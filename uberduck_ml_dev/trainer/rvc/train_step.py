from torch.cuda.amp import autocast
from ray.air import session
from datetime import datetime

from uberduck_ml_dev.models.rvc.commons import clip_grad_value_, slice_segments
from uberduck_ml_dev.data.utils import mel_spectrogram_torch, spec_to_mel_torch
from ...utils.utils import (
    to_gpu,
)
from ..log import log
from .save import save_checkpoint


# TODO (Sam): add config arguments to model / optimization / logging and remove.
# NOTE (Sam): passing dict arguments to functions is a bit of a code smell.
def _train_step(
    batch, config, models, optimization_parameters, logging_parameters, iteration
):
    data_config = config["data"]
    train_config = config["train"]

    generator = models["generator"]
    discriminator = models["discriminator"]
    discriminator_optimizer = optimization_parameters["optimizers"]["discriminator"]
    generator_optimizer = optimization_parameters["optimizers"]["generator"]
    scaler = optimization_parameters["scaler"]
    discriminator_loss = optimization_parameters["losses"]["discriminator"]["loss"]
    # NOTE (Sam): losses like l1_loss are quite generic outside of the arguments passed.
    # The reason to pass the loss as a parameter rather than import it is to reuse the _train_step function for different losses.
    # However, explicit assignments render that goal currently only half attained.
    # The reason we need explicit assignments is because I'm not sure how to parameterize the arguments passed to the loss function in the _train_step function invocation.
    # Once that's figured out, we will surely be Gucci.
    l1_loss = optimization_parameters["losses"]["l1"]["loss"]
    l1_loss_weight = optimization_parameters["losses"]["l1"]["weight"]
    generator_loss = optimization_parameters["losses"]["generator"]["loss"]
    generator_loss_weight = optimization_parameters["losses"]["generator"]["weight"]
    feature_loss = optimization_parameters["losses"]["feature"]["loss"]
    feature_loss_weight = optimization_parameters["losses"]["feature"]["weight"]
    kl_loss = optimization_parameters["losses"]["kl"]["loss"]
    kl_loss_weight = optimization_parameters["losses"]["kl"]["weight"]

    # TODO (Sam): make this a dict or better-yet a batch class.
    (
        phone,
        phone_lengths,
        pitch,
        pitchf,
        spec,
        spec_lengths,
        wave,
        wave_lengths,
        sid,
    ) = batch

    # NOTE (Sam): moving to gpu needs to be done in the training step not in the collate function (i.e. here).
    # TODO (Sam): move to batch.to_gpu().
    phone = to_gpu(phone)
    phone_lengths = to_gpu(phone_lengths)
    pitch = to_gpu(pitch)
    pitchf = to_gpu(pitchf)
    spec = to_gpu(spec)
    spec_lengths = to_gpu(spec_lengths)
    wave = to_gpu(wave)
    wave_lengths = to_gpu(wave_lengths)
    sid = to_gpu(sid)

    (
        y_hat,
        ids_slice,
        x_mask,
        z_mask,
        (z, z_p, m_p, logs_p, m_q, logs_q),
    ) = generator(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)

    # NOTE (Sam): we only train on a portion of the audio determined by the segment_size
    wave = slice_segments(
        wave, ids_slice * data_config["hop_length"], train_config["segment_size"]
    )

    y_d_hat_r, y_d_hat_g, _, _ = discriminator(wave, y_hat.detach())
    with autocast(enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
            y_d_hat_r, y_d_hat_g
        )
    discriminator_optimizer.zero_grad()
    scaler.scale(loss_disc).backward()
    scaler.unscale_(discriminator_optimizer)
    grad_norm_d = clip_grad_value_(discriminator.parameters(), None)
    scaler.step(discriminator_optimizer)

    # TODO (Sam): just compute mels directly in precompute like for RADTTS
    mel = spec_to_mel_torch(
        spec,
        data_config["filter_length"],
        data_config["n_mel_channels"],
        data_config["sampling_rate"],
        data_config["mel_fmin"],
        data_config["mel_fmax"],
    )
    y_mel = slice_segments(
        mel, ids_slice, train_config["segment_size"] // data_config["hop_length"]
    )
    with autocast(enabled=False):
        y_hat_mel = mel_spectrogram_torch(
            y_hat.float().squeeze(1),
            data_config["filter_length"],
            data_config["n_mel_channels"],
            data_config["sampling_rate"],
            data_config["hop_length"],
            data_config["win_length"],
            data_config["mel_fmin"],
            data_config["mel_fmax"],
        )
    if train_config["fp16_run"] == True:
        y_hat_mel = y_hat_mel.half()
    with autocast(enabled=train_config["fp16_run"]):
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = discriminator(wave, y_hat)
        loss_mel = l1_loss(y_mel, y_hat_mel) * train_config["c_mel"]
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * train_config["c_kl"]
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        # TODO (Sam): put these in a loss_outputs dict like radtts
        loss_gen_all = (
            loss_gen * generator_loss_weight
            + loss_fm * feature_loss_weight
            + loss_mel * l1_loss_weight
            + loss_kl * kl_loss_weight
        )
    generator_optimizer.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(generator_optimizer)
    grad_norm_g = clip_grad_value_(generator.parameters(), None)
    scaler.step(generator_optimizer)
    scaler.update()

    metrics = {"generator loss": loss_gen_all}

    print("iteration: ", iteration, datetime.now())
    log_sample = iteration % train_config["steps_per_sample"] == 0
    log_checkpoint = iteration % train_config["iters_per_checkpoint"] == 0

    # if log_sample and session.get_world_rank() == 0:
    #     generator.eval()
    #     # TODO (Sam): add sample logging
    #     images, audios = get_log_audio(
    #         batch,
    #         generator,
    #     )
    #     log(metrics, audios)
    #     generator.train()
    # else:
    log(metrics)

    if log_checkpoint and session.get_world_rank() == 0:
        checkpoint_path = f"{train_config['output_directory']}/model_{iteration}.pt"
        save_checkpoint(
            generator,
            generator_optimizer,
            discriminator,
            discriminator_optimizer,
            iteration,
            checkpoint_path,
        )
