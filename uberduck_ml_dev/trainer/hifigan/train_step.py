from torch.cuda.amp import autocast
from ray.air import session
from datetime import datetime
from einops import rearrange

from ...models.rvc.commons import clip_grad_value_, slice_segments
from ...data.utils import (
    mel_spectrogram_torch,
    spec_to_mel_torch,
)
from ..log import log
from ..rvc.save import save_checkpoint
from ...models.rvc.commons import rand_slice_segments

from ...data.data import MAX_WAV_VALUE


# NOTE (Sam): passing dict arguments to functions is a bit of a code smell.
# TODO (Sam): the data parameters have slightly different names here
# (e.g. hop_length v hop_size, filter_length v n_fft, num_mels v n_mel_channels, win_length v win_size, mel_fmin v fmin) - unify.
def train_step(
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
    # NOTE (Sam): The reason to pass the loss as a parameter rather than import it is to reuse the _train_step function for different losses.
    l1_loss = optimization_parameters["losses"]["l1"]["loss"]
    l1_loss_weight = optimization_parameters["losses"]["l1"]["weight"]
    generator_loss = optimization_parameters["losses"]["generator"]["loss"]
    generator_loss_weight = optimization_parameters["losses"]["generator"]["weight"]
    feature_loss = optimization_parameters["losses"]["feature"]["loss"]
    feature_loss_weight = optimization_parameters["losses"]["feature"]["weight"]

    batch = batch.to_gpu()
    mel_slices, ids_slice = rand_slice_segments(
        batch["mel_padded"],
        batch["mel_lengths"],
        train_config["segment_size"] // data_config["hop_size"],
    )
    # NOTE (Sam): it looks like audio_hat is a 3 way tensor to reuse the slice method between mel and audio.
    audio_hat = generator(mel_slices)

    # with autocast(enabled=False):
    audio_sliced = slice_segments(
        batch["audio_padded"].unsqueeze(0) / MAX_WAV_VALUE,
        ids_slice * data_config["hop_size"],
        train_config["segment_size"],
    )

    audio_sliced = rearrange(audio_sliced, "c b t -> b c t")

    y_d_hat_r, y_d_hat_g, _, _ = discriminator(audio_sliced, audio_hat.detach())

    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
    discriminator_optimizer.zero_grad()
    scaler.scale(loss_disc).backward()
    scaler.unscale_(discriminator_optimizer)
    grad_norm_d = clip_grad_value_(discriminator.parameters(), None)
    scaler.step(discriminator_optimizer)

    # with autocast(enabled=False):
    y_hat_mel = mel_spectrogram_torch(
        audio_hat.float().squeeze(1),
        data_config["n_fft"],
        data_config["num_mels"],
        data_config["sampling_rate"],
        data_config["hop_size"],
        data_config["win_size"],
        data_config["fmin"],
        data_config["fmax"],
    )

    # if train_config["fp16_run"] == True:
    #     y_hat_mel = y_hat_mel.half()
    # with autocast(enabled=train_config["fp16_run"]):
    # NOTE (Sam): y_d_hat are list of coordinates of real and generated data at the output of each block
    # fmap_r and fmap_g are the same except earlier in the network.
    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = discriminator(
        audio_sliced,
        audio_hat,
    )

    loss_mel = l1_loss(mel_slices, y_hat_mel) * train_config["c_mel"]
    loss_fm = feature_loss(fmap_r, fmap_g)
    loss_gen, losses_gen = generator_loss(y_d_hat_g)
    # TODO (Sam): put these in a loss_outputs dict like radtts
    loss_gen_all = (
        loss_gen * generator_loss_weight
        + loss_fm * feature_loss_weight
        + loss_mel * l1_loss_weight
    )

    generator_optimizer.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(generator_optimizer)
    grad_norm_g = clip_grad_value_(generator.parameters(), None)
    scaler.step(generator_optimizer)
    scaler.update()

    print("iteration: ", iteration, datetime.now())
    log_sample = iteration % train_config["steps_per_sample"] == 0
    log_checkpoint = iteration % train_config["iters_per_checkpoint"] == 0

    metrics = {
        "generator_total_loss": loss_gen_all,
        "generator_loss": loss_gen,
        "generator_feature_loss": loss_fm,
        "generator_loss_mel": loss_mel,
        # "discriminator_total_loss": loss_disc,
    }

    log(metrics)

    if log_sample and session.get_world_rank() == 0:
        import numpy as np

        audios = {
            "ground_truth": {
                "audio": audio_sliced[0][0] / np.abs(audio_sliced[0][0].cpu()).max()
            },
            "generated": {"audio": audio_hat[0][0]},
        }
        images = None

        log(audios=audios, images=images)
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
