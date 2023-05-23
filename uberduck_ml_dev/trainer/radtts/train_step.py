# NOTE (Sam): for use with ray trainer.
from datetime import datetime
import os

import torch
from torch.cuda.amp import autocast
from ray.air import session

from .log import get_log_audio, log
from .save import save_checkpoint
from ...utils.utils import (
    to_gpu,
)


# TODO (Sam): it seems like much of this can be made generic for multiple models.
def _train_step(
    batch,
    model,
    optim,
    iteration,
    log_decoder_samples,
    log_attribute_samples,
    steps_per_sample,
    scaler,
    iters_per_checkpoint,
    output_directory,
    criterion,
    attention_kl_loss,
    kl_loss_start_iter,
    binarization_start_iter,
    vocoder,
):
    print(datetime.now(), "entering train step:", iteration)
    if iteration >= binarization_start_iter:
        binarize = True
    else:
        binarize = False

    optim.zero_grad()

    with autocast(enabled=False):
        batch_dict = batch  # torch DataLoader?
        # TODO (Sam): move to batch.go_gpu().
        mel = to_gpu(batch_dict["mel"])
        speaker_ids = to_gpu(batch_dict["speaker_ids"])
        attn_prior = to_gpu(batch_dict["attn_prior"])
        f0 = to_gpu(batch_dict["f0"])
        voiced_mask = to_gpu(batch_dict["voiced_mask"])
        p_voiced = to_gpu(batch_dict["p_voiced"])
        text = to_gpu(batch_dict["text"])
        in_lens = to_gpu(batch_dict["input_lengths"])
        out_lens = to_gpu(batch_dict["output_lengths"])
        energy_avg = to_gpu(batch_dict["energy_avg"])
        audio_embedding = to_gpu(batch_dict["audio_embedding"])

        outputs = model(
            mel,
            speaker_ids,
            text,
            in_lens,
            out_lens,
            binarize_attention=binarize,
            attn_prior=attn_prior,
            f0=f0,
            energy_avg=energy_avg,
            voiced_mask=voiced_mask,
            p_voiced=p_voiced,
            audio_embedding=audio_embedding,
        )

        loss_outputs = criterion(outputs, in_lens, out_lens)

        print_list = []
        loss = None
        for k, (v, w) in loss_outputs.items():
            if w > 0:
                loss = v * w if loss is None else loss + v * w
            print_list.append("  |  {}: {:.3f}".format(k, v))

        w_bin = criterion.loss_weights.get("binarization_loss_weight", 1.0)
        if binarize and iteration >= kl_loss_start_iter:
            binarization_loss = attention_kl_loss(outputs["attn"], outputs["attn_soft"])
            loss += binarization_loss * w_bin
        else:
            binarization_loss = torch.zeros_like(loss)
        loss_outputs["binarization_loss"] = (binarization_loss, w_bin)
    grad_clip_val = 1.0  # TODO (Sam): make this a config option
    print(print_list)
    scaler.scale(loss).backward()
    if grad_clip_val > 0:
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)

    scaler.step(optim)
    scaler.update()

    metrics = {"loss": loss.item()}
    for k, (v, w) in loss_outputs.items():
        metrics[k] = v.item()

    print("iteration: ", iteration, datetime.now())
    log_sample = iteration % steps_per_sample == 0
    log_checkpoint = iteration % iters_per_checkpoint == 0

    if log_sample and session.get_world_rank() == 0:
        model.eval()
        # TODO (Sam): adding tf output logging and out of distribution inference
        # TODO (Sam): add logging of ground truth
        images, audios = get_log_audio(
            batch_dict,
            log_decoder_samples,
            log_attribute_samples,
            model,
            speaker_ids,
            text,
            f0,
            energy_avg,
            voiced_mask,
            vocoder,
        )
        print("audio", audios)
        # TODO (Sam): make out of sample logging cleaner.
        # gt_path = "/usr/src/app/radtts/ground_truth"
        # oos_embs = os.listdir(gt_path)
        # # this doesn't help for reasons described above
        # for oos_name in oos_embs:
        #     audio_embedding_oos = torch.load(f"{gt_path}/{oos_name}").cuda()
        #     _, audios_oos = get_log_audio(
        #         outputs,
        #         batch_dict,
        #         log_decoder_samples,
        #         log_attribute_samples,
        #         model,
        #         speaker_ids,
        #         text,
        #         f0,
        #         energy_avg,
        #         voiced_mask,
        #         vocoder,
        #         oos_name=oos_name,
        #         audio_embedding_oos=audio_embedding_oos,
        #     )
        #     audios.update(audios_oos)
        log(metrics, audios)
        model.train()
    else:
        log(metrics)

    session.report(metrics)
    if log_checkpoint and session.get_world_rank() == 0:
        checkpoint_path = f"{output_directory}/model_{iteration}.pt"
        save_checkpoint(model, optim, iteration, checkpoint_path)

    print(f"Loss: {loss.item()}")
