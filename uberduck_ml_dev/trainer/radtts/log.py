import torch
import wandb

from ray.air import session
import numpy as np

from ...utils.utils import (
    to_gpu,
)
from ...utils.plot import plot_alignment_to_numpy


@torch.no_grad()
def log(metrics, audios={}):
    # pass
    wandb_metrics = dict(metrics)

    for k, v in audios.items():
        wandb_metrics[k] = wandb.Audio(v, sample_rate=22050)

    # session.report(metrics)
    if session.get_world_rank() == 0:
        wandb.log(wandb_metrics)


# want to test out of sample but can only do proper inference with zero shot dap so lets just look at zero shot decoder samples
# in particular, the out of sample here will probably be worse than inference since the f0 and energy models are not being used.
@torch.no_grad()
def get_log_audio(
    batch_dict,
    log_decoder_samples,
    log_attribute_samples,
    model,
    speaker_ids,
    text,
    f0,
    energy_avg,
    voiced_mask,
    vocoder,  # NOTE (Sam): should this be moved on and off gpu?
    audio_embedding_oos=None,
    oos_name=None,
):
    # print( audio_embedding_oos, oos_name, bool(audio_embedding_oos is None), bool(oos_name is None))
    assert bool(audio_embedding_oos is None) == bool(
        oos_name is None
    ), "must provide both or neither of audio_embedding_oos and oos_name"

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

    # NOTE (Sam): I don't think we can reuse the previous outputs since binarize_attention must be true
    outputs = model(
        mel,
        speaker_ids,
        text,
        in_lens,
        out_lens,
        binarize_attention=True,
        attn_prior=attn_prior,
        f0=f0,
        energy_avg=energy_avg,
        voiced_mask=voiced_mask,
        p_voiced=p_voiced,
        audio_embedding=audio_embedding,
    )

    attn_used = outputs["attn"]
    attn_soft = outputs["attn_soft"]

    images = {}
    audios = {}
    if attn_used is not None:
        images["attention_weights"] = plot_alignment_to_numpy(
            attn_soft[0, 0].data.cpu().numpy().T, title="audioname"
        )
        images["attention_weights_max"] = plot_alignment_to_numpy(
            attn_used[0, 0].data.cpu().numpy().T, title="audioname"
        )
        attribute_sigmas = []
        """ NOTE: if training vanilla radtts (no attributes involved),
        use log_attribute_samples only, as there will be no ground truth
        features available. The infer function in this case will work with
        f0=None, energy_avg=None, and voiced_mask=None
        """
        if log_decoder_samples:  # decoder with gt features
            attribute_sigmas.append(-1)
        if log_attribute_samples:  # attribute prediction
            if hasattr(model, "is_attribute_unconditional"):
                if model.is_attribute_unconditional():
                    attribute_sigmas.extend([1.0])
                else:
                    attribute_sigmas.extend([0.1, 0.5, 0.8, 1.0])
            else:
                if model.module.is_attribute_unconditional():
                    attribute_sigmas.extend([1.0])
                else:
                    attribute_sigmas.extend([0.1, 0.5, 0.8, 1.0])
        if len(attribute_sigmas) > 0:
            if audio_embedding_oos is not None:
                audio_embedding = audio_embedding_oos.unsqueeze(0)
            durations = attn_used[0, 0].sum(0, keepdim=True)
            # NOTE (Sam): this is causing problems when durations are > x.5 and binarize_attention is false.
            #  In that case, durations + 0.5 . floor > durations
            # this causes issues to the length_regulator, which expects floor < durations.
            # Just keep binarize_attention = True in inference and don't think about it that hard.
            durations = (durations + 0.5).floor().int()
            # NOTE (Sam): should we load vocoder to CPU to avoid taking up valuable GPU vRAM?
            for attribute_sigma in attribute_sigmas:
                # try:
                if attribute_sigma > 0.0:
                    if hasattr(model, "infer"):
                        model_output = model.infer(
                            speaker_ids[0:1],
                            text[0:1],
                            0.8,
                            dur=durations,
                            f0=None,
                            energy_avg=None,
                            voiced_mask=None,
                            sigma_f0=attribute_sigma,
                            sigma_energy=attribute_sigma,
                            audio_embedding=audio_embedding[0:1],
                        )
                    else:
                        model_output = model.module.infer(
                            speaker_ids[0:1],
                            text[0:1],
                            0.8,
                            dur=durations,
                            f0=None,
                            energy_avg=None,
                            voiced_mask=None,
                            sigma_f0=attribute_sigma,
                            sigma_energy=attribute_sigma,
                            audio_embedding=audio_embedding[0:1],
                        )
                else:
                    if hasattr(model, "infer"):
                        model_output = model.infer(
                            speaker_ids[0:1],
                            text[0:1],
                            0.8,
                            dur=durations,
                            f0=f0[0:1, : durations.sum()],
                            energy_avg=energy_avg[0:1, : durations.sum()],
                            voiced_mask=voiced_mask[0:1, : durations.sum()],
                            audio_embedding=audio_embedding[0:1],
                        )
                    else:
                        model_output = model.module.infer(
                            speaker_ids[0:1],
                            text[0:1],
                            0.8,
                            dur=durations,
                            f0=f0[0:1, : durations.sum()],
                            energy_avg=energy_avg[0:1, : durations.sum()],
                            voiced_mask=voiced_mask[0:1, : durations.sum()],
                            audio_embedding=audio_embedding[0:1],
                        )
                # except:
                #     print("Instability or issue occured during inference, skipping sample generation for TB logger")
                #     continue
                mels = model_output["mel"]
                if hasattr(vocoder, "forward"):
                    audio = vocoder(mels.cpu()).float()[0]
                audio = audio[0].detach().cpu().numpy()
                audio = audio / np.abs(audio).max()
                if attribute_sigma < 0:
                    sample_tag = "decoder_sample_gt_attributes"
                else:
                    sample_tag = f"sample_attribute_sigma_{attribute_sigma}"
                if oos_name is not None:
                    sample_tag = f"{sample_tag}_oos_{oos_name}"
                audios[sample_tag] = audio

    return images, audios
