import csv
from io import BytesIO

import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
import wandb
from torch.cuda.amp import autocast, GradScaler

import ray
from ray.air import session, Checkpoint
from ray.air.config import ScalingConfig, RunConfig
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
import ray.data
from ray.data.datasource import FastFileMetadataProvider
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.tune import SyncConfig


from uberduck_ml_dev.models import vits
from uberduck_ml_dev.models.vits import *
from uberduck_ml_dev.text.symbols import NVIDIA_TACO2_SYMBOLS, SYMBOL_SETS
from uberduck_ml_dev.vendor.tfcompat.hparam import HParams
from uberduck_ml_dev.models.tacotron2 import Tacotron2, DEFAULTS, INFERENCE
from uberduck_ml_dev.data.collate import Collate
from uberduck_ml_dev.text.utils import text_to_sequence
from uberduck_ml_dev.text.symbols import NVIDIA_TACO2_SYMBOLS
from uberduck_ml_dev.trainer.base import sample
from uberduck_ml_dev.models.common import MelSTFT, mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch
from uberduck_ml_dev.utils.plot import (
    save_figure_to_numpy,
    plot_spectrogram,
    plot_gate_outputs,
    plot_attention,
)
from uberduck_ml_dev.monitoring.statistics import get_alignment_metrics
from uberduck_ml_dev.text.utils import random_utterance
from uberduck_ml_dev.utils.utils import intersperse, to_gpu, slice_segments, clip_grad_value_
from uberduck_ml_dev.losses import discriminator_loss, generator_loss, kl_loss, feature_loss


class TextAudioCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
        )

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

        if self.return_ids:
            return (
                text_padded,
                text_lengths,
                spec_padded,
                spec_lengths,
                wav_padded,
                wav_lengths,
                ids_sorted_decreasing,
            )
        return (
            text_padded,
            text_lengths,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
        )


MODEL_CONFIG = {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_rates": [8, 8, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "n_layers_q": 3,
    "use_spectral_norm": False,
}
DATA_CONFIG = {
    "training_files": "filelists/ljs_audio_text_train_filelist.txt.cleaned",
    "validation_files": "filelists/ljs_audio_text_val_filelist.txt.cleaned",
    "text_cleaners": ["english_cleaners2"],
    "max_wav_value": 32768.0,
    "sampling_rate": 22050,
    "filter_length": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mel_channels": 80,
    "mel_fmin": 0.0,
    "mel_fmax": None,
    "add_blank": True,
    "n_speakers": 0,
    "cleaned_text": True,
}

TRAIN_CONFIG = {
    "log_interval": 200,
    "eval_interval": 1000,
    "seed": 1234,
    "epochs": 20000,
    "learning_rate": 2e-4,
    "betas": [0.8, 0.99],
    "eps": 1e-9,
    "batch_size": 64,
    "fp16_run": True,
    "lr_decay": 0.999875,
    "segment_size": 8192,
    "init_lr_ratio": 1,
    "warmup_epochs": 0,
    "c_mel": 45,
    "c_kl": 1.0,
}

config = DEFAULTS.values()
config["with_gsts"] = False

# NOTE(zach): not using this to attempt to copy the OG vits repo precisely
# stft = MelSTFT(
#     filter_length=DEFAULTS.filter_length,
#     hop_length=DEFAULTS.hop_length,
#     win_length=DEFAULTS.win_length,
#     n_mel_channels=DEFAULTS.n_mel_channels,
#     sampling_rate=DEFAULTS.sampling_rate,
#     mel_fmin=DEFAULTS.mel_fmin,
#     mel_fmax=DEFAULTS.mel_fmax,
#     padding=None,
# )


# def sample_inference(model):
#     with torch.no_grad():
#         transcription = random_utterance()
#         utterance = torch.LongTensor(
#             text_to_sequence(
#                 transcription,
#                 ["english_cleaners"],
#                 p_arpabet=1.0,
#                 symbol_set=NVIDIA_TACO2_SYMBOLS,
#             )
#         )[None]
# 
#         input_lengths = torch.LongTensor([utterance.shape[1]])
#         speaker_id_tensor = None  # torch.LongTensor([speaker_id])
# 
#         if torch.cuda.is_available():
#             utterance = utterance.cuda()
#             input_lengths = input_lengths.cuda()
#             gst_embedding = None
#             speaker_id_tensor = None  # speaker_id_tensor.cuda()
# 
#         model.eval()
#         model_output = model.forward(
#             input_text=utterance,
#             input_lengths=input_lengths,
#             speaker_ids=speaker_id_tensor,
#             # NOTE (Sam): this is None if using old multispeaker training, not None if using new pretrained encoder.
#             audio_encoding=None,  # speaker_embedding,
#             embedded_gst=None,  # gst_embedding,
#             mode=INFERENCE,
#         )
#         model.train()
#         audio = sample(model_output["mel_outputs_postnet"][0])
#         if (audio.size(0) == 1 or audio.size(0) == 2) and audio.size(1) > 2:
#             audio = audio.transpose(0, 1)
#         return audio


def ray_df_to_batch(df):
    transcripts = df.transcript.tolist()
    audio_bytes_list = df.audio_bytes.tolist()

    collate_fn = TextAudioCollate()
    collate_input = []
    for transcript, audio_bytes in zip(transcripts, audio_bytes_list):
        bio = BytesIO(audio_bytes)
        sr, wav_data = wavfile.read(bio)
        audio = torch.FloatTensor(wav_data)
        audio_norm = audio / (np.abs(audio).max() * 2)
        audio_norm = audio_norm.unsqueeze(0)
        text_sequence = torch.LongTensor(
            intersperse(
                text_to_sequence(
                    transcript,
                    ["english_cleaners"],
                    1.0,
                    symbol_set=NVIDIA_TACO2_SYMBOLS,
                ),
                0,
            )
        )
        spec = spectrogram_torch(audio_norm)
        spec = torch.squeeze(spec, 0)
        collate_input.append((text_sequence, spec, audio_norm))
    return collate_fn(collate_input)


def get_ray_dataset():
    lj_df = pd.read_csv(
        # "s3://uberduck-audio-files/LJSpeech/metadata.csv",
        "https://uberduck-audio-files.s3.us-west-2.amazonaws.com/LJSpeech/metadata.csv",
        sep="|",
        header=None,
        quoting=csv.QUOTE_NONE,
        names=["path", "transcript"],
    )
    # lj_df = lj_df.head(100)
    paths = ("s3://uberduck-audio-files/LJSpeech/" + lj_df.path).tolist()
    transcripts = lj_df.transcript.tolist()

    audio_ds = ray.data.read_binary_files(
        paths,
        parallelism=len(paths),
        meta_provider=FastFileMetadataProvider(),
    )
    transcripts_ds = ray.data.from_items(transcripts, parallelism=len(transcripts))

    audio_ds = audio_ds.map_batches(
        lambda x: x, batch_format="pyarrow", batch_size=None
    )
    transcripts_ds = transcripts_ds.map_batches(
        lambda x: x, batch_format="pyarrow", batch_size=None
    )

    output_dataset = transcripts_ds.zip(audio_ds)
    output_dataset = output_dataset.map_batches(
        lambda table: table.rename(
            columns={"value": "transcript", "value_1": "audio_bytes"}
        )
    )
    return output_dataset

@torch.no_grad()
def log(metrics, gen_audio=None, gt_audio=None):
    wandb_metrics = dict(metrics)
    if gen_audio is not None:
        wandb_metrics.update({
            "gen/audio": wandb.Audio(gen_audio, sample_rate=22050)
        })
    if gt_audio is not None:
        wandb_metrics.update({
            "gt/audio": wandb.Audio(gt_audio, sample_rate=22050)
        })
    session.report(metrics)
    if session.get_world_rank() == 0:
        wandb.log(wandb_metrics)

def _train_step(batch, generator, discriminator, optim_g, optim_d, global_step, steps_per_sample, scaler):
    optim_d.zero_grad()
    optim_g.zero_grad()
    # Transform ray_batch_df to (x, x_lengths, spec, spec_lengths, y, y_lengths)
    with autocast():
        (x, x_lengths, spec, spec_lengths, y, y_lengths) = [
            to_gpu(el) for el in ray_df_to_batch(batch)
        ]
        generator_output = generator(x, x_lengths, spec, spec_lengths)
        y_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = generator_output

        mel = spec_to_mel_torch(spec)
        hop_length = DATA_CONFIG["hop_length"]
        segment_size = TRAIN_CONFIG["segment_size"]
        y_mel = slice_segments(mel, ids_slice, segment_size // hop_length)

        y_hat_mel = mel_spectrogram_torch(y_hat)

        y = slice_segments(y, ids_slice * hop_length, segment_size)

        discriminator_output = discriminator(y, y_hat.detach())
        y_d_hat_r, y_d_hat_g, _, _ = discriminator_output

        with autocast(enabled=False):
            # Generator step
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
            loss_disc_all = loss_disc
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = clip_grad_value_(discriminator.parameters(), 100)
    scaler.step(optim_d)

    with autocast():
        # Discriminator step
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = discriminator(y, y_hat)
        with autocast(enabled=False):
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * TRAIN_CONFIG["c_mel"]
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * TRAIN_CONFIG["c_kl"]
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = clip_grad_value_(generator.parameters(), 100)
    scaler.step(optim_g)
    scaler.update()

    metrics = {
        "loss_disc": loss_disc.item(),
        "loss_gen_all": loss_gen_all.item(),
        "loss_gen": loss_gen.item(),
        "loss_fm": loss_fm.item(),
        "loss_dur": loss_dur.item(),
        "loss_kl": loss_kl.item(),
    }
    if global_step % steps_per_sample == 0 and session.get_world_rank() == 0:
        log(metrics, y_hat[0][0][:y_lengths[0]].data.cpu().numpy().astype("float32"), y[0][0][:y_lengths[0]].data.cpu().numpy().astype("float32"))
    else:
        log(metrics)
    print(f"Disc Loss: {loss_disc_all.item()}. Gen Loss: {loss_gen_all.item()}")


def train_func(config: dict):
    setup_wandb(config, project="vits-ray", rank_zero_only=False)
    print("CUDA AVAILABLE: ", torch.cuda.is_available())
    is_cuda = torch.cuda.is_available()
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    steps_per_sample = config["batch_size"]
    is_cuda = torch.cuda.is_available()
    generator = SynthesizerTrn(
        n_vocab=len(SYMBOL_SETS[NVIDIA_TACO2_SYMBOLS]),
        spec_channels=DATA_CONFIG["filter_length"] // 2 + 1,
        segment_size=TRAIN_CONFIG["segment_size"] // DATA_CONFIG["hop_length"],
        **MODEL_CONFIG,
    )
    generator = train.torch.prepare_model(generator)
    discriminator = MultiPeriodDiscriminator(MODEL_CONFIG["use_spectral_norm"])
    discriminator = train.torch.prepare_model(discriminator)
    optim_g = torch.optim.AdamW(
        generator.parameters(),
        TRAIN_CONFIG["learning_rate"],
        betas=TRAIN_CONFIG["betas"],
        eps=TRAIN_CONFIG["eps"],
    )
    optim_d = torch.optim.AdamW(
        discriminator.parameters(),
        TRAIN_CONFIG["learning_rate"],
        betas=TRAIN_CONFIG["betas"],
        eps=TRAIN_CONFIG["eps"],
    )
    dataset_shard = session.get_dataset_shard("train")
    global_step = 0
    scaler = GradScaler()
    for epoch in range(epochs):
        for batch_idx, ray_batch_df in enumerate(
            dataset_shard.iter_batches(batch_size=batch_size)
        ):
            torch.cuda.empty_cache()
            _train_step(ray_batch_df, generator, discriminator, optim_g, optim_d, global_step, steps_per_sample, scaler)
            global_step += 1
        session.report(
            {},
            checkpoint=Checkpoint.from_dict(
                dict(
                    epoch=epoch,
                    global_step=global_step,
                    generator=generator.state_dict(),
                    discriminator=discriminator.state_dict(),
                )
            )
        )


if __name__ == "__main__":
    print("Loading dataset")
    ray_dataset = get_ray_dataset()
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "epochs": 100,
            "batch_size": 32,
            "steps_per_sample": 100,
        },
        scaling_config=ScalingConfig(
            num_workers=10, use_gpu=True, resources_per_worker=dict(CPU=4, GPU=1)
        ),
        run_config=RunConfig(
            sync_config=SyncConfig(
                upload_dir="s3://uberduck-anyscale-data/checkpoints"
            )
        ),
        datasets={"train": ray_dataset},
    )
    print("Starting trainer")
    result = trainer.fit()
    print(f"Last result: {result.metrics}")
