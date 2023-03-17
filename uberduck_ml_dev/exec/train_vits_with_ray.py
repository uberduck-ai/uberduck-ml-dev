import tempfile
import csv
from io import BytesIO

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.io import wavfile
import wandb
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ExponentialLR

import ray
from ray.air import session, Checkpoint, CheckpointConfig
from ray.air.config import ScalingConfig, RunConfig
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
import ray.data
from ray.data.datasource import FastFileMetadataProvider
import ray.train as train
from ray.train.torch import TorchTrainer, TorchCheckpoint
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
from uberduck_ml_dev.models.common import (
    MelSTFT,
    mel_spectrogram_torch,
    spec_to_mel_torch,
    spectrogram_torch,
)
from uberduck_ml_dev.text.utils import random_utterance
from uberduck_ml_dev.utils.utils import (
    intersperse,
    to_gpu,
    slice_segments,
    clip_grad_value_,
)
from uberduck_ml_dev.losses import (
    discriminator_loss,
    generator_loss,
    kl_loss,
    feature_loss,
)


def _fix_state_dict(sd):
    return {k[7:]: v for k, v in sd.items()}


def _load_checkpoint_dict():
    checkpoint = session.get_checkpoint()
    if checkpoint is None:
        return
    checkpoint_dict = checkpoint.to_dict()
    return checkpoint_dict


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
        audio_embs = torch.FloatTensor(len(batch), 512)
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

            audio_emb = row[3]
            audio_embs[i, :] = audio_emb

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
            F.normalize(audio_embs),
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
    "lr_decay": 0.9999875,
    "segment_size": 8192,
    "init_lr_ratio": 1,
    "warmup_epochs": 0,
    "c_mel": 45,
    "c_kl": 1.0,
}

config = DEFAULTS.values()
config["with_gsts"] = False


@torch.no_grad()
def sample_inference(generator, audio_embedding=None, intersperse_blank=True):
    sample_text = random_utterance()
    text_sequence = text_to_sequence(
        sample_text,
        ["english_cleaners"],
        1.0,
        symbol_set=NVIDIA_TACO2_SYMBOLS,
    )
    if intersperse_blank:
        text_sequence = intersperse(text_sequence, 0)
    text_sequence = torch.LongTensor(text_sequence).unsqueeze(0)
    text_sequence = text_sequence.cuda()
    text_lengths = torch.LongTensor([text_sequence.shape[-1]]).cuda()
    if audio_embedding is not None:
        audio_embedding = audio_embedding.cuda()
    if hasattr(generator, "infer"):
        audio, *_ = generator.infer(
            text_sequence, text_lengths, audio_embedding=audio_embedding
        )
    else:
        audio, *_ = generator.module.infer(
            text_sequence, text_lengths, audio_embedding=audio_embedding
        )
    audio = audio.data.squeeze().cpu().numpy()
    return audio


def ray_df_to_batch(df, intersperse_blank=True):
    transcripts = df.transcript.tolist()
    audio_bytes_list = df.audio_bytes.tolist()
    if hasattr(df, "emb_path"):
        emb_bytes_list = df.emb_path.tolist()
    else:
        emb_bytes_list = [None for _ in transcripts]

    collate_fn = TextAudioCollate()
    collate_input = []
    for transcript, audio_bytes, emb_bytes in zip(
        transcripts, audio_bytes_list, emb_bytes_list
    ):
        # Audio
        bio = BytesIO(audio_bytes)
        sr, wav_data = wavfile.read(bio)
        audio = torch.FloatTensor(wav_data)
        audio_norm = audio / (np.abs(audio).max() * 2)
        audio_norm = audio_norm.unsqueeze(0)
        # Text
        text_sequence = text_to_sequence(
            transcript,
            ["english_cleaners"],
            1.0,
            symbol_set=NVIDIA_TACO2_SYMBOLS,
        )
        if intersperse_blank:
            text_sequence = intersperse(text_sequence, 0)
        text_sequence = torch.LongTensor(text_sequence)
        # Spectrogram
        spec = spectrogram_torch(audio_norm)
        spec = torch.squeeze(spec, 0)
        # Audio embedding
        if emb_bytes is None:
            audio_emb = None
        else:
            bio = BytesIO(emb_bytes)
            audio_emb = torch.load(bio)

        collate_input.append((text_sequence, spec, audio_norm, audio_emb))
    return collate_fn(collate_input)


def get_ray_dataset_with_embedding():
    lj_df = pd.read_csv(
        # "https://uberduck-datasets-dirty.s3.us-west-2.amazonaws.com/vctk_mic1/all_with_embs.txt",
        # "https://uberduck-datasets-dirty.s3.amazonaws.com/vctk-plus-va/list-with-emb-2023-03-06.txt",
        # "https://uberduck-datasets-dirty.s3.us-west-2.amazonaws.com/yourtts/vctk-libritts-va-rapper-2023-03-08-with-embs.txt",
        # NOTE(zach): this is the one with the 1000 speakers from libritts and vctk.
        # "https://uberduck-datasets-dirty.s3.amazonaws.com/vctk-va-libritts/list-2023-03-13.txt",
        # NOTE(zach): Just uberduck rappers.
        # "https://uberduck-datasets-dirty.s3.amazonaws.com/yourtts-replication/rappers-with-embs-2023-03-15.txt",
        # NOTE(zach): vctk + libritts + rappers + va + synergy
        "https://uberduck-datasets-dirty.s3.amazonaws.com/yourtts-replication/vctk-libritts-va-rapper-synergyx-2023-03-16-with-emb.txt",
        sep="|",
        header=None,
        quoting=3,
        names=["path", "speaker_id", "transcript", "dataset_audio_file_id", "emb_path"],
    )
    # lj_df = lj_df.head(1000)
    paths = lj_df.path.tolist()
    transcripts = lj_df.transcript.tolist()
    dataset_audio_files = lj_df.dataset_audio_file_id.tolist()
    emb_paths = lj_df.emb_path.tolist()

    parallelism_length = 500
    audio_ds = ray.data.read_binary_files(
        paths,
        parallelism=parallelism_length,
        # meta_provider=FastFileMetadataProvider(),
        # NOTE(zach): my hypothesis is that settings this too low causes aws timeouts.
        # ray_remote_args={"num_cpus": 0.1},
    )
    transcripts_ds = ray.data.from_items(transcripts, parallelism=parallelism_length)
    dataset_audio_file_ids = ray.data.from_items(
        dataset_audio_files, parallelism=parallelism_length
    )
    paths = ray.data.from_items(paths, parallelism=parallelism_length)
    emb_paths_ds = ray.data.read_binary_files(
        emb_paths,
        parallelism=parallelism_length,
        # meta_provider=FastFileMetadataProvider(),
        # ray_remote_args={"num_cpus": 0.1},
    )

    audio_ds = audio_ds.map_batches(
        lambda x: x, batch_format="pyarrow", batch_size=None
    )
    transcripts_ds = transcripts_ds.map_batches(
        lambda x: x, batch_format="pyarrow", batch_size=None
    )
    dataset_audio_file_ids = dataset_audio_file_ids.map_batches(
        lambda x: x, batch_format="pyarrow", batch_size=None
    )
    paths_ds = paths.map_batches(lambda x: x, batch_format="pyarrow", batch_size=None)
    emb_paths_ds = emb_paths_ds.map_batches(
        lambda x: x, batch_format="pyarrow", batch_size=None
    )

    output_dataset = (
        transcripts_ds.zip(audio_ds)
        .zip(dataset_audio_file_ids)
        .zip(paths_ds)
        .zip(emb_paths_ds)
    )
    output_dataset = output_dataset.map_batches(
        lambda table: table.rename(
            columns={
                "value": "transcript",
                "value_1": "audio_bytes",
                "value_2": "dataset_audio_file_id",
                "value_3": "path",
                "value_4": "emb_path",
            }
        )
    )

    def _is_lt_15s(x):
        duration = librosa.get_duration(filename=BytesIO(x["audio_bytes"]))
        return duration < 15 and duration > 0.5

    output_dataset = output_dataset.filter(_is_lt_15s)
    return output_dataset


@torch.no_grad()
def log(metrics, gen_audio=None, gt_audio=None, sample_audio=None):
    wandb_metrics = dict(metrics)
    if gen_audio is not None:
        wandb_metrics.update({"gen/audio": wandb.Audio(gen_audio, sample_rate=22050)})
    if gt_audio is not None:
        wandb_metrics.update({"gt/audio": wandb.Audio(gt_audio, sample_rate=22050)})
    if sample_audio is not None:
        wandb_metrics.update(
            {"sample_inference": wandb.Audio(sample_audio, sample_rate=22050)}
        )
    session.report(metrics)
    if session.get_world_rank() == 0:
        wandb.log(wandb_metrics)


def _train_step(
    metrics,
    batch,
    generator,
    discriminator,
    optim_g,
    optim_d,
    steps_per_sample,
    scaler,
    scheduler_g,
    scheduler_d,
    intersperse_blank,
):
    optim_d.zero_grad()
    optim_g.zero_grad()
    # Transform ray_batch_df to (x, x_lengths, spec, spec_lengths, y, y_lengths)
    with autocast():
        (x, x_lengths, spec, spec_lengths, y, y_lengths, audio_embs) = [
            to_gpu(el) for el in ray_df_to_batch(batch, intersperse_blank)
        ]
        generator_output = generator(
            x, x_lengths, spec, spec_lengths, audio_embedding=audio_embs
        )
        (
            y_hat,
            l_length,
            attn,
            ids_slice,
            x_mask,
            z_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
        ) = generator_output

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
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                y_d_hat_r, y_d_hat_g
            )
            loss_disc_all = loss_disc
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = clip_grad_value_(discriminator.parameters(), 100)
    scaler.step(optim_d)
    scheduler_d.step()

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
    scheduler_g.step()

    global_step = metrics["global_step"]
    metrics = dict(
        metrics,
        **{
            "loss_disc": loss_disc.item(),
            "loss_gen_all": loss_gen_all.item(),
            "loss_gen": loss_gen.item(),
            "loss_fm": loss_fm.item(),
            "loss_dur": loss_dur.item(),
            "loss_kl": loss_kl.item(),
            "lr_g": scheduler_g.get_last_lr()[0],
            "lr_d": scheduler_d.get_last_lr()[0],
        },
    )
    if global_step % steps_per_sample == 0 and session.get_world_rank() == 0:
        gen_audio = y_hat[0][0][: y_lengths[0]].data.cpu().numpy().astype("float32")
        gt_audio = y[0][0][: y_lengths[0]].data.cpu().numpy().astype("float32")
        generator.eval()
        sample_audio = sample_inference(
            generator, audio_embedding=audio_embs[0].unsqueeze(0)
        )
        generator.train()

        log(metrics, gen_audio, gt_audio, sample_audio)
    else:
        log(metrics)
    print(f"Disc Loss: {loss_disc_all.item()}. Gen Loss: {loss_gen_all.item()}")


def train_func(config: dict):
    setup_wandb(
        config,
        project="yourtts-replication",
        entity="uberduck-ai",
        rank_zero_only=True,
    )
    print("CUDA AVAILABLE: ", torch.cuda.is_available())
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    steps_per_sample = config["batch_size"]
    use_audio_embedding = (config["use_audio_embedding"],)
    gin_channels = config["gin_channels"]
    intersperse_blank = config["intersperse_blank"]
    generator = SynthesizerTrn(
        n_vocab=len(SYMBOL_SETS[NVIDIA_TACO2_SYMBOLS]),
        spec_channels=DATA_CONFIG["filter_length"] // 2 + 1,
        segment_size=TRAIN_CONFIG["segment_size"] // DATA_CONFIG["hop_length"],
        **MODEL_CONFIG,
        use_audio_embedding=use_audio_embedding,
        gin_channels=gin_channels,
    )
    generator = train.torch.prepare_model(generator)
    discriminator = MultiPeriodDiscriminator(MODEL_CONFIG["use_spectral_norm"])
    discriminator = train.torch.prepare_model(discriminator)

    checkpoint_dict = _load_checkpoint_dict()
    if checkpoint_dict is None:
        global_step = 0
        start_epoch = 0
        total_files_seen = 0
    else:
        global_step = checkpoint_dict["global_step"]
        start_epoch = checkpoint_dict["epoch"]
        total_files_seen = checkpoint_dict.get("total_files_seen", 0)
        if session.get_world_size() > 1:
            generator_sd = checkpoint_dict["generator"]
            # If generator_sd does not have module. prefix, add it.
            if not any(k.startswith("module.") for k in generator_sd.keys()):
                generator_sd = {f"module.{k}": v for k, v in generator_sd.items()}

            # NOTE(zach): Add audio embedding state dict if it is not present.
            if use_audio_embedding:
                checkpoint_has_audio_emb = all(
                    f"module.emb_audio.{k}" in generator_sd
                    for k in generator.module.emb_audio.state_dict().keys()
                )
                if not checkpoint_has_audio_emb:
                    emb_state_dict = {
                        f"module.emb_audio.{k}": v
                        for k, v in generator.module.emb_audio.state_dict().items()
                    }
                    generator_sd.update(emb_state_dict)
            # NOTE(zach): Pass strict=False due to different nuber of gin_channels
            generator.load_state_dict(generator_sd, strict=False)
            # If discriminator_sd does not have module. prefix, add it.
            discriminator_sd = checkpoint_dict["discriminator"]
            if not any(k.startswith("module.") for k in discriminator_sd.keys()):
                discriminator_sd = {
                    f"module.{k}": v for k, v in discriminator_sd.items()
                }
            discriminator.load_state_dict(discriminator_sd)
        else:
            generator_sd = _fix_state_dict(checkpoint_dict["generator"])
            if use_audio_embedding:
                checkpoint_has_audio_emb = all(
                    f"emb_audio.{k}" in generator_sd
                    for k in generator.emb_audio.state_dict().keys()
                )
                emb_state_dict = {
                    f"emb_audio.{k}": v
                    for k, v in generator.emb_audio.state_dict().items()
                }
                generator_sd.update(emb_state_dict)
            # NOTE(zach): Pass strict=False due to different nuber of gin_channels
            generator.load_state_dict(generator_sd, strict=False)
            discriminator.load_state_dict(
                _fix_state_dict(checkpoint_dict["discriminator"])
            )
        del checkpoint_dict

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
    scheduler_g = ExponentialLR(
        optim_g,
        TRAIN_CONFIG["lr_decay"],
        last_epoch=-1,
    )
    scheduler_d = ExponentialLR(
        optim_d,
        TRAIN_CONFIG["lr_decay"],
        last_epoch=-1,
    )
    dataset_shard = session.get_dataset_shard("train")
    global_step = 0
    scaler = GradScaler()
    for epoch in range(start_epoch, start_epoch + epochs):
        for batch_idx, ray_batch_df in enumerate(
            dataset_shard.iter_batches(batch_size=batch_size)
        ):
            torch.cuda.empty_cache()
            _train_step(
                {
                    "epoch": epoch,
                    "total_files_seen": total_files_seen,
                    "global_step": global_step,
                },
                ray_batch_df,
                generator,
                discriminator,
                optim_g,
                optim_d,
                steps_per_sample,
                scaler,
                scheduler_g,
                scheduler_d,
                intersperse_blank,
            )
            global_step += 1
            total_files_seen += batch_size * session.get_world_size()
        checkpoint = Checkpoint.from_dict(
            dict(
                epoch=epoch,
                global_step=global_step,
                total_files_seen=total_files_seen,
                generator=generator.state_dict(),
                discriminator=discriminator.state_dict(),
            )
        )
        session.report({}, checkpoint=checkpoint)
        if session.get_world_rank() == 0:
            # TODO(zach): Also save wandb artifact here.
            artifact = wandb.Artifact(
                f"artifact_{wandb.run.name}_epoch{epoch}_step{global_step}", "model"
            )
            with tempfile.TemporaryDirectory() as tempdirname:
                checkpoint.to_directory(tempdirname)
                artifact.add_dir(tempdirname)
                wandb.log_artifact(artifact)


class TorchCheckpointFixed(TorchCheckpoint):
    def __setstate__(self, state: dict):
        if "_data_dict" in state and state["_data_dict"]:
            state = state.copy()
            state["_data_dict"] = self._decode_data_dict(state["_data_dict"])
        super(TorchCheckpoint, self).__setstate__(state)


if __name__ == "__main__":
    print("Loading dataset")
    # ray_dataset = get_ray_dataset()
    ray_dataset = get_ray_dataset_with_embedding()
    # NOTE(zach): This is an LJSpeech checkpoint
    # checkpoint_uri = "s3://uberduck-anyscale-data/checkpoints/TorchTrainer_2023-02-17_17-49-00/TorchTrainer_6a5ed_00000_0_2023-02-17_17-52-52/checkpoint_000598/"
    # NOTE(zach): This is VCTK + VA, the latest checkpoint before the job died.
    # checkpoint_uri = "s3://uberduck-anyscale-data/checkpoints/TorchTrainer_2023-03-06_10-16-53/TorchTrainer_12bb4_00000_0_2023-03-06_10-18-03/checkpoint_000338/"
    # checkpoint_uri = "s3://uberduck-anyscale-data/checkpoints/TorchTrainer_2023-03-13_11-26-04/TorchTrainer_8405d_00000_0_2023-03-13_11-27-19/checkpoint_000345/"
    # checkpoint = Checkpoint.from_uri(checkpoint_uri)
    checkpoint_uri = "s3://uberduck-anyscale-data/checkpoints/TorchTrainer_2023-03-13_20-28-33/TorchTrainer_4c425_00000_0_2023-03-13_20-28-33/checkpoint_000431/"
    checkpoint = TorchCheckpointFixed.from_uri(checkpoint_uri)
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "epochs": 300,
            "batch_size": 24,
            "steps_per_sample": 200,
            "use_audio_embedding": True,
            # NOTE(zach): Set this to 0 if use_audio_embedding is False.
            "gin_channels": 512,
            # NOTE(zach): whether to add a blank token in between characters.
            "intersperse_blank": True,
        },
        scaling_config=ScalingConfig(
            num_workers=10, use_gpu=True, resources_per_worker=dict(CPU=4, GPU=1)
        ),
        run_config=RunConfig(
            sync_config=SyncConfig(
                upload_dir="s3://uberduck-anyscale-data/checkpoints"
            ),
            checkpoint_config=CheckpointConfig(
                num_to_keep=5,
            ),
        ),
        datasets={"train": ray_dataset},
        resume_from_checkpoint=checkpoint,
    )
    print("Starting trainer")
    result = trainer.fit()
    print(f"Last result: {result.metrics}")
