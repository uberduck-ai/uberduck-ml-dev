# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/trainer.base.ipynb (unless otherwise specified).

__all__ = ['TTSTrainer', 'Tacotron2Loss', 'MellotronTrainer']

# Cell
import os
from pathlib import Path
from pprint import pprint

import torch
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from ..models.common import MelSTFT
from ..utils.plot import (
    plot_attention,
    plot_gate_outputs,
    plot_spectrogram,
)


class TTSTrainer:
    def __init__(self, hparams, rank=None, world_size=None):
        self.hparams = hparams
        for k, v in hparams.values().items():
            setattr(self, k, v)

        torch.backends.cudnn_enabled = hparams.cudnn_enabled
        self.global_step = 0
        self.rank = rank
        self.world_size = world_size
        self.writer = SummaryWriter()
        if not hasattr(self, "debug"):
            self.debug = False
        if self.debug:
            print("Running in debug mode with hparams:")
            pprint(hparams.values())
        else:
            print("Initializing trainer with hparams:")
            pprint(hparams.values())

    def init_distributed(self):
        if not self.distributed_run:
            return
        if self.rank is None or self.world_size is None:
            raise Exception(
                "Rank and world size must be provided when distributed training"
            )
        dist.init_process_group(
            "nccl",
            init_method="tcp://localhost:54321",
            rank=self.rank,
            world_size=self.world_size,
        )
        torch.cuda.set_device(self.rank)

    def save_checkpoint(self, checkpoint_name, **kwargs):
        if self.rank is not None and self.rank != 0:
            return
        checkpoint = {}
        for k, v in kwargs.items():
            if hasattr(v, "state_dict"):
                checkpoint[k] = v.state_dict()
            else:
                checkpoint[k] = v
        if not Path(self.checkpoint_path).exists():
            os.makedirs(Path(self.checkpoint_path))
        torch.save(
            checkpoint, os.path.join(self.checkpoint_path, f"{checkpoint_name}.pt")
        )

    def load_checkpoint(self, checkpoint_name):
        return torch.load(os.path.join(self.checkpoint_path, checkpoint_name))

    def log(self, tag, step, scalar=None, audio=None, image=None, figure=None):
        if self.rank is not None and self.rank != 0:
            return
        if audio is not None:
            self.writer.add_audio(tag, audio, step, sample_rate=self.sample_rate)
        if scalar:
            self.writer.add_scalar(tag, scalar, step)
        if image:
            self.writer.add_image(tag, image, step, dataformats="HWC")
        if figure:
            self.writer.add_figure(tag, figure, step)

    def sample(self, mel, algorithm="griffin-lim", **kwargs):
        if self.rank is not None and self.rank != 0:
            return
        if algorithm == "griffin-lim":
            mel_stft = MelSTFT()
            audio = mel_stft.griffin_lim(mel)
        else:
            raise NotImplemented
        return audio

    def train():
        raise NotImplemented

# Cell
from random import randint
import time
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..data_loader import TextMelDataset, TextMelCollate
from ..models.mellotron import Tacotron2
from ..utils.utils import reduce_tensor


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, model_output: List, target: List):
        mel_target, gate_target = target[0], target[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)
        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(
            mel_out_postnet, mel_target
        )
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss, gate_loss


class MellotronTrainer(TTSTrainer):
    REQUIRED_HPARAMS = [
        "audiopaths_and_text",
        "checkpoint_path",
        "dataset_path",
        "epochs",
        "mel_fmax",
        "mel_fmin",
        "n_mel_channels",
        "text_cleaners",
    ]

    def validate(self, **kwargs):
        model = kwargs["model"]
        val_set = kwargs["val_set"]
        collate_fn = kwargs["collate_fn"]
        criterion = kwargs["criterion"]
        sampler = DistributedSampler(val_set) if self.distributed_run else None
        total_loss, total_mel_loss, total_gate_loss = 0, 0, 0
        total_steps = 0
        model.eval()
        with torch.no_grad():
            val_loader = DataLoader(
                val_set,
                sampler=sampler,
                shuffle=False,
                batch_size=self.batch_size,
                collate_fn=collate_fn,
            )
            for batch in val_loader:
                total_steps += 1
                if self.distributed_run:
                    X, y = model.module.parse_batch(batch)
                else:
                    X, y = model.parse_batch(batch)
                y_pred = model(X)
                mel_loss, gate_loss = criterion(y_pred, y)
                if self.distributed_run:
                    reduced_mel_loss = reduce_tensor(mel_loss, self.world_size).item()
                    reduced_gate_loss = reduce_tensor(gate_loss, self.world_size).item()
                    reduced_val_loss = reduced_mel_loss + reduced_gate_loss
                else:
                    reduced_mel_loss = mel_loss.item()
                    reduced_gate_loss = gate_loss.item()
                reduced_val_loss = reduced_mel_loss + reduced_gate_loss
                total_mel_loss += reduced_mel_loss
                total_gate_loss += reduced_gate_loss
                total_loss += reduced_val_loss

            mean_mel_loss = total_mel_loss / total_steps
            mean_gate_loss = total_gate_loss / total_steps
            mean_loss = total_loss / total_steps
            print(f"Average loss: {mean_loss}")
            self.log("Loss/val", self.global_step, scalar=mean_loss)
            self.log("MelLoss/val", self.global_step, scalar=mean_mel_loss)
            self.log("GateLoss/val", self.global_step, scalar=mean_gate_loss)
            # Generate the sample from a random item from the last y_pred batch.
            _, mel_out_postnet, gate_outputs, alignments, *_ = y_pred
            sample_idx = randint(0, mel_out_postnet.size(0) - 1)
            mel_target, gate_target = y
            audio = self.sample(mel=mel_out_postnet[sample_idx])
            self.log("AudioSample/val", self.global_step, audio=audio)
            self.log(
                "MelPredicted/val",
                self.global_step,
                figure=plot_spectrogram(mel_out_postnet[sample_idx].data.cpu()),
            )
            self.log(
                "MelTarget/val",
                self.global_step,
                figure=plot_spectrogram(mel_target[sample_idx].data.cpu()),
            )
            self.log(
                "Gate/val",
                self.global_step,
                figure=plot_gate_outputs(
                    gate_outputs[sample_idx].data.cpu(),
                    gate_target[sample_idx].data.cpu(),
                ),
            )
            self.log(
                "Attention/val",
                self.global_step,
                figure=plot_attention(alignments[sample_idx].data.cpu()),
            )
        model.train()

    @property
    def training_dataset_args(self):
        return [
            self.dataset_path,
            self.training_audiopaths_and_text,
            self.text_cleaners,
            self.p_arpabet,
            # audio params
            self.n_mel_channels,
            self.sample_rate,
            self.mel_fmin,
            self.mel_fmax,
            self.filter_length,
            self.hop_length,
            self.win_length,
            self.max_wav_value,
            self.include_f0,
        ]

    @property
    def val_dataset_args(self):
        val_args = [a for a in self.training_dataset_args]
        val_args[1] = self.val_audiopaths_and_text
        return val_args

    def train(self):
        train_set = TextMelDataset(
            *self.training_dataset_args,
            debug=self.debug,
            debug_dataset_size=self.batch_size,
        )
        val_set = TextMelDataset(
            *self.val_dataset_args, debug=self.debug, debug_dataset_size=self.batch_size
        )
        collate_fn = TextMelCollate(n_frames_per_step=1, include_f0=self.include_f0)
        sampler = None
        if self.distributed_run:
            self.init_distributed()
            sampler = DistributedSampler(train_set, rank=self.rank)
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=collate_fn,
        )
        criterion = Tacotron2Loss()

        model = Tacotron2(self.hparams)
        if torch.cuda.is_available():
            model = model.cuda()
        if self.distributed_run:
            model = DDP(model, device_ids=[self.rank])
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        start_epoch = 0
        if self.checkpoint_name:
            checkpoint = self.load_checkpoint(self.checkpoint_name)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["iteration"]
            # self.global_step = checkpoint["global_step"]
        if self.fp16_run:
            scaler = GradScaler()

        # main training loop
        for epoch in range(start_epoch, self.epochs):
            if self.distributed_run:
                sampler.set_epoch(epoch)
            for batch in train_loader:
                start_time = time.perf_counter()
                self.global_step += 1
                model.zero_grad()
                if self.distributed_run:
                    X, y = model.module.parse_batch(batch)
                else:
                    X, y = model.parse_batch(batch)
                if self.fp16_run:
                    with autocast():
                        y_pred = model(X)
                        mel_loss, gate_loss = criterion(y_pred, y)
                        loss = mel_loss + gate_loss
                else:
                    y_pred = model(X)
                    mel_loss, gate_loss = criterion(y_pred, y)
                    loss = mel_loss + gate_loss

                if self.distributed_run:
                    reduced_mel_loss = reduce_tensor(mel_loss, self.world_size).item()
                    reduced_gate_loss = reduce_tensor(gate_loss, self.world_size).item()
                    reduced_loss = reduce_mel_loss + reduced_gate_loss
                else:
                    reduced_mel_loss = mel_loss.item()
                    reduced_gate_loss = gate_loss.item()
                reduced_loss = reduced_mel_loss + reduced_gate_loss

                if self.fp16_run:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm(
                        model.parameters(), self.grad_clip_thresh
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm(
                        model.parameters(), self.grad_clip_thresh
                    )
                    optimizer.step()
                step_duration_seconds = time.perf_counter() - start_time
                print(f"Loss: {reduced_loss}")
                self.log("Loss/train", self.global_step, scalar=reduced_loss)
                self.log("MelLoss/train", self.global_step, scalar=reduced_mel_loss)
                self.log("GateLoss/train", self.global_step, scalar=reduced_gate_loss)
                self.log("GradNorm", self.global_step, scalar=grad_norm.item())
                self.log("LearningRate", self.global_step, scalar=self.learning_rate)
                self.log(
                    "StepDurationSeconds",
                    self.global_step,
                    scalar=step_duration_seconds,
                )
                if self.global_step % self.steps_per_sample == 0:
                    _, mel_out_postnet, gate_outputs, alignments, *_ = y_pred
                    mel_target, gate_target = y
                    sample_idx = randint(0, mel_out_postnet.size(0) - 1)
                    audio = self.sample(mel=mel_out_postnet[sample_idx])
                    self.log("AudioSample/train", self.global_step, audio=audio)
                    self.log(
                        "MelPredicted/train",
                        self.global_step,
                        figure=plot_spectrogram(mel_out_postnet[sample_idx].data.cpu()),
                    )
                    self.log(
                        "MelTarget/train",
                        self.global_step,
                        figure=plot_spectrogram(mel_target[sample_idx].data.cpu()),
                    )
                    self.log(
                        "Gate/train",
                        self.global_step,
                        figure=plot_gate_outputs(
                            gate_outputs[sample_idx].data.cpu(),
                            gate_target[sample_idx].data.cpu(),
                        ),
                    )
                    self.log(
                        "Attention/train",
                        self.global_step,
                        figure=plot_attention(alignments[sample_idx].data.cpu()),
                    )
            if epoch % self.epochs_per_checkpoint == 0:
                self.save_checkpoint(
                    f"mellotron_{epoch}",
                    model=model,
                    optimizer=optimizer,
                    iteration=epoch,
                    learning_rate=self.learning_rate,
                    global_step=self.global_step,
                )

            # Generate an audio sample
            # TODO(zach)

            # There's no need to validate in debug mode since we're not really training.
            if self.debug:
                continue
            self.validate(
                model=model,
                val_set=val_set,
                collate_fn=collate_fn,
                criterion=criterion,
            )