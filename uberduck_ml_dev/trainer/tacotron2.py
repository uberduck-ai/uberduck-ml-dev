# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/trainer.tacotron2.ipynb (unless otherwise specified).

__all__ = ['Tacotron2Loss', 'Tacotron2Trainer']

# Cell
from random import choice, randint
import time
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..data_loader import TextMelDataset, TextMelCollate
from ..models.tacotron2 import Tacotron2
from ..utils.plot import save_figure_to_numpy
from ..utils.utils import reduce_tensor, get_alignment_metrics


class Tacotron2Loss(nn.Module):
    def __init__(self, pos_weight):
        if pos_weight is not None:
            self.pos_weight = torch.tensor(pos_weight)
        else:
            self.pos_weight = pos_weight

        super().__init__()

    def forward(self, model_output: List, target: List):
        mel_target, gate_target = target[0], target[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        mel_out, mel_out_postnet, gate_out, _ = model_output
        mel_loss_batch = nn.MSELoss(reduction="none")(mel_out, mel_target).mean(
            axis=[1, 2]
        ) + nn.MSELoss(reduction="none")(mel_out_postnet, mel_target).mean(axis=[1, 2])

        mel_loss = mel_loss_batch.mean()

        gate_loss_batch = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight, reduce=False
        )(gate_out, gate_target).mean(axis=[1])
        gate_loss = torch.mean(gate_loss_batch)

        return mel_loss, gate_loss, mel_loss_batch, gate_loss_batch

# Cell
import os
from pathlib import Path
from pprint import pprint
from random import choice

import torch
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import time
from torch.utils.data import DataLoader
from ..models.common import MelSTFT
from ..models.torchmoji import TorchMojiInterface
from ..utils.plot import (
    plot_attention,
    plot_gate_outputs,
    plot_spectrogram,
)
from ..text.util import text_to_sequence, random_utterance
from .base import TTSTrainer
from ..data_loader import TextMelDataset, TextMelCollate
import pdb


class Tacotron2Trainer(TTSTrainer):

    REQUIRED_HPARAMS = [
        "audiopaths_and_text",
        "checkpoint_path",
        "epochs",
        "mel_fmax",
        "mel_fmin",
        "n_mel_channels",
        "text_cleaners",
        "pos_weight",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torchmoji = TorchMojiInterface(
            "../models/vocabulary.json",
            "../models/pytorch_model.bin",
        )
        # pass

    def log_training(
        self,
        model,
        X,
        y_pred,
        y,
        loss,
        mel_loss,
        gate_loss,
        mel_loss_batch,
        gate_loss_batch,
        grad_norm,
        step_duration_seconds,
    ):
        self.log("Loss/train", self.global_step, scalar=loss)
        self.log("MelLoss/train", self.global_step, scalar=mel_loss)
        self.log("GateLoss/train", self.global_step, scalar=gate_loss)
        self.log("GradNorm", self.global_step, scalar=grad_norm.item())
        self.log("LearningRate", self.global_step, scalar=self.learning_rate)
        self.log(
            "StepDurationSeconds",
            self.global_step,
            scalar=step_duration_seconds,
        )

        batch_levels = X[5]
        batch_levels_unique = torch.unique(batch_levels)
        for l in batch_levels_unique:
            mlb = mel_loss_batch[torch.where(batch_levels == l)[0]].mean()
            self.log(
                f"MelLoss/train/speaker{l.item()}",
                self.global_step,
                scalar=mlb,
            )
            glb = gate_loss_batch[torch.where(batch_levels == l)[0]].mean()
            self.log(
                f"GateLoss/train/speaker{l.item()}",
                self.global_step,
                scalar=glb,
            )
            self.log(
                f"Loss/train/speaker{l.item()}",
                self.global_step,
                scalar=mlb + glb,
            )

        if self.global_step % self.steps_per_sample == 0:
            _, mel_out_postnet, gate_outputs, alignments, *_ = y_pred
            mel_target, gate_target = y
            alignment_metrics = get_alignment_metrics(alignments)
            alignment_diagonalness = alignment_metrics["diagonalness"]
            alignment_max = alignment_metrics["max"]
            sample_idx = randint(0, mel_out_postnet.size(0) - 1)
            audio = self.sample(mel=mel_out_postnet[sample_idx])
            self.log(
                "AlignmentDiagonalness/train",
                self.global_step,
                scalar=alignment_diagonalness,
            )
            self.log("AlignmentMax/train", self.global_step, scalar=alignment_max)
            self.log("AudioSample/train", self.global_step, audio=audio)
            self.log(
                "MelPredicted/train",
                self.global_step,
                image=save_figure_to_numpy(
                    plot_spectrogram(mel_out_postnet[sample_idx].data.cpu())
                ),
            )
            self.log(
                "MelTarget/train",
                self.global_step,
                image=save_figure_to_numpy(
                    plot_spectrogram(mel_target[sample_idx].data.cpu())
                ),
            )
            self.log(
                "Gate/train",
                self.global_step,
                image=save_figure_to_numpy(
                    plot_gate_outputs(
                        gate_targets=gate_target[sample_idx].data.cpu(),
                        gate_outputs=gate_outputs[sample_idx].data.cpu(),
                    )
                ),
            )
            input_length = X[1][sample_idx].item()
            output_length = X[4][sample_idx].item()
            self.log(
                "Attention/train",
                self.global_step,
                image=save_figure_to_numpy(
                    plot_attention(
                        alignments[sample_idx].data.cpu().transpose(0, 1),
                        encoder_length=input_length,
                        decoder_length=output_length,
                    )
                ),
            )
            self.sample_inference(model)

    def sample_inference(self, model):
        if self.rank is not None and self.rank != 0:
            return
        # Generate an audio sample
        with torch.no_grad():
            transcription = random_utterance()
            gst_embedding = self.torchmoji.encode_texts([transcription])
            utterance = torch.LongTensor(
                text_to_sequence(
                    transcription,
                    self.text_cleaners,
                    p_arpabet=self.p_arpabet,
                    symbol_set=self.symbol_set,
                )
            )[None].cuda()
            speaker_id = (
                choice(self.sample_inference_speaker_ids)
                if self.sample_inference_speaker_ids
                else randint(0, self.n_speakers - 1)
            )
            input_lengths = torch.LongTensor([utterance.shape[1]]).cuda()
            input_ = [
                utterance,
                input_lengths,
                torch.LongTensor([speaker_id]).cuda(),
                torch.FloatTensor(gst_embedding).cuda(),
            ]

            model.eval()

            _, mel, gate, attn, lengths = model.inference(input_)

            model.train()
            try:
                audio = self.sample(mel[0])
                self.log("SampleInference", self.global_step, audio=audio)
            except Exception as e:
                print(f"Exception raised while doing sample inference: {e}")
                print("Mel shape: ", mel[0].shape)
            self.log(
                "Attention/sample_inference",
                self.global_step,
                image=save_figure_to_numpy(
                    plot_attention(attn[0].data.cpu().transpose(0, 1))
                ),
            )
            self.log(
                "MelPredicted/sample_inference",
                self.global_step,
                image=save_figure_to_numpy(plot_spectrogram(mel[0].data.cpu())),
            )
            self.log(
                "Gate/sample_inference",
                self.global_step,
                image=save_figure_to_numpy(
                    plot_gate_outputs(gate_outputs=gate[0].data.cpu())
                ),
            )

    def log_validation(
        self,
        X,
        y_pred,
        y,
        mean_loss,
        mean_mel_loss,
        mean_gate_loss,
        mel_loss_val,
        gate_loss_val,
        speakers_val,
    ):
        print(f"Average loss: {mean_loss}")
        self.log("Loss/val", self.global_step, scalar=mean_loss)
        self.log("MelLoss/val", self.global_step, scalar=mean_mel_loss)
        self.log("GateLoss/val", self.global_step, scalar=mean_gate_loss)

        val_levels = speakers_val
        val_levels_unique = torch.unique(val_levels)
        for l in val_levels_unique:
            mlv = mel_loss_val[torch.where(val_levels == l)[0]].mean()
            self.log(
                f"MelLoss/val/speaker{l.item()}",
                self.global_step,
                scalar=mlv,
            )
            glv = gate_loss_val[torch.where(val_levels == l)[0]].mean()
            self.log(
                f"GateLoss/val/speaker{l.item()}",
                self.global_step,
                scalar=glv,
            )
            self.log(
                f"Loss/val/speaker{l.item()}",
                self.global_step,
                scalar=mlv + glv,
            )
        # Generate the sample from a random item from the last y_pred batch.
        mel_target, gate_target = y
        _, mel_out_postnet, gate_outputs, alignments, *_ = y_pred
        alignment_metrics = get_alignment_metrics(alignments)
        alignment_diagonalness = alignment_metrics["diagonalness"]
        alignment_max = alignment_metrics["max"]

        sample_idx = randint(0, mel_out_postnet.size(0) - 1)
        audio = self.sample(mel=mel_out_postnet[sample_idx])

        self.log(
            "AlignmentDiagonalness/val", self.global_step, scalar=alignment_diagonalness
        )
        self.log("AlignmentMax/val", self.global_step, scalar=alignment_max)
        self.log("AudioSample/val", self.global_step, audio=audio)
        self.log(
            "MelPredicted/val",
            self.global_step,
            image=save_figure_to_numpy(
                plot_spectrogram(mel_out_postnet[sample_idx].data.cpu())
            ),
        )
        self.log(
            "MelTarget/val",
            self.global_step,
            image=save_figure_to_numpy(
                plot_spectrogram(mel_target[sample_idx].data.cpu())
            ),
        )
        self.log(
            "Gate/val",
            self.global_step,
            image=save_figure_to_numpy(
                plot_gate_outputs(
                    gate_targets=gate_target[sample_idx].data.cpu(),
                    gate_outputs=gate_outputs[sample_idx].data.cpu(),
                )
            ),
        )
        input_length = X[1][sample_idx].item()
        output_length = X[4][sample_idx].item()
        self.log(
            "Attention/val",
            self.global_step,
            image=save_figure_to_numpy(
                plot_attention(
                    alignments[sample_idx].data.cpu().transpose(0, 1),
                    encoder_length=input_length,
                    decoder_length=output_length,
                )
            ),
        )

    def initialize_loader(self, include_f0: bool = False, n_frames_per_step: int = 1):
        train_set = TextMelDataset(
            **self.training_dataset_args,
            debug=self.debug,
            debug_dataset_size=self.batch_size,
        )
        val_set = TextMelDataset(
            **self.val_dataset_args,
            debug=self.debug,
            debug_dataset_size=self.batch_size,
        )
        collate_fn = TextMelCollate(
            n_frames_per_step=n_frames_per_step, include_f0=include_f0
        )
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
        return train_set, val_set, train_loader, sampler, collate_fn

    def train(self):
        print("start train", time.perf_counter())
        train_set, val_set, train_loader, sampler, collate_fn = self.initialize_loader()
        criterion = Tacotron2Loss(
            pos_weight=self.pos_weight
        )  # keep higher than 5 to make clips not stretch on

        model = Tacotron2(self.hparams)
        if self.device == "cuda":
            model = model.cuda()
        if self.distributed_run:
            model = DDP(model, device_ids=[self.rank])
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        start_epoch = 0

        if self.warm_start_name:
            model, optimizer, start_epoch = self.warm_start(model, optimizer)

        if self.fp16_run:
            scaler = GradScaler()

        # main training loop
        for epoch in range(start_epoch, self.epochs):
            #             train_loader, sampler, collate_fn = self.adjust_frames_per_step(
            #                 model, train_loader, sampler, collate_fn
            #             )
            if self.distributed_run:
                sampler.set_epoch(epoch)
            for batch_idx, batch in enumerate(train_loader):
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

                        (
                            mel_loss,
                            gate_loss,
                            mel_loss_batch,
                            gate_loss_batch,
                        ) = criterion(y_pred, y)
                        loss = mel_loss + gate_loss
                        loss_batch = mel_loss_batch + gate_loss_batch
                else:
                    y_pred = model(X)
                    mel_loss, gate_loss, mel_loss_batch, gate_loss_batch = criterion(
                        y_pred, y
                    )
                    loss = mel_loss + gate_loss
                    loss_batch = mel_loss_batch + gate_loss_batch

                if self.distributed_run:
                    reduced_mel_loss = reduce_tensor(mel_loss, self.world_size).item()
                    reduced_gate_loss = reduce_tensor(gate_loss, self.world_size).item()
                    reduced_loss = reduce_mel_loss + reduced_gate_loss
                else:
                    reduced_mel_loss = mel_loss.item()
                    reduced_gate_loss = gate_loss.item()
                    reduced_gate_loss_batch = gate_loss_batch.detach()
                    reduced_mel_loss_batch = mel_loss_batch.detach()

                reduced_loss = reduced_mel_loss + reduced_gate_loss
                reduced_loss_batch = reduced_gate_loss_batch + reduced_mel_loss_batch
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
                log_start = time.time()
                self.log_training(
                    model,
                    X,
                    y_pred,
                    y,
                    reduced_loss,
                    reduced_mel_loss,
                    reduced_gate_loss,
                    reduced_mel_loss_batch,
                    reduced_gate_loss_batch,
                    grad_norm,
                    step_duration_seconds,
                )
                log_stop = time.time()
                print(
                    f"epoch: {epoch}/{self.epochs}  |  batch: {batch_idx}/{len(train_loader)}  |  loss: {reduced_loss:.4f}"
                )
            if epoch % self.epochs_per_checkpoint == 0:
                self.save_checkpoint(
                    f"tacotron2_{epoch}",
                    model=model,
                    optimizer=optimizer,
                    iteration=epoch,
                    learning_rate=self.learning_rate,
                    global_step=self.global_step,
                )

            # There's no need to validate in debug mode since we're not really training.
            if self.debug:
                continue
            self.validate(
                model=model,
                val_set=val_set,
                collate_fn=collate_fn,
                criterion=criterion,
            )

    def validate(self, **kwargs):
        print("start validate", time.perf_counter())
        model = kwargs["model"]
        val_set = kwargs["val_set"]
        collate_fn = kwargs["collate_fn"]
        criterion = kwargs["criterion"]
        sampler = DistributedSampler(val_set) if self.distributed_run else None
        (
            total_loss,
            total_mel_loss,
            total_gate_loss,
            total_mel_loss_val,
            total_gate_loss_val,
        ) = (0, 0, 0, 0, 0)
        total_steps = 0
        model.eval()
        speakers_val = []
        total_mel_loss_val = []
        total_gate_loss_val = []
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
                    speakers_val.append(X[5])
                else:
                    X, y = model.parse_batch(batch)
                    speakers_val.append(X[5])
                y_pred = model(X)
                mel_loss, gate_loss, mel_loss_batch, gate_loss_batch = criterion(
                    y_pred, y
                )
                if self.distributed_run:
                    reduced_mel_loss = reduce_tensor(mel_loss, self.world_size).item()
                    reduced_gate_loss = reduce_tensor(gate_loss, self.world_size).item()
                    reduced_val_loss = reduced_mel_loss + reduced_gate_loss
                else:
                    reduced_mel_loss = mel_loss.item()
                    reduced_gate_loss = gate_loss.item()
                    reduced_mel_loss_val = mel_loss_batch.detach()
                    reduced_gate_loss_val = gate_loss_batch.detach()

                total_mel_loss_val.append(reduced_mel_loss_val)
                total_gate_loss_val.append(reduced_gate_loss_val)
                reduced_val_loss = reduced_mel_loss + reduced_gate_loss
                total_mel_loss += reduced_mel_loss
                total_gate_loss += reduced_gate_loss
                total_loss += reduced_val_loss

            mean_mel_loss = total_mel_loss / total_steps
            mean_gate_loss = total_gate_loss / total_steps
            mean_loss = total_loss / total_steps
            total_mel_loss_val = torch.hstack(total_mel_loss_val)
            total_gate_loss_val = torch.hstack(total_gate_loss_val)
            speakers_val = torch.hstack(speakers_val)
            self.log_validation(
                X,
                y_pred,
                y,
                mean_loss,
                mean_mel_loss,
                mean_gate_loss,
                total_mel_loss_val,
                total_gate_loss_val,
                speakers_val,
            )
        model.train()

    @property
    def val_dataset_args(self):

        args = dict(**self.training_dataset_args)
        args["audiopaths_and_text"] = self.val_audiopaths_and_text
        return args

    @property
    def training_dataset_args(self):
        return {
            "audiopaths_and_text": self.training_audiopaths_and_text,
            "text_cleaners": self.text_cleaners,
            "p_arpabet": self.p_arpabet,
            "n_mel_channels": self.n_mel_channels,
            "sampling_rate": self.sampling_rate,
            "mel_fmin": self.mel_fmin,
            "mel_fmax": self.mel_fmax,
            "filter_length": self.filter_length,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "symbol_set": self.symbol_set,
            "max_wav_value": self.max_wav_value,
            "pos_weight": self.pos_weight,
            "gst_type": self.gst_type,
            "torchmoji_model_file": self.torchmoji_model_file,
            "torchmoji_vocabulary_file": self.torchmoji_vocabulary_file,
        }