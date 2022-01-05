# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/trainer.base.ipynb (unless otherwise specified).

__all__ = ['TTSTrainer', 'DEFAULTS', 'config', 'DEFAULTS']

# Cell
import os
from pathlib import Path
from pprint import pprint

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
import numpy as np
import time

from ..models.common import MelSTFT
from ..vocoders.hifigan import HiFiGanGenerator


class TTSTrainer:
    def __init__(self, hparams, rank=None, world_size=None, device=None):
        print("TTSTrainer start", time.perf_counter())
        self.hparams = hparams
        for k, v in hparams.values().items():
            setattr(self, k, v)

        torch.backends.cudnn_enabled = hparams.cudnn_enabled
        self.global_step = 0
        self.rank = rank
        self.world_size = world_size
        self.log_dir = hparams.log_dir
        self.seed = hparams.seed
        self.symbol_set = hparams.symbol_set
        torch.manual_seed(self.seed)

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.writer = SummaryWriter(self.log_dir)
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

    def load_checkpoint(self):
        return torch.load(self.warm_start_name)

    def log(self, tag, step, scalar=None, audio=None, image=None, figure=None):
        if self.rank is not None and self.rank != 0:
            return
        if audio is not None:
            # NOTE(zach): tensorboardX add_audio requires audio to be 1D, or 2D of the shape
            # (n_samples, n_channels).
            if (audio.size(0) == 1 or audio.size(0) == 2) and audio.size(1) > 2:
                audio = audio.transpose(0, 1)
            self.writer.add_audio(tag, audio, step, sample_rate=self.sampling_rate)
        if scalar is not None:
            self.writer.add_scalar(tag, scalar, step)
        if image is not None:
            self.writer.add_image(tag, image, step, dataformats="HWC")
        if figure is not None:
            self.writer.add_figure(tag, figure, step)

    def sample(self, mel, algorithm="griffin-lim", **kwargs):
        """Invert the mel spectrogram and return the resulting audio.

        audio -> (1, N)
        """
        if self.rank is not None and self.rank != 0:
            return
        if algorithm == "griffin-lim":
            mel_stft = MelSTFT()
            audio = mel_stft.griffin_lim(mel)
        elif algorithm == "hifigan":
            assert kwargs["hifigan_config"], "hifigan_config must be set"
            assert kwargs["hifigan_checkpoint"], "hifigan_checkpoint must be set"
            cudnn_enabled = bool(kwargs["cudnn_enabled"])
            hifigan = HiFiGanGenerator(
                config=kwargs["hifigan_config"],
                checkpoint=kwargs["hifigan_checkpoint"],
                cudnn_enabled=cudnn_enabled,
            )
            audio = hifigan.infer(mel)
            audio = audio / np.max(audio)
        else:
            raise NotImplemented
        return audio

    def warm_start(self, model, optimizer, start_epoch=0):

        print("Starting warm_start", time.perf_counter())
        checkpoint = self.load_checkpoint()
        # TODO(zach): Once we are no longer using checkpoints of the old format, remove the conditional and use checkpoint["model"] only.
        model_state_dict = (
            checkpoint["model"] if "model" in checkpoint else checkpoint["state_dict"]
        )
        model.from_pretrained(
            model_dict=model_state_dict,
            device=self.device,
            ignore_layers=self.ignore_layers,
        )
        if "optimizer" in checkpoint and len(self.ignore_layers) == 0:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "iteration" in checkpoint:
            start_epoch = checkpoint["iteration"]
        if "learning_rate" in checkpoint:
            optimizer.param_groups[0]["lr"] = checkpoint["learning_rate"]
            self.learning_rate = checkpoint["learning_rate"]
        if "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]
            print(f"Adjusted global step to {self.global_step}")
        print("Ending warm_start", time.perf_counter())
        return model, optimizer, start_epoch

    def train():
        raise NotImplemented

# Cell
from ..models.base import DEFAULTS as MODEL_DEFAULTS
from ..vendor.tfcompat.hparam import HParams

DEFAULTS = HParams(
    grad_clip_thresh=1.0,
    reduction_window_schedule=[
        {"until_step": 10000, "batch_size": 16, "n_frames_per_step": 1},
        {"until_step": 50000, "batch_size": 16, "n_frames_per_step": 1},
        {"until_step": 60000, "batch_size": 16, "n_frames_per_step": 1},
        {"until_step": 70000, "batch_size": 16, "n_frames_per_step": 1},
        {"until_step": None, "batch_size": 16, "n_frames_per_step": 1},
    ],
    batch_size=16,
    decay_start=15000,
    decay_rate=8000,
    fp16_run=False,
    steps_per_sample=100,
    weight_decay=1e-6,
    sample_inference_speaker_ids=None,
)

config = DEFAULTS.values()
config.update(MODEL_DEFAULTS.values())
DEFAULTS = HParams(**config)