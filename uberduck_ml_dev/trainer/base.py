__all__ = ["TTSTrainer", "DEFAULTS", "config", "DEFAULTS"]


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
from ..models.base import DEFAULTS as MODEL_DEFAULTS
from ..vendor.tfcompat.hparam import HParams

# Note (Sam): keeping TTS specific parameters out of here actually -this shall be the pure trainer class.
class TTSTrainer:

    # Note (Sam): rewriting with explicit hparams for clarity.
    # Note (Sam): should migrate to Lightning.
    def __init__(self, hparams, rank=None, world_size=None, device=None):
        print("TTSTrainer start", time.perf_counter())

        torch.backends.cudnn_enabled = hparams.cudnn_enabled

        # NOTE (Sam): all hparams should be added to initializations and this next line removed.
        self.hparams = hparams
        self.global_step = 0
        self.rank = rank
        self.world_size = world_size
        self.log_dir = hparams.log_dir
        self.seed = hparams.seed
        self.checkpoint_name = hparams.checkpoint_name
        self.checkpoint_path = hparams.checkpoint_path
        self.epochs = hparams.epochs
        self.epochs_per_checkpoint = hparams.epochs_per_checkpoint
        self.learning_rate = hparams.learning_rate
        self.debug = hparams.debug
        self.batch_size = hparams.batch_size
        self.sample_inference_speaker_ids = hparams.sample_inference_speaker_ids
        self.weight_decay = hparams.weight_decay
        self.warm_start_name = hparams.warm_start_name
        self.ignore_layers = hparams.ignore_layers
        self.grad_clip_thresh = hparams.grad_clip_thresh
        self.steps_per_sample = hparams.steps_per_sample
        self.cudnn_enabled = hparams.cudnn_enabled
        self.is_validate = hparams.is_validate
        self.num_workers = hparams.num_workers
        self.pin_memory = hparams.pin_memory
        self.lr_decay_start = hparams.lr_decay_start
        self.lr_decay_rate = hparams.lr_decay_rate
        self.lr_decay_min = hparams.lr_decay_min

        # NOTE (Sam): these are deprecated.
        self.distributed_run = hparams.distributed_run
        self.fp16_run = hparams.fp16_run

        torch.manual_seed(self.seed)

        if device:
            self.device = device
        elif torch.cuda.is_available() and self.cudnn_enabled:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.writer = SummaryWriter(self.log_dir)
        if not hasattr(self, "debug"):
            self.debug = False
        if self.debug:
            # NOTE (Sam): a simple list representation of the loss for training tests.
            self.loss = []
            print("Running in debug mode with hparams:")
            pprint(hparams.values())
        else:
            print("Initializing trainer with hparams:")
            pprint(hparams.values())
        # NOTE (Sam): I think we should add warm starting to the init

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
        return torch.load(self.warm_start_name, map_location=self.device)

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
        if "model" in checkpoint:
            model_state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            model_state_dict = checkpoint["state_dict"]
        else:
            model_state_dict = checkpoint
        model.from_pretrained(
            model_dict=model_state_dict,
            device=self.device,
            ignore_layers=self.ignore_layers,
        )
        if "optimizer" in checkpoint and len(self.ignore_layers) == 0:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "iteration" in checkpoint:
            start_epoch = checkpoint["iteration"] + 1
        if "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]
            print(f"Adjusted global step to {self.global_step}")
        print("Ending warm_start", time.perf_counter())
        return model, optimizer, start_epoch

    def train():
        raise NotImplemented


DEFAULTS = HParams(
    grad_clip_thresh=1.0,
    # reduction_window_schedule=[
    #     {"until_step": 10000, "batch_size": 16, "n_frames_per_step": 1},
    #     {"until_step": 50000, "batch_size": 16, "n_frames_per_step": 1},
    #     {"until_step": 60000, "batch_size": 16, "n_frames_per_step": 1},
    #     {"until_step": 70000, "batch_size": 16, "n_frames_per_step": 1},
    #     {"until_step": None, "batch_size": 16, "n_frames_per_step": 1},
    # ],
    batch_size=16,
    decay_start=15000,
    decay_rate=8000,
    fp16_run=False,
    steps_per_sample=100,
    weight_decay=1e-6,
    sample_inference_speaker_ids=None,
    is_validate=True,
    learning_rate=1e-3,
    epochs=10,
    epochs_per_checkpoint=10,
    debug=False,
    warm_start_name=None,
    ignore_layers=None,
    distributed_run=False,
    num_workers=1,
    pin_memory=True,
    lr_decay_start=15000,
    lr_decay_rate=216000,
    lr_decay_min=1e-5,
    sample_inference_speaker_ids=None,
    sample_inference_text="That quick beige fox jumped in the air loudly over the thin dog fence.",
)

config = DEFAULTS.values()
config.update(MODEL_DEFAULTS.values())
DEFAULTS = HParams(**config)
