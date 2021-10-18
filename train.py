import argparse
import json
import librosa  # NOTE(zach): importing torch before librosa causes LLVM issues for some unknown reason.
import sys

import torch
from torch import multiprocessing as mp

from uberduck_ml_dev.trainer.base import TTSTrainer, MellotronTrainer
from uberduck_ml_dev.vendor.tfcompat.hparam import HParams
from uberduck_ml_dev.models.mellotron import DEFAULTS as MELLOTRON_DEFAULTS


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to JSON config")
    args = parser.parse_args(args)
    return args


def run(rank, device_count, hparams):
    trainer = MellotronTrainer(hparams, rank=rank, world_size=device_count)
    try:
        trainer.train()
    except Exception as e:
        print(f"Exception raised while training: {e}")
        
        raise e

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    config = MELLOTRON_DEFAULTS.values()
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))
    hparams = HParams(**config)
    if hparams.distributed_run:
        device_count = torch.cuda.device_count()
        mp.spawn(run, (device_count, hparams), device_count)
