# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/exec.train_tacotron2.ipynb (unless otherwise specified).

__all__ = ['parse_args', 'run']

# Cell
from ..trainer.tacotron2 import Tacotron2Trainer
from ..vendor.tfcompat.hparam import HParams
from ..trainer.tacotron2 import DEFAULTS as TACOTRON2_TRAINER_DEFAULTS
import argparse
import sys
import json
import torch
from torch import multiprocessing as mp

# Cell
def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to JSON config")
    parser.add_argument(
        "--training_audiopaths_and_text", help="Path to training filelist", default=None
    )
    parser.add_argument(
        "--val_audiopaths_and_text", help="Path to val filelist", default=None
    )
    parser.add_argument("--log_dir", help="Path to log_dir", default=None)
    parser.add_argument(
        "--checkpoint_path", help="Path to checkpoint_path", default=None
    )
    parser.add_argument(
        "--warm_start_name", help="Path to warm start model", default=None
    )
    args = parser.parse_args(args)
    return args


def run(rank, device_count, hparams):
    trainer = Tacotron2Trainer(hparams, rank=rank, world_size=device_count)
    try:
        trainer.train()
    except Exception as e:
        print(f"Exception raised while training: {e}")
        # TODO: save state.
        raise e

# Cell
try:
    from nbdev.imports import IN_NOTEBOOK
except:
    IN_NOTEBOOK = False
if __name__ == "__main__" and not IN_NOTEBOOK:
    args = parse_args(sys.argv[1:])
    config = TACOTRON2_TRAINER_DEFAULTS
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))
    config.update(vars(args))
    hparams = HParams(**config)
    if hparams.distributed_run:
        device_count = torch.cuda.device_count()
        mp.spawn(run, (device_count, hparams), device_count)
    else:
        run(None, None, hparams)