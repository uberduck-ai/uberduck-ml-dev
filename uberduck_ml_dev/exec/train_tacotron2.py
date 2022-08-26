__all__ = ["parse_args", "run"]

from ..trainer.tacotron2 import Tacotron2Trainer
from ..vendor.tfcompat.hparam import HParams
from ..trainer.tacotron2 import DEFAULTS as TACOTRON2_TRAINER_DEFAULTS
import argparse
import sys
import json
import torch
from torch import multiprocessing as mp


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to JSON config")
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


try:
    from nbdev.imports import IN_NOTEBOOK
except:
    IN_NOTEBOOK = False
if __name__ == "__main__" and not IN_NOTEBOOK:
    args = parse_args(sys.argv[1:])
    config = TACOTRON2_TRAINER_DEFAULTS.values()
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
