import sys
import json
import os

from ray.air.config import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer
from ray.tune import SyncConfig
from ray.train.torch import TorchTrainer, TorchTrainer
from ray.air.config import ScalingConfig, RunConfig

from uberduck_ml_dev.trainer.radtts.train import train_func
from uberduck_ml_dev.utils.exec import parse_args

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    os.makedirs(config["train_config"]["output_dir"], exist_ok=True)
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=config,
        scaling_config=ScalingConfig(
            num_workers=config["train_config"]["n_gpus"],
            use_gpu=True,
            resources_per_worker=dict(
                CPU=config["data_config"]["num_workers"],
                GPU=1,
            ),
        ),
        run_config=RunConfig(sync_config=SyncConfig()),
    )

    result = trainer.fit()
