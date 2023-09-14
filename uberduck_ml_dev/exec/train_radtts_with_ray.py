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
from uberduck_ml_dev.trainer.radtts.train import DEFAULTS as TRAIN_CONFIG
from uberduck_ml_dev.data.data import RADTTS_DEFAULTS as DATA_CONFIG
from uberduck_ml_dev.models.radtts import DEFAULTS as MODEL_CONFIG

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    if args.config:
        with open(args.config) as f:
            config_inputs = json.load(f)

    config = dict(
        train_config=TRAIN_CONFIG, data_config=DATA_CONFIG, model_config=MODEL_CONFIG
    )
    config["train_config"].update(config_inputs["train_config"])
    config["data_config"].update(config_inputs["data_config"])
    config["model_config"].update(config_inputs["model_config"])

    os.makedirs(config["train_config"]["output_directory"], exist_ok=True)
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
