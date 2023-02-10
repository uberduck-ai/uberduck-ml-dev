import pandas as pd
import torch

import ray
from ray.air import session, Checkpoint
from ray.air.config import ScalingConfig
import ray.train as train
from ray.train.torch import TorchTrainer
import ray.data
from ray.data.datasource import FastFileMetadataProvider

# from ..trainer.tacotron2 import Tacotron2Trainer, DEFAULTS as TACOTRON2_TRAINER_DEFAULTS
from uberduck_ml_dev.losses import Tacotron2Loss
from uberduck_ml_dev.models.tacotron2 import Tacotron2, DEFAULTS
from uberduck_ml_dev.data.collate import Collate

ray.init("ray://uberduck-1")


config = DEFAULTS.values()
config["with_gsts"] = False

def get_ray_dataset():
    lj_df = pd.read_csv(
        # "s3://uberduck-audio-files/LJSpeech/metadata.csv",
        "https://uberduck-audio-files.s3.us-west-2.amazonaws.com/LJSpeech/metadata.csv",
        sep="|",
        header=None,
        names=["path", "transcript"],
    )
    lj_df = lj_df.head(100)
    paths = ("s3://uberduck-audio-files/LJSpeech/" + lj_df.path).tolist()
    transcripts = lj_df.transcript.tolist()

    audio_ds = ray.data.read_binary_files(
        paths,
        parallelism=len(paths),
        meta_provider=FastFileMetadataProvider(),
    )
    transcripts_ds = ray.data.from_items(transcripts, parallelism=len(transcripts))

    audio_ds = audio_ds.map_batches(lambda x: x, batch_format="pyarrow", batch_size=None)
    transcripts_ds = transcripts_ds.map_batches(lambda x: x, batch_format="pyarrow", batch_size=None)

    output_dataset = transcripts_ds.zip(audio_ds)
    return output_dataset


def train_func(config: dict):
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size_per_worker = batch_size // session.get_world_size()
    is_cuda = torch.cuda.is_available()

    criterion = Tacotron2Loss(
        pos_weight=DEFAULTS.pos_weight
    )  # keep higher than 5 to make clips not stretch on
    model = Tacotron2(DEFAULTS)
    model = train.torch.prepare_model(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=DEFAULTS.weight_decay,
    )
    collate_fn = Collate(cudnn_enabled=is_cuda)
    dataset_shard = session.get_dataset_shard("train")
    for epoch in range(epochs):
        model.train()
        for batches in dataset_shard.iter_torch_batches(batch_size=batch_size):
            print("batch")
        session.report(
            {},
            checkpoint=Checkpoint.from_dict(
                dict(epoch=epoch, model=model.state_dict())
            )
        )



if __name__ == "__main__":
    print("loading dataset")
    ray_dataset = get_ray_dataset()
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={"lr": 1e-3, "batch_size": 16, "epochs": 1},
        scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
        datasets={"train": ray_dataset},
    )
    result = trainer.fit()
    print(f"Last result: {result.metrics}")