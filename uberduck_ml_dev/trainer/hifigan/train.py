import torch
from torch.cuda.amp import GradScaler
from ray.air.integrations.wandb import setup_wandb
from torch.utils.data import DataLoader
from torch.nn import functional as F

from ...data.data import Dataset
from ...models.rvc.rvc import MultiPeriodDiscriminator
from ...models.hifigan import MultiDiscriminator

from ...data.collate import Collate
from ...losses_rvc import (
    generator_loss,
    discriminator_loss,
    feature_loss,
)
from .train_epoch import train_epoch
from .train_step import train_step
from ..rvc.train import DEFAULTS as DEFAULTS
from ...models.hifigan import _load_uninitialized


def train_func(config: dict, project: str = "rvc"):
    print("Entering training function")
    setup_wandb(config, project=project, entity="uberduck-ai", rank_zero_only=False)
    train_config = config["train"]
    model_config = config["model"]
    data_config = config["data"]

    generator = _load_uninitialized(config_overrides=model_config)

    # NOTE (Sam): RVC uses MultiPeriodDiscrimator that has a single scale discriminator
    # HiFi++ paper indicates that the precise discriminator structure is not important and that reweighting the loss is sufficient
    # Vocos uses additional strcuture.
    discriminator = MultiDiscriminator(True)
    discriminator = discriminator.to("cuda")

    generator_optimizer = torch.optim.AdamW(
        generator.parameters(),
        train_config["learning_rate"],
        betas=train_config["betas"],
        eps=train_config["eps"],
    )

    discriminator_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        train_config["learning_rate"],
        betas=train_config["betas"],
        eps=train_config["eps"],
    )

    print("Loading checkpoints")
    # TODO (Sam): move to "warmstart" or "load_checkpoint" functions
    if train_config["warmstart_G_checkpoint_path"] is not None:
        generator_checkpoint = torch.load(train_config["warmstart_G_checkpoint_path"])[
            "generator"
        ]
        generator.load_state_dict(
            generator_checkpoint
        )  # NOTE (Sam): a handful of "enc_q" decoder states not present - doesn't seem to cause an issue
    if train_config["warmstart_D_checkpoint_path"] is not None:
        discriminator_checkpoint = torch.load(
            train_config["warmstart_D_checkpoint_path"]
        )["model"]
        discriminator.load_state_dict(discriminator_checkpoint)

    generator = generator.cuda()
    discriminator = discriminator.cuda()

    models = {"generator": generator, "discriminator": discriminator}
    print("Loading dataset")

    train_dataset = Dataset(
        filelist_path=data_config["filelist_path"],
        mel_suffix=data_config["mel_suffix"],
        audio_suffix=data_config["audio_suffix"],
    )

    # train_sampler = DistributedBucketSampler(
    #     train_dataset,
    #     train_config["batch_size"] * 1,
    #     [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
    #     num_replicas=1,
    #     rank=0,
    #     shuffle=True,
    # )
    train_loader = DataLoader(
        train_dataset,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        collate_fn=Collate(),
        batch_sampler=None,
        # batch_sampler=train_sampler,
        batch_size=train_config["batch_size"],
        persistent_workers=True,
        prefetch_factor=8,
    )
    optimization_parameters = {
        "optimizers": {
            "generator": generator_optimizer,
            "discriminator": discriminator_optimizer,
        },
        "scaler": GradScaler(),
        # NOTE (Sam): need to pass names rather than vector of losses since arguments differ
        "losses": {
            "l1": {"loss": F.l1_loss, "weight": 1.0},
            "feature": {"loss": feature_loss, "weight": 1.0},
            "generator": {"loss": generator_loss, "weight": 1.0},
            "discriminator": {"loss": discriminator_loss, "weight": 1},
        },
    }

    iteration = 0
    start_epoch = 0
    print("Beginning training for ", train_config["epochs"], " epochs")
    for epoch in range(start_epoch, train_config["epochs"]):
        print(f"Epoch: {epoch}")
        iteration = train_epoch(
            train_step,
            train_loader,
            config,
            models,
            optimization_parameters,
            logging_parameters={},
            iteration=iteration,
        )
