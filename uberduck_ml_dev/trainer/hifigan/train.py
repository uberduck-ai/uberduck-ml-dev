import torch
from torch.cuda.amp import GradScaler
from ray.air.integrations.wandb import setup_wandb
from torch.utils.data import DataLoader
from torch.nn import functional as F

from ...data.data import BasicDataset
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

# from ...models.common import TacotronSTFT


def train_func(config: dict, project: str = "rvc"):
    print("Entering training function")
    setup_wandb(config, project=project, entity="uberduck-ai", rank_zero_only=False)
    train_config = config["train"]
    model_config = config["model"]
    data_config = config["data"]

    from uberduck_ml_dev.models.hifigan import _load_uninitialized

    generator = _load_uninitialized(config_overrides=model_config)

    # discriminator = MultiPeriodDiscriminator(model_config["use_spectral_norm"])
    discriminator = MultiDiscriminator(True)
    # discriminator = MultiDiscriminator()
    discriminator = discriminator.to("cuda")
    # RVC uses MultiPeriodDiscrimator that has a single scale discriminator
    # multi_period_discriminator = MultiPeriodDiscriminator().to("cuda")
    # discriminator = MultiPeriodDiscriminator().to("cuda")
    # msd = MultiScaleDiscriminator().to(device)

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
            # "model"
            "generator"
        ]
        generator.load_state_dict(
            generator_checkpoint  # , False
        )  # NOTE (Sam): a handful of "enc_q" decoder states not present - doesn't seem to cause an issue
    if train_config["warmstart_D_checkpoint_path"] is not None:
        discriminator_checkpoint = torch.load(
            train_config["warmstart_D_checkpoint_path"]
        )["model"]
        discriminator.load_state_dict(discriminator_checkpoint)

    # for testing purposes
    # hifigan_path = "/usr/src/app/uberduck_ml_exp/models/g_hifi_crust"
    # hifigan_state_dict = torch.load(hifigan_path)
    # generator.load_state_dict(hifigan_state_dict["generator"])
    # hifigan_path = "/usr/src/app/uberduck_ml_exp/models/do_00000000"
    # hifigan_state_dict = torch.load(hifigan_path)
    # mpd_dict = hifigan_state_dict["mpd"]
    # new_dict = {
    #     key.replace("discriminators.", ""): value for key, value in mpd_dict.items()
    # }
    # discriminator.mpd.load_state_dict(new_dict)
    # msd_dict = hifigan_state_dict["msd"]
    # new_dict = {
    #     key.replace("discriminators.", ""): value for key, value in msd_dict.items()
    # }
    # discriminator.msd.load_state_dict(new_dict)
    # dict1 = hifigan_state_dict["mpd"]
    # dict1.update(hifigan_state_dict["msd"])

    generator = generator.cuda()
    discriminator = discriminator.cuda()
    # stft = TacotronSTFT(
    #     filter_length=data_config["filter_length"],
    #     hop_length=data_config["hop_length"],
    #     win_length=data_config["win_length"],
    #     sampling_rate=data_config["sampling_rate"],
    #     n_mel_channels=data_config["n_mel_channels"],
    #     mel_fmin=data_config["mel_fmin"],
    #     mel_fmax=data_config["mel_fmax"],
    # )
    # stft = stft.cuda()
    # models = {"generator": generator, "discriminator": discriminator, "stft": stft}
    models = {"generator": generator, "discriminator": discriminator}
    print("Loading dataset")

    train_dataset = BasicDataset(
        filelist_path=data_config["filelist_path"],
        mel_suffix=data_config["mel_suffix"],
        audio_suffix=data_config["audio_suffix"],
    )
    from ...data.data import DistributedBucketSampler

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
            # "l1": {"loss": F.l1_loss, "weight": 1.0},
            # "kl": {"loss": kl_loss, "weight": 0.0},
            # "feature": {"loss": feature_loss, "weight": 0.0},
            # "generator": {"loss": generator_loss, "weight": 0.0},
            # "discriminator": {"loss": discriminator_loss, "weight": 0.0},
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
