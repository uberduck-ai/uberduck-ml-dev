import torch
from torch.cuda.amp import GradScaler
from ray.air.integrations.wandb import setup_wandb
from torch.utils.data import DataLoader
from torch.nn import functional as F

from ...models.rvc.rvc import (
    MultiPeriodDiscriminator,
)

# from ...vendor.tfcompat.hparam import HParams

from ...data.collate import Collate  # TextAudioCollateMultiNSFsid
from ...losses_rvc import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss,
)
from .train_epoch import train_epoch
from .train_step import train_step


def train_func(config: dict, project: str = "rvc"):
    print("Entering training function")
    # setup_wandb(config, project=project, entity="uberduck-ai", rank_zero_only=False)
    train_config = config["train"]
    model_config = config["model"]
    data_config = config["data"]

    from uberduck_ml_dev.models.hifigan import _load_uninitialized

    generator = _load_uninitialized(config_overrides=model_config)

    # discriminator = MultiPeriodDiscriminator(model_config["use_spectral_norm"])

    # RVC uses MultiPeriodDiscrimator that has a single scale discriminator
    # multi_period_discriminator = MultiPeriodDiscriminator().to("cuda")
    discriminator = MultiPeriodDiscriminator().to("cuda")
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
            "model"
        ]
        generator.load_state_dict(
            generator_checkpoint, strict=False
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
    # trainset = MelDataset(
    #     data_config["filelist"],
    #     data_config["segment_size"],
    #     data_config["n_fft"],
    #     data_config["num_mels"],
    #     data_config["hop_size"],
    #     data_config["win_size"],
    #     data_config["sampling_rate"],
    #     data_config["fmin"],
    #     data_config["fmax"],
    #     n_cache_reuse=0,
    #     shuffle=False if train_config["num_gpus"] > 1 else True,
    #     fmax_loss=hi.fmax_for_loss,
    #     device="cuda",
    #     # fine_tuning=a.fine_tuning,
    #     # base_mels_path=a.input_mels_dir,
    # )

    # train_dataset = TextAudioLoaderMultiNSFsid(
    #     train_config["filelist_path"], HParams(**data_config)
    # )  # dv is sid
    # collate_fn = TextAudioCollateMultiNSFsid()
    # n_gpus = 1
    # train_sampler = DistributedBucketSampler(
    #     train_dataset,
    #     train_config["batch_size"] * n_gpus,
    #     [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
    #     num_replicas=n_gpus,
    #     rank=0,
    #     shuffle=True,
    # )
    from uberduck_ml_dev.data.data import BasicDataset

    train_dataset = BasicDataset(
        filelist_path=data_config["filelist_path"],
        mel_suffix=data_config["mel_suffix"],
        audio_suffix=data_config["audio_suffix"],
    )
    train_loader = DataLoader(
        train_dataset,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        collate_fn=Collate(),
        batch_sampler=None,
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
            "kl": {"loss": kl_loss, "weight": 1.0},
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


from ..rvc.train import DEFAULTS as DEFAULTS
