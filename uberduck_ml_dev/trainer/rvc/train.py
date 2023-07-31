import torch
from torch.cuda.amp import GradScaler
from ray.air.integrations.wandb import setup_wandb
from torch.utils.data import DataLoader
from torch.nn import functional as F

from .train_epoch import train_epoch
from ...models.rvc.rvc import (
    SynthesizerTrnMs256NSFsid,
    MultiPeriodDiscriminator,
)
from ...vendor.tfcompat.hparam import HParams
from ...data.data import (
    TextAudioLoaderMultiNSFsid,
    DistributedBucketSampler,
)
from ...data.collate import TextAudioCollateMultiNSFsid
from ...losses_rvc import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss,
)
from uberduck_ml_dev.trainer.rvc.train_epoch import train_epoch


def train_func(config: dict, project: str = "rvc"):
    print("Entering training function")
    setup_wandb(config, project=project, entity="uberduck-ai", rank_zero_only=False)
    train_config = config["train"]
    model_config = config["model"]
    data_config = config["data"]

    generator = SynthesizerTrnMs256NSFsid(
        data_config["filter_length"] // 2 + 1,
        train_config["segment_size"] // data_config["hop_length"],
        **model_config,
        is_half=train_config["fp16_run"],
        sr=data_config["sampling_rate"],
    )

    discriminator = MultiPeriodDiscriminator(model_config["use_spectral_norm"])
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
    generator_checkpoint = torch.load(train_config["warmstart_G_checkpoint_path"])[
        "model"
    ]
    discriminator_checkpoint = torch.load(train_config["warmstart_D_checkpoint_path"])[
        "model"
    ]
    discriminator.load_state_dict(discriminator_checkpoint)
    generator.load_state_dict(
        generator_checkpoint, strict=False
    )  # NOTE (Sam): a handful of "enc_q" decoder states not present - doesn't seem to cause an issue
    generator = generator.cuda()
    discriminator = discriminator.cuda()

    # # lets just train the hifi part
    # for p in discriminator.parameters():
    #     p.requires_grad = False

    # for p in generator.enc_q.parameters():
    #     p.requires_grad = False

    # for p in generator.enc_p.parameters():
    #     p.requires_grad = False

    # for p in generator.emb_g.parameters():
    #     p.requires_grad = False

    # for p in generator.flow.parameters():
    #     p.requires_grad = False

    models = {"generator": generator, "discriminator": discriminator}

    print("Loading dataset")
    train_dataset = TextAudioLoaderMultiNSFsid(
        train_config["filelist_path"], HParams(**data_config)
    )  # dv is sid
    collate_fn = TextAudioCollateMultiNSFsid()
    n_gpus = 1
    train_sampler = DistributedBucketSampler(
        train_dataset,
        train_config["batch_size"] * n_gpus,
        [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
        num_replicas=n_gpus,
        rank=0,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
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
            train_loader,
            config,
            models,
            optimization_parameters,
            logging_parameters={},
            iteration=iteration,
        )


# 40k config
DEFAULTS = {
    "log_interval": 200,
    "seed": 1234,
    "epochs": 20000,
    "learning_rate": 1e-4,
    "betas": [0.8, 0.99],
    "eps": 1e-9,
    "batch_size": 4,
    "fp16_run": False,
    "lr_decay": 0.999875,
    "segment_size": 12800,
    "init_lr_ratio": 1,
    "warmup_epochs": 0,
    "c_mel": 45,
    "c_kl": 1.0,
    "steps_per_sample": 100,
    "iters_per_checkpoint": 100,
    "output_directory": "/tmp",
}
