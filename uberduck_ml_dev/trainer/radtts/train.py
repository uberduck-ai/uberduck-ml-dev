import torch
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ExponentialLR
from ray.air.integrations.wandb import setup_wandb
from ray.air import session
import ray.train as train

from .train_epoch import train_epoch
from .load import prepare_dataloaders
from ...models.radtts import RADTTS
from ...trainer.radtts.load import warmstart
from ...losses import RADTTSLoss, AttentionBinarizationLoss
from ...optimizers.radam import RAdam
from ...vocoders.hifigan import get_vocoder


def train_func(config: dict):
    setup_wandb(
        config, project="radtts-ray", entity="uberduck-ai", rank_zero_only=False
    )
    train_config = config["train_config"]
    model_config = config["model_config"]
    data_config = config["data_config"]
    vocoder = get_vocoder(
        hifi_gan_config_path=train_config["vocoder_config_path"],
        hifi_gan_checkpoint_path=train_config["vocoder_checkpoint_path"],
    )
    print("CUDA AVAILABLE: ", torch.cuda.is_available())
    epochs = train_config["epochs"]
    steps_per_sample = train_config["steps_per_sample"]
    sigma = train_config["sigma"]
    kl_loss_start_iter = train_config["kl_loss_start_iter"]
    binarization_start_iter = train_config["binarization_start_iter"]

    model = RADTTS(
        **model_config,
    )

    if train_config["warmstart_checkpoint_path"] != "":
        warmstart(train_config["warmstart_checkpoint_path"], model)

    # NOTE (Sam): find_unused_parameters=True is necessary for num_workers >1 in ScalingConfig.
    # model = train.torch.prepare_model(model)
    model = train.torch.prepare_model(
        model, parallel_strategy_kwargs=dict(find_unused_parameters=True)
    )

    start_epoch = 0

    # NOTE (Sam): uncomment to run with torch DataLoader rather than ray dataset
    train_loader, valset, collate_fn = prepare_dataloaders(data_config, 2, 6)
    train_dataloader = train.torch.prepare_data_loader(train_loader)

    optim = RAdam(
        model.parameters(),
        train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )
    scheduler = ExponentialLR(
        optim,
        train_config["weight_decay"],
        last_epoch=-1,
    )
    # dataset_shard = session.get_dataset_shard("train")
    scaler = GradScaler()

    criterion = RADTTSLoss(
        sigma,
        model_config["n_group_size"],
        model_config["dur_model_config"],
        model_config["f0_model_config"],
        model_config["energy_model_config"],
        vpred_model_config=model_config["v_model_config"],
        loss_weights=train_config["loss_weights"],
    )
    attention_kl_loss = AttentionBinarizationLoss()
    iteration = 0
    for epoch in range(start_epoch, start_epoch + epochs):
        # NOTE (Sam): uncomment to run with torch DataLoader rather than ray dataset
        iteration = train_epoch(
            train_dataloader,
            train_config["log_decoder_samples"],
            train_config["log_attribute_samples"],
            model,
            optim,
            steps_per_sample,
            scaler,
            train_config["iters_per_checkpoint"],
            train_config["output_directory"],
            criterion,
            attention_kl_loss,
            kl_loss_start_iter,
            binarization_start_iter,
            iteration,
            vocoder,
        )
        # iteration = train_epoch(dataset_shard, batch_size, model, optim, steps_per_sample, scaler, scheduler, criterion, attention_kl_loss, kl_loss_start_iter, binarization_start_iter, epoch, iteration)


DEFAULTS = {
    "output_directory": "",
    "epochs": 10000000,
    "optim_algo": "RAdam",
    "learning_rate": 1e-4,
    "weight_decay": 1e-6,
    "sigma": 1.0,
    "iters_per_checkpoint": 50000,
    "batch_size": 32,
    "seed": 1234,
    "checkpoint_path": "",
    "ignore_layers": [],
    "ignore_layers_warmstart": [],
    "steps_per_sample": 10000,
    "include_layers": [],
    "vocoder_config_path": "hifigan_22khz_config.json",
    "vocoder_checkpoint_path": "hifigan_libritts100360_generator0p5.pt",
    "log_attribute_samples": False,
    "log_decoder_samples": False,
    "warmstart_checkpoint_path": "",  # NOTE (Sam): this also "loads_checkpoint" (i.e. no optimizer state initialization).
    "use_amp": True,
    "grad_clip_val": 1.0,
    "loss_weights": {
        "blank_logprob": -1,
        "ctc_loss_weight": 0.1,
        "binarization_loss_weight": 1.0,
        "dur_loss_weight": 1.0,
        "f0_loss_weight": 1.0,
        "energy_loss_weight": 1.0,
        "vpred_loss_weight": 1.0,
    },
    "binarization_start_iter": 18000,
    "kl_loss_start_iter": 40000,
    "unfreeze_modules": "all",
}
