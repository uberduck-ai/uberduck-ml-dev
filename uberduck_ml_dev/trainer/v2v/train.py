import torch
from torch.cuda.amp import GradScaler
from ray.air.integrations.wandb import setup_wandb
import ray.train as train

from .train_epoch import train_epoch
from .load import prepare_dataloaders, warmstart
from ...models.radtts import RADTTS
from ...losses import RADTTSLoss, AttentionBinarizationLoss
from ...optimizers.radam import RAdam
from ...vocoders.hifigan import get_vocoder


def train_func(model, optim, criterion, project: str, config: dict):
    setup_wandb(config, project=project, entity="uberduck-ai", rank_zero_only=False)
    train_config = config["train_config"]
    model_config = config["model_config"]
    data_config = config["data_config"]

    print("CUDA AVAILABLE: ", torch.cuda.is_available())
    epochs = train_config["epochs"]
    steps_per_sample = train_config["steps_per_sample"]
    sigma = train_config["sigma"]
    kl_loss_start_iter = train_config["kl_loss_start_iter"]
    binarization_start_iter = train_config["binarization_start_iter"]

    if train_config["warmstart_checkpoint_path"] != "":
        warmstart(train_config["warmstart_checkpoint_path"], model)

    # NOTE (Sam): find_unused_parameters=True is necessary for num_workers >1 in ScalingConfig.
    model = train.torch.prepare_model(
        model, parallel_strategy_kwargs=dict(find_unused_parameters=True)
    )

    start_epoch = 0

    # NOTE (Sam): uncomment to run with torch DataLoader rather than ray dataset
    train_loader, valset, collate_fn = prepare_dataloaders(data_config, 2, 6)
    train_dataloader = train.torch.prepare_data_loader(train_loader)

    scaler = GradScaler()
    logging_parameters = [
        train_config["log_decoder_samples"],
        train_config["log_attribute_samples"],
        train_config["iters_per_checkpoint"],
        train_config["output_directory"],
        steps_per_sample,
    ]
    optimization_parameters = [
        optim,
        scaler,
        criterion,
        attention_kl_loss,
        kl_loss_start_iter,
        binarization_start_iter,
    ]
    attention_kl_loss = AttentionBinarizationLoss()
    iteration = 0
    for epoch in range(start_epoch, start_epoch + epochs):
        iteration = train_epoch(
            train_dataloader,
            logging_parameters,
            model,
            optimization_parameters,
            iteration,
        )
