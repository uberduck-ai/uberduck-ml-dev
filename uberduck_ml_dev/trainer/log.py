import torch
import wandb
from ray.air import session

@torch.no_grad()
def log(metrics, audios={}):
    # pass
    wandb_metrics = dict(metrics)

    for k, v in audios.items():
        wandb_metrics[k] = wandb.Audio(v, sample_rate=22050)

    # session.report(metrics)
    if session.get_world_rank() == 0:
        wandb.log(wandb_metrics)
