import torch
import wandb
from ray.air import session


@torch.no_grad()
def log(metrics=None, audios=None, images=None):
    if session.get_world_rank() != 0:
        return
    audios = audios or {}
    images = images or {}
    wandb_metrics = {}
    if metrics is not None:
        wandb_metrics.update(metrics)

    for k, v in audios.items():
        wandb_metrics[k] = wandb.Audio(
            v["audio"].cpu(), sample_rate=22050, caption=v.get("caption")
        )

    for k, v in images.items():
        wandb_metrics[k] = wandb.Image(v)

    wandb.log(wandb_metrics)
