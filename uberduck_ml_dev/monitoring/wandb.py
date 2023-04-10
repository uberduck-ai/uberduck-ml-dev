from typing import List

import wandb
from tqdm import tqdm
import torch

from ..text.utils import UTTERANCES


def log_sample_utterances(
    project="my-project",
    name="my-model",
    dataset="my-dataset",
    architecture="my-architecture",
    speaker_ids: List = [],
    inference_function=lambda text, speaker_id: False,
):

    wandb.init(
        project=project,
        name=name,
        job_type="eval",
        config={"architecture": architecture, "dataset": dataset},
    )

    with torch.no_grad():
        for speaker_id in tqdm(speaker_ids):
            to_log = []
            for utterance in tqdm(UTTERANCES):
                inference = inference_function(utterance, speaker_id)
                to_log.append(
                    wandb.Audio(inference, caption=utterance, sample_rate=22050)
                )
                torch.cuda.empty_cache()  # might not be necessary
            wandb.log({f"Speaker {speaker_id}": to_log})

    wandb.finish()
