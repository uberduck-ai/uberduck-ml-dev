from typing import List

import wandb
from tqdm import tqdm
import torch

from ..text.utils import UTTERANCES

def log_sample_utterance(text = "I do everything, from fly fishing to badminton", speaker_id = "0",  inference_function=lambda text, speaker_id: False):

    inference = inference_function(text, speaker_id)
    return inference
    # wandb.log({f"speaker_{speaker_id}_text_{text[:20]}": wandb.Audio(inference, caption="audio", sample_rate=22050)})
    # wandb.log({"audio": wandb.Audio(inference, caption="audio", sample_rate=22050), "speaker_id": speaker_id, "text": text})
    # torch.cuda.empty_cache()

def log_sample_utterances(project = "my-project", name = "my-model", dataset = "my-dataset", architecture = "my-architecture", speaker_ids: List = [], inference_function=lambda text, speaker_id: False):

    wandb.init(project=project, name = name, job_type = "eval", config = {"architecture": architecture, "dataset": dataset})
    
    for speaker_id in tqdm(speaker_ids):
        to_log = []
        for utterance in tqdm(UTTERANCES):
            # log_sample_utterance(text = utterance, speaker_id = speaker_id , inference_function = inference_function)
            # inference = log_sample_utterance(text = utterance, speaker_id = speaker_id , inference_function = inference_function)
            inference = inference_function(utterance, speaker_id)
            to_log.append(wandb.Audio(inference, caption=f"audio_{utterance[:20]}", sample_rate=22050))
        wandb.log({f"audios_{speaker_id}": to_log})

    wandb.finish()