from io import BytesIO
import os

from scipy.io import wavfile
import torch
import numpy as np
import ray
import pandas as pd


from .utils import get_energy_average, f0_normalize
from ..models.components.encoders import ResNetSpeakerEncoderCallable

# NOTE (Sam): the ray dataset code runs mod cleanup but is seemingly slower than torch dataloader (not 100p sure if this is still true).
def ray_df_preprocessing(df, data_config, tp, stft):
    transcripts = df.transcript.tolist()
    audio_bytes_list = df.audio_bytes.tolist()
    speaker_ids = df.speaker_id.tolist()
    f0_paths = df.f0_path.tolist()
    audio_embeddings = df.audio_embedding.tolist()
    # shuffle_indices = get_shuffle_indices(speaker_ids)
    # audio_embeddings = [audio_embeddings[i] for i in shuffle_indices]
    collate_input = []
    for transcript, audio_bytes, speaker_id, f0_path, audio_embedding in zip(
        transcripts, audio_bytes_list, speaker_ids, f0_paths, audio_embeddings
    ):
        bio = BytesIO(audio_bytes)
        sr, wav_data = wavfile.read(bio)
        audio = torch.FloatTensor(wav_data)
        # NOTE (Sam): why normalize here?
        audio_norm = audio / (np.abs(audio).max() * 2)
        text_sequence = tp.get_text(transcript)
        mel = stft.get_mel(audio_norm, data_config["max_wav_value"])
        mel = torch.squeeze(mel, 0)
        dikt = torch.load(f0_path)
        f0 = dikt["f0"]
        p_voiced = dikt["p_voiced"]
        voiced_mask = dikt["voiced_mask"]
        f0 = f0_normalize(f0, f0_min=data_config["f0_min"])
        energy_avg = get_energy_average(mel)
        prior_path = "{}_{}".format(text_sequence.shape[0], mel.shape[1])
        prior_path = os.path.join("/usr/src/app/radtts/data_cache", prior_path)
        prior_path += "_prior.pth"
        attn_prior = torch.load(prior_path)
        speaker_id = torch.LongTensor([speaker_id])
        audio_embedding = torch.FloatTensor(audio_embedding)
        # NOTE (Sam): might be faster to return dictionary arrays of batched inputs instead of list
        collate_input.append(
            {
                "text_encoded": text_sequence,
                "mel": mel,
                "speaker_id": speaker_id,
                "f0": f0,
                "p_voiced": p_voiced,
                "voiced_mask": voiced_mask,
                "energy_avg": energy_avg,
                "attn_prior": attn_prior,
                "audiopath": None,
                "audio_embedding": audio_embedding,
            }
        )

    return collate_input


def get_ray_dataset(filelist_path, config_path, model_path):

    df = pd.read_csv(
        filelist_path,
        sep="|",
        header=None,
        quoting=3,
        names=["path", "transcript", "speaker_id", "f0_path", "emb_path"],
    )

    paths = df.path.tolist()
    transcripts = df.transcript.tolist()
    speaker_ids = df.speaker_id.tolist()

    pitches = df.f0_path.tolist()

    parallelism_length = 400
    audio_ds = ray.data.read_binary_files(
        paths,
        parallelism=parallelism_length,
        ray_remote_args={"num_cpus": 1.0},
    )
    audio_ds = audio_ds.map_batches(
        lambda x: x, batch_format="pyarrow", batch_size=None
    )

    paths_ds = ray.data.from_items(paths, parallelism=parallelism_length)
    paths_ds = paths_ds.map_batches(
        lambda x: x, batch_format="pyarrow", batch_size=None
    )

    transcripts = ray.data.from_items(transcripts, parallelism=parallelism_length)
    transcripts_ds = transcripts.map_batches(
        lambda x: x, batch_format="pyarrow", batch_size=None
    )

    speaker_ids_ds = ray.data.from_items(speaker_ids, parallelism=parallelism_length)
    speaker_ids_ds = speaker_ids_ds.map_batches(
        lambda x: x, batch_format="pyarrow", batch_size=None
    )
    pitches_ds = ray.data.from_items(pitches, parallelism=parallelism_length)
    pitches_ds = pitches_ds.map_batches(
        lambda x: x, batch_format="pyarrow", batch_size=None
    )

    embs_ds = ray.data.from_items(paths, parallelism=parallelism_length)
    embs_ds = embs_ds.map_batches(
        ResNetSpeakerEncoderCallable,
        fn_kwargs = {"config_path": config_path, "model_path": model_path},
        num_gpus=1.0,
        compute="actors",
    )

    output_dataset = (
        transcripts_ds.zip(audio_ds)
        .zip(paths_ds)
        .zip(speaker_ids_ds)
        .zip(pitches_ds)
        .zip(embs_ds)
    )
    output_dataset = output_dataset.map_batches(
        lambda table: table.rename(
            columns={
                "value": "transcript",
                "value_1": "audio_bytes",
                "value_2": "path",
                "value_3": "speaker_id",
                "value_4": "f0_path",
                "value_5": "emb_path",
            }
        )
    )

    processed_dataset = output_dataset.map_batches(ray_df_preprocessing)
    return processed_dataset.fully_executed()
