from io import BytesIO
import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile

import ray
from ray.air import session, Checkpoint
from ray.air.config import ScalingConfig
import ray.train as train
from ray.train.torch import TorchTrainer
import ray.data
from ray.data.datasource import FastFileMetadataProvider

# from ..trainer.tacotron2 import Tacotron2Trainer, DEFAULTS as TACOTRON2_TRAINER_DEFAULTS
from uberduck_ml_dev.losses import Tacotron2Loss
from uberduck_ml_dev.models.tacotron2 import Tacotron2, DEFAULTS
from uberduck_ml_dev.data.collate import Collate
from uberduck_ml_dev.data.batch import Batch
from uberduck_ml_dev.text.utils import text_to_sequence
from uberduck_ml_dev.text.symbols import NVIDIA_TACO2_SYMBOLS
from uberduck_ml_dev.models.common import MelSTFT

config = DEFAULTS.values()
config["with_gsts"] = False

stft = MelSTFT(
    filter_length=DEFAULTS.filter_length,
    hop_length=DEFAULTS.hop_length,
    win_length=DEFAULTS.win_length,
    n_mel_channels=DEFAULTS.n_mel_channels,
    sampling_rate=DEFAULTS.sampling_rate,
    mel_fmin=DEFAULTS.mel_fmin,
    mel_fmax=DEFAULTS.mel_fmax,
    padding=None,
)

def ray_df_to_batch(df):
    transcripts = df.transcript.tolist()
    audio_bytes_list = df.audio_bytes.tolist()

    collate_fn = Collate(cudnn_enabled=torch.cuda.is_available())
    collate_input = []
    for transcript, audio_bytes in zip(transcripts, audio_bytes_list):
        bio = BytesIO(audio_bytes)
        sr, wav_data = wavfile.read(bio)
        audio = torch.FloatTensor(wav_data)
        audio_norm = audio / (np.abs(audio).max() * 2)
        audio_norm = audio_norm.unsqueeze(0)
        text_sequence = torch.LongTensor(
            text_to_sequence(
                transcript,
                ["english_cleaners"],
                1.0,
                symbol_set=NVIDIA_TACO2_SYMBOLS,
            )
        )
        melspec = stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        collate_input.append(
            dict(
                text_sequence=text_sequence,
                mel=melspec,
            )
        )
    return collate_fn(collate_input)


def get_ray_dataset():
    lj_df = pd.read_csv(
        # "s3://uberduck-audio-files/LJSpeech/metadata.csv",
        "https://uberduck-audio-files.s3.us-west-2.amazonaws.com/LJSpeech/metadata.csv",
        sep="|",
        header=None,
        names=["path", "transcript"],
    )
    lj_df = lj_df # .head(1000)
    paths = ("s3://uberduck-audio-files/LJSpeech/" + lj_df.path).tolist()
    transcripts = lj_df.transcript.tolist()

    audio_ds = ray.data.read_binary_files(
        paths,
        parallelism=len(paths),
        meta_provider=FastFileMetadataProvider(),
    )
    transcripts_ds = ray.data.from_items(transcripts, parallelism=len(transcripts))

    audio_ds = audio_ds.map_batches(lambda x: x, batch_format="pyarrow", batch_size=None)
    transcripts_ds = transcripts_ds.map_batches(lambda x: x, batch_format="pyarrow", batch_size=None)

    output_dataset = transcripts_ds.zip(audio_ds)
    output_dataset = output_dataset.map_batches(lambda table: table.rename(columns={"value": "transcript", "value_1": "audio_bytes"}))
    return output_dataset


def train_func(config: dict):
    print("CUDA AVAILABLE: ", torch.cuda.is_available())
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size_per_worker = batch_size // session.get_world_size()
    is_cuda = torch.cuda.is_available()
    DEFAULTS.cudnn_enabled = is_cuda
    # keep pos_weight higher than 5 to make clips not stretch on
    criterion = Tacotron2Loss(pos_weight=None)
    model = Tacotron2(DEFAULTS)
    model = train.torch.prepare_model(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-6,
    )
    collate_fn = Collate(cudnn_enabled=is_cuda)
    dataset_shard = session.get_dataset_shard("train")
    global_step = 0
    for epoch in range(epochs):
        model.train()
        for ray_batch_df in dataset_shard.iter_batches(batch_size=batch_size):
            global_step += 1
            model.zero_grad()
            model_input = ray_df_to_batch(ray_batch_df)
            model_output = model(
                input_text=model_input["text_int_padded"],
                input_lengths=model_input["input_lengths"],
                speaker_ids=model_input["speaker_ids"],
                embedded_gst=model_input["gst"],
                targets=model_input["mel_padded"],
                audio_encoding=model_input["audio_encodings"],
                output_lengths=model_input["output_lengths"],
            )

            target = model_input.subset(["gate_target", "mel_padded"])
            mel_loss, gate_loss, mel_loss_batch, gate_loss_batch = criterion(
                model_output=model_output, target=target,
            )
            loss = mel_loss + gate_loss
            print(f"Loss: {loss}")
            session.report(dict(loss=loss.item()))
            grad_norm= torch.nn.utils.clip_grad_norm(
                model.parameters(), 1.0
            )
            optimizer.step()

        session.report(
            {},
            checkpoint=Checkpoint.from_dict(
                dict(epoch=epoch, model=model.state_dict())
            )
        )



if __name__ == "__main__":
    print("loading dataset")
    ray_dataset = get_ray_dataset()
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={"lr": 1e-3, "batch_size": 16, "epochs": 1},
        scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
        datasets={"train": ray_dataset},
    )
    result = trainer.fit()
    print(f"Last result: {result.metrics}")