import csv
from io import BytesIO

import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
import wandb

import ray
from ray.air import session, Checkpoint
from ray.air.config import ScalingConfig, RunConfig
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
import ray.data
from ray.data.datasource import FastFileMetadataProvider
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.tune.syncer import SyncConfig


from uberduck_ml_dev.losses import Tacotron2Loss
from uberduck_ml_dev.models.tacotron2 import Tacotron2, DEFAULTS, INFERENCE
from uberduck_ml_dev.data.collate import Collate
from uberduck_ml_dev.data.batch import Batch
from uberduck_ml_dev.text.utils import text_to_sequence
from uberduck_ml_dev.text.symbols import NVIDIA_TACO2_SYMBOLS
from uberduck_ml_dev.trainer.tacotron2 import Tacotron2Trainer
from uberduck_ml_dev.trainer.base import sample
from uberduck_ml_dev.models.common import MelSTFT
from uberduck_ml_dev.utils.plot import save_figure_to_numpy, plot_spectrogram, plot_gate_outputs, plot_attention
# from uberduck_ml_dev.utils.utils import reduce_tensor
from uberduck_ml_dev.monitoring.statistics import get_alignment_metrics
from uberduck_ml_dev.text.utils import random_utterance

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

def sample_inference(model):
    with torch.no_grad():
        transcription = random_utterance()
        utterance = torch.LongTensor(
            text_to_sequence(
                transcription,
                ["english_cleaners"],
                p_arpabet=1.0,
                symbol_set=NVIDIA_TACO2_SYMBOLS,
            )
        )[None]

        input_lengths = torch.LongTensor([utterance.shape[1]])
        speaker_id_tensor = None # torch.LongTensor([speaker_id])

        if torch.cuda.is_available():
            utterance = utterance.cuda()
            input_lengths = input_lengths.cuda()
            gst_embedding = None
            speaker_id_tensor = None # speaker_id_tensor.cuda()

        model.eval()
        model_output = model.forward(
            input_text=utterance,
            input_lengths=input_lengths,
            speaker_ids=speaker_id_tensor,
            # NOTE (Sam): this is None if using old multispeaker training, not None if using new pretrained encoder.
            audio_encoding=None, #speaker_embedding,
            embedded_gst=None, # gst_embedding,
            mode=INFERENCE,
        )
        model.train()
        audio = sample(model_output["mel_outputs_postnet"][0])
        if (audio.size(0) == 1 or audio.size(0) == 2) and audio.size(1) > 2:
            audio = audio.transpose(0, 1)
        return audio





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
        quoting=csv.QUOTE_NONE,
        names=["path", "transcript"],
    )
    # lj_df = lj_df.head(100)
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

def log(
    model,
    X,
    y_pred,
    y,
    loss,
    mel_loss,
    gate_loss,
    mel_loss_batch,
    gate_loss_batch,
    grad_norm,
    epoch,
    global_step,
    steps_per_sample
):
    metrics = {
        "Loss/train": loss.item(),
        "MelLoss/train": mel_loss.item(),
        "GateLoss/train": gate_loss.item(),
        "GradNorm": grad_norm.item(),
        "epoch": epoch,
    }
    batch_levels = X["speaker_ids"]
    batch_levels_unique = torch.unique(batch_levels) if batch_levels else []
    for l in batch_levels_unique:
        mlb = mel_loss_batch[torch.where(batch_levels == l)[0]].mean()
        metrics[f"MelLoss/train/speaker{l.item()}"] = mlb
        glb = gate_loss_batch[torch.where(batch_levels == l)[0]].mean()
        metrics[f"GateLoss/train/speaker{l.item()}"] = glb
        metrics[f"Loss/train/speaker{l.item()}"] = mlb + glb
    session.report(metrics)
    if global_step % steps_per_sample == 0:
        alignment_metrics = get_alignment_metrics(y_pred["alignments"])
        alignment_diagonalness = alignment_metrics["diagonalness"]
        alignment_max = alignment_metrics["max"]
        if session.get_world_rank() == 0 or session.get_world_size() == 1:
            teacher_forced = sample(y_pred["mel_outputs_postnet"][0]).transpose(0, 1)
            sample_audio = sample_inference(model)
            wandb_metrics = dict(metrics)
            input_length = X["input_lengths"][0].item()
            output_length = X["output_lengths"][0].item()
            mel_target = y["mel_padded"][0]
            wandb_metrics.update({
                "AlignmentDiagonalness/train": alignment_diagonalness.item(),
                "AlignmentMax/train": alignment_max.item(),
                "MelPredicted/train": wandb.Image(save_figure_to_numpy(plot_spectrogram(y_pred["mel_outputs_postnet"][0].data.cpu()))),
                "MelTarget/train": wandb.Image(save_figure_to_numpy(plot_spectrogram(mel_target.data.cpu()))),
                "TargetAudio/train": wandb.Audio(sample(mel_target).transpose(0, 1), sample_rate=22050),
                "Gate/train": wandb.Image(
                    save_figure_to_numpy(plot_gate_outputs(
                        gate_targets=y["gate_target"][0].data.cpu(),
                        gate_outputs=y_pred["gate_predicted"][0].data.cpu(),
                    ))
                ),
                "Attention/train": wandb.Image(
                    save_figure_to_numpy(
                        plot_attention(
                            y_pred["alignments"][0].data.cpu().transpose(0, 1),
                            encoder_length=input_length,
                            decoder_length=output_length,
                        )
                    )
                ),
                "SampleInference": wandb.Audio(sample_audio, sample_rate=22050),
                "AudioTeacherForced/train": wandb.Audio(teacher_forced, sample_rate=22050),
            })
            wandb.log(wandb_metrics)



def train_func(config: dict):
    setup_wandb(config, project="voice-cloning")
    print("CUDA AVAILABLE: ", torch.cuda.is_available())
    is_cuda = torch.cuda.is_available()
    DEFAULTS.cudnn_enabled = is_cuda
    DEFAULTS.steps_per_sample = 100
    device = train.torch.get_device()
    DEFAULTS.device = device
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]
    # keep pos_weight higher than 5 to make clips not stretch on
    criterion = Tacotron2Loss(pos_weight=10)
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
        session.report(dict(lr=lr, epochs=epochs, workers=session.get_world_size()))
        for ray_batch_df in dataset_shard.iter_batches(batch_size=batch_size):
            torch.cuda.empty_cache()
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
            loss.backward()
            print(f"Loss: {loss}")
            grad_norm= torch.nn.utils.clip_grad_norm(
                model.parameters(), 1.0
            )
            optimizer.step()
            log(
                model,
                model_input,
                model_output,
                target,
                loss,
                mel_loss,
                gate_loss,
                mel_loss_batch,
                gate_loss_batch,
                grad_norm,
                epoch,
                global_step,
                DEFAULTS.steps_per_sample,
            )

        session.report(
            {},
            checkpoint=Checkpoint.from_dict(
                dict(epoch=epoch, global_step=global_step, model=model.state_dict())
            )
        )



if __name__ == "__main__":
    print("loading dataset")
    ray_dataset = get_ray_dataset()
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={"lr": 1e-3, "batch_size": 24, "epochs": 5},
        scaling_config=ScalingConfig(num_workers=4, use_gpu=True, resources_per_worker=dict(CPU=4, GPU=1)),
        run_config=RunConfig(
            sync_config=SyncConfig(upload_dir="s3://uberduck-anyscale-data/checkpoints")
        ),
        datasets={"train": ray_dataset},
    )
    result = trainer.fit()
    print(f"Last result: {result.metrics}")
