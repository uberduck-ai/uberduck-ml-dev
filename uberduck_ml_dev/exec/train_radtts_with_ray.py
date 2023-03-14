import tempfile
from io import BytesIO

import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
import wandb
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ExponentialLR

import ray
from ray.air import session, Checkpoint
from ray.air.config import ScalingConfig, RunConfig
from ray.air.integrations.wandb import  setup_wandb
import ray.data
# from ray.data.datasource import FastFileMetadataProvider
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air.util.check_ingest import DummyTrainer
from ray.tune import SyncConfig


# from uberduck_ml_dev.models import vits
# from uberduck_ml_dev.models.vits import *
from uberduck_ml_dev.models.radtts import RADTTS
from uberduck_ml_dev.text.utils import text_to_sequence
from uberduck_ml_dev.text.symbols import NVIDIA_TACO2_SYMBOLS

from uberduck_ml_dev.losses import RADTTSLoss, AttentionBinarizationLoss

from uberduck_ml_dev.utils.utils import (
    intersperse,
    to_gpu,
    clip_grad_value_,
)


# config = DEFAULTS.values()
# config["with_gsts"] = False
config = {
    "train_config": {
        "output_directory": "/home/ray/default/lj_test_dontemptycache",
        "epochs": 10000,
        "optim_algo": "RAdam",
        "learning_rate": 1e-4,
        "weight_decay": 1e-6,
        "sigma": 1.0,
        "iters_per_checkpoint": 50000,
        "steps_per_sample": 4000,
        "batch_size": 16,
        "seed": None,
        "checkpoint_path": "",
        "ignore_layers": [],
        "ignore_layers_warmstart": [],
        "finetune_layers": [],
        "include_layers": [],
        "vocoder_config_path": "/usr/src/app/radtts/models/hifigan_22khz_config.json",
        "vocoder_checkpoint_path": "/usr/src/app/radtts/models/hifigan_libritts100360_generator0p5.pt",
        "log_attribute_samples": False,
        "log_decoder_samples": True,
        "warmstart_checkpoint_path": "",
        "use_amp": False,
        "grad_clip_val": 1.0,
        "loss_weights": {
            "blank_logprob": -1,
            "ctc_loss_weight": 0.1,
            "binarization_loss_weight": 1.0,
            "dur_loss_weight": 1.0,
            "f0_loss_weight": 1.0,
            "energy_loss_weight": 1.0,
            "vpred_loss_weight": 1.0
        },
        "binarization_start_iter": 6000,
        "kl_loss_start_iter": 18000,
        "unfreeze_modules": "all"
    },
    "data_config": {
    # NOTE (Sam): unused since we are getting data from s3 using the ray loader now
        "training_files": {
            "LJS": {
                "basedir": "/",
                "audiodir": "/usr/src/app/radtts/data/lj_data/LJSpeech-1.1/wavs",
                "filelist": "/usr/src/app/radtts/data/lj_data/LJSpeech-1.1/metadata_formatted_100.txt",
                "lmdbpath": ""
            }
        },
        "validation_files": {
            "LJS": {
                "basedir": "/",
                "audiodir": "/usr/src/app/radtts/data/lj_data/LJSpeech-1.1/wavs",
                "filelist": "/usr/src/app/radtts/data/lj_data/LJSpeech-1.1/metadata_formatted_100.txt",
                "lmdbpath": ""
            }
        },
    ###
        "dur_min": 0.1,
        "dur_max": 10.2,
        "sampling_rate": 22050,
        "filter_length": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mel_channels": 80,
        "mel_fmin": 0.0,
        "mel_fmax": 8000.0,
        "f0_min": 80.0,
        "f0_max": 640.0,
        "max_wav_value": 32768.0,
        "use_f0": True,
        "use_log_f0": 0,
        "use_energy_avg": True,
        "use_scaled_energy": True,
        "symbol_set": "radtts",
        "cleaner_names": ["radtts_cleaners"],
        "heteronyms_path": "/usr/src/app/uberduck_ml_dev/uberduck_ml_dev/text/heteronyms",
        "phoneme_dict_path": "/usr/src/app/uberduck_ml_dev/uberduck_ml_dev/text/cmudict-0.7b",
        # "heteronyms_path": "uberduck_ml_dev/text/heteronyms",
        # "phoneme_dict_path": "uberduck_ml_dev/text/cmudict-0.7b",
        "p_phoneme": 1.0,
        "handle_phoneme": "word",
        "handle_phoneme_ambiguous": "ignore",
        "include_speakers": None,
        "n_frames": -1,
        "betabinom_cache_path": "data_cache_2/",
        "lmdb_cache_path": "", 
        "use_attn_prior_masking": True,
        "prepend_space_to_text": True,
        "append_space_to_text": True,
        "add_bos_eos_to_text": False,
        "betabinom_scaling_factor": 1.0,
        "distance_tx_unvoiced": False,
        "mel_noise_scale": 0.0
    },
    "model_config": {
        "n_speakers": 1,
        "n_speaker_dim": 16,
        "n_text": 185,
        "n_text_dim": 512,
        "n_flows": 8,
        "n_conv_layers_per_step": 4,
        "n_mel_channels": 80,
        "n_hidden": 1024,
        "mel_encoder_n_hidden": 512,
        "dummy_speaker_embedding": False,
        "n_early_size": 2,
        "n_early_every": 2,
        "n_group_size": 2,
        "affine_model": "wavenet",
        "include_modules": "decatnvpred",
        "scaling_fn": "tanh",
        "matrix_decomposition": "LUS",
        "learn_alignments": True,
        "use_speaker_emb_for_alignment": False,
        "attn_straight_through_estimator": True,
        "use_context_lstm": True,
        "context_lstm_norm": "spectral",
        "context_lstm_w_f0_and_energy": True,
        "text_encoder_lstm_norm": "spectral",
        "n_f0_dims": 1,
        "n_energy_avg_dims": 1,
        "use_first_order_features": False,
        "unvoiced_bias_activation": "relu",
        "decoder_use_partial_padding": True,
        "decoder_use_unvoiced_bias": True,
        "ap_pred_log_f0": True,
        "ap_use_unvoiced_bias": True,
        "ap_use_voiced_embeddings": True,
        "dur_model_config": None,
        "f0_model_config": None,
        "energy_model_config": None,
        "v_model_config": {
            "name": "dap",
            "hparams": {
                "n_speaker_dim": 16,
                "take_log_of_input": False,
                "bottleneck_hparams": {
                    "in_dim": 512,
                    "reduction_factor": 16,
                    "norm": "weightnorm",
                    "non_linearity": "relu"
                },
                "arch_hparams": {
                    "out_dim": 1,
                    "n_layers": 2,
                    "n_channels": 256,
                    "kernel_size": 3,
                    "p_dropout": 0.5,
                    "lstm_type": "",
                    "use_linear": 1
                }
            }
        }
    }
}

train_config = config['train_config']
model_config = config['model_config']
data_config = config['data_config']



class DataCollate():
    """ Zero-pads model inputs and targets given number of steps """
    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate from normalized data """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x['text_encoded']) for x in batch]),
            dim=0, descending=True)

        max_input_len = input_lengths[0]
        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]]['text_encoded']
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mel_channels = batch[0]['mel'].size(0)
        max_target_len = max([x['mel'].size(1) for x in batch])

        # include mel padded, gate padded and speaker ids
        mel_padded = torch.FloatTensor(len(batch), num_mel_channels, max_target_len)
        mel_padded.zero_()
        f0_padded = None
        p_voiced_padded = None
        voiced_mask_padded = None
        energy_avg_padded = None
        if batch[0]['f0'] is not None:
            f0_padded = torch.FloatTensor(len(batch), max_target_len)
            f0_padded.zero_()

        if batch[0]['p_voiced'] is not None:
            p_voiced_padded = torch.FloatTensor(len(batch), max_target_len)
            p_voiced_padded.zero_()

        if batch[0]['voiced_mask'] is not None:
            voiced_mask_padded = torch.FloatTensor(len(batch), max_target_len)
            voiced_mask_padded.zero_()

        if batch[0]['energy_avg'] is not None:
            energy_avg_padded = torch.FloatTensor(len(batch), max_target_len)
            energy_avg_padded.zero_()

        attn_prior_padded = torch.FloatTensor(len(batch), max_target_len, max_input_len)
        attn_prior_padded.zero_()

        output_lengths = torch.LongTensor(len(batch))
        speaker_ids = torch.LongTensor(len(batch))
        audiopaths = []
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]]['mel']
            mel_padded[i, :, :mel.size(1)] = mel
            if batch[ids_sorted_decreasing[i]]['f0'] is not None:
                f0 = batch[ids_sorted_decreasing[i]]['f0']
                f0_padded[i, :len(f0)] = f0

            if batch[ids_sorted_decreasing[i]]['voiced_mask'] is not None:
                voiced_mask = batch[ids_sorted_decreasing[i]]['voiced_mask']
                voiced_mask_padded[i, :len(f0)] = voiced_mask

            if batch[ids_sorted_decreasing[i]]['p_voiced'] is not None:
                p_voiced = batch[ids_sorted_decreasing[i]]['p_voiced']
                p_voiced_padded[i, :len(f0)] = p_voiced

            if batch[ids_sorted_decreasing[i]]['energy_avg'] is not None:
                energy_avg = batch[ids_sorted_decreasing[i]]['energy_avg']
                energy_avg_padded[i, :len(energy_avg)] = energy_avg

            output_lengths[i] = mel.size(1)
            speaker_ids[i] = batch[ids_sorted_decreasing[i]]['speaker_id']
            audiopath = batch[ids_sorted_decreasing[i]]['audiopath']
            audiopaths.append(audiopath)
            cur_attn_prior = batch[ids_sorted_decreasing[i]]['attn_prior']
            if cur_attn_prior is None:
                attn_prior_padded = None
            else:
                attn_prior_padded[i, :cur_attn_prior.size(0), :cur_attn_prior.size(1)] = cur_attn_prior

        return {'mel': mel_padded,
                'speaker_ids': speaker_ids,
                'text': text_padded,
                'input_lengths': input_lengths,
                'output_lengths': output_lengths,
                'audiopaths': audiopaths,
                'attn_prior': attn_prior_padded,
                'f0': f0_padded,
                'p_voiced': p_voiced_padded,
                'voiced_mask': voiced_mask_padded,
                'energy_avg': energy_avg_padded
                }

max_wav_value = 32768
from librosa import pyin

def get_f0_pvoiced(audio, sampling_rate=22050, frame_length=1024,
                    hop_length=256, f0_min=100, f0_max=300):

    audio_norm = audio / max_wav_value
    f0, voiced_mask, p_voiced = pyin(
        y = audio_norm, fmin = f0_min, fmax = f0_max, sr = sampling_rate,
        frame_length=frame_length, win_length=frame_length // 2,
        hop_length=hop_length)
    f0[~voiced_mask] = 0.0
    f0 = torch.FloatTensor(f0)
    p_voiced = torch.FloatTensor(p_voiced)
    voiced_mask = torch.FloatTensor(voiced_mask)
    return f0, voiced_mask, p_voiced


    
use_scaled_energy = True
def energy_avg_normalize(x):
    if  use_scaled_energy == True:
        x = (x + 20.0) / 20.0
    return x
    
def get_energy_average(mel):
    energy_avg = mel.mean(0)
    energy_avg = energy_avg_normalize(energy_avg)
    return energy_avg

import os
from scipy.stats import betabinom

def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling_factor=0.05):
    P = phoneme_count
    M = mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M + 1):
        a, b = scaling_factor * i, scaling_factor * (M + 1 - i)
        rv = betabinom(P - 1, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return torch.tensor(np.array(mel_text_probs))


def get_attention_prior(n_tokens, n_frames):
    # cache the entire attn_prior by filename
    # if self.use_attn_prior_masking:
    filename = "{}_{}".format(n_tokens, n_frames)
    betabinom_cache_path = 'yourmom'
    if not os.path.exists(betabinom_cache_path):
        os.makedirs(betabinom_cache_path, exist_ok=False)
    prior_path = os.path.join(betabinom_cache_path, filename)
    prior_path += "_prior.pth"
    # if self.lmdb_cache_path != "":
    #     attn_prior = pkl.loads(
    #         self.cache_data_lmdb.get(prior_path.encode("ascii"))
    #     )
    if os.path.exists(prior_path):
        attn_prior = torch.load(prior_path)
    else:
        attn_prior = beta_binomial_prior_distribution(
            n_tokens, n_frames, scaling_factor = 1.0 # 0.05
        )
        torch.save(attn_prior, prior_path)
    # else:
        # attn_prior = torch.ones(n_frames, n_tokens)  # all ones baseline

    return attn_prior

def f0_normalize( x, f0_min):
    # if self.use_log_f0:
    # mask = x >= f0_min
    # x[mask] = torch.log(x[mask])
    # x[~mask] = 0.0

    return x
    
from uberduck_ml_dev.data.audio_processing import TacotronSTFT
stft = TacotronSTFT(
    filter_length=data_config['filter_length'],
    hop_length=data_config['hop_length'],
    win_length=data_config['win_length'],
    sampling_rate=22050,
    n_mel_channels=data_config['n_mel_channels'],
    mel_fmin=data_config['mel_fmin'],
    mel_fmax=data_config['mel_fmax'],
)

def get_speaker_id(speaker):

    return torch.LongTensor([speaker])

from uberduck_ml_dev.text.text_processing import TextProcessing

symbol_set = data_config['symbol_set']
cleaner_names = data_config['cleaner_names']
heteronyms_path = data_config['heteronyms_path']
phoneme_dict_path = data_config['phoneme_dict_path']
p_phoneme = data_config['p_phoneme']
handle_phoneme = data_config['handle_phoneme']
handle_phoneme_ambiguous = data_config['handle_phoneme_ambiguous']
prepend_space_to_text = data_config['prepend_space_to_text']
append_space_to_text = data_config['append_space_to_text']
add_bos_eos_to_text = data_config['add_bos_eos_to_text']


tp = TextProcessing(
    symbol_set,
    cleaner_names,
    heteronyms_path,
    phoneme_dict_path,
    p_phoneme=p_phoneme,
    handle_phoneme=handle_phoneme,
    handle_phoneme_ambiguous=handle_phoneme_ambiguous,
    prepend_space_to_text=prepend_space_to_text,
    append_space_to_text=append_space_to_text,
    add_bos_eos_to_text=add_bos_eos_to_text,
)
from uberduck_ml_dev.models.common import get_mel

def get_text(text):
    text = tp.encode_text(text)
    text = torch.LongTensor(text)
    return text
from datetime import datetime
def ray_df_to_batch_radtts(df):
    transcripts = df.transcript.tolist()
    audio_bytes_list = df.audio_bytes.tolist()
    speaker_ids = df.speaker_id.tolist()
    paths = df.path.tolist()
    f0_paths = df.f0_path.tolist()
    collate_fn = DataCollate()
    collate_input = []
    for transcript, audio_bytes, speaker_id, f0_path in zip(
        transcripts, audio_bytes_list, speaker_ids, f0_paths
    ):
        # print(datetime.now(), 'start')
        # Audio
        # print(datetime.now(), 'pre wav read and norm')
        bio = BytesIO(audio_bytes)
        sr, wav_data = wavfile.read(bio)
        audio = torch.FloatTensor(wav_data)
        audio_norm = audio / (np.abs(audio).max() * 2)
        # print(datetime.now(), 'pre text embed')
        text_sequence = get_text(transcript)
        # print(datetime.now(), 'pre mel compute')
        mel = get_mel(audio_norm, data_config['max_wav_value'], stft)
        mel = torch.squeeze(mel, 0)
        # print(datetime.now(), 'pre f0 load')
        dikt = torch.load(f0_path)
        f0 = dikt['f0']
        p_voiced = dikt['p_voiced']
        voiced_mask = dikt['voiced_mask']
        # f0, voiced_mask, p_voiced = get_f0_pvoiced(
        #     audio.cpu().numpy(), f0_min = data_config['f0_min'], f0_max=data_config["f0_max"], hop_length=data_config['hop_length'], frame_length=data_config['filter_length'], sampling_rate=22050)   
        f0 = f0_normalize(f0, f0_min = data_config['f0_min'])
        # print(datetime.now(), 'pre energy compute')
        energy_avg = get_energy_average(mel)
        # print(datetime.now(), 'pre prior load')
        prior_path = "{}_{}".format(text_sequence.shape[0], mel.shape[1])
        prior_path = os.path.join('/usr/src/app/radtts/data_cache', prior_path)
        prior_path += "_prior.pth"
        attn_prior = torch.load(prior_path)
        # attn_prior = get_attention_prior(text_sequence.shape[0], mel.shape[1])
        speaker_id =  get_speaker_id(speaker_id)
        collate_input.append({'text_encoded': text_sequence, 'mel':mel, 'speaker_id':speaker_id, 'f0': f0, 'p_voiced' : p_voiced, 'voiced_mask': voiced_mask, 'energy_avg': energy_avg, 'attn_prior' : attn_prior, 'audiopath': paths})
        # print(datetime.now(), 'end')
    return collate_fn(collate_input)



def get_ray_dataset():
    lj_df = pd.read_csv(
        # '/usr/src/app/radtts/data/lj_data/LJSpeech-1.1/metadata_formatted_full.txt',
        '/usr/src/app/radtts/data/lj_data/LJSpeech-1.1/metadata_formatted_full_pitch_100.txt',
        # "https://uberduck-datasets-dirty.s3.us-west-2.amazonaws.com/meta_full_s3.txt",
        # "https://uberduck-datasets-dirty.s3.us-west-2.amazonaws.com/lj_for_upload/metadata_formatted_100_edited.txt",
        sep="|",
        header=None,
        quoting=3,
        # names=["path", "transcript", "speaker_id"], # pitch path is implicit - this should be changed
        names = ['path', 'transcript', 'speaker_id', 'f0_path']
    )

    paths = lj_df.path.tolist()
    transcripts = lj_df.transcript.tolist()
    speaker_ids = lj_df.speaker_id.tolist()

    pitches = lj_df.f0_path.tolist()

    parallelism_length = 400
    audio_ds = ray.data.read_binary_files(
        paths,
        parallelism=parallelism_length,
        # meta_provider=FastFileMetadataProvider(),
        ray_remote_args={"num_cpus": 0.2},
    )
    audio_ds = audio_ds.map_batches(
        lambda x: x, batch_format="pyarrow", batch_size=None
    )

    paths = ray.data.from_items(paths, parallelism=parallelism_length)
    paths_ds = paths.map_batches(lambda x: x, batch_format="pyarrow", batch_size=None)

    transcripts = ray.data.from_items(transcripts, parallelism=parallelism_length)
    transcripts_ds = transcripts.map_batches(lambda x: x, batch_format="pyarrow", batch_size=None)

    speaker_ids_ds = ray.data.from_items(speaker_ids, parallelism=parallelism_length)
    speaker_ids_ds = speaker_ids_ds.map_batches(
        lambda x: x, batch_format="pyarrow", batch_size=None
    )
    pitches_ds = ray.data.from_items(pitches, parallelism=parallelism_length)
    pitches_ds = pitches_ds.map_batches(
        lambda x: x, batch_format="pyarrow", batch_size=None
    )
    output_dataset = (
        transcripts_ds.zip(audio_ds)
        .zip(paths_ds)
        .zip(speaker_ids_ds)
        .zip(pitches_ds)
    )
    output_dataset = output_dataset.map_batches(
        lambda table: table.rename(
            columns={
                "value": "transcript",
                "value_1": "audio_bytes",
                "value_2": "path",
                "value_3": "speaker_id",
                "value_4": "f0_path",
            }
        )
    )
    return output_dataset



# def get_ray_dataset_minimal():
#     lj_df = pd.read_csv(
#         '/usr/src/app/radtts/data/lj_data/LJSpeech-1.1/metadata_formatted_full.txt',
#         # "https://uberduck-datasets-dirty.s3.us-west-2.amazonaws.com/meta_full_s3.txt",
#         # "https://uberduck-datasets-dirty.s3.us-west-2.amazonaws.com/lj_for_upload/metadata_formatted_100_edited.txt",
#         sep="|",
#         header=None,
#         quoting=3,
#         names=["path", "transcript", "speaker_id"], # pitch path is implicit - this should be changed
#     )

#     parallelism_length = 400
#     speaker_ids = lj_df.speaker_id.tolist()
#     speaker_ids_ds = ray.data.from_items(speaker_ids, parallelism=parallelism_length)
#     speaker_ids_ds = speaker_ids_ds.map_batches(
#         lambda x: x, batch_format="pyarrow", batch_size=None
#     )
#     output_dataset = speaker_ids_ds
#     output_dataset = output_dataset.map_batches(
#         lambda table: table.rename(
#             columns={
#                 "value": "transcript",
#             }
#         )
#     )
#     return output_dataset



@torch.no_grad()
def log(metrics, audios = {}):
    # pass
    wandb_metrics = dict(metrics)
    
    for k,v in audios.items():
        wandb_metrics[k] = wandb.Audio(v, sample_rate=22050)
    # if gen_audio is not None:
    #     wandb_metrics.update({"gen/audio": wandb.Audio(gen_audio, sample_rate=22050)})
    # if gt_audio is not None:
    #     wandb_metrics.update({"gt/audio": wandb.Audio(gt_audio, sample_rate=22050)})
    # if sample_audio is not None:
    #     wandb_metrics.update(
    #         {"sample_inference": wandb.Audio(sample_audio, sample_rate=22050)}
    #     )
    # session.report(metrics)
    if session.get_world_rank() == 0:
        wandb.log(wandb_metrics)



from uberduck_ml_dev.utils.plot import plot_alignment_to_numpy

@torch.no_grad()
def get_log_audio(outputs, batch_dict, train_config, model, speaker_ids, text, f0, energy_avg, voiced_mask):

    mel = to_gpu(batch_dict['mel'])
    speaker_ids = to_gpu(batch_dict['speaker_ids'])
    attn_prior = to_gpu(batch_dict['attn_prior'])
    f0 = to_gpu(batch_dict['f0'])
    voiced_mask = to_gpu(batch_dict['voiced_mask'])
    p_voiced = to_gpu(batch_dict['p_voiced'])
    text = to_gpu(batch_dict['text'])
    in_lens = to_gpu(batch_dict['input_lengths'])
    out_lens = to_gpu(batch_dict['output_lengths'])
    energy_avg = to_gpu(batch_dict['energy_avg'])

    # NOTE (Sam): I don't think we can reuse the previous outputs since binarize_attention must be true
    outputs = model(
                mel, speaker_ids, text, in_lens, out_lens,
            binarize_attention=True, attn_prior=attn_prior, f0=f0,
            energy_avg=energy_avg, voiced_mask=voiced_mask,
            p_voiced=p_voiced)
    
    attn_used = outputs['attn']
    attn_soft = outputs['attn_soft']
    # audioname = os.path.basename(audiopaths[0])
    images = {}
    audios = {}
    if attn_used is not None:
        images['attention_weights'] = plot_alignment_to_numpy(
                attn_soft[0, 0].data.cpu().numpy().T, title="audioname")
        images['attention_weights_max'] = plot_alignment_to_numpy(
                attn_used[0, 0].data.cpu().numpy().T, title="audioname")
        attribute_sigmas = []
        """ NOTE: if training vanilla radtts (no attributes involved),
        use log_attribute_samples only, as there will be no ground truth
        features available. The infer function in this case will work with
        f0=None, energy_avg=None, and voiced_mask=None
        """
        if train_config['log_decoder_samples']: # decoder with gt features
            attribute_sigmas.append(-1)
        if train_config['log_attribute_samples']: # attribute prediction
            if model.is_attribute_unconditional():
                attribute_sigmas.extend([1.0])
            else:
                attribute_sigmas.extend([0.1, 0.5, 0.8, 1.0])
        if len(attribute_sigmas) > 0:
            durations = attn_used[0, 0].sum(0, keepdim=True)
            # NOTE (Sam): this is causing problems when durations are > x.5 and binarize_attention is false.
            #  In that case, durations + 0.5 . floor > durations
            # this causes issues to the length_regulator, which expects floor < durations
            durations = (durations + 0.5).floor().int()
            # load vocoder to CPU to avoid taking up valuable GPU vRAM
            # vocoder = get_vocoder()
            for attribute_sigma in attribute_sigmas:
                # try:
                if attribute_sigma > 0.0:
                    if hasattr(model, "infer"):
                        model_output = model.infer(
                            speaker_ids[0:1], text[0:1], 0.8,
                            dur=durations, f0=None, energy_avg=None,
                            voiced_mask=None, sigma_f0=attribute_sigma,
                            sigma_energy=attribute_sigma)
                    else:
                        model_output = model.module.infer(
                            speaker_ids[0:1], text[0:1], 0.8,
                            dur=durations, f0=None, energy_avg=None,
                            voiced_mask=None, sigma_f0=attribute_sigma,
                            sigma_energy=attribute_sigma)         
                else:
                    if hasattr(model, "infer"):
                        model_output = model.infer(
                            speaker_ids[0:1], text[0:1], 0.8,
                            dur=durations, f0=f0[0:1, :durations.sum()],
                            energy_avg=energy_avg[0:1, :durations.sum()],
                            voiced_mask=voiced_mask[0:1, :durations.sum()])
                    else:
                        model_output = model.module.infer(
                            speaker_ids[0:1], text[0:1], 0.8,
                            dur=durations, f0=f0[0:1, :durations.sum()],
                            energy_avg=energy_avg[0:1, :durations.sum()],
                            voiced_mask=voiced_mask[0:1, :durations.sum()])                      
                # except:
                #     print("Instability or issue occured during inference, skipping sample generation for TB logger")
                #     continue
                mels = model_output['mel']
                if hasattr(vocoder, 'forward'):
                    audio = vocoder(mels.cpu()).float()[0]
                audio = audio[0].detach().cpu().numpy()
                audio = audio / np.abs(audio).max()
                if attribute_sigma < 0:
                    sample_tag = "decoder_sample_gt_attributes"
                else:
                    sample_tag = f"sample_attribute_sigma_{attribute_sigma}"
                audios[sample_tag] = audio

    return images, audios


def _train_step(
    batch,
    model,
    optim,
    iteration,
    steps_per_sample,
    scaler,
    scheduler,
    criterion,
    attention_kl_loss,
    kl_loss_start_iter,
    binarization_start_iter
):
    print(datetime.now(), 'entering train step')
    if iteration >= binarization_start_iter:
        binarize = True
    else:
        binarize = False

    optim.zero_grad()

    with autocast(enabled= False):
        # batch_dict = ray_df_to_batch_radtts(batch)
        batch_dict = batch
        mel = to_gpu(batch_dict['mel'])
        speaker_ids = to_gpu(batch_dict['speaker_ids'])
        attn_prior = to_gpu(batch_dict['attn_prior'])
        f0 = to_gpu(batch_dict['f0'])
        voiced_mask = to_gpu(batch_dict['voiced_mask'])
        p_voiced = to_gpu(batch_dict['p_voiced'])
        text = to_gpu(batch_dict['text'])
        in_lens = to_gpu(batch_dict['input_lengths'])
        out_lens = to_gpu(batch_dict['output_lengths'])
        energy_avg = to_gpu(batch_dict['energy_avg'])

        outputs = model(
                    mel, speaker_ids, text, in_lens, out_lens,
                    binarize_attention=binarize, attn_prior=attn_prior,
                    f0=f0, energy_avg=energy_avg,
                    voiced_mask=voiced_mask, p_voiced=p_voiced)

        loss_outputs = criterion(outputs, in_lens, out_lens)

        print_list = []
        loss = None
        for k, (v, w) in loss_outputs.items():
            if w > 0:
                loss = v * w if loss is None else loss + v * w
            print_list.append('  |  {}: {:.3f}'.format(k, v))

        
        w_bin = criterion.loss_weights.get('binarization_loss_weight', 1.0)
        if binarize and iteration >= kl_loss_start_iter:
            binarization_loss = attention_kl_loss(
                outputs['attn'], outputs['attn_soft'])
            loss += binarization_loss * w_bin
        else:
            binarization_loss = torch.zeros_like(loss)
        loss_outputs['binarization_loss'] = (binarization_loss, w_bin)

    grad_clip_val = 1. # it is what is is ;)
    print(print_list)
    scaler.scale(loss).backward()
    if grad_clip_val > 0:
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip_val)
    # scheduler.step()
    # scaler.scale(loss).backward()
    # scaler.unscale_(optim)
    # clip_grad_value_(model.parameters(), 100)
    scaler.step(optim)
    scaler.update()
    # scaler.update()
    # scheduler.step()

    metrics = {
        "loss": loss.item()
    }
    for k, (v, w) in loss_outputs.items():
        metrics[k] = v.item()

    # if 2 == 3:
    # if global_step % steps_per_sample == 0 and session.get_world_rank() == 0:
    if iteration % steps_per_sample == 0 and session.get_world_rank() == 0:

        model.eval()
        # TODO (Sam): adding tf output logging and out of distribution inference
        images, audios = get_log_audio(outputs, batch_dict, train_config, model, speaker_ids, text, f0, energy_avg, voiced_mask)
        log(metrics, audios)
        model.train()
    else:
        log(metrics)

    print(f"Loss: {loss.item()}")


  
def train_epoch(train_dataloader, dataset_shard, batch_size, model, optim, steps_per_sample, scaler, scheduler, criterion, attention_kl_loss, kl_loss_start_iter, binarization_start_iter, epoch, iteration):
# def train_epoch(dataset_shard, batch_size, model, optim, steps_per_sample, scaler, scheduler, criterion, attention_kl_loss, kl_loss_start_iter, binarization_start_iter, epoch, iteration):
    # for batch_idx, ray_batch_df in enumerate(
    #     dataset_shard.iter_batches(batch_size=batch_size, prefetch_blocks=4)
    # ):
    for batch in train_dataloader:
        # torch.cuda.empty_cache()
        _train_step(
            # ray_batch_df,
            batch,
            model,
            optim,
            iteration,
            steps_per_sample,
            scaler,
            scheduler,
            criterion,
            attention_kl_loss,
            kl_loss_start_iter,
            binarization_start_iter,
        )
        iteration += 1
        
    checkpoint = Checkpoint.from_dict(
        dict(
            epoch=epoch,
            global_step=iteration,
            model=model.state_dict(),
        )
    )
    session.report({}, checkpoint=checkpoint)
    if session.get_world_rank() == 0:
        artifact = wandb.Artifact(
            f"artifact_epoch{epoch}_step{iteration}", "model"
        )
        with tempfile.TemporaryDirectory() as tempdirname:
            checkpoint.to_directory(tempdirname)
            artifact.add_dir(tempdirname)
            wandb.log_artifact(artifact)
        

def train_func(config: dict):
    setup_wandb(config, project="radtts-ray", entity = 'uberduck-ai', rank_zero_only=False)
    print("CUDA AVAILABLE: ", torch.cuda.is_available())
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    steps_per_sample = config["steps_per_sample"]
    sigma = config['sigma']
    kl_loss_start_iter = config['kl_loss_start_iter']
    binarization_start_iter = config['binarization_start_iter']

    model = RADTTS(
        **model_config,
    )
    # model = train.torch.prepare_model(model, parallel_strategy_kwargs = dict(find_unused_parameters=True))
    model = train.torch.prepare_model(model)
    # model = train.torch.prepare_model(model, parallel_strategy_kwargs = dict())

    start_epoch = 0
    train_loader, valset, collate_fn = prepare_dataloaders(data_config, 2, 6)
    train_dataloader = train.torch.prepare_data_loader(train_loader)

    # NOTE (Sam): replace with RAdam
    optim = torch.optim.Adam(
        model.parameters(),
        lr = config["learning_rate"],
        weight_decay = config["weight_decay"]
    )
    scheduler = ExponentialLR(
        optim,
        config["weight_decay"],
        last_epoch=-1,
    )
    dataset_shard = session.get_dataset_shard("train")
    scaler = GradScaler()

    criterion = RADTTSLoss(
        sigma,
        config['n_group_size'],
        config['dur_model_config'],
        config['f0_model_config'],
        config['energy_model_config'],
        vpred_model_config=config['v_model_config'],
        loss_weights=config['loss_weights']
    )
    attention_kl_loss = AttentionBinarizationLoss()
    iteration = 0
    for epoch in range(start_epoch, start_epoch + epochs):
        train_epoch(train_dataloader, dataset_shard, batch_size, model, optim, steps_per_sample, scaler, scheduler, criterion, attention_kl_loss, kl_loss_start_iter, binarization_start_iter, epoch, iteration)
        # train_epoch(dataset_shard, batch_size, model, optim, steps_per_sample, scaler, scheduler, criterion, attention_kl_loss, kl_loss_start_iter, binarization_start_iter, epoch, iteration)
        iteration += 1

from ray.train.torch import TorchTrainer, TorchCheckpoint, TorchTrainer
from ray.air.config import ScalingConfig, RunConfig
from ray.tune import SyncConfig


# For sample inference
import json
from uberduck_ml_dev.vocoders.hifigan import AttrDict, Generator
# , Denoiser

# def load_vocoder(vocoder_path, config_path, to_cuda=True):
def load_vocoder(vocoder_state_dict, vocoder_config, to_cuda = True):
    # with open(config_path) as f:
    #     data_vocoder = f.read()
    # config_vocoder = json.loads(data_vocoder)
    h = AttrDict(vocoder_config)
    if 'gaussian_blur' in vocoder_config:
        vocoder_config['gaussian_blur']['p_blurring'] = 0.0
    else:
        vocoder_config['gaussian_blur'] = {'p_blurring': 0.0}
        h['gaussian_blur'] = {'p_blurring': 0.0}

    # state_dict_g = torch.load(vocoder_path, map_location='cpu')['generator']

    # load hifigan
    vocoder = Generator(h)
    vocoder.load_state_dict(vocoder_state_dict)
    # denoiser = Denoiser(vocoder)
    if to_cuda:
        vocoder.cuda()
        # denoiser.cuda()
    vocoder.eval()
    # denoiser.eval()

    return vocoder #, denoiser

import requests
HIFI_GAN_CONFIG_URL = "https://uberduck-models-us-west-2.s3.us-west-2.amazonaws.com/hifigan_22khz_config.json"
HIFI_GAN_GENERATOR_URL = "https://uberduck-models-us-west-2.s3.us-west-2.amazonaws.com/hifigan_libritts100360_generator0p5.pt"



hifi_gan_config_path = '/usr/src/app/radtts/models/hifigan_22khz_config.json'
hifi_gan_generator_path = '/usr/src/app/radtts/models/hifigan_libritts100360_generator0p5.pt'

def load_pretrained(model):
    # response = requests.get(HIFI_GAN_GENERATOR_URL, stream=True)
    # bio = BytesIO(response.content)
    loaded = torch.load(hifi_gan_generator_path)
    model.load_state_dict(loaded['generator'])


def get_vocoder():
    print("Getting model config...")
    # response = requests.get(HIFI_GAN_CONFIG_URL)

    with open(hifi_gan_config_path) as f:
        hifigan_config = json.load(f)

    h = AttrDict(hifigan_config)
    if 'gaussian_blur' in hifigan_config:
        hifigan_config['gaussian_blur']['p_blurring'] = 0.0
    else:
        hifigan_config['gaussian_blur'] = {'p_blurring': 0.0}
        h['gaussian_blur'] = {'p_blurring': 0.0}
    # model_params = hifigan_config["model_params"]
    model = Generator(h)
    print("Loading pretrained model...")
    load_pretrained(model)
    print("Got pretrained model...")
    model.eval()
    return model

### pytorch dataloader for debug
from scipy.io.wavfile import read
import lmdb
import pickle as pkl
from scipy.ndimage import distance_transform_edt as distance_transform
def load_wav_to_torch(full_path):
    """ Loads wavdata into torch array """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(np.array(data)).float(), sampling_rate

class Data(torch.utils.data.Dataset):
    def __init__(self, datasets, filter_length, hop_length, win_length,
                 sampling_rate, n_mel_channels, mel_fmin, mel_fmax, f0_min,
                 f0_max, max_wav_value, use_f0, use_energy_avg, use_log_f0,
                 use_scaled_energy, symbol_set, cleaner_names, heteronyms_path,
                 phoneme_dict_path, p_phoneme, handle_phoneme='word',
                 handle_phoneme_ambiguous='ignore', speaker_ids=None,
                 include_speakers=None, n_frames=-1,
                 use_attn_prior_masking=True, prepend_space_to_text=True,
                 append_space_to_text=True, add_bos_eos_to_text=False,
                 betabinom_cache_path="", betabinom_scaling_factor=0.05,
                 lmdb_cache_path="", dur_min=None, dur_max=None,
                 combine_speaker_and_emotion=False, **kwargs):

        self.combine_speaker_and_emotion = combine_speaker_and_emotion
        self.max_wav_value = max_wav_value
        self.audio_lmdb_dict = {}  # dictionary of lmdbs for audio data
        self.data = self.load_data(datasets)
        self.distance_tx_unvoiced = False
        if 'distance_tx_unvoiced' in kwargs.keys():
            self.distance_tx_unvoiced = kwargs['distance_tx_unvoiced']
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 n_mel_channels=n_mel_channels,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)

        self.do_mel_scaling = kwargs.get('do_mel_scaling', True)
        self.mel_noise_scale = kwargs.get('mel_noise_scale', 0.0)
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.use_f0 = use_f0
        self.use_log_f0 = use_log_f0
        self.use_energy_avg = use_energy_avg
        self.use_scaled_energy = use_scaled_energy
        self.sampling_rate = sampling_rate
        self.tp = TextProcessing(
            symbol_set, cleaner_names, heteronyms_path, phoneme_dict_path,
            p_phoneme=p_phoneme, handle_phoneme=handle_phoneme,
            handle_phoneme_ambiguous=handle_phoneme_ambiguous,
            prepend_space_to_text=prepend_space_to_text,
            append_space_to_text=append_space_to_text,
            add_bos_eos_to_text=add_bos_eos_to_text)

        self.dur_min = dur_min
        self.dur_max = dur_max
        if speaker_ids is None or speaker_ids == '':
            self.speaker_ids = self.create_speaker_lookup_table(self.data)
        else:
            self.speaker_ids = speaker_ids

        print("Number of files", len(self.data))
        if include_speakers is not None:
            for (speaker_set, include) in include_speakers:
                self.filter_by_speakers_(speaker_set, include)
            print("Number of files after speaker filtering", len(self.data))

        if dur_min is not None and dur_max is not None:
            self.filter_by_duration_(dur_min, dur_max)
            print("Number of files after duration filtering", len(self.data))

        self.use_attn_prior_masking = bool(use_attn_prior_masking)
        self.prepend_space_to_text = bool(prepend_space_to_text)
        self.append_space_to_text = bool(append_space_to_text)
        self.betabinom_cache_path = betabinom_cache_path
        self.betabinom_scaling_factor = betabinom_scaling_factor
        self.lmdb_cache_path = lmdb_cache_path
        if self.lmdb_cache_path != "":
            self.cache_data_lmdb = lmdb.open(
                self.lmdb_cache_path, readonly=True, max_readers=1024,
                lock=False).begin()

        # make sure caching path exists
        if not os.path.exists(self.betabinom_cache_path):
            os.makedirs(self.betabinom_cache_path)

        print("Dataloader initialized with no augmentations")
        self.speaker_map = None
        if 'speaker_map' in kwargs:
            self.speaker_map = kwargs['speaker_map']

    def load_data(self, datasets, split='|'):
        dataset = []
        for dset_name, dset_dict in datasets.items():
            folder_path = dset_dict['basedir']
            audiodir = dset_dict['audiodir']
            filename = dset_dict['filelist']
            audio_lmdb_key = None
            if 'lmdbpath' in dset_dict.keys() and len(dset_dict['lmdbpath']) > 0:
                self.audio_lmdb_dict[dset_name] = lmdb.open(
                    dset_dict['lmdbpath'], readonly=True, max_readers=256,
                    lock=False).begin()
                audio_lmdb_key = dset_name

            wav_folder_prefix = os.path.join(folder_path, audiodir)
            filelist_path = os.path.join(folder_path, filename)
            with open(filelist_path, encoding='utf-8') as f:
                data = [line.strip().split(split) for line in f]

            for d in data:
                emotion = 'other' if len(d) == 3 else d[3]
                duration = -1 if len(d) == 3 else d[4]
                dataset.append(
                    {'audiopath': os.path.join(wav_folder_prefix, d[0]),
                     'text': d[1],
                     'speaker': d[2] + '-' + emotion if self.combine_speaker_and_emotion else d[2],
                     'emotion': emotion,
                     'duration': float(duration),
                     'lmdb_key': audio_lmdb_key
                     })
        return dataset

    def filter_by_speakers_(self, speakers, include=True):
        print("Include spaker {}: {}".format(speakers, include))
        if include:
            self.data = [x for x in self.data if x['speaker'] in speakers]
        else:
            self.data = [x for x in self.data if x['speaker'] not in speakers]

    def filter_by_duration_(self, dur_min, dur_max):
        self.data = [
            x for x in self.data
            if x['duration'] == -1 or (
                x['duration'] >= dur_min and x['duration'] <= dur_max)]

    def create_speaker_lookup_table(self, data):
        speaker_ids = np.sort(np.unique([x['speaker'] for x in data]))
        d = {speaker_ids[i]: i for i in range(len(speaker_ids))}
        print("Number of speakers:", len(d))
        print("Speaker IDS", d)
        return d

    def f0_normalize(self, x):
        if self.use_log_f0:
            mask = x >= self.f0_min
            x[mask] = torch.log(x[mask])
            x[~mask] = 0.0

        return x

    def f0_denormalize(self, x):
        if self.use_log_f0:
            log_f0_min = np.log(self.f0_min)
            mask = x >= log_f0_min
            x[mask] = torch.exp(x[mask])
            x[~mask] = 0.0
        x[x <= 0.0] = 0.0

        return x

    def energy_avg_normalize(self, x):
        if self.use_scaled_energy:
            x = (x + 20.0) / 20.0
        return x

    def energy_avg_denormalize(self, x):
        if self.use_scaled_energy:
            x = x * 20.0 - 20.0
        return x

    def get_f0_pvoiced(self, audio, sampling_rate=22050, frame_length=1024,
                       hop_length=256, f0_min=100, f0_max=300):

        audio_norm = audio / self.max_wav_value
        f0, voiced_mask, p_voiced = pyin(
            audio_norm, f0_min, f0_max, sampling_rate,
            frame_length=frame_length, win_length=frame_length // 2,
            hop_length=hop_length)
        f0[~voiced_mask] = 0.0
        f0 = torch.FloatTensor(f0)
        p_voiced = torch.FloatTensor(p_voiced)
        voiced_mask = torch.FloatTensor(voiced_mask)
        return f0, voiced_mask, p_voiced

    def get_energy_average(self, mel):
        energy_avg = mel.mean(0)
        energy_avg = self.energy_avg_normalize(energy_avg)
        return energy_avg

    def get_mel(self, audio):
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        if self.do_mel_scaling:
            melspec = (melspec + 5.5) / 2
        if self.mel_noise_scale > 0:
            melspec += torch.randn_like(melspec) * self.mel_noise_scale
        return melspec

    def get_speaker_id(self, speaker):
        if self.speaker_map is not None and speaker in self.speaker_map:
            speaker = self.speaker_map[speaker]

        return torch.LongTensor([self.speaker_ids[speaker]])

    def get_text(self, text):
        text = self.tp.encode_text(text)
        text = torch.LongTensor(text)
        return text

    def get_attention_prior(self, n_tokens, n_frames):
        # cache the entire attn_prior by filename
        if self.use_attn_prior_masking:
            filename = "{}_{}".format(n_tokens, n_frames)
            prior_path = os.path.join(self.betabinom_cache_path, filename)
            prior_path += "_prior.pth"
            if self.lmdb_cache_path != "":
                attn_prior = pkl.loads(
                    self.cache_data_lmdb.get(prior_path.encode('ascii')))
            elif os.path.exists(prior_path):
                attn_prior = torch.load(prior_path)
            else:
                attn_prior = beta_binomial_prior_distribution(
                    n_tokens, n_frames, self.betabinom_scaling_factor)
                torch.save(attn_prior, prior_path)
        else:
            attn_prior = torch.ones(n_frames, n_tokens)  # all ones baseline

        return attn_prior

    def __getitem__(self, index):
        data = self.data[index]
        audiopath, text = data['audiopath'], data['text']
        speaker_id = data['speaker']

        if data['lmdb_key'] is not None:
            data_dict = pkl.loads(
                self.audio_lmdb_dict[data['lmdb_key']].get(
                    audiopath.encode('ascii')))
            audio = data_dict['audio']
            sampling_rate = data_dict['sampling_rate']
        else:
            audio, sampling_rate = load_wav_to_torch(audiopath)

        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        mel = self.get_mel(audio)
        f0 = None
        p_voiced = None
        voiced_mask = None
        if self.use_f0:
            filename = '_'.join(audiopath.split('/')[-3:])
            f0_path = os.path.join(self.betabinom_cache_path, filename)
            f0_path += "_f0_sr{}_fl{}_hl{}_f0min{}_f0max{}_log{}.pt".format(
                self.sampling_rate, self.filter_length, self.hop_length,
                self.f0_min, self.f0_max, self.use_log_f0)

            dikt = None
            if len(self.lmdb_cache_path) > 0:
                dikt = pkl.loads(
                    self.cache_data_lmdb.get(f0_path.encode('ascii')))
                f0 = dikt['f0']
                p_voiced = dikt['p_voiced']
                voiced_mask = dikt['voiced_mask']
            elif os.path.exists(f0_path):
                try:
                    dikt = torch.load(f0_path)
                except:
                    print(f"f0 loading from {f0_path} is broken, recomputing.")

            if dikt is not None:
                f0 = dikt['f0']
                p_voiced = dikt['p_voiced']
                voiced_mask = dikt['voiced_mask']
            else:
                f0, voiced_mask, p_voiced = self.get_f0_pvoiced(
                    audio.cpu().numpy(), self.sampling_rate,
                    self.filter_length, self.hop_length, self.f0_min,
                    self.f0_max)
                print("saving f0 to {}".format(f0_path))
                torch.save({'f0': f0,
                            'voiced_mask': voiced_mask,
                            'p_voiced': p_voiced}, f0_path)
            if f0 is None:
                raise Exception("STOP, BROKEN F0 {}".format(audiopath))

            f0 = self.f0_normalize(f0)
            if self.distance_tx_unvoiced:
                mask = f0 <= 0.0
                distance_map = np.log(distance_transform(mask))
                distance_map[distance_map <= 0] = 0.0
                f0 = f0 - distance_map

        energy_avg = None
        if self.use_energy_avg:
            energy_avg = self.get_energy_average(mel)
            if self.use_scaled_energy and energy_avg.min() < 0.0:
                print(audiopath, "has scaled energy avg smaller than 0")

        speaker_id = self.get_speaker_id(speaker_id)
        text_encoded = self.get_text(text)

        attn_prior = self.get_attention_prior(
                text_encoded.shape[0], mel.shape[1])

        if not self.use_attn_prior_masking:
            attn_prior = None

        return {'mel': mel,
                'speaker_id': speaker_id,
                'text_encoded': text_encoded,
                'audiopath': audiopath,
                'attn_prior': attn_prior,
                'f0': f0,
                'p_voiced': p_voiced,
                'voiced_mask': voiced_mask,
                'energy_avg': energy_avg,
                }

    def __len__(self):
        return len(self.data)

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

def prepare_dataloaders(data_config, n_gpus, batch_size):
    # Get data, data loaders and collate function ready
    ignore_keys = ['training_files', 'validation_files']
    print("initializing training dataloader")
    trainset = Data(data_config['training_files'],
                    **dict((k, v) for k, v in data_config.items()
                    if k not in ignore_keys))

    print("initializing validation dataloader")
    data_config_val = data_config.copy()
    data_config_val['aug_probabilities'] = None  # no aug in val set
    valset = Data(data_config['validation_files'],
                  **dict((k, v) for k, v in data_config_val.items()
                  if k not in ignore_keys), speaker_ids=trainset.speaker_ids)

    collate_fn = DataCollate()

    train_sampler, shuffle = None, True
    if n_gpus > 1:
        train_sampler, shuffle = DistributedSampler(trainset), False

    train_loader = DataLoader(trainset, num_workers=8, shuffle=shuffle,
                              sampler=train_sampler, batch_size=batch_size,
                              pin_memory=False, drop_last=True,
                              collate_fn=collate_fn)

    return train_loader, valset, collate_fn


if __name__ == "__main__":

 
    # print('asdfghk \n\n\n\n\n')
    ray_dataset = get_ray_dataset()
    # ray_dataset_minimal = get_ray_dataset()
    train_config['n_group_size'] = model_config['n_group_size']
    train_config['dur_model_config'] = model_config['dur_model_config']
    train_config['f0_model_config'] = model_config['f0_model_config']
    train_config['energy_model_config'] = model_config['energy_model_config']
    train_config['v_model_config']=model_config['v_model_config']
    vocoder = get_vocoder()
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_config,
        scaling_config=ScalingConfig(
            num_workers=1, use_gpu=True, resources_per_worker=dict(CPU=8, GPU=1)
        ),
        run_config=RunConfig(
            # sync_config=SyncConfig(upload_dir="s3://uberduck-anyscale-data/checkpoints")
            sync_config=SyncConfig()
        ),
        datasets={"train": ray_dataset},
    )
    # trainer = DummyTrainer(
    #     scaling_config=ScalingConfig(
    #         num_workers=1, use_gpu=True, resources_per_worker=dict(CPU=8, GPU=1)
    #     ),
    #     run_config=RunConfig(
    #         sync_config=SyncConfig()
    #     ),
    #     datasets={"train": ray_dataset_minimal},
    # )
    result = trainer.fit()
