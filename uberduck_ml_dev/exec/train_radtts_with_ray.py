
import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
import wandb
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import sys
from collections import OrderedDict
import sys
import lmdb
import pickle as pkl
import json
from datetime import datetime
import os


from scipy.io.wavfile import read

import ray
from ray.air import session
from ray.air.config import ScalingConfig, RunConfig
from ray.air.integrations.wandb import setup_wandb
import ray.data
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.tune import SyncConfig
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from ray.train.torch import TorchTrainer, TorchTrainer
from ray.air.config import ScalingConfig, RunConfig

from ..models.radtts import RADTTS
from ..models.components.encoders.resnet_speaker_encoder import ResNetSpeakerEncoderCallable
from ..losses import RADTTSLoss, AttentionBinarizationLoss
from ..optimizers.radam import RAdam
from ..utils.utils import (
    to_gpu,
)
from ..vocoders.hifigan import AttrDict, Generator
from ..models.common import TacotronSTFT
from ..text.text_processing import TextProcessing
from ..utils.plot import plot_alignment_to_numpy

HIFI_GAN_CONFIG_URL = "https://uberduck-models-us-west-2.s3.us-west-2.amazonaws.com/hifigan_22khz_config.json"
HIFI_GAN_GENERATOR_URL = "https://uberduck-models-us-west-2.s3.us-west-2.amazonaws.com/hifigan_libritts100360_generator0p5.pt"
HIFI_GAN_CONFIG_PATH = "/usr/src/app/radtts/models/hifigan_22khz_config.json"
HIFI_GAN_GENERATOR_PATH = (
    "/usr/src/app/radtts/models/hifigan_libritts100360_generator0p5.pt"
)

from ..data.collate import DataCollateRADTTS as DataCollate
from ..data.data import DataRADTTS as Data
from ..data.utils import beta_binomial_prior_distribution

def get_attention_prior(n_tokens, n_frames):
    # cache the entire attn_prior by filename
    # if self.use_attn_prior_masking:
    filename = "{}_{}".format(n_tokens, n_frames)
    betabinom_cache_path = "yourmom"
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
            n_tokens, n_frames, scaling_factor=1.0  # 0.05
        )
        torch.save(attn_prior, prior_path)
    # else:
    # attn_prior = torch.ones(n_frames, n_tokens)  # all ones baseline

    return attn_prior






@torch.no_grad()
def log(metrics, audios={}):
    # pass
    wandb_metrics = dict(metrics)

    for k, v in audios.items():
        wandb_metrics[k] = wandb.Audio(v, sample_rate=22050)

    # session.report(metrics)
    if session.get_world_rank() == 0:
        wandb.log(wandb_metrics)


# want to test out of sample but can only do proper inference with zero shot dap so lets just look at zero shot decoder samples
# in particular, the out of sample here will probably be worse than inference since the f0 and energy models are not being used.
@torch.no_grad()
def get_log_audio(
    outputs,
    batch_dict,
    train_config,
    model,
    speaker_ids,
    text,
    f0,
    energy_avg,
    voiced_mask,
    audio_embedding_oos=None,
    oos_name=None,
):

    # print( audio_embedding_oos, oos_name, bool(audio_embedding_oos is None), bool(oos_name is None))
    assert bool(audio_embedding_oos is None) == bool(
        oos_name is None
    ), "must provide both or neither of audio_embedding_oos and oos_name"

    mel = to_gpu(batch_dict["mel"])
    speaker_ids = to_gpu(batch_dict["speaker_ids"])
    attn_prior = to_gpu(batch_dict["attn_prior"])
    f0 = to_gpu(batch_dict["f0"])
    voiced_mask = to_gpu(batch_dict["voiced_mask"])
    p_voiced = to_gpu(batch_dict["p_voiced"])
    text = to_gpu(batch_dict["text"])
    in_lens = to_gpu(batch_dict["input_lengths"])
    out_lens = to_gpu(batch_dict["output_lengths"])
    energy_avg = to_gpu(batch_dict["energy_avg"])
    audio_embedding = to_gpu(batch_dict["audio_embedding"])

    # NOTE (Sam): I don't think we can reuse the previous outputs since binarize_attention must be true
    outputs = model(
        mel,
        speaker_ids,
        text,
        in_lens,
        out_lens,
        binarize_attention=True,
        attn_prior=attn_prior,
        f0=f0,
        energy_avg=energy_avg,
        voiced_mask=voiced_mask,
        p_voiced=p_voiced,
        audio_embedding=audio_embedding,
    )

    attn_used = outputs["attn"]
    attn_soft = outputs["attn_soft"]

    images = {}
    audios = {}
    if attn_used is not None:
        images["attention_weights"] = plot_alignment_to_numpy(
            attn_soft[0, 0].data.cpu().numpy().T, title="audioname"
        )
        images["attention_weights_max"] = plot_alignment_to_numpy(
            attn_used[0, 0].data.cpu().numpy().T, title="audioname"
        )
        attribute_sigmas = []
        """ NOTE: if training vanilla radtts (no attributes involved),
        use log_attribute_samples only, as there will be no ground truth
        features available. The infer function in this case will work with
        f0=None, energy_avg=None, and voiced_mask=None
        """
        if train_config["log_decoder_samples"]:  # decoder with gt features
            attribute_sigmas.append(-1)
        if train_config["log_attribute_samples"]:  # attribute prediction
            if hasattr(model, "is_attribute_unconditional"):
                if model.is_attribute_unconditional():
                    attribute_sigmas.extend([1.0])
                else:
                    attribute_sigmas.extend([0.1, 0.5, 0.8, 1.0])
            else:
                if model.module.is_attribute_unconditional():
                    attribute_sigmas.extend([1.0])
                else:
                    attribute_sigmas.extend([0.1, 0.5, 0.8, 1.0])
        if len(attribute_sigmas) > 0:
            if audio_embedding_oos is not None:
                audio_embedding = audio_embedding_oos.unsqueeze(0)
            durations = attn_used[0, 0].sum(0, keepdim=True)
            # NOTE (Sam): this is causing problems when durations are > x.5 and binarize_attention is false.
            #  In that case, durations + 0.5 . floor > durations
            # this causes issues to the length_regulator, which expects floor < durations.
            # Just keep binarize_attention = True in inference and don't think about it that hard.
            durations = (durations + 0.5).floor().int()
            # NOTE (Sam): should we load vocoder to CPU to avoid taking up valuable GPU vRAM?
            for attribute_sigma in attribute_sigmas:
                # try:
                if attribute_sigma > 0.0:
                    if hasattr(model, "infer"):
                        model_output = model.infer(
                            speaker_ids[0:1],
                            text[0:1],
                            0.8,
                            dur=durations,
                            f0=None,
                            energy_avg=None,
                            voiced_mask=None,
                            sigma_f0=attribute_sigma,
                            sigma_energy=attribute_sigma,
                            audio_embedding=audio_embedding[0:1],
                        )
                    else:
                        model_output = model.module.infer(
                            speaker_ids[0:1],
                            text[0:1],
                            0.8,
                            dur=durations,
                            f0=None,
                            energy_avg=None,
                            voiced_mask=None,
                            sigma_f0=attribute_sigma,
                            sigma_energy=attribute_sigma,
                            audio_embedding=audio_embedding[0:1],
                        )
                else:
                    if hasattr(model, "infer"):
                        model_output = model.infer(
                            speaker_ids[0:1],
                            text[0:1],
                            0.8,
                            dur=durations,
                            f0=f0[0:1, : durations.sum()],
                            energy_avg=energy_avg[0:1, : durations.sum()],
                            voiced_mask=voiced_mask[0:1, : durations.sum()],
                            audio_embedding=audio_embedding[0:1],
                        )
                    else:
                        model_output = model.module.infer(
                            speaker_ids[0:1],
                            text[0:1],
                            0.8,
                            dur=durations,
                            f0=f0[0:1, : durations.sum()],
                            energy_avg=energy_avg[0:1, : durations.sum()],
                            voiced_mask=voiced_mask[0:1, : durations.sum()],
                            audio_embedding=audio_embedding[0:1],
                        )
                # except:
                #     print("Instability or issue occured during inference, skipping sample generation for TB logger")
                #     continue
                mels = model_output["mel"]
                if hasattr(vocoder, "forward"):
                    audio = vocoder(mels.cpu()).float()[0]
                audio = audio[0].detach().cpu().numpy()
                audio = audio / np.abs(audio).max()
                if attribute_sigma < 0:
                    sample_tag = "decoder_sample_gt_attributes"
                else:
                    sample_tag = f"sample_attribute_sigma_{attribute_sigma}"
                if oos_name is not None:
                    sample_tag = f"{sample_tag}_oos_{oos_name}"
                audios[sample_tag] = audio

    return images, audios


def save_checkpoint(model, optimizer, iteration, filepath):
    print(
        "Saving model and optimizer state at iteration {} to {}".format(
            iteration, filepath
        )
    )

    # NOTE (Sam): learning rate not accessible here
    torch.save(
        {
            "state_dict": model.state_dict(),
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
        },
        filepath,
    )


# NOTE (Sam): not necessary for ray dataset
collate_fn = DataCollate()


def _train_step(
    batch,
    model,
    optim,
    iteration,
    epoch,
    steps_per_sample,
    scaler,
    scheduler,
    criterion,
    attention_kl_loss,
    kl_loss_start_iter,
    binarization_start_iter,
):
    print(datetime.now(), "entering train step:", iteration)
    if iteration >= binarization_start_iter:
        binarize = True
    else:
        binarize = False

    optim.zero_grad()

    with autocast(enabled=False):

        batch_dict = batch # torch DataLoader
        # batch_dict = collate_fn(batch) # ray dataset
        # TODO (Sam): move to batch.go_gpu().
        mel = to_gpu(batch_dict["mel"])
        speaker_ids = to_gpu(batch_dict["speaker_ids"])
        attn_prior = to_gpu(batch_dict["attn_prior"])
        f0 = to_gpu(batch_dict["f0"])
        voiced_mask = to_gpu(batch_dict["voiced_mask"])
        p_voiced = to_gpu(batch_dict["p_voiced"])
        text = to_gpu(batch_dict["text"])
        in_lens = to_gpu(batch_dict["input_lengths"])
        out_lens = to_gpu(batch_dict["output_lengths"])
        energy_avg = to_gpu(batch_dict["energy_avg"])
        audio_embedding = to_gpu(batch_dict["audio_embedding"])

        outputs = model(
            mel,
            speaker_ids,
            text,
            in_lens,
            out_lens,
            binarize_attention=binarize,
            attn_prior=attn_prior,
            f0=f0,
            energy_avg=energy_avg,
            voiced_mask=voiced_mask,
            p_voiced=p_voiced,
            audio_embedding=audio_embedding,
        )

        loss_outputs = criterion(outputs, in_lens, out_lens)

        print_list = []
        loss = None
        for k, (v, w) in loss_outputs.items():
            if w > 0:
                loss = v * w if loss is None else loss + v * w
            print_list.append("  |  {}: {:.3f}".format(k, v))

        w_bin = criterion.loss_weights.get("binarization_loss_weight", 1.0)
        if binarize and iteration >= kl_loss_start_iter:
            binarization_loss = attention_kl_loss(outputs["attn"], outputs["attn_soft"])
            loss += binarization_loss * w_bin
        else:
            binarization_loss = torch.zeros_like(loss)
        loss_outputs["binarization_loss"] = (binarization_loss, w_bin)
    grad_clip_val = 1.0  # TODO (Sam): make this a config option
    print(print_list)
    scaler.scale(loss).backward()
    if grad_clip_val > 0:
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)

    scaler.step(optim)
    scaler.update()

    metrics = {"loss": loss.item()}
    for k, (v, w) in loss_outputs.items():
        metrics[k] = v.item()

    print("iteration: ", iteration, datetime.now())
    log_sample = iteration % steps_per_sample == 0
    log_checkpoint = iteration % train_config["iters_per_checkpoint"] == 0

    if log_sample and session.get_world_rank() == 0:
        model.eval()
        # TODO (Sam): adding tf output logging and out of distribution inference
        # TODO (Sam): add logging of ground truth
        images, audios = get_log_audio(
            outputs,
            batch_dict,
            train_config,
            model,
            speaker_ids,
            text,
            f0,
            energy_avg,
            voiced_mask,
        )
        # TODO (Sam): make this clean
        gt_path = "/usr/src/app/radtts/ground_truth"
        oos_embs = os.listdir(gt_path)
        # this doesn't help for reasons described above
        for oos_name in oos_embs:
            audio_embedding_oos = torch.load(f"{gt_path}/{oos_name}").cuda()
            _, audios_oos = get_log_audio(
                outputs,
                batch_dict,
                train_config,
                model,
                speaker_ids,
                text,
                f0,
                energy_avg,
                voiced_mask,
                oos_name=oos_name,
                audio_embedding_oos=audio_embedding_oos,
            )
            audios.update(audios_oos)
        log(metrics, audios)
        model.train()
    else:
        log(metrics)

    session.report(metrics)
    if log_checkpoint and session.get_world_rank() == 0:

        checkpoint_path = f"{train_config['output_directory']}/model_{iteration}.pt"
        save_checkpoint(model, optim, iteration, checkpoint_path)

    print(f"Loss: {loss.item()}")


# NOTE (Sam): uncomment to run with torch DataLoader rather than ray dataset
def train_epoch(
    train_dataloader,
    dataset_shard,
    batch_size,
    model,
    optim,
    steps_per_sample,
    scaler,
    scheduler,
    criterion,
    attention_kl_loss,
    kl_loss_start_iter,
    binarization_start_iter,
    epoch,
    iteration,
):
    # def train_epoch(dataset_shard, batch_size, model, optim, steps_per_sample, scaler, scheduler, criterion, attention_kl_loss, kl_loss_start_iter, binarization_start_iter, epoch, iteration):
    # for batch_idx, ray_batch_df in enumerate(
    #     dataset_shard.iter_batches(batch_size=batch_size, prefetch_blocks=6)
    # ):
    # NOTE (Sam): uncomment to run with torch DataLoader rather than ray dataset
    for batch in train_dataloader:
        _train_step(
            # ray_batch_df,
            # NOTE (Sam): uncomment to run with torch DataLoader rather than ray dataset
            batch,
            model,
            optim,
            iteration,
            epoch,
            steps_per_sample,
            # iters_per_checkpoint,
            scaler,
            scheduler,
            criterion,
            attention_kl_loss,
            kl_loss_start_iter,
            binarization_start_iter,
        )
        iteration += 1

    return iteration


def warmstart(checkpoint_path, model, include_layers=[], ignore_layers_warmstart=[]):
    pretrained_dict = torch.load(checkpoint_path, map_location="cpu")
    pretrained_dict = pretrained_dict["state_dict"]

    is_module = True
    if is_module:
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        pretrained_dict = new_state_dict

    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("Warm started from {}".format(checkpoint_path))
    model.train()
    return model


def train_func(config: dict):
    setup_wandb(
        config, project="radtts-ray", entity="uberduck-ai", rank_zero_only=False
    )
    print("CUDA AVAILABLE: ", torch.cuda.is_available())
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    steps_per_sample = config["steps_per_sample"]
    sigma = config["sigma"]
    kl_loss_start_iter = config["kl_loss_start_iter"]
    binarization_start_iter = config["binarization_start_iter"]
    model = RADTTS(
        **model_config,
    )

    if config["warmstart_checkpoint_path"] != "":
        warmstart(config["warmstart_checkpoint_path"], model)

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
        model.parameters(), config["learning_rate"], weight_decay=config["weight_decay"]
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
        config["n_group_size"],
        config["dur_model_config"],
        config["f0_model_config"],
        config["energy_model_config"],
        vpred_model_config=config["v_model_config"],
        loss_weights=config["loss_weights"],
    )
    attention_kl_loss = AttentionBinarizationLoss()
    iteration = 0
    for epoch in range(start_epoch, start_epoch + epochs):
        # NOTE (Sam): uncomment to run with torch DataLoader rather than ray dataset
        iteration = train_epoch(
            train_dataloader,
            dataset_shard,
            batch_size,
            model,
            optim,
            steps_per_sample,
            scaler,
            scheduler,
            criterion,
            attention_kl_loss,
            kl_loss_start_iter,
            binarization_start_iter,
            epoch,
            iteration,
        )
        # iteration = train_epoch(dataset_shard, batch_size, model, optim, steps_per_sample, scaler, scheduler, criterion, attention_kl_loss, kl_loss_start_iter, binarization_start_iter, epoch, iteration)

def prepare_dataloaders(data_config, n_gpus, batch_size):
    # Get data, data loaders and collate function ready
    ignore_keys = ["training_files", "validation_files"]
    print("initializing training dataloader")
    trainset = Data(
        data_config["training_files"],
        **dict((k, v) for k, v in data_config.items() if k not in ignore_keys),
    )

    print("initializing validation dataloader")
    data_config_val = data_config.copy()
    data_config_val["aug_probabilities"] = None  # no aug in val set
    valset = Data(
        data_config["validation_files"],
        **dict((k, v) for k, v in data_config_val.items() if k not in ignore_keys),
        speaker_ids=trainset.speaker_ids,
    )

    collate_fn = DataCollate()

    train_sampler, shuffle = None, True
    if n_gpus > 1:
        train_sampler, shuffle = DistributedSampler(trainset), False

    train_loader = DataLoader(
        trainset,
        num_workers=8,
        shuffle=shuffle,
        sampler=train_sampler,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )

    return train_loader, valset, collate_fn

from ..vocoders.hifigan import get_vocoder

# NOTE (Sam): denoiser not used here in contrast with radtts repo
def load_vocoder(vocoder_state_dict, vocoder_config, to_cuda=True):

    h = AttrDict(vocoder_config)
    if "gaussian_blur" in vocoder_config:
        vocoder_config["gaussian_blur"]["p_blurring"] = 0.0
    else:
        vocoder_config["gaussian_blur"] = {"p_blurring": 0.0}
        h["gaussian_blur"] = {"p_blurring": 0.0}

    vocoder = Generator(h)
    vocoder.load_state_dict(vocoder_state_dict)
    if to_cuda:
        vocoder.cuda()

    vocoder.eval()

    return vocoder

from ..utils import parse_args

if __name__ == "__main__":

    args = parse_args(sys.argv[1:])
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    data_config = config["data_config"]
    train_config = config["train_config"]
    model_config = config["model_config"]
    MAX_WAV_VALUE = data_config["max_wav_value"]

    # NOTE (Sam): we can use ray trainer with ray datasets or torch dataloader.  torch dataloader is a little faster for now.
    # See comments for optionality

    symbol_set = data_config["symbol_set"]
    cleaner_names = data_config["cleaner_names"]
    heteronyms_path = data_config["heteronyms_path"]
    phoneme_dict_path = data_config["phoneme_dict_path"]
    p_phoneme = data_config["p_phoneme"]
    handle_phoneme = data_config["handle_phoneme"]
    handle_phoneme_ambiguous = data_config["handle_phoneme_ambiguous"]
    prepend_space_to_text = data_config["prepend_space_to_text"]
    append_space_to_text = data_config["append_space_to_text"]
    add_bos_eos_to_text = data_config["add_bos_eos_to_text"]

    stft = TacotronSTFT(
        filter_length=data_config["filter_length"],
        hop_length=data_config["hop_length"],
        win_length=data_config["win_length"],
        sampling_rate=22050,
        n_mel_channels=data_config["n_mel_channels"],
        mel_fmin=data_config["mel_fmin"],
        mel_fmax=data_config["mel_fmax"],
    )

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

    train_config = config["train_config"]
    model_config = config["model_config"]
    data_config = config["data_config"]

    # NOTE (Sam): uncomment for ray dataset training
    # ray_dataset = get_ray_dataset()
    # ray_dataset.fully_executed()
    train_config["n_group_size"] = model_config["n_group_size"]
    train_config["dur_model_config"] = model_config["dur_model_config"]
    train_config["f0_model_config"] = model_config["f0_model_config"]
    train_config["energy_model_config"] = model_config["energy_model_config"]
    train_config["v_model_config"] = model_config["v_model_config"]
    vocoder = get_vocoder()
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_config,
        scaling_config=ScalingConfig(
            num_workers=2,
            use_gpu=True,
            resources_per_worker=dict(CPU=8, GPU=1)
        ),
        run_config=RunConfig(
            # NOTE (Sam): uncomment for saving on anyscale
            # sync_config=SyncConfig(upload_dir=s3_upload_folder)
            sync_config=SyncConfig()
        ),
        # NOTE (Sam): uncomment for ray dataset training
        # datasets={"train": ray_dataset},
    )

    result = trainer.fit()
