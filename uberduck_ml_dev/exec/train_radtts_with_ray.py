import sys
import json

from ray.air.config import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer
from ray.tune import SyncConfig
from ray.train.torch import TorchTrainer, TorchTrainer
from ray.air.config import ScalingConfig, RunConfig

from .train import train_func
from ..models.common import TacotronSTFT
from ..text.text_processing import TextProcessing


HIFI_GAN_CONFIG_URL = "https://uberduck-models-us-west-2.s3.us-west-2.amazonaws.com/hifigan_22khz_config.json"
HIFI_GAN_GENERATOR_URL = "https://uberduck-models-us-west-2.s3.us-west-2.amazonaws.com/hifigan_libritts100360_generator0p5.pt"
HIFI_GAN_CONFIG_PATH = "/usr/src/app/radtts/models/hifigan_22khz_config.json"
HIFI_GAN_GENERATOR_PATH = (
    "/usr/src/app/radtts/models/hifigan_libritts100360_generator0p5.pt"
)

from ..utils import parse_args

if __name__ == "__main__":

    args = parse_args(sys.argv[1:])
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    data_config = config["data_config"]

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

    # NOTE (Sam): uncomment for ray dataset training
    # ray_dataset = get_ray_dataset()
    # ray_dataset.fully_executed()
    
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=config,
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
