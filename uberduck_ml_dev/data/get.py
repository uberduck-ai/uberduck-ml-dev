from torch.utils.data import DataLoader
import librosa
from pathlib import Path
from tqdm import tqdm
import torch
import os

from ..data.data import DataMel, DataPitch
from ..data.collate import CollateBlank
from ..data.processor import Processor


def get_parallel_torch(data):
    data_loader = DataLoader(
        data, batch_size=32, collate_fn=CollateBlank(), num_workers=8
    )
    for batch in data_loader:
        pass


# TODO (Sam): use get_parallel_torch to reduce boilerplate.
# NOTE (Sam): assumes data is in a directory structure like:
# /tmp/{uuid}/resampled_normalized.wav
# These functions add spectrogram.pt, f0.pt, and coqui_resnet_512_emb.pt to each file-specific directory.
def get_mels(paths, data_config, target_paths):
    data = DataMel(audiopaths=paths, data_config=data_config, target_paths=target_paths)

    collate_fn = CollateBlank()

    data_loader = DataLoader(
        data,
        batch_size=32,
        collate_fn=collate_fn,
    )
    for batch in data_loader:
        pass  # computes in loader.


# NOTE (Sam): pitch, pitchf == f0 coarse, f0bak in rvc parlance.
# NOTE (Sam): sample_rate is also passed as part of data_config
# TODO (Sam): decide on sample_rate v sampling_rate
# NOTE (Sam): pyin (radtts) and parselmouth (rvc) methods seem to generate pitches of different lengths.
def get_pitches(
    paths,
    data_config=None,
    target_folders=None,
    method="parselmouth",
    sample_rate=None,
    recompute=False,
):
    data = DataPitch(
        audiopaths=paths,
        data_config=data_config,
        target_folders=target_folders,
        method=method,
        sample_rate=data_config["sampling_rate"],
        recompute=recompute,
    )
    get_parallel_torch(data)


HUBERT_PATH = "hubert_embedding.pt"
F0_PATH = "f0.pt"
F0F_PATH = "f0f.pt"


# NOTE (Sam): this is different from the other get functions because it doesn't use torch dataset.
def get_hubert_embeddings(
    audiopaths, hubert_model, output_layer=9, hubert_path=HUBERT_PATH
):
    """Returns the abs path w.r.t penultimate directory name in audiopaths, e.g. suitable for /tmp/{uuid}/resampled_normalized.wav."""
    hubert_abs_paths = []
    for audiopath in tqdm(audiopaths):
        folder_path = str(Path(*Path(audiopath).parts[:-1]))
        hubert_abs_path = os.path.join(folder_path, hubert_path)
        # TODO (Sam): add hashing to avoid mistakenly not recomputing.
        if not os.path.exists(hubert_abs_path):
            # NOTE (Sam): Hubert expects 16k sample rate.
            audio0, sr = librosa.load(audiopath, sr=16000)
            feats = torch.from_numpy(audio0)
            feats = feats.float()
            feats = feats.view(1, -1)
            padding_mask = torch.BoolTensor(feats.shape).to("cpu").fill_(False)
            inputs = {
                "source": feats.to("cpu"),
                "padding_mask": padding_mask,
                "output_layer": output_layer,
            }

            with torch.no_grad():
                logits = hubert_model.extract_features(**inputs)
                feats = hubert_model.final_proj(logits[0])
                torch.save(feats[0], hubert_abs_path)

        hubert_abs_paths.append(hubert_abs_path)

    return hubert_abs_paths


def get(
    processing_function,
    saving_function,
    loading_function,
    source_paths,
    target_paths,
    recompute,
):
    function_ = lambda source_path, target_path: saving_function(
        processing_function(loading_function(source_path)), target_path
    )
    processor = Processor(
        function_=function_,
        source_paths=source_paths,
        target_paths=target_paths,
        recompute=recompute,
    )

    get_parallel_torch(processor)
