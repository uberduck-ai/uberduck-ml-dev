from torch.utils.data import DataLoader

from uberduck_ml_dev.data.data import DataMel, DataPitch, DataEmbedding
from uberduck_ml_dev.data.collate import CollateBlank


# NOTE (Sam): assumes data is in a directory structure like:
# /tmp/{uuid}/resampled_normalized.wav
# These functions add spectrogram.pt, f0.pt, and coqui_resnet_512_emb.pt to each file-specific directory.
def get_mels(paths, data_config):
    data = DataMel(audiopaths=paths, data_config=data_config)

    collate_fn = CollateBlank()

    data_loader = DataLoader(
        data,
        batch_size=32,
        collate_fn=collate_fn,
    )
    for batch in data_loader:
        pass  # computes in loader.


def get_embeddings(
    paths,
    data_config,
    resnet_se_model_path,
    resnet_se_config_path,
    subpath_truncation=41,
):
    data = DataEmbedding(
        audiopaths=paths,
        resnet_se_model_path=resnet_se_model_path,
        resnet_se_config_path=resnet_se_config_path,
        subpath_truncation=subpath_truncation,
    )

    collate_fn = CollateBlank()

    data_loader = DataLoader(
        data,
        batch_size=32,
        collate_fn=collate_fn,
    )
    for batch in data_loader:
        pass  # computes in loader.


def get_pitches(paths, data_config, subpath_truncation=41):
    data = DataPitch(
        audiopaths=paths, data_config=data_config, subpath_truncation=subpath_truncation
    )

    collate_fn = CollateBlank()

    data_loader = DataLoader(
        data,
        batch_size=32,
        collate_fn=collate_fn,
    )
    for batch in data_loader:
        pass  # computes in loader.
