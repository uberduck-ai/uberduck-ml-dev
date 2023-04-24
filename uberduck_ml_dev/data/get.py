from torch.utils.data import DataLoader

from uberduck_ml_dev.data.data import DataMel, DataPitch, DataEmbedding
from uberduck_ml_dev.data.collate import CollateBlank


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


def get_embeddings(paths, data_config, resnet_se_model_path, resnet_se_config_path):
    data = DataEmbedding(
        audiopaths=paths,
        resnet_se_model_path=resnet_se_model_path,
        resnet_se_config_path=resnet_se_config_path,
    )

    collate_fn = CollateBlank()

    data_loader = DataLoader(
        data,
        batch_size=32,
        collate_fn=collate_fn,
    )
    for batch in data_loader:
        pass  # computes in loader.


def get_pitches(paths, data_config):
    data = DataPitch(audiopaths=paths, data_config=data_config)

    collate_fn = CollateBlank()

    data_loader = DataLoader(
        data,
        batch_size=32,
        collate_fn=collate_fn,
    )
    for batch in data_loader:
        pass  # computes in loader.
