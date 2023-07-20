import torch
from collections import OrderedDict

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from ...data.data import DataRADTTS as Data
from ...data.collate import DataCollateRADTTS as DataCollate


# TODO (Sam): warmstart should load optimizer state as well.
# load_pretrained should just be the state_dict
def warmstart(
    checkpoint_path, model, include_layers=[], ignore_layers_warmstart=[], strict=False
):
    pretrained_dict = torch.load(checkpoint_path, map_location="cpu")
    pretrained_dict = pretrained_dict["state_dict"]

    is_module = False
    if list(pretrained_dict.keys())[0].startswith("module."):
        is_module = True
    if is_module:
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        pretrained_dict = new_state_dict

    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=strict)
    print("Warm started from {}".format(checkpoint_path))
    model.train()
    return model


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
        num_workers=data_config["num_workers"] - 1,
        shuffle=shuffle,
        sampler=train_sampler,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )

    return train_loader, valset, collate_fn
