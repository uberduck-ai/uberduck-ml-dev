import torch
import pickle
import os
import inspect


def load_checkpoint(filepath, device, pickle_module=pickle):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(
        filepath,
        map_location=torch.device(device),
        pickle_module=pickle_module,
    )
    print("Complete.")
    return checkpoint_dict


def load_pretrained(model, checkpoint_path, key_="generator"):
    # NOTE (Sam): uncomment for download on anyscale
    # response = requests.get(HIFI_GAN_GENERATOR_URL, stream=True)
    # bio = BytesIO(response.content)
    loaded = torch.load(checkpoint_path)
    model.load_state_dict(loaded[key_])


def filter_valid_args(func, **kwargs):
    valid_keys = inspect.signature(func).parameters.keys()
    return {key: value for key, value in kwargs.items() if key in valid_keys}
