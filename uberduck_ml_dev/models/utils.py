import torch
import pickle
import os


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
