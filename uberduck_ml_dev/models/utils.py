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


def load_pretrained(model, checkpoint_path, key_="generator"):
    # NOTE (Sam): uncomment for download on anyscale
    # response = requests.get(HIFI_GAN_GENERATOR_URL, stream=True)
    # bio = BytesIO(response.content)
    loaded = torch.load(checkpoint_path)
    model.load_state_dict(loaded[key_])


# relevant part of state dict is the dec part
def filter_dict(dict_input, key_prefix):
    new_dict = {}
    for key, value in dict_input.items():
        split_key = key.split(".")
        if split_key[0] == key_prefix:
            new_dict[".".join(split_key[1:])] = value
    return new_dict


def get_subdict(state_dict, matching_keys):
    for matching_key in matching_keys:
        print(type(state_dict))
        if matching_key[0] == "dict":
            state_dict = state_dict[matching_key[1]]
        if matching_key[0] == "object":
            state_dict = filter_dict(state_dict, matching_key[1])
    return state_dict


# def filter_dict(dict_input, key_prefix):
#     new_dict = {}
#     for key, value in dict_input.items():
#         split_key = key.split(".")
#         if split_key[0] == key_prefix:
#             new_dict[".".join(split_key[1:])] = value
#     return new_dict


# def get_subdict(state_dict, matching_keys):
#     for matching_key in matching_keys:
#         print(type(state_dict))
#         if matching_key[0] == "dict":
#             state_dict = state_dict[matching_key[1]]
#         if matching_key[0] == "object":
#             state_dict = filter_dict(state_dict, matching_key[1])
#     print(type(state_dict))
#     return state_dict
