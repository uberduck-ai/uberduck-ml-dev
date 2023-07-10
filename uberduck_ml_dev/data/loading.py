from scipy.io.wavfile import read
import torch
from einops import rearrange


def load_audio(path):
    rate, data = read(path)
    data = torch.Tensor(data)
    return rearrange(data, "t -> 1 t")
