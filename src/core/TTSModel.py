import torch
import pandas as pd

class TTSModel(torch.nn.Module):
    def infer(self):
        raise NotImplemented

    def forward(self):
        raise NotImplemented

    def from_pretrained(self):
        raise NotImplemented

    @classmethod
    def create(cls, name, opts, folders, all_speakers = True):

        model_cls = cls.get_class(name)
        folders = pd.read_csv(folders)
        for folder in folders:
            

        return model_cls(opts)
