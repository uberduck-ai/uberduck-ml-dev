__all__ = ["TTSModel", "DEFAULTS"]

import torch
from torch import nn

from ..text.symbols import SYMBOL_SETS
from ..vendor.tfcompat.hparam import HParams


class TTSModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.symbol_set = hparams.symbol_set
        self.n_symbols = len(SYMBOL_SETS[self.symbol_set])
        self.n_speakers = hparams.n_speakers
        # symbols = __import__('uberduck_ml_dev.text.' + hparams.symbols)

    def infer(self):
        raise NotImplemented

    def forward(self):
        raise NotImplemented

    def from_pretrained(
        self, warm_start_path=None, device="cpu", ignore_layers=None, model_dict=None
    ):
        model_dict = model_dict or dict()
        if warm_start_path is None and model_dict is None:
            raise Exception(
                "TTSModel.from_pretrained requires a warm_start_path or state_dict"
            )
        if warm_start_path is not None:
            checkpoint = torch.load(warm_start_path, map_location=device)
            if (
                "state_dict" in checkpoint.keys()
            ):  # TODO: remove state_dict once off nvidia
                model_dict = checkpoint["state_dict"]
            if "model" in checkpoint.keys():
                model_dict = checkpoint["model"]
        if ignore_layers:
            model_dict = {k: v for k, v in model_dict.items() if k not in ignore_layers}
        dummy_dict = self.state_dict()

        for k in self.state_dict().keys():
            if k not in model_dict.keys():
                print(
                    f"WARNING! Attempting to load a model with out the {k} layer. This could lead to unexpected results during evaluation."
                )

        dummy_dict.update(model_dict)
        model_dict = dummy_dict
        self.load_state_dict(model_dict)
        if device == "cuda":
            self.cuda()

    def to_checkpoint(self):
        return dict(model=self.state_dict())

    @classmethod
    def create(cls, name, opts, folders, all_speakers=True):
        pass


DEFAULTS = HParams(
    p_arpabet=1.0,
    seed=1234,
    # NOTE (Sam): make sure users change their configurations for cudnn_enabled = True.
    cudnn_enabled=False,
)
