import torch

from typing import List, Optional, Dict

from ..utils.utils import to_gpu


class Batch(Dict):
    # NOTE (Sam): isn't gate target redundant to output length
    # NOTE (Sam): I don't see any benefit to using namedtuple over dictionary
    # Supposedly, the idea was that namedtuple enables retention of tuple ordering for backwards compatibility
    # However, both X and Y contain certain parameters (e.g. mels for teacher forcing).
    # Thus, we either need to have separate redundant classes, or unite under one batch class
    # I think a united class is much easier to understand and only slightly less efficient
    # However, a united class necessarily destroys the existing ordering

    text_int_padded: Optional[torch.LongTensor] = None
    input_lengths: Optional[torch.LongTensor] = None
    mel_padded: Optional[torch.FloatTensor] = None  # for teacher forcing
    gate_target: Optional[
        torch.LongTensor
    ] = None  # NOTE (Sam): could be bool -  for teacher forcing
    output_lengths: Optional[torch.LongTensor] = None
    speaker_ids: Optional[torch.LongTensor] = None
    gst: Optional[torch.Tensor] = None
    mel_outputs: Optional[torch.Tensor] = None  # predicted
    mel_outputs_postnet: Optional[torch.Tensor] = None
    gate_predicted: Optional[torch.LongTensor] = None  # could be bool
    alignments: Optional[torch.Tensor] = None

    def subset(self, keywords, fragile=False):
        d = {}
        for k in keywords:
            try:
                d[k] = self[k]
            except KeyError:
                if fragile:
                    raise
        return Batch(**d)

    def to_gpu(self):

        batch_gpu = Batch(**{k: to_gpu(v) for k, v in self.items()})
        return batch_gpu

    # NOTE (Sam): Dict doesn't seem to have this method so have to write it.
    # NamedTuple does have it however
    def _field_defaults(self):

        _field_defaults = dict(
            text_int_padded=None,
            input_lengths=None,
            mel_padded=None,
            gate_target=None,
            output_lengths=None,
            speaker_ids=None,
            gst=None,
            mel_outputs=None,
            mel_outputs_postnet=None,
            gate_predicted=None,
            alignments=None,
        )
        return _field_defaults
