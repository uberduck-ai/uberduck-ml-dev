import torch

from typing import List, Optional, NamedTuple


class Batch(NamedTuple):
    text_int_padded: Optional[torch.LongTensor] = None
    input_lengths: Optional[List[torch.LongTensor]] = None
    mel_padded: Optional[torch.FloatTensor] = None
    gate_target: Optional[List[torch.LongTensor]] = None  # NOTE (Sam): could be bool
    output_lengths: Optional[List[torch.LongTensor]] = None
    speaker_ids: Optional[List[torch.LongTensor]] = None
    gst: Optional[torch.Tensor] = None
    # f0_padded: List = (None,)
    # durations_padded: Optional[torch.Tensor] = (None,)
    # max_len: Optional[int] = (None,)
    # mel_outputs: Optional[torch.Tensor] = (None,)
    # mel_outputs_postnet: Optional[torch.Tensor] = (None,)
    # gate_target: Optional[torch.Tensor] = (None,)
    # alignments: Optional[torch.Tensor] = (None,)
    # predicted_durations: Optional[torch.Tensor] = (None,)
    # text: List = None
