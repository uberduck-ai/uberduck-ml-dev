import torch

from typing import List, Optional, NamedTuple


class Batch(NamedTuple):
    text_int_padded: Optional[torch.Tensor] = (None,)
    input_lengths: List = (None,)
    mel_padded: Optional[torch.Tensor] = (None,)
    gate_pred: List = (None,)
    output_lengths: List = (None,)
    speaker_ids: List = (None,)
    f0_padded: List = (None,)
    gst: Optional[torch.Tensor] = (None,)
    durations_padded: Optional[torch.Tensor] = (None,)
    max_len: Optional[int] = (None,)
    mel_outputs: Optional[torch.Tensor] = (None,)
    mel_outputs_postnet: Optional[torch.Tensor] = (None,)
    gate_target: Optional[torch.Tensor] = (None,)
    alignments: Optional[torch.Tensor] = (None,)
    predicted_durations: Optional[torch.Tensor] = (None,)
    text: List = None
