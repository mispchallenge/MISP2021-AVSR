from typing import Optional
from typing import Tuple

import torch


class AbsWavAug(torch.nn.Module):
    """Abstract class for the augmentation of spectrogram

    The process-flow:

    Frontend  -> SpecAug -> Normalization -> Encoder -> Decoder
    """

    def forward(
        self, x: torch.Tensor
    ) :
        raise NotImplementedError
