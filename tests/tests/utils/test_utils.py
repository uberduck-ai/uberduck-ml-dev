import torch
from uberduck_ml_dev.utils.utils import get_mask_from_lengths, sequence_mask


class TestUtils:
    def test_mask_from_lengths(self):

        assert (
            get_mask_from_lengths(torch.LongTensor([1, 3, 2, 1]))
            == torch.Tensor(
                [
                    [True, False, False],
                    [True, True, True],
                    [True, True, False],
                    [True, False, False],
                ]
            )
        ).all()

    def test_sequence_mask(self):

        assert (
            sequence_mask(torch.tensor([1, 3, 2, 1]))
            == torch.Tensor(
                [
                    [True, False, False],
                    [True, True, True],
                    [True, True, False],
                    [True, False, False],
                ]
            )
        ).all()
        assert (
            sequence_mask(torch.tensor([1, 3, 2, 1]), 4)
            == torch.Tensor(
                [
                    [True, False, False, False],
                    [True, True, True, False],
                    [True, True, False, False],
                    [True, False, False, False],
                ]
            )
        ).all()
