from uberduck_ml_dev.data_loader import TextMelCollate, TextMelDataset, oversample
from torch.utils.data import DataLoader


class TestTextMelCollation:
    def test_oversample(self):

        mock_fts = [
            ("speaker0/1.wav", "Test one two", "0"),
            ("speaker0/2.wav", "Test one two", "0"),
            ("speaker1/1.wav", "Test one two", "1"),
        ]
        assert oversample(mock_fts, {"1": 3}) == [
            ("speaker0/1.wav", "Test one two", "0"),
            ("speaker0/2.wav", "Test one two", "0"),
            ("speaker1/1.wav", "Test one two", "1"),
            ("speaker1/1.wav", "Test one two", "1"),
            ("speaker1/1.wav", "Test one two", "1"),
        ]

    def test_batch_structure(self):
        ds = TextMelDataset(
            "tests/fixtures/val.txt",
            ["english_cleaners"],
            0.0,
            80,
            22050,
            0,
            8000,
            1024,
            256,
            padding=None,
            win_length=1024,
            debug=True,
            debug_dataset_size=12,
            symbol_set="default",
        )
        len(ds)
        collate_fn = TextMelCollate()
        dl = DataLoader(ds, 12, collate_fn=collate_fn)
        for i, batch in enumerate(dl):
            assert len(batch) == 7

    def test_batch_dimensions(self):

        ds = TextMelDataset(
            "tests/fixtures/val.txt",
            ["english_cleaners"],
            0.0,
            80,
            22050,
            0,
            8000,
            1024,
            256,
            padding=None,
            win_length=1024,
            debug=True,
            debug_dataset_size=12,
            include_f0=True,
            symbol_set="default",
        )
        assert len(ds) == 1
        collate_fn = TextMelCollate(include_f0=True)
        dl = DataLoader(ds, 12, collate_fn=collate_fn)
        for i, batch in enumerate(dl):
            # text_padded,
            # input_lengths,
            # mel_padded,
            # gate_padded,
            # output_lengths,
            # speaker_ids,
            (
                text_padded,
                input_lengths,
                mel_padded,
                gate_padded,
                output_lengths,
                speaker_ids,
                *_,
            ) = batch
            assert output_lengths.item() == 566, print(
                "output lengths: ", output_lengths
            )
            assert gate_padded.size(1) == 566
            assert mel_padded.size(2) == 566
            assert len(batch) == 7

    def test_batch_dimensions_partial(self):

        ds = TextMelDataset(
            "tests/fixtures/val.txt",
            ["english_cleaners"],
            0.0,
            80,
            22050,
            0,
            8000,
            1024,
            256,
            padding=None,
            win_length=1024,
            debug=True,
            debug_dataset_size=12,
            include_f0=True,
            symbol_set="default",
        )
        assert len(ds) == 1
        collate_fn = TextMelCollate(n_frames_per_step=5, include_f0=True)
        dl = DataLoader(ds, 12, collate_fn=collate_fn)
        # text_padded,
        # input_lengths,
        # mel_padded,
        # gate_padded,
        # output_lengths,
        # speaker_ids,
        for i, batch in enumerate(dl):
            (
                text_padded,
                input_lengths,
                mel_padded,
                gate_padded,
                output_lengths,
                speaker_ids,
                *_,
            ) = batch
            assert output_lengths.item() == 566, output_lengths.item()
            assert mel_padded.size(2) == 570, print("actual shape: ", mel_padded.shape)
            assert gate_padded.size(1) == 570, print(
                "actual shape: ", gate_padded.shape
            )
            assert len(batch) == 7
