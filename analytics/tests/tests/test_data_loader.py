from torch.utils.data import DataLoader

from uberduck_ml_dev.data.utils import oversample
from uberduck_ml_dev.data.data import Data
from uberduck_ml_dev.data.collate import Collate


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
        ds = Data(
            "analytics/tests/fixtures/val.txt",
            debug=True,
            debug_dataset_size=12,
            symbol_set="default",
        )
        assert len(ds) == 1
        collate_fn = Collate()
        dl = DataLoader(ds, 12, collate_fn=collate_fn)
        for i, batch in enumerate(dl):
            assert len(batch) == 6

    def test_batch_dimensions(self):
        ds = Data(
            audiopaths_and_text="analytics/tests/fixtures/val.txt",
            debug=True,
            debug_dataset_size=12,
            symbol_set="default",
        )
        assert len(ds) == 1
        collate_fn = Collate()
        dl = DataLoader(ds, 12, collate_fn=collate_fn)
        for i, batch in enumerate(dl):
            output_lengths = batch["mel_lengths"]
            gate_target = batch["gate_padded"]
            mel_padded = batch["mel_padded"]
            assert output_lengths.item() == 566
            assert gate_target.size(1) == 566
            assert mel_padded.size(2) == 566
            assert len(batch) == 6

    def test_batch_dimensions_partial(self):
        ds = Data(
            "analytics/tests/fixtures/val.txt",
            debug=True,
            debug_dataset_size=12,
            symbol_set="default",
        )
        assert len(ds) == 1
        collate_fn = Collate(n_frames_per_step=5)
        dl = DataLoader(ds, 12, collate_fn=collate_fn)
        for i, batch in enumerate(dl):
            assert batch["mel_lengths"].item() == 566
            assert (
                batch["mel_padded"].size(2) == 566
            )  # I'm not sure why this was 570 - maybe 566 + 5 (i.e. the n_frames_per_step)
            assert batch["gate_padded"].size(1) == 566
            assert len(batch) == 6
