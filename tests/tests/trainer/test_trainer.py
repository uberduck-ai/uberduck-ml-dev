from uberduck_ml_dev.trainer.base import TTSTrainer
import torch
from uberduck_ml_dev.vendor.tfcompat.hparam import HParams


class TestTrainer:
    def test_trainer_base(self):
        hp = HParams(
            foo="bar",
            baz=123,
            checkpoint_path="this/is/a/test",
            cudnn_enabled=True,
            log_dir="this/is/a/test",
            seed=1234,
            symbol_set="default",
        )
        trainer = TTSTrainer(hp)
        assert trainer.hparams == hp
        assert trainer.foo == "bar"
        assert trainer.baz == 123
        mel = torch.load("tests/fixtures/stevejobs-1.pt")
        audio = trainer.sample(mel)
        assert audio.size(0) == 1
