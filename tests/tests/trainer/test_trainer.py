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


class TestTacotron2Trainer:
    def test_gradient_step(self, lj_trainer):

        torch.manual_seed(1234)
        lj_trainer.train()

        # NOTE (Sam): this number was taken from master on 8/24/22.
        # Since folks are training on master right now, it is a reasonable benchmark
        loss_target = 0.2
        assert torch.isclose(lj_trainer.loss, loss_target)
