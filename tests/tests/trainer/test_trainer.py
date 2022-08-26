from uberduck_ml_dev.trainer.base import TTSTrainer
import torch
import math
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

    # NOTE (Sam): this test could be made twice as fast by only running a single epoch
    # As it is, the second gradient step is only useful for evaluating the loss
    def test_gradient_step(self, lj_trainer):

        torch.manual_seed(1234)
        lj_trainer.train()

        # NOTE (Sam): this number was taken from master on 8/24/22.
        # Since folks are training on master right now, it is a reasonable benchmark
        train_loss_start = 0.320
        train_loss_4_datapoints_1_iteration = 0.319

        assert math.isclose(lj_trainer.loss[0], train_loss_start, abs_tol=5e-4)

        assert math.isclose(
            lj_trainer.loss[1], train_loss_4_datapoints_1_iteration, abs_tol=5e-4
        )
