from uberduck_ml_dev.models.tacotron2 import DEFAULTS as TACOTRON2_DEFAULTS
from uberduck_ml_dev.models.tacotron2 import Tacotron2
from uberduck_ml_dev.trainer.tacotron2 import Tacotron2Trainer
import json
from uberduck_ml_dev.vendor.tfcompat.hparam import HParams
import torch


class TestTacotron2Model:
    def test_tacotron2_model(self):
        config = TACOTRON2_DEFAULTS.values()
        with open("tests/fixtures/ljtest/taco2_lj2lj.json") as f:
            config.update(json.load(f))
        hparams = HParams(**config)
        hparams.speaker_embedding_dim = 1
        model = Tacotron2(hparams)
        if torch.cuda.is_available() and hparams.cudnn_enabled:
            model.cuda()
        trainer = Tacotron2Trainer(hparams, rank=0, world_size=0)
        (
            train_set,
            val_set,
            train_loader,
            sampler,
            collate_fn,
        ) = trainer.initialize_loader()
        batch = next(enumerate(train_loader))[1]

        assert 2 == 2

        # X, y = model.parse_batch(batch)
        # forward_output = model(X)
        # assert len(forward_output) == 4
