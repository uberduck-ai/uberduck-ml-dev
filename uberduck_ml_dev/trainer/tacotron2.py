__all__ = ["Tacotron2Loss", "Tacotron2Trainer", "config", "DEFAULTS"]

from random import randint
import time
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from speechbrain.pretrained import EncoderClassifier

from ..data.collate import Collate
from ..models.tacotron2 import Tacotron2
from ..utils.plot import save_figure_to_numpy
from ..utils.utils import reduce_tensor
from ..monitoring.statistics import get_alignment_metrics

from ..vendor.tfcompat.hparam import HParams
from .base import DEFAULTS as TRAINER_DEFAULTS
from ..models.tacotron2 import DEFAULTS as TACOTRON2_DEFAULTS, INFERENCE
from ..models.torchmoji import TorchMojiInterface
from ..models.components.encoders.speaker_encoder import (
    get_speaker_encoding_for_dataset,
)
from ..utils.plot import (
    plot_attention,
    plot_gate_outputs,
    plot_spectrogram,
)
from ..text.util import text_to_sequence, random_utterance
from .base import TTSTrainer
from ..losses import Tacotron2Loss
from ..data.data import Data


class Tacotron2Trainer(TTSTrainer):

    # NOTE (Sam): make arguments explicit incl. self.hparams arguments added in super().__init__.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # NOTE (Sam): many of these are generic TTS of TTMel arguments that should be added to an intermediate TTS or TTMelTrainer class,
        # but the current TTSTrainer is en route to just be Trainer so I haven't added them yet.
        self.training_audiopaths_and_text = self.hparams.training_audiopaths_and_text
        self.val_audiopaths_and_text = self.hparams.val_audiopaths_and_text
        self.symbol_set = self.hparams.symbol_set
        self.mel_fmax = self.hparams.mel_fmax
        self.mel_fmin = self.hparams.mel_fmin
        self.n_mel_channels = self.hparams.n_mel_channels
        self.text_cleaners = self.hparams.text_cleaners
        self.pos_weight = self.hparams.pos_weight
        self.n_speakers = self.hparams.n_speakers
        self.p_arpabet = self.hparams.p_arpabet
        self.sampling_rate = self.hparams.sampling_rate
        self.filter_length = self.hparams.filter_length
        self.hop_length = self.hparams.hop_length
        self.win_length = self.hparams.win_length
        self.max_wav_value = self.hparams.max_wav_value
        self.sample_inference_text = self.hparams.sample_inference_text
        self.lr_decay_start = self.hparams.lr_decay_start
        self.lr_decay_rate = self.hparams.lr_decay_rate
        self.lr_decay_min = self.hparams.lr_decay_min
        # NOTE (Sam): there is a lot of ambiguity in how to name and initialize audio / speaker encoder and torchmoji
        self.has_audio_encoder = self.hparams.audio_encoder_path is not None

        if hasattr(self.hparams, "speaker_embeddings_path"):
            # TODO (Sam): move this to "load" type method in Data.
            get_speaker_encoding_for_dataset(
                self.hparams.speaker_embeddings_path,
                self.training_audiopaths_and_text,
                embedding_dim=192,
                speaker_encoder_path="/tmp/speaker_encoder",  # TODO (Sam): make this a hparam and load from DB
            )

            self.speaker_embeddings = torch.load(self.hparams.speaker_embeddings_path)
        else:
            self.speaker_embeddings = None
        # NOTE (Sam): gst_type == torchmoji and use_gst are currently redundant although syntactically different.
        # NOTE (Sam): its not clear we should lambdafy models here rather than the data_loader or helper function (e.g. speaker_encoder)
        if self.hparams.get("with_gst"):
            assert self.hparams.get(
                "torchmoji_vocabulary_file"
            ), "torchmoji_vocabulary_file must be set"
            assert self.hparams.get(
                "torchmoji_model_file"
            ), "torchmoji_model_file must be set"
            assert self.hparams.get("gst_dim"), "gst_dim must be set"

            self.torchmoji = TorchMojiInterface(
                self.hparams.get("torchmoji_vocabulary_file"),
                self.hparams.get("torchmoji_model_file"),
            )
            # TODO (Sam): rename gst to gsts[0].
            self.get_gst = lambda texts: self.torchmoji.encode_texts(texts)
        else:
            self.get_gst = None

        # TODO (Sam): datapoint specific encoder is really unused as of now.
        if self.has_audio_encoder:

            self.audio_encoder_model = EncoderClassifier.from_hparams(
                source=self.hparams.audio_encoder_path
            )
            self.audio_encoder_forward = (
                lambda x: self.audio_encoder_model.encode_batch(x)
            )
        else:
            self.audio_encoder_forward = None

    def log_training(
        self,
        model,
        X,
        y_pred,
        y,
        loss,
        mel_loss,
        gate_loss,
        mel_loss_batch,
        gate_loss_batch,
        grad_norm,
        step_duration_seconds,
    ):
        self.log("Loss/train", self.global_step, scalar=loss)
        self.log("MelLoss/train", self.global_step, scalar=mel_loss)
        self.log("GateLoss/train", self.global_step, scalar=gate_loss)
        self.log("GradNorm", self.global_step, scalar=grad_norm.item())
        self.log("LearningRate", self.global_step, scalar=self.learning_rate)
        self.log(
            "StepDurationSeconds",
            self.global_step,
            scalar=step_duration_seconds,
        )

        batch_levels = X["speaker_ids"]
        batch_levels_unique = torch.unique(batch_levels)
        for l in batch_levels_unique:
            mlb = mel_loss_batch[torch.where(batch_levels == l)[0]].mean()
            self.log(
                f"MelLoss/train/speaker{l.item()}",
                self.global_step,
                scalar=mlb,
            )
            glb = gate_loss_batch[torch.where(batch_levels == l)[0]].mean()
            self.log(
                f"GateLoss/train/speaker{l.item()}",
                self.global_step,
                scalar=glb,
            )
            self.log(
                f"Loss/train/speaker{l.item()}",
                self.global_step,
                scalar=mlb + glb,
            )

        if self.global_step % self.steps_per_sample == 0:
            mel_target, gate_target = y.subset(["mel_padded", "gate_target"]).values()
            alignment_metrics = get_alignment_metrics(y_pred["alignments"])
            alignment_diagonalness = alignment_metrics["diagonalness"]
            alignment_max = alignment_metrics["max"]
            sample_idx = randint(0, y_pred["mel_outputs_postnet"].size(0) - 1)
            audio = self.sample(mel=y_pred["mel_outputs_postnet"][sample_idx])
            audio_target = self.sample(mel=mel_target[sample_idx])
            self.log(
                "AlignmentDiagonalness/train",
                self.global_step,
                scalar=alignment_diagonalness,
            )
            self.log("AlignmentMax/train", self.global_step, scalar=alignment_max)
            self.log("AudioTeacherForced/train", self.global_step, audio=audio)
            self.log("TargetAudio/train", self.global_step, audio=audio_target)
            self.log(
                "MelPredicted/train",
                self.global_step,
                image=save_figure_to_numpy(
                    plot_spectrogram(
                        y_pred["mel_outputs_postnet"][sample_idx].data.cpu()
                    )
                ),
            )
            self.log(
                "MelTarget/train",
                self.global_step,
                image=save_figure_to_numpy(
                    plot_spectrogram(mel_target[sample_idx].data.cpu())
                ),
            )
            self.log(
                "Gate/train",
                self.global_step,
                image=save_figure_to_numpy(
                    plot_gate_outputs(
                        gate_targets=gate_target[sample_idx].data.cpu(),
                        gate_outputs=y_pred["gate_predicted"][sample_idx].data.cpu(),
                    )
                ),
            )
            input_length = X["input_lengths"][sample_idx].item()
            output_length = X["output_lengths"][sample_idx].item()
            self.log(
                "Attention/train",
                self.global_step,
                image=save_figure_to_numpy(
                    plot_attention(
                        y_pred["alignments"][sample_idx].data.cpu().transpose(0, 1),
                        encoder_length=input_length,
                        decoder_length=output_length,
                    )
                ),
            )
            for speaker_id in self.sample_inference_speaker_ids:
                self.sample_inference(
                    model,
                    self.sample_inference_text,
                    speaker_id,
                )

    def sample_inference(self, model, transcription=None, speaker_id=None):
        if self.rank is not None and self.rank != 0:
            return
        with torch.no_grad():
            if transcription is None:
                transcription = random_utterance()

            # NOTE (Sam): treat global encoders equivalently.
            if self.compute_gst:
                gst_embedding = torch.FloatTensor(self.compute_gst([transcription]))
            else:
                gst_embedding = None
            # NOTE (Sam): audio_encoder of data points is logically like teacher forcing since it acts on the ys.
            if self.audio_encoder_forward:
                # NOTE (Sam): get pre-computed centroid.
                speaker_embedding = self.speaker_embeddings[speaker_id]
            else:
                speaker_embedding = None

            utterance = torch.LongTensor(
                text_to_sequence(
                    transcription,
                    self.text_cleaners,
                    p_arpabet=self.p_arpabet,
                    symbol_set=self.symbol_set,
                )
            )[None]

            input_lengths = torch.LongTensor([utterance.shape[1]])
            speaker_id_tensor = torch.LongTensor([speaker_id])

            if self.cudnn_enabled and torch.cuda.is_available():
                utterance = utterance.cuda()
                input_lengths = input_lengths.cuda()
                gst_embedding = (
                    gst_embedding.cuda() if gst_embedding is not None else None
                )
                speaker_id_tensor = speaker_id_tensor.cuda()

            model.eval()

            sample_inference = model.forward(
                input_text=utterance,
                input_lengths=input_lengths,
                speaker_ids=speaker_id_tensor,
                # NOTE (Sam): this is None if using old multispeaker training, not None if using new pretrained encoder.
                audio_encoding=speaker_embedding,
                embedded_gst=gst_embedding,
                mode=INFERENCE,
            )
            model.train()
            try:
                audio = self.sample(sample_inference["mel_outputs_postnet"][0])
                self.log(f"SampleInference/{speaker_id}", self.global_step, audio=audio)
            except Exception as e:
                print(f"Exception raised while doing sample inference: {e}")
                print("Mel shape: ", sample_inference["mel_outputs_postnet"][0].shape)
            self.log(
                f"Attention/{speaker_id}/sample_inference",
                self.global_step,
                image=save_figure_to_numpy(
                    plot_attention(
                        sample_inference["alignments"][0].data.cpu().transpose(0, 1)
                    )
                ),
            )
            self.log(
                f"MelPredicted/{speaker_id}/sample_inference",
                self.global_step,
                image=save_figure_to_numpy(
                    plot_spectrogram(
                        sample_inference["mel_outputs_postnet"][0].data.cpu()
                    )
                ),
            )
            self.log(
                f"Gate/{speaker_id}/sample_inference",
                self.global_step,
                image=save_figure_to_numpy(
                    plot_gate_outputs(
                        gate_outputs=sample_inference["gate_predicted"][0].data.cpu()
                    )
                ),
            )

    def log_validation(
        self,
        X,
        y_pred,
        y,
        mean_loss,
        mean_mel_loss,
        mean_gate_loss,
        mel_loss_val,
        gate_loss_val,
        speakers_val,
    ):
        self.log("Loss/val", self.global_step, scalar=mean_loss)
        self.log("MelLoss/val", self.global_step, scalar=mean_mel_loss)
        self.log("GateLoss/val", self.global_step, scalar=mean_gate_loss)

        val_levels = speakers_val
        val_levels_unique = torch.unique(val_levels)
        for l in val_levels_unique:
            mlv = mel_loss_val[torch.where(val_levels == l)[0]].mean()
            self.log(
                f"MelLoss/val/speaker{l.item()}",
                self.global_step,
                scalar=mlv,
            )
            glv = gate_loss_val[torch.where(val_levels == l)[0]].mean()
            self.log(
                f"GateLoss/val/speaker{l.item()}",
                self.global_step,
                scalar=glv,
            )
            self.log(
                f"Loss/val/speaker{l.item()}",
                self.global_step,
                scalar=mlv + glv,
            )
        # Generate the sample from a random item from the last y_pred batch.
        alignment_metrics = get_alignment_metrics(y_pred["alignments"])
        alignment_diagonalness = alignment_metrics["diagonalness"]
        alignment_max = alignment_metrics["max"]
        sample_idx = randint(0, self.batch_size)
        audio = self.sample(mel=y_pred["mel_outputs_postnet"][sample_idx])
        audio_target = self.sample(mel=X["mel_padded"][sample_idx])
        self.log(
            "AlignmentDiagonalness/val", self.global_step, scalar=alignment_diagonalness
        )
        self.log("AlignmentMax/val", self.global_step, scalar=alignment_max)
        self.log("AudioTeacherForced/val", self.global_step, audio=audio)
        self.log("AudioTarget/val", self.global_step, audio=audio_target)
        self.log(
            "MelPredicted/val",
            self.global_step,
            image=save_figure_to_numpy(
                plot_spectrogram(y_pred["mel_outputs_postnet"][sample_idx].data.cpu())
            ),
        )
        self.log(
            "MelTarget/val",
            self.global_step,
            image=save_figure_to_numpy(
                plot_spectrogram(y["mel_padded"][sample_idx].data.cpu())
            ),
        )
        self.log(
            "Gate/val",
            self.global_step,
            image=save_figure_to_numpy(
                plot_gate_outputs(
                    gate_targets=y["gate_target"][sample_idx].data.cpu(),
                    gate_outputs=y_pred["gate_predicted"][sample_idx].data.cpu(),
                )
            ),
        )
        input_length = X["input_lengths"][sample_idx].item()
        output_length = X["output_lengths"][sample_idx].item()
        self.log(
            "Attention/val",
            self.global_step,
            image=save_figure_to_numpy(
                plot_attention(
                    y_pred["alignments"][sample_idx].data.cpu().transpose(0, 1),
                    encoder_length=input_length,
                    decoder_length=output_length,
                )
            ),
        )

    def initialize_loader(self, include_f0: bool = False, n_frames_per_step: int = 1):
        train_set = Data(
            **self.training_dataset_args,
            debug=self.debug,
            debug_dataset_size=self.batch_size,
        )
        val_set = Data(
            **self.val_dataset_args,
            debug=self.debug,
            debug_dataset_size=self.batch_size,
        )
        collate_fn = Collate(**self.collate_args)
        sampler = None
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=collate_fn,
        )
        return train_set, val_set, train_loader, sampler, collate_fn

    def train(
        self,
        interrupt_condition=lambda: False,
        interrupt_action=lambda: None,
        save_function=lambda epoch: None,
    ):

        train_start_time = time.perf_counter()
        print("start train", train_start_time)
        train_set, val_set, train_loader, sampler, collate_fn = self.initialize_loader()
        criterion = Tacotron2Loss(
            pos_weight=self.pos_weight
        )  # keep higher than 5 to make clips not stretch on

        model = Tacotron2(self.hparams)
        if self.device == "cuda" and self.cudnn_enabled:
            model = model.cuda()
            # NOTE (Sam): can I just call self.cuda() here?
            if self.speaker_embeddings is not None:
                self.speaker_embeddings = self.speaker_embeddings.cuda()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        start_epoch = 0

        # NOTE (Sam): distributed_run and fp16_run code deprecated and to be removed.
        if self.warm_start_name:
            model, optimizer, start_epoch = self.warm_start(model, optimizer)

        start_time, previous_start_time = time.perf_counter(), time.perf_counter()
        for epoch in range(start_epoch, self.epochs):
            if self.distributed_run:
                sampler.set_epoch(epoch)
            for batch_idx, batch in enumerate(train_loader):
                self.global_step += 1

                # Learning Rate decay, can be disabled if lr_decay_start is == 0 or None.
                if (self.global_step > self.lr_decay_start) and (
                    self.lr_decay_start not in [0, None]
                ):
                    learning_rate = self.learning_rate * (
                        np.exp(-self.global_step / self.lr_decay_rate)
                    )
                    learning_rate = max(self.lr_decay_min, learning_rate)
                    self.learning_rate = learning_rate
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = learning_rate

                model.zero_grad()

                # NOTE (Sam): Could call subsets directly in function arguments since model_input is only reused in logging.
                model_input = batch.subset(
                    [
                        "text_int_padded",
                        "input_lengths",
                        "speaker_ids",
                        "gst",
                        "mel_padded",
                        "output_lengths",
                        "audio_encodings",
                    ]
                )
                model_output = model(
                    input_text=model_input["text_int_padded"],
                    input_lengths=model_input["input_lengths"],
                    speaker_ids=model_input["speaker_ids"],
                    embedded_gst=model_input["gst"],
                    targets=model_input["mel_padded"],
                    audio_encoding=model_input["audio_encodings"],
                    output_lengths=model_input["output_lengths"],
                )
                target = batch.subset(["gate_target", "mel_padded"])
                mel_loss, gate_loss, mel_loss_batch, gate_loss_batch = criterion(
                    model_output=model_output, target=target
                )
                loss = mel_loss + gate_loss

                reduced_mel_loss = mel_loss.item()
                reduced_gate_loss = gate_loss.item()
                reduced_gate_loss_batch = gate_loss_batch.detach()
                reduced_mel_loss_batch = mel_loss_batch.detach()

                reduced_loss = reduced_mel_loss + reduced_gate_loss
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(), self.grad_clip_thresh
                )
                optimizer.step()

                step_duration_seconds = time.perf_counter() - start_time
                # NOTE (Sam): need to unify names to match forward
                self.log_training(
                    model,
                    X=model_input,
                    y_pred=model_output,
                    y=target,
                    loss=reduced_loss,
                    mel_loss=reduced_mel_loss,
                    gate_loss=reduced_gate_loss,
                    mel_loss_batch=reduced_mel_loss_batch,
                    gate_loss_batch=reduced_gate_loss_batch,
                    grad_norm=grad_norm,
                    step_duration_seconds=step_duration_seconds,
                )
                previous_start_time = start_time
                start_time = time.perf_counter()
                log_str = f"epoch: {epoch}/{self.epochs} | batch: {batch_idx}/{len(train_loader)} | loss: {reduced_loss:.3f} | mel: {reduced_mel_loss:.3f} | gate: {reduced_gate_loss:.3f} | t: {start_time - previous_start_time:.3f}s | w: {(time.perf_counter() - train_start_time)/(60*60):.3f}h"
                if self.distributed_run:
                    log_str += f" | rank: {self.rank}"
                print(log_str)

                interrupt = interrupt_condition()
                if interrupt:
                    interrupt_action()

            if epoch % self.epochs_per_checkpoint == 0:
                self.save_checkpoint(
                    f"{self.checkpoint_name}_{epoch}",
                    model=model,
                    optimizer=optimizer,
                    iteration=epoch,
                    learning_rate=self.learning_rate,
                    global_step=self.global_step,
                )
                save_function(epoch)

            # NOTE(zach): Validation is currently broken. Comment out to fix
            # training in master.
            # if self.is_validate:
            #     self.validate(
            #         model=model,
            #         val_set=val_set,
            #         collate_fn=collate_fn,
            #         criterion=criterion,
            #     )

            if self.debug:
                self.loss.append(reduced_loss)
                continue

    def validate(self, **kwargs):
        val_start_time = time.perf_counter()
        model = kwargs["model"]
        val_set = kwargs["val_set"]
        collate_fn = kwargs["collate_fn"]
        criterion = kwargs["criterion"]
        sampler = DistributedSampler(val_set) if self.distributed_run else None
        (
            total_loss,
            total_mel_loss,
            total_gate_loss,
            total_mel_loss_val,
            total_gate_loss_val,
        ) = (0, 0, 0, 0, 0)
        model.eval()
        speakers_val = []
        total_mel_loss_val = []
        total_gate_loss_val = []
        with torch.no_grad():
            val_loader = DataLoader(
                val_set,
                sampler=sampler,
                shuffle=False,
                batch_size=self.batch_size,
                collate_fn=collate_fn,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            # TODO (Sam): train loop should be in base trainer.
            for step_counter, batch in enumerate(val_loader):

                # TODO (Sam): Could call subsets directly in function arguments since model_input is only reused in logging.
                model_input = batch.subset(
                    [
                        "text_int_padded",
                        "input_lengths",
                        "speaker_ids",
                        "gst",
                        "mel_padded",
                        "output_lengths",
                        "audio_encoding",
                    ]
                )
                model_output = model(
                    input_text=model_input["text_int_padded"],
                    input_lengths=model_input["input_lengths"],
                    speaker_ids=model_input["speaker_ids"],
                    embedded_gst=model_input["gst"],
                    targets=model_input["mel_padded"],
                    output_lengths=model_input["output_lengths"],
                    audio_encoding=model_input["audio_encoding"],
                )
                target = batch.subset(["gate_target", "mel_padded"])
                mel_loss, gate_loss, mel_loss_batch, gate_loss_batch = criterion(
                    model_output=model_output,
                    target=target,
                )
                if self.distributed_run:
                    reduced_mel_loss = reduce_tensor(mel_loss, self.world_size).item()
                    reduced_gate_loss = reduce_tensor(gate_loss, self.world_size).item()
                    reduced_val_loss = reduced_mel_loss + reduced_gate_loss
                    reduced_gate_loss_val = reduce_tensor(
                        gate_loss_batch, self.world_size
                    )
                    reduced_mel_loss_val = reduce_tensor(
                        mel_loss_batch, self.world_size
                    )

                else:
                    reduced_mel_loss = mel_loss.item()
                    reduced_gate_loss = gate_loss.item()
                    reduced_mel_loss_val = mel_loss_batch.detach()
                    reduced_gate_loss_val = gate_loss_batch.detach()

                total_mel_loss_val.append(reduced_mel_loss_val)
                total_gate_loss_val.append(reduced_gate_loss_val)
                reduced_val_loss = reduced_mel_loss + reduced_gate_loss
                total_mel_loss += reduced_mel_loss
                total_gate_loss += reduced_gate_loss
                total_loss += reduced_val_loss

            total_steps = step_counter + 1
            mean_mel_loss = total_mel_loss / total_steps
            mean_gate_loss = total_gate_loss / total_steps
            mean_loss = total_loss / total_steps
            total_mel_loss_val = torch.hstack(total_mel_loss_val)
            total_gate_loss_val = torch.hstack(total_gate_loss_val)
            # TODO (Sam): make speaker_ids None not 0 when has_speaker_embedding = False.
            speakers_val.append(batch["speaker_ids"])
            speakers_val = torch.hstack(speakers_val)
            self.log_validation(
                X=model_input,
                y_pred=model_output,
                y=target,
                mean_loss=mean_loss,
                mean_mel_loss=mean_mel_loss,
                mean_gate_loss=mean_gate_loss,
                mel_loss_val=total_mel_loss_val,
                gate_loss_val=total_gate_loss_val,
                speakers_val=speakers_val,
            )

        model.train()

        val_log_str = f"Validation loss: {mean_loss:.3f} | mel: {mel_loss:.3f} | gate: {mean_gate_loss:.3f} | t: {time.perf_counter() - val_start_time:.3f}s"
        print(val_log_str)

    @property
    def val_dataset_args(self):

        args = dict(**self.training_dataset_args)
        args["audiopaths_and_text"] = self.val_audiopaths_and_text
        return args

    @property
    def training_dataset_args(self):
        return {
            "audiopaths_and_text": self.training_audiopaths_and_text,
            # Text parameters
            "return_texts": self.return_texts,
            "text_cleaners": self.text_cleaners,
            "symbol_set": self.symbol_set,
            "p_arpabet": self.p_arpabet,
            # Audio parameters
            "return_mels": self.return_mels,
            "n_mel_channels": self.n_mel_channels,
            "sampling_rate": self.sampling_rate,
            "mel_fmin": self.mel_fmin,
            "mel_fmax": self.mel_fmax,
            "filter_length": self.filter_length,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "max_wav_value": self.max_wav_value,
            # Speaker embedding parameters
            "audio_encoder_forward": self.audio_encoder_forward,
            "speaker_embeddings": self.speaker_embeddings,
            # F0 parameters
            "return_f0": self.with_f0,
            "load_f0": self.load_f0,
            "compute_f0": self.compute_f0,
            # GST parameters
            "return_gst": self.with_gst,
            "load_gst": self.load_gst,
            "compute_gst": self.compute_gst,
            "get_gst": self.get_gst,
        }

    @property
    def collate_args(self):
        return {
            "return_f0s": self.with_f0,
            "return_gsts": self.with_gst,
            "return_mels": True,
            "return_text_sequences": True,
            "return_audio_encodings": self.with_audio_encoding,
            "cudnn_enabled": self.cudnn_enabled,
            "n_frames_per_step": self.n_frames_per_step,
        }


config = TRAINER_DEFAULTS.values()
config.update(TACOTRON2_DEFAULTS.values())
config.update({"n_frames_per_step_initial": 1})
DEFAULTS = HParams(**config)
