import torch
import pandas as pd
from tqdm import tqdm
import torchaudio
import numpy as np
from speechbrain.pretrained import EncoderClassifier
from einops import rearrange

# TODO (Sam): move this to a proper model.
def get_speaker_encoding_for_dataset(
    speaker_encoding_path, filelist_path, embedding_dim, speaker_encoder_path
):
    classifier = EncoderClassifier.from_hparams(source=speaker_encoder_path)

    audio_file_paths = pd.read_csv(filelist_path, header=None, sep="|")[0]
    embedding_dim = 192
    n_files = len(audio_file_paths)
    embeddings = torch.zeros((n_files, embedding_dim))
    for i, audio_file_path in tqdm(enumerate(audio_file_paths)):
        signal, fs = torchaudio.load(audio_file_path)
        signal = signal / (np.abs(signal).max() * 2)  # NOTE (Sam): just must be < 1.
        embeddings[i] = classifier.encode_batch(signal)
    # NOTE (Sam): hack - only works for single speaker.
    speaker_encoding = rearrange(embeddings.mean(axis=0), "s -> 1 s")
    torch.save(speaker_encoding, speaker_encoding_path)


# # NOTE (Sam): maybe rename "CategoricalEncoder"
# # TODO (Sam): this should work without audio encodings as well -> need centroid computation code.
# class SpeakerEncoder(torch.nn.Module):
#     def __init__(self, speaker_ids, audio_encodings):
#         self.speaker_ids = speaker_ids
#         self.one_hot_speakers = torch.embedding(speaker_ids)  # TODO (Sam): normalize
#         self.audio_encodings = audio_encodings
#         # TODO (Sam): project into good region rather than just averaging
#         # TODO (Sam): this needs to account for samples per class
#         self.speaker_encodings = torch.einsum(
#             "is,ie->se", self.one_hot_speakers, audio_encodings
#         )

#     def forward(self, speaker_ids):

#         return self.speaker_encodings[speaker_ids]
