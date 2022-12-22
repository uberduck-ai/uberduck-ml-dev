import torch

# NOTE (Sam): maybe rename "CategoricalEncoder"
# TODO (Sam): this should work without audio encodings as well
class SpeakerEncoder(torch.nn.Module):
    def __init__(self, speaker_ids, audio_encodings):
        self.speaker_ids = speaker_ids
        self.one_hot_speakers = torch.embedding(speaker_ids)  # TODO (Sam): normalize
        self.audio_encodings = audio_encodings
        # TODO (Sam): project into good region rather than just averaging
        self.speaker_encodings = torch.einsum(
            "is,ie->se", self.one_hot_speakers, audio_encodings
        )

    def forward(self, speaker_ids):

        return self.speaker_encodings[speaker_ids]
