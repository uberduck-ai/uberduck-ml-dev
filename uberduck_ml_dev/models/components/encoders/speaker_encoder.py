# NOTE (Sam): tried using speechbrain and had problems due to awful ruamel_yaml v ruamel.yaml issue.
# TODO (Sam): enable use of speechbrain encoder in the future.
# NOTE (Sam): I am leaving this for now just because its imported in certain locations.
def get_speaker_encoding_for_dataset(
    speaker_encoding_path, filelist_path, embedding_dim, speaker_encoder_path
):
    continue