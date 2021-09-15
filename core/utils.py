import pandas as pd
import numpy as np


def add_speaker_ids(source_file, speaker_ids=None, speaker_key=None):

    data = pd.read_csv(
        source_file + ".txt", sep="|", header=None, error_bad_lines=False
    )

    if speaker_ids == None:
        speaker_ids = np.asarray(
            np.ones(data.shape[0], dtype=int) * speaker_key, dtype=int
        )

    for i in range(data.shape[0]):
        data[0][i] = "/Users/samsonkoelle/Downloads/eminem_14/" + data[0][i]
    data[2] = speaker_ids
    data.to_csv(source_file + "_multispeaker.txt", sep="|", header=None, index=False)


# def add_speaker_id(source_file, speaker_ids=None, speaker_key=None):

#     data = pd.read_csv(
#         source_file + ".txt", sep="|", header=None, error_bad_lines=False
#     )

#     if speaker_ids == None:
#         speaker_ids = np.asarray(
#             np.ones(data.shape[0], dtype=int) * speaker_key, dtype=int
#         )

#     for i in range(data.shape[0]):
#         data[0][i] = "/Users/samsonkoelle/Downloads/eminem_14/" + data[0][i]
#     data[2] = speaker_ids
#     data.to_csv(source_file + "_multispeaker.txt", sep="|", header=None, index=False)


add_speaker_id(
    "/Users/samsonkoelle/Downloads/eminem_14/train", speaker_ids=None, speaker_key=0
)
add_speaker_id(
    "/Users/samsonkoelle/Downloads/eminem_14/val", speaker_ids=None, speaker_key=0
)
