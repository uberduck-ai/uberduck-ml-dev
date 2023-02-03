import numpy as np


def oversample(filepaths_text_sid, sid_to_weight):
    assert all([isinstance(sid, str) for sid in sid_to_weight.keys()])
    output = []
    for fts in filepaths_text_sid:
        sid = fts[2]
        for _ in range(sid_to_weight.get(sid, 1)):
            output.append(fts)
    return output


def _orig_to_dense_speaker_id(speaker_ids):
    speaker_ids = np.asarray(list(set(speaker_ids)), dtype=str)
    id_order = np.argsort(np.asarray(speaker_ids, dtype=int))
    output = {
        orig: idx for orig, idx in zip(speaker_ids[id_order], range(len(speaker_ids)))
    }
    return output
