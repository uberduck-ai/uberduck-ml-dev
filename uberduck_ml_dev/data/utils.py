import os

import numpy as np
import torch
from scipy.stats import betabinom


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


def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling_factor=0.05):
    P = phoneme_count
    M = mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M + 1):
        a, b = scaling_factor * i, scaling_factor * (M + 1 - i)
        rv = betabinom(P - 1, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return torch.tensor(np.array(mel_text_probs))


def get_attention_prior(n_tokens, n_frames):
    filename = "{}_{}".format(n_tokens, n_frames)
    betabinom_cache_path = "betabinom_cache"
    if not os.path.exists(betabinom_cache_path):
        os.makedirs(betabinom_cache_path, exist_ok=False)
    prior_path = os.path.join(betabinom_cache_path, filename)
    prior_path += "_prior.pth"

    if os.path.exists(prior_path):
        attn_prior = torch.load(prior_path)
    else:
        attn_prior = beta_binomial_prior_distribution(
            n_tokens, n_frames, scaling_factor=1.0  # 0.05
        )
        torch.save(attn_prior, prior_path)

    return attn_prior


def energy_avg_normalize(x):
    use_scaled_energy = True
    if use_scaled_energy == True:
        x = (x + 20.0) / 20.0
    return x


def get_energy_average(mel):
    energy_avg = mel.mean(0)
    energy_avg = energy_avg_normalize(energy_avg)
    return energy_avg


# NOTE (Sam): looks like this was not used in successful training runs, but could be experimented with in the future (v interesting).
def f0_normalize(x, f0_min):
    # if self.use_log_f0:
    # mask = x >= f0_min
    # x[mask] = torch.log(x[mask])
    # x[~mask] = 0.0

    return x


def get_shuffle_indices(levels):
    levels = np.asarray(levels)
    levels_unique = np.unique(levels)
    output_indices = np.zeros(len(levels), dtype=int)
    for level in levels_unique:
        indices = np.where(levels == level)[0]
        new_indices = np.random.permutation(indices)
        output_indices[indices] = new_indices
    return output_indices
