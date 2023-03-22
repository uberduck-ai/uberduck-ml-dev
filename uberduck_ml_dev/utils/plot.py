__all__ = [
    "save_figure_to_numpy",
    "plot_tensor",
    "plot_spectrogram",
    "plot_attention",
    "plot_attention_phonemes",
    "plot_gate_outputs",
]


import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..text.symbols import id_to_symbol, DEFAULT_SYMBOLS


def save_figure_to_numpy(fig):
    """Save figure to a numpy array."""
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


def plot_tensor(tensor):
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_spectrogram(mel):
    figure = plt.figure()
    plt.xlabel("Spectrogram frame")
    plt.ylabel("Channel")
    plt.imshow(mel, aspect="auto", origin="lower", interpolation="none", cmap="inferno")
    figure.canvas.draw()
    return figure


def plot_attention(attention, encoder_length=None, decoder_length=None):
    figure = plt.figure()
    plt.xlabel("Decoder timestep")
    plt.ylabel("Encoder timestep")
    plt.imshow(
        attention.data.cpu().numpy(),
        aspect="auto",
        origin="lower",
        interpolation="none",
        cmap="inferno",
    )
    title_info = []
    if encoder_length is not None:
        title_info.append(f"Encoder_length: {encoder_length}")
    if decoder_length is not None:
        title_info.append(f"Decoder length: {decoder_length}")
    title = " ".join(title_info)
    plt.title(title)
    figure.canvas.draw()
    return figure


def plot_attention_phonemes(seq, attention, symbol_set=DEFAULT_SYMBOLS):
    figure = plt.figure(figsize=(15, 8))
    phonemes = []

    for token in seq.numpy():
        if token == len(id_to_symbol[symbol_set]):
            phonemes.append("~")
        else:
            phonemes.append(id_to_symbol[symbol_set][token][1:])

    xtick_locs = np.pad(
        np.cumsum(np.sum(attention.data.cpu().numpy(), axis=1)), (1, 0)
    ).astype(np.int16)[:-1]
    ytick_locs = np.arange(seq.shape[-1])
    plt.yticks(ytick_locs, phonemes)
    plt.xticks(xtick_locs, xtick_locs)

    plt.imshow(
        attention.data.cpu().numpy(),
        aspect="auto",
        origin="lower",
        interpolation="none",
        cmap="Greys",
    )

    i = 0
    for phon, y in zip(phonemes, ytick_locs):
        if phon == "~":
            continue
        if i == 4:
            plt.axhline(y=y, color="k")
        if i == 3:
            plt.axhline(y=y, color="r")
        if i == 2:
            plt.axhline(y=y, color="g")
        if i == 1:
            plt.axhline(y=y, color="b")
        if i == 0:
            plt.axhline(y=y, color="m")
        i += 1
        i = i % 5

    plt.grid(axis="x")
    plt.title("Phoneme Alignment")
    plt.xlabel("Time (mel frames)")
    plt.ylabel("Phonemes")

    return figure


def plot_gate_outputs(gate_targets=None, gate_outputs=None):
    figure = plt.figure()
    plt.xlabel("Frames")
    plt.ylabel("Gate state")
    ax = figure.add_axes([0, 0, 1, 1])
    if gate_targets is not None:
        ax.scatter(
            range(gate_targets.size(0)),
            gate_targets,
            alpha=0.5,
            color="green",
            marker="+",
            s=1,
            label="target",
        )
    if gate_outputs is not None:
        ax.scatter(
            range(gate_outputs.size(0)),
            gate_outputs,
            alpha=0.5,
            color="red",
            marker=".",
            s=1,
            label="predicted",
        )
    figure.canvas.draw()
    return figure




def plot_alignment_to_numpy(alignment, title='', info=None, phoneme_seq=None,
                            vmin=None, vmax=None):
    if phoneme_seq:
        fig, ax = plt.subplots(figsize=(15, 10))
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    if phoneme_seq != None:
        # for debugging of phonemes and durs in maps. Not used by def in training code
        ax.set_yticks(np.arange(len(phoneme_seq)))
        ax.set_yticklabels(phoneme_seq)
        ax.hlines(np.arange(len(phoneme_seq)), xmin=0.0, xmax=max(ax.get_xticks()))

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

