__all__ = [
    "word_frequencies",
    "create_wordcloud",
    "count_frequency",
    "pace_character",
    "pace_phoneme",
    "get_sample_format",
    "AbsoluteMetrics",
]

from typing import List, Any, Dict, Union, Optional
from collections import Counter
import os

import librosa
import numpy as np
from pydub.utils import mediainfo_json
from wordfreq import word_frequency

from ..text.utils import text_to_sequence

# NOTE (Sam): this file could be refactored so that it doesn't contain both speechmetrics and wordfreqencies - very different types of statistics.


def word_frequencies(text: str, language: str = "en") -> List[float]:
    """
    Calculate the frequency [0-1] which the words appear in the english language
    """
    freqs = []
    for word in text.split():
        freqs.append(word_frequency(word, language))
    return freqs


def count_frequency(arr: List[Any]) -> Dict[Any, int]:
    """
    Calculates the frequency that a value appears in a list
    """
    return dict(Counter(arr).most_common())


def pace_character(
    text: str, audio: Union[str, np.ndarray], sr: Optional[int] = None
) -> float:
    """
    Calculates the number of characters in the text per second of the audio file. Audio can be a file path or an np array.
    """
    if isinstance(audio, str):
        audio, sr = librosa.load(audio, sr=None)
    else:
        assert sr, "Sampling rate must be provided if audio is np array"

    return len(text) / librosa.get_duration(audio, sr=sr)


def pace_phoneme(
    text: str, audio: Union[str, np.ndarray], sr: Optional[int] = None
) -> float:
    """
    Calculates the number of phonemes in the text per second of the audio. Audio can be a file path or an np array.
    """
    if isinstance(audio, str):
        audio, sr = librosa.load(audio, sr=None)
    else:
        assert sr, "Sampling rate must be provided if audio is np array"

    arpabet_seq = text_to_sequence(text, ["english_cleaners"], p_arpabet=1.0)
    return len(arpabet_seq) / librosa.get_duration(audio, sr=sr)


def get_sample_format(wav_file: str):
    """
    Get sample format of the .wav file: https://trac.ffmpeg.org/wiki/audio%20types
    """
    filename, file_extension = os.path.splitext(wav_file)
    assert file_extension == ".wav", ".wav file must be supplied"

    info = mediainfo_json(wav_file)
    audio_streams = [x for x in info["streams"] if x["codec_type"] == "audio"]
    return audio_streams[0].get("sample_fmt")


class AbsoluteMetrics:
    """This class loads and calculates the absolute metrics, MOSNet and SRMR"""

    def __init__(self, window_length: Optional[int] = None):
        # NOTE(zach): There are some problems installing speechmetrics via pip and it's not critical, so import inline to avoid issues in CI.
        import speechmetrics

        self.metrics = speechmetrics.load("absolute", window_length)

    def __call__(self, wav_file: str) -> Dict[str, float]:
        """
        Returns a Dict[str,float] with keys "mosnet" and "srmr"
        """
        filename, file_extension = os.path.splitext(wav_file)
        assert file_extension == ".wav", ".wav file must be supplied"

        return self.metrics(wav_file)
