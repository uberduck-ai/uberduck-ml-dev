__all__ = [
    "symbols_portuguese",
    "PORTUGUESE_SYMBOLS",
    "symbols_polish",
    "POLISH_SYMBOLS",
    "symbols_dutch",
    "DUTCH_SYMBOLS",
    "symbols_spanish",
    "SPANISH_SYMBOLS",
    "symbols_norwegian",
    "NORWEGIAN_SYMBOLS",
    "symbols_russian",
    "RUSSIAN_SYMBOLS",
    "symbols_ukrainian",
    "UKRAINIAN_SYMBOLS",
    "symbols",
    "symbols_nvidia_taco2",
    "symbols_with_ipa",
    "grad_tts_symbols",
    "DEFAULT_SYMBOLS",
    "IPA_SYMBOLS",
    "NVIDIA_TACO2_SYMBOLS",
    "GRAD_TTS_SYMBOLS",
    "SYMBOL_SETS",
    "symbols_to_sequence",
    "arpabet_to_sequence",
    "should_keep_symbol",
    "symbol_to_id",
    "id_to_symbol",
    "curly_re",
    "words_re",
]


""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.
The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from . import cmudict

_pad = "_"
_punctuation_nvidia_taco2 = "!'(),.:;? "
_punctuation = "!'\",.:;? "
_math = "#%&*+-/[]()"
_special = "@©°½—₩€$"
_special_nvidia_taco2 = "-"
_accented = "áçéêëñöøćž"
_numbers = "0123456789"

_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"


# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as
# uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]


# Language-specific symbol sets:

_portuguese = "áàãâéèêíìîóòõôúùûçÁÀÃÂÉÈÊÍÌÎÓÒÕÔÚÙÛÇabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

symbols_portuguese = (
    [_pad]
    + list(_special_nvidia_taco2)
    + list(_punctuation_nvidia_taco2)
    + list(_portuguese)
    + _arpabet
)

PORTUGUESE_SYMBOLS = "portuguese"

##

_polish = "AĄBCĆDEĘFGHIJKLŁMNŃOÓPRSŚTUWYZŹŻaąbcćdeęfghijklłmnńoóprsśtuwyzźż"
_punctuation_polish = "!,.? "

symbols_polish = (
    [_pad]
    + list(_special_nvidia_taco2)
    + list(_punctuation_polish)
    + list(_polish)
    + _arpabet
)

POLISH_SYMBOLS = "polish"

##

_dutch = "éèêëíìîüÉÈÊËÍÌÎÜabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

symbols_dutch = (
    [_pad]
    + list(_special_nvidia_taco2)
    + list(_punctuation_nvidia_taco2)
    + list(_dutch)
    + _arpabet
)

DUTCH_SYMBOLS = "dutch"

##

_spanish = "AÁBCDEÉFGHIÍJKLMNÑOÓPQRSTUÚÜVWXYZaábcdeéfghiíjklmnñoópqrstuúüvwxyz"
_punctuation_spanish = "!¡'(),.:;?¿ "

symbols_spanish = (
    [_pad]
    + list(_special_nvidia_taco2)
    + list(_punctuation_spanish)
    + list(_spanish)
    + _arpabet
)

SPANISH_SYMBOLS = "spanish"

##

_norwegian = "ABCDEFGHIJKLMNOPQRSTUVWXYZÆØÅabcdefghijklmnopqrstuvwxyzæøå"

symbols_norwegian = (
    [_pad]
    + list(_special_nvidia_taco2)
    + list(_punctuation_nvidia_taco2)
    + list(_norwegian)
    + _arpabet
)

NORWEGIAN_SYMBOLS = "norwegian"

##

_russian = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюя"

symbols_russian = (
    [_pad]
    + list(_special_nvidia_taco2)
    + list(_punctuation_nvidia_taco2)
    + list(_russian)
    + _arpabet
)

RUSSIAN_SYMBOLS = "russian"

##

_ukrainian = "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯабвгґдеєжзиіїйклмнопрстуфхцчшщьюя"

symbols_ukrainian = (
    [_pad]
    + list(_special_nvidia_taco2)
    + list(_punctuation_nvidia_taco2)
    + list(_ukrainian)
    + _arpabet
)

UKRAINIAN_SYMBOLS = "ukrainian"


# Export all symbols:
symbols = (
    list(_pad + _punctuation + _math + _special + _accented + _numbers + _letters)
    + _arpabet
)

symbols_nvidia_taco2 = (
    [_pad]
    + list(_special_nvidia_taco2)
    + list(_punctuation_nvidia_taco2)
    + list(_letters)
    + _arpabet
)

symbols_with_ipa = symbols + list(_letters_ipa)
grad_tts_symbols = list(_pad + "-" + "!'(),.:;? " + _letters) + _arpabet

DEFAULT_SYMBOLS = "default"
IPA_SYMBOLS = "ipa"
NVIDIA_TACO2_SYMBOLS = "nvidia_taco2"
GRAD_TTS_SYMBOLS = "gradtts"

SYMBOL_SETS = {
    DEFAULT_SYMBOLS: symbols,
    IPA_SYMBOLS: symbols_with_ipa,
    NVIDIA_TACO2_SYMBOLS: symbols_nvidia_taco2,
    GRAD_TTS_SYMBOLS: grad_tts_symbols,
    PORTUGUESE_SYMBOLS: symbols_portuguese,
    POLISH_SYMBOLS: symbols_polish,
    DUTCH_SYMBOLS: symbols_dutch,
    SPANISH_SYMBOLS: symbols_spanish,
    NORWEGIAN_SYMBOLS: symbols_norwegian,
    RUSSIAN_SYMBOLS: symbols_russian,
    UKRAINIAN_SYMBOLS: symbols_ukrainian,
}


import re

symbol_to_id = {
    DEFAULT_SYMBOLS: {s: i for i, s in enumerate(SYMBOL_SETS[DEFAULT_SYMBOLS])},
    IPA_SYMBOLS: {s: i for i, s in enumerate(SYMBOL_SETS[IPA_SYMBOLS])},
    NVIDIA_TACO2_SYMBOLS: {
        s: i for i, s in enumerate(SYMBOL_SETS[NVIDIA_TACO2_SYMBOLS])
    },
    GRAD_TTS_SYMBOLS: {s: i for i, s in enumerate(SYMBOL_SETS[GRAD_TTS_SYMBOLS])},
    PORTUGUESE_SYMBOLS: {s: i for i, s in enumerate(SYMBOL_SETS[PORTUGUESE_SYMBOLS])},
    POLISH_SYMBOLS: {s: i for i, s in enumerate(SYMBOL_SETS[POLISH_SYMBOLS])},
    DUTCH_SYMBOLS: {s: i for i, s in enumerate(SYMBOL_SETS[DUTCH_SYMBOLS])},
    SPANISH_SYMBOLS: {s: i for i, s in enumerate(SYMBOL_SETS[SPANISH_SYMBOLS])},
    NORWEGIAN_SYMBOLS: {s: i for i, s in enumerate(SYMBOL_SETS[NORWEGIAN_SYMBOLS])},
    RUSSIAN_SYMBOLS: {s: i for i, s in enumerate(SYMBOL_SETS[RUSSIAN_SYMBOLS])},
    UKRAINIAN_SYMBOLS: {s: i for i, s in enumerate(SYMBOL_SETS[UKRAINIAN_SYMBOLS])},
}
id_to_symbol = {
    DEFAULT_SYMBOLS: {i: s for i, s in enumerate(SYMBOL_SETS[DEFAULT_SYMBOLS])},
    IPA_SYMBOLS: {i: s for i, s in enumerate(SYMBOL_SETS[IPA_SYMBOLS])},
    NVIDIA_TACO2_SYMBOLS: {
        i: s for i, s in enumerate(SYMBOL_SETS[NVIDIA_TACO2_SYMBOLS])
    },
    GRAD_TTS_SYMBOLS: {i: s for i, s in enumerate(SYMBOL_SETS[GRAD_TTS_SYMBOLS])},
    PORTUGUESE_SYMBOLS: {i: s for i, s in enumerate(SYMBOL_SETS[PORTUGUESE_SYMBOLS])},
    POLISH_SYMBOLS: {i: s for i, s in enumerate(SYMBOL_SETS[POLISH_SYMBOLS])},
    DUTCH_SYMBOLS: {i: s for i, s in enumerate(SYMBOL_SETS[DUTCH_SYMBOLS])},
    SPANISH_SYMBOLS: {i: s for i, s in enumerate(SYMBOL_SETS[SPANISH_SYMBOLS])},
    NORWEGIAN_SYMBOLS: {i: s for i, s in enumerate(SYMBOL_SETS[NORGWEGIAN_SYMBOLS])},
    RUSSIAN_SYMBOLS: {i: s for i, s in enumerate(SYMBOL_SETS[RUSSIAN_SYMBOLS])},
    UKRAINIAN_SYMBOLS: {i: s for i, s in enumerate(SYMBOL_SETS[UKRAINIAN_SYMBOLS])},
}

curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")
words_re = re.compile(
    r"([a-zA-ZÀ-ž\u0400-\u04FF]+['][a-zA-ZÀ-ž\u0400-\u04FF]{1,2}|[a-zA-ZÀ-ž\u0400-\u04FF]+)|([{][^}]+[}]|[^a-zA-ZÀ-ž\u0400-\u04FF{}]+)"
)


def symbols_to_sequence(symbols, symbol_set=DEFAULT_SYMBOLS, ignore_symbols=["_", "~"]):
    return [
        symbol_to_id[symbol_set][s]
        for s in symbols
        if should_keep_symbol(s, symbol_set, ignore_symbols)
    ]


def arpabet_to_sequence(text, symbol_set=DEFAULT_SYMBOLS):
    return symbols_to_sequence(["@" + s for s in text.split()], symbol_set=symbol_set)


def should_keep_symbol(s, symbol_set=DEFAULT_SYMBOLS, ignore_symbols=["_", "~"]):
    return s in symbol_to_id[symbol_set] and s not in ignore_symbols
