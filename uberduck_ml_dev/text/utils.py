__all__ = [
    "normalize_numbers",
    "expand_abbreviations",
    "expand_numbers",
    "lowercase",
    "collapse_whitespace",
    "convert_to_ascii",
    "convert_to_arpabet",
    "basic_cleaners",
    "turkish_cleaners",
    "transliteration_cleaners",
    "english_cleaners",
    "english_cleaners_phonemizer",
    "batch_english_cleaners_phonemizer",
    "g2p",
    "batch_clean_text",
    "clean_text",
    "english_to_arpabet",
    "cleaned_text_to_sequence",
    "text_to_sequence",
    "sequence_to_text",
    "BATCH_CLEANERS",
    "CLEANERS",
    "text_to_sequence_for_editts",
    "random_utterance",
    "UTTERANCES",
]


""" from https://github.com/keithito/tacotron """

"""
Cleaners are transformations that run over the input text at both training and eval time.
Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""


import re
from typing import List

from g2p_en import G2p
from phonemizer import phonemize
from unidecode import unidecode
import torch

from .symbols import curly_re, words_re, symbols_to_sequence

g2p = G2p()

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]

import inflect
import re


_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_decimal_point(m):
    return m.group(1).replace(".", " point ")


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(
                num, andword="", zero="oh", group=2
            ).replace(", ", " ")
    else:
        return _inflect.number_to_words(num, andword="")


def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def convert_to_arpabet(text, overrides=None):
    return " ".join(
        [
            f"{{ {s.strip()} }}" if s.strip() not in ",." else s.strip()
            for s in " ".join(g2p(text, overrides=overrides)).split("  ")
        ]
    )


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def turkish_cleaners(text):
    text = text.replace("İ", "i").replace("I", "ı")
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    """Pipeline for English text, including number and abbreviation expansion."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners_phonemizer(text):
    """Pipeline for English text to phonemization, including number and abbreviation expansion."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = phonemize(
        text,
        language="en-us",
        backend="espeak",
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
    )
    text = collapse_whitespace(text)
    return text


def batch_english_cleaners_phonemizer(text: List[str]):
    batch = []
    for t in text:
        t = convert_to_ascii(t)
        t = lowercase(t)
        t = expand_numbers(t)
        t = expand_abbreviations(t)
        batch.append(t)
    batch = phonemize(
        batch,
        language="en-us",
        backend="espeak",
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
    )
    batch = [collapse_whitespace(t) for t in batch]
    return batch


import random

from .symbols import (
    DEFAULT_SYMBOLS,
    IPA_SYMBOLS,
    NVIDIA_TACO2_SYMBOLS,
    GRAD_TTS_SYMBOLS,
    id_to_symbol,
    symbols_to_sequence,
    arpabet_to_sequence,
)

BATCH_CLEANERS = {
    "english_cleaners_phonemizer": batch_english_cleaners_phonemizer,
}

CLEANERS = {
    "english_cleaners": english_cleaners,
    "english_cleaners_phonemizer": english_cleaners_phonemizer,
    "basic_cleaners": basic_cleaners,
    "turkish_cleaners": turkish_cleaners,
    "transliteration_cleaners": transliteration_cleaners,
}


def batch_clean_text(text: List[str], cleaner_names):
    for name in cleaner_names:
        cleaner = BATCH_CLEANERS[name]
        text = cleaner(text)
    return text


def clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = CLEANERS[name]
        text = cleaner(text)
    return text


def english_to_arpabet(english_text):
    arpabet_symbols = g2p(english_text)


def cleaned_text_to_sequence(cleaned_text, symbol_set):
    return symbols_to_sequence(cleaned_text, symbol_set=symbol_set, ignore_symbols=[])


def text_to_sequence(
    text,
    cleaner_names,
    p_arpabet=0.0,
    symbol_set=DEFAULT_SYMBOLS,
    arpabet_overrides=None,
):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = curly_re.match(text)
        if not m:
            cleaned = clean_text(text, cleaner_names)
            words_and_nonwords = words_re.findall(cleaned)
            cleaned_words = []
            for w, nw in words_and_nonwords:
                if w and random.random() < p_arpabet:
                    cleaned_words.append(
                        convert_to_arpabet(w, overrides=arpabet_overrides)
                    )
                elif w:
                    cleaned_words.append(w)
                else:
                    cleaned_words.append(nw)
            for word in cleaned_words:
                if word.startswith("{"):
                    sequence += arpabet_to_sequence(word, symbol_set)
                else:
                    sequence += symbols_to_sequence(word, symbol_set)
            break
        cleaned = clean_text(m.group(1), cleaner_names)
        sequence += text_to_sequence(cleaned, cleaner_names, p_arpabet, symbol_set)
        sequence += arpabet_to_sequence(m.group(2), symbol_set)
        text = m.group(3)

    return sequence


def pad_sequences(batch):
    input_lengths = torch.LongTensor([len(x) for x in batch])
    max_input_len = input_lengths.max()

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i in range(len(batch)):
        text = batch[i]
        text_padded[i, : text.size(0)] = text

    return text_padded, input_lengths


def prepare_input_sequence(
    texts,
    cpu_run=False,
    arpabet=False,
    symbol_set=NVIDIA_TACO2_SYMBOLS,
    text_cleaner=["english_cleaners"],
):
    p_arpabet = float(arpabet)
    seqs = []
    for text in texts:
        seqs.append(
            torch.IntTensor(
                # NOTE (Sam): this adds a period to the end of every text.
                text_to_sequence(
                    text,
                    text_cleaner,
                    p_arpabet=p_arpabet,
                    symbol_set=symbol_set,
                )[:]
            )
        )
    text_padded, input_lengths = pad_sequences(seqs)
    if not cpu_run:
        text_padded = text_padded.cuda().long()
        input_lengths = input_lengths.cuda().long()
    else:
        text_padded = text_padded.long()
        input_lengths = input_lengths.long()

    return text_padded, input_lengths


def sequence_to_text(sequence, symbol_set=DEFAULT_SYMBOLS):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        if symbol_id in id_to_symbol[symbol_set]:
            s = id_to_symbol[symbol_set][symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == "@":
                s = "{%s}" % s[1:]
            result += s
    return result.replace("}{", " ")


def text_to_sequence_for_editts(text, cleaner_names, symbol_set=GRAD_TTS_SYMBOLS):
    sequence = []
    emphases = []
    final_emphases = []
    space = symbols_to_sequence(" ", symbol_set=symbol_set)
    cleaned = clean_text(text, cleaner_names)

    i = 0
    result = []
    emphasis_interval = []
    for w in cleaned.split(" "):
        if w == "|":
            emphasis_interval.append(i)
            if len(emphasis_interval) == 2:
                emphases.append(emphasis_interval)
                emphasis_interval = []
        else:
            i += 1
            result.append(convert_to_arpabet(w))

    cleaned = result
    emphasis_interval = []
    cnt = 0
    for i in range(len(cleaned)):
        t = cleaned[i]
        if cnt < len(emphases) and i == emphases[cnt][0]:
            emphasis_interval.append(len(sequence))

        if t.startswith("{"):
            sequence += arpabet_to_sequence(t[1:-1], symbol_set=symbol_set)
        else:
            sequence += symbols_to_sequence(t, symbol_set=symbol_set)

        if cnt < len(emphases) and i == emphases[cnt][1] - 1:
            emphasis_interval.append(len(sequence))
            final_emphases.append(emphasis_interval)
            emphasis_interval = []
            cnt += 1

        sequence += space

    # remove trailing space
    if sequence[-1] == space[0]:
        sequence = sequence[:-1]

    return sequence, final_emphases


import random

UTTERANCES = [
    "Stop posting about Among Us, I'm tired of seeing it!",
    "My friends on TikTok send me memes, on Discord it's fucking memes.",
    "I'd just like to interject for a moment.",
    "What you're referring to as Linux, is in fact, gnu slash Linux.",
    "Wow! That was intense! Woo I just flew in from the new ruins level and boy are my arms tired.",
    "Oh my god! They killed Kenny!",
    "It needs to be about, twenty percent cooler.",
    "Hey relax guy! I'm just your average joe! Take a rest!",
    "I'm not bad, I'm just drawn that way.",
    "Alright! we're here just sitting in the car. I want you to show me if you can get far.",
    "Isn't it nice to have a computer that will talk to you?",
    "This is where we hold them. This is where we fight!",
    "I'll have two number nines, a number nine large, a number six with extra dip.",
    "A number seven, two number forty fives, one with cheese, and a large soda.",
    "Can you tell me how to get to Sesame Street?",
    "You know what they say, all toasters toast toast.",
    "Don't turn me into a marketable plushie!",
    "I am speaking straight opinions, and that's all that matters.",
    "Excuse me sir, but it appears that a package has arrived in the mailbox as of recent.",
    "I'm going to order pizza, look at me, I'm on the phone, right now.",
    "I started calling and I am hungry to the bone.",
    "so while I wait, I start to sing the song of my people I know it since I was a baby.",
    "When I was a lad, I ate four dozen eggs every morning to help me get large.",
    "Now that I'm grown I eat five dozen eggs, so I'm roughly the size of a barge!",
    "There's no crying. There's no crying in baseball.",
    "Sphinx of black quartz, judge my vow.",
    "Go to the Winchester, have a pint, and wait for all of this to blow over.",
    "You should really stop pressing this button.",
    "Minecraft is honestly a block game.",
    "I like that song. Let it play.",
    "When a zebras in the zone, leave him alone!",
    "The FitnessGram Pacer Test is a multistage aerobic capacity test that progressively gets more difficult as it continues.",
    "The 20 meter pacer test will begin in 30 seconds.",
    "The running speed starts slowly, but gets faster each minute after you hear this signal. beep.",
    "A single lap should be completed each time you hear this sound. ding.",
    "Remember to run in a straight line, and run as long as possible.",
    "The second time you fail to complete a lap before the sound, your test is over.",
    "The test will begin on the word start. On your mark, get ready, start.",
    "Oh my gosh. Nemo's swimming out to sea!",
    "Go back. I want to be monkey!",
    "Whoops! You have to put the C D in your computer.",
    "Now the animators are gonna have to draw all this fire!",
    "The mitochondria is the powerhouse of the cell.",
    "Now that's something you don't see every day!",
    "You know, what can I say? I die hard.",
    "Gosh darn it Kris, where the heck are we?",
    "This is a test voice message.",
    "I swear the toilet was full of guacamole when I bought it!",
    "Did you ever hear the Tragedy of Darth Plagueis the wise?",
    "I thought not. It's not a story the Jedi would tell you, it's a sith legend.",
    "Darth Plagueis was a dark lord of the Sith, so powerful and so wise",
    "He could use the force to influence the midichlorians to create life.",
    "Never gonna give you up. Never gonna let you down.",
    "I am the Milkman. My milk is delicious.",
    "I'm just like my country. I'm young, scrappy, and hungry, and I am not throwing away my shot.",
    "I'm still a piece of garbage.",
    "Looks like you're the first one here! Use the people tab on your watch to invite your friends to join you!",
    "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this. But, in a larger sense, we can not dedicate—we can not consecrate—we can not hallow—this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us—that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion—that we here highly resolve that these dead shall not have died in vain—that this nation, under God, shall have a new birth of freedom—and that government of the people, by the people, for the people, shall not perish from the earth."
]


def random_utterance():
    return UTTERANCES[random.randint(0, len(UTTERANCES) - 1)]
