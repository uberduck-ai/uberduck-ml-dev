""" adapted from https://github.com/keithito/tacotron """

import re
import numpy as np
from . import cleaners
from .cleaners import Cleaner
from .symbols import get_symbols
from .grapheme_dictionary import Grapheme2PhonemeDictionary


#########
# REGEX #
#########

# Regular expression matching text enclosed in curly braces for encoding
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")

# Regular expression matching words and not words
_words_re = re.compile(
    r"([a-zA-ZÀ-ž]+['][a-zA-ZÀ-ž]+|[a-zA-ZÀ-ž]+)|([{][^}]+[}]|[^a-zA-ZÀ-ž{}]+)"
)


def lines_to_list(filename):
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


class TextProcessing(object):
    def __init__(
        self,
        symbol_set,
        cleaner_name,
        heteronyms_path,
        phoneme_dict_path,
        p_phoneme,
        handle_phoneme,
        handle_phoneme_ambiguous,
        prepend_space_to_text=False,
        append_space_to_text=False,
        add_bos_eos_to_text=False,
        encoding="latin-1",
    ):
        if heteronyms_path is not None and heteronyms_path != "":
            self.heteronyms = set(lines_to_list(heteronyms_path))
        else:
            self.heteronyms = []
        # phoneme dict
        self.phonemedict = Grapheme2PhonemeDictionary(
            phoneme_dict_path, encoding=encoding
        )
        self.p_phoneme = p_phoneme
        self.handle_phoneme = handle_phoneme
        self.handle_phoneme_ambiguous = handle_phoneme_ambiguous

        self.symbols = get_symbols(symbol_set)
        self.cleaner_names = cleaner_name
        self.cleaner = Cleaner(cleaner_name, self.phonemedict)

        self.prepend_space_to_text = prepend_space_to_text
        self.append_space_to_text = append_space_to_text
        self.add_bos_eos_to_text = add_bos_eos_to_text

        if add_bos_eos_to_text:
            self.symbols.append("<bos>")
            self.symbols.append("<eos>")

        # Mappings from symbol to numeric ID and vice versa:
        self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self.id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def text_to_sequence(self, text):
        sequence = []

        # Check for curly braces and treat their contents as phoneme:
        while len(text):
            m = _curly_re.match(text)
            if not m:
                sequence += self.symbols_to_sequence(text)
                break
            sequence += self.symbols_to_sequence(m.group(1))
            sequence += self.phoneme_to_sequence(m.group(2))
            text = m.group(3)

        return sequence

    def sequence_to_text(self, sequence):
        result = ""
        for symbol_id in sequence:
            if symbol_id in self.id_to_symbol:
                s = self.id_to_symbol[symbol_id]
                # Enclose phoneme back in curly braces:
                if len(s) > 1 and s[0] == "@":
                    s = "{%s}" % s[1:]
                result += s
        return result.replace("}{", " ")

    def clean_text(self, text):
        text = self.cleaner(text)
        return text

    def symbols_to_sequence(self, symbols):
        return [self.symbol_to_id[s] for s in symbols if s in self.symbol_to_id]

    def phoneme_to_sequence(self, text):
        return self.symbols_to_sequence(["@" + s for s in text.split()])

    def get_phoneme(self, word):
        phoneme_suffix = ""

        if word.lower() in self.heteronyms:
            return word

        if len(word) > 2 and word.endswith("'s"):
            phoneme = self.phonemedict.lookup(word)
            if phoneme is None:
                phoneme = self.phonemedict.lookup(word[:-2])
                phoneme_suffix = "" if phoneme is None else " Z"

        elif len(word) > 1 and word.endswith("s"):
            phoneme = self.phonemedict.lookup(word)
            if phoneme is None:
                phoneme = self.phonemedict.lookup(word[:-1])
                phoneme_suffix = "" if phoneme is None else " Z"
        else:
            phoneme = self.phonemedict.lookup(word)

        if phoneme is None:
            return word

        if len(phoneme) > 1:
            if self.handle_phoneme_ambiguous == "first":
                phoneme = phoneme[0]
            elif self.handle_phoneme_ambiguous == "random":
                phoneme = np.random.choice(phoneme)
            elif self.handle_phoneme_ambiguous == "ignore":
                return word
        else:
            phoneme = phoneme[0]

        phoneme = "{" + phoneme + phoneme_suffix + "}"

        return phoneme

    def encode_text(self, text, return_all=False):
        text_clean = self.clean_text(text)
        text = text_clean

        text_phoneme = ""
        if self.p_phoneme > 0:
            text_phoneme = self.convert_to_phoneme(text)
            text = text_phoneme

        text_encoded = self.text_to_sequence(text)

        if self.prepend_space_to_text:
            text_encoded.insert(0, self.symbol_to_id[" "])

        if self.append_space_to_text:
            text_encoded.append(self.symbol_to_id[" "])

        if self.add_bos_eos_to_text:
            text_encoded.insert(0, self.symbol_to_id["<bos>"])
            text_encoded.append(self.symbol_to_id["<eos>"])

        if return_all:
            return text_encoded, text_clean, text_phoneme

        return text_encoded

    def convert_to_phoneme(self, text):
        if self.handle_phoneme == "sentence":
            if np.random.uniform() < self.p_phoneme:
                words = _words_re.findall(text)
                text_phoneme = [
                    self.get_phoneme(word[0])
                    if (word[0] != "")
                    else re.sub(r"\s(\d)", r"\1", word[1].upper())
                    for word in words
                ]
                text_phoneme = "".join(text_phoneme)
                text = text_phoneme
        elif self.handle_phoneme == "word":
            words = _words_re.findall(text)
            text_phoneme = [
                re.sub(r"\s(\d)", r"\1", word[1].upper())
                if word[0] == ""
                else (
                    self.get_phoneme(word[0])
                    if np.random.uniform() < self.p_phoneme
                    else word[0]
                )
                for word in words
            ]
            text_phoneme = "".join(text_phoneme)
            text = text_phoneme
        elif self.handle_phoneme != "":
            raise Exception(
                "{} handle_phoneme is not supported".format(self.handle_phoneme)
            )
        return text
