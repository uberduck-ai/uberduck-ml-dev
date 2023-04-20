# NOTE (Sam): synthesize with other methods

""" adapted from https://github.com/keithito/tacotron """

import re

_alt_re = re.compile(r"\([0-9]+\)")


class Grapheme2PhonemeDictionary:
    """Thin wrapper around g2p data."""

    def __init__(self, file_or_path, keep_ambiguous=True, encoding="latin-1"):
        with open(file_or_path, encoding=encoding) as f:
            entries = _parse_g2p(f)
        if not keep_ambiguous:
            entries = {word: pron for word, pron in entries.items() if len(pron) == 1}
        self._entries = entries

    def __len__(self):
        return len(self._entries)

    def lookup(self, word):
        """Returns list of pronunciations of the given word."""
        return self._entries.get(word.upper())


def _parse_g2p(file):
    g2p = {}
    for line in file:
        if len(line) and (line[0] >= "A" and line[0] <= "Z" or line[0] == "'"):
            parts = line.split("  ")
            word = re.sub(_alt_re, "", parts[0])
            pronunciation = parts[1].strip()
            if word in g2p:
                g2p[word].append(pronunciation)
            else:
                g2p[word] = [pronunciation]
    return g2p
