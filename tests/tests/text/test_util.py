from uberduck_ml_dev.text.utils import (
    cleaned_text_to_sequence,
    text_to_sequence,
    DEFAULT_SYMBOLS,
    sequence_to_text,
)


class TestTextUtils:
    def text_sequence_to_text(self):
        print(text_to_sequence("The pen is | blue.| ", ["english_cleaners"]))
        assert len(text_to_sequence("The pen is blue.", ["english_cleaners"])) == 16
        assert (
            len(text_to_sequence("The pen is {B L OW0}.", ["english_cleaners"])) == 15
        )
        assert (
            sequence_to_text(text_to_sequence("The pen is blue.", ["english_cleaners"]))
            == "the pen is blue."
        ), sequence_to_text(text_to_sequence("The pen is blue.", ["english_cleaners"]))
        assert (
            sequence_to_text(
                text_to_sequence("The pen is {B L OW0}.", ["english_cleaners"])
            )
            == "the pen is {B L OW0}."
        )
        assert (
            len(
                text_to_sequence(
                    "{N AA1 T} {B AE1 D} {B AA1 R T}, {N AA1 T} {B AE1 D} {AE1 T} {AO1 L}.",
                    ["english_cleaners"],
                )
            )
            == 28
        )

        assert (
            len(
                text_to_sequence(
                    "Not bad bart, not bad at all.", ["english_cleaners"], p_arpabet=1.0
                )
            )
            == 28
        )

    def test_text_to_sequence(self):
        assert cleaned_text_to_sequence(
            "Not bad bart, not bad at all", DEFAULT_SYMBOLS
        ) == [
            62,
            89,
            94,
            9,
            76,
            75,
            78,
            9,
            76,
            75,
            92,
            94,
            4,
            9,
            88,
            89,
            94,
            9,
            76,
            75,
            78,
            9,
            75,
            94,
            9,
            75,
            86,
            86,
        ]
        assert text_to_sequence(
            "Not bad bart, not bad at all", ["english_cleaners"], 0.0, DEFAULT_SYMBOLS
        ) == [
            88,
            89,
            94,
            9,
            76,
            75,
            78,
            9,
            76,
            75,
            92,
            94,
            4,
            9,
            88,
            89,
            94,
            9,
            76,
            75,
            78,
            9,
            75,
            94,
            9,
            75,
            86,
            86,
        ]
