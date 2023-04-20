from uberduck_ml_dev.text.symbols import arpabet_to_sequence, symbols_to_sequence


class TestSymbols:
    def test_arpabet_to_sequence(self):
        # NOTE: arpabet_to_sequence does not properly handle whitespace, it should take single words only.
        assert (
            len(
                arpabet_to_sequence(
                    "{ S IY } { EH M } { Y UW } { D IH K SH AH N EH R IY }"
                )
            )
            == 15
        )
        assert arpabet_to_sequence("{ S IY }") == [168, 148]
        # But symbols_to_sequence hanldes whitespace

    def test_symbols_to_sequence(self):
        assert len(symbols_to_sequence("C M U Dictionary")) == 16
