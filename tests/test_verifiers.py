"""Unit tests for the verifiers (Eq. 4) and the running score (Section 3.4)."""
import pytest

from ugra.agent.score import running_score
from ugra.tools.verifiers import CharNGramVerifier, LayoutVerifier, LexiconVerifier


@pytest.mark.unit
class TestLexicon:
    VERIFIER = LexiconVerifier(["payment", "forty", "pounds", "the"])

    def test_full_hit(self):
        assert self.VERIFIER.score("the payment") == 1.0

    def test_partial(self):
        assert self.VERIFIER.score("the zzzqqq") == pytest.approx(0.5)

    def test_empty_hypothesis_is_zero(self):
        assert self.VERIFIER.score("") == 0.0

    def test_punctuation_stripped(self):
        assert self.VERIFIER.score("payment,") == 1.0


@pytest.mark.unit
class TestCharNGram:
    VERIFIER = CharNGramVerifier(["the parish clerk received the payment",
                                  "forty pounds were paid to the clerk"])

    def test_in_domain_scores_higher(self):
        assert self.VERIFIER.score("the clerk") > self.VERIFIER.score("zqxj wvk")

    def test_range(self):
        assert 0.0 <= self.VERIFIER.score("anything at all") <= 1.0

    def test_empty_hypothesis_is_zero(self):
        assert self.VERIFIER.score("") == 0.0


@pytest.mark.unit
class TestLayout:
    VERIFIER = LayoutVerifier()

    def test_matched_length(self):
        assert self.VERIFIER.score("abcdefghij", expected_len=10) == 1.0

    def test_symmetric_ratio(self):
        short = self.VERIFIER.score("abcde", expected_len=10)
        long = self.VERIFIER.score("a" * 20, expected_len=10)
        assert short == pytest.approx(long) == pytest.approx(0.5)

    def test_empty_hypothesis_is_zero(self):
        assert self.VERIFIER.score("", expected_len=10) == 0.0

    def test_no_prior_is_uninformative(self):
        assert self.VERIFIER.score("abc", expected_len=None) == 0.5


@pytest.mark.unit
class TestRunningScore:
    def test_default_is_disagreement(self):
        assert running_score(0.3, {"lexicon": 0.9}, {"lexicon": 0.25}) == 0.3

    def test_floor_override(self):
        assert running_score(0.05, {"lexicon": 0.1}, {"lexicon": 0.25}) == 1.0

    def test_missing_score_does_not_trigger_floor(self):
        # a verifier that has not run cannot veto (z-bar convention)
        assert running_score(0.2, {}, {"lexicon": 0.25}) == 0.2

    def test_clipped_to_unit_interval(self):
        assert running_score(1.7, {}, {}) == 1.0
        assert running_score(-0.2, {}, {}) == 0.0
