"""Unit tests for the edit-distance metrics (Eqs. 1, 2, 5)."""
import pytest

from ugra.metrics.edit import cer, disagreement, levenshtein, pooled_cer, wer


@pytest.mark.unit
class TestLevenshtein:
    def test_identity(self):
        assert levenshtein("abc", "abc") == 0

    def test_known_distance(self):
        assert levenshtein("kitten", "sitting") == 3

    def test_symmetry(self):
        assert levenshtein("abcd", "xy") == levenshtein("xy", "abcd")

    def test_empty(self):
        assert levenshtein("", "abc") == 3
        assert levenshtein("", "") == 0


@pytest.mark.unit
class TestCER:
    def test_perfect(self):
        assert cer("plan", "plan") == 0.0

    def test_clip_binds_for_overlong_hypothesis(self):
        # lev("abc", "a") = 2 > |y*| = 1; unclipped ratio would be 2.0.
        assert cer("abc", "a") == 1.0

    def test_bounded(self):
        assert 0.0 <= cer("xyzw", "plan") <= 1.0

    def test_empty_reference_guard(self):
        assert cer("abc", "") == 1.0
        assert cer("", "") == 0.0


@pytest.mark.unit
class TestWER:
    def test_word_level(self):
        assert wer("the cat sat", "the dog sat") == pytest.approx(1 / 3)

    def test_empty_reference_guard(self):
        assert wer("a b", "") == 2.0  # WER is conventionally unclipped


@pytest.mark.unit
class TestDisagreement:
    def test_symmetric(self):
        assert disagreement("abc", "abd") == disagreement("abd", "abc")

    def test_range(self):
        assert 0.0 <= disagreement("hello", "world") <= 1.0

    def test_empty_empty_guard(self):
        assert disagreement("", "") == 0.0

    def test_symmetric_denominator(self):
        # normalized by the longer string, not by either as reference
        assert disagreement("ab", "abcdef") == pytest.approx(4 / 6)


@pytest.mark.unit
class TestPooledCER:
    def test_micro_weighting(self):
        # 1 edit over 4 chars + 0 edits over 6 chars = 1/10, not mean(1/4, 0)
        assert pooled_cer(["plan", "letter"], ["plon", "letter"]) == pytest.approx(0.1)

    def test_alignment_required(self):
        with pytest.raises(ValueError):
            pooled_cer(["a"], ["a", "b"])
