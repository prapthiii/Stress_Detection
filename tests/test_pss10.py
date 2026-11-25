import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.questionnaire import Questionnaire


def test_minimum_score():
    q = Questionnaire()
    responses = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    score = q.calculate_score(responses)
    assert score == 16


def test_maximum_score():
    q = Questionnaire()
    responses = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    score = q.calculate_score(responses)
    assert score == 24


def test_reverse_item_floor():
    q = Questionnaire()
    responses = [4, 4, 4, 0, 0, 4, 0, 0, 4, 4]
    score = q.calculate_score(responses)
    assert score == 40


def test_mixed_response_pattern():
    q = Questionnaire()
    responses = [2, 3, 4, 0, 2, 3, 2, 1, 4, 1]
    expected = 2 + 3 + 4 + 4 + 2 + 3 + 2 + 3 + 4 + 1
    actual = q.calculate_score(responses)
    assert actual == expected


def test_interpretation_boundaries():
    q = Questionnaire()
    assert q.interpret_score(7) == ("Very low perceived stress", 0)
    assert q.interpret_score(8) == ("Low perceived stress", 1)
    assert q.interpret_score(13) == ("Low perceived stress", 1)
    assert q.interpret_score(14) == ("Moderate perceived stress", 2)
    assert q.interpret_score(26) == ("Moderate perceived stress", 2)
    assert q.interpret_score(27) == ("High perceived stress", 3)
