from datetime import date

import pytest

from intake.recency import is_presented_as_recent

TODAY = date(2026, 10, 28)


@pytest.mark.parametrize(
    "text",
    [
        "BREAKING: flood hits city",
        "This just happened today morning",
        "Tonight at 9pm, share now",
        "कल रात की बड़ी खबर",
        "ताज़ा तस्वीरें वायरल हो रही हैं",
        "Event on 2026-10-27 was huge",
        "Photos from 25 October 2026",
    ],
)
def test_recency_cues_detected(text):
    assert is_presented_as_recent([text], TODAY) is True


@pytest.mark.parametrize(
    "text",
    [
        "A calm description of an old monument.",
        "Photos from 2015 archives.",
        "Event scheduled 2026-11-20.",  # future date, not recent
        "",
    ],
)
def test_non_recent_text_rejected(text):
    assert is_presented_as_recent([text], TODAY) is False


def test_combines_visible_text_and_caption():
    assert (
        is_presented_as_recent(["a quiet street", "forwarded many times"], TODAY)
        is False
    )
    assert is_presented_as_recent(["JUST IN", ""], TODAY) is True
    assert is_presented_as_recent([], TODAY) is False


def test_boundary_is_seven_days():
    assert is_presented_as_recent(["on 2026-10-21"], TODAY) is True
    assert is_presented_as_recent(["on 2026-10-20"], TODAY) is False
