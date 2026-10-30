from card.formatter import format_card
from court.models import Verdict


def _verdict(**overrides):
    base = {
        "verdict": "False",
        "reason": "Doctors never said this.",
        "key_source": "https://a.com/x",
        "language": "en",
        "confidence": "high",
    }
    base.update(overrides)
    return Verdict(**base)


def test_english_card():
    card = format_card(_verdict())
    assert "❌ *Forward Court Verdict: FALSE*" in card
    assert "Confidence: high" in card
    assert "Doctors never said this." in card
    assert "🔗 Source: https://a.com/x" in card


def test_hindi_card():
    card = format_card(_verdict(language="hi", reason="यह दावा गलत है।"))
    assert "फॉरवर्ड कोर्ट निर्णय" in card
    assert "विश्वास: high" in card
    assert "स्रोत: https://a.com/x" in card


def test_fact_checked_line_shown_in_both_languages():
    card = format_card(_verdict(fact_checked_by="altnews.in"))
    assert "Already fact-checked by altnews.in" in card
    card = format_card(_verdict(language="hi", fact_checked_by="altnews.in"))
    assert "पहले से तथ्य-जांचित: altnews.in" in card


def test_missing_key_source_omits_source_line():
    card = format_card(_verdict(verdict="Unverified", key_source=""))
    assert "🔗" not in card


def test_unknown_language_falls_back_to_english():
    card = format_card(_verdict(language="xx"))
    assert "Forward Court Verdict" in card
