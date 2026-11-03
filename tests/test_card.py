from card.formatter import format_card
from court.models import Verdict
from evidence.models import ClaimReview, EvidenceItem


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


def _evidence_item(url, snippet, **flags):
    return EvidenceItem(
        claim="India beat South Africa in the final",
        source_url=url,
        date="2024-06-29",
        snippet=snippet,
        **flags,
    )


def test_corroboration_lists_outlets_and_links():
    verdict = _verdict(
        verdict="True", reason="confirmed",
        key_source="https://key.example/a",
    )
    evidence = [
        _evidence_item("https://key.example/a", "India beat South Africa"),
        _evidence_item(
            "https://www.thehindu.com/r",
            "India beat South Africa in the final",
        ),
        _evidence_item(
            "https://www.bbc.com/r",
            "India beat South Africa in the final",
        ),
    ]
    card = format_card(
        verdict, evidence, ["India beat South Africa in the final"]
    )
    assert "Also confirmed by: thehindu.com, bbc.com" in card
    assert "🔗 https://www.thehindu.com/r" in card
    assert "🔗 https://www.bbc.com/r" in card


def test_corroboration_shows_factcheck_rating():
    verdict = _verdict(
        verdict="Misleading", reason="old but real",
        key_source="https://old.example/k",
    )
    evidence = [
        _evidence_item("https://old.example/k", "India beat South Africa"),
        _evidence_item(
            "https://factly.in/r", "India beat South Africa in the final",
            claim_review=ClaimReview(
                rating_raw="True", verdict="True",
                claim_text="India beat South Africa in the final",
            ),
        ),
    ]
    card = format_card(
        verdict, evidence, ["India beat South Africa in the final"]
    )
    assert "Also confirmed by: factly.in [True]" in card


def test_corroboration_absent_without_evidence():
    card = format_card(_verdict(verdict="True", reason="confirmed"))
    assert "Also confirmed by" not in card


def test_claim_review_line_shown_once_with_rating():
    card = format_card(_verdict(
        verdict="False",
        reason="Already fact-checked by altnews.in: rated False.",
        key_source="https://www.altnews.in/x",
        fact_checked_by="altnews.in",
        fact_check_rating="False",
    ))
    assert card.count("Already fact-checked by altnews.in") == 1
    assert "Already fact-checked by altnews.in [False]" in card
    assert "rated False." not in card


def test_claim_review_dedupe_hindi():
    card = format_card(_verdict(
        verdict="False", language="hi",
        reason="altnews.in द्वारा पहले से तथ्य-जांचित: रेटिंग False।",
        key_source="https://www.altnews.in/x",
        fact_checked_by="altnews.in",
        fact_check_rating="False",
    ))
    assert "पहले से तथ्य-जांचित: altnews.in [False]" in card
    assert card.count("altnews.in") == 2  # bracket line + source link


def test_debate_fact_checked_line_keeps_substantive_reason():
    card = format_card(_verdict(fact_checked_by="Vishvas News"))
    assert "Already fact-checked by Vishvas News" in card
    assert "[False]" not in card
    assert "Doctors never said this." in card
