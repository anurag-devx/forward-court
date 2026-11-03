"""Offline tests for deterministic phishing scoring (no model calls)."""
from card.formatter import format_card
from links.check import check_links
from links.scamtext import ScamReading
from safety.lexicon import normalize_text, score_message
from tests.fakes import FakeClient, dead_http_client

SBI_KYC = (
    "Your SBI account will be blocked today. Update KYC at "
    "http://sbi-kyc-update.xyz and share the OTP."
)
SBI_URL = "http://sbi-kyc-update.xyz"


def test_normalize_handles_case_punct_and_hindi():
    assert normalize_text("OTP बताएं!") == "otp बताएं"
    assert normalize_text("Share, the OTP.") == "share the otp"


def test_sbi_kyc_sample_is_dangerous():
    result = score_message(SBI_KYC, [SBI_URL])
    assert result.level == "dangerous"
    assert result.score >= 6
    assert result.reasons[0] == "Pretends to be SBI but the site isn't SBI's"
    assert "Asks for your OTP" in result.reasons
    assert len(result.reasons) <= 3


def test_official_brand_domain_has_no_penalty():
    result = score_message("Check balance", ["https://onlinesbi.sbi/personal"])
    assert result.score == 0
    assert "brand" not in result.matched
    result = score_message("SBI announces results", ["https://sbi.co.in/results"])
    assert result.level == "ok"


def test_subdomain_of_official_is_clean():
    result = score_message("x", ["https://care.sbi.co.in/alerts"])
    assert "brand" not in result.matched


def test_gov_host_counts_official_for_gov_brands():
    result = score_message("x", ["https://uidai.incometax.gov.in/services"])
    assert "brand" not in result.matched
    assert result.score == 0
    result = score_message("x", ["https://sbi.incometax.gov.in/services"])
    assert result.matched.get("brand") == ["SBI"]


def test_plain_news_link_scores_zero():
    result = score_message(
        "India won the match today", ["https://www.thehindu.com/sport/x"]
    )
    assert result.score == 0
    assert result.level == "ok"


def test_hindi_otp_scam_is_dangerous():
    text = (
        "SBI KYC turant poora karo, aaj hi OTP batayein "
        "http://sbi-kyc-update.xyz warna khata band"
    )
    result = score_message(text, [SBI_URL], language="hi")
    assert result.level == "dangerous"
    assert any("OTP" in reason for reason in result.reasons)


def test_pib_link_is_clean():
    result = score_message(
        "PIB release",
        ["https://www.pib.gov.in/PressReleasePage.aspx?PRID=2331200"],
    )
    assert result.score == 0


def test_kerala_lottery_news_is_clean():
    result = score_message(
        "Kerala lottery results announced today",
        ["https://www.thehindu.com/kerala/x"],
    )
    assert result.score == 0


def test_digit_swap_brand_is_lookalike():
    result = score_message("verify", ["https://sb1-verify.com/login"])
    assert result.level == "dangerous"
    assert "Uses a lookalike address with swapped letters" in result.reasons


def test_punycode_is_lookalike():
    result = score_message("x", ["https://xn--sbi-example.com/"])
    assert result.score >= 3
    assert "Uses a deceptive special-character address" in result.reasons


def test_risky_tld_and_free_host():
    assert score_message("x", ["https://example.xyz/d"]).matched[
        "risky_tld"
    ] == ["xyz"]
    assert score_message("x", ["https://foo.vercel.app/x"]).matched[
        "free_host"
    ] == ["vercel.app"]


def test_url_actions_capped_and_gated():
    capped = score_message("x", ["https://evil.blogspot.com/kyc-verify-claim"])
    assert capped.matched["url_actions"] == ["verify", "kyc"]
    unknown = score_message("x", ["https://example.com/kyc-verify"])
    assert unknown.score == 2
    assert score_message("x", ["https://sbi.co.in/verify"]).score == 0
    assert score_message(
        "x", ["https://www.thehindu.com/verify-account"]
    ).score == 0


def test_generic_action_word_needs_company():
    assert score_message("x", ["https://picuki.com/update"]).score == 0
    assert "update" in score_message(
        "x", ["https://evil.blogspot.com/kyc-update"]
    ).matched["url_actions"]


def test_download_extension_is_instant_dangerous():
    result = score_message("get it", ["https://evil.xyz/app.apk"])
    assert result.level == "dangerous"
    assert result.download is True
    assert "Leads to a file download (.apk)" in result.reasons
    assert score_message("x", ["https://evil.xyz/APP.EXE"]).level == "dangerous"


def test_credentials_window_and_negation():
    assert "credentials" in score_message("share the OTP", []).matched
    assert "credentials" in score_message("UPI PIN batao", []).matched
    assert "credentials" not in score_message(
        "SBI says never share your OTP with anyone", []
    ).matched
    assert "credentials" in score_message("kripya OTP batayein", []).matched
    assert "credentials" not in score_message("OTP बताएं मत", []).matched


def test_urgency_phrase_vs_single_word():
    assert "urgency" in score_message("it will be blocked", []).matched
    assert "urgency" not in score_message("do it today", []).matched
    gated = score_message("share the OTP today", [])
    assert "urgency" in gated.matched


def test_threats_money_job_remote():
    assert "threats" in score_message("bijli kat jayegi", []).matched
    assert "money_bait" in score_message("lottery jeeti hai", []).matched
    assert "job_bait" in score_message("naukri lag gayi", []).matched
    assert "remote_access" in score_message("install anydesk", []).matched


def test_task_needs_pay_word():
    assert "job_bait" not in score_message("finish this task", []).matched
    assert "job_bait" in score_message(
        "complete tasks, earn per day", []
    ).matched


def test_bare_lottery_needs_creds_or_claim_link():
    assert "money_bait" not in score_message("lottery results", []).matched
    assert "money_bait" in score_message(
        "lottery result, share the OTP", []
    ).matched
    assert "money_bait" in score_message(
        "lottery", ["https://example.com/claim-prize"]
    ).matched


def test_level_bands():
    assert score_message("x", ["https://example.xyz/d"]).level == "ok"
    assert score_message("x", ["https://sbi-login.example.com/"]).level == (
        "suspicious"
    )
    assert score_message(
        "x", ["https://sbi-alerts.example.xyz/"]
    ).level == "suspicious"
    assert score_message(SBI_KYC, [SBI_URL]).level == "dangerous"


def test_bare_lookalike_link_is_dangerous():
    result = score_message(SBI_URL, [SBI_URL])
    assert result.score == 7
    assert result.level == "dangerous"


def test_shortener_note_fills_leftover_slots():
    result = score_message("share the OTP", ["https://bit.ly/abc123"])
    assert result.level == "suspicious"
    assert "Hides the real address behind a short link" in result.reasons


def test_hindi_reasons():
    result = score_message("apna OTP batayein", [], language="hi")
    assert "OTP मांगता है" in result.reasons


def test_demo_samples_keep_levels():
    from demo.replay import load_samples

    for sample in load_samples():
        if sample["kind"] == "screenshot":
            params = sample["screenshot"]
            text = f"{params['headline']} {params['standfirst']}"
            urls = []
        else:
            text = sample["text"]
            urls = [text] if text.startswith("http") else []
            if not urls:
                from links.extract import find_urls

                urls = find_urls(text)
        level = score_message(text, urls).level
        if sample["id"] == "malicious":
            assert level == "dangerous", sample["id"]
        else:
            assert level == "ok", (sample["id"], level)


def _silent_client():
    return FakeClient(
        parsed_by_model={ScamReading: ScamReading(scam_type="", signals=[])}
    )


def test_lexicon_dangerous_wins_when_model_silent(monkeypatch):
    monkeypatch.setenv("GSB", "off")
    verdict, info = check_links(
        SBI_KYC, "en", client=_silent_client(),
        http_client=dead_http_client(),
    )
    assert verdict is not None and verdict.verdict == "DANGEROUS"
    assert "Pretends to be SBI" in verdict.reason
    assert info["lexicon"]["level"] == "dangerous"


def test_lexicon_suspicious_card(monkeypatch):
    monkeypatch.setenv("GSB", "off")
    verdict, info = check_links(
        "https://sbi-alerts.example.xyz/", "en", client=_silent_client(),
        http_client=dead_http_client(),
    )
    assert verdict is not None and verdict.verdict == "SUSPICIOUS"
    assert info["lexicon"]["level"] == "suspicious"
    card = format_card(verdict)
    assert "⚠️ *Be careful — scam signs*" in card
    assert "- Pretends to be SBI" in card
    assert "Never share OTP/PIN" in card
    assert "safe" not in card.lower()


def test_model_dangerous_kept_when_lexicon_clean(monkeypatch):
    monkeypatch.setenv("GSB", "off")
    fake = FakeClient(
        parsed_by_model={ScamReading: ScamReading(
            scam_type="Fake KYC update", signals=["otp_pin", "urgency"]
        )}
    )
    verdict, info = check_links(
        "hello", "en", client=fake, http_client=dead_http_client()
    )
    assert verdict is not None and verdict.verdict == "DANGEROUS"
    assert verdict.reason == "The message itself matches a known scam pattern."
    assert info["lexicon"]["level"] == "ok"
