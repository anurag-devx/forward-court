from datetime import date, timedelta

import httpx

import main
from card.formatter import format_card
from court.judge import judge_verdict
from court.models import Argument, ArgPoint, Verdict
from evidence.models import EvidenceItem
from links.article import (
    article_tier_note,
    fetch_article,
    is_bare_link,
    to_evidence,
)
from links.check import check_links
from links.decide import decide
from links.extract import (
    expand_shorteners,
    expand_url,
    find_urls,
    is_shortener,
)
from links.heuristics import age_signals, head_signals, url_signals
from links.safebrowsing import ENDPOINT, check_urls
from links.scamtext import ScamReading, read_scam_text
from tests.fakes import FakeClient, dead_http_client


def _scam_client(**overrides):
    parsed = {ScamReading: ScamReading(scam_type="", signals=[])}
    parsed.update(overrides.get("parsed", {}))
    return FakeClient(parsed_by_model=parsed)


# --- extract ---


def test_find_urls_dedupes_and_strips_punctuation():
    text = (
        "See https://a.com/x, and https://a.com/x! "
        "Also (http://b.org/y)."
    )
    assert find_urls(text) == ["https://a.com/x", "http://b.org/y"]


def test_find_urls_empty():
    assert find_urls("") == []
    assert find_urls("no links here") == []


def test_is_shortener_known_and_unknown():
    assert is_shortener("https://bit.ly/abc")
    assert is_shortener("https://www.tinyurl.com/abc")
    assert not is_shortener("https://example.com/abc")
    assert not is_shortener("not a url")


def test_expand_url_follows_head_hops():
    seen: list[str] = []

    def handler(request):
        seen.append(str(request.url))
        if request.url.path == "/short":
            return httpx.Response(
                301, headers={"location": "https://final.example/landing"}
            )
        return httpx.Response(200)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert expand_url("https://bit.ly/short", client) == (
        "https://final.example/landing"
    )
    # shortener hops are HEADed (no body) so GSB can check the target
    assert seen == ["https://bit.ly/short", "https://final.example/landing"]


def test_expand_url_returns_original_on_failure():
    client = dead_http_client()
    assert (
        expand_url("https://bit.ly/short", client) == "https://bit.ly/short"
    )


def test_expand_shorteners_only_touches_shorteners():
    seen: list[str] = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = expand_shorteners(
        ["https://example.com/x", "https://bit.ly/s"], client
    )
    assert out == {"https://bit.ly/s": "https://bit.ly/s"}
    assert seen == ["https://bit.ly/s"]


# --- safebrowsing ---


def _gsb_client(payload=None, status=200):
    def handler(request):
        if request.url.host != "safebrowsing.googleapis.com":
            return httpx.Response(
                200, headers={"content-type": "text/html"}, text="<p>x</p>"
            )
        assert request.url.path == "/v4/threatMatches:find"
        assert request.url.params["key"] == "k"
        return httpx.Response(status, json=payload or {})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_safebrowsing_parses_hits():
    client = _gsb_client(
        {"matches": [{
            "threatType": "SOCIAL_ENGINEERING",
            "threat": {"url": "http://evil.example/"},
        }]}
    )
    assert check_urls(["http://evil.example/"], "k", client) == {
        "http://evil.example/": ["SOCIAL_ENGINEERING"]
    }


def test_safebrowsing_empty_on_clean_error_or_missing_key():
    assert check_urls(
        ["http://ok.example/"], "k", _gsb_client({})
    ) == {}
    assert check_urls(
        ["http://evil.example/"], "k", _gsb_client(status=500)
    ) is None
    assert check_urls(["http://evil.example/"], "", _gsb_client()) == {}
    assert check_urls([], "k", _gsb_client()) == {}
    assert ENDPOINT.startswith("https://safebrowsing.googleapis.com/")


def test_gsb_error_noted_as_unavailable(monkeypatch):
    from config import UsageTracker

    monkeypatch.setenv("GSB_API_KEY", "k")
    tracker = UsageTracker()
    verdict, _ = check_links(
        "See https://example.com/x",
        "en",
        client=_scam_client(),
        http_client=_gsb_client(status=500),
        tracker=tracker,
    )
    assert verdict is None
    assert "GSB unavailable (error); heuristics only." in tracker.notes


# --- heuristics ---


def test_punycode_raw_ip_and_subdomains():
    assert "punycode_host" in url_signals("http://xn--pple-43d.com/")
    assert "raw_ip_host" in url_signals("http://93.184.216.34/x")
    assert "many_subdomains" in url_signals(
        "https://a.b.c.d.evil.example/x"
    )
    assert "many_subdomains" not in url_signals("https://a.b.example/x")


def test_at_and_plain_http_and_risky_extension():
    assert "at_in_url" in url_signals("https://ok.example@evil.example/")
    assert "plain_http" in url_signals("http://example.com/")
    assert "plain_http" not in url_signals("https://example.com/")
    assert "risky_download" in url_signals("https://example.com/app.apk")
    assert "risky_download" not in url_signals("https://example.com/app.html")


def test_lookalike_outlet_and_brand_spoof():
    outlet = url_signals("https://abcnews.com.co/story")
    assert "lookalike_domain" in outlet
    brand = url_signals("https://login.hdfcbank.verify.evil.com/")
    assert "lookalike_domain" in brand
    assert "hdfcbank" in brand["lookalike_domain"]
    legit = url_signals("https://www.hdfcbank.com/")
    assert "lookalike_domain" not in legit


def test_head_signals_risky_content_type_only():
    def handler(request):
        if "apk" in str(request.url):
            return httpx.Response(
                200, headers={"content-type": "application/octet-stream"}
            )
        return httpx.Response(200, headers={"content-type": "text/html"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert "risky_download" in head_signals(
        "https://example.com/app.apk", client
    )
    assert head_signals("https://example.com/page", client) == {}
    assert head_signals("https://example.com/x", dead_http_client()) == {}


def _rdap_client(registered: date | None):
    def handler(request):
        if request.url.host == "data.iana.org":
            return httpx.Response(
                200, json={"services": [[["xyz"], ["https://rdap.example/"]]]}
            )
        assert request.url.host == "rdap.example"
        events = (
            [{
                "eventAction": "registration",
                "eventDate": registered.isoformat(),
            }]
            if registered is not None
            else []
        )
        return httpx.Response(200, json={"events": events})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_new_domain_signal_and_old_or_unknown_silent():
    new = _rdap_client(date.today() - timedelta(days=5))
    assert "new_domain" in age_signals("https://evil.xyz/", new)
    old = _rdap_client(date.today() - timedelta(days=400))
    assert age_signals("https://evil.xyz/", old) == {}
    unknown = _rdap_client(None)
    assert age_signals("https://evil.xyz/", unknown) == {}
    assert age_signals("https://evil.xyz/", dead_http_client()) == {}


# --- scamtext ---


def test_scamtext_drops_unknown_signals():
    fake = FakeClient(
        parsed_by_model={
            ScamReading: ScamReading(
                scam_type="x", signals=["otp_pin", "made_up"]
            )
        }
    )
    reading = read_scam_text("text", "en", client=fake)
    assert reading.signals == ["otp_pin"]


# --- decide ---


def _decide(signals, scam=(), scam_type=""):
    return decide(signals, list(scam), scam_type, "en")


def test_decide_matrix():
    assert _decide({"gsb_hit": "SOCIAL_ENGINEERING"}).verdict == "DANGEROUS"
    assert _decide({"risky_download": "app.apk"}).verdict == "DANGEROUS"
    both = _decide({"lookalike_domain": "x"}, ["urgency"], "Fake KYC update")
    assert both.verdict == "DANGEROUS"
    assert both.scam_type == "Fake KYC update"
    three = _decide(
        {"punycode_host": "a", "raw_ip_host": "b", "at_in_url": "c"}
    )
    assert three.verdict == "DANGEROUS"
    assert " ".join(three.reason.split()) == three.reason
    otp = _decide({}, ["otp_pin", "urgency"], "Fake KYC update")
    assert otp.verdict == "DANGEROUS"
    install = _decide({}, ["install_app", "payment"])
    assert install.verdict == "DANGEROUS"
    assert install.scam_type is None


def test_decide_clean_cases_return_none():
    assert _decide({}) is None
    assert _decide({"plain_http": "u"}) is None  # weak signal alone
    assert _decide({"lookalike_domain": "x"}) is None  # needs scam signal
    assert _decide({}, ["urgency"]) is None  # single scam signal
    assert _decide({"punycode_host": "a", "at_in_url": "b"}) is None


def test_decide_hindi_reasons():
    verdict = decide({"gsb_hit": "MALWARE"}, [], "", "hi")
    assert verdict.verdict == "DANGEROUS"
    assert "Google Safe Browsing" in verdict.reason


# --- check orchestration ---


def _gsb_hit_client(seen: list):
    def handler(request):
        seen.append(request)
        assert request.method == "POST"
        return httpx.Response(
            200,
            json={"matches": [{
                "threatType": "MALWARE",
                "threat": {"url": "http://evil.example/app.apk"},
            }]},
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_gsb_hit_host_never_contacted(monkeypatch):
    monkeypatch.setenv("GSB_API_KEY", "k")
    seen: list = []
    verdict, info = check_links(
        "Get it here http://evil.example/app.apk",
        "en",
        client=_scam_client(),
        http_client=_gsb_hit_client(seen),
    )
    assert verdict is not None and verdict.verdict == "DANGEROUS"
    assert seen and all(
        r.url.host == "safebrowsing.googleapis.com" for r in seen
    )
    assert info["flagged_hosts"] == {"evil.example"}
    assert info["safe_urls"] == []


def test_gsb_runs_before_final_host_contact(monkeypatch):
    monkeypatch.setenv("GSB_API_KEY", "k")
    order: list[str] = []

    def handler(request):
        order.append(f"{request.method} {request.url.host}")
        if request.url.host == "safebrowsing.googleapis.com":
            return httpx.Response(200, json={})
        if request.url.host == "data.iana.org":
            return httpx.Response(200, json={"services": []})
        return httpx.Response(200, headers={"content-type": "text/html"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    verdict, info = check_links(
        "Read https://example.com/story",
        "en",
        client=_scam_client(),
        http_client=client,
    )
    assert verdict is None
    assert info["safe_urls"] == ["https://example.com/story"]
    assert order[0] == "POST safebrowsing.googleapis.com"
    assert "HEAD example.com" in order


def test_scamtext_runs_without_urls_and_flags_scam(monkeypatch):
    monkeypatch.delenv("GSB_API_KEY", raising=False)
    fake = FakeClient(
        parsed_by_model={
            ScamReading: ScamReading(
                scam_type="Fake KYC update",
                signals=["otp_pin", "urgency", "kyc_block"],
            )
        }
    )
    verdict, info = check_links(
        "Your KYC is blocked, share OTP now",
        "en",
        client=fake,
        http_client=dead_http_client(),
    )
    assert verdict is not None and verdict.verdict == "DANGEROUS"
    assert verdict.scam_type == "Fake KYC update"
    assert info["scam_signals"] == ["otp_pin", "urgency", "kyc_block"]


def test_no_key_runs_heuristics_only(monkeypatch):
    monkeypatch.delenv("GSB_API_KEY", raising=False)
    verdict, info = check_links(
        "See https://example.com/x",
        "en",
        client=_scam_client(),
        http_client=dead_http_client(),
    )
    assert verdict is None
    assert info["safe_urls"] == ["https://example.com/x"]


# --- article ---

ARTICLE_HTML = """<html><head>
<title>Big Event Happens</title>
<meta property="article:published_time" content="2026-09-01T10:00:00Z">
</head><body><article><p>{}</p></article></body></html>"""


def _article_client(html: str):
    def handler(request):
        assert request.method == "GET"
        return httpx.Response(
            200, headers={"content-type": "text/html"}, text=html
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_article_headline_outlet_date():
    html = ARTICLE_HTML.format("Body text here. " * 30)
    article = fetch_article(
        "https://www.example.com/story", _article_client(html)
    )
    assert article is not None
    assert article.headline == "Big Event Happens"
    assert article.outlet == "example.com"
    assert article.date == "2026-09-01"
    assert "Body text" in article.excerpt


def test_fetch_article_none_on_failure():
    assert (
        fetch_article("https://example.com/x", dead_http_client()) is None
    )


def test_is_bare_link():
    assert is_bare_link("https://a.com/x", ["https://a.com/x"])
    assert not is_bare_link("Look: https://a.com/x", ["https://a.com/x"])


def test_article_tier_note():
    from links.article import ArticleInfo

    spoof = ArticleInfo(
        url="https://abcnews.com.co/x", headline="h", outlet="abcnews.com.co",
        date=None, excerpt="",
    )
    assert "impersonates" in article_tier_note(spoof)
    tier1 = ArticleInfo(
        url="https://www.bbc.com/news/x", headline="h", outlet="bbc.com",
        date=None, excerpt="",
    )
    note = article_tier_note(tier1)
    assert note is not None and "tier-1" in note
    other = ArticleInfo(
        url="https://blog.example/x", headline="h", outlet="blog.example",
        date=None, excerpt="",
    )
    assert article_tier_note(other) is None


def test_shared_article_evidence_tag():
    from links.article import ArticleInfo

    article = ArticleInfo(
        url="https://example.com/x", headline="h", outlet="example.com",
        date="2026-09-01", excerpt="excerpt",
    )
    item = to_evidence(article, "Big Event Happens")
    assert item.shared_article is True
    assert item.source_url == "https://example.com/x"
    assert item.date == "2026-09-01"


# --- judge article rules ---


def _verdict(**overrides):
    base = {
        "verdict": "True",
        "reason": "supported",
        "key_source": "https://a.com/x",
        "language": "en",
        "confidence": "high",
        "claim_timeframe": None,
    }
    base.update(overrides)
    return Verdict(**base)


def _argue():
    return Argument(
        points=[ArgPoint(text="p", source_url="https://a.com/x")]
    )


def test_recycled_article_with_recency_cues_is_misleading():
    fake = FakeClient(parsed_by_model={Verdict: _verdict()})
    evidence = [
        EvidenceItem(
            claim="c", source_url="https://a.com/x", date="2026-10-01",
            snippet="s",
        )
    ]
    verdict = judge_verdict(
        ["c"], evidence, _argue(), _argue(), "en", True,
        client=fake,
        article_url="https://news.example/old",
        article_date="2020-01-01",
    )
    assert verdict.verdict == "Misleading"
    assert verdict.key_source == "https://news.example/old"


def test_bare_old_link_gets_date_line_no_auto_verdict():
    fake = FakeClient(parsed_by_model={Verdict: _verdict()})
    evidence = [
        EvidenceItem(
            claim="c", source_url="https://a.com/x", date="2026-10-01",
            snippet="s",
        )
    ]
    verdict = judge_verdict(
        ["c"], evidence, _argue(), _argue(), "en", False,
        client=fake,
        article_url="https://news.example/old",
        article_date="2020-01-01",
        article_bare=True,
    )
    assert verdict.verdict == "True"
    assert verdict.article_date == "2020-01-01"
    assert "Article from 2020-01-01" in format_card(verdict)


def test_recycled_rule_needs_recency_cues_when_not_bare():
    fake = FakeClient(parsed_by_model={Verdict: _verdict()})
    evidence = [
        EvidenceItem(
            claim="c", source_url="https://a.com/x", date="2026-10-01",
            snippet="s",
        )
    ]
    verdict = judge_verdict(
        ["c"], evidence, _argue(), _argue(), "en", False,
        client=fake,
        article_url="https://news.example/old",
        article_date="2020-01-01",
    )
    assert verdict.verdict == "True"
    assert verdict.article_date is None


def test_shared_article_cannot_be_own_key_source():
    fake = FakeClient(
        parsed_by_model={
            Verdict: _verdict(key_source="https://news.example/shared")
        }
    )
    shared = EvidenceItem(
        claim="c", source_url="https://news.example/shared",
        date="2026-10-01", snippet="s", shared_article=True,
    )
    other = EvidenceItem(
        claim="c", source_url="https://a.com/x", date="2026-10-01",
        snippet="s",
    )
    argue = Argument(
        points=[
            ArgPoint(text="p", source_url="https://news.example/shared"),
            ArgPoint(text="q", source_url="https://a.com/x"),
        ]
    )
    verdict = judge_verdict(
        ["c"], [shared, other], argue, argue, "en", False, client=fake
    )
    assert verdict.verdict == "True"
    assert verdict.key_source == "https://a.com/x"


def test_shared_article_alone_downgrades_true():
    fake = FakeClient(
        parsed_by_model={
            Verdict: _verdict(key_source="https://news.example/shared")
        }
    )
    shared = EvidenceItem(
        claim="c", source_url="https://news.example/shared",
        date="2026-10-01", snippet="s", shared_article=True,
    )
    argue = Argument(
        points=[
            ArgPoint(text="p", source_url="https://news.example/shared")
        ]
    )
    verdict = judge_verdict(
        ["c"], [shared], argue, argue, "en", False, client=fake
    )
    assert verdict.verdict == "Unverified"


# --- cards ---


def test_dangerous_card_en_and_hi():
    verdict = Verdict(
        verdict="DANGEROUS", reason="Uses a lookalike domain (x).",
        key_source="", language="en", confidence="high",
        scam_type="Fake KYC update",
    )
    card = format_card(verdict)
    assert card.splitlines()[0] == "⚠️ *DANGEROUS LINK — do not open*"
    assert "Fake KYC update" in card
    assert "cybercrime.gov.in / 1930" in card
    verdict_hi = verdict.model_copy(update={"language": "hi"})
    card_hi = format_card(verdict_hi)
    assert "खतरनाक लिंक" in card_hi.splitlines()[0]
    assert "1930" in card_hi


# --- main integration ---


def test_scam_message_short_circuits_before_grounding(monkeypatch):
    from intake.models import IntakeResult

    monkeypatch.delenv("GSB_API_KEY", raising=False)
    fake = FakeClient(
        parsed_by_model={
            IntakeResult: IntakeResult(
                language="en", claims=["kyc blocked"], presented_as_recent=True
            ),
            ScamReading: ScamReading(
                scam_type="Fake KYC update",
                signals=["otp_pin", "urgency"],
            ),
        }
    )
    card, tracker = main.run_forward(
        "KYC blocked, share OTP now", client=fake,
        http_client=dead_http_client(),
    )
    assert "DANGEROUS LINK" in card.splitlines()[0]
    assert "Fake KYC update" in card
    assert len(fake.responses.calls) == 2  # intake + scamtext only
    assert "grounding" not in "".join(tracker.breakdown())


def test_blocked_hosts_never_fetched():
    from config import UsageTracker
    from evidence.grounder import gather_evidence
    from tests.fakes import search_call

    seen: list[str] = []

    def handler(request):
        seen.append(request.url.host)
        return httpx.Response(
            200, headers={"content-type": "text/html"}, text="<p>x</p>"
        )

    fake = FakeClient(
        outputs=[
            [search_call([
                ("http://evil.example/x", "T", "bad"),
                ("https://ok.example/y", "T", "good"),
            ])],
            [search_call([])],
        ]
    )
    tracker = UsageTracker()
    items = gather_evidence(
        ["some claim"],
        client=fake,
        tracker=tracker,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        blocked_hosts=frozenset({"evil.example"}),
    )
    assert [item.source_url for item in items] == ["https://ok.example/y"]
    assert "evil.example" not in seen
    assert any("unsafe" in note for note in tracker.notes)


def test_blocked_hosts_never_fetched_in_lookup():
    from evidence.factcheck import lookup_factchecks
    from tests.fakes import search_call

    seen: list[str] = []

    def handler(request):
        seen.append(request.url.host)
        return httpx.Response(
            200, headers={"content-type": "text/html"}, text="<p>x</p>"
        )

    fake = FakeClient(lookup_outputs=[
        [search_call([
            ("https://www.altnews.in/x", "T", "bad"),
            ("https://www.boomlive.in/y", "T", "good"),
        ])]
    ])
    items = lookup_factchecks(
        ["some claim"],
        client=fake,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        blocked_hosts=frozenset({"altnews.in"}),
    )
    assert [item.source_url for item in items] == [
        "https://www.boomlive.in/y"
    ]
    assert "www.altnews.in" not in seen


def test_clean_link_prepends_headline_claim(monkeypatch):
    from unittest import mock

    from evidence.relevance import RelevanceScore, RelevanceScores
    from intake.models import IntakeResult
    from links.article import ArticleInfo
    from tests.fakes import search_call

    monkeypatch.delenv("GSB_API_KEY", raising=False)
    article = ArticleInfo(
        url="https://example.com/story", headline="Big Event Happens",
        outlet="example.com", date="2026-10-20", excerpt="excerpt",
        review=None,
    )
    fake = FakeClient(
        parsed_by_model={
            IntakeResult: IntakeResult(
                language="en", claims=["other claim"],
                presented_as_recent=False,
            ),
            ScamReading: ScamReading(scam_type="", signals=[]),
            RelevanceScores: RelevanceScores(
                scores=[RelevanceScore(index=i, score=2)
                        for i in range(12)]
            ),
            Argument: Argument(
                points=[ArgPoint(text="p", source_url="https://a.com/x")]
            ),
            Verdict: Verdict(
                verdict="True", reason="supported",
                key_source="https://a.com/x", language="en",
                confidence="high", claim_timeframe=None,
            ),
        },
        outputs=[
            [search_call([("https://a.com/1", "T", "s")])],
            [search_call([])],
            [search_call([("https://a.com/2", "T", "s")])],
            [search_call([])],
        ],
        lookup_outputs=[[search_call([])]],
    )
    with mock.patch.object(main, "fetch_article", return_value=article):
        card, _ = main.run_forward(
            "Big news https://example.com/story", client=fake,
            http_client=dead_http_client(),
        )
    assert "DANGEROUS" not in card
    grounding_inputs = [
        kwargs["input"] for verb, kwargs in fake.responses.calls
        if verb == "create"
    ]
    assert grounding_inputs
    assert any("Big Event Happens" in text for text in grounding_inputs)


def test_gateway_risky_file_replies_dangerous():
    import gateway.worker as worker_module

    sent: list[tuple[str, str]] = []

    class Sender:
        def send(self, to, body):
            sent.append((to, body))

    worker_module.handle_risky_file(
        Sender(), to="91999", filename="update.apk", caption="install"
    )
    assert sent[0][1] == "Checking this forward…"
    assert "DANGEROUS LINK" in sent[1][1].splitlines()[0]


def test_safebrowsing_key_never_logged(monkeypatch):
    from config import UsageTracker

    monkeypatch.setenv("GSB_API_KEY", "secret-key-123")
    seen: list = []
    tracker = UsageTracker()
    verdict, _ = check_links(
        "Get it here http://evil.example/app.apk",
        "en",
        client=_scam_client(),
        http_client=_gsb_hit_client(seen),
        tracker=tracker,
    )
    assert verdict is not None
    assert "secret-key-123" not in "\n".join(tracker.notes)
    assert "secret-key-123" not in verdict.reason
    assert "secret-key-123" not in format_card(verdict)
