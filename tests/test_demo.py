from pathlib import Path

from PIL import Image

import main
from config import UsageTracker
from demo import SAMPLES, make_screenshot
from evidence.grounder import GroundingError
from links.check import check_links
from links.scamtext import ScamReading
from tests.fakes import FakeClient, dead_http_client

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"


def test_samples_list():
    assert SAMPLES == ("text", "screenshot", "link", "scam")


def test_demo_text_files_present_and_nonempty():
    for name in ("text.txt", "link.txt", "scam.txt"):
        text = (DEMO_DIR / name).read_text().strip()
        assert text, name


def test_screenshot_generates_png(tmp_path):
    out = make_screenshot(str(tmp_path / "shot.png"))
    image = Image.open(out)
    assert image.size == (900, 620)
    assert image.mode == "RGB"


def test_screenshot_uses_fictional_masthead():
    from demo.make_screenshot import HEADLINE, MASTHEAD, URL_BAR

    assert MASTHEAD == "The Daily Clarion"
    assert "lemon" in HEADLINE and "cancer" in HEADLINE
    assert URL_BAR.startswith("dailyclarion.example/")


def test_demo_scam_sample_is_dangerous_offline(monkeypatch):
    monkeypatch.setenv("GSB", "off")
    monkeypatch.delenv("GSB_API_KEY", raising=False)
    text = (DEMO_DIR / "scam.txt").read_text().strip()
    assert "sbi-kyc-update.xyz" in text
    fake = FakeClient(
        parsed_by_model={
            ScamReading: ScamReading(
                scam_type="Fake KYC update",
                signals=["otp_pin", "urgency", "kyc_block"],
            )
        }
    )
    verdict, info = check_links(
        text, "en", client=fake, http_client=dead_http_client()
    )
    assert verdict is not None and verdict.verdict == "DANGEROUS"
    assert info["flagged_hosts"] == set()  # no GSB; heuristics + scamtext


def _ok_tracker():
    return UsageTracker()


def test_demo_runs_all_samples_in_order(monkeypatch, capsys):
    calls: list = []
    monkeypatch.setattr(
        main, "run_forward",
        lambda text, **_: calls.append(("text", text)) or ("CARD", _ok_tracker()),
    )
    monkeypatch.setattr(
        main, "run_image_forward",
        lambda path, *_, **__: calls.append(("image", path))
        or ("CARD", _ok_tracker()),
    )
    assert main.main(["--demo"]) == 0
    kinds = [kind for kind, _ in calls]
    assert kinds == ["text", "image", "text", "text"]
    assert Path(calls[1][1]).suffix == ".png"  # generated screenshot
    out = capsys.readouterr().out
    for name in SAMPLES:
        assert f": {name} ===" in out
        assert f"--- {name} took " in out
    assert "4/4 samples ok" in out


def test_demo_continues_after_failure(monkeypatch, capsys):
    calls: list = []

    def fail_once(text, **_):
        calls.append(text)
        if len(calls) == 1:
            raise GroundingError("no search")
        return "CARD", _ok_tracker()

    monkeypatch.setattr(main, "run_forward", fail_once)
    monkeypatch.setattr(
        main, "run_image_forward",
        lambda path, *_, **__: ("CARD", _ok_tracker()),
    )
    assert main.main(["--demo"]) == 1
    assert len(calls) == 3  # text, link, scam all attempted
    assert "3/4 samples ok" in capsys.readouterr().out


def test_demo_rejects_extra_args():
    assert main.main(["--demo", "extra"]) == 2
    assert main.main(["--demo", "--image", "x.png"]) == 2


def test_demo_forces_flags_off(monkeypatch):
    import os

    from config import gsb_enabled, reverse_search_enabled

    monkeypatch.setenv("REVERSE_SEARCH", "on")
    monkeypatch.setenv("GSB", "on")
    monkeypatch.setenv("SERPAPI_KEY", "k")
    monkeypatch.setenv("GSB_API_KEY", "k")
    observed: list = []
    monkeypatch.setattr(
        main, "run_forward",
        lambda text, **_: observed.append(
            (reverse_search_enabled(), gsb_enabled())
        ) or ("CARD", _ok_tracker()),
    )
    monkeypatch.setattr(
        main, "run_image_forward",
        lambda path, *_, **__: ("CARD", _ok_tracker()),
    )
    assert main.main(["--demo"]) == 0
    assert observed and all(state == (False, False) for state in observed)
    assert os.environ["REVERSE_SEARCH"] == "on"  # restored after demo
    assert os.environ["GSB"] == "on"


def test_env_example_documents_every_variable():
    documented = set()
    for line in (DEMO_DIR.parent / ".env.example").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            documented.add(line.split("=", 1)[0])
    assert {
        "MODEL_API_KEY",
        "FC_CONTRIBUTOR",
        "ZOOM",
        "REVERSE_SEARCH",
        "SERPAPI_KEY",
        "GSB",
        "GSB_API_KEY",
        "WHATSAPP_PROVIDER",
        "PUBLIC_BASE_URL",
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_WHATSAPP_FROM",
        "VONAGE_API_KEY",
        "VONAGE_API_SECRET",
        "VONAGE_WHATSAPP_NUMBER",
        "VONAGE_SIGNATURE_SECRET",
        "VONAGE_WEBHOOK_TOKEN",
    } <= documented


def test_explicit_provider_works_with_flag_off(monkeypatch, tmp_path):
    """REVERSE_SEARCH=off gates only the env-key auto-construction."""
    monkeypatch.setenv("REVERSE_SEARCH", "off")
    monkeypatch.setenv("SERPAPI_KEY", "k")
    seen: list = []

    class Provider:
        def search(self, image):
            seen.append(image)
            return []

    from court.models import Argument, ArgPoint, Verdict
    from evidence.relevance import RelevanceScore, RelevanceScores
    from media.models import VisionResult
    from tests.fakes import search_call

    image_path = tmp_path / "shot.png"
    make_screenshot(str(image_path))
    fake = FakeClient(
        parsed_by_model={
            VisionResult: VisionResult(
                scene_description="s",
                visible_text="",
                people_places_objects=[],
                claims_from_image=["photo shows X"],
                language="en",
                image_role="viral_claim",
            ),
            ScamReading: ScamReading(scam_type="", signals=[]),
            Argument: Argument(
                points=[ArgPoint(text="p", source_url="https://g.com/x")]
            ),
            RelevanceScores: RelevanceScores(
                scores=[RelevanceScore(index=i, score=2) for i in range(12)]
            ),
            Verdict: Verdict(
                verdict="True", reason="Looks right.",
                key_source="https://g.com/x", language="en",
                confidence="high", claim_timeframe=None,
            ),
        },
        outputs=[
            [search_call([("https://g.com/x", "T", "s")])],
            [search_call([])],
        ],
        lookup_outputs=[[search_call([])]],
    )
    card, tracker = main.run_image_forward(
        str(image_path), client=fake, http_client=dead_http_client(),
        provider=Provider(),
    )
    assert seen, "explicit provider must still be used"
    assert "reverse search disabled" not in "\n".join(tracker.notes)


def test_reverse_flag_on_without_key_notes_skip(monkeypatch, tmp_path):
    monkeypatch.setenv("REVERSE_SEARCH", "on")
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    from court.models import Argument, ArgPoint, Verdict
    from evidence.relevance import RelevanceScore, RelevanceScores
    from media.models import VisionResult
    from tests.fakes import search_call

    image_path = tmp_path / "shot.png"
    make_screenshot(str(image_path))
    fake = FakeClient(
        parsed_by_model={
            VisionResult: VisionResult(
                scene_description="s",
                visible_text="",
                people_places_objects=[],
                claims_from_image=["photo shows X"],
                language="en",
                image_role="viral_claim",
            ),
            ScamReading: ScamReading(scam_type="", signals=[]),
            Argument: Argument(
                points=[ArgPoint(text="p", source_url="https://g.com/x")]
            ),
            RelevanceScores: RelevanceScores(
                scores=[RelevanceScore(index=i, score=2) for i in range(12)]
            ),
            Verdict: Verdict(
                verdict="True", reason="Looks right.",
                key_source="https://g.com/x", language="en",
                confidence="high", claim_timeframe=None,
            ),
        },
        outputs=[
            [search_call([("https://g.com/x", "T", "s")])],
            [search_call([])],
        ],
        lookup_outputs=[[search_call([])]],
    )
    _, tracker = main.run_image_forward(
        str(image_path), client=fake, http_client=dead_http_client()
    )
    assert "reverse search skipped: SERPAPI_KEY not set." in tracker.notes
