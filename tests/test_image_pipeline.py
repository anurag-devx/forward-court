import io
from types import SimpleNamespace

from PIL import Image

import main
from court.models import Argument, ArgPoint, Verdict
from evidence.relevance import RelevanceScore, RelevanceScores
from evidence.reverse_image import ReverseMatch
from intake.models import IntakeResult
from media.models import VisionResult
from tests.fakes import FakeClient, dead_http_client, search_call


def _all_relevant():
    return RelevanceScores(
        scores=[RelevanceScore(index=i, score=2) for i in range(12)]
    )


def _image_file(tmp_path, name="fwd.jpg", width=200, height=100):
    target = tmp_path / name
    Image.new("RGB", (width, height), "blue").save(target, format="JPEG")
    return str(target)


def _sam_output():
    return [
        SimpleNamespace(
            type="message",
            content=[
                SimpleNamespace(
                    type="output_text",
                    text="<0f>1<|box;x1=10;y1=10;x2=60;y2=60;w=200;h=100;c=0.9|>",
                )
            ],
        )
    ]


def _vision():
    return VisionResult(
        scene_description="A flooded street.",
        visible_text="",
        people_places_objects=["flooded street"],
        claims_from_image=["vision claim one", "vision claim two"],
        language="en",
        image_role="viral_claim",
    )


def _client():
    return FakeClient(
        parsed_by_model={
            VisionResult: _vision(),
            IntakeResult: IntakeResult(
                language="en",
                claims=["caption claim one", "caption claim two"],
                presented_as_recent=True,
            ),
            Argument: Argument(
                points=[ArgPoint(text="p", source_url="https://g.com/0")]
            ),
            RelevanceScores: _all_relevant(),
            Verdict: Verdict(
                verdict="False",
                reason="Debunked.",
                key_source="https://g.com/0",
                language="en",
                confidence="high",
                claim_timeframe=None,
            ),
        },
        outputs=[_sam_output()]
        + [
            [search_call([(f"https://g.com/{i}", "T", "snippet")])]
            for i in range(6)
        ]
        + [[search_call([])]] * 3,
    )


class _FakeProvider:
    def __init__(self, matches):
        self.matches = matches
        self.searched = []

    def search(self, image):
        self.searched.append(image)
        return list(self.matches)


def test_full_image_pipeline_offline(tmp_path):
    provider = _FakeProvider(
        [
            ReverseMatch(
                source_url="https://old.com/pic", page_title="old pic",
                exact=True,
            ),
            ReverseMatch(
                source_url="https://blog.com/sim", page_title="similar",
                exact=False,
            ),
        ]
    )
    card, tracker = main.run_image_forward(
        _image_file(tmp_path),
        "breaking today",
        client=_client(),
        http_client=dead_http_client(),
        provider=provider,
    )
    assert "FALSE" in card.splitlines()[0]
    assert "reverse matches: 2 (1 exact) from 2 image(s)." in tracker.notes
    assert "judge agreement: 3/3 (False, False, False)." in tracker.notes
    stages = [line.split(":")[0].strip() for line in tracker.breakdown()]
    assert stages[0] == "vision"
    assert "segment" in stages


def test_claims_merge_image_first_truncated_to_3(tmp_path):
    client = _client()
    main.run_image_forward(
        _image_file(tmp_path),
        "breaking today",
        client=client,
        http_client=dead_http_client(),
        provider=_FakeProvider([]),
    )
    grounding_inputs = [
        kwargs["input"]
        for _, kwargs in client.responses.calls
        if "tools" in kwargs and not kwargs["input"].startswith("fact check:")
    ]
    assert len(grounding_inputs) == 6  # 3 merged claims x support/refute
    assert "vision claim one" in grounding_inputs[0]
    assert "caption claim two" not in "".join(grounding_inputs)


def test_ground_truth_note_reports_exact_hit(tmp_path):
    provider = _FakeProvider(
        [ReverseMatch(source_url="https://orig.com/pic", page_title="t", exact=True)]
    )
    _, tracker = main.run_image_forward(
        _image_file(tmp_path),
        "breaking today",
        client=_client(),
        http_client=dead_http_client(),
        provider=provider,
        ground_truth=["https://orig.com/pic?utm_s=1"],
    )
    assert "found ground truth: yes." in tracker.notes


def test_no_provider_skips_reverse_and_caps_verdict(tmp_path, monkeypatch):
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    client = FakeClient(
        parsed_by_model={
            VisionResult: VisionResult(
                scene_description="s",
                visible_text="",
                people_places_objects=[],
                claims_from_image=["photo shows X"],
                language="en",
                image_role="viral_claim",
            ),
            Argument: Argument(
                points=[ArgPoint(text="p", source_url="https://g.com/x")]
            ),
            RelevanceScores: _all_relevant(),
            Verdict: Verdict(
                verdict="True",
                reason="Looks right.",
                key_source="https://g.com/x",
                language="en",
                confidence="high",
                claim_timeframe=None,
            ),
        },
        outputs=[[search_call([("https://g.com/x", "T", "s")])]] * 2
        + [[search_call([])]],
    )
    card, tracker = main.run_image_forward(
        _image_file(tmp_path), None, client=client,
        http_client=dead_http_client(),
    )
    assert "UNVERIFIED" in card.splitlines()[0]
    assert "reverse search skipped: SERPAPI_KEY not set." in tracker.notes


def test_cli_rejects_text_plus_image(capsys):
    assert main.main(["hello", "--image", "x.jpg"]) == 2
    assert "only one" in capsys.readouterr().err


def test_cli_rejects_caption_without_image(capsys):
    assert main.main(["--caption", "hi"]) == 2
    assert "--caption needs --image" in capsys.readouterr().err


def _mismatch_client():
    client = _client()
    client.responses.parsed_by_model[VisionResult] = _vision().model_copy(
        update={
            "shows_claimed_event": "no",
            "mismatch_reason": "shows a test pattern",
            "consistency_agreement": 3,
        }
    )
    return client


def test_vision_mismatch_short_circuits_without_tied_evidence(tmp_path):
    client = _mismatch_client()
    card, tracker = main.run_image_forward(
        _image_file(tmp_path),
        "floods hit Patna today",
        client=client,
        http_client=dead_http_client(),
        provider=_FakeProvider([]),
    )
    assert "MISLEADING" in card.splitlines()[0]
    assert "media does not show the claimed event" in card.lower()
    assert "media mismatch verdict (agreement 3/3)." in tracker.notes
    assert not [
        kwargs
        for _, kwargs in client.responses.calls
        if kwargs.get("text_format") is Verdict
    ]


def test_vision_mismatch_reaches_judge_brief_when_tied(tmp_path):
    client = _mismatch_client()
    _, tracker = main.run_image_forward(
        _image_file(tmp_path),
        "floods hit Patna today",
        client=client,
        http_client=dead_http_client(),
        provider=_FakeProvider(
            [
                ReverseMatch(
                    source_url="https://old.com/pic", page_title="old",
                    exact=True,
                )
            ]
        ),
    )
    assert "media_mismatch: shows a test pattern" in tracker.notes
    judge_inputs = [
        kwargs["input"]
        for _, kwargs in client.responses.calls
        if kwargs.get("text_format") is Verdict
    ]
    assert judge_inputs
    assert all("Media mismatch" in sent for sent in judge_inputs)
