# WhatsApp Forward Court

Multi-agent fact-checker for viral WhatsApp forwards. A user forwards a message;
two agents debate it (one argues true, one argues false) using cited evidence;
a judge agent returns a verdict card: True / False / Misleading / Unverified,
in the user's language.

## Stack
- Python 3.11, FastAPI, pytest
- OpenAI SDK pointed at https://api.meta.ai/v1
- API key from .env as MODEL_API_KEY (python-dotenv)

## Models
- muse-spark-1.3: agents, judge, search grounding, image + video understanding
- muse-spark-1.3-contributor: same uses when contributor mode is on (default;
  set FC_CONTRIBUTOR=0 in .env for the standard model)
- sam-3.1: object segmentation in images/video frames
- muse-voice-transcribe-1.0: audio and voice notes
- Model names live in config.py only; never hardcode them elsewhere.

## Environment
- Always use the project venv at .venv. Activate before running anything.
- Install packages only inside .venv; update requirements.txt after every install.
- Never install packages globally.

## Structure
- main.py    CLI entry: `python main.py "<forward text>"` or
             `python main.py --image path.jpg [--caption "text"]` or
             `python main.py --video path.mp4 [--caption "text"]` (gateway later)
- config.py  model names, API client, token-usage tracking
- gateway/   WhatsApp webhook in/out (later)
- intake/    language detection + claim extraction
- media/     image/video understanding + SAM crops + audio transcription
- evidence/  search grounding, reverse image search, page fetching
- court/     prosecutor, defender, judge agents
- card/      verdict card formatting
- scripts/   sample tooling (collect_factchecks.py, fetch_samples.py + manifest.json)
- dates.py   shared date extraction (evidence + intake recency cues)
- tests/     pytest per module + sample forwards

## API notes
- Responses API throughout (`client.responses.parse` / `.create`).
- Evidence uses the `web_search` tool with `include=["web_search_call.results"]`;
  URLs come only from grounding metadata, never from model text.
- Reverse image search via SerpAPI Lens (key SERPAPI_KEY in .env, never logged);
  exact_matches are origin evidence, visual_matches are context only.
- SerpAPI uploads cap at 500KB; images are downscaled client-side first.
- Video: .mp4 only, 16MB WhatsApp cap; Files API upload (purpose=user_data)
  referenced by file_id, deleted after understanding; ffmpeg scene frames
  (max 3) reuse the image reverse path; audio extracted to WAV and sent to
  POST /asr/transcribe (muse-voice-transcribe-1.0), language via intake.
- Lookup stage: `fact check: <claim>` per claim, trusted domains only
  (TRUSTED_FACTCHECK_DOMAINS in config.py); skip with --no-factcheck.
- Judge votes 3x (majority wins, tie → Unverified); confidence from agreement.
- Source tiers in config.py: tier-3 (UGC) can never be key_source; prefer tier-1.
- Relevance gate (evidence/relevance.py): one structured call scores each item
  0/1/2 per claim; 0s are dropped before debate, claims left bare yield Unverified.
- ClaimReview: tier-1 JSON-LD ratings matching a claim decide the verdict
  deterministically (high confidence, skips debate + votes) unless two tier-1
  ratings contradict; the card adds "Already fact-checked by <outlet>".

## Rules
- Evidence first: both debating agents get the same evidence list and may cite only from it.
- Any argument point without a source_url from the evidence list is dropped by the judge.
- Verdict logic: True/False only on strong cited evidence; Misleading when media is
  real but context is wrong; Unverified otherwise. Never claim "deepfake" without strong evidence.
- Agents exchange structured JSON (use structured output), not free text.
- Keep modules small; one responsibility per file.
- Every module gets a pytest test.

## Security
- Never print, log or commit API keys or tokens.
- .env stays in .gitignore.

## Workflow
- Show a plan before writing code; wait for approval.
- Make small changes; run tests after each change.
- Do not add co-author lines or tool attribution to commits.
