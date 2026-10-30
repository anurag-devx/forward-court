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
             `python main.py --image path.jpg [--caption "text"]` (gateway later)
- config.py  model names, API client, token-usage tracking
- gateway/   WhatsApp webhook in/out (later)
- intake/    language detection + claim extraction
- media/     image understanding + SAM segmentation crops
- evidence/  search grounding, reverse image search, page fetching
- court/     prosecutor, defender, judge agents
- card/      verdict card formatting
- scripts/   sample-fetch tooling (fetch_samples.py + manifest.json)
- tests/     pytest per module + sample forwards

## API notes
- Responses API throughout (`client.responses.parse` / `.create`).
- Evidence uses the `web_search` tool with `include=["web_search_call.results"]`;
  URLs come only from grounding metadata, never from model text.
- Reverse image search via SerpAPI Lens (key SERPAPI_KEY in .env, never logged);
  exact_matches are origin evidence, visual_matches are context only.
- SerpAPI uploads cap at 500KB; images are downscaled client-side first.

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
