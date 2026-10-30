# WhatsApp Forward Court

A multi-agent fact-checker for viral WhatsApp forwards.

Forward a suspicious message to a WhatsApp number. Two agents debate it, one arguing it is true and one arguing it is false, each citing live sources. A judge agent returns a short verdict card (True / False / Misleading / Unverified) in your language, ready to share back to the group.

## Inputs
- Text claims
- News links (outlet, date, recycled stories)
- Images (vision + reverse image search)
- Videos (transcript, key frames, on-screen text)

## Built with
- Muse Spark (agents, search grounding, vision, video)
- SAM 3.1 (segmentation)
- Muse Voice Transcribe
- WhatsApp Cloud API

## Status
Work in progress — Meta Global AI Developer Hackathon 2026.
