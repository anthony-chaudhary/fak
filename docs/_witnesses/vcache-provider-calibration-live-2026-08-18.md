# Live provider warmth-calibration witness — 2026-08-18

**Verdict:** a real two-turn Codex session in a temporary non-FAK Git repository reported OpenAI cache reads, and the new calibration ingest/status path persisted a dated `fresh` OpenAI/model row.

- Provider/model: `openai` / `gpt-5.6-sol`.
- First turn: `28,473` input tokens, `3,456` cached input tokens.
- Resumed turn: `39,739` input tokens, `3,456` cached input tokens.
- Ingest: `fak vcache calibration-record --provider openai --model gpt-5.6-sol --source probe:codex-non-fak-repo-2026-08-18 --telemetry combined.jsonl --output calibration.jsonl`.
- Readout: `fak vcache calibration-status --file calibration.jsonl --providers openai --json` returned `ok: true`, `state: fresh`.
- Structured artifact: [`vcache-provider-calibration-live-2026-08-18.json`](vcache-provider-calibration-live-2026-08-18.json).

This is deliberately narrow evidence. It does not claim Anthropic coverage, observed writes, measured TTL/minimum-prefix constants, or that the estimator now steers runtime cache policy. Those remain requirements of #1497.
