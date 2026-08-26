---
title: "Ponytail Promptfoo pinned comparator reproduction (#6686)"
description: "This packet reproduces the benchmark definitions at DietrichGebert/ponytail@2ed6c52c9d7e5e56942508591085fd45dea277d3 with Promptfoo 0.122.0."
---
# Ponytail Promptfoo pinned comparator reproduction (#6686)

This packet reproduces the benchmark definitions at `DietrichGebert/ponytail@2ed6c52c9d7e5e56942508591085fd45dea277d3` with Promptfoo `0.122.0`. It is a comparator reproduction, not a fak value claim.

## Reproduce

```powershell
git clone https://github.com/DietrichGebert/ponytail.git $env:TEMP/ponytail-6686
git -C $env:TEMP/ponytail-6686 checkout 2ed6c52c9d7e5e56942508591085fd45dea277d3
fak armbench ponytail-promptfoo --source $env:TEMP/ponytail-6686 --out docs/_witnesses/armbench-ponytail-promptfoo-2026-08-14 --execute
```

`summary.json` hashes every benchmark-defining config, prompt, arm, skill, and deterministic assertion. Each config/provider declaration is attempted separately with caching disabled and concurrency one. Duplicate model declarations in distinct upstream configs remain distinct attempts because their sampling settings differ. The arm set is also preserved exactly: only the Anthropic config declares all three baseline/Caveman/Ponytail arms; OpenAI and Gemini configs declare baseline/Ponytail only.

## Results

- `gpt-5.4-mini` completed both separately declared configs: 10 Promptfoo rows per config (5 tasks × 2 arms), with raw outputs and telemetry in the corresponding JSON files.
- `gpt-5.5` completed: 10 rows. `analysis.json` contains per-task/per-arm correctness and output-token values derived directly from raw Promptfoo results.
- `gpt-4.1-mini` was attempted in each declaration; the configured endpoint returned HTTP 400 for all rows. The result JSONs preserve all errors and zero token telemetry.
- All three Anthropic declarations were attempted separately and were explicitly unavailable because the execution environment had no `ANTHROPIC_API_KEY`.
- Both Gemini declarations were attempted separately and were explicitly unavailable because the execution environment had no `GOOGLE_API_KEY`.
- No substitution manifest was used and no retired model was silently replaced.

## Honest upstream delta

The upstream June 12 published aggregate covers Claude 4.0-era labels and does not publish per-task raw outputs. The pinned August configs declare newer/different model IDs. Therefore there is no like-for-like numeric upstream delta to report. `analysis.json` states this basis and reports current provider/model cells independently; it makes no fak comparison or value claim.

Files named `*.stdout.txt` and `*.stderr.txt` preserve command output. Completed and provider-error cells preserve Promptfoo-compatible raw result JSON. `summary.json` is the manifest and attempt ledger; `analysis.json` is the per-task/model correctness/token rollup.
