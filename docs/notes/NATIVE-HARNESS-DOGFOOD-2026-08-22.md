# Native harness live-model and dogfood readout — 2026-08-22

## Verdict

The bounded microharness now reaches a configured real model through FAK's shared provider HTTP planner, records provider-reported usage in a private JSONL ledger, and folds weekly invocation counts. A real run on the sanctioned `fak-realmodel` node completed the #7793 repository-work goal with all six bounded completions returning non-empty evidence.

Single-arm native execution does **not** become the default in this change. The measured offline acceptance comparison proves behavioral parity with the existing FAK arm, but it is not an independent live repository-work A/B against the current harness. The existing default remains the comparison path until that stronger gate exists.

## Problem and value frame

- **For:** maintainers deciding whether FAK's native bounded harness is ready for routine repository work.
- **Problem:** the deterministic spine did not use a real model and adoption had no durable count.
- **Today:** quality and usage could only be claimed from fixture output.
- **Better because:** the same bounded host accepts the existing shared provider planner, captures token usage, and emits model evidence without importing child transcripts into the root.
- **Witness:** focused tests plus the live node receipt summarized below.

## Live repository-work dogfood (#7793)

Command shape (credential-free local gateway on the sanctioned node):

```text
FAK_PROVIDER_MAX_TOKENS=8 microharnessdemo \
  -live -base-url http://127.0.0.1:8080/v1 -model qwen2.5:14b \
  -goal "Review issue 7793: select the next real fak native-harness repository task and state its independent proof boundary." \
  -json -ledger microharness-usage.jsonl
```

Observed receipt:

- mode/model: `live` / `qwen2.5:14b`
- bounded completions: `6`; non-empty model responses: `6`
- provider usage: `239` prompt tokens, `48` completion tokens
- wall time: `19,074 ms`
- recursion: depth `2`; child budgets `1`, `2`, and `3` turns; the ambiguous irreversible task was refused at admission
- root fold: typed receipts only; model snippets were bounded to 240 bytes per child
- selected next task: scoped repository read/write tools with an independent build-plus-affected-tests proof boundary

The ledger row contained only schema, timestamp, mode, model, outcome, completion count, and token counts; it contained no prompt, repository path, hostname, or credential.

## Default acceptance gate

The current `fak agent` default runs the FAK arm and baseline arm for an A/B report. `fak agent --native` runs that exact FAK arm once. On the deterministic offline task both paths produced the same FAK-arm receipt:

| Metric | Current harness FAK arm | Native single arm |
|---|---:|---:|
| Task completed | true | true |
| Turns | 7 | 7 |
| Tool calls / errors | 6 / 0 | 6 / 0 |
| Prompt / completion tokens | 5701 / 184 | 5701 / 184 |
| Injection retained | false | false |
| Destructive effect executed | false | false |

This establishes implementation equivalence, not a measured live-repository superiority result. Therefore the promotion gate is **HOLD** and the current default is preserved.

## Defects and follow-ups

No defect surfaced in the live adapter, recursive bounds, receipt fold, or private usage ledger. The acceptance audit did surface one missing evidence class: an independent paired live repository-work benchmark between the native single arm and the current external harness. That work must be tracked before changing the default.
