# Authoritative ultracode accounting dogfood — 2026-08-23

## Verdict

The committed live pair at `docs/_witnesses/issue-8168-ultracode-live/pair.json` evaluates to **ABSTAIN**, not a cost win. The benchmark preserves provider billing and spend as authority-labelled unavailable values and identifies the receipt gaps instead of substituting local token estimates.

## Captured run

Run from committed trunk `internal/ultracodebench` at `554a2d76ef`:

```text
fak ultracode bench --pair docs/_witnesses/issue-8168-ultracode-live/pair.json
ULTRACODE PAIRED BENCH: ABSTAIN
accepted effects: single=3 fleet=3 | pass rate: 100.0% / 100.0% | contradictions: 0.0% / 0.0%
critical path: single=16277ms fleet=34576ms (0.47x) | total worker: 16277ms / 62541ms
billed tokens: unavailable (single=provider_usage/0% fleet=provider_usage/0%) | fleet cache-read share: 73.2%
spend: unavailable (single=provider_usage/0% fleet=provider_usage/0%)
accepted/wall gain: -52.9% | accepted/billed-token gain: unavailable | accepted/dollar gain: unavailable
reason: activation_unverified
reason: both modes require independent witness digests
reason: accounting_billed_tokens_unavailable
reason: accounting_spend_usd_unavailable
reason: BUDGET_RECEIPT_INCOMPLETE
```

This is a real repository campaign artifact rather than the offline self-check fixture. Its accounting readout is authoritative about availability: neither subscription-route billed tokens nor spend is reported as zero.

## Defects filed

- #8559 tracks recapturing the pair with current activation and aggregate-budget receipts while keeping unavailable provider axes typed as unavailable.
- #5971 tracks the independent-effect witness seam required before both arms can be called independently witnessed.

No additional defect surfaced in this replay.
