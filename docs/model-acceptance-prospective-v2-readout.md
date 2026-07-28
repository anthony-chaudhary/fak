---
title: "Prospective exact-model v2: infrastructure HOLD"
description: "Every preregistered attempt reached the authenticated provider boundary and hit the same weekly-limit rejection, so v2 is infrastructure evidence only."
---

# Prospective exact-model v2: infrastructure HOLD

Issue: [#4851](https://github.com/anthony-chaudhary/fak/issues/4851)  
Parent: [#4845](https://github.com/anthony-chaudhary/fak/issues/4845)  
Production-readiness parent: [#4633](https://github.com/anthony-chaudhary/fak/issues/4633)

## Verdict

**HOLD.** This is a prospective provider/infrastructure witness, not model-capability evidence. The fixed campaign was declared and published in commit `8c5edc9c3249` before launch. Every scheduled attempt reached the authenticated provider boundary, and every attempt received the same typed weekly-limit rejection. The declaration allows no replacements, so none were made.

## Preregistered envelope

- Corpus: `top3-prospective-sentinel-v2`.
- Exact IDs: `claude-opus-4-8`, `claude-sonnet-4-6`, and `claude-haiku-4-5-20251001`.
- Two task classes per model: read-only multi-record synthesis and typed transient retry recovery.
- Three repetitions per class per model: 18 fixed attempts total.
- Match contract: exact complete sentinel line after line-ending normalization; embedded substrings and padded lines do not match.
- Stop/replacement contract: stop after 18 attempts; do not replace provider/infrastructure, harness, or output-quality failures.
- Threshold: 100% eligible success; any unexplained eligible failure retains HOLD.

## Observed result

| Exact model | Scheduled | Eligible capability observations | Provider/infrastructure failures | Verdict |
|---|---:|---:|---:|---|
| `claude-opus-4-8` | 6 | 0 | 6 | HOLD |
| `claude-sonnet-4-6` | 6 | 0 | 6 | HOLD |
| `claude-haiku-4-5-20251001` | 6 | 0 | 6 | HOLD |

All 18 immutable streams contain the provider's typed `rate_limit` / HTTP 429 weekly-limit rejection. The selected seat reported a reset at **2026-07-16 06:00 America/Los_Angeles**. Since there were zero eligible model outputs, this run neither promotes nor negatively grades any exact model's capability. A fresh campaign after the reset requires a new committed declaration; these attempts cannot be silently replaced.

## Immutable evidence

Raw streams remain in operator scratch and are not committed because they include provider/session metadata. Scrubbed cryptographic bindings:

- Declaration commit: `8c5edc9c3249`.
- Declaration SHA-256: `18f4d5e8e0b94dd36cf269adcfda1bf288ffe00f9fffcf8917d83c6ea8d99182`.
- Canonical raw-manifest SHA-256 (18 JSONL entries): `22d87a91c37f925e381bc9c1998530efc2e630322b23440830a49a784b555fb7`.
- Completed report SHA-256: `1d54cbba38ce3ca3fb5eee4f27ec93a47c9c594fe268632f43d3e62cba78517f`.
- Decision SHA-256: `bc68728c58362987204d34c8b8a16fd52aab5a2eaf05570b12987d9402540852`.
- Runner exit: `4` (`HOLD`).

The retrospective corrected refold remains separate historical evidence and is not counted as this prospective campaign.
