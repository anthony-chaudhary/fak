---
title: "Work-done source provenance"
description: "fak info classifies runtime work with the versioned fak.info.work-source/1 taxonomy. Collection records the source before rendering;"
---
# Work-done source provenance

`fak info` classifies runtime work with the versioned `fak.info.work-source/1` taxonomy. Collection records the source before rendering; the UI does not infer provenance from labels.

| Stable ID | Owner | Meaning |
|---|---|---|
| `provider_cache` | provider | Input was loaded from the provider prefix cache. |
| `fak_response_reuse` | fak | A response-memo/vDSO hit served work without another model call. |
| `inline_tool_local` | fak | A read-only tool result was served inline without another model call. |
| `context_reduction` | fak | Compaction or elision removed input before provider inference. |
| `fak_prefix_reuse` | fak | Fak reused a KV/prefix token span. |
| `cold_direct` | provider | The observed path had no measured reuse or reduction. |
| `unknown` | unknown or fak | A total exists but its producer did not provide a finer source, or the source block is absent. |

Every source record carries owner, disposition (`loaded`, `served`, `reduced`, or `unknown`), event-count availability, token effect, call effect, evidence class, and an exclusivity group. Token effects in `input_token_equiv_owner/v1` partition the WORK DONE token total. Call effects in `avoided_model_call_path/v1` partition the avoided-call total. These are different units and must never be added together.

The gateway preserves response-memo and inline-served counts separately while retaining the older combined avoided-call total for compatibility. If a combined total is larger than its classified children, the remainder is emitted as `unknown`; it is not silently assigned to a mechanism. A present all-zero attribution block emits `cold_direct`, while an absent block emits `unknown` because lack of telemetry is not proof of a cold request.

Use `fak info` and open the **Cache** tab for the hierarchy, or query the same records with:

```console
fak info --gateway-url http://127.0.0.1:PORT --json
```
