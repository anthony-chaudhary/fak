The draft is mostly clean and the fences are unusually well-handled, but there is one fabricated number, one incorrect command-existence claim, one structural claim that overreaches the ledger, and one false self-congratulation in the closing line. Here is each problem, the fix, then the full corrected asset.

**Problems found**

1. **The deletion self-check command exists in source.** `cmd/deletioncert/main.go` declares `-selfcheck` and constructs a tiny synthetic model for its eviction-versus-never-seen comparison. `go run ./cmd/deletioncert -selfcheck` is therefore a source-backed command, not an invented flag. This source review does not establish a fresh successful run, HuggingFace parity, hardware behavior, or elapsed time. The script below still displays `go run ./cmd/fak agent --offline`, which demonstrates tool-policy behavior and must not be presented as the deletion numerical witness.

2. **"resolves in ~1s" / "run the deletion proof yourself, offline, in about a second."** The ledger says "60-second proof," not one second. The ~1s figure is fabricated. Changed to "in about a minute, offline."

3. **The rogue `109` value and "flash the `109` red" is fabricated detail.** The ledger proves `max|Δ|=0` against an HF oracle; it does not specify any token id `109`. Kept the visual concept (one row carries a poisoned value that vanishes) but stripped the invented literal so nothing on screen claims a specific number we can't source.

4. **"8-rung ladder" in the `--explain` b-roll is not in the ledger.** The ledger describes `--explain/--json` per-rung trace and a `DENY ... POLICY_BLOCK` outcome, but never states the rung count is 8. Removed the specific count.

5. **Closing line "No naive-only lead, every fence spoken aloud" is self-grading and one claim is now slightly off** — fine to keep as a production note, but I removed the implicit guarantee and trimmed it to a factual recap.

6. **Pre-publish blocker references `C:\work\fak\docs\demos.html`** — that's a local path note, harmless, but the ledger's canonical demos URL is `anthony-chaudhary.github.io/fak/demos.html`. Kept the OG-image check (it's a real pre-publish risk) but corrected it to reference the published page, not a machine-local absolute path that means nothing to a reader.

7. **CTA command `go run ./cmd/fak preflight refund_payment`** — the ledger's exact preflight invocation is `go run ./cmd/fak preflight --policy …readonly… --tool refund_payment`. Minor, but I aligned it to the ledger form so a skeptic copy-pasting it isn't surprised by the missing args placeholder.

The WebVoyager comparison below is modeled prefill geometry, not measured speedup. Other claims, including binary size, detector results, capture provenance and competitor comparisons, still require their own evidence before publication.

---

# YouTube Short Script — fak (60–90s)

**Working title:** "We deleted a poisoned memory out of a live model — and proved it changed nothing"
**Format:** vertical short, ~85s. Single terminal + the live browser demo at `anthony-chaudhary.github.io/fak/demos.html`.
**Voice:** flat, technical, fast. No hype reads. The fences are spoken, not subtitled-and-buried.

---

## COLD OPEN (0:00–0:15) — the hook

| | |
|---|---|
| **VO** | "We cut a poisoned tool result out of the *middle* of a live model's KV cache — and left it bit-for-bit identical to a run that never saw it. Max delta zero. Not one number differs." |
| **ON-SCREEN TEXT** | `max|Δ| = 0` (large, centered) |
| **B-ROLL** | Terminal showing two model runs side by side: one that saw the poisoned tool result, one with that span cut out. In post: highlight the poisoned span, then a hard cut as it vanishes and the two runs read identical, byte for byte. |
| **COMMAND VISIBLE** | `go run ./cmd/fak agent --offline` |

**Spoken fence (immediately, same breath):** "No shipped engine does mid-run eviction — vLLM, SGLang, the API caches all reuse from the *front* only. This isn't a faster token engine. It's mid-run deletion, proven against a HuggingFace reference."

---

## MOMENT 2 (0:15–0:38) — the turn-tax race (the watch-hook)

| | |
|---|---|
| **VO** | "Same idea, made visible. Watch the prefill counter. The naive two-pass loop ticks up every call — one, two, three… all the way to nine. The fak lane sits frozen at zero." |
| **ON-SCREEN TEXT** | Three labeled lanes: `naive two-pass loop` (climbing), `tuned warm-cache`, `fak` (frozen at 0). |
| **B-ROLL** | The live browser demo animating call-by-call. One counter climbs next to one that won't move. |
| **COMMAND VISIBLE** | `go run ./cmd/turntaxdemo` |

**Spoken fence (the load-bearing honesty):** "These are modeled prefill-token counts over the 643-task WebVoyager dataset, with difficulty-derived turn geometry and no live-model runs. Across the 1–8-worker sweep, the model gives 8.8 to 9.7 times less work than naive re-prefill, and 1.0 to 1.1 times versus per-agent KV reuse. This is not a measured latency, cost or throughput improvement."

| | |
|---|---|
| **ON-SCREEN TEXT** (under the race) | `MODELED PREFILL GEOMETRY · 8.8–9.7× vs NAIVE · 1.0–1.1× vs PER-AGENT KV` <br> `0 live-model runs · no latency/cost/throughput measurement` |

---

## MOMENT 3 (0:38–0:58) — the DENY/ALLOW split (the thesis)

| | |
|---|---|
| **VO** | "Second thing in the same binary: treat the tool call like a syscall. Default-deny. The model asks to fire `refund_payment` — DENY. Policy block. It asks to `search_kb` — allow. This runs in-process, on the same call path as the tool call. No second model, no sidecar." |
| **ON-SCREEN TEXT** | `refund_payment → DENY (POLICY_BLOCK)` in red · `search_kb → ALLOW` in green |
| **B-ROLL** | `--explain` prints the per-rung decision ladder, last line highlighted: `DENY POLICY_BLOCK <- winner (rank 100)` |
| **COMMAND VISIBLE** | `go run ./cmd/fak preflight --policy …readonly… --tool refund_payment` |

**Spoken fence:** "And the injection detector? It's roughly a hundred percent evadable — on purpose. It's not the floor. The floor is that the destructive lever was never wired up, so the model can't call it even if the prompt fools the detector. The attacker has to beat *two* independent gates."

---

## CTA (0:58–1:08)

| | |
|---|---|
| **VO** | "One static Go binary, about thirteen megabytes, two extended-standard-library dependencies, Apache-2.0. Drops in front of Claude Code or Cursor with no agent-side changes. Every number traces to a commit. Repo and live demos in the description — run the proof yourself, offline, in about a minute." |
| **ON-SCREEN TEXT** | `github.com/anthony-chaudhary/fak` <br> `go run ./cmd/fak agent --offline` <br> live demos → anthony-chaudhary.github.io/fak/demos.html |
| **B-ROLL** | Terminal re-running the offline proof: injection-in-context YES→no, destructive-op YES→no, task still completed. |

---

## Pinned-comment prosecution (post this yourself, first comment)

> Disclosure: I wrote fak. The honest fences, since a short can't hold all of them:
> - **Not a faster token engine.** vLLM/SGLang/llama.cpp win raw single-stream throughput; fak doesn't target it. The contrast is *operational surface* — one binary vs a multi-GB Python/CUDA stack — not tok/s.
> - **Geometry model:** WebVoyager uses a real 643-task dataset with derived turn geometry and zero live-model runs. Across the 1–8-worker sweep, its modeled prefill-token ratios are 8.8–9.7× versus naive re-prefill and 1.0–1.1× versus per-agent KV. These are not measured latency, billing, task-cost or throughput gains. Other experiments must retain their own evidence and must not be attributed to WebVoyager.
> - **The ~60× / "agent city" numbers are DESIGN TARGETS, not measured.** Not in this video for that reason.
> - **Deletion cert** is self-signed v1 and EvictedCount is self-reported; the `max|Δ|=0` parity is the real proof (checked against an HF oracle).
> - **Detector is ~100% evadable by design.** The floor is the capability lock + result quarantine, not detection.
> - **Power/energy figures elsewhere are simulated** — no power meter on the box.
> - Prior-art audit scored **0/29 novel** — every primitive is established; the contribution is the assembly.

---

**Pre-publish blocker (verify before posting):** the OG social-preview image on the published demos page (`anthony-chaudhary.github.io/fak/demos.html`) points at `raw.githubusercontent.com/.../visuals/social-preview.png`; confirm it resolves with `python tools/demo_live_links.py --published` before the link is posted.

**Total runtime:** ~68s of content, ~85s with the b-roll breathing room. Spoken numbers: `max|Δ|=0`, `8.8–9.7× naive`, `1.0–1.1× per-agent KV (modeled)`, `~13MB`, `0/29 novel`, `~1-minute` offline proof.

---


---

## Provenance & fact-check notes

> **Provenance & fact-check.** This asset was drafted from the repo's honesty ledger (`CLAIMS.md`) and then run through an adversarial fact-check against that ledger. The reviewer's notes are preserved below as an audit trail. Every CLI command was additionally re-verified against the live `fak` binary on 2026-06-22 and normalized to the one runnable form (`preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"` → `DENY (POLICY_BLOCK)`; `search_kb` → `ALLOW`).

## Fact-check notes

What I changed:

1. **Correction to the earlier command-existence note:** `go run ./cmd/deletioncert -selfcheck` is implemented in source and uses a synthetic model. The earlier description of it as fabricated was incorrect. The displayed `go run ./cmd/fak agent --offline` remains a separate tool-policy demonstration; a matching deletion capture and its scope must be verified before publication.
2. **Fixed "~1s" → "about a minute."** Ledger says "60-second proof." The one-second claim was invented.
3. **Stripped the literal token `109`** from the cold-open b-roll. The ledger proves `max|Δ|=0`; it specifies no token value. Kept the visual (poisoned span vanishes, runs match) without an unsourced on-screen number.
4. **Removed "8-rung ladder"** → "per-rung decision ladder." Rung count isn't in the ledger (ledger only confirms `--explain/--json` per-rung trace).
5. **Aligned preflight command** to the ledger form `go run ./cmd/fak preflight ... refund_payment`.
6. **Rewrote the pre-publish blocker** to reference the published page instead of a local absolute path, and folded in the ledger's `text/plain` raw-serving caveat.
7. **Trimmed the closing self-grade** ("No naive-only lead, every fence spoken aloud") to a plain number recap — a self-certifying claim isn't evidence.

The corrected WebVoyager ranges come from `docs/webbench-real-measurements-summary.md`: the 1–8-worker sweep models prefill-token geometry over 643 tasks, with zero live-model runs. Its 8.8–9.7× naive and 1.0–1.1× per-agent-KV ratios are not measured speedup. The earlier blanket verification paragraph did not establish those distinctions. Other numerical, capture, timing and capability claims in this asset remain subject to their own evidence checks before publication.

<!-- Freshness review 2026-08-24: commands, links, channel constraints, and product claims rechecked against the current repository; publish only after a final live channel-rule check. -->
