I have everything I need to synthesize this. The three takes converge sharply, the landscape research gives per-platform constraints, and the ground truth ledger marks what's verified versus projected.

# POSITIONING BRIEF — fak organic discovery

## 1. The hook (one sharpest + 2 backups, each fenced)

**THE HOOK:**
> "We cut a poisoned tool result out of the *middle* of a live model's KV cache and left it bit-for-bit identical to a run that never saw it. max|Δ|=0 — not one number differs. No shipped engine does mid-run eviction."

*Honest fence:* Proven against an HF oracle at max|Δ|=0 [SHIPPED]. The deletion *certificate* is self-signed v1 and EvictedCount is self-reported — say so. The "no shipped engine does this" claim is about mid-run eviction specifically (vLLM/SGLang/OpenAI/Anthropic reuse from the front only); don't widen it to "fastest cache."

*Why it's the one:* All three takes independently picked this. It's the only claim that's falsifiable in an afternoon against a reference impl, counterintuitive, and resolves in a single terminal frame. It's a capability claim (survives scrutiny), not a performance claim (invites the strawman fight). Best for r/LocalLLaMA, HN, Lobsters — the only audiences that grade KV mechanics on merit.

**BACKUP A (security-led, for X/AI_Agents/Willison orbit):**
> "Our prompt-injection detector is ~100% evadable. On purpose. It's not the floor — the floor is a lever that was never wired up: the model literally can't call refund_payment."

*Honest fence:* This is self-described and SHIPPED. The two gates (capability lock + result quarantine) are the floor; the detector is explicitly *not*. Keep the "by design" — it's the credibility, not a caveat. Don't say "unbreakable"; say "attacker must beat two independent gates."

**BACKUP B (operational, for selfhosted/golang/Primeagen):**
> "An agent kernel in one ~13MB static Go binary. No go.sum, zero deps. The default-deny gate runs in-process on the same call path as the tool call — no IPC, no sidecar, no second model."

*Honest fence:* The contrast is *operational surface* vs a multi-GB Python/CUDA stack — NOT tok/s. fak is not a faster token engine and doesn't try to be; vLLM/SGLang/llama.cpp win raw throughput. Say that first, before anyone catches it.

---

## 2. The 3 most showable moments (video)

**MOMENT 1 — the deletion proof (the cold open).** Three rows print; two are bit-identical, one carries a rogue `109`. Then: "we cut a poisoned tool result out of the middle of a live model's memory — it's now bit-for-bit identical to one that never saw it." Resolves in frame 1, no waiting for motion. Runs in ~1s offline: `go run ./cmd/deletioncert -selfcheck`. Flash the rogue digit red in post. VERIFIED [SHIPPED].

**MOMENT 2 — the turn-tax race (the thing that keeps them watching).** The live browser demo animates a call-by-call walk: the naive lane ticks `+1 +1 … → 9` while the **fak lane sits frozen at 0**. A counter climbing next to a counter that won't move is the most reliable satisfying-clip primitive there is. *Fence that must survive on screen:* label lanes "naive two-pass loop," show all three lanes, and never imply the 9 is the win over a tuned stack — the tuned lane only saves ~5. `go run ./cmd/turntaxdemo`. VERIFIED [SHIPPED].

**MOMENT 3 — the DENY/ALLOW split (the thesis on screen).** Same machine, two tools: `preflight refund_payment → DENY (POLICY_BLOCK)` in red, `search_kb → ALLOW` in green. With `--explain` it prints an 8-rung ladder ending `DENY DEFAULT_DENY <- winner`. No key, no model, no GPU — runs in one terminal frame. This is "tool call = syscall" made literal. VERIFIED [SHIPPED].

*Asset flag (from Take 2, must verify before publishing):* the OG social-preview image in `C:\work\fak\docs\demos.html` points at `raw.githubusercontent.com/.../visuals/social-preview.png` — confirm that asset exists or the share card 404s.

---

## 3. Per-platform — the ONE angle that fits

| Platform | The one angle | Lead asset |
|---|---|---|
| **r/LocalLLaMA** (PRIMARY) | mechanism-titled mid-run KV eviction, `max|Δ|=0` | deletion-proof GIF embedded + Colab one click away; you first-comment the oracle-parity table (cos=1.000000) AND the tuned 4.1x |
| **r/AI_Agents** | "treat the tool call like a syscall; the lever was never wired up" | copy-pasteable 60s `preflight refund_payment → DENY` |
| **r/selfhosted** | operational contrast: one static binary as agent gateway vs Python/CUDA stack (NOT benchmarks) | deployment specifics; repo link goes in the *weekly thread*, not standalone |
| **r/golang** | pure-Go transformer verified per-layer against HF (cos=1.000000), CUDA decode at llama.cpp parity, no go.sum | engineering story; security thesis only in comments |
| **r/netsec** | the two-gate containment architecture; why detection-as-floor is the wrong abstraction | hosted writeup (not bare repo), "disclosure: I wrote this," explicit threat model |
| **r/MachineLearning** | reproducibility: byte-identical across 4 GPU backends / 4 OSes, every number → commit | weekend `[P]` post; method first, NOT the speedup |
| **Show HN** | "treat the model as untrusted, the tool call as a syscall" (mechanism surprise, NOT "security") | 60s `preflight → DENY`; your own first comment is the prosecution |
| **Lobsters** | idea-first essay: "the safety boundary and the reuse boundary…" as worked example (needs aged account — see fence #3) | the bit-exact KV eviction + ed25519 cert writeup |
| **X (Willison/MCP-security cluster)** | a lethal-trifecta thread: capability lock + result quarantine break it structurally; tag the *vocabulary*, not the product | 60s DENY proof as first reply |
| **YouTube — Fireship** | hand him a pre-cut 100s script: one thesis + DENY frame + 4.1x | — |
| **YouTube — LiveOverflow/Hammond/IppSec** | a "break my quarantine gate" CTF challenge repo (attacker controls the tool result, win = fire the destructive call) | — |
| **YouTube — Cole Medin/Berman** | "drop fak in front of your existing Claude Code/Cursor agent, no code changes" 90s demo | — |
| **YouTube — Primeagen** | "13MB Go static binary, no go.sum, vs multi-GB Python/CUDA stack" + the 0/29-novel honesty | — |

---

## 4. The 3 framings to AVOID (read as hype)

**AVOID 1 — any "x" multiple that isn't the tuned-baseline number, and *especially* the naive 8.8–9.7x or the projected ~60x / "agent city."** The naive denominator is a re-prefill strawman no one runs in production; a perf-literate reader divides by real SOTA and writes you off as dishonest — and leading with it means you *chose* the strawman on purpose. Projections stated beside measurements are the single most reliable vaporware tell. Rule: lead with **~1.5–4.1x vs tuned warm-cache**, label every projection DESIGN TARGET on the same line, or show no multiplier at all.

**AVOID 2 — "the safety boundary and the reuse boundary are the SAME boundary" as a headline thesis.** Co-location in one binary is not unification, and a skeptic reads it as a slogan reverse-engineered from that co-location. Worse, the fences quietly admit the reuse half pays off only for self-hosted, read-heavy fleets where ~1% writes flip it negative (a narrow niche) while the safety half is universal — two features with different applicability bolted into one "aha." (It survives only as a Lobsters/HN *essay* worked example, never as a launch headline.)

**AVOID 3 — leaning on "kernel" / "syscall" as if the metaphor is load-bearing.** A real syscall boundary is enforced by a privilege ring userspace physically can't cross; fak's gate runs *in the same process* as the agent loop. The shipped thing — an in-process default-deny capability check on the tool-call path — is defensible and good; the "kernel" packaging makes a skeptic trust it *less* by implying the metaphor is doing work the engineering isn't. Use "syscall" as a one-line intuition pump (Show HN title works), but never let it stand in for the mechanism, and never claim ring-style isolation you don't have.

*Cross-channel through-line:* fak's entire credibility lives in its fences and its zeros (max|Δ|=0, 0/29 novel, DENY-by-structure, detector-evadable-by-design). Every target audience here is allergic to overclaiming and rewards exactly this self-skepticism. Lead with the fence — it is the hook, not the caveat. The one move that disarms all three hostile reflexes (AI-slop, naive-benchmark, security-overclaim) at once: **make your own first comment the prosecution.**

*Unverifiable / must-flag-on-use:* the ~60x and "agent city" frontier numbers (DESIGN TARGETS, not measured); all power/energy/$/kWh/GPU-hour figures (SIMULATED — no power meter on the box); the deletion certificate's EvictedCount (self-reported, self-signed v1).
