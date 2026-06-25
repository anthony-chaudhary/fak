---
title: "Making fak attractive to mobile, edge, and IoT providers"
description: "A grounded go-to-market strategy for fak on phones, edge gateways, and IoT. Sell the on-device capability gate and the tamper-evident audit trail, not inference speed. Every claim cites a shipped artifact or a dated external source."
date: 2026-06-24
status: strategy note (not a claim doc); proposes work, cites what is real today
---

# Making fak attractive to mobile / edge / IoT providers

**TL;DR.** Phones and edge boxes already have a fast on-device model. What they
lack is the gate that treats that model as untrusted: default-deny on every tool,
poisoned results held out of memory, and a tamper-evident log that already matches
the EU audit law landing in August 2026. fak is that gate in one pure-Go binary
that runs the same on a Raspberry Pi as on a datacenter GPU. Lead with the fence
and the law, never with tokens-per-second. The capability is shipped and
hardware-independent today; the mobile bindings and the published arm64 binary are
the net-new work, and they are small.

**The one move.** Don't sell mobile or IoT providers a faster on-device model
engine. They already have one: Gemini Nano in AICore, Apple's Foundation Models
framework, MLC, llama.cpp, ONNX Runtime. Sell them the half of the on-device-agent
stack those platforms deliberately left thin. That half is an adversarial
capability gate the model can't argue past, plus a tamper-evident audit trail that
already matches the law landing in August 2026. It is pure-Go, CPU-only, ~13 MB,
and runs the same on a Raspberry Pi as on a GPU server. The compute half is not fak's
fight, and the strategy below never pretends it is.

This note is honest about the gap. fak's security, audit, and gate story is
hardware-independent and shipped today. fak's compute story is not optimized for
constrained hardware and has no mobile bindings yet. The whole strategy is built so
fak wins on the half it already owns.

---

## 1. Why the obvious pitch is the wrong pitch

The reflexive move is "port fak's kernel to phones as a fast on-device LLM runtime."
fak's own front door tells you not to:

> `fak` is not a faster model server, and doesn't try to be. SOTA engines (vLLM,
> SGLang, llama.cpp) win raw throughput and prefix-cache reuse; that's their job.
> (`README.md`)

On a phone the engines that win throughput are even more entrenched: Qualcomm
Hexagon, Apple's Neural Engine, Google's Tensor TPU, and the vendor runtimes bound
to them. fak would lose that fight on day one and for the right reasons.

The band the platforms leave empty is enforcement, not speed. As of 2026 both big
mobile platforms ship an on-device model with a typed tool/intent contract. Both
leave the hard part in-process and under-specified: adversarial enforcement against
a model that can be jailbroken by its own input.

- **Android / Gemini Intelligence** runs Gemini Nano in the AICore system service.
  It uses "an intelligent request system, so granting a high-level permission (such
  as writing files) automatically authorizes all related sub-tools."
  ([Android developers — Gemini Nano](https://developer.android.com/ai/gemini-nano),
  [Yaabot — Gemini Intelligence](https://yaabot.com/41610/gemini-intelligence-android-ai-agent/))
  That auto-authorization from one coarse grant is exactly the hole fak's
  default-deny, per-tool capability floor closes.
- **Android's own risk docs name the problem.** They call it "excessive agency" and
  prescribe least-privilege "to every tool and function the LLM can access or
  trigger." Skip it and a malicious prompt triggers actions repeatedly: battery
  drain, data overage, local resource exhaustion.
  ([Android — Mitigate excessive agency](https://developer.android.com/privacy-and-security/risks/ai-risks/excessive-agency))
- **Apple Intelligence** ships App Intents that are "tight, typed," where
  "open-ended autonomous behavior outside the intent contract is blocked by design."
  ([buildmvpfast — Apple Intelligence Agent SDK 2026](https://www.buildmvpfast.com/blog/apple-intelligence-agent-sdk-third-party-developers-guide-2026))
  Apple isolates where data is processed (Private Cloud Compute, on-device silicon)
  and which intents exist. But the enforcement is the in-process intent resolver
  trusting its own dispatch. It is not a gate the model passes through.

The seam: these platforms isolate where the model runs (Android's pKVM earned SESIP
Level 5; Apple's PCC) and declare which tools exist. Nobody ships the gate that
adversarially decides what the on-device agent is allowed to do, fails closed, and
writes a tamper-evident line about it. That gate is fak, and it is
hardware-independent by construction.

---

## 2. What is actually real today (the honest inventory)

Everything in this column is shipped and witnessed in-repo. The strategy below
leans only on these.

| Capability | Status | Why it matters on the edge | Evidence |
|---|---|---|---|
| **Pure-Go static binary, zero external deps, `CGO_ENABLED=0`** | shipped | The whole gate is one ~13 MB file with nothing to port or link. Drops onto any box. | `release-artifacts.yml:56`, `go.mod` (stdlib only), `README.md` (~13 MB distroless) |
| **arm64 already a first-class target** | shipped (Apple Silicon) | The primary bench node is arm64 (M3 Pro); the kernel's gates are proven on arm64 + x86_64, bit-identical. | `HARDWARE-MATRIX.md:10-15,33`; `release-artifacts.yml:48-49` (darwin/arm64) |
| **CPU-only is the default; GPU is opt-in build tags** | shipped | No GPU, no NPU, no accelerator needed to run the gate or the small model. | `internal/compute/cpuref.go:21` (registers unconditionally); Metal/CUDA/Vulkan behind `-tags` |
| **Gate is a swappable layer in front of any engine** | shipped | A phone-native runtime (CoreML/TFLite/MLC/Ollama) sits *behind* fak; the gate is in-process, O(1), no IPC hop. | `--base-url` proxy mode; `EngineDriver.Complete(ctx,*ToolCall)` (`internal/abi/registry.go`); `gateway.Config.EngineID` |
| **Default-deny capability floor + closed refusal vocabulary** | shipped & live-tested | The model can't talk past the gate; refusals are structured codes, not prose. Booby-trapped policy walled 5/5. | `README.md` (the one move; `DENY (POLICY_BLOCK)`); `POLICY.md`; `docs/repro-packet.md` |
| **Result quarantine (write-time admission gate)** | shipped | Poisoned/injected tool results are held out of the model's memory at the boundary, critical when the "tool" is a sensor or actuator on an untrusted bus. | `README.md`; `internal/model/kv.go` (bit-exact `Evict`) |
| **Tamper-evident audit journal: append-only, SHA-256 hash-chained, per-row durable** | shipped | Structurally matches EU AI Act Article 12 (below). | `internal/journal/journal.go:1-26,319,334-340`; `docs/proofs/journal.md`; `TestVerifyDetectsTampering` |
| **Deletion certificates (ed25519-signed, bit-exact)** | shipped | Prove a result *left* device memory, the artifact a regulated device needs to prove "we forgot this." | `docs/proofs/deletioncert.md`; `TestMintVerifyRoundTrip` |
| **Determinism / reproducibility (cosine=1.0 forward parity)** | shipped | Same input → same decision, byte-for-byte, offline. Safety-critical buyers require this. | `docs/proofs/model-forward-parity.md:16-27`; `CLAIMS.md:85` |
| **Fully offline / air-gapped operation (3 tiers)** | shipped | The gate, the policy, and a small model all run with zero network calls. THE edge selling point. | `GETTING-STARTED.md:16-41`; `cmd/fak/main.go:171` (`--offline`); `AGENTS.md:55` |
| **Adjudicated tool calls on a 1.5B model, CPU only** | shipped & measured | The exact model class that fits a phone/edge box, proven end-to-end through the gate. | `BENCHMARK-AUTHORITY.md:30,34`; `cmd/simpledemo` (Qwen2.5-1.5B-Q8, 38 tok/s on M3 CPU) |

**The honest gaps (net-new work, not claims):**

- No `linux/arm64` *release artifact* yet (matrix ships `darwin/arm64` only). The
  self-build one-liner already works and is documented (`INSTALL.md`:
  `GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build ./cmd/fak`), so the gap is a
  *published binary*, not a port. Pi / Jetson / arm64 gateway is **one matrix line
  away**. (`release-artifacts.yml:43-51`)
- No 32-bit ARM (armv6/v7), no Android NDK / iOS Swift bindings.
- No measured footprint on real constrained hardware (Pi, Jetson, phone SoC): no
  minimum-RAM, no power/battery numbers. `HARDWARE-MATRIX.md` lists only high-end boxes.
- No registered Engine that wraps a phone-native NPU runtime (CoreML/TFLite/Hexagon).
  The seam exists (`EngineDriver`); the adapter doesn't.
- In-kernel decode streaming still synthesized post-adjudication (issue #48).

---

## 3. The three buyer segments and the pitch for each

### A. Phone platform owners (Apple, Google, and the Android OEMs): *the enforcement layer they don't ship*

**Pitch:** "Your on-device agent decides which App Intent or sub-tool to call. fak
is the gate that decision passes through before it touches the camera, contacts,
location, or a payment intent. It is default-deny, fails closed, and writes one
tamper-evident line per call. The model can't jailbreak past a gate it doesn't
control." This is the least-privilege-per-tool enforcement Android's own "excessive
agency" guidance asks for, delivered as a gate rather than developer discipline.

**Why it's credible today:** the gate is hardware-independent and already walls a
booby-trapped policy 5/5 across a weak cloud model and a local 1.5B CPU model with
identical verdicts (`docs/repro-packet.md`). The work to make it land on a phone is
*bindings*, not *capability*.

### B. Edge gateway / industrial IoT / automotive: *offline + provenance + determinism*

**Pitch:** "A device that can't phone home still needs to prove every action was
authorized and never altered. fak runs the gate, the policy, and a small model
fully air-gapped. It replays decisions deterministically at cosine=1.0. It keeps an
append-only SHA-256 hash-chained journal you verify offline with `fak audit
verify`." This segment is the strongest fit. It values exactly what fak ships and
least values the throughput fak doesn't chase.

**External tailwind:** the edge-agent attack surface is now a named research area
(["Systems-Level Attack Surface of Edge Agent Deployments on IoT"](https://arxiv.org/pdf/2602.22525),
["3D Guard-Layer: An Integrated Agentic AI Safety System for Edge AI"](https://arxiv.org/pdf/2511.08842)).
Commercial local-agent hardware is already shipping, including the SwitchBot AI Hub
and Home Assistant with MCP. Buyers are already looking for this layer.

### C. Regulated / compliance-driven (medical, automotive, EU market): *the Article 12 artifact*

**Pitch:** "The EU AI Act's high-risk audit-log obligations are enforceable
**August 2, 2026** and mandate append-only, hash-chained (SHA-256 minimum) logs
with ≥6-month retention, and most teams have *no structured audit trail at all*.
fak's journal is that artifact, already shipped: append-only, `sha256(PrevHash ‖
content)`, per-row durable flush, tamper-detected by `fak audit verify`."
([Medium — AI Agent Audit 2026](https://medium.com/@Indext_Data_Lab/ai-agent-audit-the-complete-2026-governance-and-compliance-guide-aa945b2d2f67),
[didit — AI Compliance in the LLM Era 2026](https://didit.me/blog/compliance-in-the-llm-era/))

**Why it's a wedge, not a feature:** fak isn't selling "we added logging." The
mechanism it shipped for its own integrity story (the hash-chain) *is* the regulated
artifact. That's a rare position. The compliance deliverable falls out of the
architecture instead of being bolted on.

---

## 4. The roadmap, ordered by payoff over cost

Each step is sized against the honest inventory above. The first three are cheap and
make the edge story demonstrable rather than theoretical. The rest are the sequence
that turns a demo into an adoption.

### Tier 1: make "runs on the edge" literally true and provable (days, not weeks)

1. Add `linux/arm64` to the release matrix. One line in
   `release-artifacts.yml` (`CGO_ENABLED=0` already proves nothing to port, and
   `INSTALL.md` already ships the self-build one-liner). This alone makes "download
   fak on a Raspberry Pi 5 / Jetson Orin / arm64 edge gateway" a true statement with
   an *official* binary behind it instead of a build step. **Highest-leverage move in this whole document.**
2. Witness fak on one cheap arm64 edge box (Pi 5 or Jetson Orin Nano) and add a
   row to `HARDWARE-MATRIX.md`: the gate latency (Decide ns), the offline policy
   walk, and a 1.5B-Q8 adjudicated turn. The determinism claim says the *verdicts*
   will be bit-identical to the M3, so the only new data is wall-clock + RSS.
   Converts "no constrained-hardware testing" from a gap into a proof.
3. Publish an "edge quickstart" explainer: the air-gapped tier-0/tier-2 walk,
   `fak audit verify` offline, and the Article-12 framing. The capability is
   already in `GETTING-STARTED.md`; this surfaces it for the edge buyer who won't
   read the whole guide.

### Tier 2: the phone-native seam (weeks)

4. Ship one reference `EngineDriver` that wraps a phone-class runtime. The seam
   is shipped (`EngineDriver.Complete(ctx,*ToolCall)`); build the adapter for the
   most accessible target first: Ollama or llama.cpp on-device, then MLC. This
   makes "fak is the gate, your NPU runtime is the engine" a runnable example, not a
   diagram. Pick CoreML/TFLite/Hexagon adapters by which platform partner engages.
5. Android NDK example + iOS Swift example. fak compiles to a static arm64
   library; the work is the FFI surface and a sample app that routes an on-device
   agent's tool calls through fak before they hit an App Intent / Android intent.
   Pairs with segment A.

### Tier 3: the compliance and safety package (weeks, partner-driven)

6. An EU AI Act Article-12 conformance note: map each Article-12 requirement to
   the journal mechanism that satisfies it, plus the retention-policy knob the
   operator owns. This is a doc + a `fak audit verify` demo, not new code; the
   mechanism already matches.
7. A determinism + provenance "safety-case" packet for IEC 61508 / ISO 26262 /
   ISO/IEC TR 5469 contexts: cosine=1.0 replay + deletion certificates as the
   reproducibility and data-handling evidence. Partner-gated; build when a real
   automotive/medical conversation is live.

### Tier 4: close the measured-footprint gaps (as demand pulls)

8. Power/energy and minimum-RAM profiling on the witnessed edge boxes.
9. 32-bit ARM (armv7) only if a real device target needs it. Most modern edge/IoT
   Linux is arm64.

---

## 5. The single sentence to lead every edge conversation

> **The phone and the edge box already have a fast model. What they don't have is
> the gate that treats that model as an untrusted program: default-deny on every
> tool, poisoned results held out of memory, and a tamper-evident log that already
> matches the audit law landing in August 2026. fak is that gate, in one pure-Go
> binary that runs the same on a Raspberry Pi as on a datacenter GPU.**

Lead with the fence and the law, never with tokens-per-second. That framing is true
today, defensible line-by-line, and aimed exactly at the band Apple and Google left
empty.

---

## Sources

External (dated 2026):
- [Android developers — Gemini Nano](https://developer.android.com/ai/gemini-nano)
- [Android — Mitigate excessive agency vulnerabilities](https://developer.android.com/privacy-and-security/risks/ai-risks/excessive-agency)
- [Yaabot — Gemini Intelligence: Android's AI Agent Era](https://yaabot.com/41610/gemini-intelligence-android-ai-agent/)
- [buildmvpfast — Apple Intelligence Agent SDK 2026](https://www.buildmvpfast.com/blog/apple-intelligence-agent-sdk-third-party-developers-guide-2026)
- [MindStudio — Apple WWDC AI strategy: Siri, App Intents, MCP](https://www.mindstudio.ai/blog/apple-wwdc-ai-strategy-siri-app-intents-mcp)
- [Medium — AI Agent Audit: 2026 Governance and Compliance Guide](https://medium.com/@Indext_Data_Lab/ai-agent-audit-the-complete-2026-governance-and-compliance-guide-aa945b2d2f67)
- [didit — AI Compliance in the LLM Era: Regulatory Guide 2026](https://didit.me/blog/compliance-in-the-llm-era/)
- [arXiv — Systems-Level Attack Surface of Edge Agent Deployments on IoT](https://arxiv.org/pdf/2602.22525)
- [arXiv — 3D Guard-Layer: Agentic AI Safety for Edge AI](https://arxiv.org/pdf/2511.08842)

Internal (shipped artifacts, this repo): `README.md`, `GETTING-STARTED.md`,
`AGENTS.md`, `CLAIMS.md`, `POLICY.md`, `docs/HARDWARE-MATRIX.md`,
`docs/repro-packet.md`, `docs/proofs/journal.md`, `docs/proofs/deletioncert.md`,
`docs/proofs/model-forward-parity.md`, `docs/explainers/one-binary-one-surface.md`,
`internal/journal/journal.go`, `internal/compute/cpuref.go`,
`internal/abi/registry.go`, `.github/workflows/release-artifacts.yml`,
`BENCHMARK-AUTHORITY.md`, `cmd/simpledemo`.
