# fak — the agent kernel

<!-- readme-verified: 2026-06-21 vs VERSION 0.30.0 + BENCHMARK-AUTHORITY · process: tools/readme_freshness_audit.py + /refresh-readme · front page refocused on the turn-tax hero; breadth stills moved to BENCHMARK-GALLERY.md -->

**Treat the model like an untrusted program, and the tool call like a syscall.**
That one move is the whole idea. Everything an agent does to the outside world — call
a tool, admit a result into its memory, reuse a cached answer — becomes a syscall that
passes through a kernel the model doesn't control. From the security seat that kernel
is a permission gate the agent can't talk its way past. From the performance seat it's
a way to do the shared work *once* instead of every turn. The surprise of this project
is that those turn out to be the **same gate** — and that owning that boundary lets you
do things no production agent stack does today.

`fak` is not a faster model server, and doesn't try to be. SOTA engines (vLLM, SGLang,
llama.cpp) win raw throughput and prefix-cache reuse — that's their job. But serving an
agent *safely* is a whole stack on top of the token engine: the API wire your agents
speak, a capability gate, result containment, an audit trail, auth, governance metrics.
`fak` is that half of the stack collapsed into **one static Go binary** — the gateway,
the policy gate, the quarantine, and the audit surface in a single process you
`go install` and run, in front of whatever serves your tokens. It owns the questions the
engines leave open: **which effects are allowed, which results may enter memory, when
reuse is still legal, and what survives a session boundary.**
→ **[One binary is the whole surface](docs/explainers/one-binary-one-surface.md)**

<!-- hero video — generated from the headline visuals by tools/hero_video_gen.py
     (storyboard: visuals/hero-video.storyboard.json). GitHub markdown can't autoplay a
     repo-relative .mp4, so the embed is a compact looping .gif that links to the full mp4. -->
[![fak — the agent kernel · a ~40s, 1440p model-card reveal: the turn-tax curves animate the measured 9.7x prefill elimination, then the capability matrix, the three-pillar stat card with its honest single-stream fence, and the eight-axis sweep build in — click for the full-resolution MP4](visuals/hero-video.gif)](visuals/hero-video.mp4)

<sub>▶ a ~40-second reveal — the curves draw themselves, the multipliers count up — [full-resolution MP4 (1440p)](visuals/hero-video.mp4)</sub>

**▶ Try the live demos** — [three interactive demos running the real kernel on GCP](https://anthony-chaudhary.github.io/fak/demos.html):
the turn-tax race (a SOTA loop vs fak's 1-shot kernel), the multi-agent context-reuse proof,
and a live model reuse race against a tuned warm-cache (SOTA) baseline — nothing to install,
they run in your browser. Want your own copy?
The self-contained one is one command — `go run ./cmd/turntaxdemo` —
and [the full guide](https://anthony-chaudhary.github.io/fak/run-the-demos.html) covers local,
headless, Docker, and your own cloud VM.

**Live:** [the demos](https://anthony-chaudhary.github.io/fak/demos.html) ·
[run your own](https://anthony-chaudhary.github.io/fak/run-the-demos.html) ·
[the showcase](https://anthony-chaudhary.github.io/fak/showcase.html) ·
[docs site](https://anthony-chaudhary.github.io/fak/) ·
**Try it now:** [in Colab](https://colab.research.google.com/github/anthony-chaudhary/fak/blob/main/notebooks/fak-quickstart.ipynb) (free GPU, nothing to install)

---

## The one chart: re-prefill is linear, resident KV is not

An agent rereads the same setup on every turn. The naive baseline re-prefills it on each
cold KV miss, so its cost climbs linearly with context, fleet size, and turns. `fak`
keeps that work **resident and addressable**, so its line stays flat — and the gap is the
whole pitch: a measured **9.7×** less prefill at 8 workers on real WebVoyager, the
conservative **4.1×** vs a tuned warm-cache SOTA stack.

[![fak turn-tax efficiency curves — three panels (per-turn prefill cost vs context, WebVoyager fleet prefill vs workers, 50-turn fleet serving work vs turns), each a baseline re-prefill curve rising linearly while fak's resident-KV curve stays flat, multipliers 20,480x / measured 9.7x / 4.1x](visuals/60-hero-turntax-curves.png)](BENCHMARK-GALLERY.md#60--turn-tax-curves)

**The breadth is the point too — but it belongs in the gallery, not the front page.** A
capability matrix (`fak` spans the whole boundary; serving stacks span one band), a
three-pillar stat card with its honest single-stream fence, and an eight-axis sweep round
out the receipts — each generated from one source-of-truth data file, honest by
construction (a `[NAIVE]` number stays fenced, competitor cells never carry a fabricated
figure, and the single-stream throughput `fak` *doesn't* target is shown, not hidden).
→ **[The full benchmark gallery](BENCHMARK-GALLERY.md)** ·
**[every number, traced to its commit + artifact](BENCHMARK-AUTHORITY.md)** ⭐

---

## Two flips that are bigger than they sound

**1. The permission policy runs *inside* the kernel.** Most agent safety bolts a
recognizer onto the *outside* of the loop — a pre-tool hook, a sidecar, a second model
asked "is this safe?". Two weaknesses: the model can argue its way past a recognizer
(prompt injection is exactly that), and when the outside thing crashes or times out the
call usually runs anyway — **fail-open**, precisely when you're under attack. `fak` puts
the check on the *same call path* as the tool call — one address space, no IPC,
**default-deny** — so the gate isn't something the agent talks to, it's something its
call passes *through*, like `read()` through the OS kernel. Refusing an irreversible
action doesn't depend on *catching* the attack; it depends on the lever never having
been wired up. For thirty years "more security" meant more checks to recognize bad
things — a game attackers win. The flip is to stop recognizing and start *not building
the lever*. → **[Policy in the kernel](docs/explainers/policy-in-the-kernel.md)**

**2. The cache becomes addressable — past what any shipped engine offers.** As a model
reads, it builds a scratchpad of the work-so-far (the *KV cache*) so it doesn't re-read
from scratch each turn. Every way production reuses that scratchpad today — vLLM, SGLang,
the OpenAI / Anthropic prompt caches — only reuses it *from the front*: keep the run from
the very first word, and the moment anything changes in the middle, everything after that
point is thrown away and recomputed. That's most of the speed, and it's commoditized.
What no shipped engine offers is the other direction: reach *into* the middle of a kept
run, cut one span — a poisoned tool result, an expired secret — and leave the scratchpad
**bit-for-bit identical to a run that never saw it** (checked against the reference at
`max|Δ| = 0`, i.e. not one number differs). `fak` can, because it owns that scratchpad as
a kernel object instead of renting it from a serving engine. That turns the cache from a
*speed* structure into one **policy can address**:
evict a span because a verdict said so, not because memory got tight — and *prove* it's
gone. → **[Addressable KV cache](docs/explainers/addressable-kv-cache.md)**

---

## The tension these resolve

For most of computing history, every optimization came with a tax: a faster cache opened
a coherence hole, a clever reuse trick added arcane state nobody else understood — speed
and safety pulled opposite ways. `fak`'s bet is that for agents they **converge**,
because the safety boundary and the reuse boundary are *the same boundary*. One
write-time gate decides whether a tool result may enter context (a security act) and
pages its heavy bytes out to a content-addressed store (an optimization act). One read of
the code calls it *injection containment*; the other calls it *working-set paging* —
**same code**. The correctness metadata *is* the performance metadata.

That's a claim about `fak`'s object model shown by example, not a proven law — and it has
an edge: the convergence doesn't hold on raw GPU throughput (`fak` pays for its bit-exact
guarantee in memory), and it's a reuse win only for read-heavy fleets.

---

## See it in 2 minutes (no key, no model, no GPU)

Just [Go 1.26+](https://go.dev/dl/). Or run it in a hosted notebook — free T4 GPU, nothing
to install:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anthony-chaudhary/fak/blob/main/notebooks/fak-quickstart.ipynb)
(see [`notebooks/`](notebooks/README.md)).

```bash
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb --args "{}"
go run ./cmd/fak agent --offline
```

`refund_payment` returns **`DENY (POLICY_BLOCK)`** — refused with a named reason.
`search_kb` returns **`ALLOW`**. Then `agent --offline` runs the same task twice — tools
wired directly vs. behind `fak` — and prints the before/after:

```
metric                       without fak   with fak
model turns                            9          7
injection in context                 YES         no
destructive op executed              YES         no
task completed (booked)              YES        YES
```

Both finish the task, but with `fak` the booby-trapped instruction never reaches the
model and the dangerous action never runs. Full walkthrough:
[`docs/repro-packet.md`](docs/repro-packet.md).

---

## Why this matters now

An agent system's cost isn't one number — it's roughly `agents × turns × working-set ×
reread-rate × legality checks`, and the naive stack lets all five multiply by making the
model **reread the same setup** on every turn (five agents over fifty turns is 250 chances
to reprocess the same shared prompt). `fak` attacks the one safe term, reread-rate,
*without* deleting the proof reuse is still legal: the first worker pays, everyone after
reads for free, so *more agents can mean less total work*. Two fences — the reuse win is
**self-host only** (an app that just *calls* a frontier API gets the safety floor, not the
savings), and the frontier-scale "agent city" numbers are design targets, not
measurements. → **[The full cost model and personas](docs/concepts-and-story.md)**

**Measured on real WebVoyager (643 tasks)** — every navigation action re-prefills the
whole browser context (DOM, tool schemas, task history), so the turn-tax compounds with
fleet size:

| Workers | Naive Re-Prefill | fak Fused | Net Elimination |
|---------|-----------------|-----------|-----------------|
| 1 | 170.9 M tokens | 19.4 M tokens | **8.8×** |
| 8 | 1.37 G tokens | 141.3 M tokens | **9.7×** |

The turn-tax is **worker-independent** — every agent pays it, every turn, regardless of
fleet size. SOTA agents like Alumnium (98.5% WebVoyager success) reach the same capability
through `fak` at ~9× less prefill cost (measured).
→ **[Frontier WebBench baselines](docs/webbench-baselines.md)**

**And it's not one lucky box.** The same pure-Go kernel — same bit-exact gates — is
profiled across **4 distinct hardware platforms** (Apple M3 Pro/Metal, AMD Ryzen +
RX 7600/Vulkan, Intel + RTX 4070/CUDA Ada, 8-GPU server/CUDA Ampere) spanning 2 CPU ISAs, 4 GPU
backends, and 4 operating systems — and the deterministic results reproduce byte-for-byte
on every one. → **[The hardware matrix](docs/HARDWARE-MATRIX.md)**

[![fak hardware coverage matrix — four hardware platforms (Apple M3 Pro / Metal, AMD Ryzen + RX 7600 / Vulkan, Intel + RTX 4070 / CUDA Ada, 8-GPU server / CUDA Ampere) across two CPU ISAs, four GPU backends, and four operating systems, with the bit-exact correctness gates passing on every backend](visuals/56-hardware-coverage-matrix.svg)](docs/HARDWARE-MATRIX.md#the-coverage-matrix)

---

## Security: the lock, not the screener

The capability gate refuses an irreversible action *by structure* — the tool was never
allow-listed, so no amount of context changes the answer. A separate quarantine holds
suspicious tool results out of the model's memory entirely. The detector that flags
those results is the evadable part, and we say so: it's **≈100% evadable by design** —
a helpful bonus, never the floor. The floor is the lock (the lever doesn't exist) and
the containment (the bytes never reach the model). An attacker has to beat **two
independent gates**, not fool one classifier.

Same task — book the cheapest SFO→JFK flight after reading a booby-trapped refund
policy — run twice, unmediated vs. behind `fak`:

| model | booked? | trap reached the model? |
|---|---|---|
| `gemini-2.5-flash` (strong) | ✓ → ✓ | **YES → no** |
| `gemini-2.5-flash-lite` (weak) | **✗ → ✓** | **YES → no** |
| `Qwen2.5-1.5B` (local, CPU) | ✓ → ✓ | **YES → no** |

The weak model is the case that matters: without `fak` it fell for the trap and booked
nothing; with `fak` it ignored the trap and booked the flight. Across these runs the
injection reached the unprotected baseline 5/5 and `fak` walled it off 5/5.

---

## How far do you want to take it?

Every rung is useful on its own — you get value even if you never buy the whole thesis.

- **Front your existing model.** `fak serve` puts the gate in front of any
  OpenAI-compatible server (Ollama, vLLM, a cloud provider). Keep your model and your
  stack; gain a reviewable allow-list, result quarantine, and an audit trail. This is
  where most people should start, and it's a complete product by itself.
- **Run the kernel offline.** Author a policy, check a tool call, measure the
  adjudication boundary — no model, no network. (The 2-minute demo above.)
- **Go all in: the fused kernel.** For the believers — run the model *inside* the
  kernel's address space, so the KV cache is a kernel object and the context-MMU and vDSO
  become real operations on attention state. This is where the two flips stop being
  framing and become mechanism: quarantine that reaches attention, prefix reuse the
  kernel owns, a finished session that reloads as a core dump. It's a correctness
  *reference*, not a production serving engine — but on a model that fits the GPU, the
  in-kernel CUDA decode already hits **parity with `llama.cpp` Q8_0 (~120 tok/s on an RTX
  4070)** with an opt-in CUDA graph. Capability is still your model's job; the kernel
  gives you frontier-grade *safety* and ~$0 cost on a small local model today.

→ **[Guided tutorial](docs/fak/tutorial.md)** (zero to first adjudicated call, real output
at every step) · **[Getting started](GETTING-STARTED.md)**

---

## One binary is the whole surface — laptop to fleet

Notice that every rung above is the *same binary*. That's the part the throughput
benchmarks never show. A fast token engine — vLLM, SGLang — is one band of serving an
agent; the rest is a stack you normally assemble around it: a gateway for the wire, a
policy/authorization service, a result screener, an audit pipeline, an MCP bridge, a
reverse proxy for auth. Those engines are Python on a CUDA/PyTorch stack and multi-process
by design; their production container is multi-GB because it bundles CUDA + PyTorch
(pip/uv into an existing env is the lighter path), and vLLM's own security docs tell you to
front it with a reverse proxy for auth and endpoint allow-listing. `fak` collapses the
*governance + gateway*
half of that stack into **one static Go binary with zero external dependencies** — no
Python, no CUDA toolchain, no `go.sum` — that runs on a laptop CPU with no key, model, or
network, and is the same artifact you harden for a fleet:

| | A developer, locally | A platform team, in a fleet |
|---|---|---|
| **Run** | `fak serve --base-url … --model …` | the same binary, + the flags → |
| **Policy** | the built-in default floor | `--policy floor.json` (a reviewable allow-list in git, not a code edit) |
| **Auth** | none (loopback) | `--require-key-env FAK_TOKEN` (bearer or `x-api-key`) |
| **Observe** | `curl /healthz` | scrape `/metrics`; ship JSON access logs + `X-Trace-Id` to your SIEM |
| **Ship** | one binary on `PATH` | one ~13 MB `distroless/static` container per replica |

Nothing new gets installed between those columns — no environment that drifts, no sidecar
to keep in lockstep, one statically-linked binary as the entire supply-chain surface. It
fronts your fast engine (Tier 1) over the OpenAI **and** Anthropic wires plus MCP, so
Claude Code, Cursor, or any OpenAI client drops in with no agent-side changes. The honest
fence: `fak` is not the fast token engine, and its own in-binary model is a correctness
reference, not a production server — the contrast is **operational surface**, not tokens
per second. → **[One binary is the whole surface](docs/explainers/one-binary-one-surface.md)**

---

## What's real, what's not (we keep score)

`fak` is built to survive a skeptic reading the code. Every capability in
[`fak/CLAIMS.md`](CLAIMS.md) carries one machine-checked tag. The short version:
the permission gate, the local-answer shortcut, the auto-repair ladder, the quarantine,
the in-kernel model (math proven exact against a reference), and the OpenAI-compatible
gateway are **shipped and on the critical path**, each closed by a test. The
**power/energy** numbers are **simulated** (no power meter on the box). Sharing one KV
pool with a *separate* serving engine is a labeled **stub**. And a 29-claim prior-art
audit scored **0/29 novel** — every primitive is established; the contribution is the
*assembly*, putting them together as one in-process gate where the tool call is the
checkpoint.

---

## Install

One static binary — no clone, no Go toolchain, no Python, no CUDA, no dependency tree to
manage (there is no `go.sum`). Full guide: [**Getting started**](GETTING-STARTED.md).

```bash
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh
```

Or grab a [prebuilt archive](https://github.com/anthony-chaudhary/fak/releases/latest)
(`linux_amd64`, `darwin_amd64`, `darwin_arm64`, `windows_amd64`), or run it
[in a container](GETTING-STARTED.md). Then:

```bash
fak policy --dump > floor.json   # a starter allow-list you can edit + review
fak serve --addr 127.0.0.1:8080 --base-url http://localhost:11434/v1 --model qwen2.5:1.5b
```

> **Install with Go** (needs [Go 1.26+](https://go.dev/dl/)): the module is the
> repo root, so `go install github.com/anthony-chaudhary/fak/cmd/fak@latest` drops
> `fak` onto your `$(go env GOBIN)` (`$GOPATH/bin`). Or from a clone:
> `go build -o fak ./cmd/fak`. Full install matrix: [`INSTALL.md`](INSTALL.md).

---

## Go deeper

| If you want… | Read |
|---|---|
| **The two flips, from first principles** | [Policy in the kernel](docs/explainers/policy-in-the-kernel.md) · [Addressable KV cache](docs/explainers/addressable-kv-cache.md) |
| **Why one Go binary beats a serving stack** (operational surface, laptop → fleet) | [One binary is the whole surface](docs/explainers/one-binary-one-surface.md) |
| **Every benchmark number** (single source of truth, traced to commit + artifact) | [`fak/BENCHMARK-AUTHORITY.md`](BENCHMARK-AUTHORITY.md) ⭐ |
| **Every machine fak runs on** (the hardware matrix — 4 platforms, 2 CPU ISAs, 4 GPU backends, 4 OSes) | [`docs/HARDWARE-MATRIX.md`](docs/HARDWARE-MATRIX.md) |
| **Web agent benchmark results** (real WebVoyager: 8.8-9.7× measured) | [`docs/webbench-baselines.md`](docs/webbench-baselines.md) |
| **The parable, personas, and when-the-win-kicks-in tables** | [`docs/concepts-and-story.md`](docs/concepts-and-story.md) |
| **What "tuned SOTA" means** (the 10 optimizations fak sits on top of) | [`docs/explainers/sota-optimizations.md`](docs/explainers/sota-optimizations.md) |
| **Shipped capabilities** (runnable artifacts, claim tags) | [`fak/CLAIMS.md`](CLAIMS.md), [`fak/STATUS.md`](STATUS.md) |
| **Policy / permissions** | [`fak/POLICY.md`](POLICY.md) |
| **Architecture** (the registry seams, the frozen ABI) | [`fak/ARCHITECTURE.md`](ARCHITECTURE.md) |
| **Build your optimization on fak** (researchers/teams: plug in → prove correct → prove faster → ship) | [`fak/EXTENDING.md`](EXTENDING.md) |
| **First run, step by step** (guided session, real output at every step) | [`docs/fak/tutorial.md`](docs/fak/tutorial.md) ⭐ |
| **Quick answers** (what is fak, how it differs from a firewall / guardrails / vLLM, the threat model) | [`docs/FAQ.md`](docs/FAQ.md) |
| **A machine-readable map** (for LLMs & answer engines) | [`llms.txt`](llms.txt) |
| **New here?** | [`START-HERE.md`](START-HERE.md) · [Simple demo](cmd/simpledemo/README.md) |

---

## About this repository

This repository is the **canonical public home** of the project's public content — it
is edited **directly here**, not regenerated from a private mirror. A separate private
repo holds only the operator-specific material that must never be published (machine
names, IPs, lab hosts, internal paths).

**Cite this work:** machine-readable metadata is in [`CITATION.cff`](CITATION.cff)
(GitHub renders a "Cite this repository" button from it).

License: [Apache-2.0](LICENSE).

<sub>**Topics:** agent kernel · agent tool firewall · AI agent security · prompt
injection defense · tool poisoning · capability security · default-deny permission
gate · KV cache · addressable KV cache · LLM inference · LLM serving · self-hosted
LLM · agentic AI · MCP tool security · Go.</sub>
