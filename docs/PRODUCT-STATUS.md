---
title: "fak product status — what you can pick up and use today"
description: "A person-facing snapshot of where fak's product concepts stand: which are durable products you can run on a laptop today, which are real subsystems not yet a product surface, and the next steps that turn the second group into the first. Every verdict is cross-checked against the tree by tools/product_scorecard.py."
---

# Product status — durable, real, useful-today

> **As of 2026-06-24 (fak v0.30.0).** This page answers the one question a person
> asks of a project: *of the concepts fak ships, which can I pick up and use this
> afternoon — and which are still a research seam or a named gap?* Every number and
> verdict here is re-derived from `tools/product_scorecard.data/` by
> `tools/product_scorecard.py` and cross-checked against the real tree (the CLAIMS
> tag a concept carries, whether its first command resolves, whether its witness and
> entry docs exist). Nothing on this page is hand-typed; regenerate it with the
> commands at the bottom.

## Headline

| Metric | Value |
|---|---|
| **Durable products you can run on a laptop today** | **10** |
| Usable today (needs a GPU / key / network, or a benchmark you run to see a result) | 8 |
| Real subsystems, not yet a person-facing surface | 17 |
| Coverage of the concept catalog | 100% (18/18 CLAIMS.md concept sections positioned) |
| Product-debt (honesty + coverage defects) | 0 |
| Composite map score | 100/100 (grade A) |

> **Read this right.** The score grades how *complete and honest the product map is*,
> not how much fak wins. An honest `real-not-easy` subsystem or a labeled `honest-stub`
> is not a defect; an *overclaimed* verdict is. The 17 real-but-not-yet-a-surface
> concepts are exactly where the next product gains come from — see [What's next](#whats-next).

## Standing at a glance

```text
product standing chart — 35 concepts · score 100.0/100 (grade A) · product-debt 0

verdict ladder (count of concepts, best -> roadmap):
  ★ durable-product ████████████████············ 10
  ● usable-today    █████████████··············· 8
  ◐ real-not-easy   ████████████████████████████ 17
  ○ honest-stub     ···························· 0
  · concept-only    ···························· 0

verdict mix by category (each cell = one concept):
  memory       ★◐◐              (3 concept(s); 1 durable, 0 usable-today)
  model        ●◐◐◐◐◐◐◐         (8 concept(s); 0 durable, 1 usable-today)
  performance  ●●●●◐◐◐          (7 concept(s); 0 durable, 4 usable-today)
  platform     ★★●●●            (5 concept(s); 2 durable, 3 usable-today)
  security     ★★★◐◐◐           (6 concept(s); 3 durable, 0 usable-today)
  tooling      ★★★★◐◐           (6 concept(s); 4 durable, 0 usable-today)

can a person run it today?
  laptop (offline)   █████████████··············· 16
  needs gpu/key/net  ██·························· 2
  no direct command  ██████████████·············· 17

coverage  [████████████████████████████████] 100.0%  (18/18 concept sections positioned)

legend: ★ durable-product   ● usable-today   ◐ real-not-easy   ○ honest-stub   · concept-only
```

The biggest bar is `real-not-easy` (17): real, witnessed subsystems with no surface a
person runs directly. `tooling` and `security` carry the most durable products;
the `model` lane is almost entirely subsystem-deep (the kernel-owned KV cache, the
quarantine bridge, the parity lanes) — proven in tests, not yet wired into a live run.

## What you can run today (10 durable products)

Each ships, has an offline first command (no GPU, no key), a witness that exists, and
an entry doc — usable on a laptop this afternoon.

- **Context debugger (`cdb`)** — replay a real session transcript and watch what the
  kernel kept, evicted, or quarantined.
- **One static Go binary** — the whole governed surface (gateway, capability floor,
  quarantine, audit) in a single dependency-free artifact.
- **MCP server (`fak serve --stdio`)** — the five `fak_*` adjudication tools from any
  MCP client.
- **Default-deny capability floor** — refuse an irreversible tool call structurally;
  the lever was never wired up, so it fails closed.
- **In-process adjudicator** — the DOS reference monitor on the tool-call path.
- **Write-time result quarantine (context-MMU)** — hold a poisoned tool result out of
  the model's context by structure.
- **Pre-flight ladder** — static parse + schema validation of a tool call before it runs.
- **Answer-shape witness** — catch degeneration / verbosity in a model's output.
- **Doctor** — an operator diagnostic over answer-shape and kernel admit.
- **Codelint** — language packs that check agent-written code actually parses, at the
  kernel boundary.

Another **8 are usable today** but need a GPU, an API key, a network, or are a
benchmark you run to *see* a result (the governed gateway `fak serve`, the Claude Code
passthrough, the fan-out / turn-tax / long-context benchmarks, the RSI ship-gate, and
the persistent context planner).

## What's real but not yet a product surface (17)

These are the witnessed subsystems behind the headline guarantees — bit-exact KV
eviction, the cross-engine co-residence seam, RadixAttention parity, the normalize-and-
rescan admission driver, information-flow control. They are real and proven, but a
person can't run them directly yet; they live *inside* the agent / serve path or behind
a build tag. Turning the highest-value ones into surfaces is the next frontier.

## What's next

The product-scorecard pass surfaced four next steps that aren't already tracked. Each
is filed; **#581 is resolved** (see *Resolved* below), leaving three open:

- **[#579](https://github.com/anthony-chaudhary/fak/issues/579)** — wire the model-side
  KV quarantine (evict + planned-elision) into the **live** agent/serve loop. Today the
  bit-exact eviction is proven only against a synthetic model; this makes the flagship
  KV-quarantine guarantee fire on a real run.
- **[#580](https://github.com/anthony-chaudhary/fak/issues/580)** — train the fine-tuned
  syscall/adjudication model from the harvest `LabelRow` corpus. The kernel already
  harvests its own verdicts; nothing consumes them yet. This closes the RSI loop's
  missing edge.
- **[#582](https://github.com/anthony-chaudhary/fak/issues/582)** — generalize the
  product/durability **verdict ladder** itself into a domain-free DOS primitive: an
  evidence-bound readiness score that can't be gamed by editing the claim. The same
  distrust DOS applies to agents, applied to a project's claims about itself.

### Resolved

- **[#581](https://github.com/anthony-chaudhary/fak/issues/581)** — *resolved by the
  issue's own "or document why it stays an optional leaf" branch.* The RadixAttention
  prefix-tree KV reuse **is wired into `fak serve`** on the in-kernel model path
  (`fak serve --gguf …` with no `--base-url`) and is **on by default** since `68b67c6`:
  `InKernelPlanner` looks up the longest cached KV prefix and re-prefills only the
  divergent suffix every turn (`internal/agent/inkernel_planner.go`), bit-identical to a
  full recompute. It stays an optional leaf for the **proxy governed-gateway mainline**
  (`fak serve --base-url …`): there fak is an adjudication proxy in front of an upstream
  that **owns** the KV cache, so fak holds no local `model.KVCache` for a radix tree to
  index — the only prefix-reuse lever in proxy mode is the byte-faithful `cache_control`
  passthrough (shipped), which lets the *upstream* engine's own prefix cache hit. The
  device-backend decode path (`--backend`) also bypasses the tree today (CPU-session-only
  reuse). Front-of-prompt reuse shipped; the separate mid-run KV-quarantine wiring is #579.

## Regenerate / verify

```bash
python tools/product_scorecard.py                 # the full human scorecard
python tools/product_scorecard.py --chart         # the at-a-glance chart above
python tools/product_scorecard.py --json          # machine payload
python tools/product_scorecard.py --markdown-dir docs/product-scorecard   # the generated scorecard
```

- The honesty ledger this maps: [`CLAIMS.md`](../CLAIMS.md).
- The generated, per-concept scorecard: [`docs/product-scorecard/`](product-scorecard/README.md).
- The proof-not-self-report status for the v0.2.x line: [`STATUS.md`](../STATUS.md).
