---
title: "Cache frontier operating plan - keep the SOTA reuse work on the product path"
description: "The operating spine for fak's caching value-add: multi-agent reuse, O(1) context and query, provider-cache preservation, addressable KV deletion, and dogfood/demo/evidence lanes that keep those wins from getting buried under operational work."
---

# Cache frontier operating plan

This page is the project-management spine for fak's caching value-add. It does not
introduce a new benchmark claim. It keeps the existing claims, demos, and scorecards
pointed at one product outcome:

> A long-running agent or agent fleet should use fak as the memory and reuse kernel by
> default: cheaper long sessions, bounded context, queryable history, legal KV reuse,
> and a visible proof when that value is actually paying off.

The risk this page addresses is simple: the repo already contains the pieces, but they
are spread across benchmark packets, explainers, demos, scorecards, and operational
runbooks. A large SOTA multi-agent reuse win can get treated like just another artifact
while lower-leverage hygiene work keeps moving. The rule here is that cache-frontier work
is not done until it advances at least one of these product lanes:

1. **Dogfood:** fak uses the mechanism in the real dev loop.
2. **Demo:** a person can see the value without a key, model, GPU, or live service when
   that is technically possible.
3. **Product surface:** an operator can turn it on or inspect it through a `fak` verb,
   MCP tool, gateway endpoint, or documented default.
4. **Evidence:** the result has a witness and an honesty fence, not a loose multiple.

## North-star product

The product is **agent memory and reuse as a kernel service**. The user-facing version is
not "a cache" and not "a faster model server." It is:

- a governed gateway that preserves provider cache hits when it is only riding an
  upstream engine;
- an owned KV/cache path that can share, evict, and re-materialize spans when fak runs
  the engine itself;
- a lossless context store that lets the resident view stay bounded while the full
  history remains queryable;
- a scorecard and ledger that say whether this saved work in the sessions we actually
  ran.

Every cache-frontier task should preserve that sentence. Work that only improves a
subsystem but leaves no dogfood, demo, surface, or evidence belongs behind work that does.

## The four flagship tracks

| Track | Current proof | Product surface today | Product gap |
|---|---|---|---|
| **Multi-agent reuse win** | [`SESSION-VALUE-STACK-RESULTS.md`](benchmarks/SESSION-VALUE-STACK-RESULTS.md) reports the 50-turn x 5-agent value stack: 60.3x vs naive, 4.1x vs tuned per-agent KV, with the baseline fence in the same doc. | `go run ./cmd/ctxdemo -bars`, `go run ./cmd/turntaxdemo -print`, and the benchmark authority row. | Make this the first cache demo story, not a buried benchmark packet. Tie every multi-agent dogfood run to a cache-value row and a demo/update artifact. |
| **O(1) context + query** | [`o1-context-window-economics.md`](explainers/o1-context-window-economics.md) gives the measured proxy-path crossover; `internal/ctxplan` keeps the lossless store plus bounded view; `contextq` and the MCP `fak_memory_query` surface expose query composition. | `go run ./cmd/ctxplandemo -selfcheck`, `go run ./cmd/memqdemo`, `fak_memory_query` through MCP/gateway. | Promote one operator-facing "ask the session memory" path that uses our own repo sessions, not only synthetic demos. |
| **Provider-cache preservation** | `CLAIMS.md` entries for cache-prefix-preserving compaction and oversized result elision; [`cache-value-rollup.md`](cache-value-rollup.md) separates WITNESSED kernel reuse from OBSERVED dollar savings. | `fak guard`, `fak serve`, `fak nightrun score --json`, and current-tree `fak cachevalue report --since YYYY-MM-DD` when that subcommand is available. | Make the weekly cache-value card the default review artifact. Track 2 provider-dollar join must stay separate from Track 1 kernel reuse. |
| **Addressable KV deletion/quarantine** | [`addressable-kv-cache.md`](explainers/addressable-kv-cache.md), `unseedemo`, `deletioncert`, and the KV-MMU claims prove poison/secret removal and deletion certificates under their stated witnesses. | `go run ./cmd/unseedemo -print`, `go run ./cmd/deletioncert -selfcheck`, context debugger. | Connect the deletion/quarantine demo to the O(1) context/query product so "remove this span, then query the surviving history" is a natural workflow. |

## Operating board

This board is deliberately small. It is the set of next work that keeps the cache
frontier visible and product-shaped.
The expanded ranked backlog is
[`docs/cache-frontier/DEFAULT-ENABLEMENT-NEXT-50.md`](cache-frontier/DEFAULT-ENABLEMENT-NEXT-50.md):
50 next items for making pure fak, the served API, O(1) context/query,
provider-cache preservation, and SGLang/vLLM/llama adapters useful by default
without blending their evidence planes.

| Lane | Now | Next witness | Definition of done |
|---|---|---|---|
| **Cache demo front door** | [`run-the-demos.md#cache-frontier-walkthrough`](run-the-demos.md#cache-frontier-walkthrough) groups `ctxdemo -bars`, `turntaxdemo -print`, `ctxplandemo -selfcheck`, `memqdemo`, `unseedemo -print`, and the cache ledger into one local path. | Run the walkthrough on a clean machine and capture which commands pass, fail, or need a current-tree build. | A new user can run the cache story without reading five separate docs. |
| **Dogfood ledger** | Keep `fak guard` / `fak serve` sessions writing cache-value rows and make the weekly review start from the ledger, not anecdotes. | `fak nightrun score --json` plus `fak cachevalue report --since <week>` on the current tree. | Each "cache is paying off" statement is backed by WITNESSED reuse or OBSERVED dollars, never a blended number. |
| **O(1) query product** | Pick the one path we want agents to use first: MCP `fak_memory_query`, a `fak` CLI verb, or context debugger integration. | A real fak session image can be queried for "what changed / what matters / expand this ref" with a bounded resident view. | Our own agents can query their prior work instead of relying on a growing transcript or stale recall. |
| **Multi-agent productization** | Treat the 50-turn x 5-agent result as the flagship value-stack, with tuned-baseline fence attached. | A recurring dogfood run records the same geometry class or explains why the shape differs. | The SOTA win appears in demos, status updates, and planning intake as a product lane, not only as a benchmark result. |
| **Salience guard** | Route cache-frontier tasks through this page during planning. | Each new issue/doc/update names which of Dogfood, Demo, Product surface, Evidence it moves. | Operational hygiene cannot silently outrank the cache frontier unless it unblocks one of the four lanes. |

## Intake rule for agents

Before taking a cache-adjacent task, classify it in one line:

```text
cache-frontier: dogfood=<yes/no> demo=<yes/no> surface=<yes/no> evidence=<yes/no> flagship_track=<multi-agent|o1-query|provider-cache|kv-delete|none>
```

If all four fields are `no`, the work is not cache-frontier work. It may still be worth
doing, but it should not displace this board. If `flagship_track=none`, either connect it
to a track or call it general operations.

## Weekly review loop

Run the review from evidence in this order:

```bash
fak nightrun score --json
fak cachevalue report --since 2026-06-22 --json
fak cachevalue review --since 2026-06-22 --date 2026-06-29 --source-markdown reviews/2026-06-29.md --append-ledger docs/cache-frontier/review-ledger.jsonl --markdown-out docs/cache-frontier/reviews/2026-06-29.md
go run ./cmd/ctxdemo -bars
go run ./cmd/ctxplandemo -selfcheck
go run ./cmd/memqdemo
go run ./cmd/unseedemo -print
go run ./cmd/fak maturity next
```

On builds that do not expose `fak cachevalue report`, use `fak nightrun score --json`
as the shipped Track-1 witness and leave Track 2 as missing rather than inferred.

The output of the review is not a long memo. It is three lines:

```text
1. What cache-frontier value did we use ourselves this week?
2. What can a new person demo this week?
3. What is the next missing witness or product surface?
```

Write each new dated result under [`docs/cache-frontier/`](cache-frontier/README.md): one
markdown note for humans and one appended `fak-cache-frontier-review/1` row in
[`review-ledger.jsonl`](cache-frontier/review-ledger.jsonl) for future automation. The
review command has a non-mutating `--json` form for inspecting the row first. The first
entry is [`2026-06-29`](cache-frontier/reviews/2026-06-29.md), which records the current
thin Track-1 `run` evidence and the missing Track-2 provider-dollar ledger.

## Decision fences

- Quote the 50-turn x 5-agent win only with its baseline: **60.3x vs naive** and
  **4.1x vs tuned per-agent KV** are different claims.
- Do not turn Track-1 kernel reuse into Track-2 dollar savings. A dollar claim needs the
  provider/billing join.
- Do not call O(1) context a quality win until the task-success or faithfulness witness is
  named. The current economic result prices the context bytes and the bounded prefill tail.
- Do not let vDSO hit-rate become the caching headline. It is an upside secondary unless a
  real trace proves the addressable purity is high enough for the workload.
- Do not count a synthetic demo as dogfood. Dogfood means fak used the mechanism in the
  repo's own development or operating loop.

## Source map

- Product map: [`PRODUCT-STATUS.md`](PRODUCT-STATUS.md)
- Innovation catalog: [`INNOVATIONS-INDEX.md`](INNOVATIONS-INDEX.md)
- Demo catalog: [`run-the-demos.md`](run-the-demos.md), especially the [`cache-frontier walkthrough`](run-the-demos.md#cache-frontier-walkthrough)
- Cache-value trend: [`cache-value-rollup.md`](cache-value-rollup.md)
- Multi-agent value stack: [`SESSION-VALUE-STACK-RESULTS.md`](benchmarks/SESSION-VALUE-STACK-RESULTS.md)
- O(1) context economics: [`o1-context-window-economics.md`](explainers/o1-context-window-economics.md)
- Addressable KV cache: [`addressable-kv-cache.md`](explainers/addressable-kv-cache.md)
- Maturity backlog: [`MATURITY-SCORECARD.md`](MATURITY-SCORECARD.md)
