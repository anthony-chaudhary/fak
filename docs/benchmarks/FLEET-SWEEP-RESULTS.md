# FLEET-SWEEP-RESULTS — the 2-D turn-tax surface (turns × agents), measured on the real kernel

> **New here? Read `FLEET-SWEEP-EXPLAINED.md` first** —
> the same result with no math, one analogy (a team sharing a whiteboard of
> answers), and the five concept pictures. This file is the numbers.
>
> **What this measures.** `fak turntax` and the stochastic harness price ONE agent.
> This sweep adds the missing axis the fleet thesis lives on: what does running **A
> agents together** buy, across both the session length (**T**, turns per agent)
> and the fleet size (**A**)? We sweep the full **1..50 × 1..50** grid — 2,500 cells
> — three times (headline + two controls), plus two companion-axis scans.
>
> **What is grounded vs modeled (the honest line, same discipline as TURN-TAX §3.2).**
> The agent-count lever is a **real kernel event, not a model**. The kernel's tier-2
> vDSO cache is keyed `(tool, args-sha256, world-version)` and is **process-global**,
> so it is shared across every agent in a world epoch. When A agents read the same
> reference data, the first pays a cold engine round-trip and every other agent's
> identical read is served from that one result as a **tier-2 hit** the kernel counts
> itself (`Counters.VDSOHits`). Each cell is ablated **shared-world fleet vs
> per-agent-isolated worlds** — the difference equals the extra VDSOHits the shared
> epoch produced, a **measured path-swap**, exactly like turnbench's vDSO ON/OFF
> proof. The only modeled half is the per-turn **price** (the same `CostModel` knob).
>
> Reproduce: `wsl bash ./tools/fleet_sweep_run.sh` (writes everything under
> `experiments/fleet/`), then `python tools/fleet_fit.py` and
> `python tools/fleet_heatmap.py`. Code: `internal/turnbench/fleet.go`, runner
> `cmd/fleetbench/`, tests `internal/turnbench/fleet_test.go`.

---

## §1 — The headline surface (read-only fleet)

![2-D fleet turn-tax heatmaps](../../experiments/fleet/fleet-heatmap.png)

The same result as a **fleet-vs-baseline** comparison (the shared-cache fleet
against the same agents run isolated, in **absolute** tool round-trips deleted —
the shaded gap is the cross-agent uplift):

![fleet vs baseline, absolute terms](../../experiments/fleet/fleet-compare.png)

A **read/retrieval fleet** (research, monitoring, support-lookup): agents mostly
read shared reference data (a small catalog of `shared_pool=8` popular routes),
repeat a few of their own lookups, do a little arithmetic — and **do not write**
(the write axis is §4; the zero rate is deliberate, see §4 for why). Representative
cells (medians over 64 seeded trials):

| T (turns) | A (agents) | calls = T·A | shared_saved | isolated_saved | **cross_uplift** |
|---:|---:|---:|---:|---:|---:|
| 1  | 50 | 50   | 25   | 10   | **+14** |
| 10 | 10 | 100  | 74   | 47   | **+28** |
| 10 | 50 | 500  | 399  | 233  | **+167** |
| 20 | 20 | 400  | 342  | 241  | **+102** |
| 30 | 30 | 900  | 809  | 619  | **+190** |
| 50 | 10 | 500  | 462  | 395  | **+68** |
| 50 | 1  | 50   | 39   | 39   | **0** |
| 50 | 50 | 2500 | 2344 | 1973 | **+370** |

Read three things off this table:

1. **`shared_saved` is the total turns the fleet deletes.** At the T=50×A=50
   corner the read-only fleet deletes **2344 of 2500 calls** (94%) — a read-only
   fleet eventually serves almost everything from cache. Priced at the default cost
   model (1320 tok/turn, blended $3/$15 per Mtok): **3.09M tokens / $12.66 per
   fleet-run**, of which the **cross-agent bonus alone** (the 370 uplift) is
   ~490k tokens / **$2.00** the same agents run apart would not have saved.
2. **`cross_uplift` (shared − isolated) is the fleet-only win** — the turns sharing
   buys that A independent agents cannot get. `A=1` is **exactly 0** (nobody to
   share with): the cross-agent benefit is a strictly multi-agent phenomenon.
3. **Agents matter more than turns for the uplift.** Compare `T=10,A=50` (**+167**)
   against `T=50,A=10` (**+68**): the *same 500 calls*, but the short-session big
   fleet deletes 2.5× more — because uplift is **linear in A** but **saturating in
   T** (you can only cover a finite shared catalog). That asymmetry is the whole
   point of separating the two axes, and §3 makes it a formula.

---

## §2 — The controls (why a positive number here is real)

| control | what it forces | result |
|---|---|---|
| **no-share / no-write** | agents share nothing → sharing CANNOT help | **cross_uplift = 0.0000 across all 2500 cells** (mean, p50, min, max) |
| **single agent (A=1)** | nobody to share with | cross_uplift = 0 across the whole distribution |
| **write-heavy (30% writes)** | every write bumps the global world | **93% of cells ≤ 0** (worst −100): sharing is a net LOSS |

The no-share surface is the load-bearing one: it is **exactly** zero everywhere
(not approximately), while its own `shared_saved` still ranges 0..2350 from
intra-agent dedup. So the harness credits cross-agent sharing **only** when sharing
actually happens — any positive `cross_uplift` in §1 is a real measured tier-2
event, not the benchmark flattering itself. (Tests: `TestFleet_NoShareHasZeroCrossUplift`,
`TestFleet_SingleAgentHasNoCrossUplift`.)

---

## §3 — The scaling law (derived, then confirmed — not curve-fit)

The mechanism *implies* a closed form. Think of the `pool` distinct shared reads as
coupons to collect: the **first** agent primes each one (a cold engine call); every
**other** agent reads each already-primed coupon free. So the per-extra-agent uplift
is the **expected number of distinct shared reads one agent makes in T turns** —
coupon-collector coverage of the catalog:

> **cross_uplift(T, A) = (A − 1) · pool · (1 − (1 − 1/pool)^(p_shared · T))**

With the workload's own constants (`pool=8`, `p_shared=0.45`) and **zero free
parameters**, this predicts the full kernel-measured 2,500-cell surface at
**adjusted R² = 0.9974**. Its continuous approximation `c·(1−e^(−T/τ))·(A−1)`, fit
freely, lands at **adj R² = 0.99956** and **recovers the derived constants**:

| | asymptotic per-agent uplift `c` | saturation knee `τ` (turns) | adj R² |
|---|---:|---:|---:|
| **derived** (closed form) | `pool` = **8** | `pool/p_shared` = **17.8** | 0.9974 (0 params) |
| **fit** (free) | **8.12** | **18.45** | 0.99956 |

The two agree to ~1.5%. That is the difference between "we drew a curve through the
dots" and "we know **why** the dots are there." The per-agent slope climbs and
saturates exactly as coupon-coverage predicts:

| T | 1 | 5 | 10 | 20 | 30 | 50 |
|---|---:|---:|---:|---:|---:|---:|
| per-agent uplift (turns/agent) | 0.26 | 1.83 | 3.38 | 5.41 | 6.55 | 7.55 |

The total `shared_saved(T,A)` is near-perfectly **bilinear** (`≈0.966·T·A`, adj
R²=0.99967) — a read-only fleet caches ~everything, so savings scale with the call
count T·A, while the *cross* component is the saturating coupon term above.

---

## §4 — The catch: lookers vs bookers (the most important finding)

![write-rate crossover and shared-pool slope law](../../experiments/fleet/fleet-axes.png)

The shared cache lives under a **soundness rule**: a cache hit must equal a fresh
call, so **any write bumps the world-version and invalidates the cache** (the kernel
never serves a stale read — `vdso.Emit`). Today that bump is **global**: one write
clears the *whole* shared cache, not just the one key that changed. In an
interleaved fleet, **one agent's write erases every other agent's warmed reads** — a
cost the isolated baseline (whose agents mostly never write) does not pay.

The left plot sweeps the fleet write rate at A=50, T=30. The crossover is **sharp**:

| write rate | 0% | 0.25% | 0.5% | **0.75%** | **1%** | 2% | 5% | 30% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cross_uplift (A=50) | **+322** | +145 | +62 | **+14** | **−23** | −84 | −126 | −47 |

**Even a ~1% fleet write rate flips cross-agent sharing from a big win to a net
loss.** This is the real architectural result, and it forks the recommendation:

| your fleet is mostly… | share the cross-agent cache? |
|---|---|
| **reading / looking up / researching / monitoring** | **yes** — clean win, grows ~linearly with fleet size (§1, §3) |
| **writing / booking / sending / editing** | **no** — the global invalidation back-fires (§2 write-heavy, this table) |

And it names the fix: **scoped invalidation** — evict only the key(s) a write
changed, not the whole cache. That fix is now **built, measured, and unit-proven**:
the vDSO carries a `Granularity` (`global` = v0.1 full flush · `namespace` =
per-resource-class · `resource` = per-entity; `internal/vdso/scope.go`), every sweep
runs under any eraser via `fleetbench --granularity`, and
`TestFleet_FinerEraserPushesCrossoverOut` asserts the recovery on the live kernel.
The eraser sweep (A=50, T=30, same generated work) makes the fix one glance:

![the finer eraser fixes the write-crossover](../../experiments/fleet/fleet-eraser.png)

| write rate | **global** (v0.1) | **namespace** | **resource** |
|---:|---:|---:|---:|
| 0%   | +323 | +323 | +323 |
| 0.5% | +59  | +281 | +319 |
| 1%   | **−28** | +242 | +313 |
| 5%   | −132 | +100 | +271 |
| 10%  | −102 | +55  | +235 |

The **global** eraser is the only one that ever goes negative; the **resource**
eraser keeps **97% of the no-write uplift even at a 1% write rate** (313 of 323) and
is still strongly positive (+235) at **10%** writes. So the crossover the §4 title
warns about is a property of the *coarse v0.1 eraser*, not of cross-agent sharing
itself — the finer eraser pushes it out past any realistic fleet write rate. (Raw
data: `experiments/fleet/eraser/`, summary `eraser-summary.csv`.)

(The right panel of the axes figure at the top of this section is the catalog-size
law: per-agent uplift
tracks `slope = pool` until the pool outgrows what T turns can cover, then saturates
— `min(pool, routes covered in T)`, exact for pool ≤ 8.)

---

## §5 — Caveats (read before quoting any number)

- **What a "turn" is here.** A `cross_uplift` of +370 is **370 tool round-trips the
  fleet served from a peer's cached result** instead of dispatching to the tool
  backend, priced at the same per-turn `CostModel` as the single-agent turn-tax. It
  assumes a ReAct-style loop where each tool call is a round-trip. It is **not** 370
  saved model *reasoning* turns; it is saved tool round-trips (and their latency /
  re-prime / re-consume tokens).
- **Regime — but MORE portable than the §3.1 KV story.** This is **harness-level
  result caching** (the kernel/gateway), not in-tensor KV splicing. So unlike
  TURN-TAX §3.1's "self-host only" upside, cross-agent tool-result dedup **is
  available to an API consumer** who fronts their fleet with **one fak gateway** —
  provided (a) the fleet shares that gateway, (b) the reads are genuinely identical
  (content-addressed), and (c) the workload is read-heavy (§4).
- **This is a happy-path (efficiency) axis only.** Same two-axes discipline as the
  stochastic harness: the fleet workload carries **no poison and no deny**. The
  kernel's **safety floor** (keeping injections out of context, refusing destructive
  ops) is a separate, completion-integrity axis, measured elsewhere
  (`TURN-TAX-RESULTS.md §1`, `LIVE-RESULTS.md`), and holds whether or not this
  caching is on. Cross-agent *safety* interaction (could agent 1 poison agent 2 via
  the shared cache?) is its own axis — and is structurally bounded here because the
  ctx-MMU pages a quarantined result out **before** it is cached, so a shared serve
  hands back the paged-out pointer, not live poison; the durable form of that
  guarantee is the recall/canon line.
- **Determinism.** Every cell is seeded; the same `(profile, T, A, trials, seed)`
  reproduces the identical surface byte-for-byte (`TestFleet_Determinism`).

---

## §6 — The fleet-scale ceiling (the analytic complement)

The kernel sweep is the **grounded floor**: turns the kernel **verifiably** deletes
on real syscalls, with a saturating law. Its analytic complement,
`inline_tool_roi.py` (now extended to a 2-D **agents N × calls-per-agent K**
surface), is the **fleet-scale ceiling**: the GPU cold-KV **re-prefill $** a
two-pass loop burns, which is **bilinear and UNBOUNDED** in (N, K) — no saturation.
That contrast is the point: the two axes the kernel sweep shows *saturating* (a
finite shared catalog, a finite per-session error budget) are the same two the
fleet-$ tax shows *compounding*. The kernel deletes a **bounded** tax per session,
deterministically; the unbounded tax is the self-host-regime ceiling the same
construction reaches for (with the §3.1 regime caveats). See
`inline-tool-roi-results.md`.

---

## §7 — Bottom line

- **The agent-count axis is real and measured, not modeled** (§ intro, §2): the
  cross-agent uplift is the kernel's own tier-2 VDSOHits, proven by a shared-vs-
  isolated path-swap, with an **exactly-zero** anti-inflation control across 2,500
  cells.
- **The surface has a derived closed form** (§3): `cross_uplift = (A−1)·pool·(1−(1−
  1/pool)^(p_shared·T))` predicts the kernel data at **adj R²=0.997 with zero free
  parameters** — linear in agents, coupon-collector-saturating in turns. The fleet
  win is calculable, not just observed.
- **It only holds for read fleets** (§4): a **~1% write rate** flips it from a big
  win to a net loss, because the kernel's world-bump invalidation is global. That is
  the honest negative result **and** the concrete next build (scoped invalidation).
- **The visuals carry the story for any reader** (`visuals/fleet-sweep-*.svg`,
  `experiments/fleet/fleet-heatmap.png`, `fleet-axes.png`), and
  `FLEET-SWEEP-EXPLAINED.md` says all of the above with no math.
