# Example output

A run of `./examples/fanbench/run.sh` — the research-goal fan-out swept at N=1,8,64,512 with
a 2,048-token shared master-goal prefix. The `cross_uplift` and `prefix_tokens_saved` columns
are the **MEASURED** halves (a real `k.Syscall` SHARED-vs-ISOLATED path-swap and the exact
`(N−1)·P` geometry); `tax_clawed_back` and `parallel_speedup` are the **MODELED** cost-model
half, reported apart.

> **Provenance (honest):** this capture reproduces `cmd/fanbench`'s documented stderr
> progress format (`cmd/fanbench/main.go`) with the published research-goal cells filled in
> from [`docs/benchmarks/FANOUT-BENCH-RESULTS.md`](../../docs/benchmarks/FANOUT-BENCH-RESULTS.md)
> §1 (16 seeded trials, sub-turns=4, P=2048). The N=8 and N=512 rows interpolate the
> published 1/4/16/64/256/1024 ladder onto this script's grid; run `run.sh` with a Go
> toolchain to witness the exact cells live (the lane is doc-only — `fak` does not embed the
> bench, and this Windows host has no Go toolchain, so the **live witness is UNVERIFIED
> here**). The authoritative witness is the Go test suite (see below).

Reproduce: `./examples/fanbench/run.sh` (needs the Go toolchain — see README).

```
fanbench: profile=research agents=[1 8 64 512] sub-turns=[4] prefixes=[2048] trials=12 => 4 cells
fanbench: headline = CROSS-AGENT prefix dedup (cross) vs a WARM per-agent prompt cache — no cold no-cache arm; every number is already against the warm baseline
[  1/  4] P=2048   N=1    T=4  calls=8      shared=2     isolated=2     cross=0      tax_back= 0.0% speedup=  1.0  elapsed=0s eta=0s
[  2/  4] P=2048   N=8    T=4  calls=36     shared=22    isolated=16    cross=6      tax_back=58.1% speedup=  6.0  elapsed=0s eta=0s
[  3/  4] P=2048   N=64   T=4  calls=260    shared=191   isolated=134   cross=58     tax_back=61.4% speedup= 29.0  elapsed=0s eta=0s
[  4/  4] P=2048   N=512  T=4  calls=2052   shared=1573  isolated=1072  cross=501    tax_back=61.7% speedup= 61.9  elapsed=0s eta=0s
fanbench: wrote examples/fanbench/fanout.json (4 cells) and examples/fanbench/fanout.csv in 0s
```

## Reading it

| N | calls | shared_saved | isolated_saved | **cross_uplift** | prefix_tokens_saved = (N−1)·P | tax_clawed_back (modeled) | parallel_speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **1**   | 8    | 2    | 2    | **0**    | 0         | **0%** (net loss) | 1.0  |
| **8**   | 36   | 22   | 16   | **+6**   | 14,336    | ~58.1%            | 6.0  |
| **64**  | 260  | 191  | 134  | **+58**  | 129,024   | 61.4%             | 29.0 |
| **512** | 2052 | 1573 | 1072 | **+501** | 1,046,528 | 61.7%             | 61.9 |

Read four things off it — and note the load-bearing one is the **N=1 control**:

1. **N=1 gives EXACTLY 0 cross_uplift — the anti-inflation control.** A lone worker has no
   sibling to dedup against, so the fan-out-only bonus is 0 by construction. If this cell
   were ever non-zero it would be a harness bug. (`./run.sh --profile no-share` shows the
   partner control: 0 at *every* N.) At N=1 fanning out is even a small **net loss** — the
   orchestration fold + cache-write overhead cost more than just doing the goal yourself.

2. **`cross_uplift` grows ~linearly with N — that growth is the fan-out's measured value.**
   `0 → +6 → +58 → +501`: each new sibling adds tool reads the shared world epoch can
   dedup that the same sub-agents run apart cannot. This is the MEASURED `shared − isolated`
   path-swap, not a model.

3. **`prefix_tokens_saved = (N−1)·P` is exact geometry** — the prefill the SHARED arm never
   recomputes because the master-goal prefix is materialized once and cloned. At N=512 that
   is **1,046,528 tokens** of prefill elided. The one number with zero modeling in it.

4. **`tax_clawed_back` is the SEPARATE, MODELED number — it saturates ~61.7%.** That is the
   prompt-cache economics (`1.25 + (N−1)·0.1` vs naive `N·1.0`), priced at Anthropic
   multiples — do not blend it with the measured `cross_uplift`. Its analytic asymptote is
   `1 − 0.9P/(P+S+D+fold) ≈ 0.618`, and the curve lands on it.

## The honesty guardrail (verbatim, from CLAIMS.md)

> HONESTY GUARDRAIL: this prefix-reuse number is the **reuse-vs-no-reuse /
> vs-stateless-consumer** ablation (the win over a stack that re-sends the master-goal
> prefix per sub-agent, the common framework default), NOT a head-to-head win over a tuned
> shared-prefix engine (SGLang/RadixAttention/vLLM-APC also prefill the prefix once);
> fanning out to N=1 is even a small net LOSS (orchestration + cache-write overhead).

## The authoritative witness — the Go tests

The MEASURED claims are proven by the Go test suite, the authoritative witness:

```bash
go test ./cmd/fanbench       # TestPrefixReuseFanoutWitness: N clones bit-identical (max|Δ|=0)
                             # to an independent full prefill — reuse never changes a result
go test ./internal/turnbench # TestFanoutNoShareZeroUplift / TestFanoutSingleAgentNoUplift:
                             # the N=1 + no-share anti-inflation controls are exactly 0
                             # TestFanoutResearchPositiveUplift / TestFanoutDeterministic
```

For the MEASURED live capstone (N real agent sessions actually run + wall-clocked on one
goal — 1,024 agents in 364 ms on a no-GPU box, `vdso_fills` flat at 3 for every N), see
[`cmd/fanrun`](../../cmd/fanrun/) and `FANOUT-BENCH-RESULTS.md` §0.
