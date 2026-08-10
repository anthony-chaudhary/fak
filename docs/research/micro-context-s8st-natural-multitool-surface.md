# S8s/S8t — natural multi-tool admission decision surface

**Verdict: `decision_surface`, not a universal adaptive winner.** Natural held-out windows across five evidence classes produced a real quality/cost tradeoff: the deterministic planner was fastest but only 8/10 correct; fixed cascade and both semantic policies were 10/10. At observed tool latency, fixed cascade was fastest among quality-qualified arms. At a modeled 10× tool-cost regime, semantic two-stage became fastest and opened 8 tools rather than 40.

## Corpus and independent fold

Twenty naturally phrased operator questions cover:

- packet-only / no tool;
- committed source search;
- mutable GitHub issue state;
- current git commit ancestry;
- named documentation reads.

Opaque alternating IDs freeze 10 tune and 10 test records, with two held-out records per class. Independent `gpt-5.6-sol` and Groq `llama-3.3-70b-versatile` judgments are blind to the construction label. The strict fold requires both model-distinct votes and the construction contract to agree.

Result: 19/20 unanimous, model pairwise agreement 0.95. The sole dispute (`n01`, tune) remains excluded; every held-out record is unanimous. The verifier requires at least two unanimous examples in every class.

## Real tool receipts

Every held-out required tool was executed through its actual read seam:

- `git grep` for source search;
- bounded GitHub issue-state REST/`gh` fallback;
- `git merge-base --is-ancestor` for commit state;
- bounded committed-file reads for docs;
- no dispatch for packet-only questions.

Observed receipts and latency are preserved per record. The cheap (0.1×) and expensive (10×) regimes scale those witnessed latencies without changing labels; they are explicitly modeled rather than presented as additional live measurements.

## Decision surface

| regime | policy | exact | tools | mean wall ms | mean work ms | prompt/output |
|---|---|---:|---:|---:|---:|---:|
| cheap 0.1× | deterministic | 8/10 | 6 | **20.1** | **20.1** | 0 / 0 |
| cheap 0.1× | fixed cascade | **10/10** | 40 | **102.8** | **102.8** | 0 / 0 |
| cheap 0.1× | semantic two-stage | **10/10** | 8 | 5,625.2 | 5,625.2 | 1,626 / 644 |
| cheap 0.1× | selective parallel | **10/10** | 40 | 5,604.6 | 5,707.4 | 1,626 / 644 |
| observed | deterministic | 8/10 | 6 | **201.4** | **201.4** | 0 / 0 |
| observed | fixed cascade | **10/10** | 40 | **1,028.5** | **1,028.5** | 0 / 0 |
| observed | semantic two-stage | **10/10** | 8 | 5,810.3 | 5,810.3 | 1,626 / 644 |
| observed | selective parallel | **10/10** | 40 | 5,604.6 | 6,633.1 | 1,626 / 644 |
| expensive 10× | deterministic | 8/10 | 6 | **2,014.0** | **2,014.0** | 0 / 0 |
| expensive 10× | fixed cascade | **10/10** | 40 | 10,284.7 | 10,284.7 | 0 / 0 |
| expensive 10× | **semantic two-stage** | **10/10** | **8** | **7,661.5** | **7,661.5** | 1,626 / 644 |
| expensive 10× | selective parallel | **10/10** | 40 | 8,217.4 | 15,889.3 | 1,626 / 644 |

Quality is gated before latency: deterministic cannot be called the winner because it misses two evidence classes on held-out records. Among 10/10 arms:

- cheap and observed tools: fixed cascade wins because semantic selection costs several seconds;
- expensive tools: semantic two-stage wins, reducing mean wall 25.5% versus fixed cascade and opening 80% fewer capabilities;
- selective parallel buys lower tail than serial selection only when tool latency approaches selector latency, but spends the most total work and authority.

## General micro-window interpretation

The policy should be chosen per bounded window and cost regime, not globally:

1. deterministic structural routing first where contracts are explicit;
2. fixed cascade when tools are cheap and quality requires broad evidence;
3. semantic admission when tool cost/authority exceeds selector cost and evidence need is predictable;
4. selective parallelism when tail latency matters enough to pay duplicated work;
5. typed folds and provenance regardless of admission policy.

This supports the user's general pattern while rejecting the strongest overclaim. Micro-windows provide a common scheduling unit for large inputs, filters, models, and tools; they do not imply one scheduler dominates every workload.

## Reproduction

```powershell
go run ./cmd/microcontextdemo -verify-natural-multitool-corpus experiments/microcontext/s8s-natural-multitool-corpus-2026-08-10.json -verify-natural-multitool-fold experiments/microcontext/s8s-natural-fold-2026-08-10.json
go run ./cmd/microcontextdemo -verify-natural-multitool-surface experiments/microcontext/s8t-natural-multitool-surface-2026-08-10.json
go test ./cmd/microcontextdemo -run 'TestNatural' -count=1
```
