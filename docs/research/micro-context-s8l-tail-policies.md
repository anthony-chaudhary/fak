# Micro-context S8l: live cancellation and partial-fold policies

**Status:** observed live endpoint policy matrix<br>
**Issues:** #6160, parent #6033<br>

## Question

Does splitting work into micro-contexts actually solve tail latency, or only make cancellation and partial progress possible?

S8k observed a five-minute straggler. S8l adds the missing execution policy and compares four live strategies on the same 16 held-out records:

- **wait-all:** no per-window policy deadline beyond a 45-second task cap;
- **deadline-abstain:** 12-second per-window deadline; timeout folds as typed abstention;
- **sufficiency-stop:** cancel the task after 12 confirmed records and fold remaining records as abstentions;
- **bounded-hedge:** after five seconds, open at most one duplicate and accept the first valid result.

All policies use four workers, two trials, the same endpoint/model, and the frozen 0.95 tune-selected abstention threshold.

## Receipts and semantics

Every logical record emits one receipt with status, open/hedge flags, winner, latency, usage, reason, and read-back state. The fold distinguishes:

- confirmed;
- timed out / cancelled in flight;
- cancelled before open;
- abstention synthesized because no confirmed fact exists.

Cancelled-but-billed remains `unknown`: this endpoint returns usage only for completed streams. The matrix refuses to equate “no usage response” with “free cancellation.”

## Live results

Totals cover two trials (32 logical records per policy).

| Policy | Mean wall | Completed | Timed out / cancelled in flight | Hedges | Hedge wins | Prompt tokens | Output tokens | First-trial exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Wait all | 25.90 s | 32 | 0 | 0 | 0 | 21,224 | 5,559 | **4/16** |
| Deadline → abstain | 27.38 s | 29 | 3 | 0 | 0 | 19,273 | 5,305 | 2/16 |
| Sufficiency stop at 12 | **18.17 s** | 24 | 8 | 0 | 0 | **15,796** | **3,642** | 3/16 |
| Bounded hedge | 25.39 s | 32 | 0 | 18 | 2 | 21,288 | 5,099 | 2/16 |

## Findings

1. **Decomposition is an enabling mechanism, not a tail guarantee.** Wait-all is again governed by the slowest window.
2. **Sufficiency stopping is the only policy with a clear resource/wall reduction here:** about 30% lower wall and 26% fewer prompt tokens than wait-all, at a one-record exact-quality loss.
3. **A fixed per-window deadline does not necessarily reduce all-of-set wall.** With four workers, timed-out windows consume slots before later work starts; deadline policy is slightly slower than wait-all in this sample.
4. **Hedging is not free.** Eighteen duplicate attempts produce only two hedge wins and no wall improvement. Usage from cancelled losing streams is unavailable, so reported wasted tokens are a lower bound of zero—not proof of zero waste.
5. **No policy satisfies the strict 16/16 floor.** The matrix identifies a latency/quality tradeoff, not a deployable winner.

## Steelmanned perspectives

- **Wait-all advocate:** it preserves every available answer and scores best here. For offline completeness-critical work, 26 seconds may be preferable to partial results.
- **Deadline advocate:** fixed deadlines bound individual damage even when they do not improve this small all-of-set schedule. A queue-aware scheduler could prioritize unopened work instead of letting stragglers hold slots.
- **Early-stop advocate:** many aggregate questions do not require every record. The count-based proxy demonstrates savings, but a real sufficiency predicate must be task-specific and independently checkable.
- **Hedging advocate:** hedging helps heavy-tailed services when duplicates are launched selectively. The five-second universal hedge threshold is intentionally simple and over-hedges this workload.
- **Cost skeptic:** cancelled request billing is unknown, so neither deadline nor hedge savings should be described in dollars.
- **Security/provenance advocate:** partial folds are safe only because missing records remain explicit abstentions with receipts; absence is never silently interpreted as false.

## Reproduce

```powershell
go run ./cmd/microcontextdemo `
  -tail-policy-packet experiments/microcontext/s8i-semantic-packet-2026-08-10.json `
  -tail-policy-gold experiments/microcontext/s8i-semantic-gold-2026-08-10.json `
  -tail-policy-output experiments/microcontext/s8l-tail-policy-matrix-2026-08-10.json `
  -tail-policy-endpoint $env:OPENAI_BASE_URL `
  -tail-policy-api-key $env:OPENAI_API_KEY `
  -tail-policy-model gpt-5.6-sol `
  -tail-policy-endpoint-class separate-openai-compatible-live `
  -tail-policy-hardware provider-managed-undisclosed `
  -tail-policy-trials 2 `
  -tail-policy-workers 4 `
  -tail-policy-window-ms 12000 `
  -tail-policy-task-ms 45000 `
  -tail-policy-hedge-ms 5000 `
  -tail-policy-sufficiency 12

go run ./cmd/microcontextdemo `
  -verify-tail-policy experiments/microcontext/s8l-tail-policy-matrix-2026-08-10.json
```

## Decision boundary

S8l validates the user-facing idea’s stronger form: micro-context windows make selective stopping, typed partial progress, retries, and hedges mechanically possible. They do not make those policies beneficial by default. On this workload, early stop is the only promising tradeoff, while wait-all retains best quality. #6111 must preserve both facts and still report `not-yet` at the strict floor.
