# Held-out selective adjudication under catalog perturbation

**Run date:** 2026-08-16 UTC
**Hardware:** one NVIDIA L4-class accelerator
**Models:** Qwen2.5-0.5B-Instruct draft and Qwen2.5-1.5B-Instruct adjudicator, FP16 and simultaneously resident
**Raw artifact:** [`2026-08-16-heldout-selective-adjudication.json`](../benchmarks/model-sensitivity/2026-08-16-heldout-selective-adjudication.json)
**Raw SHA-256:** `b215141aae5fa4bc17bf2ebdbf420f7bff57d24c3459a9e6ca57ace498f7e400`

## Verdict

The frozen selective-adjudication policy did **not** generalize. On unseen request paraphrases with canonical catalog order it routed 15/24 and achieved 16/24, matching always-on adjudication but far below the calibration score of 22/24. Under three catalog-order perturbations it achieved only 8/24, 10/24, and 7/24. Direct constrained 1.5B routing scored 14/24, 13/24, and 17/24 on those same permutations and was the strongest aggregate route.

The failure is larger than confidence calibration drift: catalog order changed the generated semantics themselves. Only 2/24 draft calls, 14/24 direct calls, and 6/24 adjudicated calls were invariant across all four orderings. A routing threshold cannot rescue a base route whose prediction changes this much when an equivalent catalog is reordered.

## Frozen policy and leakage boundary

Before any held-out generation, the preceding calibration artifact fixed:

- calibration artifact SHA-256: `a0838639669b2ad11f9426364b4d27410e54514491b2fb994ab9e4e0b1244e9c`;
- signal: draft mean selected-token log probability after grammar filtering;
- rule: adjudicate when confidence is **≤ -0.1951635**;
- provenance: the calibration frontier's retrospective 14/24 point, whose boundary call was `email-01`; the next unrouted calibration value was -0.1881704.

The threshold is embedded in the runner and raw artifact. It was not changed after observing held-out labels.

## Held-out protocol

- Requests: 24 new paraphrases preserving the prior tasks' exact expected calls. None appears in the calibration request set.
- Catalog semantics: unchanged 24 tools and schemas.
- Catalog orders:
  1. canonical;
  2. reversed;
  3. deterministic shuffle with seed 1729;
  4. deterministic shuffle with seed 947.
- Draft: constrained 0.5B concise-contract call plus grammar-filtered token confidence.
- Selective route: candidate-aware constrained 1.5B adjudication only when the frozen confidence rule fires.
- Comparators captured on every row: draft-only, direct constrained 1.5B, and always-on candidate-aware adjudication.
- Decode: greedy, temperature 0, 128 generated-token cap.
- Execution: both models resident; synchronous in-process calls; no network or queue handoff.

All comparator calls were generated to score the experiment. Selective token and latency totals count adjudication only on policy-routed rows.

## Results by catalog order

| Catalog order | Draft correct | Direct 1.5B | Always-on adjudication | Frozen selective | Routed | Selective tokens |
|---|---:|---:|---:|---:|---:|---:|
| Canonical | 10/24 | 15/24 | **16/24** | **16/24** | 15/24 | 1,292 |
| Reversed | 2/24 | **14/24** | 9/24 | 8/24 | 16/24 | 1,366 |
| Shuffle 1729 | 5/24 | **13/24** | 11/24 | 10/24 | 16/24 | 1,415 |
| Shuffle 947 | 4/24 | **17/24** | 11/24 | 7/24 | 9/24 | 1,129 |
| **Aggregate** | **21/96** | **59/96** | **47/96** | **41/96** | **56/96** | **5,202** |

Aggregate output tokens:

- draft only: 3,353;
- direct 1.5B: **2,767**;
- always-on adjudication: 3,147 adjudicator tokens, or 6,500 including the required drafts;
- frozen selective: 5,202 total-path tokens.

Direct 1.5B was both more accurate and lower-token than the frozen selective route over the full held-out matrix.

## Confidence transfer

Correct-versus-incorrect draft AUROC for mean log probability varied substantially:

| Catalog order | Held-out AUROC | Routed draft misses | Routed draft hits | Unrouted misses |
|---|---:|---:|---:|---:|
| Canonical | 0.750 | 12 | 3 | 2 |
| Reversed | 0.500 | 14 | 2 | 8 |
| Shuffle 1729 | 0.811 | 14 | 2 | 5 |
| Shuffle 947 | 0.650 | 7 | 2 | 13 |

The frozen threshold routed between 9 and 16 calls depending only on equivalent catalog ordering. It preserved every routed draft hit because the adjudicator did not regress those cases, but missed many low-quality drafts under the reversed and seed-947 orders.

## Catalog-order sensitivity

Across the four equivalent orderings:

- 0.5B draft parsed calls were invariant on only **2/24** tasks;
- direct 1.5B calls were invariant on **14/24**;
- candidate-aware adjudicated calls were invariant on **6/24**.

Grammar-constrained decoding guarantees that every call belongs to the legal schema language. It does not make the distribution over legal branches permutation-invariant. The tool list's serial order acts as an unintended model input and can dominate both semantic choice and confidence.

The candidate-aware adjudicator was less order-stable than direct 1.5B because it inherited catalog-order effects through both the catalog and the unstable 0.5B candidate. Under perturbation, always-on adjudication was often worse than direct 1.5B.

## Interpretation

1. **The calibration frontier was optimistic.** The same policy fell from 22/24 calibration correctness to 16/24 on canonical held-out paraphrases.
2. **Catalog order is part of the effective prompt contract.** Treating reordered tools as equivalent is not safe for these small models.
3. **Selective routing did not produce a net gain.** Aggregate direct 1.5B routing was 18 calls more accurate while using 2,435 fewer output tokens than selective routing.
4. **Confidence ranking was not stable enough.** AUROC ranged from chance (0.500) to 0.811, and routed fraction varied from 38% to 67% under equivalent schemas.
5. **Always-on adjudication also failed the robustness test.** Candidate context helped canonical requests but hurt relative to direct routing in every perturbed order.
6. **Structural validity remained misleading.** Every arm was 24/24 structurally valid in every order despite large semantic swings.
7. **No deployment threshold is supported.** The frozen threshold failed held-out evaluation and should be rejected, not retuned on these labels.

## Reproduction and validation

The raw artifact includes the immutable calibration digest and threshold, all held-out requests and expected calls, exact catalog permutations, schema/model/weight provenance, every token confidence trace, every draft/direct/adjudicated output, policy decisions, timings, and summaries.

Post-capture validation established:

- four catalog runs and 24 held-out rows per run;
- the embedded calibration digest and threshold match the committed calibration artifact;
- policy routing equals `mean_logprob <= -0.1951635` on every row;
- selective correctness, routed IDs, corrected/regressed counts, output tokens, and median path latency recompute from row records;
- every draft/direct/always-on summary and aggregate total recomputes independently;
- order-invariance counts compare parsed calls by task across all four runs;
- local and accelerator SHA-256 digests match.

## Bounded deployment conclusion

Reject the tested confidence-threshold cascade for deployment. The first robustness requirement is now catalog-order stability, not finer threshold tuning. On this held-out matrix, direct constrained Qwen2.5-1.5B is the best of the tested 0.5B/1.5B routes, but its 59/96 aggregate score is itself non-admissible. The previously witnessed SmolLM2-1.7B 24/24 result was on one catalog order and must also undergo the same permutation/paraphrase test before being treated as robust.
