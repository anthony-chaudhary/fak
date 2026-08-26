# Token confidence for selective small-model adjudication

**Run date:** 2026-08-16 UTC
**Hardware:** one NVIDIA L4-class accelerator
**Models:** Qwen2.5-0.5B-Instruct draft and Qwen2.5-1.5B-Instruct adjudicator, FP16 and simultaneously resident
**Raw artifact:** [`2026-08-16-selective-adjudication-confidence.json`](../benchmarks/model-sensitivity/2026-08-16-selective-adjudication-confidence.json)
**Raw SHA-256:** `a0838639669b2ad11f9426364b4d27410e54514491b2fb994ab9e4e0b1244e9c`

## Verdict

The 0.5B draft's grammar-filtered token confidence carried useful but imperfect semantic signal. Routing the 14 lowest-mean-log-probability calls to the 1.5B adjudicator reproduced the always-on cascade's **22/24** score while using 1,261 rather than 1,547 output tokens. Routing the 10 lowest-mean-margin calls reached **20/24**, matching direct 1.5B correctness with 1,130 tokens and about the same median warm path latency as direct routing.

These are retrospective frontier points on the same 24 examples, not deployment-ready thresholds. They show that a non-oracle signal can rank many misses early; they do not estimate held-out calibration or production prevalence.

## Question

The preceding always-on 0.5B → 1.5B cascade gained two exact calls over direct 1.5B, but approximately doubled latency and output tokens. Structural validation cannot trigger selectively because constrained decoding makes every draft structurally valid. This study asks whether confidence available *during the draft decode*, before correctness is known, can prioritize adjudication.

## Protocol

The model, catalog, grammar, prompt, direct baseline, and candidate-aware adjudication prompt match the preceding cascade study. The added draft-side signals are computed after grammar prefix filtering at each selected token:

- mean selected-token log probability;
- minimum selected-token log probability;
- sequence log probability;
- mean top-1 versus top-2 allowed-token logit margin;
- minimum top-1 versus top-2 allowed-token margin.

Lower confidence is routed first. Each signal's artifact contains all 25 route-fraction points, from zero through all 24 calls. Correctness is used only after generation to score the resulting frontier—not as an input to routing.

Three complete deterministic rounds were run. Direct and adjudicated call order alternated by task and round. Both models remained resident on one accelerator; no network or queue handoff was included.

## Base routes

All three rounds reproduced the preceding result:

| Route | Exact correct | Output tokens / 24 | Warm median path latency |
|---|---:|---:|---:|
| 0.5B draft | 13/24 | 787 | 998-1,004 ms |
| Direct 1.5B | 20/24 | **708** | 1,120-1,130 ms |
| Always-on adjudication | **22/24** | 1,547 | 2,217-2,234 ms |

## Confidence discrimination

AUROC treats a higher confidence value as predicting a correct draft:

| Draft signal | Correct-vs-incorrect AUROC |
|---|---:|
| Minimum margin | **0.836** |
| Mean log probability | 0.832 |
| Mean margin | 0.804 |
| Sequence log probability | 0.748 |
| Minimum log probability | 0.685 |

The sample has only 13 correct and 11 incorrect drafts. Values describe this exact catalog; no confidence interval or held-out claim is implied.

## Selective frontier

Useful observed operating points from the warm final round:

| Signal | Calls adjudicated | Final correct | Output tokens / 24 | Median path latency |
|---|---:|---:|---:|---:|
| None | 0/24 | 13/24 | 787 | 1,004 ms |
| Mean margin | 8/24 | 18/24 | 1,059 | 1,090 ms |
| Mean margin | 10/24 | **20/24** | 1,130 | 1,118 ms |
| Mean log probability | 11/24 | 20/24 | 1,163 | 1,202 ms |
| Mean log probability | 12/24 | 21/24 | 1,198 | 1,596 ms |
| Mean log probability | 14/24 | **22/24** | **1,261** | 2,009 ms |
| Sequence log probability | 15/24 | 22/24 | 1,296 | 2,119 ms |
| Minimum margin | 20/24 | 22/24 | 1,438 | 2,212 ms |
| Always-on | 24/24 | 22/24 | 1,547 | 2,217 ms |

At 14/24 routed, mean log probability saved 286 output tokens (18.5%) versus always-on adjudication while preserving its score. Median latency fell by about 208 ms (9.4%), but remained about 889 ms slower than direct 1.5B.

At 10/24 routed, mean margin matched direct 1.5B's 20/24 score and median latency, but used 422 more output tokens because every request still paid for the 0.5B draft. It is therefore not a net efficiency win over direct routing on this workload.

## Ranking anatomy

Mean log probability ranked nine adjudicator-correctable draft misses within its first 14 calls. It also routed three already-correct refund calls before several misses, showing imperfect calibration. The two failures shared by draft/adjudicator remained unfixable regardless of routing fraction:

- exact order lookup was interpreted as order search;
- `"Typo on homepage"` was shortened to `"Typo"`.

The frontier cannot exceed 22/24 because the adjudicator itself does not fix those calls.

Minimum margin had the highest AUROC but a poorer practical top-of-list frontier: it required routing 20 calls to reach 22/24. AUROC alone therefore does not select the operating policy; the ordering of the particular repairable misses matters.

## Interpretation

1. **Confidence is a usable routing feature, not a correctness certificate.** Mean log probability concentrated most repairable misses early, but also routed correct drafts and left two shared errors.
2. **Selective routing improves the cascade, not the best alternative.** It reduced always-on cost, yet the 22/24 point remained slower and less accurate than the witnessed 24/24 SmolLM2-1.7B concise route.
3. **Matching direct correctness did not match direct net cost.** The 10-call mean-margin point had similar median latency but 60% more output tokens than direct 1.5B.
4. **Threshold selection would overfit this sample.** Choosing “14” after seeing these labels is descriptive. A deployable threshold requires a separate calibration set and held-out evaluation.
5. **Grammar-filtered confidence has a specific meaning.** Probabilities and margins are normalized over tokens allowed by the JSON grammar at each step, not the model's unconstrained vocabulary distribution.
6. **Semantic fail-closed behavior remains unsolved.** Confidence can prioritize review, but no tested threshold proves that an unrouted call is safe to execute.

## Reproduction and validation

The raw artifact captures model and weight provenance, exact schema/catalog/tasks/prompts, every token's selected ID/text, grammar-filtered log probability, top-two margin, allowed-token count, complete generated outputs, routes, timings, and all 25 frontier points for each signal.

Post-capture checks established:

- three rounds, 24 rows per round, and all declared route summaries;
- every output, parse, correctness result, and output-token count reproduces across rounds;
- each confidence frontier contains exactly one point for every routed count 0..24;
- frontier correctness, output tokens, median warm latency, and routed IDs recompute from final-round rows;
- corrected and regressed outcomes match the underlying route records;
- local and accelerator SHA-256 digests match.

## Bounded deployment conclusion

Mean draft log probability is the strongest observed selective-adjudication policy for the desired high-correctness end of this small sample: 14/24 routed preserved 22/24 while reducing always-on cost. It does not beat direct 1.5B on cost, does not beat SmolLM2-1.7B on correctness, and must not be promoted without held-out calibration. The next proof should freeze a confidence threshold on a separate calibration catalog and evaluate it on unseen requests/catalog perturbations.
