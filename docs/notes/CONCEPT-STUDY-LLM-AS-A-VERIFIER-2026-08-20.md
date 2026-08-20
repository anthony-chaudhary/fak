---
title: "LLM-as-a-Verifier study: same-model selection, evidence gaps, and the FAK replay seam"
description: "Pinned deep study of llm-as-a-verifier's Terminal-Bench 2.1 self-verification path, reproduced baselines, provenance limits, source defects, and a witnessed FAK adoption map."
---

# LLM-as-a-Verifier: same-model selection, evidence gaps, and the FAK replay seam

## Verdict

The repository contains a real best-of-N selection implementation and a useful
445-trajectory Terminal-Bench 2.1 candidate pool. The checked-in labels reproduce
the reported Pass@1 and oracle columns exactly. They do **not** reproduce the
headline verifier columns: the score caches, task selections, error ledger,
multi-seed ledger, and uncertainty calculation are absent. Treat `86.5% ± 1.1%`
(Bo3) and `88.0% ± 0.6%` (Bo5) as **upstream-reported**, not independently
reproduced.

The strongest transferable idea is not “let a model certify itself.” It is a
bounded selector: generate several candidates, compare observable evidence under
separable criteria, cancel prompt-slot bias, and spend comparison budget around
likely winners. FAK already has the runtime generation plan, an absolute judge,
provider-prefix reuse, and a stronger witness hierarchy. Its missing first spine
is an offline, provenance-bound selection replay that measures quality, oracle
headroom recovered, billed work, errors, and seed sensitivity before a selector
is chosen for live branch-and-prune. That spine is filed as
[#8230](https://github.com/anthony-chaudhary/fak/issues/8230).

## Observation identity

| Field | Witnessed value |
|---|---|
| Observed at | `2026-08-20T09:43:27-07:00` |
| Repository | [`llm-as-a-verifier/llm-as-a-verifier`](https://github.com/llm-as-a-verifier/llm-as-a-verifier) |
| Pinned revision | [`8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770`](https://github.com/llm-as-a-verifier/llm-as-a-verifier/commit/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770) |
| Source event | `2026-08-20T00:01:30-07:00`, merge of PR #16 |
| Remote read-back | `origin/HEAD` still equaled the pinned revision at the observation time |
| Adoption signal | 2,438 stars, 188 forks; discovery signal only |
| Release state | package declares `0.2.0`; no Git tag or GitHub Release exists |
| Paper | [arXiv:2607.05391v2](https://arxiv.org/abs/2607.05391), dated 2026-07-07 |
| FAK comparison state | `internal/modelroute@r84+ga46fc02ec9`, `internal/trajctl@r34+gaaa76e85ac`, `internal/terminalbench@r20+gcad27c07b7` |

Refresh this note when upstream main moves, PR #8/#13/#17 merges or closes, a
score/result ledger appears, or FAK changes the `modelroute` best-of-N seam.

## Feynman-simple value frame

- **For:** an agent operator who can afford several candidate trajectories but
  not a full execution-based check for every one.
- **Problem:** Pass@1 leaves real oracle headroom, while a coarse model judge can
  tie, over-trust narration, or spend quadratically on pairwise ranking.
- **Today:** FAK can generate/score candidates and can control trajectories, but
  it has no frozen candidate-pool replay that tells us which selector actually
  recovers quality per billed verifier unit.
- **Better because:** a provenance-bound replay makes selector choice a measured
  decision, with the current absolute judge as the baseline and pairwise/logprob
  methods admitted only when they clear quality and cost gates.
- **Witness:** a deterministic task-level artifact reports Pass@1, oracle,
  selected success, recovered headroom, query/token/error counts, seed, and every
  selected task ID from immutable candidate and score inputs.

## Problem centrality and P1-P4

**Enabling.** This does not replace FAK's core kernel checkpoint. It supplies the
measurement gate needed before the live branch-and-prune feature (#4235) spends
extra decoding and judge calls.

- **P1 managed context:** candidate traces and rubrics need stable, hash-bound
  identities; prompt-prefix layout should make shared work reusable.
- **P2 net-true efficiency:** report recovered oracle headroom together with all
  generator and verifier work; more selected successes alone are not a gain.
- **P3 bounded adaptation:** selection is optional, budgeted, seeded, and unable
  to override stronger execution evidence or the policy floor.
- **P4 integrated operations:** one ledger must join model/config identity,
  candidate hashes, score hashes, selections, failures, usage, and claim status.

## What the upstream path actually does

### 1. Load a fixed candidate pool without showing labels to the verifier

The Terminal-Bench loader extracts the task, keeps agent narration, keystrokes,
and terminal observations, and holds `reward` back for evaluation
([`loaders.py:29-102@8db8a11`](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/llm_verifier/loaders.py#L29-L102)).
Its criterion note explicitly says to trust literal terminal evidence over the
agent's own success claim, then separates specification compliance, output match,
and unresolved error signals
([`terminal_bench.md:3-19@8db8a11`](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/criteria/terminal_bench.md#L3-L19)).

The self-verification fixture is 89 tasks × 5 trajectories. Every record names
`mini-swe-agent`, `deepseek/deepseek-v4-flash`, and ATIF conversion schema
`atif-converted-1.0`; capture ran from 2026-08-08T21:32:31Z through
2026-08-09T01:41:38Z. It does not pin the mini-swe-agent revision, the
Terminal-Bench dataset revision, a model checkpoint/API revision, or the
generation command/config.

### 2. Spend verifier calls only where selection can change success

The runner credits all-pass tasks, cannot rescue all-fail tasks, and sends only
mixed-label “swing” tasks to the verifier
([`run.py:43-53@8db8a11`](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/scripts/run.py#L43-L53)).
That optimization uses labels only in the benchmark harness; it is not available
to a live unlabeled selector.

### 3. Turn token probabilities into pairwise continuous rewards

For each criterion and repeat, the verifier emits score tokens for candidates A
and B. The implementation integrates probability mass across a 20-token score
scale instead of trusting one sampled label, averages criteria and repetitions,
and swaps A/B on odd repetitions before remapping scores to candidate order
([`fine_grained_reward.py:663-707,751-792@8db8a11`](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/llm_verifier/fine_grained_reward.py#L663-L792)).
This requires a backend that exposes token logprobs.

### 4. Rank with a ring and pivot tournament

A seeded random Hamiltonian ring puts every candidate once in slot A and once in
slot B. Bradley-Terry soft wins from that pass choose the top `k` pivots; the
second phase compares non-pivots to pivots and pivots to each other, then selects
the highest mean win mass
([`pivot_tournament.py:28-92@8db8a11`](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/llm_verifier/pivot_tournament.py#L28-L92)).
The intended comparison growth is `O(Nk)` rather than full round-robin `O(N²)`.

The shipped Bo3 and Bo5 wrappers effectively pin `N=3/5`, `k=1`, three criteria,
`K=2`, and registry seed `0`
([Bo3](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/scripts/run_bo3.py#L32-L60),
[Bo5](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/scripts/run_bo5.py#L31-L59)).

### 5. Shape calls for prefix reuse

The large task/traces prefix precedes criterion-specific text. One request per
distinct prefix completes before the remaining criterion/repeat calls fan out,
and a fully cached run creates no client
([`fine_grained_reward.py:714-747,795-876@8db8a11`](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/llm_verifier/fine_grained_reward.py#L714-L876)).
Failures become nonpersistent neutral ties by default, so retries remain possible,
but the final report omits the error count.

## Reproduction audit

The official loader was run under WSL to preserve its POSIX path semantics. The
fixture-only columns reproduce as follows:

| Configuration | All-pass | Swing | All-fail | Derived Pass@1 | Derived oracle | README verifier claim |
|---|---:|---:|---:|---:|---:|---:|
| Bo3 | 58 | 24 | 7 | 79.4007% | 92.1348% | 86.5% ± 1.1% |
| Bo5 | 50 | 36 | 3 | 78.6517% | 96.6292% | 88.0% ± 0.6% |

From a POSIX checkout at the pin, the label-only witness is:

```bash
python3 - <<'PY'
from llm_verifier.loaders import load_terminal

tasks, _ = load_terminal({
    "agent_dir": "data/terminal_bench_2.1_trajs/mini-swe-agent_deepseek-v4-flash",
}, ".")
for n in (3, 5):
    rows = [[t["reward"] for t in trials[:n]] for _, trials in sorted(tasks.items())]
    all_pass = sum(all(x == 1 for x in row) for row in rows)
    all_fail = sum(all(x == 0 for x in row) for row in rows)
    swing = len(rows) - all_pass - all_fail
    pass1 = sum(sum(row) / len(row) for row in rows) / len(rows)
    oracle = sum(any(x == 1 for x in row) for row in rows) / len(rows)
    print(n, len(rows), all_pass, swing, all_fail, pass1, oracle)
PY
```

This confirms the candidate pool and the non-verifier baselines in
[`README.md:116-134@8db8a11`](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/README.md#L116-L134).
It cannot confirm selected accuracy:

- `cache/` and `results/` are ignored and absent; no task-level score or
  selection ledger is committed
  ([`.gitignore:20-22@8db8a11`](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/.gitignore#L20-L22)).
- `report()` emits one point estimate for one run; there is no bootstrap,
  multi-seed, standard-deviation, or confidence-interval routine
  ([`run.py:198-247@8db8a11`](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/scripts/run.py#L198-L247)).
- Commit [`115de305`](https://github.com/llm-as-a-verifier/llm-as-a-verifier/commit/115de305f23ed89bc42e86e010853c40059f3f7d)
  changed only the README rows from `85.4/88.8` to the current means and error
  bars; no supporting artifact or method accompanied it.
- The paper's Terminal-Bench result is a different experiment: Terminal-Bench
  2.0 candidates from Capy/GPT-5.5, verified by Gemini 2.5 Flash. Paper v2 predates
  the August 2.1 same-model extension and does not validate its headline.

Fresh-cache work is also less simple than the wrapper prose suggests. Bo3 makes
five logical comparisons per swing task but only four unique directed pairs;
Bo5 makes nine logical comparisons but eight unique pairs. With three criteria
and two repeats, that is 576 and 1,728 unique verifier calls respectively. The
overlap is cached, so it is not billed twice, but it **is aggregated twice**.

## Material source defects and negative knowledge

### The paper and code disagree on duplicate tournament edges

Paper v2 Algorithm 1 subtracts the ring-edge set from the pivot-round set. The
implementation's `pivot_round_pairs()` has no ring input and returns the overlap;
the runner then accumulates ring and pivot pairs separately. With `k=1`, one ring
edge is always double-weighted. This does not quantify the impact on the absent
published score ledger, but it means a FAK adaptation must define and test edge
deduplication before comparing results.

### The public no-cache API loses Phase A scores

At the pinned revision, `select(cache=None)` scores the ring, rebinds its local
score map to Phase B, then tries to aggregate the ring from the Phase-B-only map
([`__init__.py:182-231@8db8a11`](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/llm_verifier/__init__.py#L182-L231)).
Open [issue #14](https://github.com/llm-as-a-verifier/llm-as-a-verifier/issues/14)
and [PR #17](https://github.com/llm-as-a-verifier/llm-as-a-verifier/pull/17)
address it. Benchmark wrappers pass a persistent cache path, so this defect does
not directly disprove the README rows; it does prove that cached/uncached
selection equivalence needs a regression property.

### Cache identity is too weak for experiment provenance

Cache keys contain criterion ID, task name, candidate indices, and repetition —
not candidate/problem hashes, criterion text, prompt version, model, reasoning
settings, media, or data revision
([`fine_grained_reward.py:775-846@8db8a11`](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/llm_verifier/fine_grained_reward.py#L775-L846)).
Open PR #8 is the upstream hardening direction. FAK should content-address the
exact score-producing inputs from its first implementation.

### “Same model” is not enforced by the scripts

The candidate blobs name DeepSeek V4 Flash, but the wrappers do not pass a
verifier model. Environment precedence chooses `OPENAI_BASE_URL` before
`DEEPSEEK_API_KEY`; an ambient endpoint can silently turn the run into a
cross-model experiment. A FAK ledger must pin and attest generator and verifier
identities independently, even when they are intended to match.

### Upstream regression coverage is absent at the pin

The pinned tree has no tracked tests or CI workflow. Four open PRs (#8, #11,
#13, #17) independently add tests around active defects. Defensive branches are
not the same as regression witnesses.

## Worldview and limits

This framework serves researchers and agent builders who already have a task,
several complete candidate traces, an evaluation rubric, and a verifier exposing
token logprobs. It assumes semantic judgment over observable traces can recover
some execution-success headroom without training a reward model. It is a
selector and progress estimator, not execution proof, policy enforcement, or a
replacement for the benchmark-native grader.

Its strengths are separable knobs and economical comparison scheduling. Its
limits are equally structural: manually designed criteria can leak development
knowledge; repeated model judgments reduce variance but not shared bias;
logprob-restricted APIs need a different backend; one model/harness/candidate
batch does not establish general same-model verification; and no selection
method can rescue an all-fail pool.

## FAK inward map

FAK self-query returned no direct capability entry for pairwise/logprob/PPT
selection. Raw code and backlog inspection found adjacent but non-equivalent
seams:

- `internal/modelroute@r84+ga46fc02ec9` ships `ScoreVotes`, which calls an
  absolute free-text judge once per candidate, then deterministic
  `Combine(ReduceBestOf)` selects the max score
  (`internal/modelroute/judge.go:100-155`,
  `internal/modelroute/modelroute.go:615-623`). This is **PARTIAL**, not pairwise
  fine-grained verification.
- `fak routebench` already replays fixed outputs and scalar scores over a
  committed corpus (`internal/modelroute/bench.go`). It is the cheapest extension
  seam for a candidate-pool selector replay; no new benchmark framework is
  needed.
- `internal/trajctl@r34+gaaa76e85ac` ships a structured rubric judge but keeps it
  at W1 below transcript-derived W2 and deterministic W3 evidence
  (`internal/trajctl/judgescorer.go:3-20,124-190`;
  `docs/observability/trajectory-control.md:50-73`). This is **PRESENT and
  deliberately stronger** than letting same-model progress directly gate done.
- `internal/terminalbench@r20+gcad27c07b7` and the submission packet already keep
  `result_claim_allowed=false` until benchmark-native evidence exists. This is
  **PRESENT** and should govern any imported replay.
- Open [#4235](https://github.com/anthony-chaudhary/fak/issues/4235) already owns
  live N-way KV-prefix branching, judge selection, and loser eviction. A second
  live best-of-N feature would duplicate it.
- Open [#897](https://github.com/anthony-chaudhary/fak/issues/897) owns the
  Terminal-Bench 2.1 rank objective. Closed #900 shipped rehearsal machinery but
  its body still records the paid raw/fak run as missing; this study does not
  silently promote that state into a result.

## Candidate-by-candidate disposition

| Technique | Axis | FAK status | Portfolio route | Disposition |
|---|---|---|---|---|
| Distrust narration; judge observable terminal evidence | false-positive resistance | PRESENT | DEFAULT | Keep FAK's W3/W2/W1 hierarchy; no issue. |
| Expected reward from score-token logprobs | score resolution | ABSENT-on-axis | WATCH / optional scorer | Evaluate only after the replay spine can compare it with absolute `ScoreVotes`; logprobs cannot be a universal default. |
| Odd-repeat A/B swap | within-pair order bias | ABSENT-on-axis | DEFAULT if pairwise scoring survives | Fold into that scorer's acceptance, not a separate product. |
| Directed Hamiltonian ring | aggregate slot bias | ABSENT-on-axis | WATCH | Admit only with a fixed-score bias test and cached/uncached equivalence. |
| Pivot tournament | comparison complexity | ABSENT-on-axis | OPTIONAL-MODULE | #4235 is the owning live seam; benchmark deduplicated edges against round-robin first. |
| Criterion decomposition and repeated judgments | semantic coverage / variance | PARTIAL | OPTIONAL | FAK rubrics exist; replay should expose criteria/repeat ablations before adding runtime calls. |
| Criterion-at-tail prefix shaping and warm-before-fanout | uncached input cost | PRESENT as substrate, unwired for this scorer | DEFAULT implementation rule | Reuse FAK's provider-prefix path if pairwise scoring ships. |
| Persist successes; retry failures; expose ties/errors | cache-poison resistance | PARTIAL | DEFAULT | FAK adaptation must fail visibly and provenance-bind the cache; never copy silent summary behavior. |
| Spend benchmark calls only on swing tasks | labeled-eval cost | ABSENT | RECIPE | Include in the offline replay only; never use hidden labels in live selection. |
| Same-model generation and verification | availability / cohort coverage | PARTIAL | RECIPE | Pin identities and compare with independent-judge and execution-based arms; never default from this one result. |
| Prefix-only progress scoring | causal online progress | PRESENT / DIVERGENT | WATCH | `trajctl` already scores live state and correctly prevents W1 from becoming proof; no issue. |
| Immutable result/usage/error ledger | reproducibility | PRESENT as FAK doctrine, missing in this lane | DEFAULT | The new replay artifact must use FAK's provenance and claim-gate conventions. |

One gap survives as immediate work: extend the existing offline `routebench`
surface with task-grouped candidate pools and selector metrics
([#8230](https://github.com/anthony-chaudhary/fak/issues/8230)). Pairwise logprob
scoring and PPT stay behind a dated review trigger: file/implement them only when
that spine can show a quality/cost gain over absolute scoring and full
round-robin on the same immutable inputs. This keeps one technique per future
ticket and avoids choosing architecture from an unreproduced headline.

## Licensing and provenance

Pinned merged code is MIT-licensed
([`LICENSE@8db8a11`](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/LICENSE)).
Direct porting is legally permissive with the copyright and permission notice,
but FAK should **ADAPT/INSPIRE** the mechanisms into stdlib-only Go and copy no
Python. Open-PR code stays inspire-only until merged or separately cleared.

The checked-in trajectories contain third-party task prompts, command output,
and possible source excerpts. The root MIT file does not prove rights to every
embedded byte. Do not vendor those 445 files into FAK without a separate data
rights/provenance review; accept an operator-supplied external bundle or
independently generated fixtures instead. This is a good-faith engineering
review, not legal advice.

## Source ledger

| Source class | Exact source | State/date | What changed the conclusion |
|---|---|---|---|
| README claim | `README.md:116-149@8db8a11` | shipped; observed 2026-08-20 | Scoped the exact same-model claim and separated it from the paper's different 86.5 result. |
| Implementation | `loaders.py`, `fine_grained_reward.py`, `pivot_tournament.py`, `__init__.py`, `scripts/run*.py` at the pin | shipped | Established the real mechanism, effective defaults, two source defects, cache identity, and missing uncertainty path. |
| Criteria | `criteria/terminal_bench.md:3-19@8db8a11` | shipped | Established output-over-narration and the three axes. |
| Dataset | all 445 Terminal-Bench 2.1 JSON metadata/reward rows | captured 2026-08-08/09 | Reproduced Pass@1/oracle and exposed incomplete generator identity. |
| Paper | arXiv:2607.05391v2 | 2026-07-07 | Supported the general mechanism/ablations, not the later same-model 2.1 claim; exposed the tournament edge-set mismatch. |
| History/blame | commits `628d8fc5`, `115de305`, `8db8a114` | 2026-08-14 through 20 | Separated code/data addition, README-only metric revision, and later maintenance. |
| Issues/PRs | upstream #8, #13, #14/#17; all read 2026-08-20 | dynamic, open | Confirmed active cache, Windows, and no-cache correctness gaps. |
| Releases/artifacts | Git tags, GitHub Releases, ignored cache/results | absent at observation | Prevented treating package version text as an attested release/result. |
| License/provenance | root MIT; data/asset inventory | shipped/unclear data rights | Allows clean-room adaptation of merged code but not blind corpus redistribution. |
| FAK self-query | `fak capabilities`; `fak-dev index docs|leaf|verbs|claims`; raw `rg`; issues #4235/#897/#595/#602/#613 | FAK HEAD `0aeef0e9` | Found existing runtime/benchmark seams and narrowed the new work to replay rather than a parallel selector stack. |

## Delegated-read witness partition

Three read-only agents covered implementation, evaluation, and history. Their
returns were treated as claims, then independently read back from the pinned Git
tree, the official POSIX loader, GitHub issue/PR APIs, the paper PDF, and FAK's
own code/indexes before synthesis.

- **CONFIRMED: 8/8 result groups** — repository identity/history; mechanism;
  full candidate metadata/reward census; Pass@1/oracle reproduction; missing
  result/uncertainty artifacts; no-cache ring loss; paper/code edge overlap;
  current issue/license state.
- **REFUTED: 0.**
- **UNWITNESSED: 0 folded.** A delegated synthetic percentage for how often
  double-weighting might change a winner was intentionally omitted because it
  was not needed for the source-level contradiction.
- **NO_CLAIM: 0.**

## Completeness critic

Read every implementation file under `llm_verifier/` and `scripts/`, the
Terminal-Bench criteria and benchmark registry, packaging/dependency files,
README, changelog, license, full Git history/blame, every issue and PR, and paper
sections for method, evaluation, ablations, progress, and limitations. Parsed
metadata/rewards across every one of the 445 relevant trajectories and manually
checked success/failure exemplars. Inventoried all 938 tracked files, tests/CI,
tags/releases, assets, other benchmark criteria, and data subtrees.

Not manually read: every step body in all 445 large trajectories, non-Terminal
fixture bodies, and plot pixels. They are input/illustration bodies rather than
distinct implementation subsystems; schema-wide parsing, provenance scanning,
and representative success/failure reads cover the claims used here. GitHub
Projects V2 was unavailable to the token, Discussions are disabled, private
community channels were not accessed, and TurboAgent is a separate repository.

No material technique remains unclassified. The one immediate gap is the
offline selector replay; algorithm-specific work is deliberately gated on its
measurement.
