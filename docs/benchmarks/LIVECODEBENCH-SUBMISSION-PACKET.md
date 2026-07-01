# LiveCodeBench code-generation submission packet - assembly index

Status: `BLOCKED_PRECREDENTIAL` - no result claim, no authority row yet.
Date: 2026-07-01.
Issue: https://github.com/anthony-chaudhary/fak/issues/2115
Parent epic: https://github.com/anthony-chaudhary/fak/issues/2085

This is the assembly index for a future LiveCodeBench leaderboard submission for
the **code generation** scenario. It pins the repo-side packet artifacts, records
the evidence still missing before any score may be promoted, and provides the
fill-after-evidence `BENCHMARK-AUTHORITY.md` row template.

It makes **no** LiveCodeBench result claim. `result_claim_allowed=false` until
the official LiveCodeBench grader has scored the exact saved generations over a
fixed release and date window. The row template below is a form to fill after
evidence exists, not a benchmark row.

## The Evidence Gate

LiveCodeBench accepts leaderboard submissions for the code-generation scenario
through the upstream submissions flow. A fak row cannot be submitted or copied
into [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md) until all of these
conditions are true:

- the benchmark release is fixed with `--release_version` instead of
  `release_latest`;
- the contest-date window is explicit (`--start_date` / `--end_date`);
- raw and fak arms generated the same problem set with the same model identity,
  budget, sampling settings, and retry policy;
- the fak arm's saved generations were exported in the official custom-evaluator
  input shape;
- the official LiveCodeBench grader produced `pass@1` and `pass@5` from those
  saved generations;
- the official grading artifact, raw generation artifact, fak generation
  artifact, and run contract are checked in and hash-pinned in this packet;
- the honesty gate (#2113) reports `result_claim_allowed=true`;
- the evidence class and promotion requirements (#2114) classify the run as
  official-gradeable evidence.

As of this packet, the campaign is **BLOCKED_PRECREDENTIAL**: there is no official
graded code-generation run, no upstream submission PR, and no authority row.

## Checked-In Artifacts

Every artifact below is tracked in this repo. Re-derive hashes with
`sha256sum <path>` and compare before promoting a result.

| Artifact | Role in the packet | SHA-256 |
|---|---|---|
| `docs/benchmarks/LIVECODEBENCH-RUNBOOK.md` | End-to-end command path and official-grading evidence contract. | `84c76b51e05839806ccfefbe7f0a95cc13b36b84de32f021356eb2b53c7d724f` |
| `docs/benchmarks/LIVECODEBENCH-RESULTS.md` | Results scaffold that stays scoreless until official grading fills it. | `b594465adb078ce3dce3c83e2d635631c50460ff2b9381ee53679f67a882395f` |
| `internal/livecodebench/testdata/fixture.json` | No-network committed fixture for the CI smoke; adapter evidence only, never a leaderboard result. | `b4afd9593e9567fe957ad161c862d95079baf126590d929b62b7acd0a040bbff` |

Verify the pinned packet artifacts:

```bash
sha256sum \
  docs/benchmarks/LIVECODEBENCH-RUNBOOK.md \
  docs/benchmarks/LIVECODEBENCH-RESULTS.md \
  internal/livecodebench/testdata/fixture.json
```

## Missing Evidence

None of the following exists yet; each must be checked in and hash-pinned before
this packet can leave `BLOCKED_PRECREDENTIAL`:

- official LiveCodeBench repository revision or release tag used for grading;
- run contract naming `release_version`, scenario `codegeneration`, date window,
  model identity, training-cutoff statement or residual, sampling settings
  (`n`, temperature), timeout, evaluator process count, and retry policy;
- raw-arm saved generations for the fixed code-generation problem set;
- fak-arm saved generations over the same problem set;
- fak custom-evaluator export JSON, preserving the upstream
  `question_id` / `code_list` shape;
- official custom-evaluator output for the fak arm;
- official raw-arm evaluation output, if this packet reports a raw-vs-fak
  comparison;
- date-windowed score output from `lcb_runner.evaluation.compute_scores`;
- upstream LiveCodeBench submissions PR URL or explicit operator decision not to
  submit yet;
- completed `result_claim_allowed=true` witness from #2113;
- completed evidence-class / promotion-requirements witness from #2114.

## Authority Row Template - Fill Only After Evidence

When, and only when, the missing evidence above exists, add one row to
[`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md). The row must keep the
official LiveCodeBench score separate from fak-specific routing or safety
evidence:

| Field | What it carries | Provenance |
|---|---|---|
| LiveCodeBench release | `release_vN` or exact upstream revision | **OFFICIAL** - upstream dataset/repo identity |
| Scenario | `codegeneration` | **OFFICIAL** - leaderboard-accepted scenario |
| Contest date window | `YYYY-MM-DD..YYYY-MM-DD` | **OFFICIAL** - date-windowed scoring boundary |
| Model / backend | model name, raw serving backend, fak serving backend | run contract |
| Official raw pass@1 / pass@5 | raw arm score, if run | **OFFICIAL** - LiveCodeBench grader output |
| Official fak pass@1 / pass@5 | fak arm score | **OFFICIAL** - LiveCodeBench grader output over fak saved generations |
| fak-specific evidence | adjudication/routing verdicts, denied unsafe actions, gateway trace, cost/tokens | **fak-SPECIFIC** - not a leaderboard number |
| Artifact paths + SHA-256 | contract, raw generations, fak generations, custom export, official eval output, score output | this packet |
| Reproduce command | exact upstream runner/custom-evaluator/compute-scores commands | run contract |
| Limitations | credential scope, release/date window, timeout sensitivity, evaluator variance, submission PR state | plain |

Until that row exists, no LiveCodeBench pass rate may appear in `README.md`, the
hero comparison, release notes, or any external claim.

## Honesty Boundary

This packet is a precredential gate. It records the submission shape, hashes the
currently committed scaffolding, and names the missing evidence. It does not run
LiveCodeBench, does not submit to the upstream leaderboard, and does not promote a
score. `status=BLOCKED_PRECREDENTIAL` and `result_claim_allowed=false` remain in
force until #2113 and #2114 clear on official grading artifacts.

## Where This Sits

- Epic: [#2085](https://github.com/anthony-chaudhary/fak/issues/2085)
- Result gate: [#2113](https://github.com/anthony-chaudhary/fak/issues/2113)
- Evidence class and promotion requirements: [#2114](https://github.com/anthony-chaudhary/fak/issues/2114)
- Authority row and submission gate: [#2115](https://github.com/anthony-chaudhary/fak/issues/2115)
- Results scaffold: [LIVECODEBENCH-RESULTS.md](LIVECODEBENCH-RESULTS.md)
- Runbook: [LIVECODEBENCH-RUNBOOK.md](LIVECODEBENCH-RUNBOOK.md)
- Upstream harness: [LiveCodeBench](https://github.com/livecodebench/livecodebench)
- Upstream submissions: [LiveCodeBench/submissions](https://github.com/LiveCodeBench/submissions)
