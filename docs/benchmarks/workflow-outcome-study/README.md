# Guarded workflow outcome study

This packet closes the gap between **workflow uptake** and **workflow value**. It freezes the five task shapes from the first 2026-08-17 dogfood run, requires direct and workflow arms to execute equivalent prompts under the same model, effort, and token budget, and withholds any gain claim until a blind grader scores every completed artifact.

## Protocol

1. Copy `study.json` to an allocated scratch run directory. Do not edit its task prompts or rubrics after either arm starts.
2. Randomize arm order per task. Run both `direct` and `workflow` with identical `model`, `effort`, and `budget_tokens`. The workflow arm may delegate; the direct arm must execute the task itself rather than record a decline.
3. Capture the final artifact, elapsed milliseconds, provider-reported input/cached-input/output tokens, and any failure in rows shaped like `result-template.json`. Hash each final artifact and assign unrelated random `candidate_id` values before grading.
4. Give only candidate artifacts, task prompt, and frozen rubric to an independent grader. Do not disclose arm names, launch logs, token counts, or elapsed time. Record one row per candidate using `grade-template.json`.
5. Append results and grades to the study copy, then run:

```sh
fak sessions workflow-outcome-study --input RUN/study.json --json
```

`gain_claim_ready=true` means only that the evidence envelope is complete: every task has equivalent completed arms and two valid blind grades. It does not itself assert that either arm won. Compare correctness/usefulness before tokens or elapsed time, report provider usage as observed, and state uncertainty for five paired tasks.

## Boundaries

- A launch receipt is not a completed outcome.
- A final-message digest proves identity, not correctness.
- Failed or missing arms remain in the report and prevent a gain claim.
- The grader must be independent of the agents that produced either candidate.
- Keep raw model logs in private scratch; commit only privacy-reviewed aggregate evidence.
