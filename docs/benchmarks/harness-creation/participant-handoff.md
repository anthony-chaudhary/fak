# Independent harness-creation participant handoff

Use this card for #6911 only after the study baseline reports `baseline_ready=true`.

## Before the clock

1. The operator records a random `person-…` slug; never put a name, email, employer, or
   account identifier in the repository artifact.
2. The participant confirms they have not worked on fak internals and records prior
   experience in broad categories only (`none`, `agent-framework-user`, etc.).
3. Explain that failures stay in the denominator, assistance is limited to this card and
   command `--help`, the transcript will be privacy-reviewed, and the participant may
   withdraw before publication by asking the operator to delete the scratch run.
4. Verify required toolchains and network access without installing or rehearsing the
   task. Allocate a clean cache and random run slug.
5. Copy `participant-receipt-template.json` into the external run directory. Choose the
   ten-minute or weekend task in [README.md](README.md); do not change tracks after start.
6. For a ten-minute comparison, freeze one random `pair_id` and its assigned
   `pair_order` (`fak-first` or `baseline-first`) before either clock. Use each assignment
   once across the first two complete pairs. Record `arm_position` 1 or 2, use separate
   run IDs and receipts, and never drop the second arm after seeing the first result.
7. Give each arm a fresh external directory and the protocol's declared empty cache. Do
   not reuse generated files, package/module caches, task answers, or process state across
   arms; record any unavoidable carryover as friction. Weekend receipts use `fak` only.
6. For a ten-minute comparison, freeze one random `pair_id` and its assigned
   `pair_order` (`fak-first` or `baseline-first`) before either clock. Use each assignment
   once across the first two complete pairs. Record `arm_position` 1 or 2, use separate
   run IDs and receipts, and never drop the second arm after seeing the first result.
7. Give each arm a fresh external directory and the protocol's declared empty cache. Do
   not reuse generated files, package/module caches, task answers, or process state across
   arms; record any unavoidable carryover as friction. Weekend receipts use `fak` only.

## During and after

Start the monotonic and wall clocks when the participant opens the selected task card.
Record every command and exit, including failures and every human hint/help request. Stop
only at the task card's success/failure boundary. Preserve the product, hashes, build and
selfcheck output, privacy-safe transcript, and conformance receipt for a weekend run.

Validate before handoff:

```text
fak harness study receipt --input participant-receipt.json \
  --study docs/benchmarks/harness-creation/study.json
```

The command fails closed on missing evidence and emits a `study_row` ready to append to
`study.json`. It does not decide promotional truth; rerun:

```text
fak harness study creation --input docs/benchmarks/harness-creation/study.json
```

The operator deduplicates run and participant slugs before append, archives artifacts,
files each material friction item, and gives the participant a final chance to withdraw.
Deletion means removing the entire external run directory and not appending its row; once
a privacy-reviewed aggregate is released, follow the repository correction process rather
than pretending an immutable published artifact vanished.
