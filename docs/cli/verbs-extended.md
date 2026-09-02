---
title: "fak CLI reference — Verb details — appended verb contracts"
description: "The later-appended per-verb contracts (launch, workpattern, stale-work, skill compile, study-*, vcache session-history, value-chain audit, git-daily, temp-artifacts, server, new-model), split out of docs/cli-reference.md."
---

# Verb details — appended verb contracts

Appended verb documentation, split out of [the CLI reference](../cli-reference.md) so the reference front door stays under its size budget.

## `fak launch`

`fak launch doctor [--json] [--repair]` diagnoses shim/provider posture; `--repair` refreshes the managed upgrade-stable fak target and owned shims.


`fak doctor launch-posture [--entrypoint agent|guard|serve] [--harness NAME] [--provider NAME] [--base-url URL] [--workspace DIR] [--json]` derives the default-on launch posture for the selected repository and wire. It reports eight mechanisms — bounded repository tools, Caveman, Ponytail, compact history, stale-read elision, cold-tool deferral, vCache anchoring, and dated provider-cache calibration — as `active`, `inert`, `disabled`, or `unsupported`, and names an action for every configured-but-inactive state. A configured vCache path is `inert` when its provider calibration is missing, stale, or fresh-but-observational; `active` now means a fresh measured constant is wired to steering. Defaults come from the same gateway/profile constants used by launch code. This is a preflight, not savings telemetry: `active` means the launch reaches the mechanism's runtime seam; use gateway/session metrics to prove realized savings.

`fak launch install [--provider claude|codex|all] [--default NAME] [--no-path]`
installs managed shims and, unless `--no-path` is set, an idempotent fak-owned PATH block
for supported PowerShell/POSIX startup files. Uninstall removes only that block.

The standalone `fak-selfupdate` executable is a recovery-sized entry point over the same package implementation as `fak self-update`. It accepts the same flags and emits the same `fak.self-update.receipt/v1` JSON. In particular, `fak-selfupdate --check --target PATH` inspects the target's embedded Go build metadata without executing that potentially stale or partially replaced binary.

The managed `fak-launch` target remains runnable while `fak self-update` replaces the
deployed binary. During that bounded transaction it defaults to `prior`, immediately
running the last known-good executable. Pass `--update-launch-policy=wait` to wait (at
most 10 seconds by default) and then run the new executable, or pass
`--update-launch-policy=fail` for a strict, actionable failure.
`--update-launch-wait=30s` changes the bounded wait (capped at five minutes).
The equivalent launch-config keys are
`"update_launch_policy": "prior|wait|fail"` and `"update_launch_wait_ms": N`.
A managed launcher accepts those flags only when they precede the provider command.
Flags after the provider or `--` remain provider arguments. These paths are
non-interactive and preserve argv boundaries, stdin, stdout, stderr, and exit status.

`fak launch add NAME --command PATH [--arg ARG ...] [--default] [--shim]` persists a
custom provider as an argv template. `fak launch remove NAME` removes the binding and
owned shim; `fak launch list [--json]` lists bindings without exposing local command paths
or argument values. Names are lowercase command aliases, cannot be path-like, and cannot
shadow reserved fak verbs. See [Zero-adoption provider launch](../zero-adoption-launch.md).
## `fak workpattern` — named coding-workload catalog and miners

`fak workpattern` is the offline front door for the versioned coding-workload vocabulary. It separates goal-shaped **patterns** from reusable ordered **subpatterns**, and reports only evidence supported by explicit detectors.

```bash
fak workpattern list --json
fak workpattern source --source . --json
fak workpattern trajectory --trajectory turns.jsonl --json
fak workpattern trajectory --chat scrubbed-chat.json --json
fak workpattern report --source . --trajectory turns.jsonl --json
```

The JSON schema is `fak.workpattern-report/1`. It records catalog and detector versions, input digests, findings, and abstentions. Source findings include paths and source ranges. Trajectory findings include trace/turn ranges and detector reasons. Default trajectory output contains tool names, ranges, hashes, counts, and reasons—not prompt, message, or tool-argument bodies. `--include-excerpts` is an explicit opt-in and excerpts remain redacted/truncated by the trajectory miner.

Supported chat input is the deliberately content-free `fak.scrubbed-chat/1` format documented by `internal/trajectory.ImportScrubbedChat`; unsupported or malformed formats fail closed. Findings are evidence-backed candidates, not semantic intent judgments, and no match means abstention rather than proof that a pattern is absent.

Research basis and vocabulary proposal: [`research/coding-workload-vocabulary.md`](../research/coding-workload-vocabulary.md). Machine companion: [`research/coding-workload-vocabulary.json`](../research/coding-workload-vocabulary.json).


## `fak workpattern` — named coding-workload catalog and miners

`fak workpattern` is the offline front door for the versioned coding-workload vocabulary. It separates goal-shaped **patterns** from reusable ordered **subpatterns**, and reports only evidence supported by explicit detectors.

```bash
fak workpattern list --json
fak workpattern source --source . --json
fak workpattern trajectory --trajectory turns.jsonl --json
fak workpattern trajectory --chat scrubbed-chat.json --json
fak workpattern report --source . --trajectory turns.jsonl --json
```

The JSON schema is `fak.workpattern-report/1`; it includes catalog/detector versions, input digests, findings, and abstentions. Source findings carry source ranges. Trajectory findings carry trace/turn ranges and reasons. Default output excludes prompt, message, and tool-argument bodies. `--include-excerpts` is explicit opt-in and remains redacted/truncated by the miner. The content-free chat adapter accepts only `fak.scrubbed-chat/1` and fails closed on malformed/unsupported formats.

Research basis: [`research/coding-workload-vocabulary.md`](../research/coding-workload-vocabulary.md); machine companion: [`research/coding-workload-vocabulary.json`](../research/coding-workload-vocabulary.json). Findings are evidence-backed candidates, not autonomous intent judgments.
# `fak stale-work`

Ranks tracked documentation into bounded, issue-ready, provenance-bearing review packets without
mutating candidates. See [`docs/stale-work.md`](../stale-work.md). `--selfcheck` proves dependency
drift outranks age-only history.

`fak stale-work loop` consumes that packet and renders one contract-valid dedicated issue unit per
candidate. It is dry-run by default, deduplicates against an `--issues` snapshot, serializes
overlapping paths into waves, and refuses dispatch until an existing issue passes the shared issue
contract. `--state`/`--state-out` read and explicitly persist evidence-digest adjudications;
`--witnesses` reconciles only independent issue/git/test evidence. GitHub creation and worker
launch are separately armed by `--live-issues` and `--live-launch`.

## `fak skill compile` — explicit skill programs

```text
fak skill compile [--json] [--dialect <name>] [--expose <canonical-name>]... <SKILL.md>
```

Compiles exactly one fenced `fak-program` JSON block from a skill file into a
content-addressed host registration and an independently content-addressed
model-visible snapshot. Natural-language skill prose is never inferred as
executable control flow.

Registration is hidden by default. `--expose` selects canonical names for the
current snapshot; `--dialect` applies declared provider/harness aliases after
selection. The model learns current availability only from the `tools` carried
in its provider request—not from installation, a skill being present, a builtin
name, or a model's training prior.

JSON output fields are stable at version `fak.skill-compile/v1`:

- `registration`: canonical program, source identity, and registration digest;
  host-only executor argv/adapter data lives here.
- `model_view.digest`: identity of the exact selected surface.
- `model_view.dialect`: requested alias dialect.
- `model_view.tools`: selected provider-visible names, descriptions, input
  schemas, canonical names, and registration digests; executor data is absent.
- `model_view.omitted`: installed registrations omitted from this snapshot with
  a reason such as `NOT_SELECTED`.

Exit status is `0` on success and `2` for usage, read, compile, unknown
selection, invalid dialect alias, collision, or JSON-encoding failure. Without
`--json`, the command prints a concise registration/exposure summary.

A deterministic command adapter is optional but must be explicit:
`fak.command-adapter/v1` declares every JSON field mapped to an argv entry,
stdin, or environment value and declares `result: "json"`. Execution uses an
argv vector, never shell-string interpolation, and refuses undeclared adapters,
missing fields, nonzero exits, and non-JSON output.

Runnable hidden/exposed example and self-check:
[`examples/skill-program/`](../../examples/skill-program).

## `fak study-classify`

Classify every record in a validated `study-forge` corpus and validate the result offline:

```bash
fak study-classify classify --corpus /tmp/vllm.corpus.json --out /tmp/vllm.classification.json --index-out docs/research/vllm.classification-index.json
fak study-classify classify --corpus /tmp/vllm.corpus.json --out /tmp/vllm.classification.json --index-out /tmp/vllm.classification-index.json --related-limit 4 --json
fak study-classify validate --classification /tmp/vllm.classification.json --corpus /tmp/vllm.corpus.json
fak study-classify validate-index --index docs/research/vllm.classification-index.json --classification /tmp/vllm.classification.json --corpus /tmp/vllm.corpus.json
fak study-classify schema > /tmp/fak-studyclass-output-1.schema.json
```

`classify` first performs the complete `study-forge` validation, binds the result to the corpus byte SHA-256 and receipt identity, and assigns exactly one primary disposition to every record. The closed disposition vocabulary distinguishes merged/landed work, open proposals, regressions/bugs, duplicates, support/questions, stale/superseded work, closed-unmerged work, and release/metadata/non-candidates. Zero or more mechanism matches use the versioned issue taxonomy for architecture/runtime, scheduling/batching, KV/cache, kernels/compilation, speculative decoding, distributed/parallelism, memory/residency, model/backend/hardware, APIs/tool calling/structured output, observability/operations, reliability/security, tests/CI/docs, and explicit non-candidates.

The command writes both outputs atomically and deterministically. `--out` contains every per-record classification and can be large, so it belongs in allocated scratch or another declared artifact location. `--index-out` contains the bounded cluster index suitable for review and commit; `--related-limit` controls how many related identity samples each compact cluster retains. Human output reports counts by source, disposition, mechanism, state, and confidence; `--json` emits the same summary as JSON.

Clusters retain upstream identities, state, dates, confidence, and the exact field/rule signal used for membership. They do not reconstruct GitHub relationships that the corpus did not capture: related members mean they share deterministic rule evidence, not that one issue links to, duplicates, implements, or supersedes another. `validate` strict-decodes the full output, revalidates the corpus, and rejects schema drift, unknown fields, checksum or input-binding mismatches, duplicate or missing identities, invalid dispositions/mechanisms, and actionable clusters without evidence. `validate-index` additionally joins the commit-sized index to that validated full output, making every omitted-membership, summary, and full-output checksum independently recomputable. `schema` emits the embedded Draft 2020-12 JSON Schema for the full output contract.

## `fak study-link`

Build or validate the bounded evidence ledger that joins a compact study cluster index
to witnessed FAK issues and repository artifacts:

```bash
fak study-link build --index docs/research/vllm-classification-2026-08-26/index.json --forge /path/to/fak-forge.json --adjacency docs/research/inventory/vllm-related-system-adjacency-v1.json --repo . --out docs/research/vllm-fak-join-2026-08-27/ledger.json --summary docs/research/vllm-fak-join-2026-08-27/README.md
fak study-link validate --ledger docs/research/vllm-fak-join-2026-08-27/ledger.json --index docs/research/vllm-classification-2026-08-26/index.json --forge /path/to/fak-forge.json --adjacency docs/research/inventory/vllm-related-system-adjacency-v1.json --repo .
```

`build` reads the complete captured study-forge corpus, compact cluster index,
adjacency manifest, and repository root. It deterministically emits a bounded
machine-readable ledger through `--out` and a Markdown review summary through
`--summary`. Use the complete captured corpus; `gh issue list --limit 1000` is not a
valid substitute.

Joins are conservative: strong matches require reproducible exact evidence, while
ambiguous candidates remain explicitly marked for manual review rather than being
promoted into fabricated semantic links. `validate` rechecks complete cluster coverage,
captured issue existence and state, duplicate exact joins, repository paths, and source
checksums against the same four inputs. The checked-in vLLM/FAK result lives under
`docs/research/vllm-fak-join-2026-08-27/`.

## `fak study-priority`

Build or validate the bounded queue derived from the uncovered actionable rows in a
`study-link` ledger:

```bash
fak study-priority build --source-ledger docs/research/vllm-fak-join-2026-08-27/ledger.json --ledger docs/research/vllm-priority-2026-08-27/ledger.json --summary docs/research/vllm-priority-2026-08-27/README.md
fak study-priority validate --source-ledger docs/research/vllm-fak-join-2026-08-27/ledger.json --ledger docs/research/vllm-priority-2026-08-27/ledger.json --summary docs/research/vllm-priority-2026-08-27/README.md
```

`build` applies the versioned rubric and separate hard gates, retains the stable
source-cluster mapping for every candidate, and emits a deterministic dependency-respecting
queue plus its Markdown review summary. `validate` recomputes the build from the same source
ledger and rejects missing or duplicate source inputs, cycles, missing dependencies, gate
violations, output drift, and checksum drift. Native-inference candidates remain fak-native;
llama.cpp evidence is reference/borrowing evidence only and cannot authorize a fallback.

## `fak study-tickets`

Construct or validate the final ticket-closure ledger from the priority queue, complete FAK
forge corpus, adjacency ledger, classification index, and FAK evidence join:

```bash
fak study-tickets build --priority docs/research/vllm-priority-2026-08-27/ledger.json --join docs/research/vllm-fak-join-2026-08-27/ledger.json --forge /path/to/fak-forge.json --adjacency docs/research/inventory/vllm-related-system-adjacency-v1.json --classification docs/research/vllm-classification-2026-08-26/index.json --ledger docs/research/vllm-ticket-closure-2026-08-27/ledger.json --report docs/research/vllm-ticket-closure-2026-08-27/README.md
fak study-tickets validate --priority docs/research/vllm-priority-2026-08-27/ledger.json --join docs/research/vllm-fak-join-2026-08-27/ledger.json --forge /path/to/fak-forge.json --adjacency docs/research/inventory/vllm-related-system-adjacency-v1.json --classification docs/research/vllm-classification-2026-08-26/index.json --ledger docs/research/vllm-ticket-closure-2026-08-27/ledger.json --report docs/research/vllm-ticket-closure-2026-08-27/README.md
```

`build` requires exact candidate-to-issue mappings, verifies that mapped issues remain open and
contain their required source-cluster and fak-native Qwen3.8 contracts, preserves complete,
partial, and inaccessible adjacency evidence separately, and emits deterministic JSON and
Markdown. `validate` rebuilds from the supplied corpora and rejects source checksum drift,
duplicate mappings, queue/dependency drift, closed or malformed tickets, and any actionable,
unclassified, selected-unmapped, or closure leftover.

## `fak study-inventory`

Render a deterministic local-checkout map for an exhaustive `study-repo` pass:

```bash
fak study-inventory --root /tmp/study-repo --repository owner/name --revision <sha>
fak study-inventory --root /tmp/study-repo --repository owner/name --revision <sha> --json
fak study-inventory --root /tmp/study-repo --repository owner/name --revision <sha> --json --out docs/research/inventory/owner-name.json
```

The command walks the checked-out tree, groups immediate subsystems, counts runtime/test/doc files, records representative paths, and emits one status row for every source class required by the exhaustive study contract. Use JSON for the registry `map_path`; Markdown is a human rendering. Non-tree classes such as open/closed issue history, the fak self-query witness, candidate matrix, and issue tracking are called out as follow-up requirements instead of being silently treated as covered.

## `fak study-monitor`

Render and validate the durable external-repository queue used by the `study-repo` and `scout-loop` skills:

```bash
fak study-monitor
fak study-monitor --due-days 7 --json
fak study-monitor --registry docs/research/monitored-repositories.json --as-of 2026-08-14
fak study-monitor --inventory-check --json
```

The command reads `docs/research/monitored-repositories.json` by default, sorts by priority, and reports each source's status, pinned checked revision, `last_checked` age, and whether it is due for refresh. `--as-of` exists for deterministic witnesses and tests. The command does not contact GitHub or mutate the registry; scouts update all check fields together after inspecting the source.

`--inventory-check` switches the readout to the stricter exhaustive-inventory contract. Candidate and studied rows are treated as needing a machine-readable map by default; the check exits nonzero until each row has an `inventory` block with a map path, matching indexed revision, positive subsystem count, completeness-critic result, and the required source-class coverage set. The map itself must include positive totals that equal its subsystem aggregates and one status row for every required source class. Local tree classes can be satisfied by `covered` rows with path evidence or by `checked_absent` rows from the complete tree walk. Forge history remains `partial` or `external_required` in that local map. Set `inventory.forge_receipt_path` to a `study-forge capture` corpus (or its standalone receipt) to satisfy the compound forge class with validated external evidence: the monitor binds schema, repository, checked revision, cutoff, complete status, all six complete source receipts, reconciled uniqueness counts, and checksums. An invalid or partial declared receipt blocks the row rather than falling back silently. Legacy traceable `source_evidence` can still name issue, pull-request, and discussion evidence, but is not replayed. Fak self-query witnesses, candidate matrices, and issue tracking remain `external_required` and need traceable `source_evidence` entries instead of bare class names.

## `fak vcache session-history`

Explore the historical session index without opening raw transcripts. The live usage contract is:

```text
fak vcache session-history --index FILE [--provider NAME] [--min-errors N]
fak vcache session-history refresh [--index FILE] [--once|--interval DURATION]
fak vcache session-history benchmark [--sizes 1000,10000,100000] [--repetitions 3]
```

Use `--json` on the query form for machine-readable rows. `--min-errors` must be non-negative and `--limit` must be at least one.

## `fak value-chain audit`

Attribute stack-stage changes to measured outcomes while keeping missing cost absent:

```text
fak value-chain audit --manifest M --observations O [--json]
```

The fixture-backed offline witness is:

```bash
fak value-chain audit --manifest examples/value-chain/support-manifest.json --observations examples/value-chain/support-observations.json --selfcheck --expect examples/value-chain/support-witness.txt
```

`--selfcheck` compares the rendered report with `--expect`; it requires both flags and exits non-zero on any mismatch.

## `fak git-daily`

Run or inspect the lock-aware daily Git maintenance job:

```text
fak git-daily [--root DIR] [--dry-run] [--force] [--prune-worktrees] [--emit-unit launchd|systemd|taskscheduler] [--interval DURATION] [--fak-bin PATH] [--label NAME] [--status N] [--score] [--json]
```

`--dry-run` is the safe preview. `--status N` and `--score` are read-only ledger views; they do not run a maintenance tick. `--emit-unit` prints a scheduler definition; install it with the operating system's own scheduler tooling after review. Use `--root` in scheduled jobs so repository discovery never depends on the scheduler's working directory.

## `fak temp-artifacts`

Inventory direct fak build/archive artifacts in the resolved OS temporary directory, with preview as the default:

```text
fak temp-artifacts --min-age DURATION [--apply] [--json]
```

`--min-age` is required and must be positive. The command examines only direct, ordinary, non-reparse `fak-*` files with a case-insensitive `.exe`, `.tar`, or `.zip` extension. Preview reports each matching file's exact canonical path, age, bytes, eligibility, and typed reason plus aggregate matching, eligible, preserved, and reaped totals. It never recurses into temporary directories.

On Windows, selection checks each exact candidate path against both `Win32_Process.ExecutablePath` and parsed command-line arguments. Prefix collisions do not count as references, and command-line contents never enter the receipt. If inspection is unavailable, candidates are preserved. `--apply` rechecks identity and references before each move, moves an eligible file into a unique quarantine under the same temporary root, rechecks source and quarantine paths, and deletes only that exact quarantined regular file. A changed, newly referenced, inaccessible, ambiguous, or failed file remains at its source or reported quarantine path; the command never terminates a process and never uses recursive or wildcard deletion.

Producer audit for this fallback: committed `os.MkdirTemp("", "fak-…")` build/cleanroom producers such as `cmd/fak/commit_buildcheck.go`, `cmd/fak/prepush_build.go`, `internal/committedtree`, `internal/devcmd/buildcheck`, `internal/nightrun/prebuild`, and `internal/workerworktree` own directories and use local cleanup where deterministic. Committed direct `os.CreateTemp` producers use non-allowlisted control extensions such as `.md`, `.json`, `.txt`, `.patch`, and `.index`. The incident's direct `.exe`, `.tar`, and `.zip` names have no committed deterministic producer to repair, so this bounded fallback owns interrupted and manual verification artifacts without widening those producer contracts.

## `fak server`: own a loopback inference server

`fak server` manages one local-process server instance through a receipt-backed lifecycle:

```text
fak server init --dir DIR --name NAME --model MODEL.gguf --sha256 HEX --executable /path/to/llama-server --json
fak server up --dir DIR --json
fak server status --dir DIR --json
fak server down --dir DIR --json
```

`init` records immutable model and executable identity. `up` starts only the declared executable and waits within its readiness deadline; `status` reports the typed lifecycle state; `down` signals only the process proven by the instance receipt. Each subcommand emits JSON. Run `fak server` with no subcommand for the live usage text.


--gate FILE strictly decodes an envelope-scoped gate request, compares the candidate to the last accepted witnessed receipt, and emits pass/investigate/regression plus suspect module revisions and a guarded bisect packet. Regression exits 3. Policy, cadence, override, and rollback evidence are defined in [the regression-gate contract](../benchmarks/NATIVE-PERFORMANCE-REGRESSION-GATE.md).

### Token destination distribution

`fak trajectory audit` records two deliberately separate views: provider-exact request-level input/output/cache token buckets, and a deterministic transcript-payload distribution measured in UTF-8 bytes. The latter shows user messages, assistant messages, reasoning, tool calls, tool results, other records, and a per-tool ranking in JSONL and Markdown. It is an attribution signal—not per-block billed tokens, which providers do not expose. `trajectory.CompactAuditDistributionLine` supplies the stable width-bounded line used by terminal/TUI status surfaces.

`trajectory audit` separates deterministic model-visible content bytes from serialized transcript storage/telemetry overhead. Runtime event mirrors such as Codex `item_completed` and Claude attachments are typed by subtype in a separate table and never inflate the model-visible denominator. `visible_unknown` is explicit and coverage-budgetable.

### Replayable private audit corpus

Use `--snapshot-out` when a later audit must replay the exact selected Claude and
Codex inputs rather than rediscover a moving live window:

```bash
fak trajectory audit --since 7d --user-contains qwen \
  --snapshot-out /private/path/qwen-corpus \
  --jsonl qwen-audit.jsonl --md qwen-audit.md
fak trajectory audit --snapshot /private/path/qwen-corpus \
  --jsonl replay.jsonl --md replay.md
```

Capture applies the live root, time, and topic selectors first, copies only selected
JSONL files, audits the captured copy, then atomically publishes a new 0700 directory
with 0600 inputs and `manifest.json` schema `fak-trajectory-audit-corpus/1`. The
manifest contains safe root labels, relative paths, byte lengths, SHA-256 values,
selection presence, audit schema, corpus digest, and captured-output digest—never
payload bytes, absolute live roots, or the topic literal. Existing targets are refused.

Replay accepts output flags but rejects live roots, `--since`, `--user-contains`, and
`--baseline`. It verifies schema, containment, exact file set, permissions, lengths,
and hashes before parsing and repeats verification afterward. Any missing, changed,
extra, malformed, incompatible, path-escaping, or concurrently mutated input exits
nonzero with `TRAJECTORY_SNAPSHOT_REFUSED`. Snapshots contain raw transcript bytes:
keep them outside Git and public witness paths, never sync them as audit output, and
delete the explicit directory when retention ends.

Pass `--snapshot-usage-ledger FILE` on capture or replay to append a deliberately
content-free adoption row to an explicit operator-owned JSONL target. The command
declares `OUT_OF_TREE_WRITE` before the append; without this option it creates no
usage ledger. Rows contain only schema, UTC observation time, `capture`/`replay`,
`success`/`refused`/`error`, and a closed uppercase reason code. They contain no
snapshot or root paths, hostnames, transcript identifiers or content, or
correlatable hashes. Appends are concurrent-safe and restrict the ledger to 0600.

```bash
fak trajectory audit --snapshot /private/path/qwen-corpus \
  --snapshot-usage-ledger /private/ops/snapshot-usage.jsonl
fak trajectory audit \
  --snapshot-usage-fold /private/ops/snapshot-usage.jsonl
```

The fold is read-only and emits deterministic counts by ascending ISO week,
operation, and outcome.

## `fak new-model`: refusal-safe native model intake

Compile a pinned model-release manifest into a deterministic fak-native onboarding packet:

```bash
fak new-model --from-manifest internal/newmodel/testdata/qwen38-valid.json --json
```

Manifest intake requires `--json` and cannot be combined with scaffold flags. The manifest pins the model identity and semantic deltas; successful output names the `fak-native` engine and never selects an external runtime fallback. Unknown or contradictory semantic deltas are refused before allocation with structured JSON on stderr and exit code 3.

The existing scaffold mode remains separate:

```bash
fak new-model --family myfamily --topology prenorm --dry-run --json
```

