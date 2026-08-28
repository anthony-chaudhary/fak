# Qwen trajectory snapshot usage outcomes — 2026-08-28

This is the public, content-free closure witness for #9699. It was captured from
synthetic repository fixtures at `origin/main@d56d9780f8d5`; no real transcript,
private root, hostname, transcript identifier, or transcript content is recorded
here.

## Shipped surface

Commit `e178eac9729c` added the explicit `--snapshot-usage-ledger` capture/replay
ledger and the read-only `--snapshot-usage-fold` CLI surface. At the witnessed
base, the affected modules are:

- `internal/trajectory r58+ge178eac97`
- `cmd/fak r3502+ge178eac97`
- `docs/cli-reference.md r207+ge178eac97`

The ledger is absent unless explicitly requested. A requested append declares
`OUT_OF_TREE_WRITE` before mutation, restricts the file to 0600, and records only
the closed schema, UTC observation time, capture/replay operation,
success/refused/error outcome, and uppercase reason code.

## Captured readout

One real synthetic-fixture capture followed by one verified replay produced:

```json
{"schema":"fak-trajectory-audit-snapshot-usage-fold/1","weeks":[{"week":"2026-W35","total":2,"operations":{"capture":1,"replay":1},"outcomes":{"success":2}}]}
```

The observed operating envelope was:

- snapshot capture/replay success: 2/2 (100 percent)
- replay output drift: 0 bytes
- manifest transcript-content matches: 0
- ledger private fixture/path matches: 0
- ledger disallowed fields: 0
- ledger correlatable 64-hex hashes: 0
- ledger permissions: 0600
- usage write declarations: one before capture append and one before replay append

## Reproduction

These are the exact public-safe commands. All raw snapshot and audit output stays
under a fresh temporary 0700 directory and is not a repository artifact.

```bash
WITNESS_ROOT="$(mktemp -d)"
mkdir -m 700 "$WITNESS_ROOT/repo" "$WITNESS_ROOT/private"
git archive d56d9780f8d5374fd237828f886643be47a24e76 | tar -x -C "$WITNESS_ROOT/repo"
cd "$WITNESS_ROOT/repo"
go build -o "$WITNESS_ROOT/private/fak" ./cmd/fak

"$WITNESS_ROOT/private/fak" trajectory audit --since 0 \
  --claude-root internal/trajectory/testdata/audit/claude/projects \
  --codex-root internal/trajectory/testdata/audit/codex/sessions \
  --snapshot-out "$WITNESS_ROOT/private/snapshot" \
  --snapshot-usage-ledger "$WITNESS_ROOT/private/usage.jsonl" \
  >"$WITNESS_ROOT/private/capture.md" \
  2>"$WITNESS_ROOT/private/capture.stderr"

"$WITNESS_ROOT/private/fak" trajectory audit \
  --snapshot "$WITNESS_ROOT/private/snapshot" \
  --snapshot-usage-ledger "$WITNESS_ROOT/private/usage.jsonl" \
  >"$WITNESS_ROOT/private/replay.md" \
  2>"$WITNESS_ROOT/private/replay.stderr"

"$WITNESS_ROOT/private/fak" trajectory audit \
  --snapshot-usage-fold "$WITNESS_ROOT/private/usage.jsonl"

cmp -s "$WITNESS_ROOT/private/capture.md" "$WITNESS_ROOT/private/replay.md"
stat -f '%Lp' "$WITNESS_ROOT/private/usage.jsonl"
jq -s '[.[] | keys_unsorted - ["schema","observed_at","operation","outcome","reason"]] | flatten | length' \
  "$WITNESS_ROOT/private/usage.jsonl"
grep -Eoc '[[:xdigit:]]{64}' "$WITNESS_ROOT/private/usage.jsonl" || true
```

The repository tests carry the synthetic content/path sentinels used for the
manifest and ledger privacy scans, so reproducing the privacy result does not
require printing those sentinels or transcript bytes:

```bash
go test ./internal/trajectory -run '^TestAuditSnapshotUsage' -count=1
go test ./cmd/fak -run '^TestRunTrajectoryAuditSnapshotUsage' -count=1
```

Both focused commands passed. The affected build and vet commands also passed:

```bash
go build -buildvcs=false ./cmd/fak
go vet ./internal/trajectory ./cmd/fak
```

## Repository-wide acceptance limitation

This witness does not claim a globally green `make test-fast`. On the macOS
control point, the literal command stopped before compilation because the
Makefile's GNU timeout dependency was unavailable:

```text
/bin/sh: timeout: command not found
make: *** [smoke-build] Error 127
```

Running the underlying all-tree checks directly independently reproduced two
base defects at pristine `origin/main@d56d9780f8d5`; neither is in the snapshot
usage diff:

```text
vet: internal/modelperfobs/bandwidth_roofline_numa_test.go:109:15: undefined: discoverNUMATopology
```

```text
worker_usage_test.go:105: assessment = {Schema:fak-qwen-empty-usage-assessment/1 State:pending Reason:observation_window_open ... TurnsStarted:1 TurnsCompleted:1 ...}, want empty/turn_completed_without_usage
```

The commands that produced those base-only failures were:

```bash
go vet ./...
go test -short ./...
```

The focused product witness above is green; the unrelated repository-wide
failures remain reported rather than being misrepresented as acceptance.
