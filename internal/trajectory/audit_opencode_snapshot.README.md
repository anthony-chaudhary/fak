# Explicit OpenCode v2 snapshot audit

This is one optional source adapter in the ordinary `RunAudit` and
`fak trajectory audit` path. It consumes an operator-selected, independently
SHA-256-bound **standalone SQLite image**, never a discovered OpenCode home.
It is not the historical `session/message/part` reader requested in #12153,
an audit of a live database, or evidence that every session was captured.

## Pinned upstream contract and license

The schema/projection is adapted from MIT-licensed
[anomalyco/opencode](https://github.com/anomalyco/opencode) v2.0.15, immutable commit
`6f3639d82ed0760091792189b78f8eeb44f699b1`:

* `packages/core/src/session/sql.ts`:
  Git blob `4465d885b33aac614689f68ed8d8683a10729355`;
  `session_v2` and `session_message` tables.
* `packages/schema/src/session-message.ts`:
  `55ef745584dd9d59a1764bc3a6251f040d8a18fb`.
* `packages/schema/src/token-usage.ts`:
  `b8ce1d2ec9c866d6662a0bc4060d4257028da906`.
* `packages/core/src/session/history.ts`:
  `aeb7cfc9f81b2d9b27d3e43ff656ed7eb4352a16`;
  stored JSON omits the authoritative row ID/type.
* MIT `LICENSE`: `6439474beed8e0271df9862eff97ffd70ec2464c`,
  reproduced in `OPENCODE-AUDIT-LICENSE.txt`.

The upstream stores a five-bucket disjoint token vector and per-message usage,
including the compaction request's own usage. FAK's privacy projection,
immutable-image binding, bounded parser and conservative refusals are local
adaptations, not inherited upstream safety or performance guarantees.

## Explicit use, without source discovery

The producer must supply a consistent, standalone rollback-format image using
its supported export/backup process. This feature does **not** generate that
image, repair a WAL header, read a journal/WAL sidecar, checkpoint a live DB,
or prove producer version. The supplied digest binds exact bytes; a digest
calculated from an untrusted response is not independent authenticity evidence.

Example only; not executed as part of this source handoff:

```text
fak trajectory audit \
  --opencode-snapshot /ABSOLUTE/immutable.sqlite \
  --opencode-snapshot-sha256 TRUSTED_LOWERCASE_SHA256 \
  --opencode-snapshot-profile opencode-2.0.15-session-v2 \
  --jsonl /ABSOLUTE/audit.jsonl
```

All three flags are required. The image-only CLI mode replaces the source list,
defaults to all image sessions (`--since 0`), and rejects live-root overrides,
prompt filters, existing capture/replay modes, external baselines and usage-ledger
modes. Explicit `--since` selects sessions by stored update time; it includes
each selected session's complete stored history, not a message-level time slice.
An output may not name the input or an already existing hardlink alias.
This is not protection against a same-user adversary racing output paths.

The Go entrypoint `NewOpenCodeSnapshotSource` constructs the corresponding
`AuditSource`. Empty options leave existing source discovery/unsupported
OpenCode behavior unchanged. A configured source cannot fall back to a live
root, and multiple snapshot images in one RunAudit are refused rather than
silently unioned. The Go API also rejects mixed snapshot/live source lists and
any baseline composition involving a snapshot.

## Admission and resource bounds

The final source is an ordinary regular file, not a final-component symlink.
A bounded read is SHA-256-checked and checked for size/identity/mtime changes.
The exact bytes are sent on stdin to isolated Python (`-I -S -B`); its SQLite
connection is only `:memory:` and uses `Connection.deserialize`.
The source path and private payload are never interpreter arguments.

This uses the existing standard-library Python/SQLite approach without a new
Go module or Python package. Python 3.11+ APIs and a linked SQLite implementing
deserialize/setlimit are required at runtime. Missing capabilities produce a
typed refusal; there is no installation, live-file or weaker-reader fallback.

Admission checks required table names are ordinary tables, exact stored column
names/types and a single ID primary key, with no hidden/generated columns or
views. Unknown layouts are refused. It does not authenticate the producing
application from schema shape or prove arbitrary semantic equivalence.

Fixed limits:
* image: 16 MiB, exact rollback-format header/page count; WAL-mode images refused;
* selected sessions: 32; total selected messages: 2,048;
* each message JSON: 64 KiB actual UTF-8 bytes; total JSON: 8 MiB;
* JSON pre-scan: nesting 32, lexical node/separator budget 8,192 before materialization;
* assistant content: 128 items; total tool records: 4,096;
* integer bucket and aggregate total: nonnegative exact integers, at most 2^53-1;
* sanitized child stdout: 2 MiB; child stderr: 4 KiB and never relayed;
* SQLite length/SQL/column/expression/variable/attachment limits, extension loading
  disabled, trusted_schema OFF and query_only ON verified, and a static authorizer
  denying ATTACH, DML/DDL, arbitrary functions and trigger/view-origin reads.

Queries preflight row/text totals before selecting JSON. SQLite has a cooperative
3-second/4-million-VM-step budget, checked again between JSON records. The child
has a 5-second cancellation deadline with Wait/stdio drain and a 1-second pipe
WaitDelay. These are **not** a hard physical-I/O, RSS or wall-clock guarantee:
file reads and native SQLite work may not yield immediately. No network request,
credential discovery, plugin loading, arbitrary SQL or skill/client execution is
introduced. Source images remain private and unmodified.

## Accounting and explicit unknowns

Only sessions with complete supported usage are applied. Missing/fractional/
negative/overflow usage, duplicate or ambiguous message identities, unfinished
assistant/tool/compaction state, unknown records, forks and aggregate mismatches
produce typed incomplete-coverage refusals. The entire affected session is
excluded, not converted to a qualified zero. Global admission errors similarly
emit a source refusal and make broad-efficiency conclusions unsupported.
Already validated sessions may remain visible beside a later source refusal;
callers must retain refusal and denominator rows, not read the numeric sum alone.

Canonical input/cache-read/cache-write remain disjoint. Canonical output equals
upstream output plus reasoning; the original reasoning bucket is retained in
`opencode_snapshot` evidence and must not be added a second time.
Per-message usage must sum exactly to the selected session's five stored totals.
IDs are stable SHA-256 labels. Tool status and explicit executed=false survive;
tool text, arguments, results, error bodies, prompts, paths, model identity and
credentials are not exported. Model attribution is explicitly `not_exported`;
it is not zero-cost or known model coverage. The explicit coverage label is
`exact_usage_and_tool_status_only`; BroadEfficiencySupported remains false even
for a completely admitted image because model/latency/repeated-failure coverage
is not established. This scope marker is persisted in the JSONL summary;
unobserved repeated-failure rate is null with unknown semantics/normalization.
ReadAuditBaseline refuses such a summary, including for a later ordinary-source
audit, and internal repeated-failure comparison keeps its values unknown.
Parent/child and fork completeness,
provider invoices and historical corpus coverage are not inferred.

## Source witnesses and outstanding qualification

The new fixtures build only synthetic in-memory SQLite images, write them into
test-owned temporary directories, and call normal RunAudit/CLI dispatch.
They cover exact five-to-four bucket accounting and tool status, privacy,
determinism, immutable binding, WAL refusal, explicit cancellation, schema/JSON/
row/integer limits, forks/missing usage, output caps, image-only CLI selection,
JSONL-to-baseline refusal, mixed/API source refusal and unchanged ordinary
baseline comparability.
No existing test assertion is weakened.

Focused commands for a separately admitted qualified build/test lane:

```text
go test ./internal/trajectory -run '^TestOpenCodeSnapshot' -count=1
go test ./cmd/fak -run '^TestTrajectoryAuditOpenCodeSnapshot' -count=1
```

These commands, the installed controller, real producer snapshots, Python/SQLite
runtime compatibility, concurrency and performance are **UNRUN** at source
publication. A host without the required Python APIs may explicitly skip fixture
construction; that is not a passing reader qualification. No live or restricted
session database was inspected to implement this module. The scope is this
pinned optional snapshot adapter, not completion of every historical OpenCode
schema/corpus or broader cross-harness audit objective.
