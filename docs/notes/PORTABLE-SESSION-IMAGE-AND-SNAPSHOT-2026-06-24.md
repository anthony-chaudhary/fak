---
title: "fak note: portable session images + a uniform dump/restore seam (2026-06-24)"
description: "Make a session — and any primitive on the loops ladder (a turn, a tool, a session, a fleet, an RSI loop) — a first-class value you can dump, archive, offload, and restore: across hosts, users, instances, VMs, and a model change. The session ships as a rich, model-agnostic, integrity-checked image; the ladder ships as one uniform snapshot envelope."
---

# Portable session images + a uniform dump/restore seam

> Date: 2026-06-24. Status: SHIPPED (the spine + the session instance + the generic seam).
> Witness: `go test ./internal/session ./internal/sessionimage ./internal/snapshot`.

## 0. The goal in one paragraph

A running agent is a stack of nested loops — a syscall inside a turn inside a session inside a
fleet inside an RSI loop ([the loops thesis](../explainers/engineering-is-building-loops.md)).
The goal is that **any** level of that stack be a value you can freeze to bytes and thaw back:
dump it, archive it, offload it to object storage, ship it to another host or user or
lightweight VM, and **resume** it — even under a different model. Two shipped pieces deliver
that: a rich, model-agnostic **session image** (`internal/sessionimage`) for the session level,
and a **uniform snapshot seam** (`internal/snapshot`) that gives every other rung the same
dump/restore shape.

## 1. Why this was not already there

The three durable session primitives existed but were disjoint, and none was portable as a unit:

| Primitive | What it held | The gap |
|---|---|---|
| `internal/session` (drive) | run-state / budget / priority / pace, live, in-memory | **never persisted** — the [SESSION-CONTROL §5](SESSION-CONTROL-STATE-AS-FIRST-CLASS-2026-06-24.md) fence: "a process restart re-attaches a session at its defaults" |
| `internal/recall` (content) | the core image: page table + content-addressed swap device + ctxplan index, quarantine-gated | query-only; no drive, no model/account identity, not resume |
| `internal/trajectory` | the per-turn audit corpus (JSONL) | a separate export again |

Nothing tied them into one object you could carry. A session on a laptop that had to move to a
server, a VM, or a different user's instance — or simply survive a model swap — had nothing to
pick up.

## 2. The session image (`internal/sessionimage`)

A bundle directory (or its single-file `.faksession` tar):

```text
image.json        Meta: identity, model/engine/account/residency/host, the portability
                  contract, a sha256 integrity index over every other part, the migration log
session.json      the drive State          ← the §5 persistence sibling, finally built
manifest.json     recall page table        ┐
cas.json          recall swap device        │ the recall core image (optional: a fresh
index.json        recall ctxplan index     ┘ session has no content yet), written by recall
trajectory.jsonl  the per-turn audit corpus (optional)
```

Three moves make it first-class:

- **`session.Table.Restore`** — the durable-resume write: load a drive `State` verbatim, **Rev
  preserved**, the load-time inverse of `Snapshot`. A `STOPPED` session restores `STOPPED` (with
  its reason), never silently revived as `RUNNING`. This is the one write that may re-establish a
  terminal session, precisely because resume must be faithful.
- **`DumpDir` / `Pack` → `Unpack` / `LoadDir` → `Rehydrate`** — dump to a bundle, pack to one
  deterministic `.faksession` (stdlib `archive/tar`, zero deps), unpack on a fresh host into a
  fresh directory, and rehydrate: re-attach the drive into a fresh `session.Table`, reload the
  gate-armed recall `Session` + ctxplan index.
- **`Migration`** — a resume that targets a different model or host appends an audited entry, so
  "this session ran under model A and resumed under model B on host X" is a fact on the record.

### The model-agnostic contract (why it survives a model change)

The image stores a session's **logical** state — the drive axes, the content-addressed page
table, the candidate index over roles+digests, the trajectory of decisions. It stores **no**
model-specific binary state: no KV cache, no tokenizer-specific token ids, no sampler RNG. The KV
cache is a *cache*, not state — a resumed session rebuilds it on the first turn — so a restore may
target a different model and re-prefill from the same logical content. `Portability.KVIncluded`
is `false` for v1 by design.

### What rides through unchanged: the moat

The load-bearing recall property survives the offload boundary: a slice the gate **quarantined**
stays sealed after the image is packed to a tar, shipped, unpacked into a fresh directory, and
reloaded under a new model — and a witness `Clear()` still does not launder it (the content
re-screen holds). Integrity is content-addressed: every part is sha256-checked on `LoadDir`, so a
truncated or tampered offload fails closed; an untrusted tar entry that tries to escape the target
directory is refused. Witness: `TestPackUnpackPreservesQuarantineAcrossBoundary`,
`TestLoadDirFailsClosedOnTamper`, `TestUnpackRejectsPathTraversal`.

## 3. The uniform seam (`internal/snapshot`) — any primitive, one shape

The session is the rich, multi-part instance. Every *other* rung of the ladder gets the same
dump/restore shape from one envelope:

```go
type Snapshot struct {
    Envelope, Kind, ID string
    CreatedUnix int64
    AppVersion  string
    Meta        map[string]string
    Body        json.RawMessage   // the primitive-specific state
    BodyDigest  string            // sha256 over the CANONICAL (compacted) body
}

func Marshal(kind, id string, body any, meta map[string]string, now int64) (Snapshot, error)
func Parse(b []byte) (Snapshot, error)   // verifies version + body digest, fail-closed
func (s Snapshot) Into(v any) error
```

Because `Body` is `json.RawMessage` and `Marshal` takes `any`, a new primitive becomes dumpable
with **no change to the package** — you call `Marshal(kind, id, yourState)` and you have a
portable, integrity-checked dump. The digest is taken over the *compacted* body, so a pretty-print
round-trip stays verified while a real content change still fails closed.

A registry names the ladder (so a tool can ask "what can I dump?"), and typed codecs ship for the
levels whose state is a single value:

| Kind | Level | Codec | State |
|---|---|---|---|
| `turn` | turn | `DumpTrace` / `RestoreTrace` | a trajectory's `Turn` rows |
| `tool` | syscall | generic `Marshal` | a tool's definition / facts |
| `session` | session | `internal/sessionimage` | the rich multi-part image (above) |
| `fleet` | fleet | `DumpFleet` / `RestoreFleet` | a `session.Table`'s whole-fleet drive snapshot, restored verbatim via `Table.Restore` |
| `rsi` | rsi | generic `Marshal` | an RSI loop's keep/revert row(s) |

The fleet codec is the proof the seam composes with a live primitive: dump every session's drive
at an instant, restore them into a fresh table on another host — a fleet, picked up and put down.
Witness: `TestFleetRoundTripViaRestore`, `TestTraceRoundTrip`, `TestGenericRoundTripAnyBody`.

## 4. What this is NOT (so it stays honest)

- **Not a KV-cache mover.** The image is model-agnostic *because* it drops the KV; a resume
  re-prefills. If you want bit-identical decode continuation on the *same* model, that is a
  different, model-pinned artifact this does not claim.
- **Not a live-migration protocol.** It is dump → carry → restore at a turn boundary, not a
  zero-downtime hand-off of an in-flight decode.
- **Not a second trust gate.** Content trust stays in recall's quarantine gate, which re-screens
  every page-in on the restoring host; the image layer only adds *integrity* (sha256 over the
  parts), not a new admission policy.

## Related

- [`SESSION-CONTROL-STATE-AS-FIRST-CLASS-2026-06-24.md`](SESSION-CONTROL-STATE-AS-FIRST-CLASS-2026-06-24.md) — the drive-state value this persists (`§5` fence, now closed).
- `internal/recall` — the content core image this composes (the durable quarantine moat).
- [`engineering-is-building-loops.md`](../explainers/engineering-is-building-loops.md) — the loops ladder the snapshot kinds map onto.
