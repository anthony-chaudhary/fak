# Session reload (recall) — persist a finished session, reload it in a fresh process

The README sells one of fak's headline capabilities as *"a finished session that
reloads as a core dump."* This example is the runnable proof of that sentence: it
persists a finished agent session to disk, re-opens it in a **separate process**,
and shows the part that makes recall more than a snapshot — a tool result the gate
**sealed** at write time stays sealed across the process boundary, and a tampered
image fails closed at load.

```text
+----------------------+     +---------------------------+     +----------------------+
| process A: fak recall| --> | core image on disk        | --> | process B: fak debug |
| (write-time gate     |     | manifest.json + cas.json  |     | (sealed pages stay   |
|  seals poison)       |     +---------------------------+     |  sealed on page-in)  |
+----------------------+                |                      +----------------------+
                                        +--> tampered blob --> load fails closed
```

No key, no model, no GPU, no network. Go is the only prerequisite. Expected runtime: the
sample run completes in seconds and is deterministic over the bundled session fixture.

```bash
examples/session-reload/run.sh
```

That builds `fak`, runs the four witnesses below, and tears everything down. The
captured output is in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

## The two files that ARE the session

`fak` persists a finished session as a *core image* — two small, self-contained
files:

| File | Role | Analogy |
|---|---|---|
| `manifest.json` | The **page table**: one row per tool result with its role, the content **digest** (its address), length, taint, the quarantine state (sealed? reason? quarantine id?), and a frozen world-version. Carries no payload bytes. | the map you always carry |
| `cas.json` | The **content-addressed swap device**: a `{ digest → bytes }` store. A blob's key *is* the sha256 of its bytes, so the store is self-verifying. | the swap the map points into |

A reload reads the page table, then demand-pages bytes out of the swap device only
when a query touches them — never by replaying the whole transcript through a model.

## What the run proves (four witnesses)

The script drives two real fak verbs in two separate OS processes:

**A — process A: build + persist.** `fak recall --dir <img>` records a finished
airline-support session — two benign results, one indirect prompt-injection
(a poisoned refund policy), one secret leak — through the **shipped write-time
gate**, so the injection and the secret are quarantined as they are recorded. It
writes `manifest.json` + `cas.json`, then (still in process A) reloads the image
into a fresh in-process session and shows:

- a benign page resolves **byte-identical**;
- the sealed injection page is **refused** with no witness;
- after a witness `Clear()` it is **still** refused — the content re-screen
  re-quarantines it. *Clearance alone does not launder poison.*

**B — the artifacts on disk.** Both files exist and are tiny (here: a 1662 B page
table over an 894 B swap device). This pair is the entire portable session.

**C — process B: reload across the boundary (the moat).** `fak debug --dir <img>`
(the context-debugger verb, a pure consumer of `recall`) attaches to the image in a
**brand-new OS process** that shares nothing with process A's memory. The page
table still marks the poisoned and secret pages `SEAL`, a benign page pages in
byte-identical, and the sealed page is **refused on page-in**. The quarantine is a
property of the *image*, not of the process that wrote it. A plain snapshot would
hand back whatever bytes it stored; recall re-adjudicates every page-in.

**D — integrity: a tampered swap device fails closed.** The script flips one byte
*inside* a stored blob while leaving its digest key unchanged, so the blob no
longer hashes to its address, then re-opens the image. Load refuses the whole
thing — `corrupt CAS entry … (digest mismatch)` — rather than serving even the
untouched pages. Content addressing is the integrity check.

## Run it by hand

```bash
go build -o fak ./cmd/fak

# A) build + persist a finished session, and watch the in-process reload guarantees
./fak recall --dir /tmp/img

# B) the image is two files
ls -l /tmp/img            # manifest.json + cas.json

# C) a SEPARATE process re-opens the same image (sealed pages stay sealed on page-in)
./fak debug --dir /tmp/img

# D) tamper one byte inside a blob (keep its digest key) and re-open -> fails closed
cp -r /tmp/img /tmp/img-bad
#   the first uppercase char in cas.json is always inside a base64 value (keys are
#   lowercase hex), so flipping it corrupts a blob, never a key:
awk 'BEGIN{d=0}{if(!d&&match($0,/[A-Z]/)){c=substr($0,RSTART,1);r=(c=="Z"?"Y":"Z");$0=substr($0,1,RSTART-1) r substr($0,RSTART+1);d=1}print}' \
  /tmp/img-bad/cas.json > /tmp/img-bad/cas.json.tmp && mv /tmp/img-bad/cas.json.tmp /tmp/img-bad/cas.json
./fak debug --dir /tmp/img-bad   # -> exit 1: recall: corrupt CAS entry … (digest mismatch)
```

## Honest scope

The seal in this demo is set by the result detector, which is **evadable by
design** — it is a bonus, never the load-bearing floor. What this example proves is
*durability and structure*, not detector accuracy: whatever decision the gate made
at write time is preserved across the process boundary and cannot be quietly
flipped by editing the image (the digest check) or by a bare clearance (the
re-screen). The capability lock and the containment are the floor; recall makes
their verdicts survive a session boundary. See the security-substrate fences in
[`CLAIMS.md`](../../CLAIMS.md).

## Where this fits

- Parent package: [`internal/recall/`](../../internal/recall) — the core-image
  recorder (`Recorder.Persist`), the reloaded `Session` (`Load` → `Resolve` /
  `Clear` / `Recall`), and the integrity + re-screen gates. The test suite there
  (`recall_test.go`) is the unit-level form of these same witnesses
  (`TestQuarantineSurvivesTheSessionBoundary`, `TestClearIsNecessaryButNotSufficient`,
  `TestCorruptCASFailsClosed`).
- Internal writeups: [`docs/benchmarks/RECALL-RESULTS.md`](../../docs/benchmarks/RECALL-RESULTS.md)
  (the recall results) and [`docs/MEMORY-ECC-INTEGRITY.md`](../../docs/MEMORY-ECC-INTEGRITY.md)
  (the page-table syndrome that catches metadata tamper the body digest does not).
- Claims: [`CLAIMS.md`](../../CLAIMS.md) § *Session core-dump + context debugger
  (recall + cdb)* — the `[SHIPPED]` rows this example exercises.
- Consumer surface: `fak debug` is the **context debugger** (`cdb`) — it attaches
  to a finished session as to a core dump and demand-pages only the working set a
  follow-up touches. This example shows the *persist + reload* path underneath it.
- Concept: [`docs/CONTEXT-IS-NOT-MEMORY.md`](../../docs/CONTEXT-IS-NOT-MEMORY.md) —
  why a fact has to earn its way across the durable boundary.
- Sibling demo: [`trace-reset/`](../trace-reset/README.md) — the operator surface
  for clearing one trace's IFC mark at a session boundary.
