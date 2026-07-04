# Expected output — verified-memory-recall

Captured from `bash examples/verified-memory-recall/run.sh` run from the repo root
(store path shown as `…/store`; yours will be your absolute checkout path).

```
== the store — three authored notes, one per read-time verdict ==
  fresh.md  names internal/memq/exec.go   (a path that EXISTS -> verifiable, fresh)
  stale.md  names internal/gonepkg/gone.go (a path that is GONE -> refused as stale)
  prose.md  a preference, nothing checkable (-> rendered hedged, unverifiable)

== the orientation block (fak memory recall --store examples/verified-memory-recall/store) ==
# Verified memory recall (memory store …/store; intent: "where does the memory algebra executor live")

## Gate helper (fresh.md) [fresh]

The memory algebra executor lives in internal/memq/exec.go.

## Preference (prose.md) [unverified]

The user prefers the outcome stated first, then the evidence.

withheld (never injected as fact):
  - stale.md [withheld:stale_recall_artifact] path "internal/gonepkg/gone.go": path missing from working tree

stats: scanned=3 rendered=2 withheld=1 ~tokens=31

The stale note is not silently dropped and it is not rendered as fact — it is
listed as a refusal with the failing claim (internal/gonepkg/gone.go) named as
evidence. That is the whole point: a moved file or renamed flag can no longer
load wearing the authority of a fact. This is a backend, not a new driver —
'fak memory drivers' is unchanged by it.
```

The two witnesses this issue (#2347) turns on:

1. **The fresh note renders.** `fresh.md` names `internal/memq/exec.go`, a path that
   exists in this checkout, so its claim verifies and it renders tagged `[fresh]`.
2. **The stale note is refused with its claim named.** `stale.md` names
   `internal/gonepkg/gone.go`, which is absent, so it is listed under
   `withheld (never injected as fact)` as `[withheld:stale_recall_artifact]` with
   `path "internal/gonepkg/gone.go": path missing from working tree` — the failing
   claim is named, not silently dropped.

The prose-only note renders hedged `[unverified]` (nothing to check), and
`fak memory drivers` output is unchanged — this is a backend, not a driver.
