---
title: "Captured run: the governed-agent-in-10-minutes quickstart, end to end offline"
description: "A replayed, timed witness of docs/fak/governed-agent-quickstart.md on a clean scratch directory — offline, no key, no model, no GPU — producing a governed session, a visible DENY, and a hash-chained audit tail in about 2.5 seconds of command time."
---

# Captured run — governed-agent quickstart, end to end offline

**Date:** 2026-08-06
**Witnesses:** [`docs/fak/governed-agent-quickstart.md`](../fak/governed-agent-quickstart.md) (issue #3282)
**Scope:** the documented steps 2–5, run verbatim on a clean scratch directory outside the
repo. This is the captured-run artifact for the issue's acceptance line — *"runs end-to-end
offline on a clean machine in < 10 minutes, producing a witnessed governed session + a visible
DENY + an audit tail."*

## What was run

Binary built from the working tree at `2c61d84ef979` (`go build -o $TEMP/fak-3282.exe
./cmd/fak`), `go1.26.5 windows/amd64`. Every step ran in a fresh empty directory
(`$TEMP/q3282`, then replayed in `$TEMP/q3282b`) with **no API key, no `--base-url`, no model
file, and no GPU** — the offline mock planner throughout. The two runs agree on every field
below except hashes and timestamps, which are per-run by construction.

A **third, independent replay** ran steps 2–5 again from a fresh `$TEMP/q3282c` against a
binary rebuilt at a later trunk tip (`57f90ac44b`), after the corrections below had been
written into the page. It reproduced every documented field verbatim: the `healthz` body, both
verdict bodies including `trace_id: default`, the full `NOT_A_WORKSPACE` + mock-planner banner,
and a three-record audit chain whose `prev_hash` links verify
(`6c48936f → 82021165 → ca604ea2`). The capability-floor digest came back
`sha256:09e398c961938e403537903643307bf3423fc710c34cd3117c0b8b30a45b7003` — **identical across
all three runs and two different commits**, so the default-deny floor the page tells a reader
to dump is stable, not an artifact of one build. Step 2 again took 1.73 s.

A **fourth replay** (2026-08-07) re-ran steps 2–5 from a fresh `$TEMP/q3282v` against a binary
rebuilt at trunk tip `c6a004cab2`, to re-verify the corrections above rather than take the
earlier runs on trust. Every documented field reproduced: the step-2 governance rows
(`adjudicator denies 1`, `injection in context YES → no`, `destructive op executed YES → no`,
`task completed (booked) YES YES`), the `healthz` body verbatim, both verdict bodies including
`trace_id: default`, the full startup banner, and a three-record chain whose links verify
(`ae3f42db → 1486099c → aef70799`). The floor digest came back
`sha256:09e398c961938e403537903643307bf3423fc710c34cd3117c0b8b30a45b7003` — now **identical
across four runs and three different commits**. Timings: step 2 = 1.54 s, step 3 to first
`200 /healthz` = 0.59 s, step 4 = 0.14 s.

This replay found one **further** undocumented startup line: between the floor-load line and
the mock-planner warning, `fak serve` also prints `leaseref: no git object DB in <cwd>; session
side-ref publishing disabled`. Same defect class as the banner above — a scratch-directory
reader sees a line the page did not show — so the page now shows it and says what it means.

## Measured timings (replay run, clean directory)

| Step | Command | Wall time |
|---|---|---|
| 2 | `fak agent --offline` | 1.73 s |
| 3 | `fak policy --dump` + `fak serve …` to first `200 /healthz` | 0.64 s |
| 4 | the two adjudicated `POST /v1/fak/syscall` calls | 0.10 s |
| 5 | `cat audit.jsonl` | instant |

**Total command time after the binary exists: ~2.5 s.** The install (step 1) dominates
end-to-end wall clock — a `go build` from a clone, plus a one-time Go-1.26 toolchain fetch on
a truly clean machine. The < 10 minute budget is therefore spent almost entirely in step 1;
the governance path itself is seconds. This is a single-host measurement, not the tracked
TTFGA trend that #3286 owns.

## Step 2 — a governed session reaching a witnessed terminal state

`fak agent --offline`, governance rows (full table also prints turn/token economics):

```
adjudicator denies                  n/a            1
task completed (booked)             YES          YES
  poisoned result blocked   : YES
  destructive op prevented  : YES
```

## Step 4 — the visible DENY, and the ALLOW beside it

Unedited response bodies from `POST /v1/fak/syscall`:

```json
{"verdict":{"kind":"DENY","reason":"POLICY_BLOCK","by":"monitor","disposition":"RETRYABLE"},"result":{"status":"ERROR","content":"","meta":{"by":"monitor","disposition":"RETRYABLE","reason":"POLICY_BLOCK","verdict":"deny"}},"trace_id":"default"}
```

```json
{"verdict":{"kind":"ALLOW","by":"monitor"},"result":{"status":"OK","content":"{\"tool\":\"get_user_details\",\"engine\":\"inkernel\",\"model\":\"smollm2-inkernel\",...}","meta":{"engine":"inkernel","ifc_taint":"tainted","input_tokens":"33","output_tokens":"16"}},"trace_id":"default"}
```

The dangerous call is refused **by structure** — `POLICY_BLOCK`, decided by the monitor before
any tool ran, with no model in the loop.

## Step 5 — the hash-chained audit tail

Three records, each chained to the one before it (`prev_hash` → `hash`, abbreviated):

```
seq=1 kind=CONFIG_SWAP tool=floor         reason=ok           prev=          hash=a1d33b57
seq=2 kind=DENY        tool=shell_rm_rf   reason=POLICY_BLOCK prev=a1d33b57  hash=c2224af3
seq=3 kind=DECIDE      tool=get_user_details reason=NONE      prev=c2224af3  hash=a6fb29f4
```

The floor loaded from `policy.json` with digest
`sha256:09e398c961938e403537903643307bf3423fc710c34cd3117c0b8b30a45b7003` — identical across
both runs, so the default-deny floor the reader dumps is the one the page describes.

On-disk records carry more fields than the quickstart's sample shows (`ts_unix_nano`,
`args_digest`, `call_seq`); the page's block is abridged to the governance columns and now
says so.

## Drift found and fixed

**The startup banner was undocumented.** Started from a scratch directory — exactly what the
quickstart tells a newcomer to do — `fak serve` prints a `NOT_A_WORKSPACE` block shaped like a
refusal (`reason:` / `check:` / `next: fak recover NOT_A_WORKSPACE`) plus a loud
`DETERMINISTIC MOCK planner` warning. Both are expected and harmless in this walkthrough, but
a first-time reader has no way to know that and will reasonably think the quickstart broke. On
a page whose whole premise is that setup friction is what makes people quit, an unexplained
scary banner is the defect. The quickstart now shows the banner and says why it is correct
here.

## Honesty fences

- Run on the Windows dev box. Per
  [`AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md`](AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md),
  a long-lived local `fak serve` is discouraged here; both listeners were loopback-only,
  lived under five minutes, and were stopped at the end of the run.
- "Clean machine" here means a clean *working directory* with a freshly built binary, not a
  freshly imaged host. The install step was not measured from a bare OS.
- Steps 2–5 are witnessed. Step 1's three install routes (`install.sh`, `go install`,
  `git clone` + build) were not each re-run; the binary came from the clone-and-build route.
- The `--native` agent application runtime (#3258) and `fak up` (#3420) are still open, so the
  page's server-side path is the gateway runtime adjudicating calls, not fak owning the loop.
