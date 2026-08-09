# Micro-context dogfood readout — 2026-08-08

## Verdict

**BLOCKED honestly, with the surfaced defect filed as #6004.** The shipped
`cmd/microcontextdemo` spine runs, but it cannot consume this repository's live
work yet, so a synthetic run cannot satisfy #5833's dogfood contract.

## Live input inspected

- Source: GitHub OPEN-issue snapshot produced by `gh issue list --state open`.
- Snapshot size at inspection: 100 bounded records (the router's live cap).
- Contract-ready leaves found by `fak issue contract --from-issues`: 6.
- Highest-priority leaf #5792 was already intent-leased; #5824 and #5825 had
  peer-dirty likely paths, which the worker did not touch.

## Captured run

The binary was built from an isolated `git archive HEAD` so peer WIP could not
mask or fabricate the result:

```text
$ microcontext-dogfood.exe -contexts 30 -workers 4 -turns 1
{
  "schema": "fak-microcontext-spine/1",
  "verdict": "PASS",
  "logical_shards": 30,
  "physical_workers": 4,
  "completed": 30,
  "failed": 0,
  "shared_base_installs": 1,
  "turn_count": 30,
  "peak_in_flight": 4,
  "scope": "synthetic endpoint; proves bounded harness fan-out and shared-base semantics, not model tokens/sec",
  "mode": "synthetic"
}
```

The result is useful negative evidence: the command accepted only an integer
shard count and explicitly reported synthetic scope. `-help` exposed no live
repository-input flag. Treating that output as dogfood would violate #5833's
confusion-risk clause.

## Defects surfaced

- #6004 — add a bounded, sanitized live-repository-work input seam and captured
  regression artifact. Marker: `dogfood-defect: microcontext-live-data-input-missing`.

## Reproduction

```powershell
gh issue list --state open --limit 100 --json number,title,body,labels > open.json
fak issue contract --from-issues open.json --json
# Build cmd/microcontextdemo from `git archive HEAD`, then:
microcontext-dogfood.exe -help
microcontext-dogfood.exe -contexts 30 -workers 4 -turns 1
```

No other defect was inferred from this blocked run. Once #6004 ships, rerun
#5833 against the same bounded live snapshot and replace this negative witness
with the required consumed-record readout.
