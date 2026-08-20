---
title: "Stale-work review packets: evidence before action"
description: "How fak stale-work finds review candidates from semantic history and freshness evidence while abstaining instead of editing or deleting work."
---

# Stale-work review packets

`fak stale-work --json` is an advisory, read-only discovery pass over tracked Markdown and
claims. It emits schema `fak.stale-work.packet.v1`; it never edits, deletes, files, or closes anything.
Age contributes at most two points and **cannot** produce `review`: a semantic dependency commit
or an unpointed live version claim (the existing `docfreshrsi` detector) is required. Missing
evidence produces `abstain` (fail-to-abstain), not a guess.

```sh
fak stale-work --selfcheck
fak stale-work --json --limit 20
fak stale-work --path docs/operator-guide.md --json
```

The scanner uses Git's semantic commit history and the existing `docfreshrsi` version-claim
rules. Its dependency provenance is compatible with `modver` history; generated and historical
surfaces follow the exemptions used by `devfresh`.

Excerpts are whitespace-normalized, capped at 240 bytes, and hash-addressed. Dependency commits,
score components, dedupe key, issue state, proposed definition of done, and witness command are
included. Packet size is reported; no efficiency gain is claimed without a measured comparison
to the real `git log` + grep + full-file-read alternative.

## Two-stage contract

1. Scan and adjudicate. Historical, generated, vendored, already-open, and previously adjudicated
   entries are exempted/deduped. A reviewer validates executable/generated checks named by the
   packet's evidence and may file one dedicated issue per dedupe key using the supplied DoD.
2. Only that dedicated issue authorizes a fresh headless worker to edit or delete candidate
   content. The discovery packet itself is never authority to mutate a candidate.

## Deterministic issue/dispatch loop

`fak stale-work loop` is the stage-2 orchestrator. It consumes the packet, an independently
captured issue snapshot, optional prior adjudications, and optional dispatch witnesses. The
default is a pure plan: it creates no issues, launches no workers, writes no state, and never
edits candidate content.

```sh
fak stale-work --json --limit 20 > packet.json
fak stale-work loop --packet packet.json --issues open-and-recent.json --json
fak stale-work loop --selfcheck
```

For each supported candidate the plan:

1. derives a stable evidence digest and one dedicated, contract-valid issue body;
2. deduplicates against open and recently closed issues by marker, exact path, or evidence
   digest;
3. refuses dispatch until the matched issue passes the existing strict
   `fak issue contract` rules;
4. prices overlapping issue paths through the existing issue-cohort wave seam;
5. assigns a distinct `resolve-<lane>-<issue>` identity to every issue and renders the existing
   `fak dispatch tick` command; and
6. reconciles only from independent issue state, `dos commit-audit` (`OK` /
   `diff-witnessed`), a green test/read-back claim, and a witnessed
   `retained|updated|deleted` decision. Worker narration is not an input.

The issue snapshot is the same subset emitted by:

```sh
gh issue list --state all --limit 500 \
  --json number,title,body,state,closedAt,url,labels > open-and-recent.json
```

Bound the file to open plus recently closed issues before a live run. The loop treats every row
supplied as part of the dedupe horizon.

Witness input is an array:

```json
[
  {
    "issue": 123,
    "sha": "abc123",
    "verdict": "OK",
    "witness": "diff-witnessed",
    "test_claim": "CLAIM_TEST_GREEN",
    "decision": "updated",
    "source": "dos commit-audit + issue read-back + focused test ledger"
  }
]
```

Reconciliation is closed-vocabulary: `SHIPPED`, `STILL_OPEN`, or `ABSTAIN`. A closed issue
without the git/test/decision rungs is `ABSTAIN`, never a ship.

### Adjudication cache and invalidation

`--state state.json` reads schema `fak.stale-work.loop-state.v1`. A terminal
`valid|retained|updated|deleted` row is reused only when both its evidence digest and independent
`witness` are present and unchanged. Any changed candidate evidence invalidates the cache and
re-plans the issue. `--state-out next-state.json` is the explicit persistence gate; absent that
flag, the loop writes nothing.

### Live gates

Live effects are separate and explicit:

```sh
# File only missing contract-valid issues; requires the dedupe snapshot.
fak stale-work loop --packet packet.json --issues open-and-recent.json --live-issues --json

# Launch only already-filed, contract-valid issue units through dispatch tick.
fak stale-work loop --packet packet.json --issues open-and-recent.json --live-launch --json
```

`--live-issues` refuses without `--issues`. `--live-launch` preserves cohort wave order and
passes the issue-specific worker identity and lease tree to `fak dispatch tick --live`.
Issue creation and launch remain independent gates so an operator can inspect the newly filed
contracts before any worker starts.

The working-spine proof is captured in
[`docs/_witnesses/stale-work-live-2026-08-13.json`](_witnesses/stale-work-live-2026-08-13.json).
