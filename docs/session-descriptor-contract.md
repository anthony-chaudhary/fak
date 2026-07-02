---
title: "Session-descriptor contract (fak.session.descriptor.v1)"
description: "The one session-identity join schema every rollup surface reads: gateway drive state, cross-host leaseref refs, and harness identity bound by exact id match, with a closed absence vocabulary and no self-reported progress fields. Issue #2214, epic #2209."
---

# Session-descriptor contract — `fak.session.descriptor.v1`

Owner: `internal/sessiondesc` (issue
[#2214](https://github.com/anthony-chaudhary/fak/issues/2214), epic
[#2209](https://github.com/anthony-chaudhary/fak/issues/2209)). Peer of the
[shared-task-record contract](shared-task-record-contract.md): that contract
carries shared TASK items; this one carries session IDENTITY. The golden wire
bytes are pinned by `TestDescriptorGoldenJSON`.

## Why

A fak session has four identities that never joined: the gateway DRIVE state
(`trace_id` on `/v1/fak/sessions` and the JSON `--log` sink), the cross-host
leaseref descriptor (`refs/fak/locks/session-<id>`), the harness identity
(which agent, which account), and the transcript/census namespace. Every
rollup surface (`fak fleet`, `tools/fleet_top.py`, `fak rollup`, the #2215
sidecar pane, the #1203 fleet fold) re-invented an ad-hoc join. This contract
is the single schema they read instead.

## The record

One `Descriptor` per session id, `schema: "fak.session.descriptor.v1"`, with
four key-space sections — `drive`, `ref`, `harness`, `census` (reserved for
#2213) — each carrying a `presence` token first:

| Token | Meaning |
|---|---|
| `BOUND` | populated from an observed source row |
| `ABSENT_NOT_OBSERVED` | that source was never consulted for this fold |
| `ABSENT_SOURCE_UNAVAILABLE` | consulted and FAILED (gateway down, git not executable) — never read an outage as "no such session" |
| `ABSENT_NO_BINDING` | consulted, answered, held no row for this id — a clean miss |

## The rules

1. **Exact join only.** Rows merge iff session ids are byte-equal. No fuzzy
   matching, no mtime heuristics; two different sessions never fold into one
   descriptor (`TestFoldNeverMergesDistinctSessions`).
2. **Typed absence.** A missing key space always states why, from the closed
   vocabulary above (`TestFoldAbsenceVocabularyPerSourceStatus`).
3. **Refuse the unidentifiable.** An empty id or a same-source duplicate id is
   an error, not a best-effort fold (`TestFoldRefusesEmptyAndDuplicateIDs`).
4. **No self-report.** A descriptor carries identity and observation pointers
   only — no progress/claim fields, by construction (the `dos_status`
   no-`claimed` discipline). Progress belongs to verified ledgers.
5. **Data-only package.** `sessiondesc` does no I/O and imports nothing
   internal; callers parse their own sources into the mirror row types
   (`DriveRow`, `RefRow`, `HarnessRow`) and state each source's status
   (`OBSERVED` / `UNAVAILABLE` / `NOT_CONSULTED`).
6. **Additive evolution.** Within v1, changes are additive; an unknown
   `schema` string means forward-incompatible — skip the record, don't blind
   the view.

## Consumers (planned)

The #2215 sidecar pane and the #1203 `fak session ls --fleet` fold read
descriptors; the #2213 cross-agent census binds the reserved `census` space.
None of these change the wire shape above.
