---
title: "Host termination provenance"
description: "How fak manage and the host-crash sensor share one evidence ledger, recording Windows console close, logoff, and shutdown without duplicating a single event."
---

# Host termination provenance

`fak manage` and the always-on `fak host-crash` sensor share one host evidence
ledger: `%FAK_STALL_DIR%/host-crashes.jsonl` when configured, otherwise the
platform user-config `fak/host/host-crashes.jsonl`.

On Windows, guard installs a console-control observer before launch. For
`CTRL_CLOSE_EVENT`, `CTRL_LOGOFF_EVENT`, and `CTRL_SHUTDOWN_EVENT` it appends a
`fak.host-termination.v1` marker containing only `control_type`, `guard_pid`,
`console_session`, and `observed_at`, then preserves normal Windows teardown.
Application Error Event 1000 crash signals from `fak host-crash` remain in the
same JSONL ledger. Readers discriminate rows by schema, so the two producers do
not duplicate an event.

`fak audit diagnose` joins a `CHILD_CRASH` wave to the nearest marker within
five seconds. A correlated wave reports the observed control type. A wave with
no marker reports `EXTERNAL_UNKNOWN`: `TerminateProcess`, power loss, and other
unobservable exits are never assigned a fabricated killer.
