---
title: "Dispatch SLO glossary"
description: "Canonical terms for issue-dispatch throughput, closure, retry, and safety reports."
---

# Dispatch SLO Glossary

This page owns the SLO vocabulary for issue-dispatch reports and runbooks. Use
these words exactly in status rows, dry-run tables, close-arm notes, and operator
summaries.

## `spawned`

A worker process was actually launched for one issue. Dry-run rows that only say
`would_spawn` are not spawned.

## `attempted`

A dispatch or close arm took a real action for an issue: spawning a worker,
auditing a candidate SHA, closing, reopening, or requeueing. A skipped candidate
is not attempted.

## `witnessed`

An external check the worker did not author confirms the claimed effect.
Accepted witnesses include a passing parent-rerun test, `dos commit-audit` with
a diff witness, `dos verify`, or an independent read-back.

## `closed`

The GitHub issue is closed after the close arm reverified the resolving SHA. A
worker saying it is done, or a local commit mentioning `#N`, is not closed.

## `retried`

A later dispatch attempt intentionally reuses an issue after a failed, blocked,
stale, or unwitnessed attempt. A retry must carry a new worker/run id and
preserve the prior witness or blocker row.

## `stale`

The remembered status is older than the source it summarizes: git, the issue
state, a worker lease, a run ledger, or a commit-audit result. Stale rows are
refreshed before they drive spawning, closing, or throughput math.

## `unsafe-to-dispatch`

The issue must not be handed to an autonomous worker in its current form.
Typical reasons are missing route/scope/witness fields, a private boundary,
dependency hold, duplicate-risk hold, live-worker overlap, self-source guard
hold, or an operator gate.
