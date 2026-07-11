---
title: "Design: Confluence-items to fak seam"
description: "A design proposal, not yet built, for a generic scrub-safe seam that turns Confluence items into fak work, capturing the intended shape."
---

# Design note: generic, scrub-safe Confluence-items → fak seam

**Status:** DESIGN / proposal — not yet built. Captures the intended shape and, per the
operator steer, *where the seam lives*. Written after the cachevalue-feed work; the inbound
importer itself is future work.

## Problem

Today the Confluence integration is **outbound only**: fak authors page sources
(`docs/confluence/*.confluence.xhtml`) and the `confluence-helpers` tool pushes them to the
wiki from the bridge (see [README.md](./README.md)). We want the **inbound mirror** — pull
Confluence *items* (pages, tasks, comments) *into* fak — with two hard constraints:

1. **Generic** — fak's side must not know Confluence's schema. It consumes an abstract
   `Item`, so a later source (Jira, Google Docs, an intranet page) reuses the same fak-side
   seam without change.
2. **Scrub-safe** — raw wiki content routinely carries secrets, internal hostnames, and PII.
   None of it may land in fak's tree, transcripts, or agent context. Scrubbing is
   **fail-closed** (redact-or-drop; never emit unknown-sensitive) and happens *before* the
   boundary, not after.

## Decision: the seam lives in `confluence-helpers`, behind its DI boundary — not in fak

This is the operator steer, and there is a structural reason it is the *only* correct home:

- **Network reach.** `<internal-confluence-host>:8090` is reachable only from the bridge
  (the internal bridge). The fak laptop and the Windows agent box cannot reach the wiki at all, so fak
  **structurally cannot be the fetch site.** The fetch must run where `confluence-helpers`
  already runs.
- **It already owns the Confluence-specific surface.** Transport, `CONFLUENCE_TOKEN` auth, the
  `confluence-meta` model, and create-or-update idempotency all live in `confluence-helpers`.
  The inbound reader is the same surface in reverse; splitting it into fak would duplicate the
  transport and drag Confluence schema into fak.
- **Scrub-at-source is fail-closed.** If scrubbing lived in fak, raw bytes would have to cross
  the boundary *first* to be scrubbed *second* — the exact window where a token leaks into a
  transcript. Scrubbing on the `confluence-helpers`/bridge side means unscrubbed content never
  leaves the bridge. This mirrors the secret-masking discipline used elsewhere: mask before it
  enters, not after.
- **One audited channel.** The scrubbed items ride the same sanctioned Slack bridge relay that
  publish already uses, so there is a single in/out path to audit, and fak depends on an
  *interface*, not on Confluence.

## The DI seam

`confluence-helpers` exposes an `ItemSource` with a `Scrubber` **injected** into it (its DI
boundary — the same place the push/pull transport is wired). fak consumes only the abstract
`Item` stream, through a thin adapter:

```
confluence-helpers (on the bridge)                     fak (laptop / agent box)
┌─────────────────────────────────────────┐            ┌────────────────────────────┐
│ ItemSource.List()/Fetch()               │            │ ItemConsumer               │
│   ── raw Confluence page/task/comment    │            │   receives Item{scrubbed}  │
│   ▼                                      │  scrubbed  │   ▼                        │
│ Scrubber (injected, fail-closed) ───────┼──items over─┤ adapter → fak artifact     │
│   redact-or-drop; record what+how-many   │  the bridge │   (issue | context | memo) │
│   never the secret itself                │            │   refuse if !scrubbed      │
└─────────────────────────────────────────┘            └────────────────────────────┘
        raw bytes never leave the bridge
```

### The generic `Item` (the boundary contract)

Deliberately Confluence-agnostic:

| field        | meaning                                                                 |
|--------------|-------------------------------------------------------------------------|
| `id`         | stable source id (opaque to fak)                                        |
| `kind`       | `page` \| `task` \| `comment` \| … (source-declared, small closed set)  |
| `title`      | scrubbed                                                                 |
| `body`       | scrubbed                                                                 |
| `source_ref` | provenance: system + url/space, so fak can link back for a human        |
| `labels`     | scrubbed, normalized                                                     |
| `scrubbed`   | `true` only if every field passed the scrubber                          |
| `redactions` | count + rule ids of what was scrubbed — **never the scrubbed value**    |

### Scrub-safe contract

- Every field crossing the boundary passes the injected `Scrubber`.
- The scrubber is **fail-closed**: on an unknown-but-sensitive-looking span it redacts (or
  drops the item), never passes it through.
- `redactions` records *that* something was scrubbed and by which rule, for audit — it never
  carries the secret.
- fak **refuses any item with `scrubbed != true`** — a missing/false flag is treated as
  unsafe, not as "probably fine."

## What fak owns (minimal)

- The `ItemConsumer` interface and a thin adapter that turns an `Item` into a fak artifact.
- Nothing Confluence-specific: no token, no wiki URL, no XHTML/`confluence-meta` parsing, no
  direct network call to the wiki.

## Open question (drives the adapter, not the seam)

**What does a Confluence item *become* in fak?** — a filed issue, read-only agent context, or
a memory entry? This is the one decision still open; it shapes only fak's adapter, and the
generic `Item` seam above is stable across all three. Recommend deciding this before building
the adapter, but the `confluence-helpers`-side `ItemSource`/`Scrubber` can be built first.

## Non-goals

- fak never holds `CONFLUENCE_TOKEN`.
- fak never fetches from the wiki directly (it can't — no network reach).
- No raw (unscrubbed) Confluence content ever lands in fak's tree, transcripts, or context.
