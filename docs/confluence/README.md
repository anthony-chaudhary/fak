---
title: "Confluence publishing (durable process)"
description: "How standalone Confluence page sources live in the fak repo as .confluence.xhtml files and get pushed to Confluence as a durable, repeatable process."
---

# Confluence publishing (durable process)

> **Inbound (Confluence → fak) is a separate, not-yet-built seam** — see
> [DESIGN-items-to-fak-seam.md](./DESIGN-items-to-fak-seam.md). This page covers the existing
> **outbound** publish flow (fak → Confluence).

Standalone Confluence page sources live here as `*.confluence.xhtml`. They are pushed to the
internal Confluence Data Center wiki with the **`confluence-helpers`** tool (a sibling repo at
`../confluence-helpers`, console script `confluence`). This directory is the tracked home for those
page sources — and **this repo is public**, so every byte committed here is published. Treat these
files as external copy and scrub *before committing*, not before pushing: no credentials, no
internal hostnames or addresses, no personal names, no unpublished internal figures. (The
`confluence-meta` `space`/`page_id` values are part of the tracked contract below and belong here.)

## Why a bridge

The Confluence host (`<internal-confluence-host>:8090`, from `CONFLUENCE_URL`) is only reachable
from an **internal bridge machine** — not from a developer
laptop or the Windows agent box. So publishing is a relay:

```
author here  ──(git)──►  fak repo (docs/confluence/*.xhtml)
                              │
                              ▼  payload / instruction over the sanctioned Slack bridge
                          the internal bridge bridge  ──(confluence push)──►  Confluence
                              ▲
                              └──(confluence pull / status)──  read-back for verification
```

## The tool's contract (confluence-helpers)

- Auth: `Authorization: Bearer $CONFLUENCE_TOKEN` (required). Base URL: `$CONFLUENCE_URL`
  (default `http://<internal-confluence-host>:8090/`).
- Target is read from a `<!-- confluence-meta -->` block **inside** each `.xhtml` file — NOT from
  CLI flags:
  ```html
  <!-- confluence-meta
  title: <exact page title, matches the <h1>>
  space: MPL
  parent_id: <optional parent page id>
  page_id: <auto-stamped after first push>
  labels: comma, separated
  -->
  ```
- `confluence push <file>` is **create-or-update & idempotent**: it matches by `page_id` if present,
  else by `title` within `space`, then stamps `page_id` back into the file so later pushes update in
  place. `--dry-run` shows the resolved target without writing.
- `confluence pull <id|title>` / `confluence status <file>` read back for verification.

## Benchmark artifact publication reruns

For the witnessed archive → analysis → attachments → page → independent-read-back workflow, including
binary-transfer failures, collision-safe attachment manifests, AMOProf evidence boundaries, and the
narrow scrubber-unavailable exception protocol, see
[`CONFLUENCE-AMOPROF-PUBLICATION-RERUN-2026-08-12.md`](../notes/CONFLUENCE-AMOPROF-PUBLICATION-RERUN-2026-08-12.md).
## Runbook: publish a page

Run these **on the bridge machine** (the internal bridge), which has `$CONFLUENCE_TOKEN` available and network
reach to the wiki:

```bash
# 0. token is provided out-of-band on the bridge (see "Credential" below) — never echoed here
# 1. install the tool once:   cd ../confluence-helpers && python install.py
# 2. dry run:
confluence push docs/confluence/<page>.confluence.xhtml --dry-run
# 3. real push (create-or-update); capture the printed URL + stamped page_id:
confluence push docs/confluence/<page>.confluence.xhtml
# 4. verify:
confluence status docs/confluence/<page>.confluence.xhtml
# 5. commit the page_id-stamped file back so future pushes update in place:
git add docs/confluence/<page>.confluence.xhtml && git commit -m "docs(confluence): stamp page_id for <page>"
```

## Credential (CONFLUENCE_TOKEN)

- The PAT is a **secret**: it is never committed and never written by an automated agent. Keep the
  real value in a gitignored `.env.confluence.local` at the repo root (already covered by the
  `.env.*` gitignore rule) **on the bridge machine**, or export `CONFLUENCE_TOKEN` in that machine's
  environment / secret store.
- `docs/confluence/confluence.env.template` (placeholder only) documents the expected keys; copy it
  to `.env.confluence.local` on the bridge and fill in the real token.
- Rotate the token if it is ever exposed in a log, transcript, or shared channel.

## Pages tracked here

| File | Title | Space |
|------|-------|-------|
| `livecodebench-kvcache-status.confluence.xhtml` | Live Coding-Benchmark Runs & Token Usage — KV Cache Value Status | MPL |
