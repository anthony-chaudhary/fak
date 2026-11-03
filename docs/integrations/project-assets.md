---
title: "Shared project assets"
description: ".claude/project-assets.json is the canonical registry for portable project skills, project memories, and reusable goal prompts."
---
# Shared project assets

`.claude/project-assets.json` is the canonical registry for portable project skills,
project memories, and reusable goal prompts. Claude continues to read the canonical
`.claude` paths. Codex, OpenCode, and pi discover generated **one-hop mirrors** under
`.agents/skills` (OpenCode configured in `opencode.json` via
`skills.paths: [".agents/skills"]`; pi passed explicitly via `fak pi --skill`): each
mirror is a regular-file, full-body copy of one canonical
`.claude/skills/<name>/SKILL.md` with provenance metadata injected in the frontmatter
(`generated-by: fak project-assets sync` plus a `canonical:` key naming the source).
Because mirrors are plain files, Windows checkouts need no symlinks; `.claude/skills`
remains the single semantic source, and regeneration is idempotent and parity-gated
(`zero_unexplained_gaps: true`).

**Why one-hop.** pi and OpenCode already page skill bodies on demand: the resident
index carries only name, description, and path. The old pointer layer was therefore
pure overhead, an extra read per hot-skill use and a resident index truncated at
220 characters mid-sentence. Mirrors restore full spec-length (up to 1024 character)
descriptions in the resident index and cut every skill load from two reads to one.
Descriptions above the 1024-rune Agent Skills limit now fail `sync` instead of being
silently truncated.

Regenerate mirrors after adding or renaming a portable skill:

```text
go run ./cmd/fak-project-assets sync --json
```

Check parity in a clean archive or checkout:

```text
go run ./cmd/fak-project-assets parity --json
```

The receipt reports canonical, imported, explicitly excluded, duplicate, and stale
assets per harness. `zero_unexplained_gaps: true` is the acceptance gate. A new file
that matches neither an include entry nor an exclusion with a reason fails closed.

Codex and fak-native startup should orient with the manifest and run:

```text
fak memory recall --intent "<task>" --json
```

That keeps project memory behind the existing freshness/dedup recall gate instead of
copying prose into `AGENTS.md`. Fak-native reads the same manifest paths for reusable
goal prompts. Issue-numbered prompt fuel is explicitly excluded as ephemeral; reusable
templates are imported once from their canonical paths.

## Native harness boundary

Project discovery is the portable layer; native harness behavior is an adapter layer. Classify an
asset before translating it:

| Layer | Portable contract | Harness adapter examples |
|---|---|---|
| Skill semantics | the canonical `SKILL.md` workflow and relative references | `$skill` versus slash invocation; UI metadata |
| Agent control | bounded ownership, explicit launch, independent witness, durable park | Codex subagent/model delegation; Claude agent/team APIs |
| Policy lifecycle | allow/deny effects and typed stop conditions | Claude hook events; Codex hooks/config; `fak guard` |
| Context/memory | named content and when it must be loaded | `CLAUDE.md` import; `AGENTS.md`; fak context-MMU |
| Goal fuel | acceptance criteria and witness requirements | harness session/goal prompt format |

When a native mechanism has no peer equivalent, preserve the effect through a fak/DOS verb when
possible. Otherwise register an explicit exclusion with rationale; never silently claim parity from
file presence. Unsupported vendor metadata is allowed to remain namespaced and fail-soft, but it is
not part of the portable semantic contract.

## Functional parity witness

`fleet-wave` demonstrates the mirror contract end to end. Codex and OpenCode discover
`.agents/skills/fleet-wave/SKILL.md`, a generated one-hop mirror whose body is the
canonical workflow: commit `0094bae9c8e668a31459c0a614aae2e0b15b97b0`'s `BOUNDED` /
`BROAD` / `ISSUE_OWNER` / `LEAF_CHILD`, one-level fan-out, independent read-back, and
durable-park effects, with the Codex-native seat routing and launch commands that
`016503cc0e` routed into the canonical body. One-hop mirrors are the default for every
portable skill; a file under `.agents/skills` that lacks the `generated-by` marker is
treated as a deliberate native artifact and preserved by sync rather than overwritten.

This is stronger than periodic blind copying: every skill is either a canonical body
with its generated one-hop mirror or an explicit exclusion, and the parity receipt
makes unexplained gaps fail.