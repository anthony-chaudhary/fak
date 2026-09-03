---
title: "Shared project assets"
description: ".claude/project-assets.json is the canonical registry for portable project skills, project memories, and reusable goal prompts."
---
# Shared project assets

`.claude/project-assets.json` is the canonical registry for portable project skills,
project memories, and reusable goal prompts. Claude continues to read the canonical
`.claude` paths. Codex and OpenCode discover generated, regular-file adapters under
`.agents/skills` (configured in `opencode.json` via `skills.paths: [".agents/skills"]`);
each adapter points to one canonical `SKILL.md` and does not copy its
maintained body, so the seam also works in Windows checkouts without symlinks.

Generate adapters after adding or renaming a portable skill:

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

`fleet-wave` demonstrates the simple-first rule and the native-adapter exception. Codex discovers
`.agents/skills/fleet-wave/SKILL.md`; the tracked adapter added in `016503cc0e` carries Codex-native
seat routing and launch commands while preserving commit
`0094bae9c8e668a31459c0a614aae2e0b15b97b0`'s `BOUNDED` / `BROAD` / `ISSUE_OWNER` /
`LEAF_CHILD`, one-level fan-out, independent read-back, and durable-park effects. Generated pointer
adapters remain the default for skills that need only discovery translation.

This is stronger than periodic blind copying: every skill is either a canonical body plus generated
pointer, a deliberate tracked native adapter, or an explicit exclusion, and the parity receipt makes
unexplained gaps fail.