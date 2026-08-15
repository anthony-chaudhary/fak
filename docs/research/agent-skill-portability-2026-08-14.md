# Agent skill portability research — 2026-08-14

## Verdict

The gap behind fak commit `0094bae9c8e668a31459c0a614aae2e0b15b97b0` was real: Claude's
canonical `fleet-wave` body changed, while Codex's adaptation lived only in one user's
`$CODEX_HOME`. The repository now uses an explicit, parity-checked rule:

- `.claude/skills/<name>/SKILL.md` is the canonical semantic body;
- `go run ./cmd/fak-project-assets sync --json` generates missing `.agents/skills/<name>/SKILL.md` discovery adapters while preserving a deliberate native Codex adapter such as `fleet-wave`;
- generated Codex adapters link to canonical bodies instead of copying them; `fleet-wave` is a deliberate native adapter because its launcher behavior differs;
- `go run ./cmd/fak-project-assets parity --json` fails on missing/stale generated adapters, duplicates, or unexplained assets. Native adapters are explicit repository artifacts and are preserved by sync.

The repository-tracked Codex adapter introduced by `016503cc0e`, plus the installed full Codex adaptation, can do the same functional `fleet-wave` workflow, including the referenced commit's
`BOUNDED`, `BROAD`, `ISSUE_OWNER`, `LEAF_CHILD`, one-level delegation, independent read-back, and
durable-park behavior. Harness-specific invocation, permissions, worker APIs, and hooks remain
adapters rather than divergent workflow copies.

## Fresh primary-source findings

The research window is 2026-07-17 through 2026-08-14. Sources outside that window are background,
not claimed as fresh findings.

1. **Agent Skills remains the portable package substrate.** At pinned revision
   `agentskills/agentskills@69ef37e9424c0a7ea9dd2293b559e43ec8176379` (latest 2026-08-09), a
   skill is a directory with `SKILL.md`, required `name` and `description`, progressive disclosure,
   and optional resources. The merge of
   [agentskills/agentskills#268](https://github.com/agentskills/agentskills/pull/268) on 2026-08-04
   clarified that `scripts/`, `references/`, and `assets/` are recommendations, not an exhaustive
   directory whitelist. This permits visible harness adapters without pretending they are standard.
2. **Codex's current repository discovery root is `.agents/skills`.** The current
   [Codex skills documentation](https://developers.openai.com/codex/skills), captured 2026-08-14,
   says Codex scans `.agents/skills` from CWD to repository root and supports symlinked skill
   folders. A machine-local `$CODEX_HOME` copy is useful for personal installation but cannot prove
   repository portability.
3. **Claude's project root remains `.claude/skills`.** The current
   [Claude Code skills documentation](https://code.claude.com/docs/en/skills), captured 2026-08-14,
   documents project skills at `.claude/skills/<name>/SKILL.md`. There is therefore no single shared
   project discovery directory across both harnesses today; generated discovery adapters are the
   smallest reliable bridge.
4. **Extensions should be namespaced and fail soft.** Agent Skills commit
   [`3f3bbec8`](https://github.com/agentskills/agentskills/commit/3f3bbec8) on 2026-08-03 clarified
   arbitrary string-keyed metadata and recommended reverse-domain namespacing. OpenAI Codex PR
   [#38467](https://github.com/openai/codex/pull/38467), merged 2026-08-14, added an optional model
   annotation while ignoring unsupported values; PR
   [#38475](https://github.com/openai/codex/pull/38475), also merged 2026-08-14, bounded the
   resulting delegation behavior. The portable contract should therefore stay semantic and small;
   vendor-native controls belong in typed extensions.
5. **Do not copy obsolete vendor-local layouts.** OpenAI Codex PR
   [#38635](https://github.com/openai/codex/pull/38635), merged 2026-08-14, removed that repository's
   local `.codex/skills` examples. Together with the public `.agents/skills` documentation, this is
   direct evidence against establishing `.codex/skills` as fak's project standard.

## Adopted fak portability standard

Use three layers:

1. **Canonical semantics:** one Agent Skills-compatible `SKILL.md` and relative resources when
   harnesses truly share behavior. Avoid absolute home paths and provider-only tool names in this
   semantic body.
2. **Discovery or native adapter:** generate a pointer when one body is sufficient. Keep a deliberate
   native adapter when invocation or launcher behavior materially differs—as `fleet-wave` does for
   Codex seats—and parity-check its presence rather than forcing byte identity.
3. **Native effect adapter:** map required effects, not syntax. Examples are explicit launch intent,
   bounded fan-out, independent witness, deny/stop policy, and memory loading. Implement them with
   Claude hooks, Codex hooks/delegation, or fak/DOS verbs as available. If no equivalent exists,
   record a typed exclusion with rationale rather than silently dropping it.

This yields functional parity without demanding byte identity. `fak-project-assets parity` proves
that every declared project asset is canonical, imported, or explicitly excluded; unit tests prove
that generated adapters load their canonical skill without copying it. The tracked Codex
`fleet-wave` native adapter is separately witnessed by its concrete route and launch commands.
## Reproduce

```powershell
# Generate Codex discovery adapters from canonical project skills.
go run ./cmd/fak-project-assets sync --json

# Read-only parity gate; zero unexplained gaps is required.
go run ./cmd/fak-project-assets parity --json

# The referenced fleet-wave semantics are in the canonical body.
Select-String .claude/skills/fleet-wave/SKILL.md `
  -Pattern 'BOUNDED|BROAD|ISSUE_OWNER|LEAF_CHILD|independent read-back|durable park'

# Codex's checked-in native adapter exposes the same functional obligations.
Select-String .agents/skills/fleet-wave/SKILL.md ` 
  -Pattern 'ISSUE_OWNER|bounded|leaf|independent|codex'
```

## Pinned sources

- Agent Skills repository/specification:
  `agentskills/agentskills@69ef37e9424c0a7ea9dd2293b559e43ec8176379`.
- Agent Skills PR [#268](https://github.com/agentskills/agentskills/pull/268), merged 2026-08-04.
- Agent Skills metadata commit
  [`3f3bbec8`](https://github.com/agentskills/agentskills/commit/3f3bbec8), 2026-08-03.
- OpenAI Codex skills docs, https://developers.openai.com/codex/skills, captured 2026-08-14.
- Claude Code skills docs, https://code.claude.com/docs/en/skills, captured 2026-08-14.
- OpenAI Codex PR [#38467](https://github.com/openai/codex/pull/38467), merged 2026-08-14.
- OpenAI Codex PR [#38475](https://github.com/openai/codex/pull/38475), merged 2026-08-14.
- OpenAI Codex PR [#38635](https://github.com/openai/codex/pull/38635), merged 2026-08-14.
