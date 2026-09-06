---
title: "Declarative Agent Descriptors"
description: "Specification and operational guide for declarative markdown agent descriptors (.fak/agents/*.md and .agents/*.md) with frontmatter capability envelopes and Prompt MMU overlay integration."
---

# Declarative Agent Descriptors

Declarative Agent Descriptors are file-based, version-controlled specifications defining an agent's persona, operational model, turn budget, and capability envelope. They decouple agent definition from runtime harness code, allowing repositories to specify specialized primary and subagent roles directly in markdown.

## Discovery and Locations

`fak` scans two canonical workspace locations for agent descriptor files:

1. `.fak/agents/*.md` (primary repository location, takes precedence)
2. `.agents/*.md` (shared convention location)

Files named `README.md` or `SKILL.md` are ignored during agent discovery.

## Descriptor Format

An agent descriptor is a Markdown document with YAML frontmatter bounded by `---` delimiters. The frontmatter defines metadata and security capabilities, while the document body contains persona instructions.

### Example Descriptor

```markdown
---
name: explore
description: Read-only codebase exploration and discovery subagent
mode: subagent
model: tier1
variant: default
max_turns: 15
capabilities:
  tools:
    - glob
    - grep
    - read
  paths:
    - internal/**
    - cmd/**
  allow_mutation: false
---

# Role: Codebase Explorer

You are an expert codebase exploration subagent. Your objective is to discover code patterns, trace symbols, and report concise findings without modifying any files.
```

### Frontmatter Schema

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string | Filename without `.md` | Unique identifier for the agent descriptor |
| `description` | string | `""` | Human- and model-readable summary of the agent's function |
| `mode` | string | `"subagent"` | Operational mode: `primary` (orchestrator) or `subagent` (delegated worker) |
| `model` | string | `"tier1"` | Execution tier alias (`tier1`, `tier2`, `tier3`) or specific model identifier |
| `variant` | string | `"default"` | Reasoning or style intensity: `default`, `high`, or `adaptive` |
| `max_turns` | int | `10` | Maximum turn budget allocated for a single execution |
| `capabilities.tools` | []string | `[]` | Explicit tool whitelist granted to the agent; empty denotes unrestricted host default |
| `capabilities.paths` | []string | `[]` | Lexical subtree paths or globs the agent is permitted to read or touch |
| `capabilities.allow_mutation` | bool | `false` | Gate for file modification and state changes; `false` enforces read-only operation |

Alternative syntax: capability fields can also be specified inline (e.g. `tools: [glob, read]`) or at the top level of the frontmatter.

## Security: Monotonic Capability Narrowing

When a parent coordinator dispatches a subagent using a descriptor, authority cannot expand. `internal/agent` enforces monotonic capability narrowing:

1. **Mutation Invariant:** A child agent can only mutate if the parent has mutation authority (`child.AllowMutation = parent.AllowMutation && requested.AllowMutation`). If a parent is read-only, all child subagents are strictly read-only.
2. **Tool Invariant:** If a parent specifies an authorized tool set, the child's tools are intersected with the parent's tools. Any requested tool outside the parent's set is denied (`ValidateChildCapabilities` rejects with `ErrAuthorityWidened`).
3. **Path Invariant:** Child path scopes must fall strictly within the parent's lexical scope. Requests for paths outside the parent's boundaries are pruned or rejected.
4. **Turn Budget Invariant:** Child turn budgets are bounded by the parent's remaining or allocated budget.

## Prompt MMU Integration

The descriptor's markdown body represents the persona system prompt. It integrates directly with `internal/syspromptmmu`:

- **Cache Stability:** The persona prompt is structured as a `TierOverlay` edit (`EditAdd`) and appended after the cache breakpoint, ensuring the resident spine prefix remains byte-identical and cache-stable (`CacheStable() == true`).
- **Prompt Rendering:** `FormatPrompt()` combines the metadata header, capability bounds, and instruction body into a structured overlay block.

## CLI Usage

List discovered declarative agent descriptors in the current workspace:

```bash
# Tabular overview
fak agents list

# Custom directory search
fak agents list --dir /path/to/repo

# Machine-readable JSON output
fak agents list --json
```

Example tabular output:

```
NAME       MODE      MODEL    VARIANT   TURNS  MUTATION  TOOLS             PATH
explore    subagent  tier1    default   15     false     glob,grep,read    .fak/agents/explore.md
reviewer   subagent  tier2    high      20     false     read,git          .agents/reviewer.md
writer     subagent  tier2    default   10     true      read,edit,write   .fak/agents/writer.md
```
