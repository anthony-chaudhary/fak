# Tool registration, model visibility, and deterministic skill programs

**Date:** 2026-08-14  
**Status:** working spine (`internal/toolcatalog`, `fak skill compile`)

## Value frame

- **For:** operators adding a custom capability without making every model learn a private command dialect.
- **Problem:** “registered”, “allowed”, “shown to the model”, and “the model already knows how to call it” are different facts, but existing systems commonly collapse them.
- **Today:** fak has lifecycle plugins (`internal/toolplugin`), skill paging, and content-addressed tool-schema pages, but no authoritative bridge from a skill-owned executable declaration to the exact provider tool set.
- **Better because:** one explicit, deterministic program block produces a host registration and a separately selected, content-addressed model view; prose never silently becomes code.
- **Witness:** `go test ./internal/toolcatalog` plus the `runSkillCompile` captured-JSON test.

Centrality is **Enabling**. P1 managed context: only selected schemas enter the model view. P2 net-true efficiency: dialect aliases reuse model priors instead of teaching duplicate private syntax, but no performance claim is made yet. P3 bounded adaptation: conversion accepts a versioned JSON block and refuses prose inference. P4 integrated operations: registration and model-view digests are inspectable CLI output.

## The four sets that must not be conflated

For a request, use these sets in order:

1. **Installed** — executable artifacts the host can resolve by pinned identity and digest.
2. **Allowed** — installed tools surviving tenant, policy, capability, and environment checks.
3. **Exposed** — allowed tools deliberately placed in this request's provider-native tool catalog.
4. **Selected** — an exposed tool the model names in a tool call.

The core invariant is `selected ⊆ exposed ⊆ allowed ⊆ installed`. A custom registration changes only the first set. This matters both for security and cognition: an installed command hidden from the request is unavailable to the model even if a prompt happens to mention it; conversely, a schema sent to the model is not executable unless its canonical registration resolves.

`internal/toolplugin` is intentionally not this registry. It hosts monotone interceptors at tool-call lifecycle stages. A lifecycle plugin can adjudicate or observe a call; it does not define the callable API presented to a model.

## How the model knows

A model can know a capability through several channels, with different authority:

| Channel | What it supplies | Authority |
|---|---|---|
| Provider `tools` schema in the current request | callable name, description, argument grammar | authoritative for availability |
| Resident skill/card text | when and why to use a capability; orchestration | advisory; not registration |
| Harness built-ins and model training priors | familiar names such as shell, Git, `gh`, popular SDKs | compatibility prior; not evidence the tool is present |
| Tool-result/error history | runtime feedback and correction | observational; bounded to the session |

Therefore the exact **exposed** catalog must be a first-class, digest-addressed request artifact. Skills should point at canonical tool identities; they should not be the sole way a tool becomes discoverable. When the catalog is paged, a small resident index must still advertise the capability name/intent and fault path, while the provider receives only the working-set schemas.

Descriptions should explain semantic affordance and constraints, not re-teach common syntax. For a model already trained on `shell_command`, `git`, or an OpenAI-style function tool, a dialect alias can preserve that prior while routing to a canonical fak registration. Alias translation is allowed only at the name boundary; arguments and results need explicit versioned adapters if their shape differs. Ambiguous aliases fail closed.

## Deterministic skill-to-program conversion

A general natural-language skill cannot be deterministically converted into a safe program. Control flow, exception policy, side effects, credentials, idempotency, and success criteria are underspecified. LLM-generated conversion can propose an artifact, but it is not deterministic compilation and must not gain execution authority automatically.

The spine instead recognizes exactly one fenced `fak-program` JSON block in `SKILL.md`:

```fak-program
{"version":"fak.skill-program/v1","name":"repo_search","input_schema":{"type":"object"},"executor":{"argv":["fak","code","search","--json"]},"aliases":{"codex":"functions.shell_command"}}
```

Compilation:

- parses only this versioned block; unknown fields and unsupported versions fail;
- defaults name/description only from bounded frontmatter, never from procedural prose;
- requires an object input schema and an argv executor (no shell-string interpolation);
- canonicalizes typed JSON and binds it with the stable source identity into a digest;
- keeps executor argv host-side and emits a separate model view with no command leakage;
- requires explicit `--expose <canonical-name>` selection;
- applies a harness/model dialect alias and rejects collisions.

Example:

```text
fak skill compile .claude/skills/repo-search/SKILL.md --json
# registered, but model_view.tools is empty and the omission says NOT_SELECTED

fak skill compile .claude/skills/repo-search/SKILL.md \
  --expose repo_search --dialect codex --json
# model_view contains functions.shell_command, canonical_name repo_search, and catalog digest
```

## Go command intersection

Go commands are a strong execution substrate when they are compiled, pinned, and invoked as argv, but CLI discovery is not automatically a good model API:

- `flag --help` text is not a stable machine schema.
- positional args, environment, stdin, and exit codes must be represented explicitly before automatic wrapping.
- a monolithic `fak` binary provides stable deployment and policy mediation, while canonical registrations preserve per-tool identity.
- deterministic wrappers should target first-class `--json` verbs and map JSON input to a declared argv/environment/stdin contract without a shell.
- popular command names can be retained as dialect aliases when that genuinely matches semantics; pretending a private command is `git` or `shell_command` when behavior differs wastes model priors and is unsafe.

The eventual request adapter should join the exposed snapshot into the existing ctxmmu content-addressed tool pages and provider request envelope. Execution should reverse-map the visible alias to the canonical registration, verify the registration digest from that request snapshot, then pass the canonical call through policy and `toolplugin` lifecycle stages.

## Non-goals of the spine

This does not yet execute custom binaries, infer workflows from arbitrary prose, introspect arbitrary Cobra/flag commands, or mutate provider requests. Those are follow-on integrations; the shipped boundary makes their invariants testable first.
