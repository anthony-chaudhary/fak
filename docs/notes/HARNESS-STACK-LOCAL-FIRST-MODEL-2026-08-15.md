# Harness stacks are typed assemblies, not uniform profile piles — 2026-08-15

**Status:** design contract for the production lock compiler in #6938  
**Scope:** the ordinary local user first; fleet and enterprise cases must extend this model rather than distort the default  
**Decision:** “harness stack” is the useful product metaphor, but its entries are not all the same kind of profile, do not all change at the same rate, and do not all have cardinality one.

## Value frame

- **For:** a developer running an agent from a laptop or workstation, often with a remote model and no local accelerator.
- **Problem:** the current person/company/project/domain/task layer vocabulary suggests five equally populated, single-choice overlays. Typical users do not have that shape. Their host is mostly stable, company and domain configuration may be absent, and tools, skills, policies, endpoints, and fallback models are naturally sets or ordered lists.
- **Today:** selection, typed asset composition, component dependency resolution, and launch locks exist as separate surfaces. The examples prove compatibility, but not a realistic local authoring model.
- **Better because:** separate observed facts, authored intent, available choices, run selection, and the immutable launch lock; then give every dimension an explicit cardinality, merge law, cadence, and authority.
- **Witness:** the local example below is representable without inventing a company profile, pretending one GPU exists, or collapsing a many-tool set into one “tool profile”; the same contract explains when a host re-probe is and is not required.

**Centrality:** Enabling. This makes the core harness-kit path usable without adding another runtime mechanism.

- **P1 managed context:** compile only selected assets and summarized host facts into the lock; do not inject the full catalog or hardware probe into model context.
- **P2 net-true efficiency:** compare with a developer manually maintaining one agent config. Automatic detection and reuse must save more work than the schema asks them to author.
- **P3 bounded adaptation:** discovery proposes; explicit constraints and fail-closed merge laws decide; the lock freezes the result.
- **P4 integrated operations:** launch consumes the lock, records environment fingerprint drift, and points failures back to a dimension and source.

## The core correction

Do not use `profile`, `layer`, `component`, and `lock entry` as synonyms.

| Kind | Question answered | Typical source | Examples |
|---|---|---|---|
| **Inventory** | What is true or installed here? | detected observation | OS/arch, RAM, accelerators, local binaries, reachable runtimes |
| **Profile** | What defaults or constraints does this scope author? | person/org/project configuration | policy floor, preferred provider, project tools |
| **Catalog** | What choices could be assembled? | installed/remote manifests | models, tools, skills, MCP servers, harness adapters |
| **Selection** | What does this run request from those choices? | invocation/task/workload | coding task, chosen harness, enabled tool subset |
| **Lock** | What exact assembly may launch? | deterministic compiler output | digests, versions, endpoints, grants, resolved providers |

A **stack** is the resulting typed assembly across these kinds. It is not a list of interchangeable YAML fragments.

This distinction prevents four category errors:

1. Hardware is observed inventory, not a user preference profile. A user may add constraints such as `gpu: forbidden` or `min_vram: 24GiB`, but those constraints do not manufacture hardware facts.
2. A catalog may contain 40 tools while a run selects 7. Cardinality of available choices is different from cardinality of active selections.
3. A singleton inventory document may contain many devices. “One host snapshot” does not mean “one GPU.”
4. A lock entry is resolved output, not another override layer. Editing it invalidates its digest and requires recompilation.

## Local-first baseline

The zero-enterprise local case should be complete with only facts fak can detect plus one explicit task request:

| Dimension | Ordinary local default | Active cardinality | Expected cadence | Authority |
|---|---|---:|---|---|
| execution target | current local host | exactly 1 | weeks/months | detected inventory |
| host devices | CPU, memory, 0..N accelerators | 0..N inside one snapshot | weeks/months | detected inventory |
| user profile | built-in safe defaults; optional user file | 0..1 authored | months | user |
| organization profile | absent | 0..1 | weeks/months | organization/admin |
| project profile | nearest explicit project manifest, if present | 0..1 | days/weeks | repository owner |
| workload | invocation task plus optional declared workload adapter | exactly 1 task, 0..1 adapter | every run | operator |
| harness/runtime | installed default or explicit selection | exactly 1 | per project/run | operator + compatibility resolver |
| model routes | one primary and optional ordered fallbacks | 1 primary, 0..N fallback | per project/run | operator + router policy |
| tools/MCP servers | compatible selected subset | 0..N | per project/run | project/operator + policy |
| skills/hooks/evaluators | compatible selected subsets | 0..N each | per project/run | project/operator |
| policy | built-in floor plus applicable overlays | 1..N constraints | mixed | system/org/project/operator |
| secrets/credentials | references only; never values in the lock | 0..N refs | days/months | credential store |

Consequences:

- `company` cannot be required. A local user often has no company-managed layer.
- `domain` cannot be required or silently classified. Most tasks need no domain adapter; when one is selected it is ordinary authored intent with advisory evidence, as #6938 requires.
- No GPU is a valid inventory result, not an incomplete profile. A remote model route can still make the stack feasible.
- Hardware should not be re-authored for every task. Cache the observation and bind the lock to a host fingerprint; re-probe on fingerprint change, explicit refresh, relevant device/runtime failure, or an expired observation policy.
- Project and task choices may change while the host snapshot is reused. Compilation must invalidate by input digest, not by pretending every dimension has one shared revision cadence.

## Cardinality is a dimension contract

Each dimension declaration needs at least:

```text
name
kind                 inventory | profile | catalog | selection
active_cardinality   exactly-one | zero-or-one | set | ordered-list
identity_key         stable key for set/list members
merge_law            replace | keyed-union | ordered-append | constraint-meet
conflict_law         refuse | explicit-winner (never implicit for privileges)
source_authority      detector | system | org | project | user | invocation
change_cadence        hint for refresh, never correctness authority
lock_projection       fields retained in the immutable launch lock
```

The production compiler should support only a small set of explicit merge algebras:

### 1. Singleton replacement

Use for an execution target, one selected harness, or one primary model route. More than one active value is an error unless a higher-authority source explicitly replaces a lower-authority default. Provenance for both must remain in the receipt.

### 2. Keyed set union

Use for tools, MCP servers, skills, hooks, evaluators, and credential references. Members merge by stable identity, never by array position. Two definitions of the same identity with incompatible executable digests, schemas, or privilege declarations refuse unless an explicit replacement rule names the winner.

### 3. Ordered append with deduplication

Use only where order is behavioral, such as fallback model routes or hook chains. Preserve authored order and deduplicate by identity. A set must not accidentally become order-sensitive because of file traversal order.

### 4. Constraint meet

Use for policy and resource ceilings. Applicable constraints combine toward the more restrictive result. A lower scope may narrow authority but may not widen a system or organization floor. An unsatisfiable meet refuses before launch.

These laws describe active values. An individual value may itself contain a collection: the one host inventory contains many devices; one policy bundle contains many rules; one MCP server exposes many tools.

## Cadence and invalidation

Cadence is an optimization hint, not a correctness shortcut. Every source is content-addressed in the lock compiler.

| Input change | Reuse allowed | Required action |
|---|---|---|
| task text changes, same declared requirements | host/catalog observations | reselect and recompile run-bound fields |
| project manifest changes | unchanged host inventory | recompose project and descendants |
| tool binary/schema digest changes | unrelated catalog entries | resolve dependents; refuse stale lock |
| provider endpoint health changes | immutable configuration, not health claim | runtime routing may fail over only within locked routes |
| OS/arch/device/driver fingerprint changes | user/org/project profiles | re-probe inventory and re-resolve compatibility |
| observation merely ages past policy | authored sources | refresh the affected inventory probe |
| secret value rotates behind same reference | lock metadata | credential store resolves at launch; never place value in lock |

A hardware fingerprint should cover compatibility-relevant facts, not volatile utilization: OS/arch, device identity and count, driver/runtime ABI, and material memory capacity. Temperature, free VRAM, load, and endpoint health are runtime telemetry, not lock identity.

## Concrete local assembly

A realistic local coding run might have:

```yaml
inventory:
  target: local
  host_snapshot: auto            # Windows/amd64, 32 GiB RAM, no GPU
profiles:
  user: null                     # built-in safe defaults apply
  organization: null
  project: .fak/project.yaml
selection:
  task: "fix the parser and run affected tests"
  workload_adapter: coding
  harness: codex-local
  model_routes:
    primary: openai/gpt-5.3-codex
    fallback: [openai/gpt-5.2-codex]
  tools: [filesystem, git, powershell]
  mcp_servers: []
  skills: [study-repo, verify]
  hooks: [trunk-guard, public-leak-guard]
constraints:
  policies: [fak-default-floor, project-contributor]
  local_gpu: forbidden           # a constraint, not a fake hardware profile
```

The compiler expands catalog identities to exact manifests, composes typed assets, resolves providers and dependencies, takes the meet of policy constraints, and emits one lock. The launch path receives no unresolved `auto`, no implicit filesystem traversal order, and no secret values.

The same user starting a second task should normally reuse the host snapshot, installed catalog, user profile, and project profile. It recompiles the task selection and any affected dependency closure. That is the practical power of the stack: stable work is amortized while volatile intent stays cheap to change.

## What the lock must say

The versioned lock proposed by #6938 should preserve, at minimum:

- schema/compiler version and whole-lock digest;
- every input source URI/path, authority, content digest, and observation timestamp where relevant;
- host fingerprint and the compatibility-relevant facts projected from inventory;
- selected singleton identities and selected keyed/ordered members;
- resolved component/provider versions and artifact digests;
- dependency edges and provider choices;
- effective policy/resource constraints plus evidence that no privilege widened;
- credential reference identities, never credential values;
- advisory classifier/recommendation evidence only when explicitly requested;
- refresh and drift conditions that make launch refuse or request recompilation.

Keep the full catalog, unselected tools, raw probes, volatile telemetry, and task prose out unless a selected component genuinely requires them. A lock is a minimal executable explanation, not a workstation backup.

## Migration from the current five layers

The existing `person/company/project/domain/task` stack is still useful as an **authored-scope precedence model**, but it is only one part of the assembly:

| Existing term | Local-first interpretation |
|---|---|
| person | optional user profile |
| company | optional organization profile |
| project | optional project profile |
| domain | optional workload adapter, never inferred by default |
| task | required invocation selection, not a persistent profile |

It must not own inventories, catalogs, or resolved lock output. Existing manifests can migrate by wrapping their typed assets as profile sources and declaring each asset dimension’s cardinality and merge law. Unknown or undeclared dimensions refuse rather than falling back to generic “later layer wins.”

## Product defaults and non-goals

1. `fak harness compile` (name illustrative until #6938 lands) should work with no company file and no GPU.
2. Auto-detection may fill inventory; it may not infer legal/domain adequacy, grant tools, relax policy, or select a paid route without an explicit default contract.
3. Simple dimensions remain simple. Do not force a one-item list where the semantic contract is singleton.
4. Many-valued dimensions stay many-valued. Do not create numbered pseudo-layers such as `tool1`, `tool2`, or one profile per skill.
5. A settings UI can later render the dimension declarations; it must not become the source of merge semantics.
6. Fleet support adds an inventory/catalog of candidate targets and a placement selection. It does not change a launched lock’s rule that one execution target is bound for each execution unit.

## Acceptance checks for the compiler issue

This model sharpens #6938’s clean-room witness:

- compile and launch the concrete no-GPU local fixture above with absent user/org profiles;
- select several tools/skills/hooks by keyed set while preserving one harness and one primary route;
- prove a project-only change reuses the host observation but changes the lock digest;
- prove a relevant driver/device fingerprint change invalidates compatibility without requiring edits to project/task profiles;
- reject duplicate keyed members with conflicting digest or privilege declarations;
- reject a lower-scope policy widening;
- prove explicit ordered model fallback survives compilation while unordered sets are deterministic across file enumeration order;
- prove no classifier call occurs in the default path.

## Relationship to existing work

- #6938 owns the production compiler and immutable lock lifecycle.
- #6886 and `internal/stackresolve` own hard dependency/compatibility feasibility.
- #6887 owns evidence-backed workload fitness; fitness may recommend a selection but cannot rewrite inventory or policy authority.
- `internal/harnesscompose` already supplies typed asset composition; its generic layer precedence should be narrowed by the per-dimension merge laws above when folded into the compiler.
- `internal/harnessselect` remains an authoring aid for scoped profile selection, not the complete stack model.

The harness stack remains powerful precisely because it can combine unlike things. Making those differences explicit is what keeps the concept realistic rather than turning “stack” into another name for a bag of overrides.
