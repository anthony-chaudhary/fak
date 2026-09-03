# OpenCode Study — 2026-09-02

**Source:** https://github.com/anomalyco/opencode  
**Pinned revision:** 4eb29a64f0054672950acf789f2b09487ebfbb20 (dev branch)  
**Study type:** Exhaustive (deep, fan-out)  
**Operator:** fak agent via study-repo skill

---

## Acquisition & Pin

```bash
git clone --depth 1 --filter=blob:none https://github.com/anomalyco/opencode.git C:\work\fak\_scratch\study-opencode\opencode
git -C C:\work\fak\_scratch\study-opencode\opencode rev-parse HEAD
# 4eb29a64f0054672950acf789f2b09487ebfbb20
```

---

## Evidence Surface Coverage (Fan-out)

| Subsystem / Directory | Coverage | Notes |
|---|---|---|
| `packages/core/src/` | ✅ Read | Session, SystemContext, ToolRegistry, Permission, Plugin, Catalog, Agent, Model, Provider, Project, Location, Event, Database, Filesystem, Policy, Skill, Integration, Reference, Command, Shell, Git, Ripgrep, Snapshot, State, Account, Credential, OAuth, Installation, Image, Control Plane, Flag, Observability, Pty, V1 compat |
| `packages/core/src/session/runner/` | ✅ Read | LLM orchestration, model resolution, publishing, message conversion, max-steps |
| `packages/core/src/system-context/` | ✅ Read | Composable context sources, snapshots, reconciliation, initialization, replacement |
| `packages/core/src/tool/` | ✅ Read | Registry, built-ins (bash, edit, glob, grep, read, write, webfetch, websearch, apply_patch, question, todo, skill), application tools, output bounding |
| `packages/core/src/permission/` | ✅ Read | Ruleset evaluation, ask/assert/reply, saved rules, agent-scoped permissions |
| `packages/core/src/plugin/` | ✅ Read | Plugin lifecycle, host, scoped loading, wait/remove, integration with Agent, AISDK, Catalog, Command, Integration, Reference, Skill |
| `packages/core/src/catalog.ts` | ✅ Read | Provider/model catalog, availability, default/small model selection, policy gating |
| `packages/opencode/src/` | ✅ Read | CLI entry, commands (run, serve, agent, models, providers, github, mcp, plugin, session, etc.), TUI runtime |
| `packages/opencode/src/cli/cmd/run.ts` | ✅ Read | Three modes: non-interactive, interactive local (--mini), interactive attach; session management, event streaming, permission handling |
| `packages/opencode/src/agent/` | ✅ Read | Agent definitions, built-ins (build, plan), subagent support |
| `packages/opencode/src/mcp/` | ✅ Read | MCP server/client integration |
| `packages/opencode/src/github/` | ✅ Read | GitHub Actions integration, issue/PR automation |
| `.github/workflows/` | ✅ Read | CI: unit tests (Linux/Windows), e2e (Playwright), typecheck, publish, pr-standards, review, triage |
| `AGENTS.md` (root) | ✅ Read | Contributor conventions, branch naming, commit style, Effect rules, module shape |
| `CONTEXT.md` | ✅ Read | Session runtime vocabulary: System Context, Session History, Context Source, Context Epoch, Baseline, Snapshot, Admitted Prompt, Prompt Promotion, Provider Turn, Session Drain, Managed Tool Output, Model Request Options, Generation Controls, Native Continuation Metadata, PTY Environment, OpenCode Client, SDK Contract IR, Embedded OpenCode, Page |
| `CONTRIBUTING.md` | ✅ Read | Contribution guidelines |
| `packages/sdk/`, `packages/client/`, `packages/protocol/`, `packages/schema/` | 📋 Noted | SDK generation from HttpApi, Protocol composition, Schema definitions — relevant for client contract architecture |

**Completeness critic:** All load-bearing subsystems opened. The monorepo structure (packages/*) was fully mapped. Core runtime (session, system-context, tool, permission, plugin, catalog) and CLI entry points were read in depth. SDK/Protocol/Schema packages noted for client contract architecture but not exhaustively read — they are interface definitions, not implementation. No material subsystem left unopened.

---

## Candidate Borrow Matrix

| # | Technique (one line) | Source `path:line@sha` | Axis | Fak diff on axis + their-worldview reason | Witness (PRESENT/PARTIAL/ABSENT/DIVERGENT) | Inspire/Integrate | Filed Issue |
|---|---|---|---|---|---|---|---|
| 1 | **Composable System Context with typed sources, snapshots, and reconciliation** | `packages/core/src/system-context/index.ts:32-39` `packages/core/src/system-context/index.ts:198-280` | Context freshness & incremental update | Fak has `ctxmmu` but not a composable, typed source registry with baseline/update/removed renderers and snapshot reconciliation. OpenCode's users (agent sessions needing live context like date, cwd, skills, project instructions) drove this — they need *lazy, sampled-at-safe-boundary* updates without async pushes. Fak's ctxmmu is lower-level memory management; this is a higher-level *context composition* layer. | **PARTIAL** — `fak capabilities "composable system context typed sources snapshot reconciliation"` → context-reuse, portable-session exist but lack Source/Registry/Snapshot composition pattern | **ADAPT** — port the Source/Registry/Snapshot pattern into fak's ctxmmu or a new layer above it | #10701 |
| 2 | **Context Epoch & Baseline System Context for provider cache prefix reuse** | `packages/core/src/system-context/index.ts:26-30` `packages/core/src/system-context/index.ts:130-133` `CONTEXT.md:26-32` | Provider cache efficiency | Fak's gateway has prompt cache awareness but no explicit *Context Epoch* abstraction that isolates baseline rendering and enables compaction-driven epoch turnover. OpenCode's users (long-running sessions with evolving context) drove this — they need the provider's prefix cache to survive across turns until compaction forces a new baseline. Fak's native inference goal explicitly wants cache reuse; this is a direct operationalization. | **PARTIAL** — `fak capabilities "context epoch baseline system context provider cache prefix"` → context-reuse has cache awareness but no epoch abstraction | **ADAPT** — add Context Epoch to gateway's cache key computation | #10702 |
| 3 | **Session Drain = process-local coordination, not durable entity** | `CONTEXT.md:51-53` `CONTEXT.md:104` `packages/core/src/session/execution.ts:9-18` | Execution ownership & recovery | Fak's engine has session-like execution but OpenCode explicitly *rejects* durable Session Drain identity — recovery reasons from prompts, history, provider attempts, tool state. Their users (agents that crash/restart) need recovery without invented execution boundaries. Fak's engine currently has more durable execution tracking; this is a philosophical divergence worth understanding. | **DIVERGENT** — fak chooses durable execution identity for witness/replay; OpenCode chooses process-local only for simplicity. Tradeoff: fak sells audited replay; OpenCode optimizes for crash-recovery simplicity. | **WORLDVIEW-FINDING** — record the divergence; no code borrow | N/A (discussion) |
| 4 | **Admitted Prompt + Promotable Delivery Modes (steer/queue)** | `CONTEXT.md:42-47` `CONTEXT.md:100-103` `packages/core/src/session.ts:147-153` `packages/core/src/session/input.ts` (referenced) | Input admission & steering UX | Fak's agent loop admits prompts directly; OpenCode has *Admitted Prompt* (durable inbox) + *steer* (promote at next safe boundary while drain continues) + *queue* (promote when idle). Their users (interactive agents) need steering without interrupting provider turns. Fak's agent could benefit from this for interactive steer UX. | **ABSENT** — `fak capabilities "admitted prompt steer queue delivery mode"` → session-control has steer but no admitted-prompt inbox or queue delivery mode | **ADAPT** — add admitted inbox + delivery modes to fak agent prompt handling | #10703 |
| 5 | **Mid-Conversation System Message for context changes** | `CONTEXT.md:22-25` `CONTEXT.md:92-99` `packages/core/src/system-context/index.ts:218-280` | Context change communication to model | Fak sends context updates via raw system prompt replacement; OpenCode emits *Mid-Conversation System Message* (chronological instruction) that lowers to provider's native role or wrapped fallback. Their users need *auditable, chronological* context changes in transcript. Fak's gateway could use this for cleaner context updates. | **ABSENT** — `fak capabilities "mid-conversation system message context change"` → no capability for chronological context change messages | **ADAPT** — add Mid-Conversation System Message emission to gateway | #10704 |
| 6 | **Tool Registry with generic output bounding + managed output files** | `packages/core/src/tool/registry.ts:50-82` `packages/core/src/tool/registry.ts:106-122` `CONTEXT.md:189-199` | Tool output size control | Fak's tool calls have no generic output bounding; OpenCode bounds *textual* output at settlement (lines/bytes), spills complete text to managed temp files, keeps bounded preview in history. Their users (agents with large tool outputs) need token pressure relief without losing data. Fak's tool calls can explode context. | **ABSENT** — `fak capabilities "tool output bounding managed spill file"` → context-compression exists but no generic output bounding at settlement with managed spill files | **ADAPT** — add generic output bounding + managed spill to fak tool settlement | #10705 |
| 7 | **Permission Ruleset with wildcard matching + saved "always" rules** | `packages/core/src/permission.ts:76-86` `packages/core/src/permission.ts:147-162` `packages/core/src/permission.ts:250-283` | Permission UX & persistence | Fak's policy is manifest-based (static); OpenCode has dynamic Ruleset with wildcard action/resource, session/agent-scoped evaluation, saved "always" rules persisted per-project. Their users (interactive agents needing per-session permission learning) drove this. Fak's policy could learn from the *saved rules* UX. | **PARTIAL** — `fak capabilities "policy manifest capability floor"` → capability-floor exists but static manifest; `fak capabilities "permission ruleset wildcard saved rules persistence"` → no results | **ADAPT** — add dynamic ruleset + saved rules to fak policy | #10706 |
| 8 | **Plugin lifecycle with scoped loading, wait/remove, KeyedMutex** | `packages/core/src/plugin.ts:31-143` `packages/core/src/plugin/host.ts` (referenced) | Plugin isolation & hot-reload | Fak has no plugin system; OpenCode has scoped plugin loading (forked Scope per plugin), KeyedMutex for load serialization, wait() for dependent code, host with service exposure. Their users (extensible agent) need safe hot-reload. Fak's leaf extension mechanism (`fak new-leaf`) is static compile-time; this is runtime plugin dynamics. | **ABSENT** — `fak capabilities "plugin lifecycle scoped loading hot reload"` → no plugin lifecycle capability | **INSPIRE** — if fak ever needs runtime plugins, this is the model | N/A (architectural divergence) |
| 9 | **Catalog with provider/model availability, default/small selection, policy gating** | `packages/core/src/catalog.ts:47-60` `packages/core/src/catalog.ts:175-286` | Model routing & cost-aware defaults | Fak's model selection is manual; OpenCode has Catalog with availability (API key/integration), default model (newest available), small model (cost/age/small-name heuristic), policy gating (provider.use deny). Their users (multi-provider, cost-sensitive) need smart defaults. Fak's gateway/model routing could use this. | **PARTIAL** — `fak capabilities "model catalog availability default small selection"` → model-routing exists but no availability/default/small selection logic with policy gating | **ADAPT** — port Catalog's availability/default/small selection to fak model routing | #10707 |
| 10 | **Embedded OpenCode = in-process HttpApi host with same client** | `CONTEXT.md:73-83` `CONTEXT.md:139-149` `packages/opencode/src/server/` (implied) | Local-first SDK & testing | Fak has no embedded in-process mode; OpenCode's SDK runs Server's HttpRouter in-memory, same client capabilities, scoped lifecycle. Their users (tests, local tools, IDE integrations) need zero-network OpenCode. Fak's gateway could expose embedded mode for testing/integration. | **ABSENT** — `fak capabilities "embedded in-process http api host"` → no embedded in-process capability | **ADAPT** — add embedded in-process mode to fak gateway | #10708 |
| 11 | **SDK Contract IR → Promise + Effect emitters from single HttpApi** | `CONTEXT.md:77-79` `CONTEXT.md:145-160` | Client SDK generation | Fak has no generated client SDK; OpenCode compiles HttpApi → IR → Promise/Effect emitters independently, preserving transport metadata, brands, schema transforms. Their users (TypeScript consumers, Effect users) need both ergonomics. Fak's integrations could use generated clients. | **ABSENT** — fak has no client SDK generation capability | **INSPIRE** — if fak adds client SDKs, this IR pattern is the model | N/A (future) |
| 12 | **Session event streams: durable (sessions.events) + live (events.subscribe)** | `CONTEXT.md:165-170` | Event streaming & replay | Fak has no public event streaming; OpenCode has durable Session event stream (replay from sequence) + live instance stream (no replay, includes lifecycle). Their users (dashboards, monitors, GitHub Actions) need both. Fak's guard/journal has decision logs but no public event API. | **ABSENT** — `fak capabilities "event stream durable live session replay"` → no public event streaming capability | **ADAPT** — add durable + live event streams to fak guard/gateway | #10711 |
| 13 | **Session list/message pagination with opaque cursors** | `CONTEXT.md:171-175` | API pagination design | Fak has no pagination; OpenCode uses opaque branded cursors (continuation state), fixed page size/order/scope per cursor. Their users (CLI, UI, SDK consumers) need stable pagination. Fak's future APIs should adopt this. | **ABSENT** — fak has no paginated APIs | **INSPIRE** — adopt opaque cursor pagination for any future fak list APIs | N/A (future) |
| 14 | **Skill Discovery: remote index.json → cached skills with versioning, atomic staging** | `packages/core/src/skill/discovery.ts:55-207` | Skill distribution & update | Fak's skills are local filesystem only; OpenCode discovers remote skill indexes, caches with version file, atomic staging (tmp → rename) with backup/rollback. Their users (shared skill repos, auto-update) need safe distributed skills. Fak's skill system could learn the distribution model. | **PARTIAL** — `fak capabilities "skill discovery remote index versioning atomic update"` → no results; fak has local skills only | **ADAPT** — add remote skill discovery + atomic update to fak skill system | #10712 |
| 15 | **Three-mode CLI: non-interactive, interactive local (--mini), interactive attach** | `packages/opencode/src/cli/cmd/run.ts:4-15` `packages/opencode/src/cli/cmd/run.ts:263-964` | CLI UX flexibility | Fak has single CLI mode; OpenCode supports one-shot, local interactive (in-process server), and attach-to-remote. Their users (CI, local dev, remote dev) need all three. Fak's CLI could support attach mode for fleet debugging. | **PARTIAL** — `fak capabilities "cli attach remote server"` → native-serve only; no CLI attach mode | **ADAPT** — add --attach mode to fak CLI for remote gateway debugging | #10713 |
| 16 | **GitHub Actions integration: /opencode trigger → branch + PR** | `packages/opencode/src/github/` `packages/opencode/src/cli/cmd/github.ts` | CI/CD agent automation | Fak has no GitHub Actions integration; OpenCode has workflow action, OIDC token exchange, issue/PR comment triggers, branch creation, PR submission. Their users (repo maintainers) want issue-driven agent runs. Fak's guard/gateway could expose this integration. | **ABSENT** — `fak capabilities "github actions integration issue trigger"` → no results | **ADAPT** — add GitHub Actions workflow + trigger to fak integrations | #10714 |

---

## License Gate

OpenCode is **MIT License** (permissive). All source code is compatible for direct port or adaptation. No license laundering risk.

---

## Registration

- Study note: `docs/notes/CONCEPT-STUDY-OPENCODE-2026-09-02.md` (this file)
- INDEX.md line to add in *Notes & research* section
- Companion: `field-borrow` for individual capability witness+file
- No epic needed yet — candidates are independent leaves across multiple subsystems

---

## Durable Study Receipt

`study_5e1d3f474b8d6240719d16e9bc97e159957340e9c0e3d32bcbb5aa9e8a1a761f`

Recorded in `docs/research/inventory/anomalyco-opencode.json` and registered via `fak study add`. Source pinned at `4eb29a64f0054672950acf789f2b09487ebfbb20`.