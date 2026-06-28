---
title: "D-006 (#243) Tool Ecosystem Expansion — the architecture-fit decision and honest on-disk status"
description: "Resolves the deferred architecture-fit question for D-006 (#243, Tool Ecosystem Expansion): does fak SHIP a tool library, or only GATE tools? Decides it on real on-disk evidence (fak already ships the confined, kernel-routed fak_read built-in #795, plus the readOnly/idempotent/destructive safety-annotation layer), reframes the LangChain-shaped acceptance into fak's gate-native idiom, corrects the now-stale #304 tracker row, and names the single remaining gate plus the smallest next code leaf (fak_glob). No feature is claimed shipped that is not on disk."
---

# D-006 (#243) Tool Ecosystem Expansion — architecture-fit decision + honest status

> **Decision note for [#243](https://github.com/anthony-chaudhary/fak/issues/243)**
> (D-006, Track D · Agent Framework, P2; umbrella epic [#304](https://github.com/anthony-chaudhary/fak/issues/304)).
> The [Track D parity tracker](track-d-agent-framework-parity-tracking-304.md) (2026-06-25)
> graded D-006 **🔴 Not implemented / design-only**, and its §3 next step was the one this
> note discharges: *"Decide the architecture fit first (does fak ship tools, or only gate
> them?); if shipping, add filesystem read/write/glob with safety annotations as the first
> leaf."* That tracker row is now **stale in one respect** — the read half already shipped
> (`fak_read`, #795) — so the decision can be made on evidence, not speculation.
> Written **2026-06-28** on a win32 dev box (native `go build`/`go vet` green; tests run
> under WSL/CI per [AGENTS.md](../../AGENTS.md)).

---

## 1. The question D-006 actually asks

The migrated issue body asks for a **built-in tool library** — *"20+ core tools (file ops,
HTTP, database, etc.), tool documentation, safety annotations, example workflows"* — with
acceptance:

- [ ] File system tools (read, write, glob)
- [ ] HTTP client tools
- [ ] Database connector tools
- [ ] Each tool documented and safety-annotated

That acceptance is shaped like **LangChain / LlamaHub**: a *provider* that ships and runs a
broad catalog of tools. fak is not that kind of system, and the issue's own dependency line
(*"None"*) hides a real architecture-fit question the tracker flagged but did not answer.

## 2. The decision

**fak's tool ecosystem is gate-native, not provider-native — and that is the correct fit.**

fak is an *agent kernel*: it sits between an agent and the tools it already has, and
adjudicates every call (deny-by-structure, repair malformed calls, quarantine poisoned
results). Shipping a LangChain-style general-purpose tool *provider* would duplicate the
ecosystem the agent's own harness already supplies, and it would put fak on the **execution**
side of the very boundary it exists to police. So D-006's literal "20+ provider tools"
framing is **partly mis-scoped for fak**. fak expands the tool ecosystem two ways, both of
which are *already underway on disk*:

1. **The safety-annotation + adjudication layer over ANY tool** (fak's actual product).
   This is shipped and is the honest reading of *"each tool documented and
   safety-annotated"*:
   - per-tool read-only / idempotent / **destructive** annotations —
     `metaFor` in [`internal/agent/tools.go`](../../internal/agent/tools.go) (`readOnlyHint`
     / `idempotentHint` / `destructive`), and the `read_only` hint on the MCP wire schema
     ([`internal/gateway/mcp.go`](../../internal/gateway/mcp.go), `toolDescriptors`);
   - capability-floor allow/deny + redaction, declared as policy manifests
     ([`examples/dev-agent-policy.json`](../../examples/dev-agent-policy.json) and ~20
     sibling presets) — the deployable form of a safety annotation;
   - tool-contract schemas + alias repair (`internal/grammar`, `internal/preflight`,
     [`internal/toolsandbox`](../../internal/toolsandbox)).

2. **A small, confined set of kernel-routed built-ins** where routing *through* fak buys
   something the raw tool cannot give: the vDSO verified-fresh cache, path confinement, and
   in-line adjudication. `fak_read` is the shipped proof of concept (#795): the model calls
   `fak_read` instead of the harness's built-in Read, and the kernel serves a
   **verified-fresh cached** result with no disk I/O on a hit, or a working-tree-**confined**
   `os.ReadFile` on a miss ([`internal/agent/readengine.go`](../../internal/agent/readengine.go),
   wired in `Configure` and advertised in `toolDescriptors`). The built-in family is the
   `fak_*` set (`fak_read`, `fak_syscall`, `fak_adjudicate`, `fak_admit`, `fak_changes`,
   `fak_revoke`, …) in [`internal/gateway/mcp.go`](../../internal/gateway/mcp.go).

The corollary that scopes the rest of the acceptance: **fak ships a built-in only where the
kernel adds value over the raw tool**, and it ships write/network built-ins only as
*gated* paths, never as raw providers. A `fak_glob` is in-architecture immediately
(read-only, idempotent, cacheable, confined — a strict subset of `fak_read`'s power). A
`fak_write` must route through the destructive-op gate. "HTTP client" and "Database
connector" tools only make sense as **gated connectors** (the SSRF/cloud-metadata egress
floor `internal/egressfloor`, plus result-side quarantine), not as raw request runners.

## 3. Honest on-disk status against each acceptance box (2026-06-28)

| Acceptance item | On-disk state | Deciding evidence |
|---|---|---|
| File system **read** | 🟢 **Shipped** — confined, kernel-routed, cache-backed | `fak_read` MCP tool: [`internal/gateway/mcp.go`](../../internal/gateway/mcp.go) (`case "fak_read"`, `fakRead`, `toolDescriptors`), engine [`internal/agent/readengine.go`](../../internal/agent/readengine.go), registered in [`internal/agent/tools.go`](../../internal/agent/tools.go) `Configure` → `RegisterReadEngine`, test `internal/gateway/fak_read_test.go` (#795) |
| File system **glob** | 🔴 Not shipped — **the named next leaf** | no `fak_glob` engine/descriptor exists (grep) |
| File system **write** | 🔴 Not shipped — must be a destructive-gated path | no `fak_write` built-in; writes are *gated* today (adjudicator deny / self-modify floor), not *provided* |
| **HTTP client** | 🔴 Not shipped as an agent-callable built-in | only a *gated* egress floor exists (`internal/egressfloor`); no `fak_http` request tool |
| **Database connector** | 🔴 Not shipped | no DB built-in; `examples/sql-analyst-policy.json` *gates* SQL, does not *run* it |
| Each tool **documented + safety-annotated** | 🟡 **Mechanism shipped, catalog partial** | `metaFor` read-only/destructive hints + the MCP `read_only` schema + ~20 policy presets are the annotation layer; only the shipped `fak_*` tools are documented in `toolDescriptors` |

Legend: 🟢 shipped · 🟡 partial · 🔴 not shipped.

So the tracker's "🔴 Not implemented / — (none)" row for D-006 should read **🟡 partial**:
the read built-in + the safety-annotation layer are real; glob/write/HTTP/DB built-ins are
not.

## 4. The gate — why this note does not close #243

The issue's literal acceptance has **four feature boxes**, three of which (glob, HTTP, DB —
plus a write path) are unbuilt, and the "20+ tools" target is a multi-day build. That gate
is unchanged by a decision note; what this note changes is that the build is now **decided
and scoped** instead of an open architecture question. Two honest paths for the operator:

- **Re-scope #243 to fak's gate-native idiom** (recommended): the acceptance becomes "the
  adjudication/safety-annotation layer over any tool + a confined kernel-routed built-in
  set." Under that reading D-006 is *mostly shipped* and closes after the next small leaf
  (`fak_glob`). This is the architecturally honest framing.
- **Keep the literal LangChain-shaped acceptance**: #243 stays open as a 7–10 day,
  partly-architecturally-inapplicable build (a raw HTTP/DB provider is the kind of
  unmediated execution fak exists to gate).

## 5. Smallest next code increment (for the agent that picks this up)

`fak_glob` — a read-only, idempotent, working-tree-**confined** directory-glob built-in that
mirrors `fak_read` exactly:

1. `globEngine` in `internal/agent/` (the read-only sibling of `readEngine`; reuse the
   `filepath.Rel` confinement check and the deny-as-value error shape);
2. `RegisterGlobEngine` called from `Configure` (like `RegisterReadEngine`);
3. a `fakGlob` handler + `case "fak_glob"` + a `toolDescriptors` entry carrying the
   read-only safety annotation, in `internal/gateway/mcp.go`;
4. a test (`go test ./internal/agent ./internal/gateway -run Glob`).

It is read-only and confined, so it adds **no** new capability surface beyond `fak_read`
(strictly less — it lists names, it does not read contents), and it ticks the remaining
in-architecture half of acceptance box 1. `fak_write` and the HTTP/DB connectors are
separate, larger leaves that must land **behind** the destructive-op gate and the egress
floor respectively.

## 6. Provenance

- **Issue:** `gh issue view 243` (live, 2026-06-28); acceptance from the migrated body.
- **Built-in tool surface:** `internal/gateway/mcp.go` (`callTool` switch + `toolDescriptors`),
  `internal/agent/readengine.go`, `internal/agent/tools.go` (`Configure`, `metaFor`).
- **Absence evidence:** grep for `fak_glob` / `fak_write` / `fak_http` / `fak_sql` over
  `internal/gateway` — no matches (2026-06-28).
- **Prior art this refines:** [`track-d-agent-framework-parity-tracking-304.md`](track-d-agent-framework-parity-tracking-304.md)
  §1 (D-006 row) and §3 (the "decide architecture fit first" next step).
- **Honesty rails:** no feature is claimed shipped that is not on disk; the three unbuilt
  boxes and the multi-day target are named as the explicit open gate.

## 7. See also

- [`track-d-agent-framework-parity-tracking-304.md`](track-d-agent-framework-parity-tracking-304.md) — the Track D roll-up (epic #304) this note feeds back into.
- [`internal/agent/readengine.go`](../../internal/agent/readengine.go) · [`internal/gateway/mcp.go`](../../internal/gateway/mcp.go) — the shipped `fak_read` built-in and the `fak_*` tool surface.
- [`examples/dev-agent-policy.json`](../../examples/dev-agent-policy.json) · [`internal/toolsandbox`](../../internal/toolsandbox) — the safety-annotation / tool-contract layer.
