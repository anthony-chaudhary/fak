---
title: "The queried skill loader — 0 cost for ∞ skills, paged on demand, versioned and hot-swappable"
description: "A skill is resolved by an in-kernel QUERY against a paged, versioned, witnessed residency — not selected from a flat menu. fak already has the MMU (ctxmmu), the witnessed residency read (ctxresidency), and the skill@version@digest procedural cache (contextq). This epic builds the query front-end on top, and generalizes the same capability object to MCP tools, A2A agents, and the next protocol."
---

# The queried skill loader

**Thesis.** A skill should be *queried* the way a KV page is faulted in — not *loaded
up front* the way a menu item is printed. Today every agent harness, fak included,
pays an O(N) tax: all N skill name+descriptions sit in the system prompt so the model
can pick. That tax is the wall between an agent and *infinitely many* skills. Remove
it and a skill becomes a paged capability: **0 marginal context cost at rest, faulted
into context only when the query says it is needed, evicted under pressure, and served
as a HIT when an identical invocation recurs.** The same object — query in, body paged,
residency witnessed, version remappable — is *not* skill-specific. A skill, an MCP
tool, an A2A agent, and the next protocol are the *same* paged capability behind one
resolver. That generality is the point.

This is not greenfield. fak already ships the three hard parts. This epic builds the
front-end that turns them into a skill loader, then generalizes it.

---

## 1. Where the field is, and the exact gap

Anthropic's own Agent Skills architecture is **progressive disclosure in three layers**:
(1) name+description metadata pre-loaded into the system prompt; (2) the full `SKILL.md`
body loaded when the model judges the skill relevant; (3) bundled files/scripts loaded
on demand during execution. It is a real advance over "load everything." But the
*selection* step is **flat**: the model sees *all* skill metadata at once and picks.
That has four consequences Anthropic's post does not solve, and the rest of the field
is now racing to:

- **O(N) metadata tax.** Layer 1 cost grows linearly with the skill count. At a few
  dozen skills it is a rounding error; at thousands it is the dominant prefix and it
  crowds out the task. "0 for ∞ skills" is precisely the property a flat menu cannot
  have.
- **No retrieval.** Selection is the model reading a list, not a query against an
  index. **RAG-MCP** (arXiv 2505.03275) shows retrieving only the relevant tool schema
  lifts tool-selection accuracy from **13% → 43%** on a large pool while shrinking the
  prompt — the menu actively *degrades* selection as it grows.
- **No active discovery.** **MCP-Zero** (arXiv 2506.01056) has the model *emit* a
  structured `<tool_assistant>{server, tool}</tool_assistant>` request when it hits a
  capability gap, then does hierarchical server→tool retrieval. Result: **98% token
  reduction** (111 vs 6,308 tokens) and **95.2% vs 69.2%** selection accuracy on the
  full pool. Decision authority moves *back to the model*, expressed as a query.
- **No residency discipline.** **MemTool** (arXiv 2507.21428) adds *and evicts* tools
  per conversation turn by relevance — a working set, not a monotonically growing
  prefix. **ScaleMCP** (arXiv 2505.06416) keeps the index in sync as tools change with
  a **SHA-256 hash-diff CRUD pipeline**, which is exactly the "0 → ∞ skills" indexing
  problem: only the changed capability is re-indexed.

None of these systems has a *kernel*. They bolt retrieval onto a stateless prompt.
**fak does have a kernel** — a context MMU with paging, eviction, and a witness gate —
and that is the moat this epic spends.

## 2. What fak already ships (the substrate this stands on)

| fak package | What it already does | Role in the skill loader |
|---|---|---|
| `internal/ctxmmu` (tier 2) | Context MMU: page-in / page-out to a bounded pointer, **CAS-pinned eviction**, quarantine + **witness-clear** gate before re-admission. | The **paging engine**. A skill body is a page that faults in on query, pages out under pressure, and re-admits only through the witness. We do not write a new pager. |
| `internal/ctxresidency` (tier 3) | Witnessable READ over the span ledger: per-span `resident / evictable / held` + **eviction blast radius** (what an evict would drop), reconciled against the kernels' own counters. | The **residency view** for skills: which capabilities are resident, which are safe to evict, and the cost of evicting one. The eviction *cost model* is already built. |
| `internal/contextq` (`skillmemory.go`, `viewindex.go`) | In-kernel **`Query()`** returning a working-set of views; **`SkillContextRecord{SkillName, Version, InvocationDigest}`** → a procedural-memory view served as a **HIT** on an identical re-invocation, **FAULT** (build once) otherwise. | The **query primitive** and the **v1/v2/v3 cache**. A versioned skill invocation is *already* a digest-keyed cache object. Doc: `docs/SKILL-CONTEXT-MEMORY.md`. |
| `internal/cachemeta`, `internal/kvmmu` | Payload-free metadata with a deterministic binding-key fold; KV-cache coherence under eviction. | **Prefix binding**: a skill's stable preamble can be a reusable KV prefix; a capability's identity is a `cachemeta.Entry`. |
| `internal/architest` | Machine-checked layered-DAG tiering (0 root → 4 integrator); a new package is added at the lowest fitting tier or CI fails. | The **layering contract** every new package below obeys. |
| `cmd/fak/skill_effectiveness.go` | `fak skill-effectiveness-scorecard` walks `.claude/skills/*/SKILL.md` and grades discoverability/operability/trust. | The **hygiene baseline**; the queried loader extends the surface it grades. |

The honest gap list: **no capability index, no query front-end, no per-capability
residency/eviction, no versioned page table, no protocol-generic resolver, no CLI, no
scorecard for the loader itself.** Those are the children below.

## 3. The architecture: a skill is a paged capability resolved by query

### 3.1 The capability — the modular core (answers "generalize to MCP / new protocols")

The loader never speaks "skill." It speaks **capability**: a descriptor + a lazily
paged body behind one interface.

```
type Capability struct {
    Ref      CapRef          // {Kind, Name, Version} — Kind ∈ {skill, mcp-tool, a2a-agent, …}
    Digest   string          // content hash of the body (the ScaleMCP sync key)
    Card     []byte          // the tiny queryable card: trigger clause + tags (layer-1 cost)
    Resolve  func() []byte   // pages in the full body on FAULT (layer-2/3)
    Scope    abi.ShareScope
}

type Resolver interface {        // one seam for every protocol
    Index() []CapCard            // cheap cards only — the at-rest cost
    Fault(ref CapRef) Capability // page in the body on demand
}
```

A `.claude/skills/*/SKILL.md` is one `Resolver` (the `skill` kind). An MCP server is
another (`mcp-tool` kind — its `tools/list` *is* an `Index()`, its `tools/call`
schema *is* a `Fault()`). An A2A endpoint is a third. **The query, the index, the
residency, the eviction, the versioning, and the witness are written once over
`Capability`** and inherited by every protocol. Adding "the next protocol" is writing
one `Resolver`, nothing else. This is the user's modularity directive made structural:
the same concepts (query, page, evict, version) apply to MCP and new protocols
*because the loader is protocol-blind by construction*.

### 3.2 The query — the in-kernel "virtual skill" angle (fak's unique play)

The model does not read a menu. It **emits a query** (MCP-Zero's active-discovery
move) — an intent string + a context budget. fak resolves it **in-kernel** via
`contextq.Query`, which already returns a *budget-bounded working set of views*. The
query ranks `CapCard`s (cheap, at-rest) and faults in only the winners' bodies, up to
the budget. The "virtual" part: what the model sees is a **view over the capability
space**, materialized per query — never the whole space. The 0-for-∞ property falls
out: at rest the model holds *no* skill bodies and only an index it queries; cost is a
function of the *query's* working set, not the *catalog's* size.

Because the query is in-kernel, it is **witnessed and auditable** — every fault, every
eviction, every version bind is a journal row `ctxresidency` can read back, the same
way fak audits KV residency. No competing skill-RAG system can say that.

### 3.3 Versioned page table — virtual skill environments (v1/v2/v3, hot-swap)

`SkillContextRecord` already keys on `Version`. We lift that into a **page table**:
a query binds a capability to a *version* (pinned, latest, or A/B). Multiple major
versions are resident *side by side* — `v1` and `v2` of the same skill occupy distinct
pages with distinct digests, exactly as two processes map distinct frames. **Hot-swap
is a remap**, not a reload: flip the page-table entry from `v1` to `v2`; in-flight
invocations keep their pinned frame (the CAS pin in `ctxmmu` already guarantees a
pinned page survives eviction), new invocations resolve the new version. Rollback is
the inverse remap — and `ctxresidency`'s blast-radius read tells you what a swap
invalidates *before* you flip. This is a virtual skill environment in the OS sense:
isolated, versioned address spaces over one physical context.

### 3.4 Residency + eviction — the working set (answers "context-window memory mgmt")

A faulted capability body is resident. Under context pressure the loader evicts the
coldest capability (MemTool's per-turn relevance, but with fak's *witnessed* eviction
and *measured* blast radius instead of a heuristic score). A re-query for an evicted
capability is a `FAULT` that re-pages it; an identical re-invocation is a `HIT` from
`contextq`'s procedural cache (re-rendering nothing). The skill working set is thus a
true cache hierarchy — query-driven admission, pressure-driven eviction, digest-keyed
reuse — over the MMU fak already operates.

### 3.5 The 0→∞ index — ScaleMCP's sync, made witnessable

The index stores only `CapCard`s keyed by `Digest`. A capability that changes gets a
new digest (SHA-256 over the body, ScaleMCP's mechanism); the indexer does a CRUD diff
— create new, update changed, delete vanished — so the at-rest index cost is
**O(catalog cards)**, never O(catalog bodies), and re-indexing is **O(changed)**. This
is the package the rest of the epic stands on (the keystone).

## 4. The epic — child issues

Build order is keystone-first; each child names the fak package it extends so no one
reimplements the pager or the procedural cache.

- **C1 (keystone) — `internal/capindex`: the capability descriptor + hash-diff index.**
  Define `Capability` / `CapRef` / `CapCard` / `Resolver`. Implement the ScaleMCP-style
  SHA-256 CRUD sync so the index is cheap at rest and re-indexes only changed
  capabilities. A `skill` `Resolver` over `.claude/skills/`. Tier: foundation/mechanism.
  *Everything else blocks on this.*
- **C2 — the in-kernel query front-end (`contextq` extension).** Accept a model-emitted
  query (intent + budget), rank `CapCard`s, fault in winners up to the budget via
  `contextq.Query`. This is the MCP-Zero active-discovery move, in-kernel and witnessed.
- **C3 — per-capability residency + eviction (`ctxresidency` + `ctxmmu` extension).**
  Track each resident capability's residency state and blast radius; evict the coldest
  under pressure through the witness gate; re-fault on re-query. The MemTool working set
  with measured (not heuristic) cost.
- **C4 — the versioned page table / virtual skill env (`SkillContextRecord` lift).**
  Resident side-by-side versions; hot-swap = remap; CAS-pinned in-flight invocations
  survive a swap; rollback is the inverse remap; blast-radius read before a flip.
- **C5 — the protocol-generic resolvers (MCP + A2A).** An `mcp-tool` `Resolver` whose
  `Index()` wraps `tools/list` and `Fault()` wraps the call schema (folds the existing
  `internal/gateway/mcp.go`); an `a2a-agent` `Resolver`. Proves the loader is
  protocol-blind and gives the next protocol a one-file on-ramp.
- **C6 — the witness + audit surface (`ctxresidency` read side).** Every fault /
  eviction / version-bind is a journal row; a `fak` read reconciles the loader's view
  with the kernel counters, the way KV residency is audited today. The trust floor.
- **C7 — CLI + scorecard.** `fak skill query <intent> [--budget N]`, `fak skill
  residency`, `fak skill swap <name> <ver>`; extend the skill-effectiveness scorecard
  with a *loader* dimension (is the catalog queryable? does it page? is the index in
  sync?). Closes the RSI loop the scorecard family expects.

## 5. Why this is fak's to win

Every other system in §1 retrofits retrieval onto a stateless prompt. fak owns the
layer underneath: a context MMU that already pages, pins, evicts, and *witnesses*.
On that substrate the queried skill loader is not a new subsystem — it is a new
*resolver and query* over machinery fak already runs in production paths. The unique
properties no competitor can match: the query is **in-kernel** (audited, not a
black-box reranker), the working set is **witnessed** (every fault and evict is a
verifiable journal row), versions are a **page table** (true hot-swap and rollback, not
a reload), and the whole thing is **protocol-blind** (skill, MCP, A2A, and the next
protocol are one capability object). That is how you get to **0 cost for ∞ skills** and
keep the trust floor fak sells.

## References

- Anthropic, [Equipping agents for the real world with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — the three-layer progressive-disclosure baseline this epic extends past flat selection.
- RAG-MCP, [arXiv 2505.03275](https://arxiv.org/html/2505.03275v1) — retrieval lifts tool selection 13%→43%; the menu degrades as it grows.
- MCP-Zero, [arXiv 2506.01056](https://arxiv.org/pdf/2506.01056) — model-emitted active discovery; 98% token cut, 95.2% accuracy.
- ScaleMCP, [arXiv 2505.06416](https://arxiv.org/html/2505.06416v1) — SHA-256 hash-diff CRUD auto-sync index (the 0→∞ mechanism).
- MemTool, [arXiv 2507.21428](https://arxiv.org/html/2507.21428v1) — add/evict tools per turn; the residency working set.
- fak internals: `internal/ctxmmu`, `internal/ctxresidency`, `internal/contextq` (`skillmemory.go`, `viewindex.go`), `internal/cachemeta`, `internal/architest`; `docs/SKILL-CONTEXT-MEMORY.md`.
