---
title: "First-class connector concepts across fak's levels — a wide survey"
description: "Positioning survey (not a build plan) for connector concepts at every fak level — install-backing stores (Postgres/DuckDB), processing frameworks, session-level agentic memory, and server-level transfer primitives. Extracts the one design recipe the repo's only shipped generic transfer primitive (KV-transfer) already proves, and applies it as a checklist to grade and sequence new connectors by minimal-effort-through-spine value vs long-term production durability."
---

# First-class connector concepts across fak's levels — a wide survey

> Positioning note, not a build plan. Every "fak already does this" claim points at a file
> in this repo; every "fak could do this" claim is unbuilt and stays unbuilt until a child
> ships with a witness (`CLAIMS.md` `[STUB]`/`[SIMULATED]`/`[SHIPPED]` discipline). Written
> 2026-07-03 as a wide survey across install-time storage backends, processing-framework
> connectors, session-level agentic memory, and server-level KV/byte transfer — the four
> levels the goal named — plus the general recipe for minting *more* primitives like
> KV-transfer, which the goal singled out as worth multiplying.

---

## 1. The one thing fak already got right — extract it before inventing more

fak has exactly one **shipped, generic, cross-tier transfer primitive**: KV-transfer. It is
not KV-specific in its design — the KV-ness is just the first tenant. Reading
`internal/cachemeta/kvtransfer.go`, `internal/abi/kvbackend.go`, `internal/abi/registry.go`
(`RegisterEngine`/`RegisterRegionBackend`/`RegisterKVBackend`), and the L3/L4 governance gates
(`internal/gateway/l3share.go`, `internal/l3region/l3region.go`,
`docs/notes/L3-DISAGGREGATED-CACHE-REIMAGINED.md`,
`docs/notes/L4-OBJECT-STORE-TIER-STUDY-2026-07-01.md`) together, the recipe is five parts:

1. **A typed byte form, content-addressed.** `hex(sha256)` identity (`blob.Digest`,
   `Ref.Digest`, L3 page keys) so identical content dedups for free across tiers and nodes,
   and a mismatch is a **detectable fact**, not a silent corruption
   (`L3_PAGE_DIGEST_MISMATCH`).
2. **A typed outcome trichotomy, never a silent fallback.** `KVResidencyOutcome` is
   `OK | Miss | Fault` (`internal/abi/kvbackend.go:13`) — the zero value maps to MISS
   (fail-closed), and a FAULT always carries a typed reason. The caller is *told* to
   recompute; it never silently does.
3. **A registration seam behind a frozen interface, not a hardcoded backend.**
   `abi.RegisterEngine`/`RegisterRegionBackend`/`RegisterKVBackend`
   (`internal/abi/registry.go:516-548`) are factory-by-name seams; a remote/off-box backend
   attaches "the *same way* the region/page-out backends do"
   (`docs/notes/THROUGHPUT-TRUST-SHARED-SPINE-2026-06-24.md` §2c). Swap the backend, never
   the ABI.
4. **Control-path/data-path separation, with the split stated honestly.** fak admits,
   verifies, and attests; bulk bytes flow client-direct where possible (presigned URLs are
   named as "the L4 form of the data-path bypass" in the L4 study). Where fak does *not* yet
   move real bytes (`BytesMoved` is today "a reported counter, no path copies KV",
   `kvtransfer.go:54`), the docs say so instead of overclaiming a mover that doesn't exist.
5. **A closed semantics-gate set scoped to the object being moved (the G1–G8 shape).**
   Verify-don't-trust the digest (G1), scope isolation — a namespace prefix is *not* an
   isolation mechanism (G4), taint travel (G5), a deletion certificate (G3), durability
   admission (G7), a closed typed refusal vocabulary (G8). The L4 study explicitly re-derives
   the *same eight gates* for the object-store tier rather than inventing new ones — that
   reuse is the evidence the recipe generalizes.

**This is the answer to "keep building more primitives like that": don't copy KV-transfer's
code, copy this five-part shape onto a new payload.** Section 5 below applies it to four
concrete non-KV candidates. Section 2–4 survey the levels the goal named against it.

One more shipped example worth naming because it is *not* KV-shaped at all and still fits the
recipe: `internal/memq/notesbackend.go`'s `NotesBackend`. It wraps an external store (markdown
memory files) behind the `memq.Backend` interface (`Cells`/`Materialize`) and applies **two
gates on every page-in regardless of backend**: a content screen (`ctxmmu.ScreenBytes` — a
secret/injection-shaped note is sealed, never rendered) and read-time re-verification
(`recall.ArtifactVerifier` — a stale concrete claim, e.g. a commit SHA that no longer exists,
refuses with `ErrStale` rather than re-entering context as fact). A new session-memory source
(a vector-DB recall store, a Codex-memory bridge) that implements `memq.Backend` inherits both
gates *for free* — this is the working template for "connect a new agentic-memory source with
minimal effort through the spine."

---

## 2. Install-time backing stores — what "back a fak installation with Postgres or DuckDB" means

### 2a. What's actually there today (honest baseline)

There is **no database backend anywhere in this module**. Every durable state surface — the
run registry, `internal/gatewayusageledger`, guard-audit journals, `docs/nightrun/*.jsonl`,
scorecard data, the honesty ledger, `.dos/` — is a flat file: JSONL, TOML, or committed
markdown, written by one process at a time and coordinated by git + advisory file locks (the
DOS lease/lane arbitration this session's own memory notes describe hitting: `index.lock`
stalls, `PATHSPEC_RACE`, shared-tree commit collisions). `docs/notes/D-006-TOOL-ECOSYSTEM-ARCH-FIT-243-2026-06-28.md`
confirms the pattern holds for the tool-facing side too: *"no DB built-in;
`examples/sql-analyst-policy.json` **gates** SQL, does not **run** it"* — fak's answer to
"database" so far has been "adjudicate access to one," never "provide one."

A hard constraint shapes every option below: **the module is zero-external-dependency, no
`go.sum`** (`AGENTS.md` line 30; restated as a design tenet in the L4 study, "the whole tier
installs with `go install`"). A real `lib/pq`, `pgx`, or cgo-linked SQLite driver breaks that
invariant. This is not a soft preference — it's load-bearing for the "60-second proof, no
key, no model, no GPU" story. Any DB connector design has to either (a) speak the wire
protocol in pure stdlib Go (the L4 study's own answer for S3 — "a pure-stdlib SigV4 +
`net/http` implementation" — generalizes: Postgres's wire protocol is a documented,
implementable-in-stdlib binary format), or (b) live as an **isolated optional leaf** that a
deployment opts into (a separate `cmd/` binary or a build-tagged package), never a default
dependency of `cmd/fak`.

### 2b. Minimal effort through the spine: DuckDB as a *query* layer, not a storage migration

The highest-leverage, lowest-risk move is not "store fak's state in DuckDB" — it's **"let
DuckDB read the state fak already writes."** DuckDB's JSON extension can query a directory of
`.jsonl` files directly with zero ingestion step and zero schema migration
(`SELECT * FROM read_json_auto('docs/nightrun/*.jsonl')`). Every JSONL surface already in this
repo (`docs/nightrun/cache-savings.jsonl`, `docs/nightrun/gateway-usage.jsonl`,
`docs/nightrun/harness-resources.jsonl`, the scorecard data files) becomes ad hoc-queryable
the moment an operator points DuckDB at the directory — **no fak code change required to get
the first value**, and no write path, so no durability or corruption risk is introduced. A
`fak query --duckdb` verb (or just a documented recipe in a `docs/integrations/` page) is the
whole first deliverable, and it composes cleanly with the release/scorecard/dispatch loops
that already emit JSONL as their canonical form — the "spine" doesn't move, a lens gets added
on top of it. This is the honest reading of "immediate value through minimal effort": the
value is in *not* migrating anything.

### 2c. Production durability: a `StateBackend` seam for the append-heavy ledgers

The place a real database earns its keep is not analytics — it's the surfaces that are
**concurrently written by multiple sessions on a shared tree today and coordinated by file
locks instead of transactions**: the run registry, the gateway usage ledger, the guard-audit
journal. Those are exactly the JSONL files this session's own memory carries scar tissue
about (shared-tree commit races, `index.lock` contention). A `StateBackend` interface —
`Append(ctx, Record) error`, `Query(ctx, Filter) (Cursor, error)`, `Compact(ctx) error` —
registered the same way `RegionBackend`/`KVBackend` are (`abi.RegisterStateBackend`, a factory
keyed by id) would let a ledger's *storage* swap from an append-only JSONL file (today's
default, zero-dependency) to a real embedded or networked store **without changing any caller
of the ledger**. That is the ABI-frozen-seam discipline this repo already applies everywhere
else, pointed at the one place it's missing.

Sequencing, worst-regret order (mirrors the spine note's S1–S5 framing):

1. **Define the `StateBackend` interface and keep the file-JSONL implementation as the
   default** — zero behavior change, but now there's a seam.
2. **A pure-Go embedded backend (SQLite-shaped, wire-protocol or a stdlib B-tree) as the
   first real alternative** — single-process durability (WAL, atomic multi-record commits)
   without a network dependency or breaking the zero-`go.sum` rule if implemented in stdlib;
   if no acceptable pure-Go path exists, this is the fork to name explicitly to an operator
   (accept one audited pure-Go driver as a deliberate policy exception vs. ship it as an
   isolated opt-in leaf) — the same shape as the spine note's named "operator forks," not a
   default this note should decide unilaterally.
3. **A real Postgres backend, hardware/network-gated, sequenced last** — the payoff is
   replacing file-lock-based multi-session coordination with row-level transactions: the DOS
   lease/lane arbiter's job (deciding whether two workers may touch overlapping trees) gets
   *easier* to reason about with a real transactional ledger under it, though the lease
   semantics themselves are a separate, already-working layer (`dos_arbitrate`) that a DB
   swap should not have to redesign.

**What does NOT belong behind this seam:** the code itself. Git remains the source of truth
for the repository; a `StateBackend` is for *operational* state (ledgers, registries,
telemetry), never a replacement for the commit graph. Keeping that boundary explicit avoids
the temptation to "just put everything in Postgres."

---

## 3. Processing frameworks (Spark, Ray, Airflow-shaped DAG runners)

The D-006 tool-ecosystem decision (`docs/notes/D-006-TOOL-ECOSYSTEM-ARCH-FIT-243-2026-06-28.md`)
already answered the architecture-fit question one level up, and it transfers directly:
**fak is gate-native, not provider-native.** It doesn't ship a Ray client or a Spark driver —
duplicating an ecosystem the caller's harness or environment already has would put fak on the
*execution* side of the boundary it exists to police. The connector concept here is not "a
fak Spark SDK," it's:

- **Adjudicate the job submission as a ToolCall** — a `spark-submit` / `ray job submit`
  invocation is a destructive-shaped or at-least-costly op; it goes through the same
  deny-by-structure / capability-floor path every other tool call does, with a
  framework-specific policy preset (mirrors the existing `examples/sql-analyst-policy.json`
  pattern: *gate*, don't *run*).
- **Quarantine the result, not just the call.** A DataFrame/RDD job's output can carry the
  same poisoned-result risk fak already guards against for any tool result — this is pure
  reuse of the existing quarantine machinery, no new primitive needed.
- **A stretch primitive worth naming: column-scoped quarantine.** Today fak's redaction/deny
  granularity is per-result. A tabular result (Arrow `RecordBatch`-shaped) is naturally
  column-addressable — redacting one PII column without denying the whole query result is the
  same *shape* as L3's `ShareScope` (a page is shareable or not; a column could be
  redactable or not) applied to structured data instead of KV pages. This is **unbuilt** and
  should stay `[STUB]`-labeled if anyone starts it; it's flagged here because it's the one
  genuinely new gate-shape a processing-framework connector would need that a plain tool-call
  gate doesn't already give for free.

Minimal effort / immediate value: policy presets + result quarantine over *existing*
`spark-submit`/`ray`/`airflow` CLI or REST invocations. Long-term durability: none of this
needs a native driver; the honesty discipline that separates "we orchestrate X" from "we
natively do X" (the spine note's "ridden vs native" distinction, §4) applies here exactly as
it does to vLLM/SGLang — never claim fak "runs" Spark when it gates a `spark-submit` call.

---

## 4. Session-level agentic connectors

Two shipped examples already are the pattern; the job here is recognizing and generalizing
them rather than inventing new machinery.

- **`memq.Backend` is the session-memory connector seam** (§1 above). A new source — a
  vector-DB-backed recall store, a live Slack-thread mirror, a Codex-memory bridge — is a
  `Cells`/`Materialize` implementation; it inherits the content screen and the staleness
  re-verification without writing any new gate code. This is the cheapest, highest-leverage
  session-level connector work available: **write a `memq.Backend` for a new source, get
  governance for free.**
- **`WarmKVStore` (`internal/session/warmsplice.go`) is the session-level KV connector**, and
  it demonstrates a second reusable pattern worth calling out on its own: **always define the
  cold/degraded path first, and make the warm path pure upside.** `Splice` on a trace with no
  parked cache returns `Warm=false` and the caller falls back to a cold re-prefill — "the
  worst case is slower, never wrong" (file header). Every connector in this survey should be
  designed with that same property: a connector failing to attach (DB unreachable, transport
  down, framework absent) degrades to the pre-connector behavior, never to a silent
  correctness gap.

What's genuinely open at this level: a **session-envelope transfer** for cross-node resume
(`internal/session/envelope.go`, named as an unbuilt WS5 item in the L4 study) is the natural
next `memq`/`WarmKV`-shaped primitive — apply the §1 recipe (content-addressed envelope bytes,
typed OK/MISS/FAULT restore outcome, a registered transport, G1/G4-shaped gates) rather than a
bespoke design.

---

## 5. Server-level transfer primitives — the recipe applied to four new payloads

Section 1's five-part recipe (typed byte form + content address, OK/MISS/FAULT outcome,
registered backend seam, control/data-path honesty, scoped G1–G8 gates) is the whole answer to
"keep building more primitives like KV-transfer." Four concrete candidates, each unbuilt
today, each a straightforward application rather than a new design:

1. **Tool-result CAS sharing.** `internal/blob` + `Ref.Digest` are *already* a generic
   content-addressed store, not KV-specific — the L3/L4 machinery (`ShareScope`,
   digest-verify admission) generalizes to *any* `Ref`-addressed blob. Sharing a memoized
   tool-call result (a web-search response, a build artifact) across agents/sessions is
   mostly already-built: it needs a producer/consumer pair that treats a tool result as the
   payload instead of a KV span, riding the same `abi.RegionBackend` seam. This is the
   cheapest of the four because the hard parts (digest identity, scope gate, deletion cert)
   are shared infrastructure already, not new code.
2. **Policy/config sync.** Propagating a capability-floor policy manifest to a fleet of
   workers with the same digest-verify + deletion-cert + typed-refusal shape as KV-transfer —
   a policy version is content-addressed, a worker either has the exact bytes (OK) or doesn't
   (MISS, fetch), and a corrupted/mismatched manifest is a FAULT, never a silently-applied
   wrong policy.
3. **Ledger/metric replication.** Aggregating `gatewayusageledger` / cache-value-ledger rows
   across fleet nodes into one view is the DB-connector question (§2c) meeting the transfer
   recipe: an append-only, digest-chained ledger replicated via a registered transport, with
   the same typed-outcome discipline (a replication gap is a reported MISS on that shard, not
   a silently incomplete rollup).
4. **Session-envelope cross-node resume** (§4, already named in the L4 study) — the direct
   sibling of `WarmKVStore`, one layer up.

None of these need a new governance model. They need the §1 recipe applied honestly, with the
same discipline this repo already enforces: name the control-path/data-path split, label
provenance (witnessed/observed/modeled), and don't let a "we can route to it" claim collapse
into "we natively transfer it."

---

## 6. Priority read (cheapest-first, cross-level)

| # | Move | Level | Effort | Payoff | Status |
|---|---|---|---|---|---|
| 1 | Document/wire a DuckDB `read_json_auto` recipe over existing `docs/nightrun/*.jsonl` | install | Tiny (docs + maybe one CLI verb) | Immediate ad hoc analytics, zero migration risk | 🔴 unbuilt |
| 2 | A new `memq.Backend` implementation for one more memory source | session | Small | Governance (screen + staleness) inherited for free | 🔴 unbuilt (pattern shipped) |
| 3 | `StateBackend` interface + keep file-JSONL as default impl | install | Small (seam only) | Unblocks every later storage swap without touching callers | 🔴 unbuilt |
| 4 | Tool-result CAS sharing (Ref-addressed, non-KV payload) | server | Medium | Reuses shipped L3 gates almost entirely | 🔴 unbuilt |
| 5 | Gate-native Spark/Ray/Airflow job-submission + quarantine policy presets | processing framework | Medium | Immediate value without owning a client SDK | 🔴 unbuilt |
| 6 | Pure-Go embedded `StateBackend` (SQLite-shaped) behind #3's seam | install | Medium–Large | Real multi-record-commit durability without breaking zero-`go.sum` | 🔴 unbuilt, operator fork on driver choice |
| 7 | Session-envelope cross-node transfer (§4/§5.4) | server + session | Large (needs S3/S4 byte-mover spine first) | Cross-node warm resume | 🔴 unbuilt, sequenced behind the shared KV-byte spine |
| 8 | Column-scoped quarantine for tabular processing-framework results | processing framework | Large (new gate shape) | Finer-grained redaction than today's per-result gate | 🔴 unbuilt, genuinely new primitive |
| 9 | Real Postgres `StateBackend`, hardware/network-gated | install | XL | Replaces file-lock coordination with transactions | 🔴 unbuilt, sequenced last (mirrors spine note's S6b posture) |

Rows 1–3 are hardware-ungated, dependency-free, and buildable today; row 9 is the most
expensive and most reversible-later, so it's deliberately last — the same worst-regret
ordering the shared-spine note already uses for the throughput/trust fork.

## 7. Honest fences

- Nothing in this note is built. Every row in §6 is `[STUB]`-or-nothing until a child ships
  with a witness, per `CLAIMS.md` discipline.
- The zero-`go.sum` constraint is treated here as load-bearing, not a preference to relax
  casually — any DB connector proposal should carry an explicit dependency-boundary answer
  (pure-stdlib wire client, or an isolated opt-in leaf) the way the L4 object-store study
  already committed to for S3.
- "Ridden vs native" discipline (spine note §4) applies to every processing-framework and
  DB connector claim exactly as it does to vLLM/SGLang: gating `spark-submit` is not "fak runs
  Spark," and a pure-stdlib Postgres wire client is not "fak IS Postgres."

## 8. See also

- `docs/notes/THROUGHPUT-TRUST-SHARED-SPINE-2026-06-24.md` — the shared-spine investment
  thesis and worst-regret sequencing this note's §6 mirrors.
- `docs/notes/L4-OBJECT-STORE-TIER-STUDY-2026-07-01.md` — the most recent prior art for "add
  a new external-store tier behind the frozen seams," reused wholesale for §2/§5's shape.
- `docs/notes/D-006-TOOL-ECOSYSTEM-ARCH-FIT-243-2026-06-28.md` — the gate-native-not-provider-native
  decision §3 extends to processing frameworks.
- `internal/abi/kvbackend.go`, `internal/abi/registry.go`, `internal/cachemeta/kvtransfer.go` —
  the shipped recipe §1 extracts.
- `internal/memq/notesbackend.go`, `internal/session/warmsplice.go` — the two working
  session-level connector examples §4 generalizes.
- `internal/engine/vllm.go` — the reference ridden-engine adapter shape (`abi.RegisterEngine`,
  public-surface-only, honesty boundary stated in the doc comment) any processing-framework
  or DB wire-protocol connector should imitate.
