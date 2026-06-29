# Context-safety debug-window surface (#1229, C11 of epic #1217)

_Research / design only. This is the **C11 surface spec** for the context-safety
epic [#1217](https://github.com/anthony-chaudhary/fak/issues/1217). It is the
**home** the C6 schemas asked for: where the six visual primitives live as
panels, the data each panel binds to, and the read route each panel scrapes. It
lands a concrete instance into the generic #196 (Performance Dashboards)
placeholder by hanging the panels off the **context debugger**
`cmd/fak/debug.go` — the surface that already attaches to a finished session and
answers `context-query` / `context-diff` / `--prefer-view summary`. No code
ships here — the deliverable is this committed surface spec. It consumes the
data schemas of C6
([#1223](https://github.com/anthony-chaudhary/fak/issues/1223),
[`context-safety-visual-primitive-schemas-1217.md`](context-safety-visual-primitive-schemas-1217.md)),
obeys the doctrine of C1
([#1218](https://github.com/anthony-chaudhary/fak/issues/1218),
[`context-safety-visuals-tracking-1217.md`](context-safety-visuals-tracking-1217.md)),
renders the failures cataloged in C2
([#1219](https://github.com/anthony-chaudhary/fak/issues/1219),
[`context-safety-failure-mode-catalog-1217.md`](context-safety-failure-mode-catalog-1217.md)),
and defers the **rendering** of each primitive to C6 and the **re-derivation**
of each panel's numbers to C9
([#1227](https://github.com/anthony-chaudhary/fak/issues/1227),
[`context-safety-rederivation-checker-1217.md`](context-safety-rederivation-checker-1217.md)).
This note owns the **placement and binding**, not the data and not the render._

---

## Design rule — a home for the panels, not a new dashboard stack

C6 specced the *data* of the six primitives and said explicitly: "The rendering
*home* (which panel lives where in the debug window) is child C11 (#1229); this
note owns the *data*, not the placement"
([`context-safety-visual-primitive-schemas-1217.md`](context-safety-visual-primitive-schemas-1217.md)
lines 15–17). This note is that home. It does **not** invent a dashboard
framework — it lands the six panels onto a surface that already exists and
already routes every byte through the shipped gate.

The home is the **context debugger**, `cmd/fak/debug.go`. That command already
"attaches to a FINISHED session as if to a core dump" (`debug.go:20-25`) and
already carries the seam the panels need: a `--cmd` switch
(`report | html | info | bt | x | ws | grep | tombstone | context-query |
context-diff`, `debug.go:31`) and an HTML render verb (`--cmd html`,
`debug.go:154-155`) that emits "the context debugger as a product" — a
self-contained static HTML inspection report with named panels (`debug.go:255-288`).
The context-safety panels are six **new** panels in that same HTML report, plus
their live read route on the running gateway.

Two render precedents already in the tree, reused (not replaced):

- **The static HTML report** — `internal/cdb/report_html.go:91`
  `(*Image).HTMLReport`, "the one existing in-tree HTML report" the C6 schemas
  named as the precedent for the richer panels
  (`context-safety-visual-primitive-schemas-1217.md` lines 39–42). It already
  renders panels (`debug.go:285`: "decomposition · backtrace timeline ·
  quarantine/seal panel · working-set residency · examine"), routes every byte
  through the gate (`report_html.go:10-11`), and is **deterministic** — "two
  HTMLReport calls on the same image produce byte-identical inspection"
  (`report_html.go:89`). That determinism is what makes C9's re-derivation
  possible: a render that is a pure function of the source can be re-derived
  from the source.
- **The text/emoji fallback** — `internal/scoreboard/render.go:41`
  `FromPayload`, the grade-fold the C6 schemas named as "the *only* render every
  primitive can claim today" (`context-safety-visual-primitive-schemas-1217.md`
  lines 33–38, 189–194). Every panel below degrades to a sourced text line when
  no HTML surface is present (e.g. `fak debug --cmd report`, the JSON+stdout
  path at `debug.go:152-153`).

> **Surface, not framework.** The panels are sections of the existing
> `HTMLReport` document and rows of the existing `--cmd report` text output.
> The live route is the existing gateway `/metrics` + `/debug/vars` mux. Nothing
> here adds a new server, a new port, or a JS dashboard.

---

## The read route — every panel scrapes a tamper-evident source

The gateway already serves two diagnostic routes off one mux, built from a
`routeTable()` (`internal/gateway/http.go:51`):

- **`/metrics`** — `s.handleMetrics` (`internal/gateway/http.go:116`, handler
  `internal/gateway/metrics.go:1041`). The Prometheus text surface. This is
  where the WITNESSED counter families live — e.g.
  `fak_gateway_kv_prefix_reused_tokens_total`
  (`internal/gateway/metrics.go:1531`), `fak_bgloop_up`
  (`internal/gateway/bgloops.go:227,233`), and the `fak_engine_cache_*` family
  the `CacheEventRecorder` emits (`internal/engine/cacheevents.go:304-317`).
- **`/debug/vars`** — `s.handleDebugVars` (`internal/gateway/http.go:117`,
  handler `internal/gateway/debug.go:286-291`). The expvar-style JSON snapshot
  (`debugVars`, `debug.go:293`), already carrying counters/uptime/memstats.

Both routes are exempt from auth **only for a loopback caller**
(`internal/gateway/http.go:287,299`) — the panels are an operator-local
diagnostic surface, not a public endpoint, which matches the security posture of
a debug window.

Each panel below names which of these two routes (and which metric family or
journal range) it reads. The **offline** home (`fak debug` over a persisted core
image) reads the same shapes from the journal on disk and the core-image
ledgers; the **live** home reads them from `/metrics` + `/debug/vars` on the
running gateway. The schema is identical (the C6 `CtxSafetyPanel` envelope,
`context-safety-visual-primitive-schemas-1217.md` lines 52–70), so a panel
binds the same way whether it is reading a dead session or a live one.

---

## The panel layout (wireframe)

The six primitives lay out as a column of panels inside the existing
`HTMLReport` document, after the working-set residency view, in **strength
order** — the event half first, then the four value guarantees, then the
contradiction light as a single status lamp at the top so an operator sees the
one-bit "is this even alive" answer before reading detail.

```
+==============================================================================+
| fak debug — context-safety surface        session: cdb-<id>   [WITNESSED|…]  |
|  (panels appended to the existing HTMLReport: bt · seal · working-set · …)   |
+==============================================================================+
| [P6] LIVENESS-CONTRADICTION LIGHT                          ●  (binary lamp)  |
|   /healthz up  vs  decode-rate>0      GREEN | RED-on-disagree (C12 #1230)     |
|   binds: fak_bgloop_up  ✕  decode-seconds accum     route: /metrics          |
+------------------------------------------------------------------------------+
| [P1] CONTEXT-EVENT LEDGER RIBBON              (time-series, one lane / kind)  |
|   CAP_EVICT ▔▔▔▁▁▔   CAP_FAULT ▁▁▔▁   tier offload→restore: ok|missed|fault   |
|   binds: journal CAP_EVICT/CAP_FAULT/CAP_VERSION_BIND + KVTransfer outcome    |
|   route: journal (offline) | /metrics fak_engine_cache_* (live)  lane: W/O    |
+------------------------------------------------------------------------------+
| [P2] SAVED-vs-REALIZED GAP                    (paired time-series; gap=error) |
|   saved ████████  realized █████░░░   gap ▒▒▒ = the error bar (C4 #1221)      |
|   binds: journal CAP_EVICT claim  ✕  cachewitness ReuseRatio()               |
|   route: /metrics fak_gateway_kv_prefix_reused_tokens_total    lane: W only   |
+------------------------------------------------------------------------------+
| [P3] S/N + FAULTS PAIRED STRIP                (paired: S/N lane + Faults lane)|
|   S/N ▁▃▅▇  Faults ▁▁▂█  ← Faults SPIKE flags raise-S/N-by-starving           |
|   grade: lean|ok|bloated|STARVING            binds: signalnoise Ratio/Fault   |
|   route: /debug/vars ctxplan.Outcome snapshot                  lane: W        |
+------------------------------------------------------------------------------+
| [P4] EVICT-CORRECTNESS HEATMAP                (per-evict / per-cell cells)    |
|   evict ▣▣▣  max|Δ|=0 → green ; ▢ refused (recurrent) ; cell rot ▨ = ERASURE  |
|   binds: kvcache CanEvict/TryEvict + recall.PageSyndrome digest re-derive     |
|   route: journal CAP_EVICT + core-image page syndromes        lane: W (REPAIR=M)|
+------------------------------------------------------------------------------+
| [P5] RESIDENCY CONSERVATION BAR               (roll-up; segments MUST sum)    |
|   resident = signal+noise+unaccounted  |▇signal|▒noise|░unacct|  refused≠0?   |
|   binds: ctxresidency Snapshot (Resident/Held/ByteHeld/ByteCleared)           |
|   route: /debug/vars residency snapshot                        lane: W        |
+------------------------------------------------------------------------------+
| Provenance legend:  ▇ WITNESSED   ▒ OBSERVED   ░ MODELED   (never summed)     |
| Re-derivation:  each number GREEN only if it re-derives from source (C9)      |
+==============================================================================+
```

The layout obeys the doctrine: every panel is a roll-up or time-series, never a
single point (the one binary lamp P6 is "driven by a **paired** time-series … so
the contradiction is re-derivable, not asserted",
`context-safety-visual-primitive-schemas-1217.md` lines 161–166); the provenance
legend is a visible channel at the foot (doctrine C); and the re-derivation
status rides each number (doctrine D, owned by C9).

---

## Per-panel data binding and read route

Each row binds to the C6 schema (the *what*) and names the route the surface
reads it from (the *where*). The render of the glyphs is C6's; the GREEN/RED
re-derivation gate is C9's. This note fixes only the binding and the route.

### P1 — Context-event ledger ribbon

- **Binds (C6 primitive 1):** the hash-chained journal `CAP_EVICT` / `CAP_FAULT`
  / `CAP_VERSION_BIND` rows (one lane per kind), plus the tier-transition
  secondary sources `internal/engine/cacheevents.go:304-317` `CacheEventRecorder`
  (demote/spill/evict → `cachemeta.Entry` + `fak_engine_cache_*`) and the
  `KVTransfer{Direction,Tokens,Outcome}` ok/missed/fault outcome.
- **Route:** offline, the journal on disk (replayable via `Verify`/`VerifyRows`,
  per C1); live, `/metrics` (`internal/gateway/metrics.go:1041`) for the
  `fak_engine_cache_*` family.
- **Lane:** WITNESSED (journal, fak's own cache events).
- **Renders (C2):** F2 (evict-on-recurrent refused), F6 (tier page-down missed).

### P2 — Saved-vs-realized gap

- **Binds (C6 primitive 2):** claimed-saved (journal `CAP_EVICT`) vs realized
  reuse `internal/cachewitness` `KVPrefixWitness.ReuseRatio()`
  (`internal/cachewitness/cachewitness.go:155`) over
  `fak_gateway_kv_prefix_reused_tokens_total`
  (`internal/gateway/metrics.go:1531`). Full datum schema is C4 (#1221).
- **Route:** `/metrics` for the reused-tokens counter; the journal for the saved
  claim. **Never blended with the OBSERVED provider `cache_read`** (fence b).
- **Lane:** WITNESSED only (the gap is the error bar; the OBSERVED provider
  number is a separate, never-summed lane).
- **Renders (C2):** F3 (saved-but-never-reused).

### P3 — S/N + Faults paired strip

- **Binds (C6 primitive 3):** `internal/ctxplan/signalnoise.go` `Ratio()`
  (`:69`, the caching/length-invariant signal axis), `FaultRatio()` (`:81`, the
  orthogonal under-resident axis), `Grade()` (`:249`, lean/ok/bloated/
  **starving**), over `ctxplan.Outcome`.
- **Route:** `/debug/vars` (`internal/gateway/debug.go:286-291`,
  `debugVars` at `:293`) carries the `ctxplan.Outcome` snapshot as JSON — the
  natural home for a structured per-turn record (the strip is a paired
  time-series, not a Prometheus scalar).
- **Lane:** WITNESSED.
- **Renders (C2):** F4 (S/N-up-but-faults-up); the `starving` grade is the named
  failure verdict.

### P4 — Evict-correctness heatmap

- **Binds (C6 primitive 4):** the G1 bit-identity witness
  `internal/model/kvcache.go:44-105` (`CanEvict`/`TryEvict`/
  `RecurrentEvictUnsupportedError`), per-evict `max|Δ|` (proven by
  `internal/model/paged_evict_test.go:84` and `longrope_test.go:179`),
  **extended** per-cell to the G4 integrity witness `internal/recall`
  `PageSyndrome` / `PageSyndromeFor` (digest re-derivation against the CAS body).
- **Route:** offline, the core-image page syndromes (a syndrome is a re-derived
  datum, not a counter); the journal `CAP_EVICT` rows for the per-evict axis.
- **Lane:** WITNESSED, with a REPAIRABLE syndrome rendered as MODELED-recoverable
  and an ERASURE as a hard WITNESSED loss — distinct lanes, never blended
  (fence b).
- **Renders (C2):** F1-adjacent (evict correctness) and F5 (cell rot). G1 surface
  is C3 (#1220); G4 is the per-cell extension.

### P5 — Residency conservation bar

- **Binds (C6 primitive 5):** `internal/ctxresidency` per-span `Span`
  (`ctxresidency.go:16-31` `State`; `:39-42` `BlastRadius{Tokens,
  DependentEntries}`) rolled up by `Snapshot` (`ctxresidency.go:73-89`:
  `ResidentTokens` / `HeldSpans` / `ByteHeld` / `ByteCleared` / `PollutionRate`),
  produced by the pure-read `Query` (`ctxresidency.go:98`).
- **Route:** `/debug/vars` (the residency snapshot is a structured roll-up). The
  conservation sum `resident = signal + noise + unaccounted` must add up, or the
  bar is *visibly* broken (doctrine B).
- **Lane:** WITNESSED (the kvmmu/ctxmmu ledgers reconcile by witness test).
- **Renders (C2):** the residency half of F2 (the `refused` evict segment is
  non-zero).

### P6 — Liveness-contradiction light

- **Binds (C6 primitive 6):** two heartbeats — `fak_bgloop_up`
  (`internal/gateway/bgloops.go:227,233`) vs the decode-seconds / decode-rate
  accumulator (`internal/gateway/metrics.go`). The single binary lamp is driven
  by a paired time-series so the contradiction is re-derivable.
- **Route:** `/metrics` for both heartbeats; the lamp reds when `/healthz` is up
  but decode-throughput is zero.
- **Lane:** WITNESSED.
- **Renders (C2):** F1 (the #1143 GDN evict panic: healthy but serving nothing).
  The RED-on-disagree rule is detailed by C12 (#1230).

---

## Panel → binds → route → render-defer at a glance

| Panel | Primitive (C6) | Binds (witnessed source) | Read route | Renders (C2) |
|---|---|---|---|---|
| P1 | event ribbon | journal `CAP_EVICT/CAP_FAULT`; `cacheevents.go:304-317`; `KVTransfer` outcome | journal (offline) / `/metrics` `fak_engine_cache_*` | F2, F6 |
| P2 | saved-vs-realized | `cachewitness.go:155` vs journal claim (C4) | `/metrics` `fak_gateway_kv_prefix_reused_tokens_total` | F3 |
| P3 | S/N + Faults | `signalnoise.go:69/:81/:249` over `ctxplan.Outcome` | `/debug/vars` Outcome snapshot | F4 |
| P4 | evict heatmap | `kvcache.go:44-105` + `recall.PageSyndrome` | journal `CAP_EVICT` + core-image syndromes | F1-adj, F5 |
| P5 | residency bar | `ctxresidency` `Snapshot` (`ctxresidency.go:73-89`) | `/debug/vars` residency snapshot | F2 |
| P6 | liveness light | `fak_bgloop_up` (`bgloops.go:227`) vs decode-rate | `/metrics` both heartbeats | F1 |

Render glyphs (ribbon bars, heatmap cells, conservation segments) are C6's
schema rendered through `cdb/report_html.go:91` (HTML) / `scoreboard/render.go:41`
(text fallback). The GREEN/RED re-derivation gate on every number is C9's
checker. This note fixes only **which panel reads which source from which
route**, in **which slot** of the debug window.

---

## Honest `not yet` gaps (per fence c)

- **No HTML render exists for five of the six panels — the surface is specced,
  the renderer is `not yet`.** `cdb/report_html.go:91` `HTMLReport` renders the
  *existing* panels (decomposition / bt / seal / working-set / examine,
  `debug.go:285`); the six context-safety panels are **new sections** that do
  not yet exist in that document. C6 names the same gap
  (`context-safety-visual-primitive-schemas-1217.md` lines 189–194): only the
  residency-style working-set report has an HTML precedent today. The text
  fallback (`scoreboard/render.go:41`) is the only render any panel can claim
  now.
- **The live `/metrics` + `/debug/vars` taps for P3 and P5 are proposed, not
  wired.** `/debug/vars` `debugVars` (`internal/gateway/debug.go:293`) carries
  counters/uptime/memstats today; it does **not** yet emit a `ctxplan.Outcome`
  S/N snapshot (P3) or a `ctxresidency.Snapshot` roll-up (P5). The structs exist
  and reconcile by witness test, but the route does not yet serialize them. This
  is build work, not design — named so the panel is not assumed live.
- **The offline-vs-live binding parity is asserted, not witnessed.** This note
  claims a panel binds the same C6 envelope whether reading a dead core image or
  the live `/metrics`/`/debug/vars`; no test yet proves the two derivations of a
  panel's points are identical. That two-derivation reconciliation is doctrine D,
  owned by C9 (#1227) — until C9's checker covers a rendered panel, parity is a
  design target.
- **The single "fak is safe" lamp temptation is fenced, not enforced.** The
  layout deliberately keeps six panels plus a provenance legend rather than one
  green light; nothing in the surface yet *prevents* a future maintainer from
  collapsing them into a single number — the fence (a) is doctrine, and the CI
  gate that would enforce it is C10 (#1228), unfiled against this surface.

---

## Acceptance check (against #1229)

The issue's acceptance: *a wireframe showing the six panels laid out, the data
each panel binds to (cross-ref C6 schemas), and the `/debug/vars`-or-`/metrics`
route each panel reads from — landing a concrete instance into the generic #196
placeholder, homed on the context debugger; defer rendering to C6 and
re-derivation to C9._

- **The home is concrete.** The panels hang off `cmd/fak/debug.go` — the context
  debugger that already runs `context-query` / `context-diff` / `--prefer-view
  summary` (`debug.go:31,107-135`) and already emits a named-panel HTML report
  via `--cmd html` (`debug.go:154-155,255-288`) over
  `internal/cdb/report_html.go:91` `(*Image).HTMLReport`. This is the concrete
  instance landed into the generic #196 (Performance Dashboards) placeholder.
- **The wireframe is present** — an ASCII column of P1–P6 plus the provenance
  legend, in strength order, with the liveness lamp first.
- **Each panel's data binding is named and cross-refs C6** — P1 ribbon
  (journal + `cacheevents.go:304-317`), P2 gap (`cachewitness.go:155` vs journal
  claim), P3 S/N (`signalnoise.go:69/:81/:249`), P4 heatmap
  (`kvcache.go:44-105` + `recall.PageSyndrome`), P5 residency
  (`ctxresidency.go:73-89` `Snapshot`), P6 lamp (`bgloops.go:227` vs
  decode-rate).
- **The read route is named per panel** — `/metrics` (`http.go:116`,
  `metrics.go:1041`) for the counter families and the liveness heartbeats;
  `/debug/vars` (`http.go:117`, `debug.go:286-293`) for the structured S/N and
  residency snapshots; the on-disk journal / core-image for the offline home.
- **Rendering is deferred to C6 and re-derivation to C9** — this note fixes
  placement, binding, and route only; the glyph render is C6's schema over the
  `report_html.go` / `scoreboard/render.go` precedents, and the GREEN/RED
  re-derivation gate on every number is C9's checker.
- **The un-built pieces are named as honest `not yet` gaps**, not assumed
  shipped (no HTML render for five panels; the P3/P5 live taps unwired; the
  offline-vs-live parity unwitnessed).

---

_Filed as research / planning only under epic #1217. Parent: #1217. Homes the C6
schemas (#1223) on the `cmd/fak/debug.go` context debugger, lands into the
generic #196 (Performance Dashboards) placeholder, obeys the C1 doctrine (#1218),
renders the C2 catalog failures (#1219), defers glyph rendering to C6 (#1223) and
the re-derivation gate to C9 (#1227); the liveness lamp's RED-on-disagree rule is
C12 (#1230) and the CI enforcement is C10 (#1228). Design-only — no
implementation ships under #1217 until the notes are reviewed._
