# System-prompt MMU — Rung 6 observability surface (#1264, feeds #1217)

_Research / design only. This is the **Rung 6 spec** for the system-prompt MMU
epic [#1258](https://github.com/anthony-chaudhary/fak/issues/1258) (design note
[`SYSTEM-PROMPT-MMU-2026-06-29.md`](SYSTEM-PROMPT-MMU-2026-06-29.md)). It is the
**observability surface**: the view that makes the live base context legible and
**proves the realized wire prefix equals the planned spine**. No code ships here —
the deliverable is this committed spec: what the view reads, the re-derivation
contract it enforces (divergence is an **alarm**, not a metric), the data each
read binds to (a real `file:line` seam), and the host it hangs off. It **consumes**
the context-safety spine of epic [#1217](https://github.com/anthony-chaudhary/fak/issues/1217)
(it reads its journal / series and its C9 re-derivation discipline) and **never
re-reports** its own numbers back into that roll-up. It depends on Rungs 1–5 for
the data it renders — all unbuilt today, named below as honest `not yet` gaps._

---

## The missing piece, in one line

> The spine is **byte-identical-or-cold by construction** — `promptmmu` already
> proves it with `bytes.Equal(raw[:prefixEnd], out[:prefixEnd])` or fail-safe
> identity (`internal/promptmmu/promptmmu.go:133`). What the epic does **not** yet
> have is a **view** that, per turn, re-derives the realized wire prefix from the
> planned spine and **reds when they disagree** — turning the splice floor's
> silent pass/fail into a legible, operator-facing **alarm** that catches an
> accidental head mutation *before* it costs a cold re-prefill. This note specs
> that view.

The splice floor is a guard: it fails safe (reverts to identity) when the prefix
would move. A guard that fires silently is invisible to the operator until the
cache-miss bill arrives. Rung 6 is the **witness** over that guard — it surfaces
the floor's verdict as a per-turn divergence series, names *which* segment moved
and at *which* byte, and reds the whole view on a single divergence.

---

## What the view shows — three reads, each consuming Rung 1–5 data

A `fak debug` / `fak prompt show` view with three reads, all bound to the
system-prompt MMU's own substrate rather than minting parallel numbers:

### Read 1 — the live segment plan (spine / policy / overlay)

The ordered `[]cachemeta.PromptSegment` the Rung-1 spine emits, rendered as a
table: one row per segment, each carrying its **kind**, **token count**, and
**witness**.

- **Binds:** `cachemeta.PromptSegment{Kind, Tokens, Content, Witness}`
  (`internal/cachemeta/prefix_stability.go:36`), keyed by `SegmentKind`
  (`prefix_stability.go:22`) — the spine is `SegStable`
  (`prefix_stability.go:25`, commented *"system prompt, static instructions —
  should be byte-identical every turn"*); the policy floor and the queried
  overlay cards are the tiers Rung 1 / Rung 3 assign below it.
- **Producer:** the Rung-1 package `internal/syspromptmmu` (the one genuinely new
  authorship surface; **not built today** — see `not yet`).
- **Per-row:** `Kind` (spine / policy / overlay), `Tokens`, `Witness` (the git
  SHA / blob hash / lease epoch the segment's content depends on, empty for a
  segment with no external dependency).

### Read 2 — resident vs paged (what's faulted in, what's evicted)

Which overlay cards are resident and which are paged out, plus the blast-radius
cost an eviction would charge — the residency decision Rung 4 records.

- **Binds:** `internal/ctxresidency` — `Snapshot()`
  (`capresidency.go:316`) for the resident/paged split, `MeasureBlastRadius(key)`
  (`capresidency.go:174`) for the dependent-entry cost of an evict, and
  `EvictColdest` (`capresidency.go:213`) for the coldest-first policy; the
  KV-level roll-up `Snapshot` (`ctxresidency.go:73`) and `BlastRadius{Tokens,
  DependentEntries}` (`ctxresidency.go:39`) reconcile against the kernel's own
  ledger (a witness test asserts `ResidentTokens == kvmmu.Context.CacheLen`).
- **Read route:** the residency snapshot is a structured roll-up (the same shape
  the C11 surface reads off `/debug/vars`), bound here to the **spine** address
  space rather than the result-side context window.

### Read 3 — the prefix-divergence series + re-derivation check (the heart)

Per turn, the **planned spine bytes** (the Rung-1 segment plan serialized through
Rung 2's `promptmmu` splice) vs the **realized wire prefix** (what actually went
on the wire). The check is `bytes.Equal(planned, realized)`. **Divergence is an
alarm, not a metric** — a diverging turn reds the view; it is never averaged into
a "% byte-equal" score that would hide the one turn that broke.

- **Binds:** the Rung-1 plan (Read 1) re-serialized vs the Rung-2 splice output,
  proven the way `promptmmu` already proves it —
  `bytes.Equal(raw[:prefixEnd], out[:prefixEnd])` or `identity(raw,
  SkipSpliceUnproven)` fail-safe (`internal/promptmmu/promptmmu.go:133`).
- **Read route:** offline, the recorded turns on the debug image; live, the
  splice floor's per-turn pair.

---

## The re-derivation contract (spine == prefix)

Rung 6 is a **consumer instance** of the C9 re-derivation checker contract
([#1227](https://github.com/anthony-chaudhary/fak/issues/1227),
[`context-safety-rederivation-checker-1217.md`](context-safety-rederivation-checker-1217.md)),
pointed at the system-prompt spine:

| C9 contract term | Rung-6 binding |
|---|---|
| the **rendered number** | the realized wire prefix bytes (what the turn actually sent) |
| the **tamper-evident re-derivation** | the planned spine, re-serialized from the Rung-1 `[]PromptSegment` plan |
| the **tolerance** | `max\|Δ\|=0` — bit-identity, exactly the `bytes.Equal` floor at `promptmmu.go:133` |
| **GREEN** | `bytes.Equal(planned, realized)` |
| **RED** | any mismatch — the divergence *is* the error bar |
| **the visual is GREEN iff every number is GREEN** | one diverging turn reds the whole series (conservation discipline) |

This is the same **rebuild-and-prove-equal** discipline the tree already runs for
the index (`internal/ctxplan/image.go` proves `RestoreIndex(ix.Image()) == ix`)
and for the inbound splice (`promptmmu`'s `bytes.Equal` prefix floor). Rung 6
extends it from "the index round-trips" / "the splice didn't move the prefix" to
"**the head the operator believes is pinned is byte-for-byte the head that
shipped**." On a RED the view names **which** segment diverged (spine vs policy)
and at **which** byte offset, so an accidental head mutation is caught at its
source rather than inferred later from a cache-miss spike.

---

## Consumes #1217, never re-reports into it (the fence)

The issue's second acceptance — *the view reads the #1217 journal / series; it
does not blend its own numbers into the context-safety roll-up* — is the C1
provenance/conservation doctrine
([#1218](https://github.com/anthony-chaudhary/fak/issues/1218)) applied here:

- The view **reads** the context-safety surface: the C6 `CtxSafetyPanel` /
  `PanelPoint` envelope ([#1223](https://github.com/anthony-chaudhary/fak/issues/1223),
  [`context-safety-visual-primitive-schemas-1217.md`](context-safety-visual-primitive-schemas-1217.md))
  and the C9 verdict discipline, plus the Rung 1–5 data.
- The view **does not** sum the spine-divergence number into the context-safety
  conservation roll-up. The spine check is its **own** panel / line, emitted
  beside the C11 panels, never folded into their total. It **consumes, never
  re-reports** — the same boundary the C4 saved-vs-realized gap keeps when it
  refuses to blend the WITNESSED reuse count with the OBSERVED provider
  `cache_read` ([#1221](https://github.com/anthony-chaudhary/fak/issues/1221)).
- **Provenance lane: WITNESSED.** Both bytes are fak's own — the planned plan and
  fak's own wire output — and both are tamper-evident (the plan re-serializes
  deterministically; the wire bytes are the recorded turn). No OBSERVED or
  MODELED number rides this check.

---

## Host: `cmd/fak/debug.go`

Consistent with the epic's host choice and with the C11 debug-window surface
([#1229](https://github.com/anthony-chaudhary/fak/issues/1229),
[`context-safety-debug-window-surface-1217.md`](context-safety-debug-window-surface-1217.md)),
which already homes the six context-safety panels on `cmd/fak/debug.go` — the
context debugger that "attaches to a FINISHED session as if to a core dump"
(`debug.go:20-25`) and routes through a `--cmd` switch
(`report | html | … | context-query | context-diff`, `debug.go:31`).

Rung 6 adds:

- a **new panel** in the existing static HTML report (`--cmd html` →
  `debugHTMLReport` → `im.HTMLReport`, `debug.go:154-155,278`), the
  spine-re-derivation panel below the C11 panels;
- a **sourced text line** in `--cmd report` (`debug.go:152`) and the
  `scoreboard/render.go` text/emoji fallback, for the no-HTML path;
- a `fak prompt show` text alias over the same read, for the live (not
  core-dump) home.

### Panel wireframe

```
+==============================================================================+
| fak debug — system-prompt MMU spine            session: cdb-<id>   [WITNESSED]|
+==============================================================================+
| [S1] LIVE SEGMENT PLAN                              (one row / PromptSegment) |
|   kind        tokens   witness                                               |
|   spine       ▇▇▇      <blob-sha>      (SegStable, byte-identical every turn) |
|   policy      ▇▇       <version>                                             |
|   overlay·a   ▒        <lease-epoch>   resident                              |
|   overlay·b   ░        <lease-epoch>   PAGED                                 |
|   binds: cachemeta.PromptSegment (prefix_stability.go:36)                     |
+------------------------------------------------------------------------------+
| [S2] RESIDENT vs PAGED                          (overlay residency + blast)   |
|   resident ▇▇  paged ░░   evict blast-radius: tokens / dependent-entries      |
|   binds: ctxresidency Snapshot/MeasureBlastRadius (capresidency.go:316/:174)  |
+------------------------------------------------------------------------------+
| [S3] SPINE RE-DERIVATION                       (per-turn; divergence = ALARM) |
|   turn   planned-spine   realized-prefix   max|Δ|   verdict                   |
|   1      <hash>          <hash>            0        GREEN                      |
|   2      <hash>          <hash>            0        GREEN                      |
|   3      <hash>          <hash>            17       RED  seg=policy @byte=842  |
|   binds: Rung-1 plan vs Rung-2 splice (promptmmu.go:133 bytes.Equal floor)    |
|   RED-on-disagree: names the diverging segment + byte offset                  |
+==============================================================================+
| Re-derivation: a turn is GREEN only if planned-spine == realized-prefix (C9)  |
| Fence: this panel is read-only over #1217; its number is NEVER summed into    |
|        the context-safety conservation roll-up (consumes, never re-reports).  |
+==============================================================================+
```

---

## Honest `not yet` gaps (per fence c)

- **`internal/syspromptmmu` does not exist — Rungs 1–5 are planning-only.** The
  parent epic note is explicit ("Planning + tickets only"), so the planned
  segment plan (Rung 1), the splice adapter (Rung 2), the queried overlay (Rung
  3), the residency policy (Rung 4), and the runtime-modification verbs (Rung 5)
  the view consumes are all unbuilt. This note is the **consumer spec**; the view
  cannot render a real turn until Rungs 1–4 land. The seams it binds to
  (`cachemeta.PromptSegment`, `promptmmu`'s `bytes.Equal` floor, `ctxresidency`)
  already exist — the gap is the Rung-1 producer that feeds them as a *spine*.
- **#1217 is itself design-only.** Every C-child of the context-safety epic
  (C1–C14) is a committed spec with "no implementation ships under #1217 until
  the notes are reviewed." So the journal / series this view reads is a specced
  shape (the C6 envelope, the C9 verdict), not a live producer yet — the
  consume-side binding is to a contract, not a running tap.
- **The host wiring is build work, not design.** A new `debug.go` panel and a
  `fak prompt show` verb do not exist today; `debug.go:31`'s `--cmd` switch has
  no `prompt-show` case. Named so the view is not assumed live.

---

## Acceptance check (against #1264)

- **The re-derivation check fails loudly on divergence.** Specced: per turn,
  `bytes.Equal(planned spine, realized wire prefix)` at `max|Δ|=0` (the same
  bit-identity floor as `promptmmu.go:133`); a single diverging turn reds the
  whole series and names the diverging segment + byte offset — catching an
  accidental head mutation before it costs a cold re-prefill. Divergence is an
  **alarm**, never averaged into a metric.
- **The view reads the #1217 journal / series and does not blend its numbers into
  the context-safety roll-up.** Specced as the consume-never-re-report fence: it
  reads the C6 envelope + C9 verdict + the Rung 1–5 data, emits the spine check
  as its **own** WITNESSED panel beside the C11 panels, and never sums it into
  the conservation roll-up.
- **Host is `cmd/fak/debug.go`.** Specced as a new panel on the existing
  `HTMLReport` / `--cmd report` surface (`debug.go:152-155,278`), sibling of the
  C11 context-safety panels, with a `fak prompt show` text alias for the live
  home.

---

_Filed as research / planning only under epic #1258. Parent: #1258
([`SYSTEM-PROMPT-MMU-2026-06-29.md`](SYSTEM-PROMPT-MMU-2026-06-29.md)). Consumes
the C6 schemas ([#1223](https://github.com/anthony-chaudhary/fak/issues/1223)),
the C9 re-derivation discipline
([#1227](https://github.com/anthony-chaudhary/fak/issues/1227)), and the C11
debug-window home ([#1229](https://github.com/anthony-chaudhary/fak/issues/1229))
of the context-safety epic
[#1217](https://github.com/anthony-chaudhary/fak/issues/1217); feeds #1217.
Design-only — no implementation ships under #1264 until Rungs 1–4 land._
