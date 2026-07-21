# RESEARCH — Org-policy precedence lattice (R3 / #5318)

**Parent epic:** #5315 (Org Policy Plane) · **Grandparent:** #5170 (Policy Amendment
Classes) · **Lane:** research (decision spike) · **Class:** research · **Date:** 2026-07-20

This note formalizes the precedence lattice that reconciles the two forces IT wants
at once — *tighten a fleet everywhere* AND *enable more usage on enrolled boxes* —
against the adjudicator's "most-restrictive-wins" fold and #5170's rule that no channel
weakens a FROZEN knob. It fixes **one** precedence model (not two) and hands W4 an
executable spec via the failing-first matrix stub at
`internal/policy/org_precedence_test.go` (turned live by #5322).

Vocabulary is taken verbatim from `internal/policy/amendment.go`: the amendment
**classes** `FROZEN` / `RATCHET` / `GATED_WIDEN` / `SELF_AMENDABLE`
(`AmendmentClass`), the **directions** `frozen` / `tighten-only` / `widen-only`
(`AmendmentDirection`), and the **channels** `compiled-in` / `operator-overlay` /
`live-reload` / `operator-escalation`. This epic adds one channel — `central` (org
policy) — and this note says exactly where it sits.

---

## 1. The lattice

Authority, most-authoritative first:

```
compiled-in FROZEN floor   (ChannelCompiledIn — a code change + release)
        >  central org overlay      (new: signed fak-org-policy/v1 envelope, out-of-band)
        >  operator overlay         (ChannelOperatorOverlay — .fak/guard/{allow,deny}.json, --policy)
        >  agent-self               (SELF_AMENDABLE — closed today; the wrapped agent)
```

`central` slots **below** the compiled-in FROZEN floor and **above** the on-disk
operator overlay. It is an out-of-band gated channel the wrapped agent cannot reach —
in the same trust tier as `live-reload` / `operator-escalation` for *who may drive it*,
but ordered **above** `operator-overlay` for *whose ceiling caps whose*.

Two flows travel this lattice in **opposite directions**, and keeping them separate is
the whole trick:

- **Tighten (RATCHET) flows DOWN, unopposed and cumulative.** Any channel may add a
  restriction; a lower channel may add more; none may subtract one. The resolved posture
  is the **union of every restriction / the most-restrictive value** across all channels.
- **Widen (GATED_WIDEN) is a descending CEILING.** The compiled-in FROZEN cap is the
  hard maximum. `central` may grant a ceiling **at or below** that cap. `operator` may
  only lower **at or below** the central grant. Agent-self may not widen at all. The
  resolved ceiling is the **minimum** (most restrictive) of the caps that apply.

> **One-line precedence rule:** *Tighten flows down the lattice unopposed (most-restrictive
> wins); widen is a ceiling each lower channel may only push lower — never past the channel
> above it — and the compiled-in FROZEN floor caps every central grant.*

Corollary — the two directions collapse to a **single fold**: for every knob the resolved
value is the **most-restrictive** contribution across `{compiled-in, central, operator,
agent-self}`, where a RATCHET contribution is a floor (union / max-restriction) and a
GATED_WIDEN contribution is a ceiling (min). There is no channel-precedence *override*
step; authority order only decides *whose widen ceiling bounds whose*. That is why a
higher channel can never re-loosen what a lower channel tightened.

---

## 2. Per-amendment-class truth tables

Inputs are the contributions of the three reachable channels — **compiled-in**,
**central**, **operator** — and the output is the **resolved verdict** the adjudicator
fold produces. `agent-self` is omitted from the input columns because it is closed
(SELF_AMENDABLE is empty today); it can contribute nothing, so it never changes a row.

### 2.1 RATCHET (tighten-only) — e.g. `Deny`, `EgressBlockHosts`, `SecretPatterns`

Model a boolean tool that is admitted unless some channel denies it. `deny` = the channel
adds a refusal; `.` = the channel contributes nothing; `widen?` = the channel *attempts*
to remove/loosen a refusal (which the direction forbids).

| compiled-in | central | operator | resolved | why |
|---|---|---|---|---|
| allow | `.` | `.` | **ALLOW** | nobody restricted |
| allow | deny | `.` | **DENY** | central tightened fleet-wide (honored by the fold) |
| allow | `.` | deny | **DENY** | operator tightened locally (a further floor) |
| allow | deny | deny | **DENY** | both tighten; union is still deny |
| allow | deny | widen? | **DENY** | operator **cannot** un-deny central's refusal |
| allow | widen? | `.` | **ALLOW** | there was nothing to widen; RATCHET has no loosen path anyway |
| deny | `.` | `.` | **DENY** | compiled-in floor already denied |
| deny | widen? | widen? | **DENY** | no channel weakens a RATCHET/FROZEN floor |

Rule: **resolved = DENY iff ANY channel denies.** No channel subtracts a peer's deny.

### 2.2 GATED_WIDEN (widen-only ceiling) — e.g. a `rate_limit` cap, a cache/model tier unlock

Model a numeric cap where **higher = more permissive**. `C` = compiled-in FROZEN hard
maximum. `G` = central grant. `O` = operator setting. `—` = channel leaves it at inherit
(no further constraint). Resolved ceiling = **min of the caps that apply**, and any grant
above the channel above it is **clamped** (never honored as a widen).

| compiled-in cap `C` | central `G` | operator `O` | resolved cap | why |
|---|---|---|---|---|
| 200 | — | — | **200** | no widen requested; sits at the compiled floor's own default¹ |
| 200 | 100 | — | **100** | central grants a ceiling ≤ C; enrolled boxes get 100 (IT enable-more) |
| 200 | 100 | 50 | **50** | operator tightens **below** the central grant (a further floor) |
| 200 | 100 | 150 | **100** | operator **cannot widen past** central grant → clamped to 100 |
| 200 | 300 | — | **200** | central **cannot exceed** the FROZEN cap → clamped to 200 |
| 200 | 100 | — → later `G`=180 | **100** if operator had set 100 | central **cannot raise a cap the operator lowered** (min wins) |
| 200 | — | 50 | **50** | operator may tighten even with no central grant |

¹ For a widen-only knob the zero/inherit value is already the tightest posture that the
compiled floor ships; the number `200` here is illustrative of the FROZEN cap, not a
default grant. Un-enrolled `fak` has no `central` column at all and behaves as today.

Rule: **resolved = min(C, G, O)** over the caps that are set, with every set value
clamped to `≤` the value of the channel immediately above it before it enters the min.
Equivalently: a widen is only ever a **ceiling that descends**; it can lower, never raise.

### 2.3 FROZEN — e.g. the egress SSRF floor, the reversibility gate, structural danger rules

| compiled-in | central | operator | resolved | why |
|---|---|---|---|---|
| floor | `.` | `.` | **FLOOR** | the shipped value |
| floor | tighten? | `.` | **FLOOR** | FROZEN does not move — even a tighten is a no-op on the floor value² |
| floor | widen? | `.` | **FLOOR** | central cannot weaken a FROZEN knob (#5170 invariant) |
| floor | `.` | widen? | **FLOOR** | operator cannot weaken it either |
| floor | widen? | widen? | **FLOOR** | no channel, in any combination, moves a FROZEN knob |

² A FROZEN knob's only authorized channel is `compiled-in` (see `amendment.go`:
`FROZEN knobs may declare only ChannelCompiledIn`). Fleet-wide *tightening* IT wants is
expressed through **RATCHET** knobs (add a deny, union a secret pattern), never by
mutating a FROZEN one. So `central` never appears as an authorized channel on a FROZEN
row; the table shows attempts precisely to assert they are inert.

### 2.4 SELF_AMENDABLE — n/a

Empty today. The `agent-self` channel is closed: it contributes nothing to any fold, so
it appears in no input column and cannot change any resolved verdict. Listed for
completeness only.

---

## 3. The two load-bearing questions (both answered: **no**)

**Q1. Can `central` raise a cap the operator lowered?** — **No.**
Under GATED_WIDEN the resolved ceiling is `min(C, G, O)`. If the operator lowered the cap
to `O`, then for any central grant `G > O` the min is still `O`. Authority order places
`central` above `operator`, but that ordering only bounds *whose grant caps whose ceiling*
— it is **not** an override that lets the higher channel re-raise the lower channel's
tightening. A lower channel's floor/tighten is always honored by the most-restrictive
fold. (Row 6 of §2.2.)

**Q2. Can the operator widen past a central grant?** — **No.**
The operator overlay is a *further floor*, direction `tighten-only` relative to the
grant above it: it may set `O ≤ G` (lower the ceiling) but any `O > G` is clamped to `G`
before the min. So the operator can only ever end at or below the central grant, never
above it. (Row 4 of §2.2.)

Symmetry: tightening always wins regardless of which channel asks (Q1 — a lower channel's
tighten beats a higher channel's widen); widening is always bounded by the channel above
(Q2 — a lower channel cannot exceed a higher channel's ceiling). That symmetry is the
single invariant the matrix test enumerates.

---

## 4. Reconciliation with #5170 Track D — ONE precedence model

#5170 Track D ("GATED-WIDEN safety rails") calls for *journaled capability-grant
provenance, TTL / expiring operator widenings, and **explicit per-scope precedence in
`policy explain`***. This note does **not** introduce a second precedence system; it
**extends** Track D's single model by:

1. **Inserting one channel** (`central`) into the existing authority order, between
   `compiled-in` and `operator-overlay`. Every amendment-class *direction* rule from
   `amendment.go` is unchanged: RATCHET still tightens-only, GATED_WIDEN still
   widens-only via a gated out-of-band channel, FROZEN still moves only via compiled-in.
   `central` is simply another out-of-band gated channel (peer to `live-reload` /
   `operator-escalation` in trust, ordered above `operator-overlay` in ceiling scope).

2. **Reusing Track D's per-scope precedence render.** The `policy explain` output Track D
   defines — which channel set which knob, with provenance — gains one more channel row
   (`central`, with issuer + envelope staleness from #5315's `fak org status`). No new
   resolution algorithm: the fold is still the most-restrictive combination, and
   precedence order only decides the widen ceiling. Track D's provenance/TTL journaling
   applies to `central` widenings exactly as to operator widenings (every central widen
   is journaled with issuer + envelope provenance, per #5315's invariants).

3. **Keeping the fold identical for enrolled and un-enrolled `fak`.** With no `central`
   manifest, the `central` column is absent and the tables in §2 degenerate to today's
   two-channel (compiled-in, operator) behavior byte-for-byte — the #5315 opt-in
   invariant. A stale / expired / bad-signature central manifest contributes **nothing to
   the widen ceiling** (refuse-to-widen → falls back to the compiled-in floor), while any
   RATCHET tightening it last delivered may still be honored per cache policy; it never
   fails open.

**Net:** one lattice, one fold, one `policy explain`. Track D owns the render and the
provenance/TTL rails; this epic contributes exactly one channel and its ceiling position.

---

## 5. What W4 must build (and what #5322 turns green)

- **W4 (#5315):** a precedence fold that, per knob, folds `{compiled-in, central,
  operator, agent-self}` into a resolved value using the §1 rule (union/max for RATCHET
  floors, min for GATED_WIDEN ceilings, identity for FROZEN), plus the `central` row in
  `fak org status` / `policy explain`.
- **#5322:** deletes the `t.Skip` guards in `internal/policy/org_precedence_test.go` and
  wires the enumerated cases to the real fold entrypoint, so §2's truth tables ship as an
  executable spec (the matrix stub already enumerates every row above).

### Invariants the fold must preserve (red-team matrix)
- No channel — `central` included — weakens a FROZEN knob.
- A RATCHET knob only ever tightens; the resolved value is the most-restrictive
  contribution across all channels.
- A GATED_WIDEN resolved ceiling never exceeds the channel above it, and never the
  compiled-in FROZEN cap.
- `central` raising a cap the operator lowered → **no-op** (Q1).
- `operator` widening past a central grant → **clamped** (Q2).
- Un-enrolled behavior is byte-for-byte unchanged (no `central` column).
