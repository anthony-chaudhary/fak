---
title: "concept(dos): the readiness / durability verdict ladder as a domain-free DOS primitive (issue #582)"
description: "Concept capture for #582: the mechanism behind the product scorecard (tools/product_scorecard.py) is not fak-specific. It is a generic DOS primitive — a closed maturity ladder where each rung's promotion is gated by evidence the claimant did not author, plus a surface-ceiling gate that stops a benchmark from posing as a durable product. This note (1) extracts the surface-gate + evidence-binding core into a domain-free form, (2) sketches a dos-grounded variant where each rung is closed by a dos witness (dos_verify / dos_commit_audit) and an overclaim is refused with a structured reason from a closed vocabulary (cf. dos_refuse_reasons), and (3) decides where it should live: keep the tree-cross-check shell as a fak tool, lift the {closed ladder + surface ceiling + evidence-bound adjudication + overclaim-refusal} core into the DOS concept vocabulary. No code commitment — the issue asked for concept capture only."
---

# The readiness / durability verdict ladder — a domain-free DOS primitive

> Captures the concept the product-scorecard work surfaced in
> [#582](https://github.com/anthony-chaudhary/fak/issues/582)
> (labels: documentation, enhancement, research). The issue's ask is
> **concept capture, no code commitment** — extract the generic primitive
> behind [`tools/product_scorecard.py`](../../tools/product_scorecard.py) /
> [`docs/PRODUCT-STATUS.md`](../PRODUCT-STATUS.md), sketch a `dos`-grounded
> variant, and decide where it should live. This note delivers all three.
>
> **Decision (§4): keep the tree-cross-check *shell* as a fak tool; lift the
> domain-free *core* — a closed maturity ladder + a surface-ceiling gate +
> evidence-bound adjudication + structured overclaim-refusal — into the DOS
> concept vocabulary, where its natural evidence binding is a `dos` witness
> (`dos_verify` / `dos_commit_audit`) rather than the tool's own
> `os.path.exists` checks.** Building a `dos readiness` verb is a dos-kernel
> change, out of this repo's tree, and is named as the next step — not done here.

## 1. What the product scorecard actually does

`tools/product_scorecard.py` answers one human question of a project: *of the
concepts you ship, which are a durable, real, useful-today product — and which
are still a named gap, a research seam, or an overclaim?* It places each concept
on a closed verdict ladder (best → worst):

```text
durable-product → usable-today → real-not-easy → honest-stub → concept-only
```

The load-bearing property — the reason #582 calls it a *DOS* mechanism and not
just a reporting tool — is that **no rung can be reached by editing the claim**.
Each promotion is gated by an artifact the author of the row did not write. From
`tools/product_scorecard.py` (`expected_verdict`, the KPIs, and the `SURFACES`
gate), the evidence behind each verdict is, concretely:

| Gate | Evidence the claimant did NOT author | Where it is read |
|---|---|---|
| Is it real / shipped? | the maturity tag the concept carries in `CLAIMS.md` (membership, not the row's own lead line) | `kpi_claim_honest` cross-checks the row's `claims_tag` against the real `CLAIMS.md` section |
| Is there a command a person runs? | the first command resolves to a real `cmd/` dir **and** a documented `fak` verb | `kpi_command_resolves` against the tree + `docs/cli-reference.md` |
| Is it proven? | the witness test / results doc exists on disk | `kpi_witnessed` — `os.path.exists` on `witness_path` |
| Is it discoverable? | the entry doc exists on disk | `kpi_discoverable` — `os.path.exists` on `entry_doc` |
| Is the stated verdict honest? | the declared verdict equals the verdict the evidence *implies* | `kpi_verdict_consistency` — `declared == expected_verdict(row)` |

On top of those evidence gates sits a **surface-ceiling gate** — the single most
important structural rule. A concept declares a *surface class*, and the class
caps the best rung the concept can reach no matter how much evidence it has:

```text
SURFACES (from product_scorecard.py)         ceiling
  product    a surface a person runs          → durable-product   (full ladder)
  benchmark  a demo/measurement you run        → usable-today      (capped)
  subsystem  an internal mechanism, no surface → real-not-easy     (capped)
  seam       a frozen ABI awaiting a backend   → real-not-easy     (capped)
```

This is what stops a benchmark from posing as a durable product, and a stub from
posing as shipped. To move a verdict you must change the **real tree**, not the
data file — exactly the "cannot be gamed by editing the data" invariant.

## 2. The domain-free core (deliverable 1: extract the primitive)

Strip away `CLAIMS.md`, `cmd/`, and the `fak` verbs and a domain-free primitive
remains. Call it a **readiness verdict ladder**. It has five parts, none of them
fak-specific:

1. **A closed rung vocabulary** `R = [r₀ … rₙ]`, a *total order* best → worst.
   Closed means: a verdict outside `R` is not a lower score, it is a bug — the
   same discipline `dos_check_reason` applies to a refusal reason outside the
   closed set (`UNCLASSIFIED` prose-drift, refused conservatively).

2. **A class taxonomy** `S` with a **ceiling function** `cap : S → R`. Each
   capability declares one class; `cap` is the best rung that class may reach.
   This is the surface gate, generalized: a *measurement* class tops out below a
   *product* class; an *internal mechanism* class can never present as a finished
   surface. The ceiling is a property of the **kind of thing**, decided before
   any evidence is weighed.

3. **A capability set** `C = {cᵢ}`. Each `cᵢ` carries a *declared* maturity, a
   *declared* class, a *declared* rung, and a set of **evidence pointers**
   (paths, commands, tags) — all author-supplied and therefore all untrusted.

4. **Evidence predicates** `P`. Each `p ∈ P` is a function over the *real
   environment*, never over the claim: `member(tag, closed_catalog)`,
   `resolves(command)`, `exists(path)`. The defining constraint: every predicate
   reads ground truth the claimant could not fabricate by editing `cᵢ`.

5. **An adjudication function** that produces an *evidence-implied* verdict:

   ```text
   adjudicate(c):
     # rungs below "shipped" are decided by the declared maturity alone
     if maturity(c) is a roadmap idea        → return ladder.bottom    # concept-only
     if maturity(c) is a labeled stub/sim     → return honest-stub      # an honest "not yet"
     # shipped: evidence + the surface ceiling decide the ceiling
     best := highest rung r in R such that every promotion predicate up to r holds
     return min(best, cap(class(c)))          # the surface ceiling caps it
   ```

   The **verdict** of `c` is `adjudicate(c)`. The *declared* rung is then checked
   against it: `declared(c) > adjudicate(c)` is an **overclaim** — and an
   overclaim is **refused**, not merely scored down. (`durable-product` on a stub,
   or on a shipped concept with no runnable command, is the canonical refusal —
   the single most valuable thing the mechanism catches.)

That five-part shape is the whole primitive. The fak product scorecard is exactly
*one binding* of it: `R` = the five product verdicts, `S` = the four surfaces,
`P` = {`CLAIMS.md` tag membership, `cmd/`+verb resolution, witness/entry
existence}. Any agent fleet making product / readiness / maturity claims about a
tree could instantiate the same primitive against its own ground truth.

### Why this is DOS-shaped, not fak-shaped

It is the same three properties as the rest of the kernel (the issue's own case):

- **A closed verdict vocabulary** — cf. `dos_refuse_reasons`' closed reason set.
  A rung outside `R` is the `UNCLASSIFIED` drift `dos_check_reason` exists to kill.
- **Verdicts grounded in evidence the claimant did not author** — cf. `dos_verify`
  reading git ancestry instead of a worker's self-report. The product ladder reads
  the tree; the generalized ladder reads whatever ground truth the host can witness.
- **Fail-closed / no-overclaim** — a top-rung claim is *refused* unless every gate
  checks out. The default on missing evidence is the lower rung, never the higher.

## 3. The `dos`-grounded variant (deliverable 2: close the verdict with a witness)

In the fak tool the evidence predicates are local existence checks
(`os.path.exists`, a `cmd/` dir lookup). A local `exists` answers *"is there a
file here now"* — it does **not** answer *"did this actually ship"*. The
`dos`-grounded variant swaps each predicate for a **kernel witness**, so the top
of the ladder is closed by evidence with the strongest possible provenance:

| Ladder gate | fak tool predicate (local) | `dos`-grounded predicate (witness) |
|---|---|---|
| Is it shipped? | the `CLAIMS.md` tag says so | **`dos_verify(plan, phase)`** — reads a run-registry row or git ship-commit grammar; `shipped` comes from git, never the row |
| Did the commit that claims it actually do it? | — (not checked) | **`dos_commit_audit(ref)`** — `diff-witnessed` (the diff did the kind of thing the subject claims) vs `subject-only` (forgeable message) |
| Is the rung in vocabulary? | `verdict in VERDICTS` | **`dos_check_reason`-style** membership in a closed set; an out-of-set rung is `UNCLASSIFIED`, refused conservatively |
| Is an overclaim refused? | the row is flagged a defect | the refusal is **emitted as a structured reason** from a closed vocabulary (cf. `dos_refuse_reasons`) — emittable, verifiable, refusable — not free-text prose |

Sketch of the adjudication with the witness binding:

```text
adjudicate_dos(c):
  if maturity(c) is roadmap/stub  → low rung as before (no shipped claim to witness)
  shipped := dos_verify(plan(c), phase(c)).shipped          # git evidence, not the claim
  honest  := dos_commit_audit(ship_ref(c)).witness == "diff-witnessed"
  best := highest r in R such that the promotion predicate up to r holds,
          where the "is it shipped/real" predicate is (shipped AND honest)
  verdict := min(best, cap(class(c)))
  if declared(c) > verdict:
      refuse(reason = READINESS_OVERCLAIM,                  # a closed, verifiable reason
             evidence = {shipped, honest, cap(class(c))})   # the deciding facts, not prose
  return verdict
```

The shape is identical to the product scorecard; only the *provenance of the
evidence* is upgraded from "a file exists" to "the kernel witnessed it from git."
That upgrade is precisely the `dos-skillify` move the issue cites — swap a
self-certified claim for a `dos` witness check — applied to the readiness ladder.

A new structured reason, `READINESS_OVERCLAIM` (category `STALE_CLAIM` /
`UNCLASSIFIED`), is the natural closed-vocabulary token for "a capability claims a
rung its evidence does not support." It is **proposed, not added** — minting a
reason in the kernel's closed set is a dos-kernel change, deferred with the rest
of §4's next step.

## 4. Where it should live (deliverable 3: the decision)

The mechanism cleanly factors into two layers, and they belong in two different
places:

- **The domain-free core** — `{closed rung ladder, surface-ceiling `cap`,
  evidence-bound `adjudicate`, structured overclaim-refusal}` — is genuinely a
  **DOS primitive**. It names no fak path and carries the kernel's three
  invariants exactly. This is what belongs in the **DOS concept vocabulary**,
  alongside `dos_verify` (witnessed completion) and `dos_refuse_reasons` (closed
  refusal set), as a *readiness verdict ladder* concept whose top rung is closed
  by a `dos` witness.

- **The tree-cross-check shell** — reading `CLAIMS.md`, resolving `cmd/` dirs and
  `fak` verbs, rendering `docs/PRODUCT-STATUS.md`, driving `product-debt` into the
  scorecard control pane — is **inherently fak-specific** and should **stay a fak
  tool**. It is one binding of the primitive to one tree's ground truth; other
  fleets bind their own.

This is the same split the issue proposed in its own words — *"the category
vocabulary is already data-defined; the tree cross-check is the fak-specific shell
to generalize."* The data-defined vocabulary (the `_meta.json` category list, the
`SURFACES` map, the `VERDICTS` ladder) is already the domain-free core sitting
inside a fak-specific shell. The generalization is therefore **documentation +
one kernel concept**, not a rewrite of the tool.

### Concretely, the next step (named, not committed here)

1. **This note** is the concept-capture artifact #582 asked for.
2. A `dos`-kernel follow-up (cross-repo, out of this tree) would add a
   `readiness verdict ladder` concept page and, if it earns its keep, a thin
   `dos readiness` adjudicator that takes a host-supplied `{ladder, cap, capabilities}`
   and closes the "is it shipped" rung with `dos_verify` / `dos_commit_audit`,
   refusing an overclaim with the proposed `READINESS_OVERCLAIM` reason.
3. fak's `tools/product_scorecard.py` stays the reference *binding* — the worked
   example a new host forks, the same way `dos-skillify` is the worked example for
   witness-gated skills.

No change is made to `tools/product_scorecard.py` or the kernel here: the issue
scoped this to capture, and the kernel surface is a different repository's tree.

## 5. Honest scope of this resolution

- **Deliverable:** a concept-capture note (this file) + its INDEX entry. The issue
  is labeled documentation / enhancement / research and explicitly says *"concept
  capture — no code commitment yet."* The smallest correct change that closes it is
  the captured concept, not new code.
- **Gate run:** the docs gate the repo enforces for a new note —
  `tools/check_doc_placement.py`, `tools/check_index_sync.py`, `tools/check_links.py`.
  There is no Go/test gate to run because there is no code change (and the issue
  forbids one).
- **What remains (out of this repo's tree):** the optional `dos`-kernel concept
  page + `dos readiness` verb and the `READINESS_OVERCLAIM` reason. Those are
  dos-kernel changes; this note names them as the next step so the kernel side can
  pick them up with full context.

## See also

- [`tools/product_scorecard.py`](../../tools/product_scorecard.py) — the reference binding (the fak-specific shell).
- [`docs/PRODUCT-STATUS.md`](../PRODUCT-STATUS.md) — the rendered product-status page the ladder produces.
- [`/scorecard`](../../.claude/skills) — the generic scoring doctrine every fak scorecard instantiates; the readiness ladder is the same machine pointed at "is this a durable product."
- DOS kernel concepts this generalizes alongside: `dos_verify` (witnessed completion from git), `dos_refuse_reasons` / `dos_check_reason` (closed, verifiable refusal vocabulary), `dos_commit_audit` (a commit's claim vs its diff).
