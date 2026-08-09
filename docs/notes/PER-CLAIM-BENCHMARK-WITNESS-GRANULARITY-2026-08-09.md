---
title: "Per-claim benchmark witness granularity"
description: "Decision for #3431: keep benchmark rows as compatibility envelopes, but decompose mixed rows into stable assertions with their own status and witness kind so one measured sentence cannot promote a functional, simulated, modeled, or retracted sibling."
---

# Per-claim benchmark witness granularity — decision for #3431

**Verdict: adopt per-claim witness granularity for mixed-witness authority
rows, as an optional additive field.** Keep the existing row-level `status` and
`provenance` fields as a conservative compatibility envelope; do not require
decomposition for a row that contains one independently citable assertion.

This is a schema decision, not a migration. No registry row, authority renderer,
runtime path, or CI gate changes in this issue.

## Why the scalar row is insufficient

The generation-aware authority view currently classifies one registry row from
one `status` and one `provenance` value. The executable fold states its known
limitation directly: the coarse enum cannot represent the
`gen/second-next` simulation/fixture rung, so
[`AuthorityClaim.EntitledHorizon`](../../internal/bench/authorityview.go) never
emits `second-next`.

The typed authority model is scalar too:
[`benchauthority.Claim`](../../internal/benchauthority/benchauthority.go) has one
`Status` and one `Provenance` for the complete headline. That shape is honest
only while a row is one citable claim.

The current cache-savings row is the counterexample. The
[`compaction-shed-per-fire` registry record](../benchmarks/registry.jsonl) and
its rendered row in
[`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md) carry at least three
independently citable assertions:

| Stable assertion | Current evidence | Correct claim-level classification |
|---|---|---|
| `shed-per-fire` — the longest 2026-07-06 session shed about `106,708 tokens/fire` across seven real fires | committed real-traffic ledger; reproduce by folding `compaction_shed_tokens/compaction_fired` | `status: live`, `witness_kind: measured`, entitled `gen/now` |
| `fleet-token-equiv-share` — fak-authored share is about `0.3–16%`, with 2026-W28 pooled at about `16.4%` | observed provider/fak inputs passed through the deterministic proportional valuation in [`cacheprice.ShedTokenEquiv`](../../internal/cacheprice/cacheprice.go); it is a counterfactual token-equivalent fold, not a directly metered provider quantity | `status: live`, `witness_kind: functional`, entitled `gen/next` |
| `retracted-session-share` — the former per-session `~15%→~75%` share | disproved by the cumulative-across-fires accounting analysis retained in the row's retraction and fences | `status: retracted`, no live entitled horizon |

The row-level record says `status: live`, `provenance: measured`. That is correct
for `shed-per-fire`, too strong for the functional token-equivalent share, and
incapable of expressing the retracted sibling as a tombstone. A citing agent can
therefore inherit the strongest witness in the row and silently apply it to every
sentence in the headline.

The failure is not that the prose lacks fences. The fences are unusually strong.
The failure is that the machine-readable witness is attached one level above the
claims those fences distinguish.

## Decision boundary

Decompose a row when **either** condition is true:

1. two sentences in the headline or fences can be cited independently and clear
   different witness kinds or statuses; or
2. one sentence remains live while a sibling is gated, stale, superseded, or
   retracted.

Do not decompose merely because a row has several metrics from the same artifact
and all of them share one witness. A decode/prefill pair produced by the same
measured run is still one scalar-witness row unless its claims acquire different
status or provenance.

This keeps per-claim metadata exceptional and evidence-driven rather than turning
every benchmark row into an unbounded document tree.

## Additive schema

Add an optional `assertions` array to a registry row. Existing scalar fields
remain present.

```json
{
  "id": "compaction-shed-per-fire",
  "status": "live",
  "provenance": "functional",
  "assertions": [
    {
      "id": "shed-per-fire",
      "statement": "~106708 tokens shed per compaction fire on the pinned session",
      "status": "live",
      "witness_kind": "measured",
      "metric": "compaction_shed_tokens_per_fire",
      "value": "106708",
      "unit": "tokens/fire",
      "artifact": "docs/nightrun/cache-savings.jsonl",
      "reproduce": "fold compaction_shed_tokens/compaction_fired for the pinned session"
    },
    {
      "id": "fleet-token-equiv-share",
      "statement": "fak-authored share is ~0.3-16%; 2026-W28 pooled ~16.4%",
      "status": "live",
      "witness_kind": "functional",
      "artifact": "docs/nightrun/cache-savings.jsonl",
      "reproduce": "fak cachevalue report"
    },
    {
      "id": "retracted-session-share",
      "statement": "per-session fak share rose from ~15% to ~75%",
      "status": "retracted",
      "witness_kind": "unknown",
      "replacement": "fleet-token-equiv-share"
    }
  ]
}
```

### Closed claim-level witness vocabulary

| `witness_kind` | Entitled horizon | Meaning |
|---|---|---|
| `measured` | `gen/now` | captured real-workload number with its baseline, artifact, and reproduce handle |
| `functional` | `gen/next` | real behavior or deterministic derived fold, but not a captured default-grade measurement for the exact claim |
| `simulated` | `gen/second-next` | fixture, simulation, or micro-measurement with the counterfactual named |
| `modeled` | `gen/future` | analytic geometry, projection, or formula whose inputs are stated |
| `unknown` | `gen/future` | no stronger witness has been resolved |

`simulated` is intentionally claim-level even though the current row-level
registry does not carry it. This is the missing distinction that recovers the
authority view's collapsed `gen/second-next` rung instead of continuing to fold
fixtures into `functional`/`gen/next` or `modeled`/`gen/future`.

Claim-level `status` reuses the existing closed vocabulary:
`canonical | live | gated | pending | stale | retracted`. Status overrides the
live horizon for tombstones and withheld claims exactly as it does today.

### Compatibility roll-up

Scalar consumers must fail closed, not guess which assertion was intended.
For a decomposed row:

1. Exclude `stale` and `retracted` assertions from the live witness roll-up, but
   keep them rendered as tombstones.
2. Set row-level `provenance` to the **least-entitling live
   `witness_kind`** (`unknown ≡ modeled < simulated < functional < measured`;
   `unknown` and `modeled` are tied because both entitle only `gen/future`).
3. Set row-level `status` to `canonical` only if every live assertion is
   canonical and entitled to `gen/now`; otherwise use `live` when any live
   assertion exists, `gated`/`pending` when all unresolved assertions are
   withheld, and a tombstone status when none remain live.
4. Assertion-aware consumers cite `row-id/assertion-id` and derive the horizon
   from that assertion only.

For the worked row, the compatibility envelope becomes `live / functional`.
That under-claims the measured per-fire assertion to scalar readers, while the
assertion-aware view recovers its `gen/now` entitlement. Conservative loss of
precision is preferable to promoting the functional share to `gen/now`.

The JSONL change is additive: the current subset loader in
[`LoadAuthorityRegistry`](../../internal/bench/authorityview.go) ignores fields
it does not read. A future typed transcription must add the same optional shape
to `benchauthority.Claim`; it must not maintain a second incompatible schema.

## Assumptions

1. Independently citable assertions can receive stable IDs that survive prose
   edits.
2. Existing consumers can continue reading the conservative scalar envelope
   while assertion-aware consumers are introduced.
3. Witness kind plus status is sufficient to derive entitled horizon; regime,
   model, and baseline remain row-level unless a real mixed row proves otherwise.
4. The cache-savings row is not the only long-lived mixed-witness row. It is the
   first proven case and the migration can census before expanding.

## Kill criteria

Retire this design in favor of separate flat rows if any of these is observed:

- a census finds no second mixed-witness row and the cache-savings row can be
  split cleanly without losing its shared fences;
- stable assertion IDs churn whenever headline prose changes;
- the conservative row roll-up breaks a current consumer that cannot be moved
  to assertion-aware reads without changing the public contract; or
- an assertion needs its own model, regime, baseline, and artifact often enough
  that it has become a row in everything but name.

If those conditions do not fire, nested assertions are cheaper than duplicating
the cache-savings row's long shared provenance and retraction history across
three flat rows.

## Implementation handoff

The implementation leaf should be additive and ordered:

1. census `docs/benchmarks/registry.jsonl` for rows whose headline/fences mix
   live and retired claims or more than one witness kind;
2. add the optional assertion type and conservative roll-up to the single
   registry authority seam;
3. migrate `compaction-shed-per-fire` as the first fixture;
4. update the generation-aware authority view so its histogram can emit
   `second-next` from `witness_kind: simulated`; and
5. add a compatibility test proving scalar consumers receive the conservative
   row envelope while assertion-aware consumers recover the measured,
   functional, simulated, and retracted children separately.

Do not wire a refusing CI gate in that leaf. Derive and render the new field
first; refusal remains a later, independently witnessed promotion.

## Decision witness

This note is linked from the generation authority memo, names the current
cache-savings mixed row, provides the schema that preserves scalar consumers,
and explicitly recovers the collapsed `gen/second-next` rung. The repository
link/placement gates are the acceptance witness for this note-only issue.
