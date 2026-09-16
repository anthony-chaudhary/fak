# Adding the next protocol

**The one file a new protocol implements.** A protocol joins the queried skill loader by
implementing exactly one interface — `capindex.Resolver` in `internal/capindex/capindex.go`:

```go
type Resolver interface {
    Index() []CapCard             // cheap cards only — the at-rest cost
    Fault(ref CapRef) (Capability, error) // page in the full body on demand
}
```

That is the whole on-ramp. The query, the index, the residency, the eviction, and the
versioning are written once over `Capability`; a new protocol inherits them. See
[`SKILL-LOADER-QUERY-EPIC.md`](SKILL-LOADER-QUERY-EPIC.md) (issue #1108) for the design.

## The two methods

- **`Index()` returns cheap cards, never bodies.** One `CapCard` per capability, carrying a
  name, a digest, a trigger clause, and tags. Keep it O(cards): do **not** read a body (an
  input schema, a SKILL.md, a tool list) here, or every at-rest session pays O(bodies) and
  the "0-for-∞" property is lost. `internal/capindexgw/mcp_resolver.go` documents this at
  its `Index()`.
- **`Fault(ref)` pages in the full body on demand.** This is the protocol's "get": the
  complete invocation contract for one capability — enough for a caller to actually invoke
  it, not just its label. Return `capindex.ErrKindMismatch` for a foreign `ref.Kind` and
  `capindex.ErrNotFound` for an unknown name.

## Where to put the file

Put the resolver in `internal/<proto>_resolver.go` when the protocol's source types live in
`internal/` at a tier `capindex` may import. If it must import a **higher** tier, put it in a
new adapter package **above** `capindex` instead.

**The tier rule: an adapter sits at the higher of the two tiers it bridges.** `capindex` is
tier-3; `internal/gateway` is tier-5. A resolver coupling them cannot live in `capindex` or
it would pin the whole keystone up to tier-5 — so it lives in `internal/capindexgw`
(tier-5), the higher of the two. `internal/architest` machine-checks this; a mis-tiered
package fails CI. Read the numbers off `internal/architest/architest_test.go`, not from
memory — the table is the single source of truth.

## Wiring it into the query front-end (C2)

A resolver becomes queryable through `internal/contextq` by wrapping it, not by editing the
query code:

```go
res := []contextq.Resolver{ contextq.NewCapIndexResolver(myResolver) }
got := contextq.QueryCapabilities(res, contextq.CapQueryRequest{Intent: "...", BudgetBytes: n}, nil)
```

`contextq.NewCapIndexResolver` translates a `capindex.Resolver` onto the C2 seam, so the
C2/C3/C4 code stays unchanged. The same wrapper carries the residency ledger for free.

## Worked example

`internal/capindexgw` holds `mcp_resolver.go` and `a2a_resolver.go` — two protocols behind
the same seam. The A2A one is the fuller template: its `Index()` emits name+scope+description
cards, and its `Fault()` pages in the method's full invocation contract (`method`,
`transport`, `input_schema`). Read `a2a_resolver.go` first; copy its shape.