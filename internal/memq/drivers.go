package memq

import "sort"

// Params parameterize a driver's compiled query: the agent's intent (for relevance),
// an optional K (limit), an optional byte Budget, and a Reason stamped on suppressions.
type Params struct {
	Intent string `json:"intent,omitempty"`
	K      int    `json:"k,omitempty"`
	Budget int64  `json:"budget,omitempty"`
	Reason string `json:"reason,omitempty"`
}

// Driver is a NAMED, pre-composed memory strategy — a "canned query" in the algebra.
// The whole point of memq is that a Driver is a few lines of Ops, not a bespoke Go
// function: recall, render, clean, compact, and dream below are five sentences in one
// language, and an agent (or a plugin) authors a sixth by emitting its own Query — no
// kernel edit. Build is pure and deterministic.
type Driver struct {
	Name  string             `json:"name"`
	Doc   string             `json:"doc"`
	Build func(Params) Query `json:"-"`
}

var registry = map[string]Driver{}

// Register adds (or replaces) a driver. A plugin/agent registers its own strategy here
// the same way the built-ins do — the registry is the open extension seam.
func Register(d Driver) { registry[d.Name] = d }

// Get returns a registered driver by name.
func Get(name string) (Driver, bool) { d, ok := registry[name]; return d, ok }

// Drivers returns every registered driver, name-sorted (deterministic).
func Drivers() []Driver {
	out := make([]Driver, 0, len(registry))
	for _, d := range registry {
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func init() {
	// recall — the top-K most relevant benign cells, rendered into context. This is
	// recall.Session.Recall expressed in the algebra: filter to the un-sealed,
	// un-tombstoned candidates, rank by descending relevance to the intent, take K,
	// render. (Equivalence to recall.Recall is witnessed in drivers_test.go.)
	Register(Driver{
		Name: "recall",
		Doc:  "render the top-K cells most relevant to the intent (≡ recall.Recall)",
		Build: func(p Params) Query {
			k := p.K
			if k <= 0 {
				k = 5
			}
			return Query{
				Intent: p.Intent,
				Ops: []Op{
					{Kind: OpScan},
					{Kind: OpFilter, Pred: &Pred{Op: PredAnd, Args: []Pred{
						{Op: PredEq, Field: "sealed", Value: "false"},
						{Op: PredEq, Field: "tombstoned", Value: "false"},
					}}},
					{Kind: OpRank, By: RankRelevance, Desc: true},
					{Kind: OpLimit, K: k},
					{Kind: OpRender},
				},
			}
		},
	})

	// render — a budget-bounded context: cells that match the intent OR are durable
	// (a standing preference is always eligible — the durability axis), ranked by
	// relevance and trimmed to the byte budget, then rendered. The generalization of
	// contextq's fixed-shape materializer into one authored sentence.
	Register(Driver{
		Name: "render",
		Doc:  "render the intent-relevant + durable cells within a byte budget",
		Build: func(p Params) Query {
			return Query{
				Intent: p.Intent,
				Ops: []Op{
					{Kind: OpScan},
					{Kind: OpFilter, Pred: &Pred{Op: PredAnd, Args: []Pred{
						{Op: PredEq, Field: "sealed", Value: "false"},
						{Op: PredEq, Field: "tombstoned", Value: "false"},
						{Op: PredOr, Args: []Pred{
							{Op: PredMatch, Value: p.Intent},
							{Op: PredEq, Field: "durability", Value: DurabilityDurable},
						}},
					}}},
					{Kind: OpRank, By: RankRelevance, Desc: true},
					{Kind: OpBudget, Bytes: p.Budget},
					{Kind: OpRender},
				},
			}
		},
	})

	// clean — forget the ephemeral: tombstone every turn-class observation (the
	// "it's 3pm" expiry). Negative-only — the cell row and bytes survive for audit,
	// only future recall skips it. Sealed cells need no tombstone (the trust gate
	// already refuses them on page-in), so the filter targets the turn class.
	Register(Driver{
		Name: "clean",
		Doc:  "tombstone turn-class (ephemeral) cells so future recall skips them",
		Build: func(p Params) Query {
			reason := p.Reason
			if reason == "" {
				reason = "expired turn-class observation"
			}
			return Query{
				Intent: p.Intent,
				Ops: []Op{
					{Kind: OpScan},
					{Kind: OpFilter, Pred: &Pred{Op: PredAnd, Args: []Pred{
						{Op: PredEq, Field: "durability", Value: DurabilityTurn},
						{Op: PredEq, Field: "tombstoned", Value: "false"},
					}}},
					{Kind: OpTombstone, Reason: reason},
				},
			}
		},
	})

	// compact — fold the unreferenced, non-durable cells into ONE derived disposition
	// (a deterministic extractive summary), then tombstone the sources. The compaction
	// path expressed as a query: the kernel does not guess the strategy; this driver
	// IS the strategy, and an agent can write a different one.
	Register(Driver{
		Name: "compact",
		Doc:  "consolidate unreferenced non-durable cells into a disposition, then tombstone the sources",
		Build: func(p Params) Query {
			reason := p.Reason
			if reason == "" {
				reason = "compacted into a derived disposition"
			}
			return Query{
				Intent: p.Intent,
				Ops: []Op{
					{Kind: OpScan},
					{Kind: OpFilter, Pred: &Pred{Op: PredAnd, Args: []Pred{
						{Op: PredEq, Field: "sealed", Value: "false"},
						{Op: PredEq, Field: "tombstoned", Value: "false"},
						{Op: PredNe, Field: "durability", Value: DurabilityDurable},
						{Op: PredEq, Field: "refcount", Value: "0"},
					}}},
					{Kind: OpRank, By: RankBytes, Desc: true},
					{Kind: OpBudget, Bytes: p.Budget},
					{Kind: OpConsolidate},
					{Kind: OpTombstone, Reason: reason},
				},
			}
		},
	})

	// dream — the storage-GC half of an offline sleep pass: reclaim unreferenced CAS
	// blobs. The trust-gate reseal/refresh half stays in recall.Dream, which memq
	// complements; a recall-image backend reports prune as proposal-only and points
	// the operator at `fak dream`.
	Register(Driver{
		Name: "dream",
		Doc:  "reclaim unreferenced storage (the GC half of a sleep pass; reseal stays in recall.Dream)",
		Build: func(p Params) Query {
			return Query{Intent: p.Intent, Ops: []Op{{Kind: OpScan}, {Kind: OpPrune}}}
		},
	})
}
