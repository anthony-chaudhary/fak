package memq

import "sort"

// Params parameterize a driver's compiled query: the agent's intent (for relevance),
// an optional K (limit), an optional byte Budget, and a Reason stamped on suppressions.
type Params struct {
	Intent string `json:"intent,omitempty"`
	K      int    `json:"k,omitempty"`
	Budget int64  `json:"budget,omitempty"`
	Reason string `json:"reason,omitempty"`

	// NarrowBytes is the OPT-IN per-cell width cap for the second compression axis
	// (#4019): when > 0, render and compact chain an OpNarrow immediately before
	// their OpBudget, so an over-wide cell is slimmed (least-relevant Attrs fields
	// dropped, Descriptor re-headLined) instead of tail-dropped whole. Default 0
	// (off): the compiled Query is identical to before this field existed.
	NarrowBytes int64 `json:"narrow_bytes,omitempty"`

	// RetentionCount is the OPT-IN divergence-outlier exemption (MiniCache; #4018):
	// when > 0, the compact driver carves the RetentionCount most-divergent fold
	// candidates out of the set (OpExempt) before consolidating, so the cells a lossy
	// fold would approximate worst survive bit-exact. 0 (the zero value) builds the
	// exact pre-#4018 pipeline.
	RetentionCount int `json:"retention_count,omitempty"`

	// StarveK opts a driver's cutline op into the deterministic anti-starvation credit
	// (#4021; see Op.StarveK). 0 (the default) builds the exact same query as before.
	StarveK int `json:"starve_k,omitempty"`
	// ProtectFloor opts the render driver's budget stage into the protected-floor
	// split (#4017, NACL/Scissorhands): durable cells plus the ProtectRecent
	// most-recent cells are charged against the byte budget FIRST, so a standing
	// preference or the fresh tail is never tail-dropped by a low relevance rank.
	// Both zero values compile to exactly the query the driver built before the
	// knob existed — default off, byte-identical.
	ProtectFloor bool `json:"protect_floor,omitempty"`
	// ProtectRecent is the byte-exact recent-window size joined to the floor
	// (top-N by Step). Meaningful only with ProtectFloor. Note the render
	// driver's filter precedes its budget, so the window spans the filtered
	// candidate set (intent-matching ∪ durable); author a raw scan→rank→budget
	// query to protect the unconditional recency tail.
	ProtectRecent int `json:"protect_recent,omitempty"`
	// MergeFloor opts the render driver's budget stage into merge-on-evict (#4015):
	// a cell evicted by the byte budget is folded into its nearest surviving near-twin
	// (simhash cosine >= this floor) instead of tail-dropped, preserving the evicted
	// cell's refs on the survivor. Cosine is in [-1,1]; a meaningful floor is in (0,1].
	// 0 (the default) tail-drops as before — byte-identical.
	MergeFloor float64 `json:"merge_floor,omitempty"`
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
	// un-tombstoned candidates, collapse byte-identical twins, rank by descending
	// relevance to the intent, take K, render. (Equivalence to recall.Recall on a
	// non-duplicate corpus is witnessed in drivers_test.go.)
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
					{Kind: OpDedup},
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
			q := Query{
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
					{Kind: OpDedup},
					{Kind: OpRank, By: RankRelevance, Desc: true},
					{Kind: OpBudget, Bytes: p.Budget, Protect: p.ProtectFloor, Recent: p.ProtectRecent, StarveK: p.StarveK, MergeFloor: p.MergeFloor},
					{Kind: OpRender},
				},
			}
			if p.NarrowBytes > 0 {
				// The opt-in width axis (#4019): narrow over-wide cells BEFORE the
				// count-axis budget so the two compression axes compose.
				q.Ops = insertNarrowBeforeBudget(q.Ops, p.NarrowBytes)
			}
			return q
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
			ops := []Op{
				{Kind: OpScan},
				{Kind: OpFilter, Pred: &Pred{Op: PredAnd, Args: []Pred{
					{Op: PredEq, Field: "sealed", Value: "false"},
					{Op: PredEq, Field: "tombstoned", Value: "false"},
					{Op: PredNe, Field: "durability", Value: DurabilityDurable},
					{Op: PredEq, Field: "refcount", Value: "0"},
				}}},
				{Kind: OpRank, By: RankBytes, Desc: true},
				{Kind: OpBudget, Bytes: p.Budget},
			}
			// MiniCache outlier exemption (#4018), opt-in via RetentionCount: carve the
			// top-K most-divergent candidates out of the fold set so the cells the fold
			// would approximate worst survive bit-exact. Unset (0) appends nothing —
			// the pipeline is exactly the pre-#4018 query.
			if p.RetentionCount > 0 {
				ops = append(ops, Op{Kind: OpExempt, By: RankDivergence, K: p.RetentionCount})
			}
			ops = append(ops,
				Op{Kind: OpConsolidate},
				Op{Kind: OpTombstone, Reason: reason},
			)
			q := Query{Intent: p.Intent, Ops: ops}
			if p.NarrowBytes > 0 {
				// The opt-in width axis (#4019): slim over-wide cells before the
				// count-axis budget picks what to fold.
				q.Ops = insertNarrowBeforeBudget(q.Ops, p.NarrowBytes)
			}
			return q
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
