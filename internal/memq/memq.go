package memq

import (
	"cmp"
	"fmt"
	"slices"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

// Durability classes — the temporal axis from CONTEXT-IS-NOT-MEMORY.md. memq mirrors
// the recall/ctxmmu vocabulary as plain strings (it does not import the mechanism
// packages) and normalizes any missing/unknown class to the shortest-lived one.
const (
	DurabilityTurn    = "turn"
	DurabilitySession = "session"
	DurabilityBounded = "bounded"
	DurabilityDurable = "durable"
)

// durabilityRank orders the classes from shortest- to longest-lived, so reclassify
// can refuse a PROMOTION (a move toward a longer-lived class) while allowing a hold
// or a demotion. An unknown class normalizes to turn (rank 0).
var durabilityRank = map[string]int{
	DurabilityTurn: 0, DurabilitySession: 1, DurabilityBounded: 2, DurabilityDurable: 3,
}

// NormDurability maps any class string to the canonical vocabulary, failing closed to
// turn for a missing/unknown/reserved value — the reader-side default of the
// expire-by-default posture.
func NormDurability(s string) string {
	if _, ok := durabilityRank[s]; ok {
		return s
	}
	return DurabilityTurn
}

// Cell is one addressable unit of memory: a recall page, an in-memory note, or a
// derived disposition. It carries only SAFE metadata — never the bytes of a sealed
// span (Descriptor is the extractive-or-sealed-metadata descriptor, exactly as
// recall.Page records it). Bytes are fetched lazily through a Backend's trust-gated
// Materialize, never carried on the cell.
type Cell struct {
	ID         string            `json:"id"`             // stable address within the backend (e.g. "step:3")
	Step       int               `json:"step"`           // ordinal position (recall page step; -1 for unordered)
	Role       string            `json:"role,omitempty"` // the producer (tool name, "user", "system", "memq/consolidate")
	Kind       string            `json:"kind,omitempty"` // tool_result | reasoning | user | system | episode | disposition
	Descriptor string            `json:"descriptor"`     // SAFE extractive/sealed descriptor; never sealed bytes
	Digest     string            `json:"digest,omitempty"`
	Bytes      int64             `json:"bytes"`                // size (token-cost proxy)
	Durability string            `json:"durability,omitempty"` // turn | session | bounded | durable
	Sealed     bool              `json:"sealed,omitempty"`     // quarantined by the trust gate
	Tombstoned bool              `json:"tombstoned,omitempty"` // suppressed by context control
	Witness    string            `json:"witness,omitempty"`    // external trust witness
	Refs       []string          `json:"refs,omitempty"`       // digests/ids this cell references
	Attrs      map[string]string `json:"attrs,omitempty"`      // OPEN attribute bag (forward-compatible, drop-unknown)
}

// Op kinds. The pure SELECT side transforms the working set; the EFFECT side records
// (and, with caps, applies) a mutation or materialization.
const (
	OpScan        = "scan"        // source: (re)load every cell from the backend
	OpFilter      = "filter"      // keep cells matching Pred (WHERE)
	OpRank        = "rank"        // sort by By (ORDER BY)
	OpLimit       = "limit"       // keep the first K (LIMIT)
	OpBudget      = "budget"      // keep the prefix whose cumulative Bytes <= Bytes
	OpDedup       = "dedup"       // collapse digest-identical cells to one (read-side recall dedup; #2506)
	OpNarrow      = "narrow"      // shrink kept cells' width to a per-cell byte cap (opt-in 2nd compression axis; ThinK #4019)
	OpRender      = "render"      // materialize the set into context (read-only page-in via the gate)
	OpTombstone   = "tombstone"   // negative-only suppression (recall.RequestContextChange)
	OpConsolidate = "consolidate" // fold the set into one derived extractive disposition
	OpReclassify  = "reclassify"  // change durability class (never promotes to durable)
	OpPrune       = "prune"       // reclaim unreferenced storage (GC); no model-visible effect
	OpExempt      = "exempt"      // carve the top-K most-divergent cells out of the set before a lossy fold, kept bit-exact (MiniCache; #4018)
)

// Pred ops. Comparison ops resolve a field; the boolean ops compose sub-predicates.
const (
	PredTrue   = "true"
	PredAnd    = "and"
	PredOr     = "or"
	PredNot    = "not"
	PredEq     = "eq"
	PredNe     = "ne"
	PredLt     = "lt"
	PredLe     = "le"
	PredGt     = "gt"
	PredGe     = "ge"
	PredMatch  = "match"  // token overlap of Value against (role + descriptor) > 0
	PredHasRef = "hasref" // Value is a member of the cell's Refs (the [[wikilink]] edge set); "references <id>"
)

// Rank keys.
const (
	RankRelevance  = "relevance"  // token overlap with the query Intent (the recall ranker)
	RankBytes      = "bytes"      // size
	RankStep       = "step"       // ordinal position
	RankDurability = "durability" // shortest-lived first (asc) / longest-lived first (desc)
	RankRefcount   = "refcount"   // graph in-degree — how many other cells reference this one (desc = most backlinks first)
	// RankRelevanceSpan is the neighborhood-pooled relevance variant (#4014, SnapKV):
	// each cell's intent-overlap score is summed over an odd, centered window of its
	// step-adjacent neighbors before ranking, so a coherent multi-cell span outranks an
	// isolated one-cell spike. Opt-in only — it requires Op.Window (odd, >= 1); window 1
	// is the identity kernel and reproduces RankRelevance's ordering exactly.
	RankRelevanceSpan = "relevance_span"
	RankDivergence    = "divergence" // simhash dissimilarity (1 - cosine) to the working set's consolidation centroid (desc = most divergent first; #4018)
)

// Score-aggregation ops for multi-intent recall (#4020 — the GQA group-reduce: N
// consumer intents share ONE retained set, so their per-intent relevance scores fold
// to one shared score per cell before the single top-k). The op choice is deliberately
// a knob, exposed exactly as upstream exposes it: mean dilutes a strong minority
// signal, amax over-keeps for any single consumer. Empty means ScoreAggMean.
const (
	ScoreAggMean = "mean" // fold to the per-intent average (the diluting default)
	ScoreAggAmax = "amax" // fold to the per-intent maximum (any one consumer's need keeps the cell)
	ScoreAggSum  = "sum"  // fold to the per-intent total (mean without the /N normalization)
)

// Pred is a serializable predicate expression — the WHERE clause an agent authors as
// JSON. A zero Pred (or Op == "" / PredTrue) matches every cell.
type Pred struct {
	Op    string `json:"op"`
	Field string `json:"field,omitempty"` // durability|role|kind|descriptor|digest|witness|bytes|step|sealed|tombstoned|refcount|attr:<k> (unused by hasref, which reads Value against Refs)
	Value string `json:"value,omitempty"`
	Args  []Pred `json:"args,omitempty"` // sub-predicates for and/or/not
}

// Op is one stage of the pipeline. The Kind discriminates which fields apply; the
// rest are an additive union (the same discriminated-union shape abi.Verdict uses).
type Op struct {
	Kind   string `json:"kind"`
	Pred   *Pred  `json:"pred,omitempty"`   // filter
	By     string `json:"by,omitempty"`     // rank key, or reclassify target class
	Desc   bool   `json:"desc,omitempty"`   // rank direction (default ascending)
	K      int    `json:"k,omitempty"`      // limit
	Bytes  int64  `json:"bytes,omitempty"`  // budget cap, or narrow's per-cell width cap (bytes)
	Window int    `json:"window,omitempty"` // rank relevance_span: pooling width in cells (odd, >= 1; 1 = identity kernel)
	Reason string `json:"reason,omitempty"` // tombstone/effect reason
	// StarveK is the OPT-IN deterministic anti-starvation credit for the cutline ops
	// (limit/budget), #4021. When > 0, every pass a cell falls below the op's cutline
	// advances a per-cell counter in Attrs[StarveAttr]; once the cell JUST below the
	// cutline is non-durable, unreferenced, and has sat below it for StarveK
	// consecutive passes, it is retained for ONE pass (counter reset) — the
	// NACL/Scissorhands anti-starvation axis with no RNG (see starve.go). 0 (the
	// default) is byte-identical to today.
	StarveK int `json:"starve_k,omitempty"`
	// Protect opts a budget op into the protected-floor split (#4017, NACL/
	// Scissorhands): every durable-class cell, plus the Recent most-recent cells
	// by Step, is charged against the cap FIRST — surviving regardless of
	// relevance rank — before the ephemeral remainder spends the leftover
	// headroom in working-set order. False (the default) is today's plain
	// prefix budget, byte-for-byte.
	Protect bool `json:"protect,omitempty"` // budget: opt-in protected floor (#4017)
	// Recent is the byte-exact recent-window size joined to the protected floor:
	// the top-N most-recent cells by Step, regardless of class (the NACL
	// pins ∪ recent union). Meaningful only with Protect; 0 protects the durable
	// class alone.
	Recent int `json:"recent,omitempty"` // budget: recent-window size (requires protect)
}

// Query is an authored memory request: an intent string (for relevance ranking and as
// the default match terms) plus an ordered pipeline of Ops.
type Query struct {
	Intent string `json:"intent,omitempty"`
	// NearDupThreshold is the OPT-IN advisory near-duplicate flag (#2506). When > 0,
	// Run pages each non-sealed cell's body in through the trust gate, embeds it via
	// internal/simhash, and reports body-similar pairs (cosine >= threshold) on
	// Result.Advisory for a compactor (memory-compact) to consume. It is ADVISORY
	// only — a fuzzy signal never silently decides (the vdso/neardup.go discipline),
	// so the pair is reported, never collapsed. Default 0 (off).
	NearDupThreshold float64 `json:"near_dup_threshold,omitempty"`
	// Intents is the OPT-IN multi-intent recall slice (#4020, GQA group-reduce): the
	// N consumer intents that must share this one retained set. When non-empty it
	// supersedes Intent for relevance scoring — every cell is scored against every
	// intent and the per-intent scores fold to ONE shared score via ScoreAgg (see
	// reduceGroupScores in exec.go). Empty (the default) keeps the single-Intent
	// scoring path byte-identical to today; Intents={x} ranks identically to Intent=x.
	Intents []string `json:"intents,omitempty"`
	// ScoreAgg is the tunable reduction operator folding per-intent scores when
	// Intents is set: ScoreAggMean (the default when empty), ScoreAggAmax, or
	// ScoreAggSum. An unknown value fails Validate (fail-closed); without Intents
	// the field is inert.
	ScoreAgg string `json:"score_agg,omitempty"`
	Ops      []Op   `json:"ops"`
}

// effectKinds and mutationKinds classify ops. An effect reads/derives or mutates; a
// mutation additionally requires a Caps grant to be APPLIED (otherwise it is proposed).
var effectKinds = map[string]bool{
	OpRender: true, OpTombstone: true, OpConsolidate: true, OpReclassify: true, OpPrune: true, OpDedup: true, OpNarrow: true, OpExempt: true,
}

// mutationKinds are the effects that change durable backend state. They are
// proposal-only without an explicit Caps grant. render and consolidate are NOT here:
// render is a read (page-in), and consolidate produces an in-memory artifact this rung
// never writes back (see the package doc's honest scope).
var mutationKinds = map[string]bool{
	OpTombstone: true, OpPrune: true,
}

// IsEffect reports whether an op kind is an effect (vs a pure select op).
func IsEffect(kind string) bool { return effectKinds[kind] }

// IsMutation reports whether an op kind mutates durable backend state (needs caps).
func IsMutation(kind string) bool { return mutationKinds[kind] }

// Validate fails closed on an unknown op kind, a missing/!malformed parameter, or a
// reclassify target that is not a known class. A query that does not validate never
// runs — the same posture as an unknown verdict kind resolving to its fallback.
func Validate(q Query) error {
	// #4020: an unknown score-aggregation op never runs — the same fail-closed
	// posture as an unknown op kind. Empty means the default (mean).
	switch q.ScoreAgg {
	case "", ScoreAggMean, ScoreAggAmax, ScoreAggSum:
	default:
		return fmt.Errorf("memq: unknown score_agg %q (want %s|%s|%s)", q.ScoreAgg, ScoreAggMean, ScoreAggAmax, ScoreAggSum)
	}
	for i, op := range q.Ops {
		switch op.Kind {
		case OpScan, OpRender, OpTombstone, OpConsolidate, OpPrune, OpDedup:
			// no required parameters
		case OpFilter:
			if op.Pred == nil {
				return fmt.Errorf("memq: op %d (filter) has no predicate", i)
			}
			if err := validatePred(*op.Pred); err != nil {
				return fmt.Errorf("memq: op %d (filter): %w", i, err)
			}
		case OpRank:
			switch op.By {
			case RankRelevance, RankBytes, RankStep, RankDurability, RankRefcount, RankDivergence:
			case RankRelevanceSpan:
				// The pooling kernel is part of the authored query, never guessed: a
				// centered window must be an explicit odd width (1 = identity kernel).
				if op.Window < 1 || op.Window%2 == 0 {
					return fmt.Errorf("memq: op %d (rank %s) requires an odd window >= 1, got %d", i, RankRelevanceSpan, op.Window)
				}
			default:
				return fmt.Errorf("memq: op %d (rank) has unknown key %q", i, op.By)
			}
		case OpLimit:
			if op.K < 0 {
				return fmt.Errorf("memq: op %d (limit) has negative k=%d", i, op.K)
			}
			if op.StarveK < 0 {
				return fmt.Errorf("memq: op %d (limit) has negative starve_k=%d", i, op.StarveK)
			}
		case OpBudget:
			if op.Bytes < 0 {
				return fmt.Errorf("memq: op %d (budget) has negative bytes=%d", i, op.Bytes)
			}
			if op.Recent < 0 {
				return fmt.Errorf("memq: op %d (budget) has negative recent=%d", i, op.Recent)
			}
			if op.Recent > 0 && !op.Protect {
				return fmt.Errorf("memq: op %d (budget) sets recent=%d without protect", i, op.Recent)
			}
			if op.StarveK < 0 {
				return fmt.Errorf("memq: op %d (budget) has negative starve_k=%d", i, op.StarveK)
			}
		case OpExempt:
			if op.K < 0 {
				return fmt.Errorf("memq: op %d (exempt) has negative k=%d", i, op.K)
			}
			switch op.By {
			case "", RankDivergence:
			default:
				return fmt.Errorf("memq: op %d (exempt) has unknown carve key %q (only %q is supported)", i, op.By, RankDivergence)
			}
		case OpNarrow:
			if op.Bytes < 0 {
				return fmt.Errorf("memq: op %d (narrow) has negative bytes=%d", i, op.Bytes)
			}
		case OpReclassify:
			if _, ok := durabilityRank[op.By]; !ok {
				return fmt.Errorf("memq: op %d (reclassify) target %q is not a durability class", i, op.By)
			}
		case "":
			return fmt.Errorf("memq: op %d has no kind", i)
		default:
			return fmt.Errorf("memq: op %d has unknown kind %q (no hard-delete op exists by design)", i, op.Kind)
		}
	}
	return nil
}

func validatePred(p Pred) error {
	switch p.Op {
	case "", PredTrue:
		return nil
	case PredAnd, PredOr:
		for _, a := range p.Args {
			if err := validatePred(a); err != nil {
				return err
			}
		}
		return nil
	case PredNot:
		if len(p.Args) != 1 {
			return fmt.Errorf("not requires exactly one sub-predicate, got %d", len(p.Args))
		}
		return validatePred(p.Args[0])
	case PredEq, PredNe, PredLt, PredLe, PredGt, PredGe, PredMatch:
		if p.Field == "" && p.Op != PredMatch {
			return fmt.Errorf("comparison op %q requires a field", p.Op)
		}
		return nil
	case PredHasRef:
		if p.Value == "" {
			return fmt.Errorf("hasref requires a value (the referenced cell id)")
		}
		return nil
	default:
		return fmt.Errorf("unknown predicate op %q", p.Op)
	}
}

// eval evaluates the predicate against one cell. refcount is the precomputed count of
// other cells that reference this cell's content (see Run); it is the only field that
// is not intrinsic to the cell.
func (p Pred) eval(c Cell, refcount int) bool {
	switch p.Op {
	case "", PredTrue:
		return true
	case PredAnd:
		for _, a := range p.Args {
			if !a.eval(c, refcount) {
				return false
			}
		}
		return true
	case PredOr:
		for _, a := range p.Args {
			if a.eval(c, refcount) {
				return true
			}
		}
		return false
	case PredNot:
		if len(p.Args) == 0 {
			return false
		}
		return !p.Args[0].eval(c, refcount)
	case PredMatch:
		return overlap(tokenize(p.Value), tokenize(c.Role+" "+c.Descriptor)) > 0
	case PredHasRef:
		return slices.Contains(c.Refs, p.Value)
	}
	if n, isNum := numField(c, p.Field, refcount); isNum {
		rv, err := strconv.ParseInt(strings.TrimSpace(p.Value), 10, 64)
		if err != nil {
			return false // a non-numeric value against a numeric field never matches
		}
		return cmpOrdered(p.Op, n, rv)
	}
	return cmpOrdered(p.Op, strField(c, p.Field), p.Value)
}

// numField returns (value, true) for the numeric fields, else (0, false).
func numField(c Cell, field string, refcount int) (int64, bool) {
	switch field {
	case "bytes":
		return c.Bytes, true
	case "step":
		return int64(c.Step), true
	case "refcount":
		return int64(refcount), true
	}
	return 0, false
}

// strField resolves a string-or-bool field to its string form ("true"/"false" for
// the bool fields), so eq/ne work uniformly.
func strField(c Cell, field string) string {
	switch field {
	case "durability":
		return NormDurability(c.Durability)
	case "role":
		return c.Role
	case "kind":
		return c.Kind
	case "descriptor":
		return c.Descriptor
	case "digest":
		return c.Digest
	case "witness":
		return c.Witness
	case "sealed":
		return strconv.FormatBool(c.Sealed)
	case "tombstoned":
		return strconv.FormatBool(c.Tombstoned)
	}
	if rest, ok := strings.CutPrefix(field, "attr:"); ok {
		return c.Attrs[rest]
	}
	return ""
}

// cmpOrdered evaluates a predicate operator against two ordered operands. One generic
// body serves both the numeric (int64) and lexical (string) field comparisons — Go's
// native ordering is the same shape for each, so the six-arm switch lives in one place.
func cmpOrdered[T cmp.Ordered](op string, a, b T) bool {
	switch op {
	case PredEq:
		return a == b
	case PredNe:
		return a != b
	case PredLt:
		return a < b
	case PredLe:
		return a <= b
	case PredGt:
		return a > b
	case PredGe:
		return a >= b
	}
	return false
}

// rankLess is the comparison used by OpRank. relevance ranks by descending token
// overlap with the intent (the recall ranker); the others rank by the named field.
// ra/rb are the cells' relevance scores; fa/fb their graph in-degrees (refcount).
// Ties always break by ascending Step then ID, so the order is total and
// deterministic regardless of the input slice order.
func rankLess(by string, desc bool, a, b Cell, ra, rb, fa, fb int) bool {
	prim := 0
	switch by {
	case RankRelevance:
		prim = compareInt(ra, rb) // ra/rb are overlap scores; default DESC handled below
		// relevance is inherently "higher is better": treat higher score as "less" (earlier)
		// unless desc is explicitly false.
		if !desc {
			// caller asked ascending relevance (rare); leave prim as-is
		} else {
			prim = -prim
		}
	case RankBytes:
		prim = compareInt64(a.Bytes, b.Bytes)
		if desc {
			prim = -prim
		}
	case RankStep:
		prim = compareInt(a.Step, b.Step)
		if desc {
			prim = -prim
		}
	case RankDurability:
		prim = compareInt(durabilityRank[NormDurability(a.Durability)], durabilityRank[NormDurability(b.Durability)])
		if desc {
			prim = -prim
		}
	case RankRefcount:
		prim = compareInt(fa, fb) // graph in-degree; desc = most-backlinked first
		if desc {
			prim = -prim
		}
	}
	if prim != 0 {
		return prim < 0
	}
	if a.Step != b.Step {
		return a.Step < b.Step
	}
	return a.ID < b.ID
}

func compareInt(a, b int) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	}
	return 0
}

func compareInt64(a, b int64) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	}
	return 0
}

// sortByRank stably sorts cells in place by the rank key. relevance needs each cell's
// overlap score against the intent (score); refcount needs each cell's graph in-degree
// (refcount) — both precomputed once per Run.
func sortByRank(cells []Cell, by string, desc bool, score, refcount map[string]int) {
	sort.SliceStable(cells, func(i, j int) bool {
		return rankLess(by, desc, cells[i], cells[j],
			score[cells[i].ID], score[cells[j].ID],
			refcount[cells[i].ID], refcount[cells[j].ID])
	})
}

// sortByDivergence stably sorts cells in place by their divergence score — the simhash
// dissimilarity (1 - cosine) to the working set's own consolidation centroid, computed
// per rank op by the executor (divergence is working-set-relative, so it is never
// precomputed once per Run the way relevance/refcount are; #4018). desc puts the most
// divergent first (the MiniCache outlier order). A cell with no computed score (sealed
// or unpageable) scores 0 — least divergent. Ties always break by ascending Step then
// ID, the same total-order discipline as rankLess.
func sortByDivergence(cells []Cell, desc bool, div map[string]float64) {
	sort.SliceStable(cells, func(i, j int) bool {
		di, dj := div[cells[i].ID], div[cells[j].ID]
		if di != dj {
			if desc {
				return di > dj
			}
			return di < dj
		}
		if cells[i].Step != cells[j].Step {
			return cells[i].Step < cells[j].Step
		}
		return cells[i].ID < cells[j].ID
	})
}

// --- tokenization (the recall ranker's extractive overlap, kept local) ---

func tokenize(s string) []string {
	return strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

func overlap(query, doc []string) int {
	set := make(map[string]bool, len(query))
	for _, t := range query {
		if len(t) > 2 {
			set[t] = true
		}
	}
	n := 0
	seen := map[string]bool{}
	for _, t := range doc {
		if set[t] && !seen[t] {
			seen[t] = true
			n++
		}
	}
	return n
}
