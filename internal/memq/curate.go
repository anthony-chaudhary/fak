package memq

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// This file is BUDGET-CURATED FORGETTING (#3908): value-ranked eviction under a HARD
// byte cap, witnessed by a regret metric. It is the eviction policy for the
// capped-verified-memory epic (#2395), which makes the index cap a kernel byte budget.
//
// SOTA anchor: "Forget to Improve: On-Device LLM-Agent Continual Learning via
// Budget-Curated Memory" (arXiv 2606.25115) frames memory as hard-bounded and exposed —
// the agent must continuously decide what to retain or discard under a strict budget.
// fak's differentiator is the WITNESS: not just "evict low value," but *prove the
// forgetting was net-positive* via a regret signal (CurateRegret below).
//
// The safety posture is inherited unchanged from the rest of memq (doc.go): there is NO
// hard-delete. Curate SELECTS the lowest-witnessed-value cells; the eviction itself is a
// negative-only `tombstone` (bytes / row survive for audit — recall.RequestContextChange)
// and is fail-closed under caps (ApplyCurate proposes without a grant). A durable,
// referenced, or intentional-floor cell is NEVER an eviction candidate — the same
// guard-not-estimate distinction as dojo.IntentionalFloor.

// ValueAttr is the Attrs key carrying a cell's WITNESSED value: the realized
// recall-events contribution attributed to this cell, on the memvaluescore FrontierUnits
// weights (fresh_rendered ×2, lesson_distilled ×4, stale_withheld ×8). It rides the OPEN
// Attrs bag like StarveAttr, so no Cell schema change is needed and any backend that
// round-trips Attrs persists it for free. It is NEVER store size or capability presence —
// internal/memvaluescore/score.go:21-23 explicitly forbids those as value ("unbounded
// ephemera is not value"), so curate ranks by realized value and uses size only as the
// COST axis. A missing / negative / non-numeric value fails closed to 0, so an
// un-witnessed cell has the lowest value and is the FIRST evicted.
const ValueAttr = "memq.value"

// FloorAttr is the Attrs key marking an INTENTIONAL FLOOR cell: a value the operator
// declared must-keep regardless of witnessed value — a guard, not an estimate (the
// dojo.IntentionalFloor mirror). A truthy value ("1"/"true"/"yes") protects the cell from
// eviction exactly like the durable class does, so a safety-critical standing note with
// zero recall events yet is never curated away.
const FloorAttr = "memq.floor"

// CurateReason is the closed-vocabulary verdict name stamped on a budget-curated pass —
// a TYPED event a consumer keys off, never free text (like OverflowReason / StarveReason).
const CurateReason = "MEMORY_BUDGET_CURATED"

// RegretReason is the closed-vocabulary name of the curated-forgetting regret event —
// the keep-bit that records when a later recall needed a cell curate evicted (the #3908
// DoD-3 regret metric). The wire name is spelled root-free so the concept-admission gate
// reads it as this package's own typed event, not a new canonical concept.
const RegretReason = "curate_regret"

// witnessedValue reads a cell's realized recall value: the injected per-cell value map
// (folded from the recall-events ledger, the same decoupling memvaluescore uses) takes
// precedence, falling back to the persisted Attrs[ValueAttr]. Fails closed to 0.
func witnessedValue(c Cell, override map[string]int) int {
	if override != nil {
		if v, ok := override[c.ID]; ok {
			if v < 0 {
				return 0
			}
			return v
		}
	}
	n, err := strconv.Atoi(strings.TrimSpace(c.Attrs[ValueAttr]))
	if err != nil || n < 0 {
		return 0
	}
	return n
}

// intentionalFloor reports whether a cell is a declared must-keep floor (the
// dojo.IntentionalFloor mirror). A missing / falsey attr is not a floor.
func intentionalFloor(c Cell) bool {
	switch strings.TrimSpace(strings.ToLower(c.Attrs[FloorAttr])) {
	case "1", "true", "yes":
		return true
	}
	return false
}

// isProtected reports whether a cell survives eviction REGARDLESS of its witnessed value
// — the safety floor that is never a curate candidate: the durable class (an earned,
// consented standing keep), a referenced cell (refcount>0 — it survives through its
// referrers, and dropping it would strand an edge), and an intentional-floor cell (an
// operator guard). Sealed / tombstoned cells are handled separately by the caller (they
// are already refused on page-in, so tombstoning them adds nothing).
func isProtected(c Cell, refcount int) bool {
	if intentionalFloor(c) {
		return true
	}
	if NormDurability(c.Durability) == DurabilityDurable {
		return true
	}
	return refcount > 0
}

// Eviction names one cell curate selected to tombstone — never an anonymous tail-drop.
// Value is the witnessed value it evicted at; Bytes is the cost it reclaims; Descriptor
// is the recovery handle for the operator report.
type Eviction struct {
	ID         string `json:"id"`
	Descriptor string `json:"descriptor,omitempty"`
	Bytes      int64  `json:"bytes"`
	Value      int    `json:"value"`
}

// CurateReport is the typed verdict of one BudgetCurate pass (#3908): the byte cap, the
// surviving set, the protected floor that was never eligible, and the evicted set
// (lowest-value-first, the deterministic tombstone work-list). FloorOverBudget is set
// when the protected floor alone exceeds the cap — the fail-closed case where NO
// protected cell is dropped even though the target cannot be met.
type CurateReport struct {
	Reason          string     `json:"reason"` // always CurateReason
	Budget          int64      `json:"budget"`
	Kept            int        `json:"kept"`
	KeptBytes       int64      `json:"kept_bytes"`
	Protected       []string   `json:"protected,omitempty"`
	Evicted         []Eviction `json:"evicted,omitempty"`
	FloorOverBudget bool       `json:"floor_over_budget,omitempty"`
}

// BudgetCurate selects the lowest-witnessed-value cells to evict so the surviving store
// fits a HARD byte budget (#3908 DoD 1-2). It is a pure, deterministic pass — a fixed
// (cells, budget, value) yields a byte-identical report, no RNG / clock / map-order
// dependence — that SELECTS; the negative-only tombstone is applied by ApplyCurate under
// caps. Ranking is by realized value (never size or capability presence); size is only
// the cost that decides how many low-value cells must go. A durable, referenced, or
// intentional-floor cell is never a candidate, and a protected floor that alone exceeds
// the cap is kept in full (fail-closed: a protected cell is never dropped). budget<=0 is
// unbounded (keep everything). A tombstoned cell is already suppressed and is ignored.
func BudgetCurate(cells []Cell, budget int64, value map[string]int) CurateReport {
	rep := CurateReport{Reason: CurateReason, Budget: budget}
	refcount := computeRefcount(cells)

	type scored struct {
		c   Cell
		val int
	}
	var floorBytes int64 // protected floor + sealed: charged first, never evicted
	var protectedIDs []string
	var cand []scored
	for _, c := range cells {
		if c.Tombstoned {
			continue // already suppressed — not eligible, not counted
		}
		if c.Sealed {
			floorBytes += c.Bytes // refused on page-in; not a curate target, but occupies bytes
			continue
		}
		if isProtected(c, refcount[c.ID]) {
			protectedIDs = append(protectedIDs, c.ID)
			floorBytes += c.Bytes
			continue
		}
		cand = append(cand, scored{c: c, val: witnessedValue(c, value)})
	}
	sort.Strings(protectedIDs)
	rep.Protected = protectedIDs

	if budget <= 0 {
		// Unbounded: keep everything, no eviction.
		rep.Kept, rep.KeptBytes = keptTally(cells, nil)
		return rep
	}
	if floorBytes > budget {
		rep.FloorOverBudget = true
	}

	// Keep the highest-value ephemera first; the low-value remainder that no longer
	// fits is evicted. Greedy + deterministic, mirroring applyBudget's keep-if-it-fits.
	sort.SliceStable(cand, func(i, j int) bool {
		return keepPriorityLess(cand[i].c, cand[j].c, cand[i].val, cand[j].val)
	})
	used := floorBytes
	droppedSet := map[string]bool{}
	var evicted []Eviction
	for _, s := range cand {
		if used+s.c.Bytes <= budget {
			used += s.c.Bytes
			continue // kept
		}
		droppedSet[s.c.ID] = true
		evicted = append(evicted, Eviction{ID: s.c.ID, Descriptor: s.c.Descriptor, Bytes: s.c.Bytes, Value: s.val})
	}
	// Report the evicted set worst-first: lowest value, then oldest step, then ID — the
	// deterministic compaction order the operator reads.
	sort.SliceStable(evicted, func(i, j int) bool {
		return droppedWorstFirst(evicted[i], evicted[j])
	})
	rep.Evicted = evicted
	rep.Kept, rep.KeptBytes = keptTally(cells, droppedSet)
	return rep
}

// keptTally counts the survivors (non-tombstoned, non-evicted) and their byte weight.
func keptTally(cells []Cell, evicted map[string]bool) (int, int64) {
	n := 0
	var bytes int64
	for _, c := range cells {
		if c.Tombstoned || evicted[c.ID] {
			continue
		}
		n++
		bytes += c.Bytes
	}
	return n, bytes
}

// keepPriorityLess orders candidates best-to-keep first: highest witnessed value, then
// the more recent cell (higher Step), then ID for a total, RNG-free order.
func keepPriorityLess(a, b Cell, va, vb int) bool {
	if va != vb {
		return va > vb
	}
	if a.Step != b.Step {
		return a.Step > b.Step
	}
	return a.ID < b.ID
}

// droppedWorstFirst orders the evicted report worst-first: lowest value, then oldest step,
// then ID — value drives the order, never size.
func droppedWorstFirst(a, b Eviction) bool {
	if a.Value != b.Value {
		return a.Value < b.Value
	}
	if a.ID != b.ID {
		return a.ID < b.ID
	}
	return a.Bytes < b.Bytes
}

// RegretReport is the keep-bit for one curated pass (#3908 DoD 3): when a later recall
// would have needed a cell curate evicted (a stale/absent claim the evicted card
// carried), that is regret. The no-eviction baseline regrets nothing (with nothing
// evicted, the needed set can never intersect the empty evicted set), so ANY regret
// rises over baseline and REVERTS the policy. The gross/net split mirrors the cache-value
// ledger: gross bytes the eviction reclaimed, minus the bytes of the regretted cells a
// later recall had to recover, is the NET saving.
type RegretReport struct {
	Reason         string   `json:"reason"` // always RegretReason
	Needed         []string `json:"needed,omitempty"`
	Regretted      []string `json:"regretted,omitempty"` // needed ∩ evicted
	Regret         int      `json:"regret"`
	Baseline       int      `json:"baseline"` // the no-eviction regret (0 by construction)
	Reverts        bool     `json:"reverts"`
	GrossReclaimed int64    `json:"gross_reclaimed"` // bytes the eviction reclaimed
	RegretCost     int64    `json:"regret_cost"`     // bytes of the regretted (needed-but-evicted) cells
	NetReclaimed   int64    `json:"net_reclaimed"`   // gross - regret cost (the net-not-gross line)
}

// CurateRegret witnesses a curated pass against the set of cell IDs a LATER recall
// needed. A needed cell that was evicted is regret; regret over the no-eviction baseline
// (0) reverts the policy. Deterministic and read-only.
func CurateRegret(rep CurateReport, needed []string) RegretReport {
	w := RegretReport{Reason: RegretReason, Baseline: 0}
	evicted := make(map[string]int64, len(rep.Evicted))
	for _, e := range rep.Evicted {
		evicted[e.ID] = e.Bytes
		w.GrossReclaimed += e.Bytes
	}
	seen := map[string]bool{}
	for _, id := range needed {
		if seen[id] {
			continue
		}
		seen[id] = true
		w.Needed = append(w.Needed, id)
	}
	sort.Strings(w.Needed)
	for _, id := range w.Needed {
		if b, ok := evicted[id]; ok {
			w.Regretted = append(w.Regretted, id)
			w.RegretCost += b
		}
	}
	w.Regret = len(w.Regretted)
	w.Reverts = w.Regret > w.Baseline
	w.NetReclaimed = w.GrossReclaimed - w.RegretCost
	return w
}

// ApplyCurate applies (or, without caps, PROPOSES) the negative-only tombstone eviction a
// CurateReport selected — the fail-closed, caps-gated half of budget-curated forgetting
// (#3908 DoD 2). It mirrors applyTombstone: a backend with no Tombstoner, or a caller
// with no OpTombstone grant, gets a proposal-only Effect and mutates nothing.
func ApplyCurate(ctx context.Context, b Backend, rep CurateReport, caps Caps) Effect {
	ts, canApply := b.(Tombstoner)
	apply := caps.may(OpTombstone) && canApply
	ids := make([]string, 0, len(rep.Evicted))
	appliedAny := false
	for _, e := range rep.Evicted {
		ids = append(ids, e.ID)
		if apply {
			if ok, err := ts.Tombstone(ctx, e.ID, CurateReason, "memq/curate"); err == nil && ok {
				appliedAny = true
			}
		}
	}
	note := ""
	switch {
	case !canApply:
		note = "backend does not support tombstone; proposal only"
	case !apply:
		note = "no caps granted; proposal only (run with --apply to evict)"
	}
	return Effect{Kind: OpTombstone, Applied: appliedAny, Reason: CurateReason, Cells: ids, Note: note}
}

// CurateReportText is the operator-facing report (#3908 DoD 4): the byte budget, the evicted
// set (with what each cost in value and bytes), and — when a regret witness is supplied —
// the running regret rate plus the gross/net split so the operator can see what
// forgetting cost. Deterministic; the CLI shell renders exactly this string.
func CurateReportText(rep CurateReport, reg *RegretReport) string {
	var b strings.Builder
	fmt.Fprintf(&b, "budget-curated forgetting: %d cell(s) kept (%d byte(s)) under a %d-byte cap; %d evicted\n",
		rep.Kept, rep.KeptBytes, rep.Budget, len(rep.Evicted))
	if rep.FloorOverBudget {
		b.WriteString("  NOTE: the protected floor alone exceeds the cap — no protected cell is dropped (fail-closed)\n")
	}
	if len(rep.Protected) > 0 {
		fmt.Fprintf(&b, "  protected (never evicted): %s\n", strings.Join(rep.Protected, ", "))
	}
	for _, e := range rep.Evicted {
		desc := e.Descriptor
		if desc != "" {
			desc = "  " + desc
		}
		fmt.Fprintf(&b, "  evict %s  value=%d bytes=%d%s\n", e.ID, e.Value, e.Bytes, desc)
	}
	if reg != nil {
		rate := 0.0
		if len(reg.Needed) > 0 {
			rate = float64(reg.Regret) / float64(len(reg.Needed))
		}
		verdict := "net-positive (keep)"
		if reg.Reverts {
			verdict = "REVERTS — regret rose over the no-eviction baseline"
		}
		fmt.Fprintf(&b, "  regret: %d/%d needed recall(s) hit an evicted cell (rate %.3f); reclaimed gross=%d net=%d byte(s); %s\n",
			reg.Regret, len(reg.Needed), rate, reg.GrossReclaimed, reg.NetReclaimed, verdict)
	}
	return b.String()
}
