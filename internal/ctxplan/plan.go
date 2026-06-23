package ctxplan

import (
	"fmt"
	"sort"
	"strings"
)

// Candidate is one span the planner may keep resident: a Span (SAFE metadata only —
// never the bytes of a sealed span) plus the two numbers the optimizer trades off — the
// token Cost to keep it resident and the estimated Benefit of doing so. The field is named
// Cell for the relational reading (a candidate row), but its type is the local Span.
type Candidate struct {
	Cell    Span    `json:"cell"`
	Cost    int     `json:"cost"`    // resident token cost (see CostModel)
	Benefit float64 `json:"benefit"` // Forecast.Benefit score
}

// density is benefit per token — the planner's ORDER BY key (the greedy knapsack ratio).
// A zero-cost candidate is treated as infinitely dense (free to keep, so always taken if
// it has any benefit); a zero-benefit candidate has density 0.
func (c Candidate) density() float64 {
	if c.Cost <= 0 {
		if c.Benefit > 0 {
			return inf
		}
		return 0
	}
	return c.Benefit / float64(c.Cost)
}

const inf = 1e308

// CostModel estimates the resident token cost of keeping a cell in context. The default
// is TokenCost; a caller with a real tokenizer can supply its own (the planner only
// needs a monotone, deterministic size proxy).
type CostModel func(Span) int

// TokenCost is the default resident-cost estimate: ceil(bytes/4), the standard 4-bytes/token
// proxy (the same one memq/recall use), so a ctxplan budget and a downstream render report
// the same token units. An empty span costs 0.
func TokenCost(c Span) int {
	if c.Bytes <= 0 {
		return 0
	}
	return int((c.Bytes + 3) / 4)
}

// Budget is the O(1) cap on the resident view: at most Tokens tokens may be resident.
// It is the constant the whole concept rests on — the resident context stays bounded by
// Tokens no matter how many turns the session runs.
type Budget struct {
	Tokens int `json:"tokens"`
}

// Selection is one span the plan keeps resident, carried in RENDER order (step ascending,
// so the materialized view reads like a coherent history even though SELECTION was by
// density). It carries the cost/benefit/density the optimizer used, for EXPLAIN.
type Selection struct {
	ID         string  `json:"id"`
	Step       int     `json:"step"`
	Role       string  `json:"role,omitempty"`
	Descriptor string  `json:"descriptor,omitempty"`
	Cost       int     `json:"cost"`
	Benefit    float64 `json:"benefit"`
	Density    float64 `json:"density"`
	Pinned     bool    `json:"pinned,omitempty"`
}

// Elision is one span the plan keeps COLD — out of the resident view but RECOVERABLE.
// Digest (and ID) are the recovery handle: a pointer back into the lossless store, so the
// span pages back in on demand (a forecast miss is a page fault, not a lost fact). The
// presence of a recovery handle is exactly what faithful.go checks to distinguish a
// planned view from lossy compaction.
type Elision struct {
	ID      string  `json:"id"`
	Step    int     `json:"step"`
	Role    string  `json:"role,omitempty"`
	Digest  string  `json:"digest,omitempty"` // content address — the page-back-in handle
	Cost    int     `json:"cost"`
	Benefit float64 `json:"benefit"`
	Reason  string  `json:"reason"` // over_budget | sealed | tombstoned
}

// Elision reasons.
const (
	ElideOverBudget = "over_budget" // lost the knapsack: a denser span took the room
	ElideSealed     = "sealed"      // quarantined by the trust gate — never a candidate
	ElideTombstoned = "tombstoned"  // suppressed by context control — never a candidate
)

// Objectives.
const (
	ObjGreedy = "greedy-density" // sort by benefit/cost, take while it fits (the production planner)
	ObjExact  = "exact-dp"       // 0/1-knapsack DP — the optimal-benefit oracle (small inputs only)
)

// Plan is the planner's chosen O(1) view: the resident Selected set, the cold-but-
// recoverable Elided set, and the cost/benefit accounting — the analogue of a query
// plan plus its EXPLAIN. Selected+Elided partition every candidate (faithful.go proves
// it), so a Plan never destroys a span; it only decides which are resident.
type Plan struct {
	Budget       int         `json:"budget"`
	Objective    string      `json:"objective"`
	Horizon      int         `json:"horizon,omitempty"`
	Selected     []Selection `json:"selected"`
	Elided       []Elision   `json:"elided"`
	Candidates   int         `json:"candidates"`
	CostUsed     int         `json:"cost_used"`
	PinnedTokens int         `json:"pinned_tokens"`
	Benefit      float64     `json:"benefit"`     // sum of selected benefits — the objective achieved
	OverBudget   bool        `json:"over_budget"` // pins alone exceeded the budget (they stay; nothing else fits)
}

// Optimize is the planner: it chooses the resident view that maximizes total benefit
// subject to the token budget, treating pins and the trust gate as hard constraints.
//
//  1. SEALED / TOMBSTONED candidates are elided up front (they are never resident —
//     the poison/suppression invariant). They keep their recovery handle for audit but
//     can never be selected; a pin naming one is refused here too.
//  2. PINNED candidates (ID in pins, and not sealed/tombstoned) are forced resident and
//     charged to the budget FIRST. If the pins alone exceed the budget they STILL stay
//     (a turn cannot proceed without them) and OverBudget is set; nothing else is added.
//  3. The remaining budget is filled by a 0/1 knapsack over the rest — greedy-by-density
//     (ObjGreedy, the deterministic production planner) or exact DP (ObjExact, the
//     optimal-benefit oracle, used to bound the greedy gap on small inputs).
//
// pins may be nil. The result is deterministic: every sort has a total tie-break, so the
// same (candidates, budget, pins, objective) yields a byte-identical Plan.
func Optimize(cands []Candidate, b Budget, pins map[string]bool, objective string) Plan {
	p := Plan{Budget: b.Tokens, Objective: objective, Candidates: len(cands)}

	var pinned, free []Candidate
	for _, c := range cands {
		switch {
		case c.Cell.Sealed:
			p.Elided = append(p.Elided, elisionOf(c, ElideSealed))
		case c.Cell.Tombstoned:
			p.Elided = append(p.Elided, elisionOf(c, ElideTombstoned))
		case pins[c.Cell.ID]:
			pinned = append(pinned, c)
		default:
			free = append(free, c)
		}
	}

	// Pins are non-negotiable: select them all and charge their cost first.
	used := 0
	for _, c := range pinned {
		p.Selected = append(p.Selected, selectionOf(c, true))
		used += c.Cost
		p.Benefit += c.Benefit
	}
	p.PinnedTokens = used
	remaining := b.Tokens - used
	if remaining < 0 {
		// The pins alone overrun the budget. They stay (correctness over thrift); the
		// optimizer adds nothing more. A higher rung that must shrink pins reports this.
		p.OverBudget = true
		remaining = 0
		for _, c := range free {
			p.Elided = append(p.Elided, elisionOf(c, ElideOverBudget))
		}
		finalize(&p, used)
		return p
	}

	// Fill the remaining budget over the free candidates.
	var chosen map[string]bool
	switch objective {
	case ObjExact:
		chosen = knapsackExact(free, remaining)
	default:
		p.Objective = ObjGreedy
		chosen = knapsackGreedy(free, remaining)
	}
	for _, c := range free {
		if chosen[c.Cell.ID] {
			p.Selected = append(p.Selected, selectionOf(c, false))
			used += c.Cost
			p.Benefit += c.Benefit
		} else {
			p.Elided = append(p.Elided, elisionOf(c, ElideOverBudget))
		}
	}
	finalize(&p, used)
	return p
}

// finalize sets the cost accounting and sorts Selected into RENDER order (step asc, then
// ID) and Elided into a stable audit order (step asc, then ID). SELECTION already
// happened by density; the resident view is presented chronologically so it reads as a
// coherent history.
func finalize(p *Plan, used int) {
	p.CostUsed = used
	sort.SliceStable(p.Selected, func(i, j int) bool {
		if p.Selected[i].Step != p.Selected[j].Step {
			return p.Selected[i].Step < p.Selected[j].Step
		}
		return p.Selected[i].ID < p.Selected[j].ID
	})
	sort.SliceStable(p.Elided, func(i, j int) bool {
		if p.Elided[i].Step != p.Elided[j].Step {
			return p.Elided[i].Step < p.Elided[j].Step
		}
		return p.Elided[i].ID < p.Elided[j].ID
	})
}

// knapsackGreedy is the production planner: sort by density (benefit/token) descending
// with a total deterministic tie-break, then take greedily while the item fits. O(n log
// n). It is not optimal in general (the classic 0/1 counterexample exists), but it is
// fast, replay-stable, and within a bounded gap of the exact optimum — the gap is what
// the DP oracle measures in tests.
func knapsackGreedy(cands []Candidate, budget int) map[string]bool {
	order := make([]Candidate, len(cands))
	copy(order, cands)
	sort.SliceStable(order, func(i, j int) bool {
		di, dj := order[i].density(), order[j].density()
		if di != dj {
			return di > dj
		}
		if order[i].Benefit != order[j].Benefit {
			return order[i].Benefit > order[j].Benefit
		}
		if order[i].Cell.Step != order[j].Cell.Step {
			return order[i].Cell.Step < order[j].Cell.Step
		}
		return order[i].Cell.ID < order[j].Cell.ID
	})
	chosen := map[string]bool{}
	used := 0
	for _, c := range order {
		if c.Benefit <= 0 {
			continue // a span the forecast predicts no use for is not worth a resident slot
		}
		if used+c.Cost <= budget {
			chosen[c.Cell.ID] = true
			used += c.Cost
		}
	}
	return chosen
}

// knapsackExact is the optimal-benefit oracle: a 0/1-knapsack DP over integer token
// costs. It is O(n*budget) time and O(budget) space, so it is only run when that product
// is small (the DPExactLimit guard) — its job is to bound the greedy planner's optimality
// gap in tests, not to plan a million-span store. When the input is too large it falls
// back to the greedy choice (so ObjExact never blows up); callers that need the oracle
// keep their inputs small. Benefits are scaled to integers (BenefitScale) so the DP
// compares total benefit exactly and deterministically.
func knapsackExact(cands []Candidate, budget int) map[string]bool {
	// Drop non-positive-benefit and over-budget-singleton items: they never help.
	var items []Candidate
	for _, c := range cands {
		if c.Benefit > 0 && c.Cost <= budget {
			items = append(items, c)
		}
	}
	if budget <= 0 || len(items) == 0 {
		return map[string]bool{}
	}
	if int64(len(items))*int64(budget) > DPExactLimit {
		return knapsackGreedy(cands, budget) // too large for the oracle; degrade to greedy
	}
	n := len(items)
	val := make([]int64, n)
	for i, c := range items {
		val[i] = int64(c.Benefit * BenefitScale)
	}
	// dp[w] = best scaled benefit achievable with capacity w after considering items so
	// far; take[i][w] records whether item i was taken at capacity w (for reconstruction).
	dp := make([]int64, budget+1)
	take := make([][]bool, n)
	for i := 0; i < n; i++ {
		take[i] = make([]bool, budget+1)
		ci, vi := items[i].Cost, val[i]
		for w := budget; w >= ci; w-- {
			if cand := dp[w-ci] + vi; cand > dp[w] {
				dp[w] = cand
				take[i][w] = true
			}
		}
	}
	chosen := map[string]bool{}
	w := budget
	for i := n - 1; i >= 0; i-- {
		if take[i][w] {
			chosen[items[i].Cell.ID] = true
			w -= items[i].Cost
		}
	}
	return chosen
}

// DPExactLimit bounds n*budget for the exact oracle so a stray ObjExact on a huge store
// degrades to greedy instead of allocating a giant table. 2e7 cells is ~160MB of bool
// table worst case — comfortably a test-only oracle, never the hot path.
const DPExactLimit = 20_000_000

// BenefitScale turns a float benefit into an integer the DP can compare exactly. 1e6
// keeps six significant digits — far finer than the benefit signals' resolution.
const BenefitScale = 1_000_000

func selectionOf(c Candidate, pinned bool) Selection {
	return Selection{
		ID: c.Cell.ID, Step: c.Cell.Step, Role: c.Cell.Role, Descriptor: c.Cell.Descriptor,
		Cost: c.Cost, Benefit: c.Benefit, Density: c.density(), Pinned: pinned,
	}
}

func elisionOf(c Candidate, reason string) Elision {
	return Elision{
		ID: c.Cell.ID, Step: c.Cell.Step, Role: c.Cell.Role, Digest: c.Cell.Digest,
		Cost: c.Cost, Benefit: c.Benefit, Reason: reason,
	}
}

// Explain renders the plan as an operator-readable EXPLAIN — the "step through this
// before you trust it" surface, the analogue of EXPLAIN ANALYZE. It shows every resident
// span (step, role, cost, benefit, density, pin marker) and every elided span (with its
// recovery handle and the reason it stayed cold), plus the cost/benefit footer and the
// faithfulness verdict so a reader can see at a glance that nothing was destroyed.
func (p Plan) Explain() string {
	var b strings.Builder
	fmt.Fprintf(&b, "ctxplan %s: budget=%d tokens, %d candidate(s)", p.Objective, p.Budget, p.Candidates)
	if p.Horizon > 0 {
		fmt.Fprintf(&b, ", horizon=%d turn(s)", p.Horizon)
	}
	b.WriteByte('\n')
	if p.OverBudget {
		fmt.Fprintf(&b, "  WARNING: pins (%d tokens) exceed the budget; they stay resident, nothing else fits\n", p.PinnedTokens)
	}
	fmt.Fprintf(&b, "  RESIDENT (%d span(s), %d tokens):\n", len(p.Selected), p.CostUsed)
	for _, s := range p.Selected {
		pin := " "
		if s.Pinned {
			pin = "*" // forced resident
		}
		fmt.Fprintf(&b, "   %s [step %d] %-16s cost=%-5d benefit=%.3f density=%.4f\n",
			pin, s.Step, truncate(s.Role, 16), s.Cost, s.Benefit, s.Density)
	}
	fmt.Fprintf(&b, "  ELIDED (%d span(s), recoverable on demand):\n", len(p.Elided))
	for _, e := range p.Elided {
		fmt.Fprintf(&b, "     [step %d] %-16s cost=%-5d reason=%-11s handle=%s\n",
			e.Step, truncate(e.Role, 16), e.Cost, e.Reason, short(e.Digest))
	}
	w := Audit(p)
	fmt.Fprintf(&b, "  total benefit=%.3f, residency=%d/%d tokens, faithful=%v (recoverable %d/%d elided)\n",
		p.Benefit, p.CostUsed, p.Budget, w.Faithful, w.Recoverable, w.Elided)
	return b.String()
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

func short(d string) string {
	if len(d) > 12 {
		return d[:12]
	}
	if d == "" {
		return "(id-only)"
	}
	return d
}
