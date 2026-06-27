package ctxplan

import (
	"fmt"
	"math"
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
	Benefit float64 `json:"benefit"` // Forecast.Benefit score (the ISOLATED, per-span score)
	Area    string  `json:"area,omitempty"`
	// Precision is the layout policy that produced this candidate. The optimizer only
	// needs the candidate's cost/benefit; layout-aware callers use this metadata to explain
	// why a span was forced resident, planned normally, or kept as a pointer.
	Precision string `json:"precision,omitempty"`

	// Coverage geometry — precomputed by Candidates so the coverage objective can score
	// MARGINAL benefit (a candidate's relevance discounted by the intents the resident set
	// already covers) without re-scoring against the forecast. coverHit is the distinct
	// forecast intent tokens this cell's role+descriptor match; coverQTotal is |intent
	// tokens| (the relevance denominator); coverRelWeight is the relevance weight;
	// coverRest is the weighted NON-relevance benefit (utility+durability+recency), the
	// modular part the coverage objective keeps additive while it discounts relevance. A
	// hand-built Candidate leaves them zero, and the coverage objective degrades to the
	// isolated Benefit (no discounting) — fail-safe, never a mis-selection.
	coverHit       map[string]bool
	coverQTotal    int
	coverRelWeight float64
	coverRest      float64
}

// density is benefit per token — the planner's ORDER BY key (the greedy knapsack ratio).
// A zero-cost candidate is treated as infinitely dense (free to keep, so always taken if
// it has any benefit); a zero-benefit candidate has density 0.
func (c Candidate) density() float64 {
	// Fail closed on a non-finite benefit: a NaN/Inf signal (e.g. a poisoned utility) must
	// never sort ahead of a real candidate or corrupt the budgeted selection.
	if math.IsNaN(c.Benefit) || math.IsInf(c.Benefit, 0) {
		return 0
	}
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

// Tokenizer counts the tokens of a text — the one-method contract a real BPE /
// SentencePiece tokenizer satisfies. ctxplan defines it LOCALLY (the leaf imports nothing
// internal, by the foundation-tier rule) so the planner can take a REAL per-span token
// cost in place of the bytes/4 proxy: a higher-tier adapter wraps
// internal/tokenizer.Encoder (or any tokenizer) into this interface and passes
// TokenizerCost(it) as the CostModel. The planner only ever needs a monotone,
// deterministic size proxy, so a real count is a strictly-more-honest CostModel, not a
// correctness change.
type Tokenizer interface {
	Count(text string) int
}

// TokenizerCost returns a CostModel that prices a span by tokenizing its renderable text
// (role + descriptor — the SAFE metadata the planner actually sees; the full resident body
// is behind the store's gate and is a higher-tier concern) with tok. It is the
// "real-tokenizer CostModel" wiring: pass a Tokenizer, get the CostModel the planner,
// Candidates, and Materialize already accept. A nil tok falls back to TokenCost
// (fail-closed to the documented bytes/4 proxy), so a caller with no tokenizer on hand
// gets the original estimate with no special case. A non-positive count costs 0.
func TokenizerCost(tok Tokenizer) CostModel {
	if tok == nil {
		return TokenCost
	}
	return func(c Span) int {
		n := tok.Count(c.Role + " " + c.Descriptor)
		if n < 0 {
			return 0
		}
		return n
	}
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
	Area       string  `json:"area,omitempty"`
	Precision  string  `json:"precision,omitempty"`
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
	ID        string  `json:"id"`
	Step      int     `json:"step"`
	Role      string  `json:"role,omitempty"`
	Area      string  `json:"area,omitempty"`
	Precision string  `json:"precision,omitempty"`
	Digest    string  `json:"digest,omitempty"` // content address — the page-back-in handle
	Cost      int     `json:"cost"`
	Benefit   float64 `json:"benefit"`
	Reason    string  `json:"reason"` // over_budget | sealed | tombstoned | pointer
}

// Elision reasons.
const (
	ElideOverBudget = "over_budget" // lost the knapsack: a denser span took the room
	ElideSealed     = "sealed"      // quarantined by the trust gate — never a candidate
	ElideTombstoned = "tombstoned"  // suppressed by context control — never a candidate
	ElideDuplicate  = "duplicate"   // byte-identical to an already-resident span (equal Digest) — kept cold, recoverable
	ElidePointer    = "pointer"     // intentionally kept as a recoverable pointer by the area layout
)

// Objectives.
const (
	ObjGreedy   = "greedy-density" // sort by benefit/cost, take while it fits (the production planner)
	ObjExact    = "exact-dp"       // 0/1-knapsack DP — the optimal-benefit oracle (small inputs only)
	ObjCoverage = "coverage"       // submodular greedy: marginal benefit/cost, relevance discounted by covered intents (1-1/e)
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
//  2. DUPLICATE-DIGEST candidates are collapsed to one representative (the content-dedup
//     phase). Two byte-identical spans (equal Digest) would otherwise both ride into the
//     resident view and both charge the budget for the same bytes — pure redundancy. The
//     non-representatives are elided as ElideDuplicate (recoverable via their Digest) so
//     Audit still partitions them; a pin wins the representative slot when one is pinned.
//  3. PINNED candidates (ID in pins, and not sealed/tombstoned/duplicate) are forced
//     resident and charged to the budget FIRST. If the pins alone exceed the budget they
//     STILL stay (a turn cannot proceed without them) and OverBudget is set; nothing else
//     is added.
//  4. The remaining budget is filled by a 0/1 knapsack over the rest — greedy-by-density
//     (ObjGreedy, the deterministic production planner), exact DP (ObjExact, the optimal-
//     benefit oracle, used to bound the greedy gap on small inputs), or the submodular
//     coverage greedy (ObjCoverage, which discounts relevance by the intents already
//     resident — the broad-coverage axis).
//
// pins may be nil. The result is deterministic: every sort has a total tie-break, so the
// same (candidates, budget, pins, objective) yields a byte-identical Plan.
func Optimize(cands []Candidate, b Budget, pins map[string]bool, objective string) Plan {
	// A negative budget is meaningless; clamp to 0 (nothing non-pinned fits) so the
	// OverBudget branch below is reached ONLY when real pins overrun, never on a stray
	// negative budget with no pins.
	budgetTokens := b.Tokens
	if budgetTokens < 0 {
		budgetTokens = 0
	}
	p := Plan{Budget: budgetTokens, Objective: objective, Candidates: len(cands)}

	var live []Candidate
	for _, c := range cands {
		switch {
		case c.Cell.Sealed:
			p.Elided = append(p.Elided, elisionOf(c, ElideSealed))
		case c.Cell.Tombstoned:
			p.Elided = append(p.Elided, elisionOf(c, ElideTombstoned))
		default:
			live = append(live, c)
		}
	}

	// Content-dedup: collapse equal-Digest spans to one representative BEFORE the
	// knapsack, so two byte-identical spans never both charge the budget. The duplicates
	// are elided (ElideDuplicate) with their Digest recovery handle, so Audit still
	// partitions every candidate and the originals stay one demand-page away.
	live, dupElided := dedup(live, pins)
	p.Elided = append(p.Elided, dupElided...)

	var pinned, free []Candidate
	for _, c := range live {
		if pins[c.Cell.ID] {
			pinned = append(pinned, c)
		} else {
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
	remaining := budgetTokens - used
	if remaining < 0 {
		// The pins alone overrun the budget (used > budgetTokens >= 0 implies used > 0, so
		// pins genuinely exist). They stay (correctness over thrift); the optimizer adds
		// nothing more. A higher rung that must shrink pins reports this.
		p.OverBudget = true
		for _, c := range free {
			p.Elided = append(p.Elided, elisionOf(c, ElideOverBudget))
		}
		finalize(&p, used)
		return p
	}

	// Fill the remaining budget over the free candidates. The knapsack returns a mask over
	// `free` BY INDEX (not keyed by Cell.ID): selection and cost-charging are per ROW, so
	// two candidates that happen to share an ID can never both ride in on one knapsack
	// slot and double-charge the budget — the O(1) bound holds regardless of ID collisions.
	var chosen []bool
	switch objective {
	case ObjExact:
		chosen = knapsackExact(free, remaining)
	case ObjCoverage:
		chosen = knapsackCoverage(free, remaining)
	default:
		p.Objective = ObjGreedy
		chosen = knapsackGreedy(free, remaining)
	}
	for i, c := range free {
		if chosen[i] {
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
func knapsackGreedy(cands []Candidate, budget int) []bool {
	// order indexes INTO cands (so the returned mask is by row, never by ID).
	order := make([]int, len(cands))
	for i := range cands {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		ca, cb := cands[order[a]], cands[order[b]]
		di, dj := ca.density(), cb.density()
		if di != dj {
			return di > dj
		}
		if ca.Benefit != cb.Benefit {
			return ca.Benefit > cb.Benefit
		}
		if ca.Cell.Step != cb.Cell.Step {
			return ca.Cell.Step < cb.Cell.Step
		}
		return ca.Cell.ID < cb.Cell.ID
	})
	chosen := make([]bool, len(cands))
	used := 0
	for _, idx := range order {
		c := cands[idx]
		if c.Benefit <= 0 {
			continue // a span the forecast predicts no use for is not worth a resident slot
		}
		if used+c.Cost <= budget {
			chosen[idx] = true
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
func knapsackExact(cands []Candidate, budget int) []bool {
	chosen := make([]bool, len(cands))
	// Eligible items, each remembering its ORIGINAL index into cands so the returned mask
	// is by row. Drop non-positive-benefit and over-budget-singleton items: they never help.
	type item struct {
		idx, cost int
		val       int64
	}
	var items []item
	for i, c := range cands {
		if c.Benefit > 0 && c.Cost <= budget {
			items = append(items, item{idx: i, cost: c.Cost, val: scaleBenefit(c.Benefit)})
		}
	}
	if budget <= 0 || len(items) == 0 {
		return chosen
	}
	if int64(len(items))*int64(budget) > DPExactLimit {
		return knapsackGreedy(cands, budget) // too large for the oracle; degrade to greedy
	}
	n := len(items)
	// dp[w] = best scaled benefit achievable with capacity w after considering items so
	// far; take[i][w] records whether item i was taken at capacity w (for reconstruction).
	dp := make([]int64, budget+1)
	take := make([][]bool, n)
	for i := 0; i < n; i++ {
		take[i] = make([]bool, budget+1)
		ci, vi := items[i].cost, items[i].val
		for w := budget; w >= ci; w-- {
			if c := dp[w-ci] + vi; c > dp[w] {
				dp[w] = c
				take[i][w] = true
			}
		}
	}
	w := budget
	for i := n - 1; i >= 0; i-- {
		if take[i][w] {
			chosen[items[i].idx] = true
			w -= items[i].cost
		}
	}
	return chosen
}

// scaleBenefit turns a float benefit into the integer the DP compares. It ROUNDS (not
// truncates, so near-ties rank correctly) and clamps to a safe ceiling so a pathological
// (or maliciously large) benefit cannot overflow int64 and flip a huge-value item to a
// negative DP value. A non-finite benefit (NaN/Inf) maps to 0 — fail-closed, the same
// posture the benefit model uses (forecast.go), so a poisoned signal can never corrupt
// the oracle's value table.
func scaleBenefit(b float64) int64 {
	if math.IsNaN(b) || b <= 0 {
		return 0
	}
	v := math.Round(b * BenefitScale)
	if v > maxScaledBenefit {
		return maxScaledBenefit
	}
	return int64(v)
}

// maxScaledBenefit caps a single item's scaled benefit far below int64 overflow even
// summed across a very large plan (1e15 * millions of items stays within int64's ~9.2e18).
const maxScaledBenefit = 1_000_000_000_000_000

// DPExactLimit bounds n*budget for the exact oracle so a stray ObjExact on a huge store
// degrades to greedy instead of allocating a giant table. 2e7 cells is ~160MB of bool
// table worst case — comfortably a test-only oracle, never the hot path.
const DPExactLimit = 20_000_000

// BenefitScale turns a float benefit into an integer the DP can compare exactly. 1e6
// keeps six significant digits — far finer than the benefit signals' resolution.
const BenefitScale = 1_000_000

// dedup is the content-dedup phase: it collapses equal-Digest candidates to ONE
// representative and elides the rest as ElideDuplicate. Two byte-identical spans (equal
// Digest) carry the same content, so keeping both resident charges the budget twice for
// the same bytes and fills the view with redundant takes — exactly what a budget-bounded
// resident view must not do. The representative is chosen to preserve caller intent: a
// PINNED candidate wins (so a pin's content stays resident), else the lowest step then
// lowest ID (the deterministic order the other planners use). A span with an empty Digest
// is never deduped (no content address -> identity is unprovable; the row-vs-ID charging
// guard handles a pure ID collision separately). The elided duplicates keep their Digest
// recovery handle, so Audit still partitions them and they page back in on demand — a
// duplicate is recoverable cold storage, never a destroyed fact.
//
// This runs BEFORE the knapsack, so the dedup decision is independent of objective: the
// greedy, exact, and coverage planners all benefit from a non-redundant candidate set.
func dedup(live []Candidate, pins map[string]bool) (representatives []Candidate, duplicates []Elision) {
	// Group live indexes by Digest (non-empty only), remembering first-seen order so the
	// representative pick is deterministic regardless of map iteration order.
	groups := map[string][]int{}
	order := []string{}
	for i, c := range live {
		d := c.Cell.Digest
		if d == "" {
			continue
		}
		if _, ok := groups[d]; !ok {
			order = append(order, d)
		}
		groups[d] = append(groups[d], i)
	}
	elide := make([]bool, len(live))
	for _, d := range order {
		idxs := groups[d]
		if len(idxs) <= 1 {
			continue
		}
		best := idxs[0]
		for _, idx := range idxs[1:] {
			if betterRep(live[idx], live[best], pins) {
				best = idx
			}
		}
		for _, idx := range idxs {
			if idx == best {
				continue
			}
			elide[idx] = true
			duplicates = append(duplicates, elisionOf(live[idx], ElideDuplicate))
		}
	}
	for i, c := range live {
		if !elide[i] {
			representatives = append(representatives, c)
		}
	}
	return
}

// betterRep reports whether a is a better dedup representative than b. A pinned candidate
// wins (it preserves the caller's pin intent); among equals the lower step then lower ID
// wins, matching the deterministic tie-break the other planners use.
func betterRep(a, b Candidate, pins map[string]bool) bool {
	pa, pb := pins[a.Cell.ID], pins[b.Cell.ID]
	if pa != pb {
		return pa // a pinned candidate is the better representative
	}
	return candidateStepIDLess(a, b)
}

// candidateStepIDLess is the deterministic total tie-break the planners share: lower step
// wins, then lower ID. It is the final fallback after each planner's primary key (pin
// preference, marginal benefit, …), so identical inputs always order byte-identically.
func candidateStepIDLess(a, b Candidate) bool {
	if a.Cell.Step != b.Cell.Step {
		return a.Cell.Step < b.Cell.Step
	}
	return a.Cell.ID < b.Cell.ID
}

// knapsackCoverage is the submodular greedy for the coverage objective: it maximizes a
// monotone submodular benefit (the relevance term, discounted by the intents the resident
// set already covers) subject to the token budget. At each step it takes the still-free
// candidate that FITS and has the highest marginal-benefit/cost density, then folds that
// candidate's covered intents into the resident set — so a span whose intents are already
// covered scores zero marginal relevance and loses its slot to a span covering something
// new. Greedy over a monotone submodular function under a cardinality/knapsack constraint
// gives the classic (1-1/e) approximation. The non-relevance benefit terms stay modular
// (additive) — only relevance is the coverage signal. O(n^2) worst case; the same
// test/demo scale the exact oracle serves.
//
// It is deterministic: every selection round picks the max (density desc, marginal desc,
// step asc, ID asc) with a total tie-break, so identical inputs yield a byte-identical
// mask. Candidates without precomputed coverage geometry (a hand-built Candidate) degrade
// to their isolated Benefit as the marginal — fail-safe, never a mis-selection.
func knapsackCoverage(cands []Candidate, budget int) []bool {
	chosen := make([]bool, len(cands))
	covered := map[string]bool{}
	used := 0
	for {
		best := -1
		bestDensity := 0.0
		bestMarg := 0.0
		for i, c := range cands {
			if chosen[i] {
				continue
			}
			if used+c.Cost > budget {
				continue // does not fit
			}
			marg := coverageMarginal(c, covered)
			if marg <= 0 {
				continue // no diminishing-returns gain -> not worth a resident slot
			}
			d := inf
			if c.Cost > 0 {
				d = marg / float64(c.Cost)
			} // a zero-cost candidate is infinitely dense (free to keep)
			if best < 0 || d > bestDensity ||
				(d == bestDensity && coverageTieBreak(c, cands[best], marg, bestMarg)) {
				best = i
				bestDensity = d
				bestMarg = marg
			}
		}
		if best < 0 {
			break
		}
		chosen[best] = true
		for t := range cands[best].coverHit {
			covered[t] = true
		}
		used += cands[best].Cost
	}
	return chosen
}

// coverageMarginal is the benefit of adding c to a resident set that already covers
// `covered`: the relevance term counts only the intent tokens c covers that are NOT yet
// covered (the diminishing-returns discount), plus the modular non-relevance benefit. A
// candidate with no precomputed geometry degrades to its isolated Benefit (no discounting).
// Fail-closed on a non-finite result: a poisoned signal collapses to 0 so it can never win
// a slot over a clean candidate.
func coverageMarginal(c Candidate, covered map[string]bool) float64 {
	var marg float64
	if c.coverQTotal > 0 {
		fresh := 0
		for t := range c.coverHit {
			if !covered[t] {
				fresh++
			}
		}
		marg = c.coverRelWeight*float64(fresh)/float64(c.coverQTotal) + c.coverRest
	} else {
		marg = c.Benefit
	}
	if math.IsNaN(marg) || math.IsInf(marg, 0) {
		return 0
	}
	return marg
}

// coverageTieBreak reports whether a should be preferred over b given equal density. It
// breaks the tie by higher marginal benefit, then lower step, then lower ID — the same
// deterministic total order the greedy planner uses.
func coverageTieBreak(a, b Candidate, margA, margB float64) bool {
	if margA != margB {
		return margA > margB
	}
	return candidateStepIDLess(a, b)
}

func selectionOf(c Candidate, pinned bool) Selection {
	return Selection{
		ID: c.Cell.ID, Step: c.Cell.Step, Role: c.Cell.Role, Descriptor: c.Cell.Descriptor,
		Area: c.Area, Precision: c.Precision,
		Cost: c.Cost, Benefit: c.Benefit, Density: c.density(), Pinned: pinned,
	}
}

func elisionOf(c Candidate, reason string) Elision {
	return Elision{
		ID: c.Cell.ID, Step: c.Cell.Step, Role: c.Cell.Role, Area: c.Area, Precision: c.Precision, Digest: c.Cell.Digest,
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
