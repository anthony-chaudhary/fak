package ctxplan

import (
	"fmt"
	"math"
	"strings"
)

// Regime is one strategy for managing a growing session's context. The three span the
// design space the planned view sits between (see the package doc): the two extremes
// each sacrifice one of {bounded resident tokens, exact recall}; the planned view keeps
// both, paying only a bounded forecast-miss rate.
type Regime int

const (
	// Linear keeps the whole transcript resident. Resident tokens grow Theta(N) with the
	// turn count; recall is exact (it is all there); it blows any finite window.
	Linear Regime = iota
	// Compaction summarizes at a cap and DROPS the originals. Resident tokens stay
	// bounded by the working set W, but recall of an arbitrary past fact decays
	// geometrically as it survives more compactions — and it is irreversible.
	Compaction
	// Planned keeps an O(1) resident VIEW plus the lossless store behind it, re-planned
	// each turn. Resident tokens stay bounded by W AND recall stays exact (any elided
	// span pages back in); the only cost is the forecast-MISS rate — a cheap page fault,
	// never a lost fact.
	Planned
)

func (r Regime) String() string {
	switch r {
	case Linear:
		return "linear"
	case Compaction:
		return "compaction"
	case Planned:
		return "planned"
	default:
		return "unknown"
	}
}

// ResidentBigO is the asymptotic resident-token growth of the regime in N (turns).
func (r Regime) ResidentBigO() string {
	switch r {
	case Linear:
		return "Θ(N)"
	case Compaction, Planned:
		return "Θ(1)"
	default:
		return "?"
	}
}

// RecallBigO is the asymptotic EXACT-recall fidelity of the regime as N grows: the
// probability an arbitrary past fact is still recoverable byte-exact.
func (r Regime) RecallBigO() string {
	switch r {
	case Linear, Planned:
		return "1.0"
	case Compaction:
		return "ρ^Θ(N) → 0"
	default:
		return "?"
	}
}

// Params are the inputs to the scaling model. They are deterministic constants — there
// is no wall clock and no randomness, so Model is fully reproducible and testable (the
// repo's "no Date.now in a model" rule).
type Params struct {
	// TokensPerTurn (b) is the mean number of resident tokens a turn adds to the context
	// in the linear regime — the per-turn growth rate.
	TokensPerTurn float64
	// WorkingSet (W) is the O(1) resident cap for the Compaction and Planned regimes —
	// the constant the resident context never exceeds.
	WorkingSet int
	// ForecastHit (p_hit, in [0,1]) is the planner's forecast quality: the fraction of
	// turns whose needed span was already resident (no page fault). Planned regime only.
	ForecastHit float64
	// Retain (ρ, in (0,1]) is the fraction of a fact's detail a single compaction keeps.
	// Compaction regime only; a fact that has survived k compactions retains ρ^k.
	Retain float64
}

// Point is the modeled cost/quality of a regime at one turn count N.
type Point struct {
	Turn           int     `json:"turn"`            // N
	Resident       int     `json:"resident"`        // C(N): tokens that must sit in the live window
	Store          int64   `json:"store"`           // lossless backing store on disk (cold)
	RecallExact    float64 `json:"recall_exact"`    // P(an arbitrary past fact is recoverable byte-exact)
	RetrieveFaults float64 `json:"retrieve_faults"` // expected demand-pages so far (Planned): (1-p_hit)*N
	PromptCostCum  int64   `json:"prompt_cost_cum"` // cumulative resident tokens processed — the prefill/$ proxy
}

// Model computes the regime's curve at each turn count in turns (which must be positive
// and is treated as arbitrary, e.g. {50, 100, 1000, 10000, 1000000}). It is the
// instrument behind the scaling-law claim: it makes the resident-vs-turns curve and the
// recall-fidelity term reproducible numbers rather than asserted ones.
func Model(r Regime, p Params, turns []int) []Point {
	out := make([]Point, 0, len(turns))
	for _, n := range turns {
		if n < 1 {
			continue
		}
		out = append(out, modelAt(r, p, n))
	}
	return out
}

func modelAt(r Regime, p Params, n int) Point {
	b := p.TokensPerTurn
	W := float64(p.WorkingSet)
	pt := Point{Turn: n}
	linearTokens := b * float64(n) // tokens a fully-resident transcript would carry at turn N

	switch r {
	case Linear:
		pt.Resident = int(math.Round(linearTokens))
		pt.Store = 0 // nothing is paged out; it is all resident
		pt.RecallExact = 1.0
		pt.PromptCostCum = cumLinear(b, n) // Σ b*i = b*N*(N+1)/2 — the Θ(N²) prefill tax
	case Compaction:
		pt.Resident = int(math.Round(math.Min(linearTokens, W)))
		pt.Store = 0 // the originals were dropped — lossy
		pt.RecallExact = compactionRecall(p, n)
		pt.PromptCostCum = cumCapped(b, p.WorkingSet, n)
	case Planned:
		pt.Resident = int(math.Round(math.Min(linearTokens, W)))
		pt.Store = int64(math.Round(linearTokens)) // every turn's bytes preserved (lossless, cold)
		pt.RecallExact = 1.0                       // any elided span pages back in — exact by reference
		hit := clamp01(p.ForecastHit)
		pt.RetrieveFaults = (1 - hit) * float64(n) // misses are cheap page faults, not lost facts
		pt.PromptCostCum = cumCapped(b, p.WorkingSet, n)
	}
	return pt
}

// compactionRecall models the geometric decay of exact recall under compaction: a fact
// laid down early has survived ~floor(b*N/W) compaction events by turn N (each time the
// resident set refilled to W), retaining ρ per event. With ρ=1 (lossless summary — the
// degenerate case) recall stays 1.0; with ρ<1 it decays toward 0, the irreversible loss
// the planned view avoids.
func compactionRecall(p Params, n int) float64 {
	rho := p.Retain
	if rho <= 0 {
		rho = 0.5 // a missing ρ defaults to "half the detail survives each compaction"
	}
	if rho > 1 {
		rho = 1
	}
	if p.WorkingSet <= 0 {
		return rho
	}
	compactions := math.Floor(p.TokensPerTurn * float64(n) / float64(p.WorkingSet))
	if compactions < 0 {
		compactions = 0
	}
	return math.Pow(rho, compactions)
}

// cumLinear is Σ_{i=1..n} b*i = b*n*(n+1)/2 — the cumulative resident tokens a linear
// transcript forces the model to (re)process, the Θ(N²) prefill tax that makes long
// linear sessions intractable.
func cumLinear(b float64, n int) int64 {
	return int64(math.Round(b * float64(n) * float64(n+1) / 2))
}

// cumCapped is Σ_{i=1..n} min(b*i, W): the resident set grows linearly until it hits the
// cap W at turn k=ceil(W/b), then stays flat at W. So the cumulative cost is the small
// triangle below the cap plus a W-by-(n-k) rectangle — Θ(W·N), linear in N, the bent
// curve the cap buys. Both Compaction and Planned pay this resident cost; they differ
// only in recall fidelity and the (cheap, separate) retrieve faults.
func cumCapped(b float64, wInt, n int) int64 {
	if wInt <= 0 || b <= 0 {
		return cumLinear(b, n)
	}
	W := float64(wInt)
	k := int(math.Ceil(W / b)) // first turn at which b*i >= W (the cap engages)
	if k > n {
		k = n
	}
	// Σ_{i=1..k} b*i  (still below the cap)  +  W*(n-k)  (flat at the cap)
	below := b * float64(k) * float64(k+1) / 2
	flat := W * float64(n-k)
	return int64(math.Round(below + flat))
}

// Compare runs all three regimes over the same turn schedule and returns a parallel
// table — the headline artifact for the scaling-law doc and the demo. The slices are
// index-aligned with turns.
type Comparison struct {
	Turns      []int   `json:"turns"`
	Linear     []Point `json:"linear"`
	Compaction []Point `json:"compaction"`
	Planned    []Point `json:"planned"`
}

// Compare computes the three-regime comparison for the given params and turn schedule.
func Compare(p Params, turns []int) Comparison {
	return Comparison{
		Turns:      append([]int(nil), turns...),
		Linear:     Model(Linear, p, turns),
		Compaction: Model(Compaction, p, turns),
		Planned:    Model(Planned, p, turns),
	}
}

// Table renders the comparison as an operator-readable block: resident tokens and exact
// recall per regime at each horizon, so the "bend" is visible at a glance.
func (c Comparison) Table() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%-12s | %-14s %-8s | %-14s %-8s | %-14s %-8s %-10s\n",
		"turns", "linear-resident", "recall", "compact-resident", "recall", "planned-resident", "recall", "faults")
	b.WriteString(strings.Repeat("-", 104) + "\n")
	for i, n := range c.Turns {
		l, cm, pl := c.Linear[i], c.Compaction[i], c.Planned[i]
		fmt.Fprintf(&b, "%-12s | %-14s %-8.3f | %-14s %-8.3f | %-14s %-8.3f %-10s\n",
			human(n),
			human(l.Resident), l.RecallExact,
			human(cm.Resident), cm.RecallExact,
			human(pl.Resident), pl.RecallExact, human(int(pl.RetrieveFaults)))
	}
	return b.String()
}

// human renders an int with K/M/B suffixes so a 700,000,000-token resident set is
// readable in the table.
func human(n int) string {
	switch {
	case n >= 1_000_000_000:
		return fmt.Sprintf("%.1fB", float64(n)/1e9)
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(n)/1e6)
	case n >= 1_000:
		return fmt.Sprintf("%.1fK", float64(n)/1e3)
	default:
		return fmt.Sprintf("%d", n)
	}
}

func clamp01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}
