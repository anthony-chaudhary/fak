// authorityview.go is the generation-aware benchmark authority VIEW (issue #1669,
// epic #1625, stream gen/future) as an EXECUTABLE fold — the computed form of the
// design memo docs/generation-benchmark-authority-view.md.
//
// The memo answers the one question BENCHMARK-AUTHORITY.md raises but does not
// resolve: the authority table proves each number is TRUE; it does not say what
// each number ENTITLES you to claim. Does a committed number prove CURRENT VALUE,
// or FUTURE POTENTIAL? The memo slices every row of the typed benchmark registry
// (docs/benchmarks/registry.jsonl) by three axes:
//
//   - witness type       — the provenance rung the row's evidence clears
//     (measured / functional / modeled / unknown).
//   - entitled horizon   — the strongest generation that witness supports, DERIVED
//     from the witness (never declared by prose): now / next / second-next / future.
//   - promotion relevance — whether the number moves a claim toward now (PROMOTING),
//     defends one already there (HOLDING), is withheld pending a gate (GATED), or is
//     a tombstone (RETIRED). A closed four-token set; every claim maps to exactly one.
//
// The load-bearing invariant is `claimed <= entitled`: a row's CITABILITY (its
// status, which a citing agent reads first) may never outrank what its WITNESS
// entitles it to claim. A top-tier "quote this" row (status canonical) whose witness
// is entitled below gen/now is HORIZON LAUNDERING — the memo's one confirmed
// structural hit is the single `canonical x modeled` cell (webbench-webvoyager-hero):
// filed at the tier-1 quote status while its witness is a geometry MODEL entitled
// only to gen/future.
//
// WHY THIS FILE. The memo's own promotion evidence is "the view is COMPUTED, not
// merely described" (its future->second-next->next path). This fold promotes it: the
// three-axis classification is now a pure, re-runnable function bound by
// authorityview_test.go to the LIVE committed registry, so an edit that breaks the
// memo's stated finding reds the bench gate instead of silently drifting the prose.
// It is the memo's named "cheapest first gate" (handoff step 2) — a check that reads
// the status/provenance pair no view previously reconciled.
//
// It stays doc-adjacent and INERT by construction: no runtime gate, no CI wiring, no
// default exposure, no new provenance vocabulary — matching the memo's gen/future
// status and the generation orthogonality restated in metrics.OrthogonalityNote,
// which this view reuses rather than re-authoring (single source of truth).
package bench

import (
	"bufio"
	"encoding/json"
	"io"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/metrics"
)

// AuthorityClaim is the subset of a docs/benchmarks/registry.jsonl row this view
// reads. The registry row carries more (headline, model, baseline, artifact, ...);
// the horizon derivation depends ONLY on the two independent enums that set the
// witness and the citability, plus the id and the named blocker issue. The view
// introduces no new provenance words — it reads what the registry already carries.
type AuthorityClaim struct {
	// ID is the registry row's stable kebab-case slug.
	ID string `json:"id"`
	// Status is the citability / governance word: canonical | live | gated |
	// pending | stale | retracted. A citing agent reads this FIRST.
	Status string `json:"status"`
	// Provenance is the witness rung: measured | functional | modeled | unknown.
	Provenance string `json:"provenance"`
	// Issue is the named blocker/gate this claim would retire, or nil.
	Issue *int `json:"issue"`
}

// Horizon values. The four live horizons are exactly metrics.RoadmapGenerations
// (now -> future), reused so the bench view and the roadmap dashboard cannot drift
// their horizon vocabulary. HorizonRetired is the sentinel for a tombstone row that
// entitles no live horizon.
const HorizonRetired = "retired"

// Promotion-relevance tokens — the closed four-token set from the memo's axis 3.
// A value outside this set is a bug, not a new token.
const (
	// RelPromoting: the number is the witness that would retire a named blocker and
	// move a claim one generation toward now.
	RelPromoting = "PROMOTING"
	// RelHolding: the number substantiates a claim already at its entitled horizon —
	// it defends the row, it does not advance it.
	RelHolding = "HOLDING"
	// RelGated: the row exists deliberately with its number withheld pending a gate.
	RelGated = "GATED"
	// RelRetired: stale, superseded, or retracted — zero promotion relevance.
	RelRetired = "RETIRED"
)

// horizonRank orders the horizons by how strong a claim they license (now is the
// strongest current-value claim). Laundering is claimed-rank strictly above
// entitled-rank; a retired tombstone claims nothing and ranks below every horizon.
var horizonRank = map[string]int{
	HorizonRetired: -1,
	"future":       0,
	"second-next":  1,
	"next":         2,
	"now":          3,
}

// EntitledHorizon derives the strongest generation the claim's WITNESS supports —
// the memo's "entitled horizon read straight off the benchmark column of the witness
// ladder." It is a pure function of (status, provenance), never of prose:
//
//   - a tombstone (stale/retracted) entitles no live horizon;
//   - a deliberately-withheld number (gated/pending) entitles only gen/future — the
//     absence of a number published with the condition that would produce one;
//   - a MODELED geometry/projection is a floor or a plan, never a measured headline
//     -> gen/future;
//   - a FUNCTIONAL witness (real execution / behavior, but not a captured perf
//     default) -> gen/next;
//   - a MEASURED, reproducible number with a baseline -> gen/now (current value).
//
// LIMITATION (memo assumption 1, restated in code): the registry's coarse provenance
// enum does not separate a gen/second-next simulation/fixture rung from gen/next, so
// this derivation never emits "second-next". A future per-claim (not per-row) witness
// field is what would recover that rung; until then the collapse is stated, not hidden.
func (c AuthorityClaim) EntitledHorizon() string {
	switch c.Status {
	case "stale", "retracted":
		return HorizonRetired
	case "gated", "pending":
		return "future"
	}
	switch c.Provenance {
	case "measured":
		return "now"
	case "functional":
		return "next"
	default: // modeled, unknown
		return "future"
	}
}

// ClaimedHorizon is the horizon a row's CITABILITY asserts — what a citing agent
// reads off the status field before any prose fence. Only `canonical` — the tier-1
// "the number to quote" status — asserts a current headline (gen/now) by status
// alone, so it is the one status that can silently outrank its witness. `live` is a
// weaker, self-fencing citability: it claims exactly what its witness entitles (it
// does not inflate by status), which is why the memo's confirmed laundering hit is a
// single `canonical` row, not all six citable-modeled rows. `gated`/`pending` publish
// an explicitly withheld future number; a tombstone claims nothing.
func (c AuthorityClaim) ClaimedHorizon() string {
	switch c.Status {
	case "canonical":
		return "now"
	case "gated", "pending":
		return "future"
	case "stale", "retracted":
		return HorizonRetired
	default: // live — self-fencing: claims exactly its witness entitlement
		return c.EntitledHorizon()
	}
}

// PromotionRelevance maps the claim onto the closed four-token relevance set.
func (c AuthorityClaim) PromotionRelevance() string {
	switch c.Status {
	case "stale", "retracted":
		return RelRetired
	case "gated", "pending":
		return RelGated
	default: // canonical, live
		if c.Issue != nil {
			return RelPromoting // a named blocker this number would retire
		}
		return RelHolding
	}
}

// Laundered reports the memo's mechanically-checkable `claimed <= entitled`
// violation: the row's citability outranks what its witness entitles. This is the
// horizon-laundering failure the view exists to surface on the authority rows
// themselves. GATED and tombstone rows are correctly never flagged (their claimed
// and entitled horizons match by construction).
func (c AuthorityClaim) Laundered() bool {
	return horizonRank[c.ClaimedHorizon()] > horizonRank[c.EntitledHorizon()]
}

// ProvesCurrentValue is the memo's headline rule: a row proves CURRENT VALUE iff its
// entitled horizon is gen/now. Every other row proves FUTURE POTENTIAL. This does not
// rank the two (a gen/now row can record a loss; a gen/future witness can be
// bit-exact) — it only says they answer different questions.
func (c AuthorityClaim) ProvesCurrentValue() bool {
	return c.EntitledHorizon() == "now"
}

// AuthorityRow is one classified registry row — the executable form of a cell in the
// memo's "The view, against real rows" table.
type AuthorityRow struct {
	ID                 string `json:"id"`
	Witness            string `json:"witness"`             // provenance rung
	ClaimedHorizon     string `json:"claimed_horizon"`     // read off status
	EntitledHorizon    string `json:"entitled_horizon"`    // derived from witness
	PromotionRelevance string `json:"promotion_relevance"` // closed four-token set
	Laundered          bool   `json:"laundered"`           // claimed > entitled
	ProvesCurrentValue bool   `json:"proves_current_value"`
}

// AuthorityView is the computed generation-aware view over the whole registry: the
// per-row classification plus the aggregate slices the memo's first-classification
// pass names. It is a pure fold of the input claims — no clock, no disk — so a test
// can assert its exact shape and a future gen/next verb can render it.
type AuthorityView struct {
	// OrthogonalityNote restates how generation stays orthogonal to priority, shared
	// trunk, and runtime feature gates. Reused from metrics so the statement has one
	// source of truth across the bench view and the roadmap dashboard.
	OrthogonalityNote string `json:"orthogonality_note"`
	Total             int    `json:"total"`
	// Rows are the classified registry rows in stable ID order.
	Rows []AuthorityRow `json:"rows"`
	// ByEntitled / ByRelevance are the closed-vocabulary histograms (every row lands
	// in exactly one bucket of each).
	ByEntitled  map[string]int `json:"by_entitled_horizon"`
	ByRelevance map[string]int `json:"by_promotion_relevance"`
	// LaunderedIDs are the rows whose citability outranks their witness — the memo's
	// confirmed horizon-laundering hits.
	LaunderedIDs []string `json:"laundered_ids"`
	// CitableModeled counts rows carrying a MODELED witness in a citable status
	// (canonical or live) — the broader watch surface the memo names beyond the
	// confirmed laundering hits.
	CitableModeled int `json:"citable_modeled"`
	// UnknownProvenance / CurrentValue / FuturePotential are the memo's headline counts.
	UnknownProvenance int `json:"unknown_provenance"`
	CurrentValue      int `json:"current_value"`
	FuturePotential   int `json:"future_potential"`
}

// ClassifyAuthorityRegistry folds a set of registry claims into the generation-aware
// authority view. Pure and deterministic: rows are emitted in stable ID order and two
// calls on the same input are identical.
func ClassifyAuthorityRegistry(claims []AuthorityClaim) AuthorityView {
	v := AuthorityView{
		OrthogonalityNote: metrics.OrthogonalityNote,
		Total:             len(claims),
		ByEntitled:        map[string]int{},
		ByRelevance:       map[string]int{},
	}
	sorted := make([]AuthorityClaim, len(claims))
	copy(sorted, claims)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].ID < sorted[j].ID })

	for _, c := range sorted {
		row := AuthorityRow{
			ID:                 c.ID,
			Witness:            c.Provenance,
			ClaimedHorizon:     c.ClaimedHorizon(),
			EntitledHorizon:    c.EntitledHorizon(),
			PromotionRelevance: c.PromotionRelevance(),
			Laundered:          c.Laundered(),
			ProvesCurrentValue: c.ProvesCurrentValue(),
		}
		v.Rows = append(v.Rows, row)
		v.ByEntitled[row.EntitledHorizon]++
		v.ByRelevance[row.PromotionRelevance]++
		if row.Laundered {
			v.LaunderedIDs = append(v.LaunderedIDs, c.ID)
		}
		if c.Provenance == "modeled" && (c.Status == "canonical" || c.Status == "live") {
			v.CitableModeled++
		}
		if c.Provenance == "unknown" {
			v.UnknownProvenance++
		}
		if row.ProvesCurrentValue {
			v.CurrentValue++
		} else {
			v.FuturePotential++
		}
	}
	return v
}

// LoadAuthorityRegistry parses the JSONL benchmark registry (one AuthorityClaim per
// non-blank line) from r. It reads only the fields the view needs; unknown fields are
// ignored. Blank lines are skipped so the committed docs/benchmarks/registry.jsonl
// feeds straight in.
func LoadAuthorityRegistry(r io.Reader) ([]AuthorityClaim, error) {
	var out []AuthorityClaim
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		trimmed := len(line)
		for trimmed > 0 && (line[trimmed-1] == ' ' || line[trimmed-1] == '\t' || line[trimmed-1] == '\r') {
			trimmed--
		}
		if trimmed == 0 {
			continue
		}
		var c AuthorityClaim
		if err := json.Unmarshal(line[:trimmed], &c); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return out, nil
}
