// Package spendrollup builds the cross-account `fak spend` rollup: one or more labeled
// spend figures per account, each carrying a valuation basis AND a WITNESSED/OBSERVED
// provenance label, plus a gate that fails any figure that forgets either label.
//
// It applies the conflation-scorecard's provenance discipline (WITNESSED vs OBSERVED,
// internal/conflationscore) to spend. The Hermes inspiration
// (agent/credential_pool.py, credits_tracker.py, account_usage.py) pools and tracks spend
// across accounts, but it relays the PROVIDER's own billed number as if it were an
// authored fact — easy to conflate "what we spent" with "what the provider billed". fak
// refuses that conflation: a figure fak measured locally is WITNESSED; a figure fak merely
// relays from a provider is OBSERVED, and the two are never folded into one bare "spend"
// number without their labels. The gate is the enforcement — it fails an unlabeled figure
// exactly the way `/conflation-score` fails an unlabeled external value.
package spendrollup

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

// Schema is the rollup's stable identifier for JSON consumers.
//
// Invariant: Payload consumers inspect Schema to detect serialization and format drift.
// Guard: Any mismatch against Schema indicates unsupported version skew.
const Schema = "fak-spend-rollup/1"

// Provenance labels WHO authored a spend figure — the same WITNESSED/OBSERVED axis the
// conflation scorecard enforces on every reported number.
//
// Invariant: Every spend figure in a verified rollup must carry either Witnessed or Observed provenance.
// Guard: Gate fails closed on empty, unknown, or unclassified provenance values.
type Provenance string

const (
	// Witnessed marks a figure fak MEASURED locally and stands behind (fak authored it).
	//
	// Invariant: Author-attributed to fak; valued strictly on locally observed net usage.
	// Guard: Never applied to external provider-billed dollar amounts.
	Witnessed Provenance = "WITNESSED"
	// Observed marks a figure fak RELAYS from an external provider (fak did not author it).
	//
	// Invariant: Relayed as-is from external telemetry; zero fak authorial claim.
	// Guard: Never conflated into fak-authored session or compute totals.
	Observed Provenance = "OBSERVED"
)

// ValidProvenance reports whether p is a recognized WITNESSED/OBSERVED label. An empty or
// unknown provenance is what the gate fails.
//
// Invariant: Returns true strictly for canonical Witnessed or Observed values.
// Guard: Fails closed (false) on empty strings, lower-case forms, or undefined provenance tokens.
func ValidProvenance(p Provenance) bool { return p == Witnessed || p == Observed }

// ValuationBasis names the pricing basis a spend figure was valued on, so a session count
// fak authored is never silently read as a provider-billed dollar. The token enum reuses
// the conflation scorecard's valuation-basis vocabulary (#2796).
//
// Invariant: Every spend figure must explicitly define its measurement or pricing basis.
// Guard: Unspecified or unrecognized bases are rejected fail-closed by Gate.
type ValuationBasis string

const (
	// BasisObservedNet is fak's own locally observed net usage (a WITNESSED basis).
	//
	// Invariant: Represents internal metrics directly metered by fak.
	// Guard: Must only pair with Witnessed provenance figures.
	BasisObservedNet ValuationBasis = "OBSERVED_NET"
	// BasisProviderBilled is the provider's own billed number, relayed (an OBSERVED basis).
	//
	// Invariant: External billing ledger quantity reported by the remote vendor.
	// Guard: Must only pair with Observed provenance figures.
	BasisProviderBilled ValuationBasis = "PROVIDER_BILLED"
	// BasisFullInput prices a token at 1.0x full-input.
	//
	// Invariant: Baseline un-discounted token input accounting.
	// Guard: Cannot be combined with marginal read basis in subtotal aggregation.
	BasisFullInput ValuationBasis = "FULL_INPUT"
	// BasisCacheReadMarginal prices a token at 0.1x cache-read marginal.
	//
	// Invariant: Applies solely to verified prompt cache hits.
	// Guard: Cannot be folded into un-discounted full input subtotals.
	BasisCacheReadMarginal ValuationBasis = "CACHE_READ_MARGINAL"
)

// ValidBasis reports whether b is a recognized valuation basis. An empty or unknown basis
// is what the gate fails.
//
// Invariant: Returns true strictly for declared members of ValuationBasis.
// Guard: Fails closed (false) for unrecognized, blank, or malformed basis tokens.
func ValidBasis(b ValuationBasis) bool {
	switch b {
	case BasisObservedNet, BasisProviderBilled, BasisFullInput, BasisCacheReadMarginal:
		return true
	default:
		return false
	}
}

// Figure is one labeled spend number in the rollup. Amount is deliberately paired with a
// Unit so a "live-sessions" count is never conflated with a provider dollar.
//
// Invariant: Amount is dimensionless without pairing to Unit, Basis, and Provenance.
// Guard: Gate verifies that Provenance and Basis satisfy ValidProvenance and ValidBasis.
type Figure struct {
	Account    string         `json:"account"`
	Provider   string         `json:"provider"`
	Label      string         `json:"label"`
	Amount     float64        `json:"amount"`
	Unit       string         `json:"unit"`
	Basis      ValuationBasis `json:"valuation_basis"`
	Provenance Provenance     `json:"provenance"`
	Source     string         `json:"source"`
}

// Subtotal folds figures that share a (provenance, unit, basis) key. There is deliberately
// NO single grand total: summing a WITNESSED session count and an OBSERVED provider-window
// state into one number is the exact conflation this rollup exists to prevent.
//
// Invariant: Subtotals partition figures strictly by the tuple (Provenance, Unit, Basis).
// Guard: Prevents dimensional collapse; cross-provenance or cross-basis summation is refused.
type Subtotal struct {
	Provenance Provenance     `json:"provenance"`
	Unit       string         `json:"unit"`
	Basis      ValuationBasis `json:"valuation_basis"`
	Amount     float64        `json:"amount"`
	Figures    int            `json:"figures"`
}

// Rollup is the cross-account spend rollup behind `fak spend`.
//
// Invariant: Schema is initialized to Schema ("fak-spend-rollup/1") by Build.
// Guard: Callers must invoke Gate() to ensure no unlabeled or malformed figures are accepted.
type Rollup struct {
	Schema    string     `json:"schema"`
	Figures   []Figure   `json:"figures"`
	Subtotals []Subtotal `json:"subtotals"`
	Warnings  []string   `json:"warnings"`
}

// Build folds the annotated account roster into a labeled spend rollup. Every figure Build
// emits is labeled by construction, so Build's output always passes Gate; the gate exists to
// catch a hand-constructed or future figure that forgets its label. Only routable worker
// rows contribute — duplicate/excluded/non-account rows carry no spend of their own.
//
// Invariant: Figures and subtotals are generated strictly from routable worker accounts.
// Guard: Non-routable accounts (duplicates, excluded identities, non-accounts) are filtered out fail-closed.
func Build(rows []fleetaccounts.Account) Rollup {
	rollup := Rollup{Schema: Schema, Figures: []Figure{}, Subtotals: []Subtotal{}, Warnings: []string{}}
	for _, row := range rows {
		if !fleetaccounts.RoutableWorker(row) {
			continue
		}
		provider := fleetaccounts.ProviderFamily(row)

		// WITNESSED: fak's own count of the sessions it is running against this account.
		// fak authored this number (it counts its own dispatches), so it is a figure fak
		// stands behind — valued as fak's locally observed net, NOT as provider dollars.
		rollup.Figures = append(rollup.Figures, Figure{
			Account:    row.Account,
			Provider:   provider,
			Label:      "fak live-session load",
			Amount:     float64(intVal(row.LiveSessions)),
			Unit:       "live-sessions",
			Basis:      BasisObservedNet,
			Provenance: Witnessed,
			Source:     "fak local session ledger (sessions.json)",
		})

		// OBSERVED: a provider-reported usage-window posture, relayed. This is the
		// provider's own number — fak did not author it and attributes nothing about it to
		// a fak action — so it is labeled OBSERVED / PROVIDER_BILLED. Emitted only when the
		// provider actually reported a posture (throttle / weekly window / reset).
		if providerUsageReported(row) {
			rollup.Figures = append(rollup.Figures, Figure{
				Account:    row.Account,
				Provider:   provider,
				Label:      "provider-reported usage-window state",
				Amount:     1,
				Unit:       "provider-usage-window",
				Basis:      BasisProviderBilled,
				Provenance: Observed,
				Source:     "provider-reported usage window (relayed; not a fak claim)",
			})
		}
	}
	rollup.Subtotals = subtotals(rollup.Figures)
	rollup.Warnings = warnings(rollup.Figures)
	return rollup
}

// Gate fails the rollup if ANY figure is unlabeled — missing a WITNESSED/OBSERVED
// provenance label OR missing/unknown a valuation basis. It returns nil when every figure
// is labeled. This is the "gate failing an unlabeled spend figure" witness (issue #2903).
//
// Invariant: A valid rollup has zero unlabeled figures across its entire Figures slice.
// Guard: Fails closed on any invalid Provenance or ValuationBasis; returns aggregated defects.
func (r Rollup) Gate() error {
	var defects []string
	for i, f := range r.Figures {
		id := f.Account
		if id == "" {
			id = fmt.Sprintf("figure[%d]", i)
		}
		if !ValidProvenance(f.Provenance) {
			defects = append(defects, fmt.Sprintf(
				"%s %q: spend figure has no WITNESSED/OBSERVED provenance label (got %q)",
				id, f.Label, f.Provenance))
		}
		if !ValidBasis(f.Basis) {
			defects = append(defects, fmt.Sprintf(
				"%s %q: spend figure names no valuation basis (got %q)",
				id, f.Label, f.Basis))
		}
	}
	if len(defects) == 0 {
		return nil
	}
	return fmt.Errorf("%d unlabeled spend figure(s):\n  %s",
		len(defects), strings.Join(defects, "\n  "))
}

func providerUsageReported(row fleetaccounts.Account) bool {
	return boolVal(row.Throttled) ||
		strings.TrimSpace(strVal(row.Weekly)) != "" ||
		strings.TrimSpace(strVal(row.Reset)) != "" ||
		strings.EqualFold(strings.TrimSpace(strVal(row.BlockKind)), "usage")
}

func subtotals(figures []Figure) []Subtotal {
	type key struct {
		p Provenance
		u string
		b ValuationBasis
	}
	byKey := map[key]*Subtotal{}
	var order []key
	for _, f := range figures {
		k := key{f.Provenance, f.Unit, f.Basis}
		sub, ok := byKey[k]
		if !ok {
			sub = &Subtotal{Provenance: f.Provenance, Unit: f.Unit, Basis: f.Basis}
			byKey[k] = sub
			order = append(order, k)
		}
		sub.Amount += f.Amount
		sub.Figures++
	}
	out := make([]Subtotal, 0, len(order))
	for _, k := range order {
		out = append(out, *byKey[k])
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Provenance != out[j].Provenance {
			return out[i].Provenance < out[j].Provenance
		}
		return out[i].Unit < out[j].Unit
	})
	return out
}

func warnings(figures []Figure) []string {
	var out []string
	hasObserved := false
	for _, f := range figures {
		if f.Provenance == Observed {
			hasObserved = true
			break
		}
	}
	if len(figures) > 0 && !hasObserved {
		out = append(out, "no provider-relayed figure present: fak holds no WITNESSED provider-dollar spend, so no dollar figure here is fak-authored")
	}
	return out
}

// Render draws the compact human rollup. It leads with the provenance discipline so a
// reader is never left to guess whether a number is fak-authored or provider-relayed.
//
// Invariant: Formats figures and subtotals with provenance explicitly rendered in column 1.
// Guard: When Figures is empty, prints a safe notice rather than invalid or empty tables.
func Render(r Rollup) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak spend — cross-account rollup (%s)\n", r.Schema)
	b.WriteString("  every figure labels its provenance: WITNESSED = fak authored, OBSERVED = provider-relayed\n\n")
	if len(r.Figures) == 0 {
		b.WriteString("  (no routable worker accounts)\n")
		return b.String()
	}
	b.WriteString("figures:\n")
	for _, f := range r.Figures {
		fmt.Fprintf(&b, "  [%-9s] %-18s %-30s %10.2f %-22s basis=%-16s %s\n",
			f.Provenance, f.Provider, f.Account, f.Amount, f.Unit, f.Basis, f.Label)
	}
	b.WriteString("\nsubtotals (never summed across provenance/unit):\n")
	for _, s := range r.Subtotals {
		fmt.Fprintf(&b, "  %-9s %-22s basis=%-16s %10.2f over %d figure(s)\n",
			s.Provenance, s.Unit, s.Basis, s.Amount, s.Figures)
	}
	if len(r.Warnings) > 0 {
		b.WriteString("\nnotes:\n")
		for _, w := range r.Warnings {
			fmt.Fprintf(&b, "  %s\n", w)
		}
	}
	return b.String()
}

func intVal(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}

func boolVal(p *bool) bool {
	if p == nil {
		return false
	}
	return *p
}

func strVal(p *string) string {
	if p == nil {
		return ""
	}
	return *p
}
