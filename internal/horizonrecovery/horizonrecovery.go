// Package horizonrecovery grounds the budget-recovery term r of the horizon
// multiplier in docs/explainers/compounding-benefits-of-a-saved-call.md, from a
// REAL ctxplanbench replay over real Claude Code transcripts.
//
// The compounding-benefits doc states effective_horizon = budget /
// effective_cost_per_call and the gain r/d, where r > 1 is the budget a long
// session keeps getting back (window/resident slots reclaimed by the bounded
// ctxplan view). The doc deliberately ships NO number for r, because quoting it
// soundly needs a task-success eval proving the reclaimed budget did not cost an
// answer. This package keeps that discipline: it surfaces r's MEASURED INPUTS and
// the recall-faithfulness FENCE that travels with them, and it STRUCTURALLY
// refuses to multiply them into a printed r.
//
// What it reads: a `fak ctxplanbench --out report.json` artifact, whose `total`
// (and per-session `sessions[]`) rows already carry, measured over real
// transcripts:
//   - linear_cum_tokens vs compact_cum_tokens -- the resident-token budget the
//     linear transcript would have forced vs what the bounded view actually held.
//     Their ratio is the budget-recovery OPERAND (kept as two fields + the ratio,
//     never multiplied by anything else).
//   - fault_rate / served / refused / fault_tax_cum -- the FENCE. A reclaim that
//     elided a span a later turn referenced is a fault; served = recovered via
//     DemandPage, refused = the gate held. A high recovery ratio is meaningless
//     without its fault rate, so the two NEVER ship apart.
//   - compaction_loss_turns / facts_recovered -- the faithfulness witness: turns a
//     naive compaction destroyed a fact that the planned view recovered.
//
// The honesty contract is enforced by Selfcheck: the emitted band has no `r` and
// no `horizon_multiplier`; the recovery ratio and the fault fence co-occur; and a
// single session never prints a population claim (an aggregate band requires a
// session-count floor, mirroring the doc's own 20/100-session cohorts).
package horizonrecovery

import (
	"encoding/json"
	"fmt"
	"math"
)

// jsonKeys marshals v and returns the set of its top-level JSON object keys, so a
// structural assertion can prove a forbidden field (r, horizon_multiplier) is NOT
// present -- and would catch a future field addition that smuggled one in.
func jsonKeys(v any) (map[string]bool, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	keys := make(map[string]bool, len(m))
	for k := range m {
		keys[k] = true
	}
	return keys, nil
}

// MinAggregateSessions is the floor below which an aggregate recovery band is
// refused: one transcript's gap is not the population claim r makes. The doc's own
// committed practice uses 20- and 100-session cohorts (o1-context-window-economics.md).
const MinAggregateSessions = 20

// Session mirrors the slice of a ctxplanbench sessionResult this lens reads. Field
// tags match cmd/ctxplanbench exactly so a report decodes without adaptation. Every
// field is already MEASURED on a real transcript upstream; nothing new is measured.
type Session struct {
	Source string `json:"source"`
	Turns  int    `json:"turns"`
	Budget int    `json:"budget"`

	// budget-recovery OPERANDS (resident-token cost, cumulative).
	LinearCumTok  int64 `json:"linear_cum_tokens"`
	CompactCumTok int64 `json:"compact_cum_tokens"`
	PlannedCumTok int64 `json:"planned_cum_tokens"`
	FaultTaxCum   int64 `json:"fault_tax_cum"`

	// the FENCE (forecast-miss cost on the real reference signal).
	References int     `json:"references"`
	Faults     int     `json:"faults"`
	Served     int     `json:"served"`
	Refused    int     `json:"refused"`
	FaultRate  float64 `json:"fault_rate"`

	// faithfulness witness vs naive compaction.
	CompactionLossTurns int `json:"compaction_loss_turns"`
	FactsRecovered      int `json:"facts_recovered"`
}

// Total mirrors the ctxplanbench aggregate row (the `total` key).
type Total struct {
	Sessions            int     `json:"sessions"`
	Turns               int     `json:"turns"`
	LinearCumTok        int64   `json:"linear_cum_tokens"`
	CompactCumTok       int64   `json:"compact_cum_tokens"`
	PlannedCumTok       int64   `json:"planned_cum_tokens"`
	FaultTaxCum         int64   `json:"fault_tax_cum"`
	References          int     `json:"references"`
	Faults              int     `json:"faults"`
	Served              int     `json:"served"`
	Refused             int     `json:"refused"`
	FaultRate           float64 `json:"fault_rate"`
	CompactionLossTurns int     `json:"compaction_loss_turns"`
	FactsRecovered      int     `json:"facts_recovered"`
}

// Report mirrors the ctxplanbench --out top-level object.
type Report struct {
	Budget   int       `json:"budget"`
	Window   int       `json:"window"`
	Sessions []Session `json:"sessions"`
	Total    Total     `json:"total"`
}

// RecoveryBand is the measured-inputs band for r on one row (a session or the
// aggregate). It carries the recovery OPERANDS and the FENCE together, and -- by
// construction -- NO r and NO horizon_multiplier field. Provenance is per the doc:
// every field here is measured over a real transcript by ctxplanbench.
type RecoveryBand struct {
	Scope string `json:"scope"` // "session:<source>" | "aggregate"
	Turns int    `json:"turns"`

	// the recovery OPERANDS -- two fields plus their ratio, never multiplied into r.
	LinearResidentTok  int64   `json:"linear_resident_tokens"`  // what the linear transcript forced resident
	BoundedResidentTok int64   `json:"bounded_resident_tokens"` // what the bounded view actually held
	RecoveryRatio      float64 `json:"recovery_ratio"`          // linear / bounded; the measured r OPERAND (>=1 means budget came back)
	ReclaimedTok       int64   `json:"reclaimed_tokens"`        // linear - bounded; resident budget returned to the pool

	// the FENCE -- co-located with the operands by invariant; never ships apart.
	FaultRate      float64 `json:"fault_rate"`            // faults / references on the real reference signal
	FaultsServed   int     `json:"faults_served"`         // recovered via DemandPage (recoverable miss)
	FaultsRefused  int     `json:"faults_refused"`        // gate held (sealed) -- the floor, not a loss
	FaultTaxTokens int64   `json:"fault_tax_tokens"`      // re-prefill tokens the misses cost (the recovery's price)
	CompactionLoss int     `json:"compaction_loss_turns"` // turns naive compaction destroyed >=1 fact
	FactsRecovered int     `json:"facts_recovered"`       // facts the planned view recovered that compaction lost

	Provenance string `json:"provenance"` // always "measured" -- every field is a real-transcript event
}

// BandsFromReport builds a per-session recovery band for every session in the
// report. The aggregate band is built separately by AggregateBand, which enforces
// the session-count floor.
func BandsFromReport(r Report) []RecoveryBand {
	out := make([]RecoveryBand, 0, len(r.Sessions))
	for _, s := range r.Sessions {
		out = append(out, sessionBand(s))
	}
	return out
}

func sessionBand(s Session) RecoveryBand {
	return RecoveryBand{
		Scope:              "session:" + s.Source,
		Turns:              s.Turns,
		LinearResidentTok:  s.LinearCumTok,
		BoundedResidentTok: s.CompactCumTok,
		RecoveryRatio:      ratio(s.LinearCumTok, s.CompactCumTok),
		ReclaimedTok:       s.LinearCumTok - s.CompactCumTok,
		FaultRate:          s.FaultRate,
		FaultsServed:       s.Served,
		FaultsRefused:      s.Refused,
		FaultTaxTokens:     s.FaultTaxCum,
		CompactionLoss:     s.CompactionLossTurns,
		FactsRecovered:     s.FactsRecovered,
		Provenance:         "measured",
	}
}

// AggregateBand returns the population recovery band, or an error if the report
// holds fewer than MinAggregateSessions real sessions -- a single transcript's gap
// is not the population claim r makes.
func AggregateBand(r Report) (RecoveryBand, error) {
	n := r.Total.Sessions
	if n == 0 {
		n = len(r.Sessions)
	}
	if n < MinAggregateSessions {
		return RecoveryBand{}, fmt.Errorf(
			"aggregate recovery band refused: %d sessions < floor %d "+
				"(one transcript's gap is not a population claim; run ctxplanbench --heaviest %d)",
			n, MinAggregateSessions, MinAggregateSessions)
	}
	t := r.Total
	return RecoveryBand{
		Scope:              "aggregate",
		Turns:              t.Turns,
		LinearResidentTok:  t.LinearCumTok,
		BoundedResidentTok: t.CompactCumTok,
		RecoveryRatio:      ratio(t.LinearCumTok, t.CompactCumTok),
		ReclaimedTok:       t.LinearCumTok - t.CompactCumTok,
		FaultRate:          t.FaultRate,
		FaultsServed:       t.Served,
		FaultsRefused:      t.Refused,
		FaultTaxTokens:     t.FaultTaxCum,
		CompactionLoss:     t.CompactionLossTurns,
		FactsRecovered:     t.FactsRecovered,
		Provenance:         "measured",
	}, nil
}

// ratio is linear/bounded with the bounded-cost guard. >=1 means the bounded view
// reclaimed resident budget; ==1 means it held everything (no recovery, the floor
// case). It is an OPERAND of r, never r itself.
func ratio(linear, bounded int64) float64 {
	if bounded <= 0 {
		return 0
	}
	return float64(linear) / float64(bounded)
}

// Selfcheck enforces the anti-overclaim contract structurally, with no model run.
// It is the package's honesty witness: the four guards the doc and the grounding
// synthesis demand.
func Selfcheck() error {
	rep := syntheticReport()

	// GUARD 1 -- the band NEVER carries r or horizon_multiplier. Enforced by the
	// type itself (RecoveryBand has no such field), re-asserted over the JSON keys
	// so a future field addition that smuggles r in fails here.
	band := sessionBand(rep.Sessions[0])
	keys, err := jsonKeys(band)
	if err != nil {
		return err
	}
	for _, forbidden := range []string{"r", "horizon_multiplier", "horizon", "multiplier"} {
		if keys[forbidden] {
			return fmt.Errorf("band emits forbidden field %q (r must stay structural, never a printed number)", forbidden)
		}
	}

	// GUARD 2 -- the recovery operand and the fault fence CO-OCCUR. A recovery
	// number may never ship without its fault-rate fence in the same row.
	for _, need := range []string{"recovery_ratio", "reclaimed_tokens", "fault_rate", "faults_refused"} {
		if !keys[need] {
			return fmt.Errorf("band missing required co-occurring field %q (recovery and its fence must travel together)", need)
		}
	}

	// GUARD 3 -- every field is labeled measured (every operand is a real-transcript
	// event; nothing here is modeled, unlike savingsvector's prefill/wall-clock axes).
	if band.Provenance != "measured" {
		return fmt.Errorf("band provenance is %q, want measured", band.Provenance)
	}

	// GUARD 4 -- a single session NEVER yields an aggregate population band; the
	// floor refuses it.
	if _, err := AggregateBand(Report{Sessions: rep.Sessions[:1], Total: Total{Sessions: 1}}); err == nil {
		return fmt.Errorf("aggregate band accepted 1 session; the population floor (%d) must refuse it", MinAggregateSessions)
	}

	// GUARD 5 -- the recovery ratio is a faithful re-projection: reclaimed_tokens ==
	// linear - bounded, and ratio == linear/bounded. A drift here would mean the
	// operand was computed, not read.
	s := rep.Sessions[0]
	if band.ReclaimedTok != s.LinearCumTok-s.CompactCumTok {
		return fmt.Errorf("reclaimed_tokens drift: %d != %d-%d", band.ReclaimedTok, s.LinearCumTok, s.CompactCumTok)
	}
	if math.Abs(band.RecoveryRatio-ratio(s.LinearCumTok, s.CompactCumTok)) >= 1e-12 {
		return fmt.Errorf("recovery_ratio drift")
	}

	// GUARD 6 -- a valid aggregate (>= floor) is accepted and carries the fence.
	agg, err := AggregateBand(rep)
	if err != nil {
		return fmt.Errorf("aggregate band refused a valid %d-session report: %v", rep.Total.Sessions, err)
	}
	if agg.Scope != "aggregate" {
		return fmt.Errorf("aggregate band scope is %q", agg.Scope)
	}
	return nil
}

// syntheticReport is a ctxplanbench-shaped report with the floor met, for Selfcheck.
// Values are illustrative (a recovery ratio > 1 with a low fault rate), NOT a
// benchmark claim.
func syntheticReport() Report {
	s := Session{
		Source: "synthetic", Turns: 50, Budget: 8000,
		LinearCumTok: 500000, CompactCumTok: 100000, PlannedCumTok: 100000, FaultTaxCum: 1200,
		References: 40, Faults: 2, Served: 2, Refused: 0, FaultRate: 0.05,
		CompactionLossTurns: 3, FactsRecovered: 2,
	}
	sessions := make([]Session, MinAggregateSessions)
	for i := range sessions {
		sessions[i] = s
	}
	return Report{
		Budget: 8000, Window: 6,
		Sessions: sessions,
		Total: Total{
			Sessions: MinAggregateSessions, Turns: 50 * MinAggregateSessions,
			LinearCumTok: 500000 * MinAggregateSessions, CompactCumTok: 100000 * MinAggregateSessions,
			PlannedCumTok: 100000 * MinAggregateSessions, FaultTaxCum: 1200 * MinAggregateSessions,
			References: 40 * MinAggregateSessions, Faults: 2 * MinAggregateSessions,
			Served: 2 * MinAggregateSessions, Refused: 0, FaultRate: 0.05,
			CompactionLossTurns: 3 * MinAggregateSessions, FactsRecovered: 2 * MinAggregateSessions,
		},
	}
}
