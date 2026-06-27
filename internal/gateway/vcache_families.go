package gateway

// vcache_families.go -- the PER-FAMILY live-observe view (#935, #788 follow-on).
//
// The cumulative fak_vcache_* family (writeVCacheMetrics) + the /debug/vars `vcache`
// block expose ONE aggregate row: the session's NET realized provider-cache economics.
// `fak vcache observe` gives more offline — the PER-FAMILY (per session/trace prefix)
// breakdown, the measured Zipf concentration + flat/defeated gate, the per-family
// governor verdict, the warmth-belief false-warm/false-cold, and the measured-vs-
// synthetic grade. This wires that same view onto LIVE traffic.
//
// How it reconciles with the offline verb: the live gateway retains the per-turn,
// family-tagged telemetry (observeVCacheTurn) and renders the view by feeding it to the
// SAME vcacheobserve.Observe engine the CLI uses. Same engine + same turns => the live
// per-family view is byte-identical to `fak vcache observe` on the same traffic (the
// acceptance), with no parallel math to drift.
//
// Provenance + Law A2: every value carries an OBSERVED (provider-relayed: hit rate,
// arrival rate, the token-equiv economics derived from the provider's own counters) or
// DECISION (fak's deterministic verdict over those counters: the governor decision, the
// warmth-belief prediction, the concentration gate, the proof status, the grade) label,
// surfaced in the block's `provenance` map. The accumulator is purely observational —
// nothing in the request path reads it — so correctness never depends on a cache hit.
//
// Cardinality note: the family key is a session/trace id (high cardinality), so this
// view is a periodic /debug/vars SNAPSHOT, not a per-family Prometheus label set (which
// would explode series cardinality). The issue allows either; the snapshot is the
// correct surface for per-session data.

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

// vcacheTurnCap bounds the per-family live-observe window so a 24/7 gateway stays flat
// in memory. At ~48 bytes per retained Turn this is well under a megabyte; the live
// view is over this rolling window (the block reports turns_observed + window_capped so
// an operator knows when drop-oldest has trimmed it).
const vcacheTurnCap = 4096

// observeVCacheTurn records one served turn's provider-cache telemetry under its prefix
// family (the session/trace id) for the live per-family observe view. It clamps negative
// counts to 0 (a planner that omits a count never wraps), normalizes an empty family to
// "unknown" so the bucket reads clearly, and drops the oldest turn past vcacheTurnCap so
// the window stays bounded. It is a no-op on a nil metrics object so every caller is
// safe to invoke unconditionally.
func (m *gatewayMetrics) observeVCacheTurn(family string, unixMillis int64, input, read, create int) {
	if m == nil {
		return
	}
	family = strings.TrimSpace(family)
	if family == "" {
		family = "unknown"
	}
	turn := vcacheobserve.Turn{
		Family:        family,
		UnixMillis:    unixMillis,
		InputTokens:   int64(clampNonNeg(input)),
		CacheRead:     int64(clampNonNeg(read)),
		CacheCreation: int64(clampNonNeg(create)),
	}
	m.vcacheMu.Lock()
	m.vcacheTurns = append(m.vcacheTurns, turn)
	if len(m.vcacheTurns) > vcacheTurnCap {
		// Drop-oldest: keep the most recent vcacheTurnCap turns. Re-slice onto a fresh
		// backing array so the dropped head can be GC'd instead of pinned by the cap.
		trimmed := make([]vcacheobserve.Turn, vcacheTurnCap)
		copy(trimmed, m.vcacheTurns[len(m.vcacheTurns)-vcacheTurnCap:])
		m.vcacheTurns = trimmed
		m.vcacheTurnsDropped = true
	}
	m.vcacheMu.Unlock()
}

// vcacheTurnsSnapshot returns a copy of the retained per-family window and whether
// drop-oldest has trimmed it, so the render runs over a stable slice without holding the
// lock across the Observe pass.
func (m *gatewayMetrics) vcacheTurnsSnapshot() ([]vcacheobserve.Turn, bool) {
	if m == nil {
		return nil, false
	}
	m.vcacheMu.Lock()
	defer m.vcacheMu.Unlock()
	if len(m.vcacheTurns) == 0 {
		return nil, m.vcacheTurnsDropped
	}
	out := make([]vcacheobserve.Turn, len(m.vcacheTurns))
	copy(out, m.vcacheTurns)
	return out, m.vcacheTurnsDropped
}

func clampNonNeg(n int) int {
	if n < 0 {
		return 0
	}
	return n
}

// debugVCacheFamiliesVars is the /debug/vars `vcache_families` block: the per-family
// observe view over the live rolling window, reconciling with `fak vcache observe` on
// the same traffic. It is nil (the block is omitted) until a turn carries provider cache
// activity — the same no-phantom guard the cumulative `vcache` block keeps.
type debugVCacheFamiliesVars struct {
	TurnsObserved   int                      `json:"turns_observed"`    // OBSERVED: turns in the rolling window
	FamilyCount     int                      `json:"family_count"`      // distinct prefix families seen
	WindowCapped    bool                     `json:"window_capped"`     // true once drop-oldest trimmed the window
	SavedTokenEquiv float64                  `json:"saved_token_equiv"` // NET OBSERVED-derived, over the window
	Status          string                   `json:"status"`            // DECISION: PROVEN/REFUTED over the window
	Concentration   debugVCacheConcentration `json:"concentration"`     // DECISION: measured Zipf + flat/defeated gate
	GradeMeasured   string                   `json:"grade_measured"`    // DECISION: scored with the account's measured skew
	GradeSynthetic  string                   `json:"grade_synthetic"`   // DECISION: scored with the scorecard's synthetic defaults
	Families        []debugVCacheFamilyVars  `json:"families"`
	Provenance      map[string]string        `json:"provenance"` // value-class -> OBSERVED|DECISION
}

// debugVCacheFamilyVars is one prefix family's live observe slice.
type debugVCacheFamilyVars struct {
	Key               string  `json:"key"`
	Turns             int     `json:"turns"`
	HitRate           float64 `json:"hit_rate"`             // OBSERVED
	SavedTokenEquiv   float64 `json:"saved_token_equiv"`    // NET OBSERVED-derived
	Status            string  `json:"status"`               // DECISION: PROVEN/REFUTED
	GovernorDecision  string  `json:"governor_decision"`    // DECISION: the M5 steady-state verdict
	ArrivalRatePerSec float64 `json:"arrival_rate_per_sec"` // OBSERVED
	WarmthTrueWarm    int     `json:"warmth_true_warm"`     // DECISION: belief vs observed read
	WarmthFalseWarm   int     `json:"warmth_false_warm"`    // DECISION: the lethal believed-warm-but-cold miss
	WarmthTrueCold    int     `json:"warmth_true_cold"`     // DECISION
	WarmthFalseCold   int     `json:"warmth_false_cold"`    // DECISION: a warming chance the belief missed
}

// debugVCacheConcentration is the measured workload-concentration gate (#716): the
// fitted Zipf exponent s and the structurally-defeated flag (s <= 1 -> vCache cannot
// help). It is a DECISION over the OBSERVED family distribution.
type debugVCacheConcentration struct {
	ZipfS          float64 `json:"zipf_s"`
	Measured       bool    `json:"measured"`
	Defeated       bool    `json:"defeated"`
	Recommendation string  `json:"recommendation,omitempty"`
}

// vcacheFamiliesVars renders the per-family live observe block from the retained window.
// It returns nil (no phantom) until a turn carried provider cache activity, mirroring the
// cumulative `vcache` block's zero guard. The whole view is produced by the SAME
// vcacheobserve.Observe engine `fak vcache observe` uses, so it reconciles with the
// offline verb on the same turns by construction.
func vcacheFamiliesVars(turns []vcacheobserve.Turn, windowCapped bool) *debugVCacheFamiliesVars {
	if len(turns) == 0 {
		return nil
	}
	active := false
	for _, t := range turns {
		if t.CacheRead > 0 || t.CacheCreation > 0 {
			active = true
			break
		}
	}
	if !active {
		return nil // a no-cache workload emits no per-family block (no phantom)
	}

	rep := vcacheobserve.Observe(turns, vcacheobserve.DefaultMultipliers())
	out := &debugVCacheFamiliesVars{
		TurnsObserved:   rep.Turns,
		FamilyCount:     rep.FamilyCount,
		WindowCapped:    windowCapped,
		SavedTokenEquiv: rep.Aggregate.SavedTokenEquiv,
		Status:          string(rep.Aggregate.Status),
		Concentration: debugVCacheConcentration{
			ZipfS:          rep.Concentration.ZipfS,
			Measured:       rep.Concentration.Measured,
			Defeated:       rep.Concentration.Defeated,
			Recommendation: rep.Concentration.Recommendation,
		},
		GradeMeasured:  rep.GradeMeasured,
		GradeSynthetic: rep.GradeSynthetic,
		Provenance: map[string]string{
			"hit_rate":             string(vcacheobserve.Observed),
			"arrival_rate_per_sec": string(vcacheobserve.Observed),
			"saved_token_equiv":    string(vcacheobserve.Observed),
			"governor_decision":    string(vcacheobserve.Decision),
			"warmth":               string(vcacheobserve.Decision),
			"concentration":        string(vcacheobserve.Decision),
			"grade":                string(vcacheobserve.Decision),
			"status":               string(vcacheobserve.Decision),
		},
	}
	for _, fam := range rep.Families {
		out.Families = append(out.Families, debugVCacheFamilyVars{
			Key:               fam.Key,
			Turns:             fam.Turns,
			HitRate:           fam.HitRate,
			SavedTokenEquiv:   fam.Economics.SavedTokenEquiv,
			Status:            string(fam.Economics.Status),
			GovernorDecision:  string(fam.GovernorDecision),
			ArrivalRatePerSec: fam.ArrivalRatePerSec,
			WarmthTrueWarm:    fam.Prediction.TrueWarm,
			WarmthFalseWarm:   fam.Prediction.FalseWarm,
			WarmthTrueCold:    fam.Prediction.TrueCold,
			WarmthFalseCold:   fam.Prediction.FalseCold,
		})
	}
	return out
}
