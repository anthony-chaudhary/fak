package scorecard

import (
	"fmt"
	"sort"
)

// --- Fleet cache-health fold (issue #3643, epic #3569 cache-verify) ---
//
// Cache health today is spread across many surfaces -- managed-cache posture adoption,
// realized reuse ratio, shed effectiveness, WITNESSED/OBSERVED divergence, and the
// upgrade-fired rate -- with no single 0..1 number a human can read at a glance or a gate
// can ratchet. This fold is that number: it folds the cache FAMILIES into one 0..1
// cache-health headline plus a worst-first component worklist, and `fak score cache-health`
// renders it ALONGSIDE the D1/D2/D3 cache-VALUE aspects (it does not replace them).
//
// Like D1/D2/D3 this is PURE over caller-supplied facts and imports nothing but fmt/sort, so
// it stays deterministic, unit-testable with fixtures, and reusable by the leased cmd/report
// shell without this scoring core reaching up into cachevaluereport (which would red architest
// with an ARCH_LAYER_VIOLATION). It RE-USES the existing component metrics rather than
// re-deriving them: the caller maps each family's already-derived signal into a 0..1 health
// (1.0 == healthy) and hands it in; the two count-based conversions and the divergence->agreement
// conversion live here as exported helpers (ShedEffectivenessHealth / UpgradeFiredHealth /
// WitnessedObservedAgreement) so the family semantics are unit-tested at this tier and the CLI
// wiring stays a thin mapping.
//
// Each family is INDIVIDUALLY RETIRABLE: its health is a standalone 0..1 fraction, its KPI a
// standalone builder, and its defect is retired by raising the real component (turn managed
// cache on, improve reuse, fix the shed gate, converge witnessed->observed, let the upgrade
// lever fire) -- never by weakening the pass line.

// CacheHealthSchema tags the fleet cache-health SCORE card (ComposeCacheHealth), distinct from
// the D1 score / D2 gate / D3 accuracy schemas so a roster/consumer reads all four cards apart.
const CacheHealthSchema = "fak-cache-health-scorecard/1"

// CacheHealthDebtKey is the corpus debt integer this card writes (the count of families below
// the pass line). Exported so the CLI names it in the shared --json/--markdown/--compare tail.
const CacheHealthDebtKey = "cache_health_debt"

// CacheHealthPassLine is the 0..1 health floor below which a cache family books a defect (and so
// reds ok / joins the ratchet debt). It is deliberately a single, conservative starting floor an
// operator TIGHTENS over time -- the "gate can ratchet" knob the issue names -- NOT a per-family
// tuned threshold. The worst-first worklist orders every scored family regardless of this floor;
// the floor only decides which families count as debt.
const CacheHealthPassLine = 0.5

// The canonical, ordered cache-family component keys -- the same exported-canonical-list
// treatment CacheValueSubAspectKeys / CacheValueAccuracyClasses get. The order is the
// deterministic tie-break when two families carry equal health AND the enumeration a consumer
// iterates to address each family standalone.
const (
	CacheHealthPosture      = "managed_cache_posture"        // fraction of exit sessions with managed-cache posture active
	CacheHealthReuse        = "reuse_ratio"                  // realized multi-turn cache reuse fraction (WITNESSED)
	CacheHealthShed         = "shed_effectiveness"           // compaction shed fired / (fired + bailed)
	CacheHealthAgreement    = "witnessed_observed_agreement" // 1 - |gross - net| WITNESSED/OBSERVED divergence
	CacheHealthUpgradeFired = "upgrade_fired_rate"           // TTL upgrades fired / (fired + refused)
)

// CacheHealthComponents is the canonical ordered family set. len == the fixed denominator
// components_total; a family is folded only when the caller supplies evidence for it.
var CacheHealthComponents = []string{
	CacheHealthPosture,
	CacheHealthReuse,
	CacheHealthShed,
	CacheHealthAgreement,
	CacheHealthUpgradeFired,
}

// cacheHealthLabels give each family a human phrase for the worklist detail + the retire hint.
var cacheHealthLabels = map[string]string{
	CacheHealthPosture:      "managed-cache posture adoption",
	CacheHealthReuse:        "realized multi-turn reuse",
	CacheHealthShed:         "compaction shed effectiveness",
	CacheHealthAgreement:    "witnessed/observed agreement",
	CacheHealthUpgradeFired: "TTL upgrade-fired rate",
}

// CacheHealthFacts carries each cache family's already-derived 0..1 health (1.0 == healthy).
// Every field is a POINTER so "no evidence this window" (nil) is never conflated with a measured
// 0.0 -- a nil family is EXCLUDED from the fold, exactly like the nil-able weekly-digest ratios
// it is mapped from. The caller (the leased `fak score cache-health` shell) fills these from the
// EXISTING component metrics; this scoring core reads no ledger and imports no shell.
type CacheHealthFacts struct {
	ManagedCachePosture        *float64 // posture-active adoption fraction (see PostureHealth)
	ReuseRatio                 *float64 // realized multi-turn reuse fraction (already 0..1)
	ShedEffectiveness          *float64 // shed fired / (fired + bailed) (see ShedEffectivenessHealth)
	WitnessedObservedAgreement *float64 // 1 - gross/net divergence (see WitnessedObservedAgreement)
	UpgradeFiredRate           *float64 // upgrades / (upgrades + refusals) (see UpgradeFiredHealth)
}

// componentHealth returns the caller-supplied health for one family key, or nil when absent.
func (f CacheHealthFacts) componentHealth(component string) *float64 {
	switch component {
	case CacheHealthPosture:
		return f.ManagedCachePosture
	case CacheHealthReuse:
		return f.ReuseRatio
	case CacheHealthShed:
		return f.ShedEffectiveness
	case CacheHealthAgreement:
		return f.WitnessedObservedAgreement
	case CacheHealthUpgradeFired:
		return f.UpgradeFiredRate
	}
	return nil
}

// PostureHealth maps a 0..100 posture-adoption percentage (the weekly digest's PostureAdoptionPct,
// nil when no exit sessions) into a 0..1 family health. nil in -> nil out (no evidence).
func PostureHealth(adoptionPct *float64) *float64 {
	if adoptionPct == nil {
		return nil
	}
	h := clamp01(*adoptionPct / 100)
	return &h
}

// ShedEffectivenessHealth maps the WITNESSED compaction counters into a 0..1 family health: the
// fraction of shed decisions where the shed actually FIRED (removed resident tokens) rather than
// bailed. No shed decisions (fired+bailed == 0) -> nil (no evidence this window).
func ShedEffectivenessHealth(fired, bailed uint64) *float64 {
	total := fired + bailed
	if total == 0 {
		return nil
	}
	h := clamp01(float64(fired) / float64(total))
	return &h
}

// UpgradeFiredHealth maps the WITNESSED TTL-upgrade outcomes into a 0..1 family health: the
// fraction of upgrade heads where the lever FIRED (upgraded) rather than refused. No heads
// (upgrades+refusals == 0) -> nil (no evidence: the lever saw nothing). NOTE the framing --
// upgrade-FIRED is read as healthy; a fleet may deliberately refuse upgrades that do not pay
// back, so an operator tightens/loosens the pass line rather than treating every refusal as a bug.
func UpgradeFiredHealth(upgrades, refusals uint64) *float64 {
	total := upgrades + refusals
	if total == 0 {
		return nil
	}
	h := clamp01(float64(upgrades) / float64(total))
	return &h
}

// WitnessedObservedAgreement maps the WITNESSED/OBSERVED (gross/net) shares into a 0..1 family
// health: 1 - their divergence, so gross == net reads a healthy 1.0 and a maximal divergence
// reads 0.0. It REUSES the existing GrossNetDivergence sub-aspect rather than re-deriving it.
func WitnessedObservedAgreement(gross, net float64) float64 {
	return clamp01(1 - GrossNetDivergence(gross, net))
}

// CacheHealthRow is one cache family's row in the worst-first worklist: its key, its 0..1 health,
// the pass line, whether it is in debt (below the pass line), and a human detail. The worklist is
// EVERY scored family, sorted worst-first (lowest health first, CacheHealthComponents order
// breaking ties), so a human reads the weakest family at a glance and a gate ratchets on the debt.
type CacheHealthRow struct {
	Component string  `json:"component"`
	Health    float64 `json:"health"`
	PassLine  float64 `json:"pass_line"`
	InDebt    bool    `json:"in_debt"`
	Detail    string  `json:"detail"`
}

// cacheHealthDetail renders one family's worklist/KPI detail line.
func cacheHealthDetail(component string, health float64) string {
	status := "clears pass line"
	if health+gateEps < CacheHealthPassLine {
		status = "BELOW pass line"
	}
	return fmt.Sprintf("%s health %.3f (pass line %.2f, %s)", cacheHealthLabels[component], health, CacheHealthPassLine, status)
}

// CacheHealth is the pure core: it folds the present cache-family healths into the single 0..1
// fleet cache-health number (the mean of the families that HAVE evidence) and the worst-first
// worklist (every scored family, lowest health first, canonical order breaking ties). present is
// the count of families folded; when it is 0 the number is 1.0 (nothing is known-unhealthy) and
// the worklist empty. A degraded family lowers the number AND floats to the top of the worklist.
func CacheHealth(f CacheHealthFacts) (number float64, present int, worklist []CacheHealthRow) {
	rank := map[string]int{}
	for i, c := range CacheHealthComponents {
		rank[c] = i
	}
	var sum float64
	worklist = make([]CacheHealthRow, 0, len(CacheHealthComponents))
	for _, c := range CacheHealthComponents {
		h := f.componentHealth(c)
		if h == nil {
			continue
		}
		v := clamp01(*h)
		sum += v
		present++
		worklist = append(worklist, CacheHealthRow{
			Component: c,
			Health:    Round3(v),
			PassLine:  CacheHealthPassLine,
			InDebt:    v+gateEps < CacheHealthPassLine,
			Detail:    cacheHealthDetail(c, v),
		})
	}
	sort.SliceStable(worklist, func(i, j int) bool {
		if worklist[i].Health != worklist[j].Health {
			return worklist[i].Health < worklist[j].Health
		}
		return rank[worklist[i].Component] < rank[worklist[j].Component]
	})
	if present == 0 {
		return 1, 0, worklist
	}
	return sum / float64(present), present, worklist
}

// cacheHealthKPI builds one family KPI. Score is 100*health so the Fold composite (mean of scores)
// is exactly 100*number; PassLine feeds the unbounded pressure layer (deficit below the health
// bar). A family below the pass line owns exactly one Defect, so Fold's debt is the count of
// below-pass-line families and ok flips iff any family is in debt.
func cacheHealthKPI(component string, health float64) KPI {
	v := clamp01(health)
	k := KPI{
		Key:      component,
		Group:    "cache_health",
		Score:    100 * v,
		PassLine: 100 * CacheHealthPassLine,
		Detail:   cacheHealthDetail(component, v),
	}
	if v+gateEps < CacheHealthPassLine {
		k.Defects = []string{fmt.Sprintf("%s: %s health %.3f < %.2f pass line -- raise the real component (reuse the existing metric; never weaken the floor)", component, cacheHealthLabels[component], v, CacheHealthPassLine)}
	}
	return k
}

// cacheHealthKPIs builds one KPI per family that has evidence, in canonical order. With no
// evidence it returns a single healthy INSUFFICIENT KPI so the fold's value stays a coherent 1.0
// (nothing is known-unhealthy) instead of collapsing an empty slice to a spurious 0/F.
func cacheHealthKPIs(f CacheHealthFacts) []KPI {
	kpis := make([]KPI, 0, len(CacheHealthComponents))
	for _, c := range CacheHealthComponents {
		if h := f.componentHealth(c); h != nil {
			kpis = append(kpis, cacheHealthKPI(c, *h))
		}
	}
	if len(kpis) == 0 {
		return []KPI{{
			Key:      "cache_health_evidence",
			Group:    "cache_health",
			Score:    100,
			PassLine: 100 * CacheHealthPassLine,
			Detail:   "no cache-family evidence this window -- nothing to fold (INSUFFICIENT); cache_health defaults to 1.0",
		}}
	}
	return kpis
}

// ComposeCacheHealth folds the cache families into the single fleet cache-health control-pane
// payload (symmetric with ComposeD1/D2/D3). corpus["cache_health"] is the 0..1 headline (identical
// to corpus.value by construction); corpus["cache_health_worklist"] is the worst-first family
// order; corpus[CacheHealthDebtKey] is the count of families below the pass line; ok == (debt == 0).
// The standard grade curve (GradeStd) is used because this is an OPERATIONAL health card, not a
// provenance-honesty card (which is why it uses GradeStd where D1/D2/D3 use GradeStrict).
func ComposeCacheHealth(f CacheHealthFacts) Payload {
	number, present, worklist := CacheHealth(f)
	if worklist == nil {
		worklist = []CacheHealthRow{}
	}
	return Fold(CacheHealthSchema, cacheHealthKPIs(f), CacheHealthDebtKey, nil, Messages{
		Finding:         "fleet cache-health carries debt: a cache family fell below the health pass line -- work the worst-first worklist",
		FindingClean:    "fleet cache-health clean: every cache family with evidence clears the health pass line",
		NextAction:      "raise the worst-first family (reuse the metric behind it): managed-cache posture / realized reuse / shed effectiveness / witnessed-observed agreement / upgrade-fired rate",
		NextActionClean: "hold the line; keep every cache family above the health pass line and tighten the ratchet",
		Grade:           GradeStd,
		ExtraCorpus: map[string]any{
			"cache_health":          Round3(number),
			"cache_health_worklist": worklist,
			"components_present":    present,
			"components_total":      len(CacheHealthComponents),
			"pass_line":             CacheHealthPassLine,
		},
	})
}
