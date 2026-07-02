package vcachescore

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/vcachecal"
	"github.com/anthony-chaudhary/fak/internal/vcachechain"
	"github.com/anthony-chaudhary/fak/internal/vcachegov"
)

const (
	DefaultTwoXThreshold     = 2.0
	DefaultTargetCoverage    = 0.85
	DefaultMaxFalseWarmRate  = 0.05
	AnchorSourceSynthetic    = "synthetic"
	AnchorSourceMeasured     = "measured"
	defaultConcentrationZipf = 1.74
	defaultAnchorUniverse    = 1000
)

// Input is the complete deterministic input to the vCache scorecard.
type Input struct {
	Star vcachegov.StarSavingsInput

	AgenticActivation AgenticActivationInput
	KernelKV          PlaneEvidenceInput
	Context           PlaneEvidenceInput
	ExternalEngine    PlaneEvidenceInput
	TelemetryRows     []vcachegov.TelemetryRow
	TelemetryReadMult float64
	TelemetryWrite5m  float64
	TelemetryWrite1h  float64
	Ranked            []vcachecal.RankedVBlock
	AnchorSource      string
	TurnsObserved     int
	TargetCoverage    float64
	Prediction        vcachecal.PredictionError
	Recall            vcachechain.ProveRecallInput
	TwoXThreshold     float64
	MaxFalseWarmRate  float64
}

// PlaneEvidenceInput is optional cache-plane evidence supplied by an embedding
// surface such as the gateway. It keeps provider telemetry, fak-owned KV reuse,
// O(1) context value, and external serving-engine hit-rate evidence separate.
type PlaneEvidenceInput struct {
	Available          bool    `json:"available"`
	Provenance         string  `json:"provenance,omitempty"`
	Multiplier         float64 `json:"multiplier,omitempty"`
	BaselineTokenEquiv float64 `json:"baseline_token_equiv,omitempty"`
	SavedTokenEquiv    float64 `json:"saved_token_equiv,omitempty"`
	CostTokenEquiv     float64 `json:"cost_token_equiv,omitempty"`
	HitRate            float64 `json:"hit_rate,omitempty"`
	Reason             string  `json:"reason,omitempty"`
}

// AgenticActivationInput counts fak-authored cache activation events. Provider
// cache counters are observed economics, not fak-authored activation.
type AgenticActivationInput struct {
	KernelKVEvents          int `json:"kernel_kv_events,omitempty"`
	ContextEvents           int `json:"context_events,omitempty"`
	ProviderVCacheDecisions int `json:"provider_vcache_decisions,omitempty"`
	ExternalEngineEvents    int `json:"external_engine_events,omitempty"`
}

// DefaultInput is the committed Codex-like star workload plus a skewed Zipf
// workload. It is deliberately model/provider-free and reproduces the default
// vCache proof posture.
func DefaultInput() Input {
	return Input{
		Star: vcachegov.StarSavingsInput{
			AnchorTokens:    4096,
			SuffixTokens:    10,
			Requests:        7,
			MinPrefixTokens: 1024,
			ReadMult:        0.1,
			WriteMult:       vcachegov.WriteMult5Minutes,
			Secret:          vcachegov.Cacheable,
		},
		TelemetryReadMult: 0.1,
		TelemetryWrite5m:  vcachegov.WriteMult5Minutes,
		TelemetryWrite1h:  vcachegov.WriteMult1Hour,
		Ranked:            SyntheticZipfWorkload(defaultConcentrationZipf, defaultAnchorUniverse),
		TargetCoverage:    DefaultTargetCoverage,
		Recall: vcachechain.ProveRecallInput{
			PrefixTokens: 30000,
			UnitTokens:   10,
			ReadMult:     0.1,
			Siblings:     1,
		},
		TwoXThreshold:    DefaultTwoXThreshold,
		MaxFalseWarmRate: DefaultMaxFalseWarmRate,
	}
}

// SyntheticZipfWorkload builds a deterministic ranked workload for benchmark and
// fixture use. The weight is rank^-s; callers that have real anchor counts should
// pass real RankedVBlock rows instead.
func SyntheticZipfWorkload(s float64, anchors int) []vcachecal.RankedVBlock {
	if anchors < 0 {
		anchors = 0
	}
	out := make([]vcachecal.RankedVBlock, anchors)
	for i := 0; i < anchors; i++ {
		w := math.Pow(float64(i+1), -s)
		out[i] = vcachecal.RankedVBlock{
			Key:          fmt.Sprintf("anchor-%04d", i+1),
			Frequency:    w,
			Size:         1,
			ReuseDensity: 1,
		}
	}
	return out
}

// IndexPlan is the hot-anchor index the agent should build to capture the target
// coverage.
type IndexPlan struct {
	TargetCoverage float64 `json:"target_coverage"`
	TotalAnchors   int     `json:"total_anchors"`
	AnchorCount    int     `json:"anchor_count"`
	Coverage       float64 `json:"coverage"`
	Defeated       bool    `json:"defeated"`
	Recommendation string  `json:"recommendation"`
}

// AnchorIndexEntry is one ranked anchor selected for the hot-anchor index
// artifact. It records only stable metadata, never prompt payload.
type AnchorIndexEntry struct {
	Rank     int     `json:"rank"`
	Key      string  `json:"key"`
	Weight   float64 `json:"weight"`
	Coverage float64 `json:"coverage"`
}

// AnchorIndexArtifact is the provider-neutral index plan an agent can persist
// from fak vcache score --index-out. It is a ranked list of reusable prompt
// anchors sufficient to hit the configured coverage target.
type AnchorIndexArtifact struct {
	Schema         string             `json:"schema"`
	TargetCoverage float64            `json:"target_coverage"`
	TotalAnchors   int                `json:"total_anchors"`
	AnchorCount    int                `json:"anchor_count"`
	Coverage       float64            `json:"coverage"`
	Defeated       bool               `json:"defeated"`
	Recommendation string             `json:"recommendation"`
	TotalWeight    float64            `json:"total_weight"`
	Entries        []AnchorIndexEntry `json:"entries"`
}

// EconomicsReport surfaces the four prompt-cache economics the agent-dev gate
// reports from observed provider telemetry -- hit, read, rebate, cost -- alongside
// the realized 2x multiplier. Every value is OBSERVED: it is relayed straight from
// the provider's own cache counters, not a fak-caused effect. Budgeting still
// happens at the uncached price; a cache hit is a realized rebate, never a trust
// claim. It is emitted only when telemetry is supplied, so a value here always has
// a provider witness behind it.
type EconomicsReport struct {
	Source              string  `json:"source"`                // "telemetry"
	Witness             string  `json:"witness"`               // "observed"
	HitRate             float64 `json:"hit_rate"`              // hit: cache_read / baseline
	CacheReadTokens     float64 `json:"cache_read_tokens"`     // read: cached input tokens served
	CacheCreationTokens float64 `json:"cache_creation_tokens"` // read companion: cache writes
	RebateTokenEquiv    float64 `json:"rebate_token_equiv"`    // rebate: token-equivalents saved
	RebatePct           float64 `json:"rebate_pct"`
	CostTokenEquiv      float64 `json:"cost_token_equiv"`     // cost: token-equivalents actually paid
	BaselineTokenEquiv  float64 `json:"baseline_token_equiv"` // cost companion: uncached baseline
	Multiplier          float64 `json:"multiplier"`           // 2x readiness: baseline / cost
}

// PredictionReport surfaces prediction risk as rates and raw counts.
type PredictionReport struct {
	Total         int     `json:"total"`
	TrueWarm      int     `json:"true_warm"`
	FalseWarm     int     `json:"false_warm"`
	TrueCold      int     `json:"true_cold"`
	FalseCold     int     `json:"false_cold"`
	FalseWarmRate float64 `json:"false_warm_rate"`
	FalseColdRate float64 `json:"false_cold_rate"`
}

// PlaneValueReport is one non-overlapping cache plane in the default-usefulness
// view. It keeps provider rebates, fak-owned kernel reuse, O(1) context value,
// external-engine cache observations, and forecasts from collapsing into one
// ambiguous "cache hit" number.
type PlaneValueReport struct {
	Available          bool    `json:"available"`
	Provenance         string  `json:"provenance"` // OBSERVED | WITNESSED | FORECAST | MISSING
	Multiplier         float64 `json:"multiplier,omitempty"`
	HitRate            float64 `json:"hit_rate,omitempty"`
	SavedTokenEquiv    float64 `json:"saved_token_equiv,omitempty"`
	BaselineTokenEquiv float64 `json:"baseline_token_equiv,omitempty"`
	CostTokenEquiv     float64 `json:"cost_token_equiv,omitempty"`
	Reason             string  `json:"reason,omitempty"`
}

// PlaneReport is the additive vCache score contract requested by the
// cache-frontier default-enablement plan. Each plane keeps its own provenance so
// provider rebates, pure-fak KV reuse, O(1) context value, and external-engine
// cache observations do not collapse into one ambiguous "cache hit" number.
type PlaneReport struct {
	ProviderObserved       PlaneValueReport `json:"provider_observed"`
	KernelWitnessed        PlaneValueReport `json:"kernel_witnessed"`
	ContextWitnessed       PlaneValueReport `json:"context_witnessed"`
	ExternalEngineObserved PlaneValueReport `json:"external_engine_observed"`
	Forecast               PlaneValueReport `json:"forecast"`
}

// AgenticActivationReport is the scorecard's visible answer to "did fak's own
// cache machinery fire, or did we only observe an upstream provider rebate?"
type AgenticActivationReport struct {
	KernelKVEvents          int  `json:"kernel_kv_events"`
	ContextEvents           int  `json:"context_events"`
	ProviderVCacheDecisions int  `json:"provider_vcache_decisions"`
	ExternalEngineEvents    int  `json:"external_engine_events"`
	Total                   int  `json:"total"`
	Active                  bool `json:"active"`
}

// DefaultUsefulnessReport is the conservative "useful by default" fold. It is
// not the legacy 2x gate: a provider-only run can clear the 2x economics gate
// while still scoring low here because no fak-authored cache mechanism fired.
type DefaultUsefulnessReport struct {
	Schema          string                  `json:"schema"`
	Score           int                     `json:"score"`
	Grade           string                  `json:"grade"`
	Verdict         string                  `json:"verdict"`
	ColdPathCorrect bool                    `json:"cold_path_correct"`
	Facets          DefaultUsefulnessFacets `json:"facets"`
	Reason          string                  `json:"reason"`
}

// DefaultUsefulnessFacets are the weighted ingredients of the
// default-usefulness score. They intentionally preserve low sub-scores instead
// of smoothing them away.
type DefaultUsefulnessFacets struct {
	NetRealizedValue    int `json:"net_realized_value"`
	AgenticActivation   int `json:"agentic_activation"`
	ColdPathCorrectness int `json:"cold_path_correctness"`
	Granularity         int `json:"granularity"`
	DefaultCoverage     int `json:"default_coverage"`
	DriftResistance     int `json:"drift_resistance"`
	OperatorActionable  int `json:"operator_actionable"`
}

// Report is the scorecard emitted by the CLI and tests.
type Report struct {
	Schema            string                           `json:"schema"`
	Status            string                           `json:"status"`
	Grade             string                           `json:"grade"`
	Score             int                              `json:"score"`
	ActiveSource      string                           `json:"active_source"`
	AnchorSource      string                           `json:"anchor_source"`
	TurnsObserved     int                              `json:"turns_observed"`
	ActiveMultiplier  float64                          `json:"active_multiplier"`
	TwoXThreshold     float64                          `json:"two_x_threshold"`
	TwoXBetter        bool                             `json:"two_x_better"`
	Planned           vcachegov.StarSavingsProof       `json:"planned"`
	Observed          *vcachegov.TelemetrySavingsProof `json:"observed,omitempty"`
	Economics         *EconomicsReport                 `json:"economics,omitempty"`
	Planes            PlaneReport                      `json:"planes"`
	AgenticActivation AgenticActivationReport          `json:"agentic_activation"`
	DefaultUsefulness DefaultUsefulnessReport          `json:"default_usefulness"`
	ColdPathCorrect   bool                             `json:"cold_path_correct"`
	Concentration     vcachecal.Concentration          `json:"concentration"`
	Index             IndexPlan                        `json:"index"`
	Prediction        PredictionReport                 `json:"prediction"`
	Recall            vcachechain.RecallProof          `json:"recall"`
	Actions           []string                         `json:"actions"`
	Risks             []string                         `json:"risks"`
}

// Score folds the existing proof leaves into one deterministic vCache scorecard.
func Score(in Input) Report {
	in = normalize(in)
	planned := vcachegov.ProveStarSavings(in.Star)
	concentration := vcachecal.FitConcentration(in.Ranked)
	index := PlanIndex(concentration, in.TargetCoverage)
	recall := vcachechain.ProveRecall(in.Recall)
	prediction := PredictionReport{
		Total:         in.Prediction.Total,
		TrueWarm:      in.Prediction.TrueWarm,
		FalseWarm:     in.Prediction.FalseWarm,
		TrueCold:      in.Prediction.TrueCold,
		FalseCold:     in.Prediction.FalseCold,
		FalseWarmRate: in.Prediction.FalseWarmRate(),
		FalseColdRate: in.Prediction.FalseColdRate(),
	}

	activeSource := "planned"
	activeProven := planned.Status == vcachegov.ProofProven
	activeMultiplier := ratio(planned.BaselineTokenEquiv, planned.VCacheTokenEquiv)
	var observed *vcachegov.TelemetrySavingsProof
	if len(in.TelemetryRows) > 0 {
		p := vcachegov.ProveTelemetrySavings(vcachegov.TelemetrySavingsInput{
			Rows:        in.TelemetryRows,
			ReadMult:    in.TelemetryReadMult,
			Write5mMult: in.TelemetryWrite5m,
			Write1hMult: in.TelemetryWrite1h,
		})
		observed = &p
		activeSource = "telemetry"
		activeProven = p.Status == vcachegov.ProofProven
		activeMultiplier = ratio(p.BaselineTokenEquiv, p.ActualTokenEquiv)
	}
	twoX := activeProven &&
		activeMultiplier >= in.TwoXThreshold &&
		!concentration.Defeated &&
		prediction.FalseWarmRate <= in.MaxFalseWarmRate

	score := overallScore(activeMultiplier, in.TwoXThreshold, concentration, index, prediction, in.MaxFalseWarmRate)
	rep := Report{
		Schema:           "fak.vcache.score.v1",
		Status:           status(twoX, activeProven, activeMultiplier),
		Grade:            grade(score),
		Score:            score,
		ActiveSource:     activeSource,
		AnchorSource:     in.AnchorSource,
		TurnsObserved:    in.TurnsObserved,
		ActiveMultiplier: activeMultiplier,
		TwoXThreshold:    in.TwoXThreshold,
		TwoXBetter:       twoX,
		Planned:          planned,
		Observed:         observed,
		Concentration:    concentration,
		Index:            index,
		Prediction:       prediction,
		Recall:           recall,
	}
	rep.Economics = economics(observed, activeMultiplier)
	rep.AgenticActivation = activationReport(in.AgenticActivation)
	rep.Planes = planeReport(planned, observed, in.KernelKV, in.Context, in.ExternalEngine)
	rep.ColdPathCorrect = coldPathCorrect(planned, observed, recall)
	rep.Actions, rep.Risks = actionsAndRisks(rep, in.MaxFalseWarmRate)
	rep.DefaultUsefulness = defaultUsefulness(rep, in.MaxFalseWarmRate)
	return rep
}

// economics folds the observed telemetry proof into the hit/read/rebate/cost
// economics the agent-dev gate reports. It returns nil when no telemetry was
// supplied, so the block is present exactly when a provider witnessed the reads.
func economics(observed *vcachegov.TelemetrySavingsProof, multiplier float64) *EconomicsReport {
	if observed == nil {
		return nil
	}
	hit := 0.0
	if observed.BaselineTokenEquiv > 0 {
		hit = observed.CacheReadTokens / observed.BaselineTokenEquiv
	}
	return &EconomicsReport{
		Source:              "telemetry",
		Witness:             "observed",
		HitRate:             hit,
		CacheReadTokens:     observed.CacheReadTokens,
		CacheCreationTokens: observed.CacheCreationTokens,
		RebateTokenEquiv:    observed.SavedTokenEquiv,
		RebatePct:           observed.SavedPct,
		CostTokenEquiv:      observed.ActualTokenEquiv,
		BaselineTokenEquiv:  observed.BaselineTokenEquiv,
		Multiplier:          multiplier,
	}
}

func planeReport(planned vcachegov.StarSavingsProof, observed *vcachegov.TelemetrySavingsProof, kernelKV, context, external PlaneEvidenceInput) PlaneReport {
	rep := PlaneReport{
		ProviderObserved: PlaneValueReport{
			Available:  false,
			Provenance: "MISSING",
			Reason:     "no provider telemetry supplied",
		},
		KernelWitnessed:        planeEvidence(kernelKV, "WITNESSED", "no fak-owned KV reuse witness supplied to this scorecard"),
		ContextWitnessed:       planeEvidence(context, "WITNESSED", "no O(1) context/query witness supplied to this scorecard"),
		ExternalEngineObserved: planeEvidence(external, "OBSERVED", "no external-engine cache capability witness supplied to this scorecard"),
		Forecast: PlaneValueReport{
			Available:          true,
			Provenance:         "FORECAST",
			Multiplier:         ratio(planned.BaselineTokenEquiv, planned.VCacheTokenEquiv),
			SavedTokenEquiv:    planned.SavedTokenEquiv,
			BaselineTokenEquiv: planned.BaselineTokenEquiv,
			CostTokenEquiv:     planned.VCacheTokenEquiv,
			Reason:             planned.Reason,
		},
	}
	if observed != nil {
		rep.ProviderObserved = PlaneValueReport{
			Available:          true,
			Provenance:         "OBSERVED",
			Multiplier:         ratio(observed.BaselineTokenEquiv, observed.ActualTokenEquiv),
			SavedTokenEquiv:    observed.SavedTokenEquiv,
			BaselineTokenEquiv: observed.BaselineTokenEquiv,
			CostTokenEquiv:     observed.ActualTokenEquiv,
			Reason:             observed.Reason,
		}
	}
	return rep
}

func planeEvidence(in PlaneEvidenceInput, defaultProvenance, missingReason string) PlaneValueReport {
	if !planeEvidenceAvailable(in) {
		return PlaneValueReport{
			Available:  false,
			Provenance: "MISSING",
			Reason:     missingReason,
		}
	}
	provenance := strings.TrimSpace(in.Provenance)
	if provenance == "" {
		provenance = defaultProvenance
	}
	reason := strings.TrimSpace(in.Reason)
	if reason == "" {
		reason = "cache plane witness supplied"
	}
	cost := nonNegativeFloat(in.CostTokenEquiv)
	saved := nonNegativeFloat(in.SavedTokenEquiv)
	baseline := nonNegativeFloat(in.BaselineTokenEquiv)
	if baseline == 0 && saved > 0 && cost > 0 {
		baseline = saved + cost
	}
	multiplier := in.Multiplier
	if multiplier == 0 && baseline > 0 && cost > 0 {
		multiplier = ratio(baseline, cost)
	}
	return PlaneValueReport{
		Available:          true,
		Provenance:         provenance,
		Multiplier:         multiplier,
		HitRate:            clamp01(in.HitRate),
		SavedTokenEquiv:    saved,
		BaselineTokenEquiv: baseline,
		CostTokenEquiv:     cost,
		Reason:             reason,
	}
}

func planeEvidenceAvailable(in PlaneEvidenceInput) bool {
	return in.Available ||
		in.Multiplier > 0 ||
		in.HitRate > 0 ||
		in.SavedTokenEquiv > 0 ||
		in.BaselineTokenEquiv > 0 ||
		in.CostTokenEquiv > 0
}

func activationReport(in AgenticActivationInput) AgenticActivationReport {
	rep := AgenticActivationReport{
		KernelKVEvents:          nonNegative(in.KernelKVEvents),
		ContextEvents:           nonNegative(in.ContextEvents),
		ProviderVCacheDecisions: nonNegative(in.ProviderVCacheDecisions),
		ExternalEngineEvents:    nonNegative(in.ExternalEngineEvents),
	}
	rep.Total = rep.KernelKVEvents + rep.ContextEvents + rep.ProviderVCacheDecisions + rep.ExternalEngineEvents
	rep.Active = rep.Total > 0
	return rep
}

func coldPathCorrect(planned vcachegov.StarSavingsProof, observed *vcachegov.TelemetrySavingsProof, recall vcachechain.RecallProof) bool {
	if planned.CorrectnessDependsOn || recall.CorrectnessDependsOn {
		return false
	}
	return observed == nil || !observed.CorrectnessDependsOn
}

func defaultUsefulness(rep Report, maxFalseWarm float64) DefaultUsefulnessReport {
	f := DefaultUsefulnessFacets{}
	if value := bestRealizedValue(rep); value > 0 {
		f.NetRealizedValue = int(math.Round(25 * value))
	}
	if rep.AgenticActivation.Active {
		f.AgenticActivation = 20
	}
	if rep.ColdPathCorrect {
		f.ColdPathCorrectness = 15
	}
	if hasObservedPlane(rep) {
		f.Granularity = 15
	} else if rep.Index.AnchorCount > 0 {
		f.Granularity = 8
	}
	f.DefaultCoverage = defaultCoverage(rep.Planes)
	if rep.Prediction.Total > 0 && rep.Prediction.FalseWarmRate <= maxFalseWarm {
		f.DriftResistance = 10
	} else if rep.Prediction.Total == 0 {
		f.DriftResistance = 3
	}
	if len(rep.Actions) > 0 {
		f.OperatorActionable = 5
	}
	score := f.NetRealizedValue + f.AgenticActivation + f.ColdPathCorrectness + f.Granularity +
		f.DefaultCoverage + f.DriftResistance + f.OperatorActionable
	return DefaultUsefulnessReport{
		Schema:          "fak.cache.default_usefulness.v1",
		Score:           score,
		Grade:           grade(score),
		Verdict:         defaultUsefulnessVerdict(score),
		ColdPathCorrect: rep.ColdPathCorrect,
		Facets:          f,
		Reason:          defaultUsefulnessReason(rep),
	}
}

func defaultCoverage(p PlaneReport) int {
	score := 0
	if p.ProviderObserved.Available {
		score += 3
	}
	if p.KernelWitnessed.Available {
		score += 2
	}
	if p.ContextWitnessed.Available {
		score += 2
	}
	if p.ExternalEngineObserved.Available {
		score += 2
	}
	if p.Forecast.Available {
		score++
	}
	if score > 10 {
		return 10
	}
	return score
}

func defaultUsefulnessVerdict(score int) string {
	switch {
	case score >= 75:
		return "default_ready"
	case score >= 50:
		return "partial"
	default:
		return "not_ready"
	}
}

func defaultUsefulnessReason(rep Report) string {
	switch {
	case !rep.ColdPathCorrect:
		return "cold-path correctness is not proven"
	case !hasObservedPlane(rep):
		return "no realized provider/kernel/context/external evidence supplied; score is mostly forecast"
	case bestRealizedValue(rep) <= 0 && !rep.AgenticActivation.Active:
		return "cache plane evidence was observed, but no realized value witness or fak-authored cache activation was supplied"
	case !rep.AgenticActivation.Active:
		if onlyProviderObserved(rep) {
			return "provider rebate observed, but no fak-authored cache activation was supplied"
		}
		return "realized cache value was observed, but no fak-authored cache activation was supplied"
	case bestRealizedValue(rep) <= 0:
		return "fak-authored cache activation was supplied, but realized cache value is not yet witnessed"
	default:
		return "realized cache value and fak-authored cache activation were both supplied"
	}
}

func bestRealizedValue(rep Report) float64 {
	best := 0.0
	if mult := bestRealizedMultiplier(rep); mult > 1 {
		best = clamp01((mult - 1) / (rep.TwoXThreshold - 1))
	}
	for _, p := range realizedPlanes(rep) {
		if p.Available && p.BaselineTokenEquiv > 0 && p.SavedTokenEquiv > 0 {
			if v := clamp01(p.SavedTokenEquiv / p.BaselineTokenEquiv); v > best {
				best = v
			}
		}
	}
	return best
}

func bestRealizedMultiplier(rep Report) float64 {
	best := 0.0
	for _, p := range realizedPlanes(rep) {
		if p.Available && p.Multiplier > best && !math.IsInf(p.Multiplier, 0) {
			best = p.Multiplier
		}
	}
	return best
}

func onlyProviderObserved(rep Report) bool {
	return rep.Planes.ProviderObserved.Available &&
		!rep.Planes.KernelWitnessed.Available &&
		!rep.Planes.ContextWitnessed.Available &&
		!rep.Planes.ExternalEngineObserved.Available
}

func hasObservedPlane(rep Report) bool {
	for _, p := range realizedPlanes(rep) {
		if p.Available {
			return true
		}
	}
	return false
}

func realizedPlanes(rep Report) []PlaneValueReport {
	return []PlaneValueReport{
		rep.Planes.ProviderObserved,
		rep.Planes.KernelWitnessed,
		rep.Planes.ContextWitnessed,
		rep.Planes.ExternalEngineObserved,
	}
}

func nonNegative(n int) int {
	if n < 0 {
		return 0
	}
	return n
}

func nonNegativeFloat(n float64) float64 {
	if n < 0 || math.IsNaN(n) {
		return 0
	}
	return n
}

func normalize(in Input) Input {
	def := DefaultInput()
	if in.Star.AnchorTokens == 0 && in.Star.Requests == 0 {
		in.Star = def.Star
	}
	if in.Star.ReadMult == 0 {
		in.Star.ReadMult = def.Star.ReadMult
	}
	if in.Star.WriteMult == 0 {
		in.Star.WriteMult = def.Star.WriteMult
	}
	if in.Star.MinPrefixTokens == 0 {
		in.Star.MinPrefixTokens = def.Star.MinPrefixTokens
	}
	if in.TelemetryReadMult == 0 {
		in.TelemetryReadMult = in.Star.ReadMult
	}
	if in.TelemetryWrite5m == 0 {
		in.TelemetryWrite5m = vcachegov.WriteMult5Minutes
	}
	if in.TelemetryWrite1h == 0 {
		in.TelemetryWrite1h = vcachegov.WriteMult1Hour
	}
	if len(in.Ranked) == 0 {
		in.Ranked = def.Ranked
	} else {
		in.Ranked = NormalizeRanked(in.Ranked)
	}
	switch strings.ToLower(strings.TrimSpace(in.AnchorSource)) {
	case AnchorSourceMeasured:
		in.AnchorSource = AnchorSourceMeasured
	default:
		in.AnchorSource = AnchorSourceSynthetic
	}
	if in.TurnsObserved < 0 {
		in.TurnsObserved = 0
	}
	if len(in.TelemetryRows) > 0 && in.TurnsObserved == 0 {
		in.TurnsObserved = len(in.TelemetryRows)
	}
	if in.TargetCoverage <= 0 || in.TargetCoverage > 1 {
		in.TargetCoverage = DefaultTargetCoverage
	}
	if in.Recall.PrefixTokens == 0 && in.Recall.UnitTokens == 0 {
		in.Recall = def.Recall
	}
	if in.Recall.ReadMult == 0 {
		in.Recall.ReadMult = in.Star.ReadMult
	}
	if in.TwoXThreshold <= 1 {
		in.TwoXThreshold = DefaultTwoXThreshold
	}
	if in.MaxFalseWarmRate <= 0 {
		in.MaxFalseWarmRate = DefaultMaxFalseWarmRate
	}
	return in
}

// NormalizeRanked removes non-positive anchor rows, fills size/reuse defaults,
// and sorts the list by descending vCache ranking weight.
func NormalizeRanked(rows []vcachecal.RankedVBlock) []vcachecal.RankedVBlock {
	out := make([]vcachecal.RankedVBlock, 0, len(rows))
	for i, row := range rows {
		if row.Key == "" {
			row.Key = fmt.Sprintf("anchor-%04d", i+1)
		}
		if row.Size == 0 {
			row.Size = 1
		}
		if row.ReuseDensity == 0 {
			row.ReuseDensity = 1
		}
		if row.Weight() <= 0 {
			continue
		}
		out = append(out, row)
	}
	sort.SliceStable(out, func(i, j int) bool {
		wi, wj := out[i].Weight(), out[j].Weight()
		if wi == wj {
			return out[i].Key < out[j].Key
		}
		return wi > wj
	})
	return out
}

// PlanIndex chooses the smallest hot-anchor index that reaches target coverage.
func PlanIndex(c vcachecal.Concentration, target float64) IndexPlan {
	if target <= 0 || target > 1 {
		target = DefaultTargetCoverage
	}
	p := IndexPlan{
		TargetCoverage: target,
		TotalAnchors:   len(c.TopNCoverage),
		Defeated:       c.Defeated,
		Recommendation: c.Recommendation,
	}
	if len(c.TopNCoverage) == 0 {
		p.Defeated = true
		if p.Recommendation == "" {
			p.Recommendation = "no coverage curve: rank anchors before building a vCache index"
		}
		return p
	}
	keys := make([]int, 0, len(c.TopNCoverage))
	for n := range c.TopNCoverage {
		keys = append(keys, n)
	}
	sort.Ints(keys)
	last := keys[len(keys)-1]
	p.AnchorCount = last
	p.Coverage = c.TopNCoverage[last]
	for _, n := range keys {
		cov := c.TopNCoverage[n]
		if cov >= target {
			p.AnchorCount = n
			p.Coverage = cov
			break
		}
	}
	return p
}

// BuildIndexArtifact emits the selected hot-anchor index rows for the supplied
// workload. The artifact is deterministic and payload-free, so it is safe to
// check into benchmark output or feed to later cache planners.
func BuildIndexArtifact(ranked []vcachecal.RankedVBlock, target float64) AnchorIndexArtifact {
	ranked = NormalizeRanked(ranked)
	concentration := vcachecal.FitConcentration(ranked)
	plan := PlanIndex(concentration, target)
	artifact := AnchorIndexArtifact{
		Schema:         "fak.vcache.anchor_index.v1",
		TargetCoverage: plan.TargetCoverage,
		TotalAnchors:   plan.TotalAnchors,
		AnchorCount:    plan.AnchorCount,
		Coverage:       plan.Coverage,
		Defeated:       plan.Defeated,
		Recommendation: plan.Recommendation,
	}
	for _, row := range ranked {
		artifact.TotalWeight += row.Weight()
	}
	if artifact.TotalWeight <= 0 || plan.AnchorCount <= 0 {
		return artifact
	}
	limit := plan.AnchorCount
	if limit > len(ranked) {
		limit = len(ranked)
	}
	cum := 0.0
	artifact.Entries = make([]AnchorIndexEntry, 0, limit)
	for i := 0; i < limit; i++ {
		w := ranked[i].Weight()
		cum += w
		artifact.Entries = append(artifact.Entries, AnchorIndexEntry{
			Rank:     i + 1,
			Key:      ranked[i].Key,
			Weight:   w,
			Coverage: cum / artifact.TotalWeight,
		})
	}
	return artifact
}

func overallScore(multiplier, threshold float64, c vcachecal.Concentration, idx IndexPlan, p PredictionReport, maxFalseWarm float64) int {
	score := 0.0
	if multiplier > 1 && !math.IsInf(multiplier, 1) {
		score += 45 * clamp01((multiplier-1)/(threshold-1))
	}
	if c.Measured && !c.Defeated {
		score += 20 * clamp01((c.ZipfS-1)/0.74)
	}
	if idx.AnchorCount > 0 && !idx.Defeated {
		score += 15 * clamp01(idx.Coverage/idx.TargetCoverage)
	}
	risk := 1.0
	if p.Total > 0 {
		falseWarmPenalty := clamp01(p.FalseWarmRate / maxFalseWarm)
		falseColdPenalty := clamp01(p.FalseColdRate / 0.5)
		risk = 1 - 0.75*falseWarmPenalty - 0.25*falseColdPenalty
	}
	score += 20 * clamp01(risk)
	if score > 100 {
		score = 100
	}
	return int(math.Round(score))
}

func actionsAndRisks(rep Report, maxFalseWarm float64) ([]string, []string) {
	var actions []string
	var risks []string
	if rep.TwoXBetter {
		actions = append(actions, "ship the star-anchor path behind telemetry; keep uncached-first budgeting and prove realized savings per run")
	}
	if rep.ActiveSource == "planned" {
		actions = append(actions, "collect provider telemetry with fak vcache prove-telemetry, then re-score on observed cache_read counters")
	}
	if !rep.Planes.KernelWitnessed.Available {
		actions = append(actions, "capture a fak-owned KV witness: run guard/serve long enough to append the cache-value ledger, or pass --kernel-ledger/--kernel-kv-* evidence")
	}
	if !rep.Planes.ContextWitnessed.Available {
		actions = append(actions, "capture a context-plane witness: run guard/serve until compact-history or ctx-view fires, or pass --context-shed-tokens with resident-token cost")
	}
	if rep.Planned.Status != vcachegov.ProofProven {
		risks = append(risks, rep.Planned.Reason)
		actions = append(actions, "raise sibling reuse, anchor length, or read discount before spending dedicated vCache engineering")
	}
	if rep.Index.AnchorCount > 0 && !rep.Index.Defeated {
		actions = append(actions, fmt.Sprintf("build a hot-anchor index for the top %d anchors; expected coverage %.1f%%", rep.Index.AnchorCount, 100*rep.Index.Coverage))
	}
	if rep.Concentration.Defeated {
		risks = append(risks, rep.Concentration.Recommendation)
		actions = append(actions, "manufacture skew first: canonicalize prompts and aggregate tiny anchors before warming")
	}
	if rep.Prediction.FalseWarmRate > maxFalseWarm {
		risks = append(risks, fmt.Sprintf("false-warm rate %.2f%% exceeds %.2f%%", 100*rep.Prediction.FalseWarmRate, 100*maxFalseWarm))
		actions = append(actions, "tighten the warmth estimator: demote on zero-read and refresh calibration before enabling automatic warms")
	}
	if rep.Recall.Status == vcachechain.ProofRefuted {
		actions = append(actions, fmt.Sprintf("keep chain recall off for single units; batch at least %d siblings before rebuild", rep.Recall.BreakEvenSiblings))
	}
	if len(actions) == 0 {
		actions = append(actions, "no vCache action recommended until telemetry or concentration evidence improves")
	}
	return actions, risks
}

func status(twoX, proven bool, multiplier float64) string {
	switch {
	case twoX:
		return "2x_ready"
	case proven && multiplier > 1:
		return "promising"
	default:
		return "not_ready"
	}
}

func grade(score int) string {
	switch {
	case score >= 90:
		return "A"
	case score >= 75:
		return "B"
	case score >= 60:
		return "C"
	case score >= 40:
		return "D"
	default:
		return "F"
	}
}

func ratio(num, den float64) float64 {
	if den == 0 {
		if num > 0 {
			return math.Inf(1)
		}
		return 0
	}
	return num / den
}

func clamp01(x float64) float64 {
	if x < 0 || math.IsNaN(x) {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}
