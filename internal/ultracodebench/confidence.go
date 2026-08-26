package ultracodebench

import (
	"errors"
	"math"
	"math/rand"
	"sort"
)

const ConfidenceCampaignSchema = "fak-ultracode-factorial-campaign/2"

type ConfidenceInterval struct {
	Low  float64 `json:"low"`
	High float64 `json:"high"`
}

type ConfidenceEffect struct {
	Estimate float64            `json:"estimate"`
	Interval ConfidenceInterval `json:"confidence_interval_95"`
}

type AttributionConfidence struct {
	ScopedPercent                    ConfidenceEffect `json:"scoped_percent"`
	PrefixPercent                    ConfidenceEffect `json:"prefix_percent"`
	FixedClaimPercent                float64          `json:"fixed_claim_percent"`
	MaterialityBoundPercentagePoints float64          `json:"materiality_bound_percentage_points"`
	FixedClaimVerdict                string           `json:"fixed_claim_verdict"`
}

type ConfidenceWidthResult struct {
	Width                     int                   `json:"width"`
	Verdict                   string                `json:"verdict"`
	Reason                    string                `json:"reason,omitempty"`
	AcceptedRuns              int                   `json:"accepted_runs"`
	OutcomeAbstentions        int                   `json:"outcome_abstentions"`
	WithinRunMeasurementNoise float64               `json:"within_run_measurement_noise"`
	Scope                     ConfidenceEffect      `json:"scope"`
	Prefix                    ConfidenceEffect      `json:"prefix"`
	Combined                  ConfidenceEffect      `json:"combined"`
	Interaction               ConfidenceEffect      `json:"interaction"`
	Attribution               AttributionConfidence `json:"attribution"`
}

type ConfidenceReport struct {
	Schema                 string                  `json:"schema"`
	SourceArtifact         string                  `json:"source_artifact"`
	Metric                 string                  `json:"metric"`
	ConfidenceLevel        float64                 `json:"confidence_level"`
	BootstrapSamples       int                     `json:"bootstrap_samples"`
	RandomSeed             int64                   `json:"random_seed"`
	Widths                 []ConfidenceWidthResult `json:"widths"`
	ReplayCommand          string                  `json:"replay_command"`
	PromotionEvidence      string                  `json:"promotion_evidence"`
	DemotionEvidence       string                  `json:"demotion_evidence"`
	InvalidatingAssumption string                  `json:"invalidating_assumption"`
}

type runCell struct {
	run, order int
	work       float64
}

func EvaluateConfidenceCampaign(c FactorialCampaign, widths []int) (ConfidenceReport, error) {
	r := ConfidenceReport{Schema: "fak-ultracode-factorial-confidence/1", SourceArtifact: c.SourceArtifact, Metric: c.Metric,
		ConfidenceLevel: c.ConfidenceLevel, BootstrapSamples: c.BootstrapSamples, RandomSeed: c.RandomSeed,
		ReplayCommand:     "fak ultracode bench --factorial internal/ultracodebench/testdata/issue8672-confidence-campaign.json --json",
		PromotionEvidence: c.PromotionEvidence, DemotionEvidence: c.DemotionEvidence, InvalidatingAssumption: c.InvalidatingAssumption}
	if c.Schema != ConfidenceCampaignSchema {
		return r, errors.New("confidence campaign schema is required")
	}
	if c.OrderPolicy != "randomized" || c.BootstrapSamples < 1000 || c.ConfidenceLevel != .95 || c.MaterialityBound <= 0 {
		return r, errors.New("predeclared randomized bootstrap design is required")
	}
	for _, width := range widths {
		r.Widths = append(r.Widths, evaluateConfidenceWidth(c, width))
	}
	return r, nil
}

func evaluateConfidenceWidth(c FactorialCampaign, width int) ConfidenceWidthResult {
	out := ConfidenceWidthResult{Width: width, Verdict: "ABSTAIN"}
	cells := map[string]map[int]runCell{}
	abstain := map[int]bool{}
	for _, cell := range c.Cells {
		if cell.Width != width {
			continue
		}
		if cells[cell.Treatment] == nil {
			cells[cell.Treatment] = map[int]runCell{}
		}
		for _, rep := range cell.Replicates {
			if !rep.Accepted || rep.OutcomeDigest != c.OutcomeDigest {
				abstain[rep.Run] = true
				continue
			}
			if rep.Run <= 0 || rep.Order < 1 || rep.Order > 4 || rep.Work <= 0 || rep.ResetReceipt == "" || (cell.Cache == "warm" && (rep.WarmupReceipt == "" || rep.AuthoritativeReadReceipt == "" || rep.CachedTokens <= 0)) {
				out.Reason = "missing run, telemetry, or reset/warmup receipt"
				return out
			}
			cells[cell.Treatment][rep.Run] = runCell{rep.Run, rep.Order, rep.Work}
		}
	}
	runs := make([]int, 0)
	for run := range cells["A"] {
		if !abstain[run] {
			runs = append(runs, run)
		}
	}
	sort.Ints(runs)
	out.OutcomeAbstentions = len(abstain)
	if len(runs) < 5 {
		out.Reason = "fewer than five equal-outcome repetitions"
		return out
	}
	effects := make([][4]float64, 0, len(runs))
	orderPatterns := map[[4]int]bool{}
	for _, run := range runs {
		a, aok := cells["A"][run]
		b, bok := cells["B"][run]
		cc, cok := cells["C"][run]
		d, dok := cells["D"][run]
		if !aok || !bok || !cok || !dok {
			out.Reason = "incomplete randomized run"
			return out
		}
		pattern := [4]int{a.order, b.order, cc.order, d.order}
		seen := map[int]bool{}
		for _, x := range pattern {
			if seen[x] {
				out.Reason = "duplicate order within run"
				return out
			}
			seen[x] = true
		}
		orderPatterns[pattern] = true
		effects = append(effects, [4]float64{cc.work - a.work, b.work - a.work, d.work - a.work, d.work - b.work - cc.work + a.work})
	}
	if len(orderPatterns) < 2 {
		out.Reason = "run order was not randomized"
		return out
	}
	out.AcceptedRuns = len(effects)
	estimates := meanEffects(effects)
	samples := make([][][4]float64, 0)
	_ = samples
	boots := [4][]float64{}
	rng := rand.New(rand.NewSource(c.RandomSeed + int64(width)))
	for i := 0; i < c.BootstrapSamples; i++ {
		draw := make([][4]float64, len(effects))
		for j := range draw {
			draw[j] = effects[rng.Intn(len(effects))]
		}
		m := meanEffects(draw)
		for k := 0; k < 4; k++ {
			boots[k] = append(boots[k], m[k])
		}
	}
	cis := [4]ConfidenceInterval{}
	for k := 0; k < 4; k++ {
		sort.Float64s(boots[k])
		cis[k] = ConfidenceInterval{percentile(boots[k], .025), percentile(boots[k], .975)}
	}
	out.Scope = ConfidenceEffect{estimates[0], cis[0]}
	out.Prefix = ConfidenceEffect{estimates[1], cis[1]}
	out.Combined = ConfidenceEffect{estimates[2], cis[2]}
	out.Interaction = ConfidenceEffect{estimates[3], cis[3]}
	var noise float64
	for _, e := range effects {
		for k := 0; k < 4; k++ {
			d := e[k] - estimates[k]
			noise += d * d
		}
	}
	out.WithinRunMeasurementNoise = math.Sqrt(noise / float64(len(effects)*4))
	shares := make([]float64, 0, c.BootstrapSamples)
	for i := 0; i < c.BootstrapSamples; i++ {
		denom := boots[0][i] + boots[1][i]
		if denom != 0 {
			shares = append(shares, 100*boots[0][i]/denom)
		}
	}
	sort.Float64s(shares)
	point := 100 * estimates[0] / (estimates[0] + estimates[1])
	sci := ConfidenceInterval{percentile(shares, .025), percentile(shares, .975)}
	verdict := "REJECT"
	lower := 62.7 - c.MaterialityBound
	upper := 62.7 + c.MaterialityBound
	if sci.Low >= lower && sci.High <= upper {
		verdict = "RETAIN"
	}
	out.Attribution = AttributionConfidence{ConfidenceEffect{point, sci}, ConfidenceEffect{100 - point, ConfidenceInterval{100 - sci.High, 100 - sci.Low}}, 62.7, c.MaterialityBound, verdict}
	out.Verdict = "GAIN"
	if out.Combined.Interval.High >= 0 {
		out.Verdict = "NO_GAIN"
		out.Reason = "combined confidence interval includes no gain"
	}
	return out
}
func meanEffects(xs [][4]float64) (m [4]float64) {
	for _, x := range xs {
		for k := 0; k < 4; k++ {
			m[k] += x[k]
		}
	}
	for k := 0; k < 4; k++ {
		m[k] /= float64(len(xs))
	}
	return
}
func percentile(xs []float64, p float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	return xs[int(p*float64(len(xs)-1))]
}
