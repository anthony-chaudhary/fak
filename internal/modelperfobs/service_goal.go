package modelperfobs

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strings"
)

const ServiceGoalSchema = "fak-service-goal/1"

// ServiceGoal separates reusable run evidence from the latency limits being
// evaluated. References are caller attestations, not verified artifact contents.
type ServiceGoal struct {
	Schema                string             `json:"schema"`
	ID                    string             `json:"id"`
	WorkloadRef           string             `json:"workload_ref"`
	TTFTLimitMS           float64            `json:"ttft_limit_ms"`
	TPOTLimitMS           float64            `json:"tpot_limit_ms"`
	WarmupPolicy          string             `json:"warmup_policy"`
	SingleTokenTPOTPolicy string             `json:"single_token_tpot_policy"`
	Run                   ServiceRunEvidence `json:"run"`
}

// ServiceRunEvidence describes one common offered-workload accounting window.
// OfferedRequests excludes explicitly marked warmups and includes absent rows,
// rejections and failures. WallSeconds must include the corresponding drain.
type ServiceRunEvidence struct {
	SourceRef       string               `json:"source_ref"`
	ModelRef        string               `json:"model_ref"`
	ConfigRef       string               `json:"config_ref"`
	WallSeconds     float64              `json:"wall_seconds"`
	OfferedRequests int64                `json:"offered_requests"`
	Quality         QualityQualification `json:"quality"`
	QualityRef      string               `json:"quality_ref"`
	Cost            *float64             `json:"cost"`
	CostUnit        string               `json:"cost_unit"`
}

type ServiceGoodput struct {
	RequestsPerSecond         float64 `json:"requests_per_second"`
	CompletionTokensPerSecond float64 `json:"completion_tokens_per_second"`
}

// ServiceTimingCoverage is a marginal timing statistic, not a joint-SLO test.
// A missing value is null, and its denominator always retains the offered load.
type ServiceTimingCoverage struct {
	ValueMS          *float64 `json:"value_ms"`
	MeasuredRequests int64    `json:"measured_requests"`
	OfferedRequests  int64    `json:"offered_requests"`
	CoverageFraction float64  `json:"coverage_fraction"`
}

type ServiceGoalReport struct {
	Schema                    string                `json:"schema"`
	Goal                      ServiceGoal           `json:"goal"`
	ObservedRequests          int64                 `json:"observed_requests"`
	MissingRequests           int64                 `json:"missing_requests"`
	WarmupRequests            int64                 `json:"warmup_requests"`
	Errors                    int64                 `json:"errors"`
	SLOPassedRequests         int64                 `json:"slo_passed_requests"`
	SLOFailedRequests         int64                 `json:"slo_failed_requests"`
	SLOUnknownRequests        int64                 `json:"slo_unknown_requests"`
	TPOTNotApplicableRequests int64                 `json:"tpot_not_applicable_requests"`
	SLOAttainment             float64               `json:"slo_attainment"`
	MeasuredSLOGoodput        ServiceGoodput        `json:"measured_slo_goodput"`
	QualityQualifiedGoodput   *ServiceGoodput       `json:"quality_qualified_goodput"`
	TTFTP95                   ServiceTimingCoverage `json:"ttft_p95"`
	TPOTP95                   ServiceTimingCoverage `json:"tpot_p95"`
	CostPerQualifiedRequest   *float64              `json:"cost_per_qualified_request"`
	Unknowns                  []string              `json:"unknowns"`
}

func ReadServiceGoal(r io.Reader) (ServiceGoal, error) {
	var g ServiceGoal
	d := json.NewDecoder(r)
	d.DisallowUnknownFields()
	if err := d.Decode(&g); err != nil {
		return g, fmt.Errorf("decode service goal: %w", err)
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return g, fmt.Errorf("service goal must contain exactly one JSON object")
	}
	return g, g.Validate()
}

func (g ServiceGoal) Validate() error {
	if g.Schema != ServiceGoalSchema {
		return fmt.Errorf("service goal schema %q, want %q", g.Schema, ServiceGoalSchema)
	}
	for _, field := range []struct{ name, value string }{
		{"id", g.ID}, {"workload_ref", g.WorkloadRef}, {"run.source_ref", g.Run.SourceRef},
		{"run.model_ref", g.Run.ModelRef}, {"run.config_ref", g.Run.ConfigRef},
	} {
		if strings.TrimSpace(field.value) == "" {
			return fmt.Errorf("service goal %s is required", field.name)
		}
	}
	if !finite(g.TTFTLimitMS) || g.TTFTLimitMS <= 0 || !finite(g.TPOTLimitMS) || g.TPOTLimitMS <= 0 {
		return fmt.Errorf("service goal TTFT and TPOT limits must be finite and positive")
	}
	if g.WarmupPolicy != "exclude" {
		return fmt.Errorf("service goal warmup_policy must explicitly be exclude")
	}
	if g.SingleTokenTPOTPolicy != "ttft-only" && g.SingleTokenTPOTPolicy != "unknown" {
		return fmt.Errorf("service goal single_token_tpot_policy must be ttft-only or unknown")
	}
	if !finite(g.Run.WallSeconds) || g.Run.WallSeconds <= 0 || g.Run.OfferedRequests <= 0 {
		return fmt.Errorf("service goal run requires positive finite wall_seconds and positive offered_requests")
	}
	if !validQuality(g.Run.Quality) {
		return fmt.Errorf("service goal run quality must be qualified, failed or unknown")
	}
	if g.Run.Quality != QualityUnknown && strings.TrimSpace(g.Run.QualityRef) == "" {
		return fmt.Errorf("service goal known quality requires an external quality_ref")
	}
	if g.Run.Cost != nil {
		if !finite(*g.Run.Cost) || *g.Run.Cost < 0 || strings.TrimSpace(g.Run.CostUnit) == "" {
			return fmt.Errorf("service goal cost must be finite and nonnegative with an explicit cost_unit")
		}
	} else if g.Run.CostUnit != "" {
		return fmt.Errorf("service goal cost_unit requires cost")
	}
	return nil
}

// EvaluateServiceGoal counts joint per-request SLO passes, never an intersection
// inferred from marginal percentiles. Unknown timing cannot contribute goodput.
// QualityQualification is externally supplied and applies to this entire run.
func EvaluateServiceGoal(g ServiceGoal, rows []Observation) (ServiceGoalReport, error) {
	r := ServiceGoalReport{Schema: "fak-service-goal-report/1", Goal: g, Unknowns: []string{}}
	if err := g.Validate(); err != nil {
		return r, err
	}
	seen := make(map[string]bool, len(rows))
	var allPrompt, allCompletion, goodTokens int64
	var ttft, tpot []float64
	for i, row := range rows {
		if err := validateServiceObservation(row, seen); err != nil {
			return r, fmt.Errorf("service goal observation %d: %w", i+1, err)
		}
		if row.PromptTokens > math.MaxInt64-allPrompt || row.CompletionTokens > math.MaxInt64-allCompletion {
			return r, fmt.Errorf("service goal aggregate token count overflow")
		}
		allPrompt += row.PromptTokens
		allCompletion += row.CompletionTokens
		if row.Warmup {
			r.WarmupRequests++
			continue
		}
		r.ObservedRequests++
		if row.Error != "" || (row.Status != 0 && (row.Status < 200 || row.Status >= 300)) {
			r.Errors++
			continue
		}
		if row.Status == 0 || !row.Streaming || row.CompletionTokens == 0 || row.DurationMS == 0 {
			r.SLOUnknownRequests++
			continue
		}
		if row.TTFTMS > 0 {
			ttft = append(ttft, row.TTFTMS)
		}
		if row.CompletionTokens > 1 && row.TPOTMS > 0 {
			tpot = append(tpot, row.TPOTMS)
		}
		if row.CompletionTokens == 1 {
			r.TPOTNotApplicableRequests++
		}
		if row.TTFTMS > g.TTFTLimitMS || (row.CompletionTokens > 1 && row.TPOTMS > g.TPOTLimitMS) {
			r.SLOFailedRequests++
		} else if row.TTFTMS == 0 || (row.CompletionTokens > 1 && row.TPOTMS == 0) ||
			(row.CompletionTokens == 1 && g.SingleTokenTPOTPolicy == "unknown") {
			r.SLOUnknownRequests++
		} else {
			r.SLOPassedRequests++
			goodTokens += row.CompletionTokens // bounded by validated allCompletion
		}
	}
	if r.ObservedRequests > g.Run.OfferedRequests {
		return r, fmt.Errorf("service goal offered_requests is smaller than observed non-warmup requests")
	}
	r.MissingRequests = g.Run.OfferedRequests - r.ObservedRequests
	r.SLOUnknownRequests += r.MissingRequests
	r.SLOAttainment = float64(r.SLOPassedRequests) / float64(g.Run.OfferedRequests)
	r.MeasuredSLOGoodput = ServiceGoodput{float64(r.SLOPassedRequests) / g.Run.WallSeconds, float64(goodTokens) / g.Run.WallSeconds}
	if !finite(r.MeasuredSLOGoodput.RequestsPerSecond) || !finite(r.MeasuredSLOGoodput.CompletionTokensPerSecond) {
		return r, fmt.Errorf("service goal goodput overflow")
	}
	r.TTFTP95, r.TPOTP95 = serviceTimingCoverage(ttft, g.Run.OfferedRequests), serviceTimingCoverage(tpot, g.Run.OfferedRequests)
	qualified := int64(0)
	switch g.Run.Quality {
	case QualityQualified:
		value := r.MeasuredSLOGoodput
		r.QualityQualifiedGoodput, qualified = &value, r.SLOPassedRequests
	case QualityFailed:
		r.QualityQualifiedGoodput = &ServiceGoodput{}
	case QualityUnknown:
		r.Unknowns = append(r.Unknowns, "external quality is unknown; quality-qualified goodput is null")
	}
	if g.Run.Cost != nil && qualified > 0 {
		value := *g.Run.Cost / float64(qualified)
		r.CostPerQualifiedRequest = &value
	} else {
		r.Unknowns = append(r.Unknowns, "cost per qualified request requires cost and at least one quality-qualified SLO pass")
	}
	if r.SLOUnknownRequests > 0 {
		r.Unknowns = append(r.Unknowns, "missing or insufficiently timed requests do not contribute measured SLO goodput")
	}
	if r.TTFTP95.MeasuredRequests < g.Run.OfferedRequests || r.TPOTP95.MeasuredRequests < g.Run.OfferedRequests {
		r.Unknowns = append(r.Unknowns, "marginal p95 timing has incomplete offered-request coverage; one-token TPOT is undefined")
	}
	return r, nil
}

func validateServiceObservation(r Observation, seen map[string]bool) error {
	if r.Schema != Schema || strings.TrimSpace(r.RequestID) == "" || seen[r.RequestID] {
		return fmt.Errorf("invalid schema, missing request_id or duplicate request_id %q", r.RequestID)
	}
	seen[r.RequestID] = true
	if r.Status < 0 || r.Status > 599 || r.PromptTokens < 0 || r.CompletionTokens < 0 || r.TotalTokens < 0 || r.InterChunkCount < 0 ||
		r.PromptTokens > math.MaxInt64-r.CompletionTokens {
		return fmt.Errorf("invalid status/token count or token count overflow")
	}
	if r.TotalTokens != 0 && r.TotalTokens != r.PromptTokens+r.CompletionTokens {
		return fmt.Errorf("total_tokens does not match prompt_tokens + completion_tokens")
	}
	for _, v := range []float64{r.DurationMS, r.TTFTMS, r.TPOTMS, r.OutputTokensPerSec, r.InterChunkP50MS, r.InterChunkP95MS} {
		if !finite(v) || v < 0 {
			return fmt.Errorf("timing/rate values must be finite and nonnegative")
		}
	}
	return nil
}

func serviceTimingCoverage(values []float64, offered int64) ServiceTimingCoverage {
	c := ServiceTimingCoverage{MeasuredRequests: int64(len(values)), OfferedRequests: offered, CoverageFraction: float64(len(values)) / float64(offered)}
	if len(values) != 0 {
		v := percentile(values, .95)
		c.ValueMS = &v
	}
	return c
}
