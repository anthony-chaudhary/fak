package webbench

// K3.1 bench-campaign arm-matrix plan and per-arm receipt fold (#14374).
//
// The campaign measures the fak-native serving engine against itself across a
// fixed arm matrix: speculative-decode arms (no-spec baseline, DSpark block 4,
// DSpark block 5 — #14369) crossed with the concurrency ladder {1, 8, 32, 64},
// plus hot-expert CPU-RAM offload placeholders gated on #14372. Every witnessed
// arm is one fak.serving-sweep.v1 artifact (RunServingSweep /
// WriteServingSweepReport) for that arm; FoldCampaignIndex projects the
// per-arm receipts onto the plan matrix and folds the comparable metrics into
// one index.
//
// This file is pure and deterministic: no network, no clock, no CLI wiring
// (the `fak webbench campaign` surface is a filed follow-on leaf). Nothing here
// relaxes the sweep claim gate (serving_sweep_claim.go): peak/knee claims
// still require the per-arm sweep artifact.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Campaign receipt schemas. The plan and the index are derived artifacts over
// the existing per-arm serving-sweep receipt; they mint their own schema names
// so a consumer can tell the three artifact classes apart.
const (
	CampaignPlanSchema  = "fak.k31-campaign-plan.v1"
	CampaignIndexSchema = "fak.k31-campaign-index.v1"
)

// Canonical campaign arm names. A per-arm report is bound to its arm by this
// name; FoldCampaignIndex refuses reports for arms outside the plan matrix so
// a typo'd arm cannot silently vanish from the index.
const (
	CampaignArmSpecNone    = "spec-none"
	CampaignArmSpecDSpark4 = "spec-dspark4"
	CampaignArmSpecDSpark5 = "spec-dspark5"
)

// Speculative-decode arm configuration strings. "none" is the no-spec
// baseline; "dspark:N" is the DSpark drafter with block length N (#14369).
const (
	CampaignSpecConfigNone    = "none"
	CampaignSpecConfigDSpark4 = "dspark:4"
	CampaignSpecConfigDSpark5 = "dspark:5"
)

// Campaign tracking issues (spine map:
// docs/plans/k31-top-items-spine-2026-11-12.md).
const (
	CampaignIssue       = "#14374"
	CampaignOffloadGate = "#14372"
)

// Arm and row lifecycle statuses. Offload placeholders are gated, never
// planned: their receipt fields do not exist until #14372 lands them.
const (
	CampaignArmStatusGated = "gated"

	CampaignRowMeasured                  = "measured"
	CampaignRowMissing                   = "missing"
	CampaignRowPresentWithoutSpecMetrics = "present-without-spec-metrics"
)

// CampaignArm is one speculative-decode arm coordinate of the campaign matrix:
// an arm configuration measured at one concurrency point.
type CampaignArm struct {
	Name              string `json:"name"`
	SpeculativeConfig string `json:"speculative_config"`
	Concurrency       int    `json:"concurrency"`
}

// OffloadArm is the #14372 hot-expert CPU-RAM offload arm placeholder. The
// offload arm has no receipt seam yet (the hot-expert hit-rate fields are a
// filed follow-on leaf), so it stays a plan placeholder gated on #14372.
type OffloadArm struct {
	Name        string  `json:"name"`
	HotFraction float64 `json:"hot_fraction"`
	Status      string  `json:"status"`
	Gate        string  `json:"gate"`
}

// CampaignPlan is the deterministic arm-matrix plan receipt.
type CampaignPlan struct {
	Schema            string        `json:"schema"`
	CampaignIssue     string        `json:"campaign_issue"`
	PromotionRequires []string      `json:"promotion_requires"`
	ReceiptSchema     string        `json:"receipt_schema"`
	SpecArms          []CampaignArm `json:"spec_arms"`
	OffloadArms       []OffloadArm  `json:"offload_arms"`
}

// K31CampaignPlan returns the canonical #14374 arm matrix: the three
// speculative-decode arms (no-spec baseline, DSpark block 4, DSpark block 5)
// crossed with the concurrency ladder {1, 8, 32, 64} = 12 planned sweep
// points, plus the hot-expert offload placeholders at h=0.5/0.75/0.85 gated on
// #14372. Deterministic: fixed arm order, no network, no clock. Witnessed arms
// are a later rung (Promotion-requires #14368/#14371/#14370); this plan is the
// contract every per-arm fak.serving-sweep.v1 artifact folds into.
func K31CampaignPlan() *CampaignPlan {
	plan := &CampaignPlan{
		Schema:            CampaignPlanSchema,
		CampaignIssue:     CampaignIssue,
		PromotionRequires: []string{"#14368", "#14371", "#14370"},
		ReceiptSchema:     ServingSweepSchema,
	}
	specArms := []struct{ name, config string }{
		{CampaignArmSpecNone, CampaignSpecConfigNone},
		{CampaignArmSpecDSpark4, CampaignSpecConfigDSpark4},
		{CampaignArmSpecDSpark5, CampaignSpecConfigDSpark5},
	}
	for _, spec := range specArms {
		for _, concurrency := range []int{1, 8, 32, 64} {
			plan.SpecArms = append(plan.SpecArms, CampaignArm{
				Name:              spec.name,
				SpeculativeConfig: spec.config,
				Concurrency:       concurrency,
			})
		}
	}
	for _, hotFraction := range []float64{0.5, 0.75, 0.85} {
		plan.OffloadArms = append(plan.OffloadArms, OffloadArm{
			Name:        fmt.Sprintf("offload-h%03d", int(hotFraction*100)),
			HotFraction: hotFraction,
			Status:      CampaignArmStatusGated,
			Gate:        CampaignOffloadGate,
		})
	}
	return plan
}

// CampaignSpecMetrics is the additive spec-decode join a spec arm folds into
// its fak.serving-sweep.v1 track point. Documented source: the engine spec
// counters in internal/enginestep — the fak_engine_spec_accepted_per_round
// histogram (sum/count) and the snapshot SpecStat (AcceptedTokens/Rounds);
// AcceptanceLength is the mean accepted tokens per verify round. The field is
// optional: receipts written before the join parse unchanged, and a spec arm
// whose point lacks it folds as present-without-spec-metrics rather than
// inventing a number.
type CampaignSpecMetrics struct {
	AcceptanceLength *float64 `json:"acceptance_length,omitempty"`
}

// campaignSpecJoin extracts the optional additive spec join from raw
// fak.serving-sweep.v1 JSON. The base receipt struct (ServingSweepReport) is
// frozen, so the join is read in a second targeted pass over the same bytes
// instead of extending serving_sweep.go.
type campaignSpecJoin struct {
	Points []struct {
		Concurrency int `json:"concurrency"`
		Tracks      []struct {
			Track ServingTrack         `json:"track"`
			Spec  *CampaignSpecMetrics `json:"spec,omitempty"`
		} `json:"tracks"`
	} `json:"points"`
}

// CampaignArmReport binds one per-arm fak.serving-sweep.v1 artifact — the raw
// JSON exactly as written by WriteServingSweepReport — to its campaign arm
// name.
type CampaignArmReport struct {
	Arm    string
	Report []byte
}

// CampaignIndexRow is one (arm, concurrency) projection of a per-arm sweep
// receipt onto the plan matrix. SingleStreamTokS is the arm's c=1 point (the
// single-stream anchor, repeated on every row of that arm); AggregateTokS,
// TTFTP99Millis, and ITLP99Millis are this row's own point. AcceptanceLength
// is present only when the arm carries the additive spec join.
type CampaignIndexRow struct {
	Arm               string   `json:"arm"`
	SpeculativeConfig string   `json:"speculative_config"`
	Concurrency       int      `json:"concurrency"`
	SingleStreamTokS  *float64 `json:"single_stream_tok_s,omitempty"`
	AggregateTokS     *float64 `json:"aggregate_tok_s,omitempty"`
	TTFTP99Millis     *float64 `json:"ttft_p99_ms,omitempty"`
	ITLP99Millis      *float64 `json:"itl_p99_ms,omitempty"`
	AcceptanceLength  *float64 `json:"acceptance_length,omitempty"`
	Status            string   `json:"status"`
	Reason            string   `json:"reason,omitempty"`
}

// CampaignIndex is the folded campaign artifact: the plan matrix projected
// onto the witnessed per-arm receipts.
type CampaignIndex struct {
	Schema            string             `json:"schema"`
	CampaignIssue     string             `json:"campaign_issue"`
	ReceiptSchema     string             `json:"receipt_schema"`
	PromotionRequires []string           `json:"promotion_requires,omitempty"`
	Model             string             `json:"model,omitempty"`
	WorkloadDigest    string             `json:"workload_digest,omitempty"`
	Rows              []CampaignIndexRow `json:"rows"`
}

// parsedCampaignArm is one loaded arm artifact: the evaluated sweep receipt
// plus the ours-track spec join indexed by concurrency.
type parsedCampaignArm struct {
	report *ServingSweepReport
	spec   map[int]*CampaignSpecMetrics
}

// FoldCampaignIndex folds per-arm fak.serving-sweep.v1 report JSONs into one
// campaign index over the plan's arm matrix. Missing arms fold as missing
// rows; a measured spec point without the additive spec join folds as
// present-without-spec-metrics. Pure and deterministic: row order follows the
// plan, never the input order, and offload placeholders are excluded until
// #14372 defines their receipt fields.
func FoldCampaignIndex(plan *CampaignPlan, reports []CampaignArmReport) (*CampaignIndex, error) {
	if plan == nil {
		return nil, errors.New("nil campaign plan")
	}
	parsed := make(map[string]*parsedCampaignArm, len(reports))
	for _, entry := range reports {
		arm := strings.TrimSpace(entry.Arm)
		if arm == "" {
			return nil, errors.New("campaign arm report has an empty arm name")
		}
		if _, exists := parsed[arm]; exists {
			return nil, fmt.Errorf("duplicate campaign arm report for %q", arm)
		}
		parsedArm, err := parseCampaignArmReport(entry.Report)
		if err != nil {
			return nil, fmt.Errorf("campaign arm %s: %w", arm, err)
		}
		parsed[arm] = parsedArm
	}
	if err := validateCampaignPlanCovers(plan, parsed); err != nil {
		return nil, err
	}
	// Comparability across arms: every receipt must describe the same model
	// and the same immutable workload, or the index would compare unlike
	// envelopes. Engine identity may differ across arms by design (the spec
	// arms run a different engine configuration and receipt).
	var model, workloadDigest string
	for arm, parsedArm := range parsed {
		if model == "" {
			model, workloadDigest = parsedArm.report.Model, parsedArm.report.Workload.Digest
			continue
		}
		if parsedArm.report.Model != model {
			return nil, fmt.Errorf("campaign arm %s model %q differs from %q; campaign arms must share one model identity", arm, parsedArm.report.Model, model)
		}
		if parsedArm.report.Workload.Digest != workloadDigest {
			return nil, fmt.Errorf("campaign arm %s workload digest differs from the other arms; campaign arms must share one immutable workload", arm)
		}
	}
	index := &CampaignIndex{
		Schema:            CampaignIndexSchema,
		CampaignIssue:     plan.CampaignIssue,
		ReceiptSchema:     plan.ReceiptSchema,
		PromotionRequires: append([]string(nil), plan.PromotionRequires...),
		Model:             model,
		WorkloadDigest:    workloadDigest,
	}
	for _, arm := range plan.SpecArms {
		index.Rows = append(index.Rows, foldCampaignArmRow(arm, parsed[arm.Name]))
	}
	return index, nil
}

// validateCampaignPlanCovers refuses reports for arms outside the plan matrix.
func validateCampaignPlanCovers(plan *CampaignPlan, parsed map[string]*parsedCampaignArm) error {
	known := make(map[string]bool, len(plan.SpecArms)+len(plan.OffloadArms))
	for _, arm := range plan.SpecArms {
		known[arm.Name] = true
	}
	for _, arm := range plan.OffloadArms {
		known[arm.Name] = true
	}
	for arm := range parsed {
		if !known[arm] {
			return fmt.Errorf("campaign arm report %q is not in the plan matrix", arm)
		}
	}
	return nil
}

// parseCampaignArmReport loads one arm artifact: base receipt first, then the
// additive spec join, then the shipped sweep evaluator so the fold only
// projects points the comparability gate certified valid.
func parseCampaignArmReport(data []byte) (*parsedCampaignArm, error) {
	if len(bytes.TrimSpace(data)) == 0 {
		return nil, errors.New("empty fak.serving-sweep.v1 report JSON")
	}
	var report ServingSweepReport
	if err := json.Unmarshal(data, &report); err != nil {
		return nil, fmt.Errorf("parse fak.serving-sweep.v1 JSON: %w", err)
	}
	if report.Schema != ServingSweepSchema {
		return nil, fmt.Errorf("serving sweep schema = %q, want %q", report.Schema, ServingSweepSchema)
	}
	if err := EvaluateServingSweep(&report); err != nil {
		return nil, fmt.Errorf("evaluate serving-sweep report: %w", err)
	}
	var join campaignSpecJoin
	if err := json.Unmarshal(data, &join); err != nil {
		return nil, fmt.Errorf("parse additive spec join: %w", err)
	}
	spec := make(map[int]*CampaignSpecMetrics)
	for _, point := range join.Points {
		for _, track := range point.Tracks {
			if track.Track == TrackOurs && track.Spec != nil {
				spec[point.Concurrency] = track.Spec
			}
		}
	}
	return &parsedCampaignArm{report: &report, spec: spec}, nil
}

// campaignOursTrackPoint returns the evaluated-valid ours-track point at the
// given concurrency. The campaign folds fak-native arms, so ours is the folded
// track; a point that exists but failed the sweep evaluator comes back with
// its typed reason instead of its numbers.
func campaignOursTrackPoint(report *ServingSweepReport, concurrency int) (*ServingSweepTrackPoint, string) {
	for i := range report.Points {
		if report.Points[i].Concurrency != concurrency {
			continue
		}
		for j := range report.Points[i].Tracks {
			trackPoint := &report.Points[i].Tracks[j]
			if trackPoint.Track != TrackOurs {
				continue
			}
			if trackPoint.Status == "valid" {
				return trackPoint, ""
			}
			reason := trackPoint.ReasonCode
			if reason == "" {
				reason = trackPoint.Status
			}
			return nil, "ours-track point at this concurrency is " + reason
		}
	}
	return nil, "no ours-track point at this concurrency"
}

// campaignArmIsSpec reports whether an arm configuration expects the additive
// spec join. The none arm is the no-spec baseline and never requires it.
func campaignArmIsSpec(config string) bool {
	return config != "" && config != CampaignSpecConfigNone
}

func foldCampaignArmRow(arm CampaignArm, parsedArm *parsedCampaignArm) CampaignIndexRow {
	row := CampaignIndexRow{
		Arm:               arm.Name,
		SpeculativeConfig: arm.SpeculativeConfig,
		Concurrency:       arm.Concurrency,
		Status:            CampaignRowMissing,
	}
	if parsedArm == nil {
		row.Reason = "no report supplied for this arm"
		return row
	}
	if single, _ := campaignOursTrackPoint(parsedArm.report, 1); single != nil {
		row.SingleStreamTokS = single.Stats.ThroughputTokensS.Value
	}
	point, reason := campaignOursTrackPoint(parsedArm.report, arm.Concurrency)
	if point == nil {
		row.Reason = reason
		return row
	}
	row.AggregateTokS = point.Stats.ThroughputTokensS.Value
	row.TTFTP99Millis = point.Stats.TTFTMillis.P99
	row.ITLP99Millis = point.Stats.ITLMillis.P99
	if spec := parsedArm.spec[arm.Concurrency]; spec != nil {
		row.AcceptanceLength = spec.AcceptanceLength
	}
	if campaignArmIsSpec(arm.SpeculativeConfig) && row.AcceptanceLength == nil {
		row.Status = CampaignRowPresentWithoutSpecMetrics
		row.Reason = "spec arm carries no acceptance_length join (source: fak_engine_spec_accepted_per_round sum/count)"
		return row
	}
	row.Status = CampaignRowMeasured
	return row
}

// JSON renders the index as indented JSON (the machine artifact shape).
func (idx *CampaignIndex) JSON() ([]byte, error) {
	if idx == nil {
		return nil, errors.New("nil campaign index")
	}
	return json.MarshalIndent(idx, "", "  ")
}

// Markdown renders the index as a compact receipt table: one row per plan
// point, "-" for an unfilled value.
func (idx *CampaignIndex) Markdown() string {
	if idx == nil {
		return ""
	}
	var b strings.Builder
	fmt.Fprintf(&b, "# K3.1 bench-campaign index (%s)\n\n", idx.CampaignIssue)
	b.WriteString("| arm | spec | c | single tok/s | aggregate tok/s | ttft p99 ms | itl p99 ms | accept len | status |\n")
	b.WriteString("|---|---|---:|---:|---:|---:|---:|---:|---|\n")
	for _, row := range idx.Rows {
		fmt.Fprintf(&b, "| %s | %s | %d | %s | %s | %s | %s | %s | %s |\n",
			row.Arm, row.SpeculativeConfig, row.Concurrency,
			campaignFormatMetric(row.SingleStreamTokS),
			campaignFormatMetric(row.AggregateTokS),
			campaignFormatMetric(row.TTFTP99Millis),
			campaignFormatMetric(row.ITLP99Millis),
			campaignFormatMetric(row.AcceptanceLength),
			row.Status,
		)
	}
	return b.String()
}

func campaignFormatMetric(value *float64) string {
	if value == nil {
		return "-"
	}
	rounded := math.Round(*value*1000) / 1000
	return strconv.FormatFloat(rounded, 'f', -1, 64)
}
