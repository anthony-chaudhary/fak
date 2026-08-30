package webbench

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sweepcert"
)

const ServingSweepSchema = "fak.serving-sweep.v1"

// ServingSweepTrackContract binds every measured point to one declared engine
// and capacity envelope. EngineReceiptDigest is a digest/ID, never a receipt
// path: artifacts must remain portable and must not leak machine-local paths.
type ServingSweepTrackContract struct {
	Track               ServingTrack `json:"track"`
	Model               string       `json:"model"`
	Engine              string       `json:"engine"`
	EngineReceiptDigest string       `json:"engine_receipt_digest"`
	BatchCapacity       int          `json:"batch_capacity"`
	CapacitySource      string       `json:"capacity_source"`
}

type ServingSweepConfig struct {
	GeneratedAt   string
	MachineID     string
	Model         string
	Tracks        []ServingTrackConfig
	Contracts     map[ServingTrack]ServingSweepTrackContract
	Workload      []ServingRequest
	Concurrencies []int
	GoodputSLO    time.Duration
	TTFTP99Budget time.Duration
	ITLP99Budget  time.Duration
	Timeout       time.Duration
	Client        *http.Client
}

type ServingSweepReport struct {
	Schema      string                      `json:"schema"`
	GeneratedAt string                      `json:"generated_at"`
	MachineID   string                      `json:"machine_id"`
	Model       string                      `json:"model"`
	Workload    ServingSweepWorkload        `json:"workload"`
	Contracts   []ServingSweepTrackContract `json:"contracts"`
	SLA         ServingSweepSLA             `json:"sla"`
	Points      []ServingSweepPoint         `json:"points"`
	Tracks      []ServingSweepTrackSummary  `json:"tracks"`
	Honesty     ServingSweepHonestyContract `json:"honesty"`
	Artifact    string                      `json:"artifact,omitempty"`
}

type ServingSweepWorkload struct {
	Digest        string `json:"digest"`
	Requests      int    `json:"requests"`
	Concurrencies []int  `json:"concurrencies"`
}

type ServingSweepSLA struct {
	TTFTP99Millis int64 `json:"ttft_p99_ms,omitempty"`
	ITLP99Millis  int64 `json:"itl_p99_ms,omitempty"`
}

type ServingSweepHonestyContract struct {
	ComparablePointRule string `json:"comparable_point_rule"`
	CapacityRule        string `json:"capacity_rule"`
	PeakRule            string `json:"peak_rule"`
	SLAKneeRule         string `json:"sla_knee_rule"`
	UnknownRule         string `json:"unknown_rule"`
}

type ServingSweepPoint struct {
	Concurrency    int                      `json:"concurrency"`
	WorkloadDigest string                   `json:"workload_digest"`
	Tracks         []ServingSweepTrackPoint `json:"tracks"`
}

type ServingSweepTrackPoint struct {
	Track               ServingTrack `json:"track"`
	Status              string       `json:"status"`
	ReasonCode          string       `json:"reason_code,omitempty"`
	Reason              string       `json:"reason,omitempty"`
	MeasurementStatus   string       `json:"measurement_status,omitempty"`
	Model               string       `json:"model"`
	Engine              string       `json:"engine"`
	EngineReceiptDigest string       `json:"engine_receipt_digest"`
	BatchCapacity       int          `json:"batch_capacity"`
	CapacitySource      string       `json:"capacity_source"`
	Stats               ServingStats `json:"stats"`
}

type ServingSweepTrackSummary struct {
	Track          ServingTrack           `json:"track"`
	Status         string                 `json:"status"`
	Reason         string                 `json:"reason,omitempty"`
	ValidPoints    int                    `json:"valid_points"`
	EnvelopeDigest string                 `json:"envelope_digest,omitempty"`
	PeakStatus     string                 `json:"peak_status,omitempty"`
	PeakReason     string                 `json:"peak_reason,omitempty"`
	Peak           *ServingSweepSelection `json:"peak,omitempty"`
	SLAStatus      string                 `json:"sla_status"`
	SLAReason      string                 `json:"sla_reason,omitempty"`
	SLAKnee        *ServingSweepSelection `json:"sla_knee,omitempty"`
}

type ServingSweepSelection struct {
	Concurrency      int      `json:"concurrency"`
	ThroughputTokens float64  `json:"throughput_tok_s"`
	GoodputRPS       *float64 `json:"goodput_rps,omitempty"`
	TTFTP99Millis    *float64 `json:"ttft_p99_ms,omitempty"`
	ITLP99Millis     *float64 `json:"itl_p99_ms,omitempty"`
}

// ServingWorkloadDigest content-addresses the exact ordered request set. The
// digest includes IDs, messages, token limits, and prompt estimates, so points
// cannot silently change prompt/output shape while remaining comparable.
func ServingWorkloadDigest(workload []ServingRequest) (string, error) {
	if len(workload) == 0 {
		return "", errors.New("serving sweep workload is empty")
	}
	b, err := json.Marshal(workload)
	if err != nil {
		return "", fmt.Errorf("marshal serving sweep workload: %w", err)
	}
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

func normalizeServingConcurrencies(values []int) ([]int, error) {
	seen := make(map[int]bool, len(values))
	out := make([]int, 0, len(values))
	for _, value := range values {
		if value <= 0 {
			return nil, fmt.Errorf("serving sweep concurrency must be positive (got %d)", value)
		}
		if !seen[value] {
			seen[value] = true
			out = append(out, value)
		}
	}
	sort.Ints(out)
	if len(out) < 2 {
		return nil, errors.New("serving sweep requires at least two distinct concurrency points")
	}
	return out, nil
}

// RunServingSweep repeats the existing parity measurement with one immutable
// workload and folds the point reports through the capacity/identity gate.
func RunServingSweep(ctx context.Context, cfg ServingSweepConfig) (*ServingSweepReport, error) {
	concurrencies, err := normalizeServingConcurrencies(cfg.Concurrencies)
	if err != nil {
		return nil, err
	}
	digest, err := ServingWorkloadDigest(cfg.Workload)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(cfg.Model) == "" || strings.EqualFold(strings.TrimSpace(cfg.Model), "unknown") {
		return nil, errors.New("serving sweep requires an exact --model identity")
	}
	if cfg.GeneratedAt == "" {
		cfg.GeneratedAt = time.Now().UTC().Format(time.RFC3339)
	}
	if cfg.MachineID == "" {
		cfg.MachineID = "unknown"
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 60 * time.Second
	}
	if cfg.Client == nil {
		cfg.Client = &http.Client{Timeout: cfg.Timeout + 5*time.Second}
	}
	tracks := append([]ServingTrackConfig(nil), cfg.Tracks...)
	if len(tracks) == 0 {
		for _, track := range AllServingTracks {
			tracks = append(tracks, ServingTrackConfig{Track: track, Model: cfg.Model})
		}
	}

	report := &ServingSweepReport{
		Schema:      ServingSweepSchema,
		GeneratedAt: cfg.GeneratedAt,
		MachineID:   cfg.MachineID,
		Model:       cfg.Model,
		Workload: ServingSweepWorkload{
			Digest:        digest,
			Requests:      len(cfg.Workload),
			Concurrencies: concurrencies,
		},
		SLA: ServingSweepSLA{
			TTFTP99Millis: cfg.TTFTP99Budget.Milliseconds(),
			ITLP99Millis:  cfg.ITLP99Budget.Milliseconds(),
		},
		Honesty: ServingSweepHonestyContract{
			ComparablePointRule: "workload digest, model, engine, engine receipt, capacity, and capacity source must match the declared track contract",
			CapacityRule:        "a point above batch capacity, or with unknown capacity provenance, is invalid and cannot support peak or knee claims",
			PeakRule:            "peak is the maximum measured token throughput across at least two comparable valid points; ties choose lower concurrency",
			SLAKneeRule:         "when p99 budgets are configured, the knee is the maximum-throughput valid point satisfying every configured budget; ties choose lower concurrency",
			UnknownRule:         "failed, sparse, invalid, or missing measurements remain not_measured/invalid and are never converted to zero",
		},
	}

	contracts := make(map[ServingTrack]ServingSweepTrackContract, len(tracks))
	for _, trackCfg := range tracks {
		track, parseErr := ParseServingTrack(string(trackCfg.Track))
		if parseErr != nil {
			return nil, parseErr
		}
		contract := cfg.Contracts[track]
		contract.Track = track
		if contract.Model == "" {
			contract.Model = trackCfg.Model
		}
		if contract.Model == "" {
			contract.Model = cfg.Model
		}
		contracts[track] = contract
		report.Contracts = append(report.Contracts, contract)
	}

	for _, concurrency := range concurrencies {
		point := ServingSweepPoint{Concurrency: concurrency, WorkloadDigest: digest}
		for _, trackCfg := range tracks {
			contract := contracts[trackCfg.Track]
			trackCfg.Model = contract.Model
			measured, measureErr := measureServingSweepTrack(ctx, cfg, trackCfg, concurrency)
			if measureErr != nil {
				return nil, fmt.Errorf("serving sweep concurrency %d track %s: %w", concurrency, trackCfg.Track, measureErr)
			}
			point.Tracks = append(point.Tracks, ServingSweepTrackPoint{
				Track:               measured.Track,
				Status:              measured.Status,
				Reason:              measured.Reason,
				MeasurementStatus:   measured.Status,
				Model:               contract.Model,
				Engine:              contract.Engine,
				EngineReceiptDigest: contract.EngineReceiptDigest,
				BatchCapacity:       contract.BatchCapacity,
				CapacitySource:      contract.CapacitySource,
				Stats:               measured.Stats,
			})
		}
		report.Points = append(report.Points, point)
	}
	if err := EvaluateServingSweep(report); err != nil {
		return nil, err
	}
	return report, nil
}

// measureServingSweepTrack keeps point execution on the shipped SSE/metrics
// seam while making concurrency an explicit sweep coordinate.
func measureServingSweepTrack(ctx context.Context, cfg ServingSweepConfig, trackCfg ServingTrackConfig, concurrency int) (ServingTrackResult, error) {
	track, err := ParseServingTrack(string(trackCfg.Track))
	if err != nil {
		return ServingTrackResult{}, err
	}
	trackCfg.Track = track
	model := trackCfg.Model
	if model == "" {
		model = cfg.Model
	}
	result := ServingTrackResult{
		Track:      track,
		BaseURL:    trackCfg.BaseURL,
		MetricsURL: trackCfg.MetricsURL,
	}
	if strings.TrimSpace(trackCfg.BaseURL) == "" {
		result.Status = "not_measured"
		result.Reason = "no base URL configured for this track"
		result.Stats = emptyServingStats("track was not measured")
		return result, nil
	}
	start := time.Now()
	samples := runSamples(ctx, cfg.Client, trackCfg, model, cfg.Workload, concurrency, cfg.Timeout)
	result.Samples = samples
	result.Status = "measured"
	result.Stats = FoldServingSamples(samples, time.Since(start).Seconds(), cfg.GoodputSLO)
	result.Stats.PrefixCacheHitRate = FetchPrefixCacheHitRate(ctx, cfg.Client, trackCfg.MetricsURL)
	return result, nil
}

// EvaluateServingSweep validates comparability and derives summaries from a
// report's point data. It is exported so deterministic synthetic curves can
// witness the fold without wall-clock timing noise.
func EvaluateServingSweep(report *ServingSweepReport) error {
	if report == nil {
		return errors.New("nil serving sweep report")
	}
	if report.Schema == "" {
		report.Schema = ServingSweepSchema
	}
	if report.Schema != ServingSweepSchema {
		return fmt.Errorf("serving sweep schema = %q, want %q", report.Schema, ServingSweepSchema)
	}
	contracts := make(map[ServingTrack]ServingSweepTrackContract, len(report.Contracts))
	for _, contract := range report.Contracts {
		track, err := ParseServingTrack(string(contract.Track))
		if err != nil {
			return err
		}
		if _, exists := contracts[track]; exists {
			return fmt.Errorf("duplicate serving sweep contract for %s", track)
		}
		contract.Track = track
		contracts[track] = contract
	}
	seenConcurrency := make(map[int]bool, len(report.Points))
	for pointIndex := range report.Points {
		point := &report.Points[pointIndex]
		if point.Concurrency <= 0 {
			return fmt.Errorf("serving sweep point concurrency must be positive (got %d)", point.Concurrency)
		}
		if seenConcurrency[point.Concurrency] {
			return fmt.Errorf("duplicate serving sweep concurrency %d", point.Concurrency)
		}
		seenConcurrency[point.Concurrency] = true
		for trackIndex := range point.Tracks {
			trackPoint := &point.Tracks[trackIndex]
			contract, ok := contracts[trackPoint.Track]
			if !ok {
				invalidateServingSweepPoint(trackPoint, "contract_missing", "track has no declared sweep contract")
				continue
			}
			if point.WorkloadDigest != report.Workload.Digest {
				invalidateServingSweepPoint(trackPoint, "workload_identity_mismatch", "point workload digest differs from the sweep workload")
				continue
			}
			if trackPoint.Model != contract.Model {
				invalidateServingSweepPoint(trackPoint, "model_identity_mismatch", "point model differs from the declared track model")
				continue
			}
			if strings.TrimSpace(contract.Engine) == "" || !validSHA256Digest(contract.EngineReceiptDigest) {
				invalidateServingSweepPoint(trackPoint, "engine_identity_unknown", "engine and a sha256 engine receipt digest are required")
				continue
			}
			if trackPoint.Engine != contract.Engine || trackPoint.EngineReceiptDigest != contract.EngineReceiptDigest {
				invalidateServingSweepPoint(trackPoint, "engine_identity_mismatch", "point engine identity differs from the declared track contract")
				continue
			}
			if contract.BatchCapacity <= 0 || strings.TrimSpace(contract.CapacitySource) == "" {
				invalidateServingSweepPoint(trackPoint, "capacity_unknown", "positive batch capacity and provenance are required")
				continue
			}
			if trackPoint.BatchCapacity != contract.BatchCapacity || trackPoint.CapacitySource != contract.CapacitySource {
				invalidateServingSweepPoint(trackPoint, "capacity_identity_mismatch", "point capacity differs from the declared track contract")
				continue
			}
			if point.Concurrency > contract.BatchCapacity {
				invalidateServingSweepPoint(trackPoint, "capacity_exceeded", fmt.Sprintf("concurrency %d exceeds declared batch capacity %d", point.Concurrency, contract.BatchCapacity))
				continue
			}
			if trackPoint.MeasurementStatus == "" {
				trackPoint.MeasurementStatus = trackPoint.Status
			}
			if trackPoint.MeasurementStatus != "measured" || trackPoint.Stats.OK <= 0 {
				markServingSweepNotMeasured(trackPoint, "measurement_missing", "track produced no successful measured requests")
				continue
			}
			throughput := trackPoint.Stats.ThroughputTokensS
			if throughput.Status != "measured" || throughput.Value == nil || *throughput.Value <= 0 {
				markServingSweepNotMeasured(trackPoint, "throughput_missing", "positive measured token throughput is required")
				continue
			}
			trackPoint.Status = "valid"
			trackPoint.ReasonCode = ""
			trackPoint.Reason = ""
		}
	}

	report.Tracks = report.Tracks[:0]
	for _, contract := range report.Contracts {
		summary := summarizeServingSweepTrack(report, contract.Track)
		report.Tracks = append(report.Tracks, summary)
	}
	return nil
}

func validSHA256Digest(value string) bool {
	const prefix = "sha256:"
	if !strings.HasPrefix(value, prefix) || len(value) != len(prefix)+sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(strings.TrimPrefix(value, prefix))
	return err == nil
}

func invalidateServingSweepPoint(point *ServingSweepTrackPoint, code, reason string) {
	point.Status = "invalid"
	point.ReasonCode = code
	point.Reason = reason
}

func markServingSweepNotMeasured(point *ServingSweepTrackPoint, code, reason string) {
	point.Status = "not_measured"
	point.ReasonCode = code
	point.Reason = reason
}

func summarizeServingSweepTrack(report *ServingSweepReport, track ServingTrack) ServingSweepTrackSummary {
	summary := ServingSweepTrackSummary{Track: track, Status: "not_measured"}
	var firstInvalid string
	var hardInvalid string
	var valid []*ServingSweepTrackPoint
	for pointIndex := range report.Points {
		point := &report.Points[pointIndex]
		for trackIndex := range point.Tracks {
			trackPoint := &point.Tracks[trackIndex]
			if trackPoint.Track != track {
				continue
			}
			if trackPoint.Status == "valid" {
				valid = append(valid, trackPoint)
			} else if firstInvalid == "" && trackPoint.Reason != "" {
				firstInvalid = trackPoint.Reason
			}
			if trackPoint.Status == "invalid" && hardServingSweepInvalidity(trackPoint.ReasonCode) && hardInvalid == "" {
				hardInvalid = trackPoint.Reason
			}
		}
	}
	summary.ValidPoints = len(valid)
	if hardInvalid != "" {
		summary.Status = "invalid"
		summary.Reason = hardInvalid
		summary.PeakStatus = string(sweepcert.FindingInvalid)
		summary.SLAStatus = string(sweepcert.FindingInvalid)
		summary.SLAReason = "identity/capacity invalidity prevents a sweep claim"
		return summary
	}
	evidence, selections, err := servingSweepEvidence(report, track)
	if err != nil {
		summary.Status = "invalid"
		summary.Reason = err.Error()
		summary.PeakStatus = string(sweepcert.FindingInvalid)
		summary.SLAStatus = string(sweepcert.FindingInvalid)
		return summary
	}
	summary.EnvelopeDigest = evidence.EnvelopeDigest
	if len(valid) < 2 {
		summary.Reason = "fewer than two comparable valid points"
		summary.PeakStatus = string(sweepcert.FindingNotIdentifiable)
		if firstInvalid != "" {
			summary.Reason += ": " + firstInvalid
		}
		summary.SLAStatus = servingSweepSLAStatus(report)
		if summary.SLAStatus == "configured" {
			summary.SLAStatus = string(sweepcert.FindingNotIdentifiable)
			summary.SLAReason = summary.Reason
		}
		return summary
	}
	summary.Status = "measured"
	peakFinding := sweepcert.ObservedExtremum(evidence, "throughput_tok_s", sweepcert.Maximum)
	summary.PeakStatus = string(peakFinding.Status)
	summary.PeakReason = peakFinding.Reason
	summary.Peak = selections[peakFinding.PointID]
	if !servingSweepSLAConfigured(report) {
		summary.SLAStatus = "not_configured"
		summary.SLAReason = "no p99 latency budget configured; no SLA knee claimed"
		return summary
	}
	constraints := make([]sweepcert.Constraint, 0, 2)
	if report.SLA.TTFTP99Millis > 0 {
		constraints = append(constraints, sweepcert.Constraint{Metric: "ttft_p99_ms", Operator: sweepcert.AtOrBelow, Threshold: float64(report.SLA.TTFTP99Millis)})
	}
	if report.SLA.ITLP99Millis > 0 {
		constraints = append(constraints, sweepcert.Constraint{Metric: "itl_p99_ms", Operator: sweepcert.AtOrBelow, Threshold: float64(report.SLA.ITLP99Millis)})
	}
	slaFinding := sweepcert.ConstrainedExtremum(evidence, "throughput_tok_s", sweepcert.Maximum, constraints)
	summary.SLAStatus = string(slaFinding.Status)
	summary.SLAReason = slaFinding.Reason
	summary.SLAKnee = selections[slaFinding.PointID]
	if summary.SLAKnee == nil && summary.SLAReason == "" {
		summary.SLAReason = "no comparable valid point satisfies every configured p99 budget"
	}
	return summary
}

func servingSweepEvidence(report *ServingSweepReport, track ServingTrack) (sweepcert.Evidence, map[string]*ServingSweepSelection, error) {
	contract := ServingSweepTrackContract{}
	for _, candidate := range report.Contracts {
		if candidate.Track == track {
			contract = candidate
			break
		}
	}
	coordinates := make([]float64, len(report.Points))
	for i := range report.Points {
		coordinates[i] = float64(report.Points[i].Concurrency)
	}
	sort.Float64s(coordinates)
	axis := sweepcert.Axis{Name: "serving_concurrency", Unit: "requests", Coordinates: coordinates, LowerClosed: len(coordinates) > 0 && coordinates[0] == 1}
	axis.UpperClosed = len(coordinates) > 0 && int(coordinates[len(coordinates)-1]) == contract.BatchCapacity
	envelope := sweepcert.Envelope{Axis: axis, Bindings: []sweepcert.Binding{
		{Name: "model", Value: nonemptySweepBinding(contract.Model)},
		{Name: "workload", Value: nonemptySweepBinding(report.Workload.Digest)},
		{Name: "engine", Value: nonemptySweepBinding(contract.Engine + "/" + contract.EngineReceiptDigest)},
		{Name: "configuration", Value: fmt.Sprintf("track=%s;ttft=%d;itl=%d", track, report.SLA.TTFTP99Millis, report.SLA.ITLP99Millis)},
		{Name: "capacity", Value: strconv.Itoa(contract.BatchCapacity) + "/" + nonemptySweepBinding(contract.CapacitySource)},
		{Name: "reset_order", Value: "ascending-concurrency;declared-track-order;no-reset"},
		{Name: "environment", Value: nonemptySweepBinding(report.MachineID)},
	}}
	digest, err := sweepcert.CanonicalEnvelopeDigest(envelope)
	if err != nil {
		return sweepcert.Evidence{}, nil, err
	}
	evidence := sweepcert.Evidence{
		Envelope: envelope, EnvelopeDigest: digest,
		DeclaredInvalidReasons: []string{"contract_missing", "workload_identity_mismatch", "model_identity_mismatch", "engine_identity_unknown", "engine_identity_mismatch", "capacity_unknown", "capacity_identity_mismatch", "capacity_exceeded"},
	}
	selections := make(map[string]*ServingSweepSelection)
	for _, coordinate := range coordinates {
		id := "concurrency:" + strconv.Itoa(int(coordinate))
		point := sweepcert.Point{ID: id, Coordinate: coordinate, Status: sweepcert.PointNotMeasured, EnvelopeDigest: digest, Observations: make(map[string]sweepcert.Observation)}
		for pointIndex := range report.Points {
			if report.Points[pointIndex].Concurrency != int(coordinate) {
				continue
			}
			for trackIndex := range report.Points[pointIndex].Tracks {
				trackPoint := &report.Points[pointIndex].Tracks[trackIndex]
				if trackPoint.Track != track {
					continue
				}
				switch trackPoint.Status {
				case "valid":
					point.Status = sweepcert.PointMeasured
				case "invalid":
					point.Status, point.InvalidReason = sweepcert.PointInvalid, trackPoint.ReasonCode
				default:
					point.Status = sweepcert.PointNotMeasured
				}
				addServingSweepObservation(point.Observations, "throughput_tok_s", "tok/s", trackPoint.Stats.ThroughputTokensS.Value, digest, track)
				addServingSweepObservation(point.Observations, "ttft_p99_ms", "ms", trackPoint.Stats.TTFTMillis.P99, digest, track)
				addServingSweepObservation(point.Observations, "itl_p99_ms", "ms", trackPoint.Stats.ITLMillis.P99, digest, track)
				if trackPoint.Stats.ThroughputTokensS.Value != nil {
					selections[id] = chooseServingSweepPoint([]*ServingSweepTrackPoint{trackPoint}, []int{int(coordinate)}, nil)
				}
				break
			}
			break
		}
		evidence.Points = append(evidence.Points, point)
	}
	return evidence, selections, nil
}

func addServingSweepObservation(observations map[string]sweepcert.Observation, metric, unit string, value *float64, digest string, track ServingTrack) {
	if value == nil {
		return
	}
	observations[metric] = sweepcert.Observation{Status: sweepcert.ObservationMeasured, Value: value, Provenance: sweepcert.Provenance{Source: string(track), Method: "serving-sse-fold", Unit: unit, EnvelopeDigest: digest}}
}

func nonemptySweepBinding(value string) string {
	if strings.TrimSpace(value) == "" {
		return "<missing>"
	}
	return value
}

func hardServingSweepInvalidity(code string) bool {
	switch code {
	case "contract_missing",
		"workload_identity_mismatch",
		"model_identity_mismatch",
		"engine_identity_unknown",
		"engine_identity_mismatch",
		"capacity_unknown",
		"capacity_identity_mismatch",
		"capacity_exceeded":
		return true
	default:
		return false
	}
}

func servingSweepSLAConfigured(report *ServingSweepReport) bool {
	return report.SLA.TTFTP99Millis > 0 || report.SLA.ITLP99Millis > 0
}

func servingSweepSLAStatus(report *ServingSweepReport) string {
	if servingSweepSLAConfigured(report) {
		return "configured"
	}
	return "not_configured"
}

func chooseServingSweepPoint(points []*ServingSweepTrackPoint, concurrencies []int, include func(*ServingSweepTrackPoint) bool) *ServingSweepSelection {
	best := -1
	for i, point := range points {
		if include != nil && !include(point) {
			continue
		}
		if point.Stats.ThroughputTokensS.Value == nil {
			continue
		}
		if best < 0 || *point.Stats.ThroughputTokensS.Value > *points[best].Stats.ThroughputTokensS.Value ||
			(*point.Stats.ThroughputTokensS.Value == *points[best].Stats.ThroughputTokensS.Value && concurrencies[i] < concurrencies[best]) {
			best = i
		}
	}
	if best < 0 {
		return nil
	}
	point := points[best]
	selection := &ServingSweepSelection{
		Concurrency:      concurrencies[best],
		ThroughputTokens: *point.Stats.ThroughputTokensS.Value,
		TTFTP99Millis:    point.Stats.TTFTMillis.P99,
		ITLP99Millis:     point.Stats.ITLMillis.P99,
	}
	if point.Stats.GoodputRPS.Value != nil {
		selection.GoodputRPS = point.Stats.GoodputRPS.Value
	}
	return selection
}

func DefaultServingSweepArtifactPath(outDir, machineID, generatedAt string, tracks []ServingTrack) string {
	if outDir == "" {
		outDir = filepath.Join("experiments", "benchmark", "runs", "by-machine")
	}
	if machineID == "" {
		machineID = "unknown"
	}
	ts := generatedAt
	if parsed, err := time.Parse(time.RFC3339, generatedAt); err == nil {
		ts = parsed.UTC().Format("20060102T150405Z")
	}
	if ts == "" {
		ts = time.Now().UTC().Format("20060102T150405Z")
	}
	labels := make([]string, 0, len(tracks))
	for _, track := range tracks {
		labels = append(labels, sanitizePathPart(string(track)))
	}
	if len(labels) == 0 {
		labels = []string{"serving"}
	}
	return filepath.Join(outDir, sanitizePathPart(machineID), fmt.Sprintf("%s-serving-sweep-%s", ts, strings.Join(labels, "-")), "result.json")
}

func WriteServingSweepReport(report *ServingSweepReport, path string) error {
	if report == nil {
		return errors.New("nil serving sweep report")
	}
	report.Artifact = path
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0644)
}

func LoadServingSweepReport(path string) (*ServingSweepReport, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var report ServingSweepReport
	if err := json.Unmarshal(data, &report); err != nil {
		return nil, err
	}
	return &report, nil
}
