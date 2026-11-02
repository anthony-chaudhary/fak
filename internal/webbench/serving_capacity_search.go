package webbench

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"
)

// ADAPT: Microsoft Vidur CapacitySearch.search, MIT, abae7f63aa857300f5cdc6f5e0d27860cd24721b.
// Attribution and deliberate bounded-search differences: docs/notes/WEBBENCH-CAPACITY-SEARCH.md.
var (
	ErrCapacityInvalid     = errors.New("invalid capacity search")
	ErrCapacityUnknown     = errors.New("capacity measurement unknown")
	ErrCapacityProbeBudget = errors.New("capacity probe budget exhausted")
	ErrCapacityBracket     = errors.New("capacity bracket not established")
)

// CapacitySearch verifies both endpoints before bisecting; best is always a
// witnessed passing rate, never an assumed lower bound. Every call counts.
func CapacitySearch(lo, hi, tol float64, maxProbes int, probe func(float64) (bool, error)) (best float64, probes int, err error) {
	if !capacityFinite(lo) || !capacityFinite(hi) || !capacityFinite(tol) || lo <= 0 || hi <= lo || tol <= 0 || maxProbes < 2 || probe == nil {
		return 0, 0, ErrCapacityInvalid
	}
	call := func(q float64) (bool, error) { probes++; return probe(q) }
	pass, e := call(lo)
	if e != nil {
		return 0, probes, e
	}
	if !pass {
		return 0, probes, ErrCapacityBracket
	}
	best = lo
	pass, e = call(hi)
	if e != nil {
		return best, probes, e
	}
	if pass {
		return hi, probes, nil
	}
	for hi-lo > tol {
		if probes >= maxProbes {
			return best, probes, ErrCapacityProbeBudget
		}
		mid := lo + (hi-lo)/2
		if mid == lo || mid == hi {
			break
		}
		pass, e = call(mid)
		if e != nil {
			return best, probes, e
		}
		if pass {
			lo = mid
			best = mid
		} else {
			hi = mid
		}
	}
	return best, probes, nil
}
func capacityFinite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

type ServingCapacitySearchConfig struct {
	MinQPS       float64       `json:"min_qps"`
	MaxQPS       float64       `json:"max_qps"`
	ToleranceQPS float64       `json:"tolerance_qps"`
	MaxProbes    int           `json:"max_probes"`
	MaxRequests  int           `json:"max_requests"`
	MaxDuration  time.Duration `json:"max_duration_ns"`
	Concurrency  int           `json:"concurrency"`
}
type ServingCapacityObservation struct {
	QPS          float64      `json:"requested_qps"`
	DeliveredQPS float64      `json:"dispatch_qps"`
	Status       string       `json:"status"`
	Stats        ServingStats `json:"stats"`
}
type ServingCapacityResult struct {
	TTFTP99Budget  time.Duration                `json:"ttft_p99_budget_ns"`
	ITLP99Budget   time.Duration                `json:"itl_p99_budget_ns"`
	Status         string                       `json:"status"`
	ReasonCode     string                       `json:"reason_code,omitempty"`
	BestPassingQPS *float64                     `json:"best_passing_qps,omitempty"`
	Probes         int                          `json:"probes"`
	Config         ServingCapacitySearchConfig  `json:"bounds"`
	Contract       ServingSweepTrackContract    `json:"contract"`
	WorkloadDigest string                       `json:"workload_digest"`
	Observations   []ServingCapacityObservation `json:"observations"`
	Qualification  string                       `json:"qualification"`
}

func validateServingCapacity(cfg ServingSweepConfig) error {
	return ValidateServingCapacitySearch(cfg, len(cfg.Workload))
}

// ValidateServingCapacitySearch validates declared bounds before workload expansion or endpoint resolution.
func ValidateServingCapacitySearch(cfg ServingSweepConfig, requests int) error {
	c := cfg.CapacitySearch
	if c == nil {
		return nil
	}
	if len(cfg.Concurrencies) != 0 {
		return ErrCapacityInvalid
	}
	if !capacityFinite(c.MinQPS) || !capacityFinite(c.MaxQPS) || !capacityFinite(c.ToleranceQPS) || c.MinQPS <= 0 || c.MaxQPS <= c.MinQPS || c.ToleranceQPS <= 0 || c.MaxProbes < 2 || c.MaxDuration <= 0 || c.MaxRequests <= 0 || c.Concurrency <= 0 || requests < 2 || requests > c.MaxRequests/c.MaxProbes || len(cfg.Tracks) != 1 {
		return ErrCapacityInvalid
	}
	// A positive nanosecond dispatch interval must be representable.
	if float64(time.Second)/c.MaxQPS < 1 || float64(time.Second)/c.MinQPS >= float64(math.MaxInt64) {
		return ErrCapacityInvalid
	}
	if cfg.TTFTP99Budget < 0 || cfg.ITLP99Budget < 0 || (cfg.TTFTP99Budget == 0 && cfg.ITLP99Budget == 0) {
		return ErrCapacityInvalid
	}
	if strings.TrimSpace(cfg.Model) == "" || strings.EqualFold(strings.TrimSpace(cfg.Model), "unknown") {
		return ErrCapacityInvalid
	}
	t := cfg.Tracks[0]
	contract := cfg.Contracts[t.Track]
	if contract.Model != cfg.Model || t.Model != "" && t.Model != cfg.Model || strings.TrimSpace(contract.Engine) == "" || !validSHA256Digest(contract.EngineReceiptDigest) || contract.BatchCapacity < c.Concurrency || strings.TrimSpace(contract.CapacitySource) == "" {
		return ErrCapacityInvalid
	}
	return nil
}
func servingCapacitySLO(stats ServingStats, cfg ServingSweepConfig) (bool, error) {
	if stats.Failed != 0 || stats.Requests != len(cfg.Workload) || stats.OK != stats.Requests {
		return false, ErrCapacityUnknown
	}
	pass := true
	for _, b := range []struct {
		metric QuantileMetric
		budget time.Duration
	}{{stats.TTFTMillis, cfg.TTFTP99Budget}, {stats.ITLMillis, cfg.ITLP99Budget}} {
		if b.budget == 0 {
			continue
		}
		if b.metric.Status != "measured" || b.metric.P99 == nil || !capacityFinite(*b.metric.P99) || *b.metric.P99 < 0 {
			return false, ErrCapacityUnknown
		}
		if *b.metric.P99 > float64(b.budget)/float64(time.Millisecond) {
			pass = false
		}
	}
	return pass, nil
}
func servingCapacityProbePass(measured ServingTrackResult, qps float64, cfg ServingSweepConfig) (bool, error) {
	if measured.Status != "measured" || !capacityFinite(measured.dispatchQPS) || measured.dispatchQPS < qps*0.95 {
		return false, ErrCapacityUnknown
	}
	return servingCapacitySLO(measured.Stats, cfg)
}

func runServingCapacity(ctx context.Context, cfg ServingSweepConfig, report *ServingSweepReport) (*ServingSweepReport, error) {
	c := *cfg.CapacitySearch
	ctx, cancel := context.WithTimeout(ctx, c.MaxDuration)
	defer cancel()
	result := &ServingCapacityResult{TTFTP99Budget: cfg.TTFTP99Budget, ITLP99Budget: cfg.ITLP99Budget, Config: c, Contract: report.Contracts[0], WorkloadDigest: report.Workload.Digest, Qualification: "bounded_client_probe_not_capacity_certification", Observations: []ServingCapacityObservation{}}
	report.CapacitySearch = result
	best, n, err := CapacitySearch(c.MinQPS, c.MaxQPS, c.ToleranceQPS, c.MaxProbes, func(qps float64) (bool, error) {
		if err := ctx.Err(); err != nil {
			return false, err
		}
		probeCfg := cfg
		probeCfg.requestQPS = qps
		measured, e := measureServingSweepTrack(ctx, probeCfg, cfg.Tracks[0], c.Concurrency)
		obs := ServingCapacityObservation{QPS: qps, DeliveredQPS: measured.dispatchQPS, Stats: measured.Stats, Status: "unknown"}
		pass := false
		if e == nil {
			e = ctx.Err()
		}
		if e == nil {
			pass, e = servingCapacityProbePass(measured, qps, cfg)
		}
		if e == nil {
			if pass {
				obs.Status = "pass"
			} else {
				obs.Status = "fail"
			}
		}
		result.Observations = append(result.Observations, obs)
		return pass, e
	})
	result.Probes = n
	if best > 0 {
		result.BestPassingQPS = &best
	}
	switch {
	case err == nil:
		result.Status = "bounded"
		if best == c.MaxQPS {
			result.ReasonCode = "upper_bound_passed_not_maximum"
		}
	case errors.Is(err, ErrCapacityUnknown):
		result.Status = "unknown"
		result.ReasonCode = "measurement_or_dispatch_incomplete"
	case errors.Is(err, ErrCapacityBracket):
		result.Status = "infeasible"
		result.ReasonCode = "lower_bound_failed"
	case errors.Is(err, ErrCapacityProbeBudget):
		result.Status = "inconclusive"
		result.ReasonCode = "probe_budget"
	case errors.Is(err, context.DeadlineExceeded), errors.Is(err, context.Canceled):
		result.Status = "unknown"
		result.ReasonCode = "cancelled_or_deadline"
	default:
		return report, fmt.Errorf("capacity probe: %w", err)
	}
	return report, nil
}
