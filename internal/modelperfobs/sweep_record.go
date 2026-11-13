// Sweep-result record schema: the serializable document a serving-config sweep
// emits, joining the swept server/client arguments to per-config TTFT/ITL/TPOT/
// E2E mean/median/p95/p99 statistics and a best_configurations block.
//
// Adapted from bentoml/llm-optimizer, src/llm_optimizer/bench_client.py
// (BenchmarkMetrics + calculate_metrics at :590/:1140-1210) and
// src/llm_optimizer/cli.py (the enhanced result structure and best_configs
// selection at :142-170/:558-570) at
// 7be87d776e60a978516a2ae53e4a4ff745d04c20 (Apache-2.0). Re-expressed in
// idiomatic Go; no upstream code is copied. Study note:
// docs/research/STUDY-PRODUCTION-SCALING-ECON-2026-11-07.md, candidate
// llm-optimizer-4.
//
// Changed for Fak: the twenty flat *_ttft_ms/*_itl_ms/*_tpot_ms/*
// *_e2e_latency_ms donor fields become one SweepLatencyStats block per metric;
// native-arg dictionaries become a typed SweepArgs map with a
// content-addressed SweepConfig.ID so a row's identity survives map ordering;
// best-configuration ties resolve by config identity instead of row order; the
// constraint check reuses the llm-optimizer-1/2 SLO vocabulary and its full
// six-operator comparison plus unit conversion rather than the donor's two
// operators; and record integrity is a typed sentinel error rather than an
// unvalidated dict. The constrained-selection axis stays the borrowed
// ttft/itl/e2e_latency metric set - TPOT is carried as a recorded statistic,
// not a constraint axis.
package modelperfobs

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
)

// SweepRecordSchema identifies the serialized sweep-result record.
const SweepRecordSchema = "fak-sweep-record/1"

// ErrSweepRecordInvalid reports a sweep row or header that cannot be part of a
// closed record: unmeasured statistics, a non-finite value, a row without a
// config identity, or a total_tests count that disagrees with the row set.
var ErrSweepRecordInvalid = errors.New("modelperfobs: invalid sweep record")

// SweepLatencyStats is the closed four-statistic summary of one latency metric
// over the requests of a single config. Every field is milliseconds, so a
// record needs no per-field unit lookup.
type SweepLatencyStats struct {
	MeanMS   float64 `json:"mean_ms"`
	MedianMS float64 `json:"median_ms"`
	P95MS    float64 `json:"p95_ms"`
	P99MS    float64 `json:"p99_ms"`
}

// SweepMetrics is one config's measured outcome: request/input/output
// throughput plus the four latency metrics the sweep optimizes against.
type SweepMetrics struct {
	Completed            int               `json:"completed"`
	RequestThroughputRPS float64           `json:"request_throughput"`
	InputThroughputTPS   float64           `json:"input_throughput"`
	OutputThroughputTPS  float64           `json:"output_throughput"`
	TTFT                 SweepLatencyStats `json:"ttft"`
	ITL                  SweepLatencyStats `json:"itl"`
	TPOT                 SweepLatencyStats `json:"tpot"`
	E2ELatency           SweepLatencyStats `json:"e2e_latency"`
}

// SweepArgs is one side's native serving arguments. The map holds whatever the
// swept backend accepts (tp_size, max_concurrency, request_rate, ...); this
// schema deliberately does not interpret them.
type SweepArgs map[string]string

// SweepConfig is one swept configuration: the server-side args the harness
// launched with and the client-side args the load generator drove with.
type SweepConfig struct {
	ServerArgs SweepArgs `json:"server_args,omitempty"`
	ClientArgs SweepArgs `json:"client_args,omitempty"`
}

// ID returns a stable content address for the configuration, independent of
// map iteration order. Two rows that swept identical args share an ID, so a
// record's best-configuration choice and its tie-breaks do not depend on the
// order rows were appended.
func (c SweepConfig) ID() string {
	h := sha256.New()
	for _, side := range []struct {
		name string
		args SweepArgs
	}{{"server", c.ServerArgs}, {"client", c.ClientArgs}} {
		keys := make([]string, 0, len(side.args))
		for k := range side.args {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			fmt.Fprintf(h, "%s\x00%s\x00%s\x00", side.name, k, side.args[k])
		}
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))[:16]
}

// SweepTestResult is one row of a sweep: the config that was run, the exact
// launch command when the harness captured one, and what it measured.
type SweepTestResult struct {
	Config  SweepConfig  `json:"config"`
	Cmd     string       `json:"cmd,omitempty"`
	Results SweepMetrics `json:"results"`
}

// SweepMetadata is the record header describing the sweep as a whole.
type SweepMetadata struct {
	GPUType    string `json:"gpu_type,omitempty"`
	GPUCount   int    `json:"gpu_count,omitempty"`
	ModelTag   string `json:"model_tag,omitempty"`
	InputLen   int    `json:"input_len,omitempty"`
	OutputLen  int    `json:"output_len,omitempty"`
	TotalTests int    `json:"total_tests"`
}

// SweepBestConfiguration is one selected best: the winning throughput value,
// the config that produced it, and its launch command.
type SweepBestConfiguration struct {
	ThroughputTPS float64     `json:"throughput"`
	Config        SweepConfig `json:"config"`
	Cmd           string      `json:"cmd,omitempty"`
}

// SweepBestConfigurations is the record's verdict block. The unconstrained
// pair always selects the highest throughput; the constrained pair selects the
// highest throughput among rows that satisfy the sweep's SLO constraints, and
// stays nil when the sweep declared no constraints.
type SweepBestConfigurations struct {
	BestInputThroughput             *SweepBestConfiguration `json:"best_input_throughput,omitempty"`
	BestOutputThroughput            *SweepBestConfiguration `json:"best_output_throughput,omitempty"`
	BestInputThroughputConstrained  *SweepBestConfiguration `json:"best_input_throughput_constrained,omitempty"`
	BestOutputThroughputConstrained *SweepBestConfiguration `json:"best_output_throughput_constrained,omitempty"`
}

// SweepRecord is the whole sweep-result document: the schema stamp, the sweep
// header, the verdict block, and every measured row.
type SweepRecord struct {
	Schema             string                  `json:"schema"`
	Metadata           SweepMetadata           `json:"metadata"`
	BestConfigurations SweepBestConfigurations `json:"best_configurations"`
	TestResults        []SweepTestResult       `json:"test_results"`
}

// SweepLatencyStatsFromSamples summarizes one unordered sample set with the
// four closed statistics. Percentiles use the package's nearest-rank rule so a
// record agrees with the observation summary it was folded from. Empty or
// non-finite samples are refused: an unmeasured metric must not become zeros.
func SweepLatencyStatsFromSamples(samples []float64) (SweepLatencyStats, error) {
	if len(samples) == 0 {
		return SweepLatencyStats{}, fmt.Errorf("%w: no samples for latency stats", ErrSweepRecordInvalid)
	}
	for _, v := range samples {
		if !finite(v) || v < 0 {
			return SweepLatencyStats{}, fmt.Errorf("%w: sample %g must be finite and non-negative", ErrSweepRecordInvalid, v)
		}
	}
	var sum float64
	for _, v := range samples {
		sum += v
	}
	return SweepLatencyStats{
		MeanMS:   sum / float64(len(samples)),
		MedianMS: percentile(samples, .5),
		P95MS:    percentile(samples, .95),
		P99MS:    percentile(samples, .99),
	}, nil
}

// validateSweepLatencyStats refuses a statistics block that carries an
// unmeasured (non-finite or negative) value.
func validateSweepLatencyStats(name string, s SweepLatencyStats) error {
	for _, f := range []struct {
		field string
		v     float64
	}{{"mean_ms", s.MeanMS}, {"median_ms", s.MedianMS}, {"p95_ms", s.P95MS}, {"p99_ms", s.P99MS}} {
		if !finite(f.v) || f.v < 0 {
			return fmt.Errorf("%w: %s.%s must be finite and non-negative", ErrSweepRecordInvalid, name, f.field)
		}
	}
	return nil
}

// validateSweepMetrics refuses a row whose numeric fields are unmeasured.
func validateSweepMetrics(m SweepMetrics) error {
	if m.Completed < 0 {
		return fmt.Errorf("%w: completed must be non-negative", ErrSweepRecordInvalid)
	}
	for _, f := range []struct {
		field string
		v     float64
	}{
		{"request_throughput", m.RequestThroughputRPS},
		{"input_throughput", m.InputThroughputTPS},
		{"output_throughput", m.OutputThroughputTPS},
	} {
		if !finite(f.v) || f.v < 0 {
			return fmt.Errorf("%w: %s must be finite and non-negative", ErrSweepRecordInvalid, f.field)
		}
	}
	for _, metric := range []struct {
		name string
		s    SweepLatencyStats
	}{{"ttft", m.TTFT}, {"itl", m.ITL}, {"tpot", m.TPOT}, {"e2e_latency", m.E2ELatency}} {
		if err := validateSweepLatencyStats(metric.name, metric.s); err != nil {
			return err
		}
	}
	return nil
}

// validateSweepConfig refuses a row without an identity: a config that swept
// no args cannot be told apart from any other, so it must not enter a record.
func validateSweepConfig(c SweepConfig) error {
	if len(c.ServerArgs)+len(c.ClientArgs) == 0 {
		return fmt.Errorf("%w: config must carry at least one server or client arg", ErrSweepRecordInvalid)
	}
	return nil
}

// sweepStatValueMS returns the measured statistic a constraint names, in
// milliseconds. It selects from the row's measured statistics directly (a
// median constraint reads the median, not a scaled mean), which is why a
// constrained sweep needs no tail-inflation factor.
func sweepStatValueMS(m SweepMetrics, c SLOConstraint) (float64, bool) {
	var block SweepLatencyStats
	switch c.Metric {
	case SLOMetricTTFT:
		block = m.TTFT
	case SLOMetricITL:
		block = m.ITL
	case SLOMetricE2ELatency:
		block = m.E2ELatency
	default:
		return 0, false
	}
	switch c.StatType {
	case SLOStatMean:
		return block.MeanMS, true
	case SLOStatMedian:
		return block.MedianMS, true
	case SLOStatP95:
		return block.P95MS, true
	case SLOStatP99:
		return block.P99MS, true
	default:
		return 0, false
	}
}

// sweepRowSatisfiesConstraints reports whether a row holds every constraint.
// Both sides are converted to one unit before the shared operator comparison,
// so a seconds constraint and a milliseconds row compare correctly.
func sweepRowSatisfiesConstraints(m SweepMetrics, constraints []SLOConstraint) (bool, error) {
	for _, c := range constraints {
		if err := validateSLOConstraint(c); err != nil {
			return false, err
		}
		value, ok := sweepStatValueMS(m, c)
		if !ok {
			return false, fmt.Errorf("%w: metric %q is not a sweep constraint axis", ErrSweepRecordInvalid, c.Metric)
		}
		if !sloConstraintHolds(value, sloConstraintValueMS(c), c.Operator) {
			return false, nil
		}
	}
	return true, nil
}

// betterSweepBest reports whether candidate beats incumbent for a best slot:
// higher throughput wins, and an exact tie resolves to the lexicographically
// smaller config ID so the verdict is stable across row order.
func betterSweepBest(candidateThroughput float64, candidate SweepConfig, incumbent *SweepBestConfiguration) bool {
	if incumbent == nil {
		return true
	}
	if candidateThroughput != incumbent.ThroughputTPS {
		return candidateThroughput > incumbent.ThroughputTPS
	}
	return candidate.ID() < incumbent.Config.ID()
}

// BuildSweepRecord validates every row, computes the best_configurations
// block, and returns the closed record. Rows stay in caller order in
// TestResults (the record is also a run log); only the verdict block is
// order-independent. Constrained bests stay nil when constraints is empty.
func BuildSweepRecord(results []SweepTestResult, meta SweepMetadata, constraints []SLOConstraint) (SweepRecord, error) {
	record := SweepRecord{
		Schema:      SweepRecordSchema,
		Metadata:    meta,
		TestResults: make([]SweepTestResult, 0, len(results)),
	}
	if meta.TotalTests == 0 {
		record.Metadata.TotalTests = len(results)
	} else if meta.TotalTests != len(results) {
		return SweepRecord{}, fmt.Errorf("%w: total_tests %d disagrees with %d row(s)", ErrSweepRecordInvalid, meta.TotalTests, len(results))
	}
	for _, c := range constraints {
		if err := validateSLOConstraint(c); err != nil {
			return SweepRecord{}, err
		}
	}

	var bests SweepBestConfigurations
	for _, row := range results {
		if err := validateSweepConfig(row.Config); err != nil {
			return SweepRecord{}, err
		}
		if err := validateSweepMetrics(row.Results); err != nil {
			return SweepRecord{}, err
		}
		record.TestResults = append(record.TestResults, row)

		if betterSweepBest(row.Results.InputThroughputTPS, row.Config, bests.BestInputThroughput) {
			bests.BestInputThroughput = &SweepBestConfiguration{ThroughputTPS: row.Results.InputThroughputTPS, Config: row.Config, Cmd: row.Cmd}
		}
		if betterSweepBest(row.Results.OutputThroughputTPS, row.Config, bests.BestOutputThroughput) {
			bests.BestOutputThroughput = &SweepBestConfiguration{ThroughputTPS: row.Results.OutputThroughputTPS, Config: row.Config, Cmd: row.Cmd}
		}
		if len(constraints) == 0 {
			continue
		}
		feasible, err := sweepRowSatisfiesConstraints(row.Results, constraints)
		if err != nil {
			return SweepRecord{}, err
		}
		if !feasible {
			continue
		}
		if betterSweepBest(row.Results.InputThroughputTPS, row.Config, bests.BestInputThroughputConstrained) {
			bests.BestInputThroughputConstrained = &SweepBestConfiguration{ThroughputTPS: row.Results.InputThroughputTPS, Config: row.Config, Cmd: row.Cmd}
		}
		if betterSweepBest(row.Results.OutputThroughputTPS, row.Config, bests.BestOutputThroughputConstrained) {
			bests.BestOutputThroughputConstrained = &SweepBestConfiguration{ThroughputTPS: row.Results.OutputThroughputTPS, Config: row.Config, Cmd: row.Cmd}
		}
	}
	record.BestConfigurations = bests
	return record, nil
}
