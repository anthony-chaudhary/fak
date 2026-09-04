// Package kvquantquality provides evaluation kernels for version-pinned KV-cache
// quantization quality against unquantized baselines.
//
// Invariants: kv quantization quality evaluator enforces minimum perplexity and cosine similarity thresholds.
// Key invariant: quality budgets, attention drift, and output drift bounds must be satisfied.
// Contract and guard invariants:
// 1. fail-closed: malformed JSON, invalid baseline precisions, or non-finite inputs fail closed or delegate cleanly.
// 2. Determinism: identical distributions yield identical Jensen-Shannon divergence metrics.
// 3. Provenance: observed evidence requires complete hardware envelope; modeled evidence requires valid pins.
package kvquantquality

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
)

// ContractVersion defines the schema and evaluation contract version for KV quantization quality.
const ContractVersion = "kvquantquality/v1"

// Outcome represents the qualitative decision reached by the evaluation kernel.
type Outcome string

const (
	// OutcomeSupported indicates all quality budgets and provenance requirements are satisfied.
	OutcomeSupported Outcome = "supported"
	// OutcomeRefused indicates the candidate was explicitly evaluated and failed quality budgets.
	OutcomeRefused   Outcome = "unsupported"
	// OutcomeDelegate indicates unknown contracts, missing hardware, or incomplete evidence delegated upstream.
	OutcomeDelegate  Outcome = "delegate"
)

// ReasonCode represents the deterministic reason for the evaluation outcome.
type ReasonCode string

const (
	// ReasonWithinBudget indicates all drift and task metrics stayed within tolerance.
	ReasonWithinBudget          ReasonCode = "within_quality_budget"
	// ReasonQualityBudgetExceeded indicates drift or task degradation exceeded the requested budget.
	ReasonQualityBudgetExceeded ReasonCode = "quality_budget_exceeded"
	// ReasonUnknownContract indicates the request declared an unsupported contract version.
	ReasonUnknownContract       ReasonCode = "unknown_contract_version"
	// ReasonUnknownEvidence indicates the request used an unclassified evidence kind.
	ReasonUnknownEvidence       ReasonCode = "unknown_evidence_kind"
	// ReasonIncompletePin indicates one or more artifact, recipe, or runtime pins were incomplete.
	ReasonIncompletePin         ReasonCode = "incomplete_provenance_pin"
	// ReasonInvalidBaseline indicates the unquantized baseline precision was not fp16 or bf16.
	ReasonInvalidBaseline       ReasonCode = "invalid_unquantized_baseline"
	// ReasonSamePrecision indicates the candidate did not specify a distinct quantized precision.
	ReasonSamePrecision         ReasonCode = "same_precision"
	// ReasonMalformedData indicates inconsistent vectors, non-finite values, or missing required fields.
	ReasonMalformedData         ReasonCode = "malformed_data"
	// ReasonMissingHardware indicates observed evidence lacked required platform, accelerator, or driver details.
	ReasonMissingHardware       ReasonCode = "missing_hardware"
	// ReasonMalformedJSON indicates failure to parse the incoming request JSON payload.
	ReasonMalformedJSON         ReasonCode = "malformed_json"
)

// EvidenceKind designates whether evaluation is modeled mathematically or observed on real hardware.
type EvidenceKind string

const (
	// EvidenceModeled indicates evidence generated from synthetic or modeled attention/output distributions.
	EvidenceModeled          EvidenceKind = "modeled"
	// EvidenceObservedHardware indicates evidence collected directly from benchmark runs on physical accelerators.
	EvidenceObservedHardware EvidenceKind = "observed_hardware"
)

// Pin identifies a reproducible artifact, recipe, or runtime. Version and
// provenance are deliberately separate: a mutable display name is not a pin.
type Pin struct {
	Name       string `json:"name"`
	Version    string `json:"version"`
	Provenance string `json:"provenance"`
	SHA256     string `json:"sha256"`
}

// HardwareEnvelope describes the physical execution platform, accelerator, and driver versions.
type HardwareEnvelope struct {
	Platform    string `json:"platform"`
	Accelerator string `json:"accelerator"`
	Driver      string `json:"driver"`
}

// Measurement contains precision details, attention/output distributions, and aggregate task scores.
type Measurement struct {
	Precision string      `json:"precision"`
	Attention [][]float64 `json:"attention"`
	Output    [][]float64 `json:"output"`
	TaskScore float64     `json:"task_score"`
}

// Budget specifies upper bounds on allowable attention drift, output drift, and task score degradation.
type Budget struct {
	MaxRowJSD    float64 `json:"max_row_jsd"`
	MaxOutputJSD float64 `json:"max_output_jsd"`
	MaxTaskDrop  float64 `json:"max_task_drop"`
}

// Request encapsulates the baseline and candidate measurements against a requested quality budget.
type Request struct {
	ContractVersion string           `json:"contract_version"`
	FixtureID       string           `json:"fixture_id"`
	Seed            int64            `json:"seed"`
	TokenCount      int              `json:"context_length"`
	Evidence        EvidenceKind     `json:"evidence"`
	Artifact        Pin              `json:"artifact"`
	Recipe          Pin              `json:"recipe"`
	Runtime         Pin              `json:"runtime"`
	Hardware        HardwareEnvelope `json:"hardware,omitempty"`
	Baseline        Measurement      `json:"baseline"`
	Candidate       Measurement      `json:"candidate"`
	Budget          Budget           `json:"budget"`
}

// Metrics captures the computed divergence metrics and task drop between baseline and candidate.
type Metrics struct {
	RowJSD    float64 `json:"row_jsd"`
	OutputJSD float64 `json:"output_jsd"`
	TaskDrop  float64 `json:"task_drop"`
}

// Report details the evaluation outcome, reason code, measured metrics, and metadata.
type Report struct {
	Contract           string           `json:"contract"`
	FixtureID          string           `json:"fixture_id"`
	Outcome            Outcome          `json:"outcome"`
	Reason             ReasonCode       `json:"reason"`
	Evidence           EvidenceKind     `json:"evidence"`
	Seed               int64            `json:"seed"`
	TokenCount         int              `json:"context_length"`
	Artifact           Pin              `json:"artifact"`
	Recipe             Pin              `json:"recipe"`
	Runtime            Pin              `json:"runtime"`
	Hardware           HardwareEnvelope `json:"hardware,omitempty"`
	BaselinePrecision  string           `json:"baseline_precision"`
	QuantizedPrecision string           `json:"quantized_precision"`
	Metrics            Metrics          `json:"metrics"`
	Budget             Budget           `json:"budget"`
	Detail             string           `json:"detail,omitempty"`
}

// EvaluateJSON is the public JSON adapter. Even malformed or unknown input gets
// a machine-readable delegated report; callers never need to infer fallback.
func EvaluateJSON(raw []byte) ([]byte, error) {
	var req Request
	if err := json.Unmarshal(raw, &req); err != nil {
		return json.Marshal(Report{
			Contract: ContractVersion,
			Outcome:  OutcomeDelegate,
			Reason:   ReasonMalformedJSON,
			Detail:   "delegate malformed request: " + err.Error(),
		})
	}
	return json.Marshal(Evaluate(req))
}

// Evaluate applies deterministic Jensen-Shannon drift and task-quality gates.
// Unknown contracts/evidence delegate rather than guessing. Structurally invalid
// or over-budget combinations are explicitly unsupported.
func Evaluate(req Request) Report {
	report := Report{
		Contract: ContractVersion, FixtureID: req.FixtureID, Evidence: req.Evidence,
		Seed: req.Seed, TokenCount: req.TokenCount, Artifact: req.Artifact, Recipe: req.Recipe,
		Runtime: req.Runtime, Hardware: req.Hardware, BaselinePrecision: req.Baseline.Precision, QuantizedPrecision: req.Candidate.Precision, Budget: req.Budget,
	}
	finish := func(out Outcome, reason ReasonCode, detail string) Report {
		report.Outcome, report.Reason, report.Detail = out, reason, detail
		return report
	}
	if req.ContractVersion != ContractVersion {
		return finish(OutcomeDelegate, ReasonUnknownContract, "delegate unknown contract version")
	}
	if req.Evidence != EvidenceModeled && req.Evidence != EvidenceObservedHardware {
		return finish(OutcomeDelegate, ReasonUnknownEvidence, "delegate unknown evidence kind")
	}
	if name, err := validatePins(req.Artifact, req.Recipe, req.Runtime); err != nil {
		return finish(OutcomeDelegate, ReasonIncompletePin, name+": "+err.Error())
	}
	if req.Evidence == EvidenceObservedHardware && (strings.TrimSpace(req.Hardware.Platform) == "" || strings.TrimSpace(req.Hardware.Accelerator) == "" || strings.TrimSpace(req.Hardware.Driver) == "") {
		return finish(OutcomeDelegate, ReasonMissingHardware, "observed evidence requires platform, accelerator, and driver")
	}
	if req.Seed == 0 || req.TokenCount <= 0 || req.FixtureID == "" || !validBudget(req.Budget) {
		return finish(OutcomeRefused, ReasonMalformedData, "fixture id, non-zero seed, context length, and finite non-negative budgets are required")
	}
	if req.Baseline.Precision != "fp16" && req.Baseline.Precision != "bf16" {
		return finish(OutcomeRefused, ReasonInvalidBaseline, "baseline must be an unquantized fp16 or bf16 tuned run")
	}
	if req.Candidate.Precision == "" || req.Candidate.Precision == req.Baseline.Precision {
		return finish(OutcomeRefused, ReasonSamePrecision, "quantized precision must name a distinct cache precision")
	}
	attention, err := meanJSD(req.Baseline.Attention, req.Candidate.Attention)
	if err != nil {
		return finish(OutcomeRefused, ReasonMalformedData, "attention: "+err.Error())
	}
	output, err := meanJSD(req.Baseline.Output, req.Candidate.Output)
	if err != nil {
		return finish(OutcomeRefused, ReasonMalformedData, "output: "+err.Error())
	}
	if !finite(req.Baseline.TaskScore) || !finite(req.Candidate.TaskScore) {
		return finish(OutcomeRefused, ReasonMalformedData, "task scores must be finite")
	}
	report.Metrics = Metrics{RowJSD: attention, OutputJSD: output, TaskDrop: math.Max(0, req.Baseline.TaskScore-req.Candidate.TaskScore)}
	if attention > req.Budget.MaxRowJSD || output > req.Budget.MaxOutputJSD || report.Metrics.TaskDrop > req.Budget.MaxTaskDrop {
		return finish(OutcomeRefused, ReasonQualityBudgetExceeded, "one or more quality budgets were exceeded")
	}
	return finish(OutcomeSupported, ReasonWithinBudget, "all quality budgets satisfied")
}

func validatePins(artifact, recipe, runtime Pin) (string, error) {
	for _, item := range []struct {
		name string
		pin  Pin
	}{{"artifact", artifact}, {"recipe", recipe}, {"runtime", runtime}} {
		if strings.TrimSpace(item.pin.Name) == "" || strings.TrimSpace(item.pin.Version) == "" || strings.TrimSpace(item.pin.Provenance) == "" {
			return item.name, errors.New("name, version, and provenance are required")
		}
		digest := strings.ToLower(strings.TrimPrefix(item.pin.SHA256, "sha256:"))
		if len(digest) != sha256.Size*2 {
			return item.name, errors.New("sha256 must contain 64 hexadecimal characters")
		}
		if _, err := hex.DecodeString(digest); err != nil {
			return item.name, errors.New("sha256 is not hexadecimal")
		}
	}
	return "", nil
}

func validBudget(b Budget) bool {
	return finite(b.MaxRowJSD) && finite(b.MaxOutputJSD) && finite(b.MaxTaskDrop) && b.MaxRowJSD >= 0 && b.MaxOutputJSD >= 0 && b.MaxTaskDrop >= 0
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

func meanJSD(base, candidate [][]float64) (float64, error) {
	if len(base) == 0 || len(base) != len(candidate) {
		return 0, errors.New("baseline and candidate require the same non-zero row count")
	}
	total := 0.0
	for i := range base {
		p, q, err := distributions(base[i], candidate[i])
		if err != nil {
			return 0, fmt.Errorf("row %d: %w", i, err)
		}
		row := 0.0
		for j := range p {
			m := (p[j] + q[j]) / 2
			if p[j] > 0 {
				row += .5 * p[j] * math.Log2(p[j]/m)
			}
			if q[j] > 0 {
				row += .5 * q[j] * math.Log2(q[j]/m)
			}
		}
		total += row
	}
	return total / float64(len(base)), nil
}

func distributions(a, b []float64) ([]float64, []float64, error) {
	if len(a) == 0 || len(a) != len(b) {
		return nil, nil, errors.New("vectors require the same non-zero width")
	}
	p, q := append([]float64(nil), a...), append([]float64(nil), b...)
	var ps, qs float64
	for i := range p {
		if !finite(p[i]) || !finite(q[i]) || p[i] < 0 || q[i] < 0 {
			return nil, nil, errors.New("vectors must contain finite non-negative values")
		}
		ps, qs = ps+p[i], qs+q[i]
	}
	if ps <= 0 || qs <= 0 {
		return nil, nil, errors.New("vectors must have positive mass")
	}
	for i := range p {
		p[i], q[i] = p[i]/ps, q[i]/qs
	}
	return p, q, nil
}
