package ultracodebench

import (
	"fmt"
	"sort"
)

const AccessModeFrontierSchema = "fak-ultracode-access-mode-frontier/1"

// OptionalInt64 keeps missing telemetry distinct from a measured zero.
type OptionalInt64 struct {
	Observed bool  `json:"observed"`
	Value    int64 `json:"value,omitempty"`
}

type AccessModeMetrics struct {
	CriticalPathWallMS         OptionalInt64 `json:"critical_path_wall_ms"`
	TotalWorkerMS              OptionalInt64 `json:"total_worker_ms"`
	InputTokens                OptionalInt64 `json:"input_tokens"`
	OutputTokens               OptionalInt64 `json:"output_tokens"`
	CacheReadTokens            OptionalInt64 `json:"cache_read_tokens"`
	LeaseWaitMS                OptionalInt64 `json:"lease_wait_ms"`
	LeaseHoldMS                OptionalInt64 `json:"lease_hold_ms"`
	DeniedCollisions           OptionalInt64 `json:"denied_collisions"`
	StaleFindings              OptionalInt64 `json:"stale_findings"`
	StaleReruns                OptionalInt64 `json:"stale_reruns"`
	AcceptedIndependentEffects OptionalInt64 `json:"accepted_independent_effects"`
	WitnessMS                  OptionalInt64 `json:"witness_ms"`
	ReconcileMS                OptionalInt64 `json:"reconcile_ms"`
	Contradictions             OptionalInt64 `json:"contradictions"`
}

type AccessModeCell struct {
	Mode            string            `json:"mode"`
	Width           int               `json:"width"`
	AcceptedOutcome bool              `json:"accepted_outcome"`
	OutcomeDigest   string            `json:"outcome_digest,omitempty"`
	Metrics         AccessModeMetrics `json:"metrics"`
}

type AccessModeArtifact struct {
	EvidenceKind   string           `json:"evidence_kind"`
	SourceArtifact string           `json:"source_artifact"`
	Cells          []AccessModeCell `json:"cells"`
}

type AccessModeFrontier struct {
	Schema            string               `json:"schema"`
	Scenario          string               `json:"scenario"`
	TaskDigest        string               `json:"task_digest"`
	ModelDigest       string               `json:"model_digest"`
	EnvironmentDigest string               `json:"environment_digest"`
	OperatingEnvelope string               `json:"operating_envelope"`
	Artifacts         []AccessModeArtifact `json:"artifacts"`
}

type AccessModeCellResult struct {
	Mode            string            `json:"mode"`
	Width           int               `json:"width"`
	AcceptedOutcome bool              `json:"accepted_outcome"`
	OutcomeDigest   string            `json:"outcome_digest,omitempty"`
	Metrics         AccessModeMetrics `json:"metrics"`
	Verdict         string            `json:"verdict"`
	Reasons         []string          `json:"reasons,omitempty"`
	WallGainPPM     int64             `json:"wall_gain_ppm,omitempty"`
	BilledTokens    int64             `json:"billed_tokens,omitempty"`
	CoordinationMS  int64             `json:"coordination_ms,omitempty"`
}

type AccessModeArtifactResult struct {
	EvidenceKind   string                 `json:"evidence_kind"`
	SourceArtifact string                 `json:"source_artifact"`
	Cells          []AccessModeCellResult `json:"cells"`
}

type AccessModeFrontierReport struct {
	// EvidenceKind and SourceArtifact retain top-level provenance when decoding the
	// legacy single-artifact frontier report through this compatibility contract.
	EvidenceKind      string                     `json:"evidence_kind,omitempty"`
	SourceArtifact    string                     `json:"source_artifact,omitempty"`
	Schema            string                     `json:"schema"`
	Scenario          string                     `json:"scenario"`
	Widths            []int                      `json:"widths"`
	TaskDigest        string                     `json:"task_digest"`
	ModelDigest       string                     `json:"model_digest"`
	EnvironmentDigest string                     `json:"environment_digest"`
	OperatingEnvelope string                     `json:"operating_envelope"`
	VerdictRule       string                     `json:"verdict_rule"`
	MissingTelemetry  string                     `json:"missing_telemetry"`
	Artifacts         []AccessModeArtifactResult `json:"artifacts"`
	// Compatibility metadata retained for consumers of the earlier context frontier.
	ClaimScope        string      `json:"claim_scope"`
	ExcludedBaselines []string    `json:"excluded_baselines"`
	HillClimb         []HillClimb `json:"hill_climb,omitempty"`
}

func observed(v int64) OptionalInt64 { return OptionalInt64{Observed: true, Value: v} }

func completeAccessModeMetrics(wall, worker, in, out, cache, wait, hold, collisions, stale, reruns, effects, witness, reconcile, contradictions int64) AccessModeMetrics {
	return AccessModeMetrics{
		CriticalPathWallMS: observed(wall), TotalWorkerMS: observed(worker), InputTokens: observed(in), OutputTokens: observed(out), CacheReadTokens: observed(cache),
		LeaseWaitMS: observed(wait), LeaseHoldMS: observed(hold), DeniedCollisions: observed(collisions), StaleFindings: observed(stale), StaleReruns: observed(reruns),
		AcceptedIndependentEffects: observed(effects), WitnessMS: observed(witness), ReconcileMS: observed(reconcile), Contradictions: observed(contradictions),
	}
}

func AccessModeFrontierFixture() AccessModeFrontier {
	const outcome = "sha256:accepted-identical-task-v1"
	widths := []int{1, 2, 4, 8}
	offline := AccessModeArtifact{EvidenceKind: "offline_fixture", SourceArtifact: "internal/ultracodebench.AccessModeFrontierFixture"}
	live := AccessModeArtifact{EvidenceKind: "scrubbed_live_artifact", SourceArtifact: "docs/_witnesses/issue-8337-ultracode-access-frontier/live-input.json"}
	for _, width := range widths {
		offline.Cells = append(offline.Cells,
			AccessModeCell{Mode: "single_agent", Width: width, AcceptedOutcome: true, OutcomeDigest: outcome, Metrics: completeAccessModeMetrics(10000, 10000, 10000, 1000, 0, 0, 0, 0, 0, 0, 3, 700, 300, 0)},
			AccessModeCell{Mode: "scout_plus_writer", Width: width, AcceptedOutcome: true, OutcomeDigest: outcome, Metrics: completeAccessModeMetrics(10500-int64(width)*900, 9000+int64(width)*3500, 9000+int64(width)*2200, 900+int64(width)*250, int64(width)*1200, int64(width)*60, 2500, 0, max64(0, int64(width)-4), max64(0, int64(width)-4), 3, 650, 350+int64(width)*80, 0)},
			AccessModeCell{Mode: "multi_writer", Width: width, AcceptedOutcome: width < 4, OutcomeDigest: map[bool]string{true: outcome, false: "sha256:unequal-conflicting-outcome"}[width < 4], Metrics: completeAccessModeMetrics(10200-int64(width)*550, 9500+int64(width)*4200, 9200+int64(width)*2800, 1000+int64(width)*350, int64(width)*900, int64(width)*300, int64(width)*2600, max64(0, int64(width)-2), max64(0, int64(width)-2), max64(0, int64(width)-2), 3, 700, 500+int64(width)*180, max64(0, int64(width)-2))},
		)
		live.Cells = append(live.Cells,
			AccessModeCell{Mode: "single_agent", Width: width},
			AccessModeCell{Mode: "scout_plus_writer", Width: width},
			AccessModeCell{Mode: "multi_writer", Width: width},
		)
	}
	return AccessModeFrontier{
		Schema: AccessModeFrontierSchema, Scenario: "access-frontier", TaskDigest: "sha256:frozen-access-frontier-task-v1", ModelDigest: "sha256:pinned-model-config-v1", EnvironmentDigest: "sha256:pinned-environment-v1",
		OperatingEnvelope: "same task, model, environment, acceptance witness, and widths 1/2/4/8; offline fixture is illustrative; scrubbed live cells explicitly mark telemetry unobserved and cannot support a gain claim",
		Artifacts:         []AccessModeArtifact{offline, live},
	}
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func EvaluateAccessModeFrontier(input AccessModeFrontier, widths []int) (AccessModeFrontierReport, error) {
	if input.Schema != AccessModeFrontierSchema {
		return AccessModeFrontierReport{}, fmt.Errorf("schema %q, want %q", input.Schema, AccessModeFrontierSchema)
	}
	if input.TaskDigest == "" || input.ModelDigest == "" || input.EnvironmentDigest == "" {
		return AccessModeFrontierReport{}, fmt.Errorf("task, model, and environment digests are required")
	}
	wanted := map[int]bool{}
	for _, w := range widths {
		if w < 1 {
			return AccessModeFrontierReport{}, fmt.Errorf("width must be positive")
		}
		wanted[w] = true
	}
	widths = append([]int(nil), widths...)
	sort.Ints(widths)
	report := AccessModeFrontierReport{Schema: input.Schema, Scenario: input.Scenario, Widths: widths, TaskDigest: input.TaskDigest, ModelDigest: input.ModelDigest, EnvironmentDigest: input.EnvironmentDigest, OperatingEnvelope: input.OperatingEnvelope,
		VerdictRule:      "GAIN requires the identical accepted outcome, complete telemetry, positive critical-path wall gain, coordination time below wall time saved, no contradictions, and at least the baseline accepted effects; NO_GAIN is a complete comparable cell that misses that bar; unequal outcomes or unobserved telemetry ABSTAIN",
		MissingTelemetry: "every metric carries observed=false when unavailable; unavailable values are never coerced to zero", ClaimScope: "same-task agentic context work only", ExcludedBaselines: []string{"single-request throughput", "traditional batching throughput", "provider billed tokens", "provider spend"}}
	for _, artifact := range input.Artifacts {
		baseline, ok := findAccessModeBaseline(artifact.Cells)
		ar := AccessModeArtifactResult{EvidenceKind: artifact.EvidenceKind, SourceArtifact: artifact.SourceArtifact}
		for _, cell := range artifact.Cells {
			if !wanted[cell.Width] {
				continue
			}
			result := evaluateAccessModeCell(cell, baseline, ok)
			ar.Cells = append(ar.Cells, result)
		}
		report.Artifacts = append(report.Artifacts, ar)
	}
	return report, nil
}

func findAccessModeBaseline(cells []AccessModeCell) (AccessModeCell, bool) {
	for _, c := range cells {
		if c.Mode == "single_agent" && c.Width == 1 {
			return c, true
		}
	}
	return AccessModeCell{}, false
}

func evaluateAccessModeCell(cell, baseline AccessModeCell, baselineOK bool) AccessModeCellResult {
	r := AccessModeCellResult{Mode: cell.Mode, Width: cell.Width, AcceptedOutcome: cell.AcceptedOutcome, OutcomeDigest: cell.OutcomeDigest, Metrics: cell.Metrics}
	if !baselineOK || !metricsComplete(baseline.Metrics) || !metricsComplete(cell.Metrics) {
		r.Verdict = "ABSTAIN"
		r.Reasons = []string{"missing_or_unobserved_telemetry"}
		return r
	}
	if !cell.AcceptedOutcome || !baseline.AcceptedOutcome || cell.OutcomeDigest == "" || cell.OutcomeDigest != baseline.OutcomeDigest {
		r.Verdict = "ABSTAIN"
		r.Reasons = []string{"unequal_or_unaccepted_outcome"}
		return r
	}
	r.BilledTokens = cell.Metrics.InputTokens.Value + cell.Metrics.OutputTokens.Value - cell.Metrics.CacheReadTokens.Value
	r.CoordinationMS = cell.Metrics.LeaseWaitMS.Value + cell.Metrics.WitnessMS.Value + cell.Metrics.ReconcileMS.Value
	if baseline.Metrics.CriticalPathWallMS.Value > 0 {
		r.WallGainPPM = (baseline.Metrics.CriticalPathWallMS.Value - cell.Metrics.CriticalPathWallMS.Value) * 1_000_000 / baseline.Metrics.CriticalPathWallMS.Value
	}
	saved := baseline.Metrics.CriticalPathWallMS.Value - cell.Metrics.CriticalPathWallMS.Value
	if cell.Mode != "single_agent" && saved > 0 && r.CoordinationMS < saved && cell.Metrics.Contradictions.Value == 0 && cell.Metrics.AcceptedIndependentEffects.Value >= baseline.Metrics.AcceptedIndependentEffects.Value {
		r.Verdict = "GAIN"
		return r
	}
	r.Verdict = "NO_GAIN"
	r.Reasons = []string{"coordination_adjusted_gain_not_proven"}
	return r
}

func metricsComplete(m AccessModeMetrics) bool {
	values := []OptionalInt64{m.CriticalPathWallMS, m.TotalWorkerMS, m.InputTokens, m.OutputTokens, m.CacheReadTokens, m.LeaseWaitMS, m.LeaseHoldMS, m.DeniedCollisions, m.StaleFindings, m.StaleReruns, m.AcceptedIndependentEffects, m.WitnessMS, m.ReconcileMS, m.Contradictions}
	for _, v := range values {
		if !v.Observed {
			return false
		}
	}
	return true
}
