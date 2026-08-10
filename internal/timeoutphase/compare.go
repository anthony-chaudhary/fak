package timeoutphase

import "time"

type ComparisonArm struct {
	Name      string        `json:"name"`
	Kind      string        `json:"kind"`
	Available bool          `json:"available"`
	Correct   bool          `json:"correct"`
	Latency   time.Duration `json:"latency"`
	Rows      int           `json:"rows"`
	Bytes     int64         `json:"bytes"`
	CostUSD   float64       `json:"cost_usd"`
	Note      string        `json:"note,omitempty"`
}
type ComparisonResult struct {
	Workload string          `json:"workload"`
	Arms     []ComparisonArm `json:"arms"`
}

func CompareLocal() ComparisonResult {
	attempts := []Attempt{{ID: "unknown"}, {ID: "startup", Started: true}, {ID: "edit", Started: true, LastStage: StageEdit}, {ID: "test", Started: true, LastStage: StageTest}, {ID: "commit", Started: true, LastStage: StageCommit}, {ID: "push", Started: true, LastStage: StagePush}}
	start := time.Now()
	report := Record(attempts)
	elapsed := time.Since(start)
	correct := len(report.Rows) == 6 && len(report.PhaseCount) == 6
	return ComparisonResult{Workload: "classify six timeout attempts spanning every closed phase", Arms: []ComparisonArm{
		{Name: "fak native timeout-phase classifier", Kind: "native", Available: true, Correct: correct, Latency: elapsed, Rows: len(report.Rows)},
		{Name: "one undifferentiated timeout bucket", Kind: "baseline", Available: true, Correct: false, Rows: 6, Note: "zero-structure baseline loses the failure phase"},
		{Name: "OpenTelemetry spans", Kind: "external", Note: "requires real instrumented stages and collector trace"},
		{Name: "Datadog APM", Kind: "external", Note: "requires real tracer, agent, and backend trace"},
		{Name: "AWS X-Ray", Kind: "external", Note: "requires real segments and service backend"},
	}}
}
