package cachevalueledger

import "time"

type ComparisonArm struct {
	Name         string
	Kind         string
	Available    bool
	Correct      bool
	Latency      time.Duration
	Rows         int
	Alerts       int
	PromptTokens uint64
	Bytes        int64
	CostUSD      float64
	Note         string
}
type ComparisonResult struct {
	Workload string
	Arms     []ComparisonArm
}

func trendFixture() []Row {
	return []Row{
		{UnixMillis: 1, Turns: 4, PromptTokens: 400, ReusedTokens: 320},
		{UnixMillis: 2, Turns: 4, PromptTokens: 400, ReusedTokens: 320},
		{UnixMillis: 3, Turns: 1, PromptTokens: 100, ReusedTokens: 0},
		{UnixMillis: 4, Turns: 4, PromptTokens: 400, ReusedTokens: 160},
		{UnixMillis: 5, Turns: 4, PromptTokens: 400, ReusedTokens: 120},
	}
}
func representedPromptTokens(rows []Row) uint64 {
	var n uint64
	for _, r := range rows {
		n += r.PromptTokens
	}
	return n
}

// CompareLocal executes fak and raw-ledger inspection. Monitoring adapters are
// not witnesses for real ingestion, storage, query evaluation, or alerting.
func CompareLocal() ComparisonResult {
	rows := trendFixture()
	start := time.Now()
	report := FoldTrendGate(rows)
	elapsed := time.Since(start)
	correct := !report.OK && report.Verdict == "REGRESSED" && report.BaselineTurns == 8 && report.RecentTurns == 8 && report.BaselineReuseRatio == .8 && report.RecentReuseRatio == .35
	start = time.Now()
	tokens := representedPromptTokens(rows)
	baselineElapsed := time.Since(start)
	return ComparisonResult{Workload: "detect one trailing-window reuse regression across five chronological ledger rows including one ignored single-turn row", Arms: []ComparisonArm{
		{Name: "fak native trailing-window trend gate", Kind: "native", Available: true, Correct: correct, Latency: elapsed, Rows: len(rows), Alerts: 1, PromptTokens: tokens},
		{Name: "raw JSONL ledger without trend gate", Kind: "baseline", Available: true, Correct: false, Latency: baselineElapsed, Rows: len(rows), PromptTokens: tokens, Note: "tuned incumbent preserves every row but emits no regression verdict"},
		{Name: "fak + Prometheus", Kind: "integration", Note: "requires the real first-class metrics scrape, query, and alert path"},
		{Name: "fak + OpenTelemetry", Kind: "integration", Note: "requires the real first-class telemetry export and collector path"},
		{Name: "Prometheus recording and alerting rules", Kind: "external", Note: "requires a real Prometheus server and equivalent trailing-window rule"},
		{Name: "Datadog change and anomaly monitor", Kind: "external", Note: "requires a real Datadog monitor and equivalent query"},
	}}
}
