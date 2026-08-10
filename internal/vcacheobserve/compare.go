package vcacheobserve

import "time"

type ComparisonArm struct {
	Name             string
	Kind             string
	Available        bool
	Correct          bool
	Latency          time.Duration
	Turns            int
	InputTokens      int64
	CacheWriteTokens int64
	CacheReadTokens  int64
	SavedTokenEquiv  float64
	Bytes            int64
	CostUSD          float64
	Note             string
}
type ComparisonResult struct {
	Workload string
	Arms     []ComparisonArm
}

func economicsFixture() []Turn {
	return []Turn{{Family: "a", UnixMillis: 1, InputTokens: 1000}, {Family: "a", UnixMillis: 2, InputTokens: 100, CacheCreation: 900, Ephemeral5m: 900}, {Family: "a", UnixMillis: 3, InputTokens: 100, CacheRead: 900}, {Family: "b", UnixMillis: 4, InputTokens: 500, ContextEvents: 1, ContextShedTokens: 200, ContextBaselineTokens: 700, ContextCostTokens: 500}}
}
func rawTotals(ts []Turn) (in, write, read int64) {
	for _, t := range ts {
		in += t.InputTokens
		write += t.CacheCreation
		read += t.CacheRead
	}
	return
}
func CompareLocal() ComparisonResult {
	turns := economicsFixture()
	start := time.Now()
	rep := Observe(turns, DefaultMultipliers())
	elapsed := time.Since(start)
	in, w, r := rawTotals(turns)
	correct := rep.Turns == 4 && rep.FamilyCount == 2 && rep.Aggregate.CacheCreationTokens == 900 && rep.Aggregate.CacheReadTokens == 900 && rep.Aggregate.SavedTokenEquiv != 0
	start = time.Now()
	_, _, _ = rawTotals(turns)
	bl := time.Since(start)
	return ComparisonResult{Workload: "fold four cold, write, read, and context provider-cache turns across two families", Arms: []ComparisonArm{{Name: "fak native provider-cache economics fold", Kind: "native", Available: true, Correct: correct, Latency: elapsed, Turns: 4, InputTokens: in, CacheWriteTokens: w, CacheReadTokens: r, SavedTokenEquiv: rep.Aggregate.SavedTokenEquiv}, {Name: "raw provider usage without economics fold", Kind: "baseline", Available: true, Correct: false, Latency: bl, Turns: 4, InputTokens: in, CacheWriteTokens: w, CacheReadTokens: r, Note: "preserves provider counters but emits no net-value or family report"}, {Name: "fak + Prometheus", Kind: "integration", Note: "requires real scrape, query, and storage"}, {Name: "fak + OpenTelemetry", Kind: "integration", Note: "requires real export and collector"}, {Name: "Anthropic usage and cost reporting", Kind: "external", Note: "requires real Anthropic usage records"}, {Name: "OpenAI usage and cost reporting", Kind: "external", Note: "requires real OpenAI usage records"}, {Name: "Datadog LLM Observability", Kind: "external", Note: "requires real Datadog ingestion and queries"}, {Name: "LangSmith", Kind: "external", Note: "requires real LangSmith traces and cost analysis"}}}
}
