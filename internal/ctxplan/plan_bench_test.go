package ctxplan

import (
	"context"
	"fmt"
	"testing"
)

var (
	sinkPlan           Plan
	sinkView           View
	sinkSpans          []Span
	sinkCacheVerdict   PlanCacheVerdict
	sinkBudget         Budget
	sinkReconciliation PlanReconciliation
	sinkFault          Fault
	sinkIndex          *Index
	sinkPlanView       PlanView
	sinkString         string
	sinkCands          []Candidate
)

func createBenchmarkStore(n int) (*MemStore, []Span) {
	st := NewMemStore()
	// Pin 0: system prompt
	st.Add("system", DurabilityDurable, []byte("system: you are a helpful autonomous agent kernel assistant"), false)
	// Pin 1: user goal
	st.Add("user", DurabilitySession, []byte("goal: implement auth token rotation across billing services"), false)
	// Durable runbooks and knowledge
	st.Add("WebSearch", DurabilityDurable, []byte("runbook: auth token rotation steps require mint roll and revoke"), false)
	st.Add("WebSearch", DurabilityDurable, []byte("runbook: database schema migration requires backup and read lock"), false)
	st.Add("WebSearch", DurabilityDurable, []byte("runbook: kubernetes pod deployment rolling restart guide"), false)

	for i := 5; i < n-5; i++ {
		role := "Bash"
		if i%3 == 0 {
			role = "Read"
		} else if i%3 == 1 {
			role = "Write"
		}
		st.Add(role, DurabilityTurn, []byte(fmt.Sprintf("log output turn %d: execution completed with status ok", i)), false)
	}

	for i := n - 5; i < n; i++ {
		st.Add("Bash", DurabilitySession, []byte(fmt.Sprintf("recent interaction %d: auth token verification on billing node", i)), false)
	}

	spans, _ := st.Spans(context.Background())
	return st, spans
}

func BenchmarkPlanCellsGreedy(b *testing.B) {
	_, spans := createBenchmarkStore(250)
	f := Forecast{
		Intents: []string{"auth token rotation", "database schema migration"},
		Pins:    []string{"span:0", "span:1"},
		Horizon: 4,
	}
	budget := Budget{Tokens: 128}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		p := PlanCells(spans, f, budget, nil)
		sinkPlan = p
	}
}

func BenchmarkOptimizeExactDP(b *testing.B) {
	_, spans := createBenchmarkStore(50)
	f := Forecast{
		Intents: []string{"auth token rotation"},
		Pins:    []string{"span:0"},
		Horizon: 2,
	}
	cands := Candidates(spans, f, nil)
	budget := Budget{Tokens: 100}
	pins := pinSet(f.Pins)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		p := Optimize(cands, budget, pins, ObjExact)
		sinkPlan = p
	}
}

func BenchmarkOptimizeCoverage(b *testing.B) {
	_, spans := createBenchmarkStore(100)
	f := Forecast{
		Intents: []string{"auth token rotation", "database schema migration", "kubernetes rolling restart"},
		Pins:    []string{"span:0"},
		Horizon: 3,
	}
	cands := Candidates(spans, f, nil)
	budget := Budget{Tokens: 128}
	pins := pinSet(f.Pins)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		p := Optimize(cands, budget, pins, ObjCoverage)
		sinkPlan = p
	}
}

func BenchmarkCandidates(b *testing.B) {
	_, spans := createBenchmarkStore(250)
	f := Forecast{
		Intents: []string{"auth token rotation", "billing node verification"},
		Pins:    []string{"span:0", "span:1"},
		Horizon: 3,
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		cands := Candidates(spans, f, nil)
		sinkCands = cands
	}
}

func BenchmarkBuildIndex(b *testing.B) {
	_, spans := createBenchmarkStore(500)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		ix := BuildIndex(spans)
		sinkIndex = ix
	}
}

func BenchmarkIndexProbe(b *testing.B) {
	_, spans := createBenchmarkStore(1000)
	ix := BuildIndex(spans)
	f := Forecast{
		Intents: []string{"auth token rotation", "billing node verification"},
		Pins:    []string{"span:0", "span:1"},
		Horizon: 4,
	}
	opts := ProbeOptions{
		RecencyWindow: DefaultRecencyWindow,
		MaxCandidates: DefaultMaxCandidates,
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		probed := ix.Probe(f, opts)
		sinkSpans = probed
	}
}

func BenchmarkIndexPlanCells(b *testing.B) {
	_, spans := createBenchmarkStore(1000)
	ix := BuildIndex(spans)
	f := Forecast{
		Intents: []string{"auth token rotation", "billing node verification"},
		Pins:    []string{"span:0", "span:1"},
		Horizon: 4,
	}
	budget := Budget{Tokens: 128}
	opts := ProbeOptions{
		RecencyWindow: DefaultRecencyWindow,
		MaxCandidates: DefaultMaxCandidates,
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		p := ix.PlanCells(f, budget, nil, opts)
		sinkPlan = p
	}
}

func BenchmarkPlanLayout(b *testing.B) {
	_, spans := createBenchmarkStore(500)
	ix := BuildIndex(spans)
	f := Forecast{
		Intents: []string{"auth token rotation", "runbook guide"},
		Pins:    []string{"span:0"},
		Horizon: 3,
	}
	budget := Budget{Tokens: 256}
	layout := DefaultLayout()

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		p := ix.PlanLayout(f, budget, nil, layout)
		sinkPlan = p
	}
}

func BenchmarkPlanCacheLookupHit(b *testing.B) {
	_, spans := createBenchmarkStore(200)
	f := Forecast{
		Intents: []string{"auth token rotation"},
		Pins:    []string{"span:0", "span:1"},
		Horizon: 2,
	}
	budget := Budget{Tokens: 128}
	p := PlanCells(spans, f, budget, nil)
	var cache PlanCache
	cache.Store(spans, f, budget, p)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		v := cache.Lookup(spans, f, budget)
		sinkCacheVerdict = v
	}
}

func BenchmarkPlanWithCache(b *testing.B) {
	_, spans := createBenchmarkStore(200)
	f := Forecast{
		Intents: []string{"auth token rotation"},
		Pins:    []string{"span:0", "span:1"},
		Horizon: 2,
	}
	budget := Budget{Tokens: 128}
	p := PlanCells(spans, f, budget, nil)
	var cache PlanCache
	cache.Store(spans, f, budget, p)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res, hit := cache.PlanWithCache(spans, f, budget, nil)
		if !hit {
			b.Fatal("expected cache hit")
		}
		sinkPlan = res
	}
}

func BenchmarkMaterialize(b *testing.B) {
	ctx := context.Background()
	store, _ := createBenchmarkStore(100)
	f := Forecast{
		Intents: []string{"auth token rotation", "database schema migration"},
		Pins:    []string{"span:0", "span:1"},
		Horizon: 2,
	}
	budget := Budget{Tokens: 128}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		view, err := Materialize(ctx, store, f, budget, nil)
		if err != nil {
			b.Fatalf("Materialize failed: %v", err)
		}
		sinkView = view
	}
}

func BenchmarkDemandPage(b *testing.B) {
	ctx := context.Background()
	store := NewMemStore()
	store.Add("system", DurabilityDurable, []byte("system: autonomous kernel agent"), false)
	store.Add("user", DurabilitySession, []byte("goal: rotate auth credentials"), false)
	store.Add("Read", DurabilityTurn, []byte("ephemeral compiler log data line"), false)
	f := Forecast{Intents: []string{"auth credentials"}, Pins: []string{"span:0", "span:1"}}
	baseView, err := Materialize(ctx, store, f, Budget{Tokens: 13}, nil)
	if err != nil {
		b.Fatalf("setup failed: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		v, fault, err := DemandPage(ctx, store, baseView, "span:2")
		if err != nil || fault.Status != FaultServed {
			b.Fatalf("DemandPage failed: status=%s, err=%v", fault.Status, err)
		}
		sinkView = v
		sinkFault = fault
	}
}

func BenchmarkRecommendBudget(b *testing.B) {
	diff := Difficulty{
		Intents:   4,
		Pins:      2,
		Horizon:   3,
		FaultRate: 0.25,
	}
	bounds := DefaultBudgetBounds()

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := RecommendBudget(diff, bounds)
		sinkBudget = res
	}
}

func BenchmarkReconcilePlan(b *testing.B) {
	stepsBefore := make([]StepPin, 20)
	stepsAfter := make([]StepPin, 20)
	for i := 0; i < 20; i++ {
		id := fmt.Sprintf("step-%d", i)
		text := fmt.Sprintf("execute task step %d in pipeline", i)
		stepsBefore[i] = StepPin{StepID: id, Text: text}
		if i == 15 {
			text += " with additional flag"
		}
		stepsAfter[i] = StepPin{StepID: id, Text: text}
	}
	pinBefore := NewPlanPin(stepsBefore)
	pinAfter := NewPlanPin(stepsAfter)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		rec := ReconcilePlan(pinBefore, pinAfter)
		sinkReconciliation = rec
	}
}

func BenchmarkPlanQuery(b *testing.B) {
	_, spans := createBenchmarkStore(150)
	query := PlanQuery{
		Intents: []string{"auth token rotation", "runbook guide"},
		Pins:    []string{"span:0", "span:1"},
		Horizon: 3,
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pv := query.Plan(spans, nil)
		sinkPlanView = pv
	}
}

func BenchmarkForecastFingerprint(b *testing.B) {
	f := Forecast{
		Intents: []string{"auth token rotation", "database schema migration", "billing node verification"},
		Pins:    []string{"span:0", "span:1", "span:2"},
		Horizon: 4,
		Weights: DefaultWeights(),
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		fp := ForecastFingerprint(f)
		sinkString = fp
	}
}

func BenchmarkStoreVersion(b *testing.B) {
	_, spans := createBenchmarkStore(100)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sv := StoreVersion(spans)
		sinkString = sv
	}
}
