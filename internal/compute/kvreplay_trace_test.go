package compute

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

func TestKVReplayTraceGoldenCorpusScoresCostAwareAgainstOracle(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "kvbm_trace_issue2675_synthetic.json"))
	if err != nil {
		t.Fatalf("read golden trace: %v", err)
	}
	trace, err := ParseKVReplayTrace(data)
	if err != nil {
		t.Fatalf("parse golden trace: %v", err)
	}
	report, err := ReplayKVTrace(trace, KVEvictLRU, KVEvictCostAware)
	if err != nil {
		t.Fatalf("replay golden trace: %v", err)
	}
	if !report.Oracle.Exact {
		t.Fatalf("golden corpus oracle should be exact: %+v", report.Oracle)
	}
	lru := report.Policies[KVEvictLRU]
	cost := report.Policies[KVEvictCostAware]
	if cost.HitTokens < lru.HitTokens {
		t.Fatalf("cost-aware regressed below LRU on golden corpus: LRU=%+v cost=%+v", lru, cost)
	}
	if report.Oracle.HitTokens < cost.HitTokens || report.Oracle.HitTokens < lru.HitTokens {
		t.Fatalf("oracle did not bound policies: oracle=%+v LRU=%+v cost=%+v", report.Oracle, lru, cost)
	}
	if cost.EvictionsPerHit >= lru.EvictionsPerHit {
		t.Fatalf("cost-aware should be no-thrashier than LRU on golden corpus: LRU=%+v cost=%+v", lru, cost)
	}
	if cost.GoodDecisionRatio <= lru.GoodDecisionRatio {
		t.Fatalf("cost-aware should improve good-decision ratio on the golden corpus: LRU=%+v cost=%+v oracle=%+v", lru, cost, report.Oracle)
	}
}

func TestGatewayUsageRowsToKVReplayTraceDerivesPrefixTouches(t *testing.T) {
	rows := []gatewayusageledger.Row{
		{
			Kind:        "exit",
			SessionType: "serve",
			Context:     "stdio",
			SessionID:   "later",
			PID:         20,
			UnixMillis:  2000,
			Counters: gatewayusageledger.Counters{
				InputTokens:          900,
				KVPrefixPromptTokens: 300,
				KVPrefixReusedTokens: 300,
			},
		},
		{
			Kind:        "exit",
			SessionType: "serve",
			Context:     "stdio",
			SessionID:   "earlier",
			PID:         10,
			UnixMillis:  1000,
			Counters: gatewayusageledger.Counters{
				InputTokens:        700,
				CachedPromptTokens: 200,
				CachedTurns:        1,
			},
		},
	}

	trace := GatewayUsageRowsToKVReplayTrace(rows, KVReplayGatewayTraceOptions{
		Name:          "unit-gateway-trace",
		BudgetTokens:  256,
		MaxSpanTokens: 128,
	})
	if err := trace.validate(); err != nil {
		t.Fatalf("gateway trace did not validate: %v", err)
	}
	if len(trace.Events) != 6 {
		t.Fatalf("derived %d events, want 6: %+v", len(trace.Events), trace.Events)
	}
	if trace.Events[0].Session != "earlier" {
		t.Fatalf("rows were not sorted by timestamp: %+v", trace.Events)
	}
	if trace.Events[0].SpanID != trace.Events[1].SpanID {
		t.Fatalf("prefix reuse did not repeat the same span id: %+v", trace.Events[:2])
	}
	if trace.Events[0].Tokens != 128 || trace.Events[2].Tokens != 128 {
		t.Fatalf("token clamping changed: %+v", trace.Events[:3])
	}
	if _, err := ReplayKVTrace(trace, KVEvictLRU, KVEvictCostAware); err != nil {
		t.Fatalf("derived gateway trace should replay: %v", err)
	}
}

func TestGenerateKVReplaySyntheticTraceDeterministic(t *testing.T) {
	opts := KVReplaySyntheticOptions{
		Name:         "unit-synthetic",
		Seed:         2675,
		Events:       16,
		HotSpans:     3,
		SpanTokens:   50,
		BudgetTokens: 100,
	}
	first := GenerateKVReplaySyntheticTrace(opts)
	second := GenerateKVReplaySyntheticTrace(opts)
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("synthetic generator is not deterministic:\nfirst=%+v\nsecond=%+v", first, second)
	}
	report, err := ReplayKVTrace(first, KVEvictLRU, KVEvictCostAware)
	if err != nil {
		t.Fatalf("replay generated trace: %v", err)
	}
	if report.Policies[KVEvictCostAware].HitTokens < report.Policies[KVEvictLRU].HitTokens {
		t.Fatalf("generated corpus regressed cost-aware below LRU: %+v", report)
	}
}
