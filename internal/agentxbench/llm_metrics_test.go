package agentxbench

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestLLMMetricDefinitions(t *testing.T) {
	tr := LLMRequestTrace{
		RequestStartNS:    1_000_000_000,
		TokenTimestampsNS: []int64{1_100_000_000, 1_120_000_000, 1_140_000_000}, // TTFT=100ms, TTST=20ms, ITL=20ms
		OutputTokens:      3,
	}
	m, err := ComputeLLMRequestMetrics(tr)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(m.TTFTMS-100.0) > 1e-9 || math.Abs(m.TTSTMS-20.0) > 1e-9 || math.Abs(m.ITLMS-20.0) > 1e-9 {
		t.Fatalf("unexpected latency metrics: %+v", m)
	}
	// 3 tokens over 0.14s = 21.42857 tok/s
	wantUserTPS := 3.0 / 0.14
	if math.Abs(m.OutputTokPerSecPerUser-wantUserTPS) > 1e-6 {
		t.Fatalf("OutputTokPerSecPerUser = %v, want %v", m.OutputTokPerSecPerUser, wantUserTPS)
	}

	sys, err := ComputeLLMSystemMetrics([]LLMRequestTrace{tr, tr}, 500_000_000) // 6 tokens / 0.5s = 12 tok/s
	if err != nil {
		t.Fatal(err)
	}
	if sys.TotalOutputTokens != 6 || math.Abs(sys.SystemTokPerSec-12.0) > 1e-9 {
		t.Fatalf("unexpected system metrics: %+v", sys)
	}
}
