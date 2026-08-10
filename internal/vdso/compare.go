package vdso

import (
	"bytes"
	"context"
	"runtime"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

const ComparisonSchema = "fak-tool-result-cache-comparison/1"

type ComparisonArm struct {
	Name              string  `json:"name"`
	Class             string  `json:"class"`
	Available         bool    `json:"available"`
	Calls             int     `json:"calls,omitempty"`
	Correct           int     `json:"correct,omitempty"`
	Correctness       float64 `json:"correctness,omitempty"`
	ElapsedNS         int64   `json:"elapsed_ns,omitempty"`
	ElapsedPerCallNS  float64 `json:"elapsed_per_call_ns,omitempty"`
	UpstreamCalls     int     `json:"upstream_calls,omitempty"`
	PeakRSSBytes      int64   `json:"peak_rss_bytes,omitempty"`
	TotalCostUSD      float64 `json:"total_cost_usd,omitempty"`
	UnavailableReason string  `json:"unavailable_reason,omitempty"`
}

type ComparisonReport struct {
	Schema   string          `json:"schema"`
	Workload string          `json:"workload"`
	GOOS     string          `json:"goos"`
	GOARCH   string          `json:"goarch"`
	Arms     []ComparisonArm `json:"arms"`
	Complete bool            `json:"complete"`
	Verdict  string          `json:"verdict"`
}

func comparisonCall() abi.ToolCall {
	return abi.ToolCall{Tool: "system_version", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)}}
}

// CompareLocal runs a real static-tier vDSO lookup and an optimized uncached
// in-process upstream over the same call/result bytes. External caches stay
// unavailable until real networked processes are measured.
func CompareLocal(iterations int) ComparisonReport {
	if iterations <= 0 {
		iterations = 10000
	}
	want := []byte(`{"version":"1.2.3"}`)
	v := New(16)
	v.RegisterStatic("system_version", want)
	call := comparisonCall()
	ctx := context.Background()
	nativeCorrect := 0
	start := time.Now()
	for i := 0; i < iterations; i++ {
		if r, ok := v.Lookup(ctx, &call); ok && bytes.Equal(r.Payload.Inline, want) {
			nativeCorrect++
		}
	}
	nativeElapsed := time.Since(start)
	baselineCorrect := 0
	start = time.Now()
	for i := 0; i < iterations; i++ {
		got := append([]byte(nil), want...)
		if bytes.Equal(got, want) {
			baselineCorrect++
		}
	}
	baselineElapsed := time.Since(start)
	return ComparisonReport{Schema: ComparisonSchema, Workload: "identical deterministic tool calls and result bytes; live completion fixes upstream service, cache budget, concurrency, warmup, invalidation trace, and correctness oracle", GOOS: runtime.GOOS, GOARCH: runtime.GOARCH,
		Arms: []ComparisonArm{
			localArm("fak native tool vDSO", "native", iterations, nativeCorrect, 0, nativeElapsed),
			localArm("uncached optimized upstream", "tuned_baseline", iterations, baselineCorrect, iterations, baselineElapsed),
			unavailableArm("Redis client-side/server-assisted cache", "next_best", "requires a real Redis process and identical deterministic tool workload including invalidation"),
			unavailableArm("Momento Cache", "next_best", "requires a real Momento service and identical deterministic tool workload including network and total cost"),
		}, Complete: false, Verdict: "local exact-hit overhead only; no net-true cache winner until networked alternatives report output equivalence, hit rate, latency, upstream calls, resources, and total cost"}
}

func localArm(name, class string, calls, correct, upstream int, elapsed time.Duration) ComparisonArm {
	return ComparisonArm{Name: name, Class: class, Available: true, Calls: calls, Correct: correct, Correctness: float64(correct) / float64(calls), ElapsedNS: elapsed.Nanoseconds(), ElapsedPerCallNS: float64(elapsed.Nanoseconds()) / float64(calls), UpstreamCalls: upstream}
}
func unavailableArm(name, class, reason string) ComparisonArm {
	return ComparisonArm{Name: name, Class: class, Available: false, UnavailableReason: reason}
}
