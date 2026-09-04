package faultlab

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"
	"time"
)

// Invariant: Fault injection lifecycle rules transition deterministically between enabled, disabled, and removed states.
func TestFaultLifecycle_RuleTransitions(t *testing.T) {
	fi := NewFaultInjector()
	ctx := context.Background()

	rule := NewFaultRule("rule_lifecycle", Truncation, "worker/task")
	rule.TruncateBytes = 5
	if err := fi.RegisterRule(rule); err != nil {
		t.Fatalf("unexpected error registering rule: %v", err)
	}

	// Verify initial registered state
	fetched, err := fi.GetRule("rule_lifecycle")
	if err != nil {
		t.Fatalf("failed to retrieve registered rule: %v", err)
	}
	if !fetched.Active {
		t.Fatalf("expected rule to be active initially")
	}

	payload := []byte("hello world 12345")

	// 1. Injected when active
	res, err := fi.Inject(ctx, "worker/task", payload)
	if !errors.Is(err, ErrTruncated) {
		t.Fatalf("expected ErrTruncated when active, got: %v", err)
	}
	if string(res) != "hello" {
		t.Fatalf("expected truncated payload 'hello', got %q", string(res))
	}

	// 2. Disable rule -> clean passthrough (recovery)
	if err := fi.DisableRule("rule_lifecycle"); err != nil {
		t.Fatalf("failed to disable rule: %v", err)
	}
	res, err = fi.Inject(ctx, "worker/task", payload)
	if err != nil {
		t.Fatalf("expected nil error when rule disabled, got: %v", err)
	}
	if string(res) != string(payload) {
		t.Fatalf("expected original payload when rule disabled, got %q", string(res))
	}

	// 3. Re-enable rule -> fault returns
	if err := fi.EnableRule("rule_lifecycle"); err != nil {
		t.Fatalf("failed to re-enable rule: %v", err)
	}
	res, err = fi.Inject(ctx, "worker/task", payload)
	if !errors.Is(err, ErrTruncated) {
		t.Fatalf("expected ErrTruncated after re-enabling, got: %v", err)
	}
	if len(res) != 5 {
		t.Fatalf("expected 5 bytes after re-enabling, got %d", len(res))
	}

	// 4. Remove rule -> clean passthrough permanently
	if err := fi.RemoveRule("rule_lifecycle"); err != nil {
		t.Fatalf("failed to remove rule: %v", err)
	}
	if _, err := fi.GetRule("rule_lifecycle"); !errors.Is(err, ErrRuleNotFound) {
		t.Fatalf("expected ErrRuleNotFound after removal, got: %v", err)
	}
	res, err = fi.Inject(ctx, "worker/task", payload)
	if err != nil {
		t.Fatalf("expected nil error after rule removal, got: %v", err)
	}
	if string(res) != string(payload) {
		t.Fatalf("expected unmodified payload after rule removal")
	}
}

// Invariant: Fault scenarios activate and deactivate multi-rule failure conditions cohesively.
func TestFaultLifecycle_ScenarioLifecycle(t *testing.T) {
	fi := NewFaultInjector()
	ctx := context.Background()

	scenario := FaultScenario{
		ID:          "multi_failure_mode",
		Name:        "Multi-Failure Simulation",
		Description: "Simulates mixed network drop and corrupted JSON",
		Enabled:     false,
		Rules: []FaultRule{
			NewFaultRule("drop_rule", NetworkDrop, "network/*"),
			NewFaultRule("json_rule", CorruptedJSON, "payload/*"),
		},
	}

	if err := fi.RegisterScenario(scenario); err != nil {
		t.Fatalf("unexpected error registering scenario: %v", err)
	}

	// Because enabled was false, rules should not trigger
	res, err := fi.Inject(ctx, "network/socket1", []byte("ping"))
	if err != nil {
		t.Fatalf("expected no injection while scenario is disabled, got %v", err)
	}
	if string(res) != "ping" {
		t.Fatalf("unexpected data change while scenario disabled")
	}

	// Enable scenario
	if err := fi.EnableScenario("multi_failure_mode"); err != nil {
		t.Fatalf("failed to enable scenario: %v", err)
	}

	// network/* should now drop
	_, err = fi.Inject(ctx, "network/socket1", []byte("ping"))
	if !errors.Is(err, ErrNetworkDrop) {
		t.Fatalf("expected ErrNetworkDrop, got: %v", err)
	}

	// payload/* should corrupt json
	jsonIn := []byte(`{"status":"ok","code":200}`)
	cRes, err := fi.Inject(ctx, "payload/tool", jsonIn)
	if !errors.Is(err, ErrCorruptedJSON) {
		t.Fatalf("expected ErrCorruptedJSON, got: %v", err)
	}
	if json.Valid(cRes) {
		t.Fatalf("expected corrupted payload to fail json.Valid")
	}

	// Disable scenario -> both rules deactivate
	if err := fi.DisableScenario("multi_failure_mode"); err != nil {
		t.Fatalf("failed to disable scenario: %v", err)
	}

	_, err = fi.Inject(ctx, "network/socket1", []byte("ping"))
	if err != nil {
		t.Fatalf("expected recovery after scenario disabled, got %v", err)
	}
	cRes, err = fi.Inject(ctx, "payload/tool", jsonIn)
	if err != nil {
		t.Fatalf("expected json recovery after scenario disabled, got %v", err)
	}
	if !json.Valid(cRes) {
		t.Fatalf("expected valid JSON after recovery")
	}
}

// Invariant: Simulates operational failures and verifies that callers can recover gracefully.
func TestFaultSimulation_ErrorRecovery(t *testing.T) {
	fi := NewFaultInjector()
	ctx := context.Background()

	// Simulate a service with retry and fallback mechanics
	t.Run("NetworkDropWithFallbackRecovery", func(t *testing.T) {
		rule := NewFaultRule("drop_flaky", NetworkDrop, "remote/endpoint")
		_ = fi.RegisterRule(rule)

		caller := func(target string) (string, error) {
			data, err := fi.Inject(ctx, target, []byte("request_body"))
			if err != nil {
				return "", err
			}
			return string(data), nil
		}

		// Initial attempt hits fault
		_, err := caller("remote/endpoint")
		if !errors.Is(err, ErrNetworkDrop) {
			t.Fatalf("expected ErrNetworkDrop, got: %v", err)
		}

		// Recovery mechanism: fallback to local secondary endpoint
		recoveredData, err := caller("local/secondary")
		if err != nil {
			t.Fatalf("fallback failed: %v", err)
		}
		if recoveredData != "request_body" {
			t.Fatalf("unexpected fallback data: %q", recoveredData)
		}
	})

	t.Run("CorruptedJSONWithRetryRecovery", func(t *testing.T) {
		rule := NewFaultRule("json_transient", CorruptedJSON, "agent/output")
		rule.MaxHits = 1 // Fault triggers once, then self-exhausts
		_ = fi.RegisterRule(rule)

		type Result struct {
			Status string `json:"status"`
		}

		callAndParse := func() (*Result, error) {
			raw, err := fi.Inject(ctx, "agent/output", []byte(`{"status":"success"}`))
			if err != nil && !errors.Is(err, ErrCorruptedJSON) {
				return nil, err
			}
			var res Result
			if err := json.Unmarshal(raw, &res); err != nil {
				return nil, err
			}
			return &res, nil
		}

		// First call fails JSON parsing due to injected corruption
		res, err := callAndParse()
		if err == nil || res != nil {
			t.Fatalf("expected JSON parse error on first attempt")
		}

		// Second call succeeds because MaxHits quota was exhausted (recovery)
		res, err = callAndParse()
		if err != nil {
			t.Fatalf("expected recovery on second attempt, got: %v", err)
		}
		if res.Status != "success" {
			t.Fatalf("expected status 'success', got %q", res.Status)
		}
	})

	t.Run("MemoryPressureRecovery", func(t *testing.T) {
		rule := NewFaultRule("oom_burst", MemoryPressure, "compute/heavy")
		_ = fi.RegisterRule(rule)

		// Call with simulated OOM
		_, err := fi.Inject(ctx, "compute/heavy", []byte("large_matrix"))
		if !errors.Is(err, ErrMemoryPressure) {
			t.Fatalf("expected ErrMemoryPressure, got: %v", err)
		}

		// Caller sheds load, disables rule, and recovers
		_ = fi.DisableRule("oom_burst")
		out, err := fi.Inject(ctx, "compute/heavy", []byte("large_matrix"))
		if err != nil {
			t.Fatalf("expected recovery after shedding load, got: %v", err)
		}
		if string(out) != "large_matrix" {
			t.Fatalf("expected intact data after recovery")
		}
	})

	t.Run("HostResetRecovery", func(t *testing.T) {
		rule := NewFaultRule("panic_test", HostReset, "kernel/boot")
		_ = fi.RegisterRule(rule)

		_, err := fi.Inject(ctx, "kernel/boot", []byte("boot_config"))
		if !errors.Is(err, ErrHostReset) {
			t.Fatalf("expected ErrHostReset, got: %v", err)
		}

		// Operator resets faultlab state and re-executes
		fi.Reset()
		out, err := fi.Inject(ctx, "kernel/boot", []byte("boot_config"))
		if err != nil {
			t.Fatalf("expected clean execution after reset: %v", err)
		}
		if string(out) != "boot_config" {
			t.Fatalf("expected successful boot config")
		}
	})
}

// Invariant: Latency spikes respect context deadlines and allow timeouts to trigger fail-closed bounds.
func TestFaultSimulation_LatencyTimeoutAndRecovery(t *testing.T) {
	fi := NewFaultInjector()

	rule := NewFaultRule("slow_stream", LatencySpike, "stream/slow")
	rule.Delay = 50 * time.Millisecond
	_ = fi.RegisterRule(rule)

	// Context with tighter deadline than delay
	ctxTimeout, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	_, err := fi.Inject(ctxTimeout, "stream/slow", []byte("data"))
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected context.DeadlineExceeded, got: %v", err)
	}

	// Context with sufficient deadline recovers cleanly
	ctxSuccess, cancelSuccess := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancelSuccess()

	out, err := fi.Inject(ctxSuccess, "stream/slow", []byte("data"))
	if err != nil {
		t.Fatalf("expected success with sufficient deadline, got: %v", err)
	}
	if string(out) != "data" {
		t.Fatalf("expected data intact")
	}
}

// Invariant: Streaming reader transitions cleanly across injection and recovery phases.
func TestFaultSimulation_StreamReaderRecovery(t *testing.T) {
	fi := NewFaultInjector()
	ctx := context.Background()

	rule := NewFaultRule("stream_drop", NetworkDrop, "stream/chat")
	_ = fi.RegisterRule(rule)

	sourceData := "streamed token output chunk 1 and chunk 2"

	// 1. Read fails on fault
	r1 := strings.NewReader(sourceData)
	ir1 := fi.InterceptReader(ctx, "stream/chat", r1)
	buf := make([]byte, 128)
	_, err := ir1.Read(buf)
	if !errors.Is(err, ErrNetworkDrop) {
		t.Fatalf("expected ErrNetworkDrop on stream read, got %v", err)
	}

	// 2. Disable rule and reconnect stream (recovery)
	if err := fi.DisableRule("stream_drop"); err != nil {
		t.Fatalf("failed to disable rule: %v", err)
	}

	r2 := strings.NewReader(sourceData)
	ir2 := fi.InterceptReader(ctx, "stream/chat", r2)
	allBytes, err := io.ReadAll(ir2)
	if err != nil {
		t.Fatalf("expected clean read after disabling fault rule, got: %v", err)
	}
	if string(allBytes) != sourceData {
		t.Fatalf("expected %q, got %q", sourceData, string(allBytes))
	}
}

// Invariant: Quota exhaustion permits automatic self-healing without manual rule deactivation.
func TestFaultLifecycle_MaxHitsSelfHealing(t *testing.T) {
	fi := NewFaultInjector()
	ctx := context.Background()

	rule := NewFaultRule("quota_rule", Truncation, "api/fetch")
	rule.MaxHits = 2
	rule.TruncateBytes = 3
	_ = fi.RegisterRule(rule)

	payload := []byte("ABCDEFG")

	// Hit 1: Truncated
	r1, err1 := fi.Inject(ctx, "api/fetch", payload)
	if !errors.Is(err1, ErrTruncated) || len(r1) != 3 {
		t.Fatalf("expected hit 1 to truncate, got len=%d, err=%v", len(r1), err1)
	}

	// Hit 2: Truncated
	r2, err2 := fi.Inject(ctx, "api/fetch", payload)
	if !errors.Is(err2, ErrTruncated) || len(r2) != 3 {
		t.Fatalf("expected hit 2 to truncate, got len=%d, err=%v", len(r2), err2)
	}

	// Hit 3: Quota reached, passes through untouched (automatic recovery)
	r3, err3 := fi.Inject(ctx, "api/fetch", payload)
	if err3 != nil {
		t.Fatalf("expected hit 3 to pass cleanly, got: %v", err3)
	}
	if string(r3) != string(payload) {
		t.Fatalf("expected hit 3 to be intact, got: %q", string(r3))
	}

	// Verify report totals
	rep := fi.Report()
	if rep.HitsByRule["quota_rule"] != 2 {
		t.Fatalf("expected exactly 2 hits for quota_rule, got %d", rep.HitsByRule["quota_rule"])
	}
}

// Invariant: Reset clears all states while ResetMetrics clears only the counters.
func TestFaultLifecycle_ResetSemantics(t *testing.T) {
	fi := NewFaultInjector()
	ctx := context.Background()

	rule := NewFaultRule("r1", NetworkDrop, "target1")
	_ = fi.RegisterRule(rule)

	_, _ = fi.Inject(ctx, "target1", []byte("hello"))

	rep := fi.Report()
	if rep.TotalInjections != 1 {
		t.Fatalf("expected 1 injection, got %d", rep.TotalInjections)
	}

	// ResetMetrics: rule remains, hit count clears
	fi.ResetMetrics()
	rep = fi.Report()
	if rep.TotalInjections != 0 {
		t.Fatalf("expected 0 injections after ResetMetrics, got %d", rep.TotalInjections)
	}
	if rep.ActiveRules != 1 {
		t.Fatalf("expected 1 active rule to remain after ResetMetrics, got %d", rep.ActiveRules)
	}

	// Reset: rules and metrics cleared
	fi.Reset()
	rep = fi.Report()
	if rep.ActiveRules != 0 {
		t.Fatalf("expected 0 active rules after Reset, got %d", rep.ActiveRules)
	}
	if len(fi.GetRules()) != 0 {
		t.Fatalf("expected empty rules slice after Reset")
	}
}

// Invariant: Concurrent injectors, readers, and rule modifications operate safely without race conditions.
func TestFaultLifecycle_ConcurrentSimulation(t *testing.T) {
	fi := NewFaultInjector()
	ctx := context.Background()

	_ = fi.RegisterRule(NewFaultRule("trunc_conc", Truncation, "conc/trunc"))
	_ = fi.RegisterRule(NewFaultRule("drop_conc", NetworkDrop, "conc/drop"))

	var wg sync.WaitGroup
	workers := 8
	iterations := 50

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				switch i % 3 {
				case 0:
					_, _ = fi.Inject(ctx, "conc/trunc", []byte("payload data to truncate"))
				case 1:
					_, _ = fi.Inject(ctx, "conc/drop", []byte("payload to drop"))
				case 2:
					_, _ = fi.Inject(ctx, "conc/clean", []byte("passthrough data"))
				}

				if workerID == 0 && i%10 == 0 {
					_ = fi.DisableRule("trunc_conc")
					_ = fi.EnableRule("trunc_conc")
				}
			}
		}(w)
	}

	wg.Wait()

	report := fi.Report()
	if report.TotalInjections == 0 {
		t.Fatalf("expected non-zero total injections")
	}
}

// BenchmarkFaultLab benchmarks the primary Inject path under alternating fault and clean workloads.
func BenchmarkFaultLab(b *testing.B) {
	fi := NewFaultInjector()
	ctx := context.Background()

	rule := NewFaultRule("bench_trunc", Truncation, "bench/target")
	rule.TruncateBytes = 16
	_ = fi.RegisterRule(rule)

	data := bytes.Repeat([]byte("A"), 128)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		if i%2 == 0 {
			_, _ = fi.Inject(ctx, "bench/target", data)
		} else {
			_, _ = fi.Inject(ctx, "bench/passthrough", data)
		}
	}
}

// BenchmarkFaultLab_Lifecycle benchmarks dynamic rule registration, enablement, and removal cycles.
func BenchmarkFaultLab_Lifecycle(b *testing.B) {
	fi := NewFaultInjector()
	rule := NewFaultRule("bench_cycle", Truncation, "cycle/target")

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = fi.RegisterRule(rule)
		_ = fi.DisableRule("bench_cycle")
		_ = fi.EnableRule("bench_cycle")
		_ = fi.RemoveRule("bench_cycle")
	}
}

// BenchmarkFaultLab_SimulationPassThrough benchmarks the zero-cost bypass path when no rules match.
func BenchmarkFaultLab_SimulationPassThrough(b *testing.B) {
	fi := NewFaultInjector()
	ctx := context.Background()
	data := []byte("clean payload without active rules")

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = fi.Inject(ctx, "unmatched/target", data)
	}
}
