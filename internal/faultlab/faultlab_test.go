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

func TestFaultTypeValidation(t *testing.T) {
	types := []FaultType{
		Truncation,
		CorruptedJSON,
		LatencySpike,
		MemoryPressure,
		NetworkDrop,
		HostReset,
	}

	for _, ft := range types {
		if !ft.IsValid() {
			t.Errorf("expected FaultType %s to be valid", ft)
		}
		if ft.String() == "" {
			t.Errorf("expected non-empty String() for %s", ft)
		}
	}

	invalid := FaultType("cosmic_ray")
	if invalid.IsValid() {
		t.Errorf("expected invalid FaultType to return false for IsValid()")
	}
}

func TestRegisterAndManageRules(t *testing.T) {
	fi := NewFaultInjector()

	// Empty ID error
	err := fi.RegisterRule(FaultRule{Type: Truncation, Target: "*"})
	if !errors.Is(err, ErrInvalidRule) {
		t.Fatalf("expected ErrInvalidRule for empty ID, got %v", err)
	}

	// Invalid FaultType error
	err = fi.RegisterRule(FaultRule{ID: "r1", Type: "unknown", Target: "*"})
	if !errors.Is(err, ErrInvalidRule) {
		t.Fatalf("expected ErrInvalidRule for invalid type, got %v", err)
	}

	// Invalid probability
	err = fi.RegisterRule(FaultRule{ID: "r1", Type: Truncation, Probability: 1.5, Target: "*"})
	if !errors.Is(err, ErrInvalidRule) {
		t.Fatalf("expected ErrInvalidRule for prob > 1.0, got %v", err)
	}

	// Valid rule registration
	rule1 := NewFaultRule("r1", Truncation, "tool:search")
	if err := fi.RegisterRule(rule1); err != nil {
		t.Fatalf("unexpected error registering rule1: %v", err)
	}

	rule2 := NewFaultRule("r2", CorruptedJSON, "stream/*")
	if err := fi.RegisterRule(rule2); err != nil {
		t.Fatalf("unexpected error registering rule2: %v", err)
	}

	rules := fi.GetRules()
	if len(rules) != 2 {
		t.Fatalf("expected 2 rules, got %d", len(rules))
	}
	if rules[0].ID != "r1" || rules[1].ID != "r2" {
		t.Fatalf("expected order [r1, r2], got [%s, %s]", rules[0].ID, rules[1].ID)
	}

	// Disable and enable rule
	if err := fi.DisableRule("r1"); err != nil {
		t.Fatalf("unexpected error disabling r1: %v", err)
	}
	r1, err := fi.GetRule("r1")
	if err != nil {
		t.Fatalf("failed to get r1: %v", err)
	}
	if r1.Active {
		t.Fatalf("expected r1 to be inactive")
	}

	if err := fi.EnableRule("r1"); err != nil {
		t.Fatalf("unexpected error enabling r1: %v", err)
	}
	r1, _ = fi.GetRule("r1")
	if !r1.Active {
		t.Fatalf("expected r1 to be active")
	}

	// Remove rule
	if err := fi.RemoveRule("r1"); err != nil {
		t.Fatalf("unexpected error removing r1: %v", err)
	}
	if _, err := fi.GetRule("r1"); !errors.Is(err, ErrRuleNotFound) {
		t.Fatalf("expected ErrRuleNotFound, got %v", err)
	}
	if len(fi.GetRules()) != 1 {
		t.Fatalf("expected 1 rule remaining, got %d", len(fi.GetRules()))
	}
}

func TestInjectTruncation(t *testing.T) {
	fi := NewFaultInjector()
	ctx := context.Background()

	data := []byte("1234567890abcdefghij") // 20 bytes

	// 1. Default cut in half
	ruleDefault := NewFaultRule("r_half", Truncation, "target_half")
	_ = fi.RegisterRule(ruleDefault)

	res, err := fi.Inject(ctx, "target_half", data)
	if !errors.Is(err, ErrTruncated) {
		t.Fatalf("expected ErrTruncated, got %v", err)
	}
	if len(res) != 10 {
		t.Fatalf("expected 10 bytes truncated, got %d (%s)", len(res), string(res))
	}

	// 2. TruncateBytes
	ruleBytes := NewFaultRule("r_bytes", Truncation, "target_bytes")
	ruleBytes.TruncateBytes = 5
	_ = fi.RegisterRule(ruleBytes)

	res, err = fi.Inject(ctx, "target_bytes", data)
	if !errors.Is(err, ErrTruncated) {
		t.Fatalf("expected ErrTruncated, got %v", err)
	}
	if len(res) != 5 || string(res) != "12345" {
		t.Fatalf("expected '12345', got %q", string(res))
	}

	// 3. TruncateRatio
	ruleRatio := NewFaultRule("r_ratio", Truncation, "target_ratio")
	ruleRatio.TruncateRatio = 0.25 // 25% of 20 = 5 bytes
	_ = fi.RegisterRule(ruleRatio)

	res, err = fi.Inject(ctx, "target_ratio", data)
	if !errors.Is(err, ErrTruncated) {
		t.Fatalf("expected ErrTruncated, got %v", err)
	}
	if len(res) != 5 {
		t.Fatalf("expected 5 bytes, got %d", len(res))
	}

	// 4. CustomPayload override
	ruleCustom := NewFaultRule("r_custom", Truncation, "target_custom")
	ruleCustom.CustomPayload = []byte("short")
	_ = fi.RegisterRule(ruleCustom)

	res, err = fi.Inject(ctx, "target_custom", data)
	if !errors.Is(err, ErrTruncated) {
		t.Fatalf("expected ErrTruncated, got %v", err)
	}
	if string(res) != "short" {
		t.Fatalf("expected 'short', got %q", string(res))
	}
}

func TestTruncateDataEdgeCases(t *testing.T) {
	// Empty data
	if res := TruncateData(nil, 0.5, 0); len(res) != 0 {
		t.Fatalf("expected empty result on nil data")
	}

	// maxBytes > len
	data := []byte("short")
	if res := TruncateData(data, 0, 100); len(res) != 5 {
		t.Fatalf("expected len 5, got %d", len(res))
	}

	// Single byte
	single := []byte("x")
	if res := TruncateData(single, 0, 0); len(res) != 1 {
		t.Fatalf("expected 1 byte, got %d", len(res))
	}
}

func TestInjectCorruptedJSON(t *testing.T) {
	fi := NewFaultInjector()
	ctx := context.Background()

	rule := NewFaultRule("r_json", CorruptedJSON, "api/stream")
	_ = fi.RegisterRule(rule)

	testCases := [][]byte{
		[]byte(`{"model": "qwen", "max_tokens": 100}`),
		[]byte(`[1, 2, 3, {"nested": true}]`),
		[]byte(`"quoted string payload"`),
		[]byte(`12345`),
		[]byte(``),
	}

	for i, tc := range testCases {
		corrupted, err := fi.Inject(ctx, "api/stream", tc)
		if !errors.Is(err, ErrCorruptedJSON) {
			t.Fatalf("case %d: expected ErrCorruptedJSON, got %v", i, err)
		}
		if json.Valid(corrupted) {
			t.Fatalf("case %d: corrupted payload must not be valid JSON: %q", i, string(corrupted))
		}
	}

	// Custom payload
	ruleCustom := NewFaultRule("r_json_custom", CorruptedJSON, "api/custom")
	ruleCustom.CustomPayload = []byte(`{"unclosed": `)
	_ = fi.RegisterRule(ruleCustom)

	corrupted, err := fi.Inject(ctx, "api/custom", []byte(`{"valid": true}`))
	if !errors.Is(err, ErrCorruptedJSON) {
		t.Fatalf("expected ErrCorruptedJSON, got %v", err)
	}
	if string(corrupted) != `{"unclosed": ` {
		t.Fatalf("expected custom payload, got %q", string(corrupted))
	}
}

func TestInjectLatencySpike(t *testing.T) {
	slept := false
	var sleepDuration time.Duration

	fi := NewFaultInjector(
		WithSleep(func(ctx context.Context, d time.Duration) error {
			slept = true
			sleepDuration = d
			return nil
		}),
	)

	rule := NewFaultRule("r_latency", LatencySpike, "slow_endpoint")
	rule.Delay = 250 * time.Millisecond
	_ = fi.RegisterRule(rule)

	data := []byte("normal_response")
	res, err := fi.Inject(context.Background(), "slow_endpoint", data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !slept || sleepDuration != 250*time.Millisecond {
		t.Fatalf("expected sleep of 250ms, got slept=%v, dur=%v", slept, sleepDuration)
	}
	if string(res) != "normal_response" {
		t.Fatalf("expected original data returned, got %q", string(res))
	}

	// Test latency with context cancellation
	fiCanceled := NewFaultInjector(
		WithSleep(func(ctx context.Context, d time.Duration) error {
			return ctx.Err()
		}),
	)
	_ = fiCanceled.RegisterRule(rule)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = fiCanceled.Inject(ctx, "slow_endpoint", data)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestInjectSystemFaults(t *testing.T) {
	fi := NewFaultInjector()
	ctx := context.Background()

	_ = fi.RegisterRule(NewFaultRule("r_oom", MemoryPressure, "mem:*"))
	_ = fi.RegisterRule(NewFaultRule("r_drop", NetworkDrop, "net:*"))
	_ = fi.RegisterRule(NewFaultRule("r_panic", HostReset, "host:*"))

	// Memory pressure
	_, err := fi.Inject(ctx, "mem:alloc", []byte("buffer"))
	if !errors.Is(err, ErrMemoryPressure) {
		t.Fatalf("expected ErrMemoryPressure, got %v", err)
	}

	// Network drop
	_, err = fi.Inject(ctx, "net:socket_42", []byte("stream"))
	if !errors.Is(err, ErrNetworkDrop) {
		t.Fatalf("expected ErrNetworkDrop, got %v", err)
	}

	// Host reset
	_, err = fi.Inject(ctx, "host:vdso_call", []byte("call"))
	if !errors.Is(err, ErrHostReset) {
		t.Fatalf("expected ErrHostReset, got %v", err)
	}

	// Unmatched target
	res, err := fi.Inject(ctx, "safe_target", []byte("safe_payload"))
	if err != nil {
		t.Fatalf("unexpected error on safe target: %v", err)
	}
	if string(res) != "safe_payload" {
		t.Fatalf("expected unmodified payload, got %q", string(res))
	}
}

func TestTargetPatternMatching(t *testing.T) {
	patterns := []struct {
		pattern string
		target  string
		matches bool
	}{
		{"*", "anything", true},
		{"", "anything", true},
		{"exact_name", "exact_name", true},
		{"exact_name", "other_name", false},
		{"tool:*", "tool:search", true},
		{"tool:*", "tool:calc", true},
		{"tool:*", "api:search", false},
		{"agent/v1/*", "agent/v1/stream", true},
		{"agent/v1/*", "agent/v2/stream", false},
	}

	for _, tc := range patterns {
		res := matchTarget(tc.pattern, tc.target)
		if res != tc.matches {
			t.Errorf("matchTarget(%q, %q) = %v; want %v", tc.pattern, tc.target, res, tc.matches)
		}
	}
}

func TestMaxHitsQuota(t *testing.T) {
	fi := NewFaultInjector()
	ctx := context.Background()

	rule := NewFaultRule("r_quota", NetworkDrop, "unstable_api")
	rule.MaxHits = 2
	_ = fi.RegisterRule(rule)

	// Hit 1
	_, err := fi.Inject(ctx, "unstable_api", []byte("msg1"))
	if !errors.Is(err, ErrNetworkDrop) {
		t.Fatalf("hit 1: expected ErrNetworkDrop, got %v", err)
	}

	// Hit 2
	_, err = fi.Inject(ctx, "unstable_api", []byte("msg2"))
	if !errors.Is(err, ErrNetworkDrop) {
		t.Fatalf("hit 2: expected ErrNetworkDrop, got %v", err)
	}

	// Hit 3 (quota exceeded -> bypass fault)
	res, err := fi.Inject(ctx, "unstable_api", []byte("msg3"))
	if err != nil {
		t.Fatalf("hit 3: expected bypass with nil error, got %v", err)
	}
	if string(res) != "msg3" {
		t.Fatalf("hit 3: expected msg3, got %q", string(res))
	}
}

func TestProbabilisticFaults(t *testing.T) {
	// With fixed seed for deterministic outcome
	fi := NewFaultInjector(WithSeed(42))
	ctx := context.Background()

	rule := NewFaultRule("r_prob", NetworkDrop, "flaky_call")
	rule.Probability = 0.5
	_ = fi.RegisterRule(rule)

	dropped := 0
	passed := 0
	trials := 200

	for i := 0; i < trials; i++ {
		_, err := fi.Inject(ctx, "flaky_call", []byte("ping"))
		if errors.Is(err, ErrNetworkDrop) {
			dropped++
		} else {
			passed++
		}
	}

	if dropped == 0 || passed == 0 {
		t.Fatalf("expected both drops and passes with prob=0.5, got dropped=%d, passed=%d", dropped, passed)
	}
}

func TestInterceptReader(t *testing.T) {
	ctx := context.Background()

	// 1. Truncation reader
	t.Run("TruncationReader", func(t *testing.T) {
		fi := NewFaultInjector()
		rule := NewFaultRule("r_trunc_stream", Truncation, "stream:trunc")
		rule.TruncateBytes = 10
		_ = fi.RegisterRule(rule)

		orig := strings.Repeat("a", 100)
		r := strings.NewReader(orig)
		intercepted := fi.InterceptReader(ctx, "stream:trunc", r)

		buf := make([]byte, 50)
		n, err := intercepted.Read(buf)
		if err != nil && !errors.Is(err, io.EOF) {
			t.Fatalf("read error: %v", err)
		}
		if n != 10 {
			t.Fatalf("expected 10 bytes, got %d", n)
		}

		// Next read should return EOF
		n2, err2 := intercepted.Read(buf)
		if !errors.Is(err2, io.EOF) {
			t.Fatalf("expected EOF on truncated stream end, got n=%d, err=%v", n2, err2)
		}
	})

	// 2. CorruptedJSON reader
	t.Run("CorruptedJSONReader", func(t *testing.T) {
		fi := NewFaultInjector()
		rule := NewFaultRule("r_json_stream", CorruptedJSON, "stream:json")
		_ = fi.RegisterRule(rule)

		validJSON := `{"response": "agent output completed", "ok": true}`
		r := strings.NewReader(validJSON)
		intercepted := fi.InterceptReader(ctx, "stream:json", r)

		data, err := io.ReadAll(intercepted)
		if err != nil {
			t.Fatalf("unexpected read error: %v", err)
		}
		if json.Valid(data) {
			t.Fatalf("expected stream payload to be corrupted JSON, got: %q", string(data))
		}
	})

	// 3. NetworkDrop reader
	t.Run("NetworkDropReader", func(t *testing.T) {
		fi := NewFaultInjector()
		rule := NewFaultRule("r_net_stream", NetworkDrop, "stream:net")
		_ = fi.RegisterRule(rule)

		r := strings.NewReader("some valid network traffic")
		intercepted := fi.InterceptReader(ctx, "stream:net", r)

		buf := make([]byte, 10)
		_, err := intercepted.Read(buf)
		if !errors.Is(err, ErrNetworkDrop) {
			t.Fatalf("expected ErrNetworkDrop, got %v", err)
		}
	})

	// 4. HostReset reader
	t.Run("HostResetReader", func(t *testing.T) {
		fi := NewFaultInjector()
		rule := NewFaultRule("r_hr_stream", HostReset, "stream:host")
		_ = fi.RegisterRule(rule)

		r := strings.NewReader("bytes")
		intercepted := fi.InterceptReader(ctx, "stream:host", r)

		buf := make([]byte, 10)
		_, err := intercepted.Read(buf)
		if !errors.Is(err, ErrHostReset) {
			t.Fatalf("expected ErrHostReset, got %v", err)
		}
	})

	// 5. MemoryPressure reader
	t.Run("MemoryPressureReader", func(t *testing.T) {
		fi := NewFaultInjector()
		rule := NewFaultRule("r_mp_stream", MemoryPressure, "stream:mem")
		_ = fi.RegisterRule(rule)

		r := strings.NewReader("bytes")
		intercepted := fi.InterceptReader(ctx, "stream:mem", r)

		buf := make([]byte, 10)
		_, err := intercepted.Read(buf)
		if !errors.Is(err, ErrMemoryPressure) {
			t.Fatalf("expected ErrMemoryPressure, got %v", err)
		}
	})

	// 6. LatencySpike reader
	t.Run("LatencySpikeReader", func(t *testing.T) {
		slept := false
		fi := NewFaultInjector(
			WithSleep(func(ctx context.Context, d time.Duration) error {
				slept = true
				return nil
			}),
		)
		rule := NewFaultRule("r_delay_stream", LatencySpike, "stream:delay")
		rule.Delay = 50 * time.Millisecond
		_ = fi.RegisterRule(rule)

		r := strings.NewReader("delayed payload")
		intercepted := fi.InterceptReader(ctx, "stream:delay", r)

		data, err := io.ReadAll(intercepted)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !slept {
			t.Fatalf("expected sleep to be triggered")
		}
		if string(data) != "delayed payload" {
			t.Fatalf("expected 'delayed payload', got %q", string(data))
		}
	})

	// 7. Passthrough reader
	t.Run("PassthroughReader", func(t *testing.T) {
		fi := NewFaultInjector()
		r := strings.NewReader("unaffected payload")
		intercepted := fi.InterceptReader(ctx, "unmatched_stream", r)

		data, err := io.ReadAll(intercepted)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if string(data) != "unaffected payload" {
			t.Fatalf("expected 'unaffected payload', got %q", string(data))
		}
	})

	// 8. Close delegation
	t.Run("CloseDelegation", func(t *testing.T) {
		fi := NewFaultInjector()
		closed := false
		rc := &dummyReadCloser{
			Reader: strings.NewReader("hello"),
			closeFn: func() error {
				closed = true
				return nil
			},
		}
		intercepted := fi.InterceptReader(ctx, "test", rc)
		if closer, ok := intercepted.(io.Closer); ok {
			_ = closer.Close()
		}
		if !closed {
			t.Fatalf("expected Close() to be forwarded to underlying reader")
		}
	})
}

type dummyReadCloser struct {
	io.Reader
	closeFn func() error
}

func (d *dummyReadCloser) Close() error {
	if d.closeFn != nil {
		return d.closeFn()
	}
	return nil
}

func TestScenarioLifecycle(t *testing.T) {
	fi := NewFaultInjector()
	ctx := context.Background()

	scenario := FaultScenario{
		ID:          "chaos_network",
		Name:        "Intermittent Network Chaos",
		Description: "Simulates packet truncation and socket drops on tool calls",
		Enabled:     false,
		Rules: []FaultRule{
			NewFaultRule("chaos_drop", NetworkDrop, "tool:external_*"),
			NewFaultRule("chaos_trunc", Truncation, "tool:weather"),
		},
	}

	if err := fi.RegisterScenario(scenario); err != nil {
		t.Fatalf("failed to register scenario: %v", err)
	}

	// Because Enabled = false, rules should not fire
	res, err := fi.Inject(ctx, "tool:external_api", []byte("safe"))
	if err != nil || string(res) != "safe" {
		t.Fatalf("expected rule to be inactive initially, got res=%q, err=%v", string(res), err)
	}

	// Enable scenario
	if err := fi.EnableScenario("chaos_network"); err != nil {
		t.Fatalf("failed to enable scenario: %v", err)
	}

	// Now rules should fire
	_, err = fi.Inject(ctx, "tool:external_api", []byte("call"))
	if !errors.Is(err, ErrNetworkDrop) {
		t.Fatalf("expected ErrNetworkDrop after enabling, got %v", err)
	}

	// Disable scenario
	if err := fi.DisableScenario("chaos_network"); err != nil {
		t.Fatalf("failed to disable scenario: %v", err)
	}

	res, err = fi.Inject(ctx, "tool:external_api", []byte("safe_again"))
	if err != nil || string(res) != "safe_again" {
		t.Fatalf("expected bypass after disabling scenario, got %v", err)
	}

	// Scenario not found errors
	if err := fi.EnableScenario("non_existent"); !errors.Is(err, ErrScenarioNotFound) {
		t.Fatalf("expected ErrScenarioNotFound, got %v", err)
	}
	if err := fi.DisableScenario("non_existent"); !errors.Is(err, ErrScenarioNotFound) {
		t.Fatalf("expected ErrScenarioNotFound, got %v", err)
	}
	if _, err := fi.GetScenario("non_existent"); !errors.Is(err, ErrScenarioNotFound) {
		t.Fatalf("expected ErrScenarioNotFound, got %v", err)
	}
}

func TestReportAndMetrics(t *testing.T) {
	fi := NewFaultInjector(WithMaxRecentHits(5))
	ctx := context.Background()

	_ = fi.RegisterRule(NewFaultRule("r1", NetworkDrop, "target1"))
	_ = fi.RegisterRule(NewFaultRule("r2", MemoryPressure, "target2"))

	_, _ = fi.Inject(ctx, "target1", []byte("1"))
	_, _ = fi.Inject(ctx, "target1", []byte("2"))
	_, _ = fi.Inject(ctx, "target2", []byte("3"))

	// Manual record hit
	fi.RecordHit("manual_rule", LatencySpike, "target3", nil)

	report := fi.Report()
	if report.TotalInjections != 4 {
		t.Fatalf("expected 4 total injections, got %d", report.TotalInjections)
	}
	if report.HitsByType[NetworkDrop] != 2 {
		t.Fatalf("expected 2 NetworkDrop hits, got %d", report.HitsByType[NetworkDrop])
	}
	if report.HitsByType[MemoryPressure] != 1 {
		t.Fatalf("expected 1 MemoryPressure hit, got %d", report.HitsByType[MemoryPressure])
	}
	if report.HitsByType[LatencySpike] != 1 {
		t.Fatalf("expected 1 LatencySpike hit, got %d", report.HitsByType[LatencySpike])
	}
	if report.HitsByTarget["target1"] != 2 {
		t.Fatalf("expected 2 hits for target1, got %d", report.HitsByTarget["target1"])
	}
	if report.ActiveRules != 2 {
		t.Fatalf("expected 2 active rules, got %d", report.ActiveRules)
	}

	// Reset metrics
	fi.ResetMetrics()
	cleared := fi.Report()
	if cleared.TotalInjections != 0 {
		t.Fatalf("expected 0 injections after ResetMetrics(), got %d", cleared.TotalInjections)
	}
	if cleared.ActiveRules != 2 {
		t.Fatalf("rules should remain active after ResetMetrics()")
	}

	// Reset completely
	fi.Reset()
	emptyReport := fi.Report()
	if emptyReport.ActiveRules != 0 || len(emptyReport.HitsByType) != 0 {
		t.Fatalf("expected fully empty report after Reset()")
	}
}

func TestConcurrentInjectionAndReader(t *testing.T) {
	fi := NewFaultInjector(WithMaxRecentHits(50))
	ctx := context.Background()

	_ = fi.RegisterRule(NewFaultRule("r_trunc", Truncation, "agent/trunc/*"))
	_ = fi.RegisterRule(NewFaultRule("r_json", CorruptedJSON, "agent/json/*"))
	_ = fi.RegisterRule(NewFaultRule("r_net", NetworkDrop, "agent/net/*"))

	const numGoroutines = 40
	const iterations = 50

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				switch id % 4 {
				case 0:
					data := []byte(`{"turn": 1, "action": "browse", "payload": "abcdefghij"}`)
					_, _ = fi.Inject(ctx, "agent/trunc/step", data)
				case 1:
					data := []byte(`{"turn": 2, "action": "reply", "payload": "valid json"}`)
					_, _ = fi.Inject(ctx, "agent/json/step", data)
				case 2:
					r := bytes.NewReader([]byte(`{"turn": 3, "action": "stream"}`))
					reader := fi.InterceptReader(ctx, "agent/net/step", r)
					buf := make([]byte, 64)
					_, _ = reader.Read(buf)
				case 3:
					_ = fi.Report()
				}
			}
		}(i)
	}

	wg.Wait()

	report := fi.Report()
	if report.TotalInjections == 0 {
		t.Fatalf("expected positive total injections after concurrent runs")
	}
}
