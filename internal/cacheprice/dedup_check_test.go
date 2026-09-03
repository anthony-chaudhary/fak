package cacheprice

import (
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestDedupNetSavings(t *testing.T) {
	cases := []struct {
		name            string
		hit             bool
		overhead        time.Duration
		avoided         time.Duration
		wantNetDuration time.Duration
	}{
		{
			name:            "hit with positive savings",
			hit:             true,
			overhead:        2 * time.Millisecond,
			avoided:         10 * time.Millisecond,
			wantNetDuration: 8 * time.Millisecond,
		},
		{
			name:            "hit at break-even",
			hit:             true,
			overhead:        5 * time.Millisecond,
			avoided:         5 * time.Millisecond,
			wantNetDuration: 0,
		},
		{
			name:            "hit with negative net savings (check slower than transfer)",
			hit:             true,
			overhead:        8 * time.Millisecond,
			avoided:         3 * time.Millisecond,
			wantNetDuration: -5 * time.Millisecond,
		},
		{
			name:            "miss pays check overhead without avoided transfer",
			hit:             false,
			overhead:        3 * time.Millisecond,
			avoided:         15 * time.Millisecond,
			wantNetDuration: -3 * time.Millisecond,
		},
		{
			name:            "negative inputs clamp defensively to zero",
			hit:             true,
			overhead:        -2 * time.Millisecond,
			avoided:         -5 * time.Millisecond,
			wantNetDuration: 0,
		},
		{
			name:            "miss with negative overhead clamps to zero",
			hit:             false,
			overhead:        -4 * time.Millisecond,
			avoided:         10 * time.Millisecond,
			wantNetDuration: 0,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := DedupNetSavings(tc.hit, tc.overhead, tc.avoided)
			if got != tc.wantNetDuration {
				t.Fatalf("DedupNetSavings(%v, %v, %v) = %v, want %v",
					tc.hit, tc.overhead, tc.avoided, got, tc.wantNetDuration)
			}
		})
	}
}

func TestDedupCheckWorthwhile(t *testing.T) {
	cases := []struct {
		name     string
		overhead time.Duration
		avoided  time.Duration
		minRatio float64
		want     bool
	}{
		{
			name:     "favorable 5x ratio",
			overhead: 2 * time.Millisecond,
			avoided:  10 * time.Millisecond,
			minRatio: 1.0,
			want:     true,
		},
		{
			name:     "exact break-even 1.0",
			overhead: 5 * time.Millisecond,
			avoided:  5 * time.Millisecond,
			minRatio: 1.0,
			want:     true,
		},
		{
			name:     "unfavorable below 1.0",
			overhead: 6 * time.Millisecond,
			avoided:  5 * time.Millisecond,
			minRatio: 1.0,
			want:     false,
		},
		{
			name:     "clears custom 2.0 ratio",
			overhead: 2 * time.Millisecond,
			avoided:  5 * time.Millisecond,
			minRatio: 2.0,
			want:     true,
		},
		{
			name:     "fails custom 2.0 ratio",
			overhead: 3 * time.Millisecond,
			avoided:  5 * time.Millisecond,
			minRatio: 2.0,
			want:     false,
		},
		{
			name:     "non-positive minRatio defaults to 1.0",
			overhead: 4 * time.Millisecond,
			avoided:  4 * time.Millisecond,
			minRatio: 0,
			want:     true,
		},
		{
			name:     "zero overhead is not worthwhile",
			overhead: 0,
			avoided:  5 * time.Millisecond,
			minRatio: 1.0,
			want:     false,
		},
		{
			name:     "zero avoided transfer is not worthwhile",
			overhead: 2 * time.Millisecond,
			avoided:  0,
			minRatio: 1.0,
			want:     false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := DedupCheckWorthwhile(tc.overhead, tc.avoided, tc.minRatio)
			if got != tc.want {
				t.Fatalf("DedupCheckWorthwhile(%v, %v, %f) = %v, want %v",
					tc.overhead, tc.avoided, tc.minRatio, got, tc.want)
			}
		})
	}
}

func TestTransferDuration(t *testing.T) {
	// 10 MB at 100 MB/s = 0.1s = 100ms
	const tenMB = 10 * 1024 * 1024
	const hundredMBPerSec = 100 * 1024 * 1024

	dur := TransferDuration(tenMB, hundredMBPerSec)
	if dur != 100*time.Millisecond {
		t.Fatalf("TransferDuration() = %v, want 100ms", dur)
	}

	if TransferDuration(0, hundredMBPerSec) != 0 {
		t.Fatal("TransferDuration(0, rate) should return 0")
	}
	if TransferDuration(tenMB, 0) != 0 {
		t.Fatal("TransferDuration(bytes, 0) should return 0")
	}
	if TransferDuration(-10, hundredMBPerSec) != 0 {
		t.Fatal("TransferDuration(-bytes, rate) should return 0")
	}
}

func TestDedupActionMethods(t *testing.T) {
	if ActionSkip.String() != "skip" {
		t.Fatalf("ActionSkip.String() = %q, want \"skip\"", ActionSkip.String())
	}
	if ActionCheck.String() != "check" {
		t.Fatalf("ActionCheck.String() = %q, want \"check\"", ActionCheck.String())
	}
	if ActionProbe.String() != "probe" {
		t.Fatalf("ActionProbe.String() = %q, want \"probe\"", ActionProbe.String())
	}
	if DedupAction(99).String() != "unknown" {
		t.Fatalf("DedupAction(99).String() = %q, want \"unknown\"", DedupAction(99).String())
	}

	if ActionSkip.ShouldCheck() {
		t.Fatal("ActionSkip.ShouldCheck() should be false")
	}
	if !ActionCheck.ShouldCheck() {
		t.Fatal("ActionCheck.ShouldCheck() should be true")
	}
	if !ActionProbe.ShouldCheck() {
		t.Fatal("ActionProbe.ShouldCheck() should be true")
	}
}

func TestDedupCheckController_ZeroTelemetry_ConservativeDefault(t *testing.T) {
	// Default config declares ActionSkip as ConservativeDefault.
	cfg := DefaultDedupCheckConfig()
	c := NewDedupCheckController(cfg)

	cases := []struct {
		name  string
		input DedupCheckInput
	}{
		{
			name:  "completely empty input",
			input: DedupCheckInput{},
		},
		{
			name: "missing check overhead",
			input: DedupCheckInput{
				AvoidedTransfer: 10 * time.Millisecond,
			},
		},
		{
			name: "missing avoided transfer and no payload/bandwidth",
			input: DedupCheckInput{
				CheckOverhead: 2 * time.Millisecond,
			},
		},
		{
			name: "negative durations",
			input: DedupCheckInput{
				CheckOverhead:   -2 * time.Millisecond,
				AvoidedTransfer: -5 * time.Millisecond,
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rcpt := c.ShouldCheck(tc.input)
			if rcpt.Action != ActionSkip {
				t.Fatalf("expected ActionSkip on zero telemetry, got %v", rcpt.Action)
			}
			if rcpt.Reason != ReasonZeroTelemetry {
				t.Fatalf("expected ReasonZeroTelemetry, got %v", rcpt.Reason)
			}
			if rcpt.Admitted {
				t.Fatal("expected Admitted == false for ActionSkip")
			}
			if rcpt.IsProbe {
				t.Fatal("expected IsProbe == false for zero telemetry")
			}
		})
	}

	// Verify declared conservative default can be configured to ActionCheck.
	cfgWithCheck := DefaultDedupCheckConfig()
	cfgWithCheck.ConservativeDefault = ActionCheck
	cCheck := NewDedupCheckController(cfgWithCheck)

	rcptCheck := cCheck.ShouldCheck(DedupCheckInput{})
	if rcptCheck.Action != ActionCheck {
		t.Fatalf("expected declared ConservativeDefault ActionCheck, got %v", rcptCheck.Action)
	}
	if rcptCheck.Reason != ReasonZeroTelemetry {
		t.Fatalf("expected ReasonZeroTelemetry, got %v", rcptCheck.Reason)
	}
	if !rcptCheck.Admitted {
		t.Fatal("expected Admitted == true when ConservativeDefault is ActionCheck")
	}
}

func TestDedupCheckController_CandidateNegativeSavings(t *testing.T) {
	cfg := DefaultDedupCheckConfig()
	c := NewDedupCheckController(cfg)

	// Check overhead (5ms) > Avoided transfer (1ms) -> net savings would be -4ms even on hit.
	unfavorable := DedupCheckInput{
		CheckOverhead:   5 * time.Millisecond,
		AvoidedTransfer: 1 * time.Millisecond,
	}

	rcpt := c.ShouldCheck(unfavorable)
	if rcpt.Action != ActionSkip {
		t.Fatalf("expected ActionSkip for unfavorable candidate, got %v", rcpt.Action)
	}
	if rcpt.Reason != ReasonNegativeSavings {
		t.Fatalf("expected ReasonNegativeSavings, got %v", rcpt.Reason)
	}
	if rcpt.Admitted {
		t.Fatal("expected Admitted == false")
	}

	// Controller should remain enabled for subsequent favorable candidates.
	if !c.IsEnabled() {
		t.Fatal("controller should still be enabled overall")
	}

	favorable := DedupCheckInput{
		CheckOverhead:   1 * time.Millisecond,
		AvoidedTransfer: 20 * time.Millisecond,
	}
	rcptFav := c.ShouldCheck(favorable)
	if rcptFav.Action != ActionCheck {
		t.Fatalf("expected ActionCheck for favorable candidate, got %v", rcptFav.Action)
	}
	if rcptFav.Reason != ReasonSavingsFavorable {
		t.Fatalf("expected ReasonSavingsFavorable, got %v", rcptFav.Reason)
	}
	if !rcptFav.Admitted {
		t.Fatal("expected Admitted == true")
	}
}

func TestDedupCheckController_LosingStreak(t *testing.T) {
	cfg := DefaultDedupCheckConfig()
	cfg.MaxLosingStreak = 3
	c := NewDedupCheckController(cfg)

	input := DedupCheckInput{
		CheckOverhead:   1 * time.Millisecond,
		AvoidedTransfer: 10 * time.Millisecond,
	}

	// Initially enabled.
	if !c.IsEnabled() {
		t.Fatal("controller should start enabled")
	}

	// Miss 1
	rcpt1 := c.ShouldCheck(input)
	if rcpt1.Action != ActionCheck {
		t.Fatalf("expected ActionCheck, got %v", rcpt1.Action)
	}
	c.Observe(DedupObservation{
		Hit:           false,
		CheckOverhead: 1 * time.Millisecond,
	})
	if c.ConsecutiveMisses() != 1 {
		t.Fatalf("expected 1 consecutive miss, got %d", c.ConsecutiveMisses())
	}
	if !c.IsEnabled() {
		t.Fatal("should still be enabled after 1 miss")
	}

	// Miss 2
	c.Observe(DedupObservation{
		Hit:           false,
		CheckOverhead: 1 * time.Millisecond,
	})
	if c.ConsecutiveMisses() != 2 {
		t.Fatalf("expected 2 consecutive misses, got %d", c.ConsecutiveMisses())
	}
	if !c.IsEnabled() {
		t.Fatal("should still be enabled after 2 misses")
	}

	// Miss 3 reaches MaxLosingStreak -> self-disables!
	_, enabled := c.Observe(DedupObservation{
		Hit:           false,
		CheckOverhead: 1 * time.Millisecond,
	})
	if enabled {
		t.Fatal("controller should report enabled == false upon reaching MaxLosingStreak")
	}
	if c.IsEnabled() {
		t.Fatal("controller should be disabled")
	}
	if c.ConsecutiveMisses() != 3 {
		t.Fatalf("expected 3 consecutive misses, got %d", c.ConsecutiveMisses())
	}

	// Next ShouldCheck should be ActionSkip with ReasonLosingStreak.
	rcptAfter := c.ShouldCheck(input)
	if rcptAfter.Action != ActionSkip {
		t.Fatalf("expected ActionSkip, got %v", rcptAfter.Action)
	}
	if rcptAfter.Reason != ReasonLosingStreak {
		t.Fatalf("expected ReasonLosingStreak, got %v", rcptAfter.Reason)
	}
	if rcptAfter.Admitted {
		t.Fatal("expected Admitted == false")
	}
}

func TestDedupCheckController_HitResetsLosingStreak(t *testing.T) {
	cfg := DefaultDedupCheckConfig()
	cfg.MaxLosingStreak = 3
	c := NewDedupCheckController(cfg)

	// 2 misses
	c.Observe(DedupObservation{Hit: false, CheckOverhead: 1 * time.Millisecond})
	c.Observe(DedupObservation{Hit: false, CheckOverhead: 1 * time.Millisecond})
	if c.ConsecutiveMisses() != 2 {
		t.Fatalf("expected 2 misses, got %d", c.ConsecutiveMisses())
	}

	// 1 hit resets streak
	c.Observe(DedupObservation{
		Hit:             true,
		CheckOverhead:   1 * time.Millisecond,
		AvoidedTransfer: 10 * time.Millisecond,
	})
	if c.ConsecutiveMisses() != 0 {
		t.Fatalf("expected 0 consecutive misses after hit, got %d", c.ConsecutiveMisses())
	}
	if !c.IsEnabled() {
		t.Fatal("controller should still be enabled")
	}
}

func TestDedupCheckController_NegativeCumulativeSavings(t *testing.T) {
	cfg := DefaultDedupCheckConfig()
	cfg.MaxLosingStreak = 10
	cfg.MinChecksForSavings = 3
	c := NewDedupCheckController(cfg)

	input := DedupCheckInput{
		CheckOverhead:   5 * time.Millisecond,
		AvoidedTransfer: 10 * time.Millisecond,
	}

	// Check 1: Hit with small savings (overhead 5ms, avoided 6ms -> net +1ms)
	c.Observe(DedupObservation{
		Hit:             true,
		CheckOverhead:   5 * time.Millisecond,
		AvoidedTransfer: 6 * time.Millisecond,
	})
	if c.CumulativeSavings() != 1*time.Millisecond {
		t.Fatalf("expected +1ms cumulative savings, got %v", c.CumulativeSavings())
	}

	// Check 2: Miss (overhead 5ms -> net -5ms, cumulative -4ms)
	c.Observe(DedupObservation{
		Hit:           false,
		CheckOverhead: 5 * time.Millisecond,
	})
	if c.CumulativeSavings() != -4*time.Millisecond {
		t.Fatalf("expected -4ms cumulative savings, got %v", c.CumulativeSavings())
	}
	// Total checks is 2, less than MinChecksForSavings (3), so still enabled.
	if !c.IsEnabled() {
		t.Fatal("should still be enabled before MinChecksForSavings")
	}

	// Check 3: Miss (overhead 5ms -> net -5ms, cumulative -9ms)
	// Reaches MinChecksForSavings with negative cumulative savings -> disables!
	_, enabled := c.Observe(DedupObservation{
		Hit:           false,
		CheckOverhead: 5 * time.Millisecond,
	})
	if enabled {
		t.Fatal("controller should disable when cumulative savings is negative after min checks")
	}
	if c.IsEnabled() {
		t.Fatal("controller should be disabled")
	}

	rcpt := c.ShouldCheck(input)
	if rcpt.Action != ActionSkip {
		t.Fatalf("expected ActionSkip, got %v", rcpt.Action)
	}
	if rcpt.Reason != ReasonNegativeSavings {
		t.Fatalf("expected ReasonNegativeSavings, got %v", rcpt.Reason)
	}
}

func TestDedupCheckController_Disabled_AdmitsSparseProbes_AndRecovery(t *testing.T) {
	cfg := DefaultDedupCheckConfig()
	cfg.MaxLosingStreak = 2
	cfg.ProbeInterval = 4
	c := NewDedupCheckController(cfg)

	input := DedupCheckInput{
		CheckOverhead:   1 * time.Millisecond,
		AvoidedTransfer: 20 * time.Millisecond,
	}

	// Trip into disabled state via 2 consecutive misses.
	c.Observe(DedupObservation{Hit: false, CheckOverhead: 1 * time.Millisecond})
	c.Observe(DedupObservation{Hit: false, CheckOverhead: 1 * time.Millisecond})
	if c.IsEnabled() {
		t.Fatal("controller should be disabled")
	}

	// Calls 1, 2, 3 should be skipped.
	for i := 1; i <= 3; i++ {
		rcpt := c.ShouldCheck(input)
		if rcpt.Action != ActionSkip {
			t.Fatalf("call %d: expected ActionSkip, got %v", i, rcpt.Action)
		}
		if rcpt.Admitted {
			t.Fatalf("call %d: expected Admitted == false", i)
		}
		if rcpt.IsProbe {
			t.Fatalf("call %d: expected IsProbe == false", i)
		}
	}

	// Call 4: Probe interval reached -> emits ActionProbe!
	probeRcpt := c.ShouldCheck(input)
	if probeRcpt.Action != ActionProbe {
		t.Fatalf("call 4: expected ActionProbe, got %v", probeRcpt.Action)
	}
	if !probeRcpt.Admitted {
		t.Fatal("call 4: expected Admitted == true for probe")
	}
	if !probeRcpt.IsProbe {
		t.Fatal("call 4: expected IsProbe == true")
	}
	if probeRcpt.Reason != ReasonProbeAdmitted {
		t.Fatalf("call 4: expected ReasonProbeAdmitted, got %v", probeRcpt.Reason)
	}

	// Unfavorable probe: it misses. Controller stays disabled.
	_, reenabled := c.Observe(DedupObservation{
		Hit:           false,
		CheckOverhead: 1 * time.Millisecond,
		IsProbe:       true,
	})
	if reenabled {
		t.Fatal("controller should NOT re-enable after unfavorable probe")
	}
	if c.IsEnabled() {
		t.Fatal("controller should remain disabled")
	}

	// Next 3 calls skip again.
	for i := 1; i <= 3; i++ {
		rcpt := c.ShouldCheck(input)
		if rcpt.Action != ActionSkip {
			t.Fatalf("second cycle call %d: expected ActionSkip, got %v", i, rcpt.Action)
		}
	}

	// Next call is another probe.
	probeRcpt2 := c.ShouldCheck(input)
	if probeRcpt2.Action != ActionProbe {
		t.Fatalf("expected ActionProbe on second probe cycle, got %v", probeRcpt2.Action)
	}

	// Favorable probe: it hits with avoided transfer > check overhead.
	net, reenabled := c.Observe(DedupObservation{
		Hit:             true,
		CheckOverhead:   1 * time.Millisecond,
		AvoidedTransfer: 20 * time.Millisecond,
		IsProbe:         true,
	})
	if !reenabled {
		t.Fatal("controller SHOULD re-enable after favorable probe")
	}
	if !c.IsEnabled() {
		t.Fatal("controller should be enabled")
	}
	if net != 19*time.Millisecond {
		t.Fatalf("expected net savings +19ms, got %v", net)
	}
	if c.ConsecutiveMisses() != 0 {
		t.Fatalf("expected consecutive misses reset to 0, got %d", c.ConsecutiveMisses())
	}

	// Subsequent request should be an ordinary ActionCheck!
	afterRcpt := c.ShouldCheck(input)
	if afterRcpt.Action != ActionCheck {
		t.Fatalf("expected ActionCheck after re-enabling, got %v", afterRcpt.Action)
	}
	if afterRcpt.Reason != ReasonSavingsFavorable {
		t.Fatalf("expected ReasonSavingsFavorable, got %v", afterRcpt.Reason)
	}
	if !afterRcpt.Admitted {
		t.Fatal("expected Admitted == true")
	}
	if afterRcpt.IsProbe {
		t.Fatal("expected IsProbe == false for ordinary check")
	}
}

func TestDedupCheckReceipt_NoContentKeys_AndJSON(t *testing.T) {
	cfg := DefaultDedupCheckConfig()
	c := NewDedupCheckController(cfg)

	input := DedupCheckInput{
		CheckOverhead:   2 * time.Millisecond,
		AvoidedTransfer: 15 * time.Millisecond,
		PayloadBytes:    1024 * 1024,
	}

	rcpt := c.ShouldCheck(input)
	data, err := json.Marshal(rcpt)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	raw := string(data)

	// Verify required fields are present in JSON.
	requiredFields := []string{
		`"action"`,
		`"action_name"`,
		`"reason"`,
		`"admitted"`,
		`"is_probe"`,
		`"inputs"`,
		`"cumulative_savings"`,
		`"consecutive_misses"`,
		`"controller_enabled"`,
	}
	for _, field := range requiredFields {
		if !strings.Contains(raw, field) {
			t.Fatalf("JSON output missing field %s: %s", field, raw)
		}
	}

	// Verify ABSOLUTELY NO content keys, hashes, digests, paths, or keys exist.
	forbiddenSubstrings := []string{
		"content_key",
		"key",
		"hash",
		"digest",
		"token_content",
		"prompt",
		"path",
	}
	// We want to check object keys specifically to ensure no content fields leaked.
	var parsed map[string]any
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}
	for _, forbidden := range forbiddenSubstrings {
		if _, exists := parsed[forbidden]; exists {
			t.Fatalf("forbidden key %q found in receipt top-level JSON", forbidden)
		}
	}
	inputsMap, ok := parsed["inputs"].(map[string]any)
	if !ok {
		t.Fatalf("inputs is not a JSON object: %v", parsed["inputs"])
	}
	for _, forbidden := range forbiddenSubstrings {
		if _, exists := inputsMap[forbidden]; exists {
			t.Fatalf("forbidden key %q found in inputs JSON", forbidden)
		}
	}

	// Verify roundtrip unmarshal into DedupCheckReceipt.
	var unmarshaled DedupCheckReceipt
	if err := json.Unmarshal(data, &unmarshaled); err != nil {
		t.Fatalf("json.Unmarshal into DedupCheckReceipt failed: %v", err)
	}
	if unmarshaled.Action != rcpt.Action {
		t.Fatalf("roundtrip Decision mismatch: got %v, want %v", unmarshaled.Action, rcpt.Action)
	}
	if unmarshaled.Reason != rcpt.Reason {
		t.Fatalf("roundtrip Reason mismatch: got %v, want %v", unmarshaled.Reason, rcpt.Reason)
	}
	if unmarshaled.Admitted != rcpt.Admitted {
		t.Fatalf("roundtrip Admitted mismatch: got %v, want %v", unmarshaled.Admitted, rcpt.Admitted)
	}
}

func TestDedupCheckController_DerivedAvoidedTransfer(t *testing.T) {
	cfg := DefaultDedupCheckConfig()
	c := NewDedupCheckController(cfg)

	// 50 MB at 100 MB/s = 500ms
	input := DedupCheckInput{
		CheckOverhead:                2 * time.Millisecond,
		PayloadBytes:                 50 * 1024 * 1024,
		TransferBandwidthBytesPerSec: 100 * 1024 * 1024,
	}

	if !input.HasTelemetry() {
		t.Fatal("input with payload and bandwidth should report HasTelemetry() == true")
	}
	derived := input.EffectiveAvoidedTransfer()
	if derived != 500*time.Millisecond {
		t.Fatalf("EffectiveAvoidedTransfer() = %v, want 500ms", derived)
	}

	rcpt := c.ShouldCheck(input)
	if rcpt.Action != ActionCheck {
		t.Fatalf("expected ActionCheck for derived transfer, got %v", rcpt.Action)
	}
	if rcpt.Reason != ReasonSavingsFavorable {
		t.Fatalf("expected ReasonSavingsFavorable, got %v", rcpt.Reason)
	}
}

func TestDedupCheckController_StatsAndReset(t *testing.T) {
	cfg := DefaultDedupCheckConfig()
	cfg.MaxLosingStreak = 5
	c := NewDedupCheckController(cfg)

	c.Observe(DedupObservation{
		Hit:             true,
		CheckOverhead:   1 * time.Millisecond,
		AvoidedTransfer: 10 * time.Millisecond,
	})
	c.Observe(DedupObservation{
		Hit:           false,
		CheckOverhead: 1 * time.Millisecond,
	})
	c.Observe(DedupObservation{
		Hit:             true,
		CheckOverhead:   1 * time.Millisecond,
		AvoidedTransfer: 5 * time.Millisecond,
		IsProbe:         true,
	})

	stats := c.Stats()
	if stats.TotalChecks != 3 {
		t.Fatalf("TotalChecks = %d, want 3", stats.TotalChecks)
	}
	if stats.TotalHits != 2 {
		t.Fatalf("TotalHits = %d, want 2", stats.TotalHits)
	}
	if stats.TotalMisses != 1 {
		t.Fatalf("TotalMisses = %d, want 1", stats.TotalMisses)
	}
	if stats.TotalProbes != 1 {
		t.Fatalf("TotalProbes = %d, want 1", stats.TotalProbes)
	}
	if stats.ProbeHits != 1 {
		t.Fatalf("ProbeHits = %d, want 1", stats.ProbeHits)
	}
	// Expected hit rate = 2 / 3 ≈ 0.6667
	if stats.HitRate < 0.66 || stats.HitRate > 0.67 {
		t.Fatalf("HitRate = %f, want ~0.667", stats.HitRate)
	}
	// Cumulative:
	// Check 1: +9ms
	// Check 2: -1ms
	// Check 3: +4ms
	// Total: +12ms
	if stats.CumulativeSavings != 12*time.Millisecond {
		t.Fatalf("CumulativeSavings = %v, want 12ms", stats.CumulativeSavings)
	}

	c.Reset()
	statsAfter := c.Stats()
	if statsAfter.TotalChecks != 0 || statsAfter.CumulativeSavings != 0 {
		t.Fatalf("Reset did not clear statistics: %+v", statsAfter)
	}
	if !c.IsEnabled() {
		t.Fatal("controller should be enabled after Reset")
	}
}

func TestDedupCheckController_NilSafety(t *testing.T) {
	var c *DedupCheckController

	rcpt := c.ShouldCheck(DedupCheckInput{})
	if rcpt.Action != ActionSkip {
		t.Fatalf("nil controller ShouldCheck() = %v, want ActionSkip", rcpt.Action)
	}
	if rcpt.Admitted {
		t.Fatal("nil controller ShouldCheck() Admitted should be false")
	}

	net, enabled := c.Observe(DedupObservation{})
	if net != 0 || enabled {
		t.Fatalf("nil controller Observe() = (%v, %v), want (0, false)", net, enabled)
	}

	if c.IsEnabled() {
		t.Fatal("nil controller IsEnabled() should be false")
	}
	if c.CumulativeSavings() != 0 {
		t.Fatal("nil controller CumulativeSavings() should be 0")
	}
	if c.ConsecutiveMisses() != 0 {
		t.Fatal("nil controller ConsecutiveMisses() should be 0")
	}
	c.Reset() // should not panic
	stats := c.Stats()
	if stats.TotalChecks != 0 {
		t.Fatal("nil controller Stats() should be empty")
	}
}

func TestDedupCheckController_Concurrency(t *testing.T) {
	cfg := DefaultDedupCheckConfig()
	cfg.MaxLosingStreak = 4
	cfg.ProbeInterval = 8
	c := NewDedupCheckController(cfg)

	const goroutines = 20
	const iterations = 100

	var wg sync.WaitGroup
	wg.Add(goroutines)

	for g := 0; g < goroutines; g++ {
		go func(id int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				in := DedupCheckInput{
					CheckOverhead:   time.Duration(1+i%5) * time.Millisecond,
					AvoidedTransfer: time.Duration(5+i%10) * time.Millisecond,
				}
				rcpt := c.ShouldCheck(in)
				if rcpt.Admitted {
					c.Observe(DedupObservation{
						Hit:             (i+id)%3 == 0,
						CheckOverhead:   in.CheckOverhead,
						AvoidedTransfer: in.AvoidedTransfer,
						IsProbe:         rcpt.IsProbe,
					})
				}
				_ = c.IsEnabled()
				_ = c.CumulativeSavings()
				_ = c.Stats()
			}
		}(g)
	}

	wg.Wait()

	stats := c.Stats()
	if stats.TotalChecks == 0 {
		t.Fatal("expected non-zero TotalChecks after concurrent operations")
	}
}
