package stallscan

import (
	"strings"
	"testing"
	"time"
)

// The whole point of the arming axis is that "calm" and "unmeasured" must never be
// spelled the same way. Every test below is a variation on that one claim.

func TestClassifyArmingMissingIsNotCalm(t *testing.T) {
	got := ClassifyArming(LedgerRead{}, time.Now(), 90*time.Second, true)
	if got.Armed() {
		t.Fatalf("a host with no reading armed the gate: %+v", got)
	}
	if got.State != ArmStateMissing {
		t.Errorf("state = %q, want %q", got.State, ArmStateMissing)
	}
	// The detail must say NOT MEASURED, because a reader who sees a zero burst and
	// no explanation concludes "calm" — the exact confusion this axis removes.
	if !strings.Contains(got.Detail, "NOT MEASURED") {
		t.Errorf("missing-reading detail does not say the host is unmeasured: %q", got.Detail)
	}
}

func TestClassifyArmingZeroValueIsMissing(t *testing.T) {
	// A struct nobody populated must read as unmeasured, not as armed-and-calm.
	var z Arming
	if z.Armed() {
		t.Fatal("the zero Arming value reads as armed")
	}
	// Status (not the raw field) is what renderers emit, so it is what must never
	// go blank: `"state": ""` in a payload is skimmed past like a pass.
	if z.Status() != ArmStateMissing {
		t.Errorf("zero-value Status() = %q, want %q", z.Status(), ArmStateMissing)
	}
}

func TestClassifyArmingGarbledIsDistinctFromMissing(t *testing.T) {
	got := ClassifyArming(LedgerRead{Found: true}, time.Now(), 90*time.Second, true)
	if got.Armed() {
		t.Fatalf("an unparseable reading armed the gate: %+v", got)
	}
	// Garbled must not collapse into missing: missing means "install a monitor",
	// garbled means "a monitor is running and writing the wrong shape" — different
	// operator action, so the states stay distinct.
	if got.State != ArmStateGarbled {
		t.Errorf("state = %q, want %q", got.State, ArmStateGarbled)
	}
}

func TestClassifyArmingStaleReadingCannotGate(t *testing.T) {
	now := time.Now()
	r := LedgerRead{Found: true, Parsed: true, Timestamp: now.Add(-10 * time.Minute), SpawnBurst: 40}
	got := ClassifyArming(r, now, 90*time.Second, true)
	if got.Armed() {
		t.Fatalf("a 10-minute-old reading armed the gate: %+v", got)
	}
	if got.State != ArmStateStale {
		t.Errorf("state = %q, want %q", got.State, ArmStateStale)
	}
	if got.AgeSeconds < 599 || got.AgeSeconds > 601 {
		t.Errorf("age = %.1fs, want ~600s", got.AgeSeconds)
	}
	// A stale reading's burst count must not leak out as if it were current — it
	// would gate on a storm that drained ten minutes ago.
	if got.SpawnBurst != 0 {
		t.Errorf("stale reading exposed SpawnBurst = %d, want 0", got.SpawnBurst)
	}
}

func TestClassifyArmingDisabledShortCircuits(t *testing.T) {
	// Disabled is the ONE inert state that needs no alarm, so it must win even when
	// there is also no reading — otherwise turning the term off produces a
	// permanent "install a monitor" nag for a host that opted out on purpose.
	got := ClassifyArming(LedgerRead{}, time.Now(), 90*time.Second, false)
	if got.State != ArmStateDisabled {
		t.Errorf("state = %q, want %q", got.State, ArmStateDisabled)
	}
	if got.Armed() {
		t.Error("a disabled term reported itself armed")
	}
}

func TestClassifyArmingFreshReadingArms(t *testing.T) {
	now := time.Now()
	r := LedgerRead{Found: true, Parsed: true, Timestamp: now.Add(-5 * time.Second), SpawnBurst: 12}
	got := ClassifyArming(r, now, 90*time.Second, true)
	if !got.Armed() {
		t.Fatalf("a 5s-old reading did not arm: %+v", got)
	}
	if got.SpawnBurst != 12 {
		t.Errorf("SpawnBurst = %d, want 12", got.SpawnBurst)
	}
}

func TestClassifyArmingZeroFreshnessDisablesStalenessCheck(t *testing.T) {
	now := time.Now()
	r := LedgerRead{Found: true, Parsed: true, Timestamp: now.Add(-10 * time.Minute), SpawnBurst: 3}
	got := ClassifyArming(r, now, 0, true)
	if !got.Armed() {
		t.Fatalf("freshness=0 should disable the staleness check, got %+v", got)
	}
}

func TestClassifyArmingFutureStampIsNotNegativeAge(t *testing.T) {
	// Clock skew between the writer and the reader must not render as a negative
	// age in an operator-facing string.
	now := time.Now()
	r := LedgerRead{Found: true, Parsed: true, Timestamp: now.Add(2 * time.Second), SpawnBurst: 1}
	got := ClassifyArming(r, now, 90*time.Second, true)
	if got.AgeSeconds < 0 {
		t.Errorf("age = %.1f, want >= 0", got.AgeSeconds)
	}
}

func TestArmingDetailAlwaysPopulated(t *testing.T) {
	now := time.Now()
	cases := map[string]Arming{
		"missing":  ClassifyArming(LedgerRead{}, now, time.Minute, true),
		"garbled":  ClassifyArming(LedgerRead{Found: true}, now, time.Minute, true),
		"stale":    ClassifyArming(LedgerRead{Found: true, Parsed: true, Timestamp: now.Add(-time.Hour)}, now, time.Minute, true),
		"disabled": ClassifyArming(LedgerRead{}, now, time.Minute, false),
		"armed":    ClassifyArming(LedgerRead{Found: true, Parsed: true, Timestamp: now}, now, time.Minute, true),
	}
	for name, a := range cases {
		if strings.TrimSpace(a.Detail) == "" {
			t.Errorf("%s: Detail is empty — an unexplained state is what makes inertness invisible", name)
		}
	}
}
