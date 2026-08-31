package incidentrsi

import (
	"strings"
	"sync"
	"testing"
	"time"
)

func TestFingerprintNormalizesStructuralFields(t *testing.T) {
	left := Input{
		Source:        SourceUnexpectedHook,
		Operation:     "  Stop-Hook / Dispatch  ",
		ErrorClass:    "IO.Timeout",
		CauseIdentity: "Connection Reset",
	}
	right := Input{
		Source:        Source("UNEXPECTED HOOK"),
		Operation:     "stop_hook_dispatch",
		ErrorClass:    "io timeout",
		CauseIdentity: "connection-reset",
	}
	if got, want := Fingerprint(left), Fingerprint(right); got != want {
		t.Fatalf("normalized fingerprints differ: %q != %q", got, want)
	}
}

func TestFingerprintExcludesRawAndSecretContent(t *testing.T) {
	input := Input{
		Source:        SourceGatewayTransport,
		Operation:     "dial",
		ErrorClass:    "transport_timeout",
		CauseIdentity: "upstream_unreachable",
	}
	fingerprint := Fingerprint(input)
	for _, forbidden := range []string{"dial", "transport", "upstream", "secret-token", `C:\\private`, "host.example"} {
		if strings.Contains(fingerprint, forbidden) {
			t.Fatalf("fingerprint %q exposed %q", fingerprint, forbidden)
		}
	}
	if len(fingerprint) != len("irsi-v1-")+64 {
		t.Fatalf("fingerprint has unexpected format: %q", fingerprint)
	}
}

func TestExpectedFailureIsSuppressed(t *testing.T) {
	kernel := New(Config{Threshold: 1})
	decision := kernel.Observe(baseInput(time.Unix(1, 0), true))
	if decision.Action != ActionNoop || decision.Count != 0 {
		t.Fatalf("expected noop without state, got %+v", decision)
	}
}

func TestEndUserGetsDoctorInsteadOfLaunch(t *testing.T) {
	kernel := New(Config{Threshold: 1})
	input := baseInput(time.Unix(1, 0), false)
	input.Developer = false
	decision := kernel.Observe(input)
	if decision.Action != ActionDoctor {
		t.Fatalf("expected doctor, got %+v", decision)
	}
	if decision.Recommendation == "" || strings.Contains(decision.Recommendation, input.Operation) {
		t.Fatalf("unsafe or missing recommendation: %+v", decision)
	}
}

func TestThresholdCrossingAndCooldown(t *testing.T) {
	start := time.Unix(100, 0)
	kernel := New(Config{Threshold: 2, Cooldown: 10 * time.Minute, MaxFingerprints: 8, StaleAfter: time.Hour})

	first := kernel.Observe(baseInput(start, false))
	if first.Action != ActionObserve || first.Count != 1 {
		t.Fatalf("first observation = %+v", first)
	}
	second := kernel.Observe(baseInput(start.Add(time.Second), false))
	if second.Action != ActionLaunch || second.Count != 2 {
		t.Fatalf("threshold crossing = %+v", second)
	}
	wantEligible := start.Add(time.Second).Add(10 * time.Minute)
	if !second.NextEligibleAt.Equal(wantEligible) {
		t.Fatalf("next eligible = %v, want %v", second.NextEligibleAt, wantEligible)
	}
	third := kernel.Observe(baseInput(start.Add(2*time.Second), false))
	if third.Action != ActionUpdate || third.Count != 3 || !third.NextEligibleAt.Equal(wantEligible) {
		t.Fatalf("cooldown observation = %+v", third)
	}
	after := kernel.Observe(baseInput(wantEligible, false))
	if after.Action != ActionLaunch || after.Count != 4 {
		t.Fatalf("post-cooldown observation = %+v", after)
	}
}

func TestDistinctFingerprintsAreIndependent(t *testing.T) {
	kernel := New(Config{Threshold: 2, Cooldown: time.Hour})
	at := time.Unix(200, 0)
	first := baseInput(at, false)
	second := first
	second.CauseIdentity = "tls_handshake"

	if got := kernel.Observe(first); got.Action != ActionObserve || got.Count != 1 {
		t.Fatalf("first identity = %+v", got)
	}
	if got := kernel.Observe(second); got.Action != ActionObserve || got.Count != 1 {
		t.Fatalf("second identity = %+v", got)
	}
	if Fingerprint(first) == Fingerprint(second) {
		t.Fatal("distinct structural causes shared a fingerprint")
	}
}

func TestConcurrentThresholdCrossingLaunchesExactlyOnce(t *testing.T) {
	const observations = 128
	kernel := New(Config{Threshold: 8, Cooldown: time.Hour, MaxFingerprints: 8, StaleAfter: 2 * time.Hour})
	input := baseInput(time.Unix(300, 0), false)

	var wg sync.WaitGroup
	decisions := make(chan Decision, observations)
	wg.Add(observations)
	for range observations {
		go func() {
			defer wg.Done()
			decisions <- kernel.Observe(input)
		}()
	}
	wg.Wait()
	close(decisions)

	launches := 0
	updates := 0
	for decision := range decisions {
		switch decision.Action {
		case ActionLaunch:
			launches++
		case ActionUpdate:
			updates++
		}
	}
	if launches != 1 {
		t.Fatalf("launches = %d, want 1", launches)
	}
	if updates != observations-8 {
		t.Fatalf("updates = %d, want %d", updates, observations-8)
	}
}

func TestBoundedStateEvictsOldestFingerprint(t *testing.T) {
	kernel := New(Config{Threshold: 10, Cooldown: time.Hour, MaxFingerprints: 2, StaleAfter: 24 * time.Hour})
	start := time.Unix(400, 0)

	oldest := baseInput(start, false)
	oldest.CauseIdentity = "oldest"
	middle := baseInput(start.Add(time.Second), false)
	middle.CauseIdentity = "middle"
	newest := baseInput(start.Add(2*time.Second), false)
	newest.CauseIdentity = "newest"

	kernel.Observe(oldest)
	kernel.Observe(middle)
	kernel.Observe(newest)

	oldest.OccurredAt = start.Add(3 * time.Second)
	if got := kernel.Observe(oldest); got.Count != 1 {
		t.Fatalf("evicted identity retained state: %+v", got)
	}
}

func baseInput(at time.Time, expected bool) Input {
	return Input{
		Source:        SourceUnexpectedHook,
		Operation:     "post_tool_use",
		ErrorClass:    "hook_execution",
		CauseIdentity: "unexpected_exit",
		OccurredAt:    at,
		Developer:     true,
		Expected:      expected,
	}
}
