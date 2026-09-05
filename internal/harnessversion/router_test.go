package harnessversion

import (
	"fmt"
	"sync"
	"testing"
)

func TestExplicitWireNegotiation(t *testing.T) {
	r := NewStickySessionRouter()
	if err := r.Register(VersionDescriptor{
		Version:  "v1",
		Weight:   100,
		Active:   true,
		Metadata: map[string]string{"role": "stable"},
	}); err != nil {
		t.Fatalf("Register v1 failed: %v", err)
	}

	if err := r.Register(VersionDescriptor{
		Version:  "v2",
		Weight:   50,
		Active:   true,
		Metadata: map[string]string{"role": "canary"},
	}); err != nil {
		t.Fatalf("Register v2 failed: %v", err)
	}

	// 1. Header negotiation
	if got := r.Negotiate("v1", ""); got != "v1" {
		t.Errorf("expected Negotiate header v1 to return v1, got %q", got)
	}
	if got := r.Negotiate("v2", ""); got != "v2" {
		t.Errorf("expected Negotiate header v2 to return v2, got %q", got)
	}

	// 2. Path parameter negotiation
	if got := r.Negotiate("", "v1"); got != "v1" {
		t.Errorf("expected Negotiate path v1 to return v1, got %q", got)
	}
	if got := r.Negotiate("", "/v2"); got != "v2" {
		t.Errorf("expected Negotiate path /v2 to return v2, got %q", got)
	}
	if got := r.Negotiate("", "/v2/execute"); got != "v2" {
		t.Errorf("expected Negotiate path /v2/execute to return v2, got %q", got)
	}

	// 3. Header takes precedence over path
	if got := r.Negotiate("v1", "v2"); got != "v1" {
		t.Errorf("expected header v1 to override path v2, got %q", got)
	}

	// 4. Empty wire signals return empty string (defers to canary)
	if got := r.Negotiate("", ""); got != "" {
		t.Errorf("expected empty inputs to return empty string, got %q", got)
	}

	// 5. Route with explicit wire signals
	ver, wasPinned := r.Route("sess-wire-1", "v2", "")
	if ver != "v2" || wasPinned {
		t.Errorf("expected Route to return (v2, false), got (%s, %v)", ver, wasPinned)
	}
	ver, wasPinned = r.Route("sess-wire-2", "", "/v1")
	if ver != "v1" || wasPinned {
		t.Errorf("expected Route to return (v1, false), got (%s, %v)", ver, wasPinned)
	}
}

func TestStickySessionPinning(t *testing.T) {
	r := NewStickySessionRouter()
	if err := r.Register(VersionDescriptor{Version: "v1", Weight: 50, Active: true}); err != nil {
		t.Fatalf("Register v1 failed: %v", err)
	}
	if err := r.Register(VersionDescriptor{Version: "v2", Weight: 50, Active: true}); err != nil {
		t.Fatalf("Register v2 failed: %v", err)
	}

	sessionID := "session-sticky-42"

	// First call routes and pins
	firstVer, wasPinned := r.Route(sessionID, "", "")
	if firstVer == "" {
		t.Fatalf("expected non-empty version")
	}
	if wasPinned {
		t.Errorf("expected first call to have wasPinned=false")
	}

	// Subsequent calls must return the exact same pinned version with wasPinned=true
	for i := 0; i < 25; i++ {
		ver, pinned := r.Route(sessionID, "", "")
		if ver != firstVer {
			t.Fatalf("call %d: expected pinned version %q, got %q", i, firstVer, ver)
		}
		if !pinned {
			t.Fatalf("call %d: expected wasPinned=true", i)
		}
	}

	// Conflicting header or path MUST NOT override an already pinned session
	conflictingVer := "v2"
	if firstVer == "v2" {
		conflictingVer = "v1"
	}
	ver, pinned := r.Route(sessionID, conflictingVer, "")
	if ver != firstVer || !pinned {
		t.Errorf("conflicting wire header should not break sticky pin: got (%s, %v), want (%s, true)", ver, pinned, firstVer)
	}

	// Verify GetPinnedVersion
	pinnedVer, ok := r.GetPinnedVersion(sessionID)
	if !ok || pinnedVer != firstVer {
		t.Errorf("GetPinnedVersion: got (%q, %v), want (%q, true)", pinnedVer, ok, firstVer)
	}

	// Release session unpins
	r.ReleaseSession(sessionID)
	pinnedVer, ok = r.GetPinnedVersion(sessionID)
	if ok || pinnedVer != "" {
		t.Errorf("after ReleaseSession: got (%q, %v), want ('', false)", pinnedVer, ok)
	}

	// Routing after release is treated as new session
	nextVer, wasPinnedNext := r.Route(sessionID, "", "")
	if nextVer == "" {
		t.Fatalf("expected non-empty version after release")
	}
	if wasPinnedNext {
		t.Errorf("expected wasPinned=false on newly routed session after release")
	}
}

func TestCanarySplittingDistribution(t *testing.T) {
	r := NewStickySessionRouter()
	if err := r.Register(VersionDescriptor{
		Version:  "v1",
		Weight:   90,
		Active:   true,
		Metadata: map[string]string{"channel": "stable"},
	}); err != nil {
		t.Fatalf("Register v1 failed: %v", err)
	}
	if err := r.Register(VersionDescriptor{
		Version:  "v2",
		Weight:   10,
		Active:   true,
		Metadata: map[string]string{"channel": "canary"},
	}); err != nil {
		t.Fatalf("Register v2 failed: %v", err)
	}

	// Verify ActiveVersions returns defensive copies
	active := r.ActiveVersions()
	if len(active) != 2 {
		t.Fatalf("expected 2 active versions, got %d", len(active))
	}
	if active[0].Version != "v1" || active[1].Version != "v2" {
		t.Errorf("expected sorted active versions [v1, v2], got [%s, %s]", active[0].Version, active[1].Version)
	}

	const totalSessions = 10000
	counts := make(map[string]int)

	for i := 0; i < totalSessions; i++ {
		sid := fmt.Sprintf("session-canary-%d", i)
		selected, wasPinned := r.Route(sid, "", "")
		if wasPinned {
			t.Fatalf("initial route for session %s returned wasPinned=true", sid)
		}
		counts[selected]++

		// Verify sticky pin on immediate second call
		reselected, rePinned := r.Route(sid, "", "")
		if reselected != selected || !rePinned {
			t.Fatalf("session %s re-route failed: got (%s, %v), want (%s, true)", sid, reselected, rePinned, selected)
		}
	}

	v1Ratio := float64(counts["v1"]) / float64(totalSessions)
	v2Ratio := float64(counts["v2"]) / float64(totalSessions)

	t.Logf("Canary distribution across %d sessions: v1(90%%)=%d (%.2f%%), v2(10%%)=%d (%.2f%%)",
		totalSessions, counts["v1"], v1Ratio*100, counts["v2"], v2Ratio*100)

	// Tolerance check within 3% of expected weights (v1: 87%-93%, v2: 7%-13%)
	if v1Ratio < 0.87 || v1Ratio > 0.93 {
		t.Errorf("v1 distribution out of bounds: expected ~0.90, got %.4f", v1Ratio)
	}
	if v2Ratio < 0.07 || v2Ratio > 0.13 {
		t.Errorf("v2 distribution out of bounds: expected ~0.10, got %.4f", v2Ratio)
	}
}

func TestInvalidUnregisteredVersionsFailClosed(t *testing.T) {
	r := NewStickySessionRouter()
	if err := r.Register(VersionDescriptor{
		Version:  "v1",
		Weight:   90,
		Active:   true,
		Metadata: map[string]string{"stable": "true"},
	}); err != nil {
		t.Fatalf("Register v1 failed: %v", err)
	}
	if err := r.Register(VersionDescriptor{
		Version:  "v2",
		Weight:   10,
		Active:   true,
		Metadata: map[string]string{"role": "canary"},
	}); err != nil {
		t.Fatalf("Register v2 failed: %v", err)
	}
	if err := r.Register(VersionDescriptor{
		Version:  "v3-disabled",
		Weight:   50,
		Active:   false,
		Metadata: map[string]string{"deprecated": "true"},
	}); err != nil {
		t.Fatalf("Register v3 failed: %v", err)
	}

	defaultStable := r.DefaultVersion()
	if defaultStable != "v1" {
		t.Fatalf("expected default stable version to be v1, got %q", defaultStable)
	}

	// 1. Negotiate with unregistered string fails closed
	if got := r.Negotiate("v999", ""); got != "v1" {
		t.Errorf("expected Negotiate unregistered header to fail closed to v1, got %q", got)
	}
	if got := r.Negotiate("", "unregistered_path"); got != "v1" {
		t.Errorf("expected Negotiate unregistered path to fail closed to v1, got %q", got)
	}

	// 2. Negotiate with inactive version fails closed
	if got := r.Negotiate("v3-disabled", ""); got != "v1" {
		t.Errorf("expected Negotiate inactive version to fail closed to v1, got %q", got)
	}

	// 3. Negotiate with malicious / traversal strings fails closed
	if got := r.Negotiate("../../etc/passwd", ""); got != "v1" {
		t.Errorf("expected Negotiate malicious path to fail closed to v1, got %q", got)
	}

	// 4. Route with invalid headers across 100 sessions: 100% must route to default stable v1
	for i := 0; i < 100; i++ {
		sid := fmt.Sprintf("invalid-sess-%d", i)
		ver, wasPinned := r.Route(sid, "v999-bad-version", "")
		if ver != "v1" {
			t.Fatalf("session %s failed closed to wrong version: got %q, want 'v1'", sid, ver)
		}
		if wasPinned {
			t.Fatalf("session %s initial invalid route had wasPinned=true", sid)
		}

		// Subsequent call for pinned session
		ver2, wasPinned2 := r.Route(sid, "v999-bad-version", "")
		if ver2 != "v1" || !wasPinned2 {
			t.Fatalf("session %s second call failed: got (%s, %v), want (v1, true)", sid, ver2, wasPinned2)
		}
	}

	// 5. Inactive version via path fails closed
	ver, _ := r.Route("inactive-path-test", "", "v3-disabled")
	if ver != "v1" {
		t.Errorf("expected Route inactive path to fail closed to v1, got %q", ver)
	}
}

func TestConcurrencySafety(t *testing.T) {
	r := NewStickySessionRouter()
	_ = r.Register(VersionDescriptor{Version: "v1", Weight: 80, Active: true})
	_ = r.Register(VersionDescriptor{Version: "v2", Weight: 20, Active: true})

	var wg sync.WaitGroup
	const goroutines = 30
	const iterations = 100

	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				sid := fmt.Sprintf("concurrent-session-%d-%d", workerID, i%10)

				// Concurrent Route
				ver, _ := r.Route(sid, "", "")
				if ver != "v1" && ver != "v2" {
					t.Errorf("unexpected version: %s", ver)
				}

				// Concurrent GetPinnedVersion
				_, _ = r.GetPinnedVersion(sid)

				// Concurrent ActiveVersions
				_ = r.ActiveVersions()

				// Concurrent Negotiate
				_ = r.Negotiate("v1", "")
				_ = r.Negotiate("", "v2")
				_ = r.Negotiate("invalid", "")

				// Occasional ReleaseSession
				if i%5 == 0 {
					r.ReleaseSession(sid)
				}

				// Occasional Register update
				if i%20 == 0 {
					_ = r.Register(VersionDescriptor{
						Version: "v2",
						Weight:  20 + (i % 5),
						Active:  true,
					})
				}
			}
		}(g)
	}

	wg.Wait()
}

func TestDescriptorValidationAndEdgeCases(t *testing.T) {
	r := NewStickySessionRouter()

	// Empty version error
	if err := r.Register(VersionDescriptor{Version: "", Weight: 10, Active: true}); err == nil {
		t.Errorf("expected error registering empty version")
	}

	// Negative weight error
	if err := r.Register(VersionDescriptor{Version: "v1", Weight: -5, Active: true}); err == nil {
		t.Errorf("expected error registering negative weight")
	}

	// No versions registered
	emptyRouter := NewStickySessionRouter()
	if got := emptyRouter.Negotiate("v1", ""); got != "" {
		t.Errorf("expected empty string for negotiate on empty router, got %q", got)
	}
	ver, pinned := emptyRouter.Route("any", "", "")
	if ver != "" || pinned {
		t.Errorf("expected ('', false) for route on empty router, got (%q, %v)", ver, pinned)
	}

	// Stateless routing (empty session ID)
	_ = r.Register(VersionDescriptor{Version: "v1", Weight: 100, Active: true})
	v, p := r.Route("", "", "")
	if v != "v1" || p {
		t.Errorf("stateless route should return (v1, false), got (%s, %v)", v, p)
	}
	if _, ok := r.GetPinnedVersion(""); ok {
		t.Errorf("empty session should not be pinned")
	}

	// Explicit SetDefaultVersion
	_ = r.Register(VersionDescriptor{Version: "v2", Weight: 10, Active: true})
	if err := r.SetDefaultVersion("v2"); err != nil {
		t.Fatalf("SetDefaultVersion failed: %v", err)
	}
	if r.DefaultVersion() != "v2" {
		t.Errorf("expected default version v2, got %q", r.DefaultVersion())
	}

	// SetDefaultVersion fails for unregistered / inactive
	if err := r.SetDefaultVersion("unregistered"); err == nil {
		t.Errorf("expected error setting unregistered version as default")
	}
	_ = r.Register(VersionDescriptor{Version: "v3-inactive", Weight: 10, Active: false})
	if err := r.SetDefaultVersion("v3-inactive"); err == nil {
		t.Errorf("expected error setting inactive version as default")
	}
}
