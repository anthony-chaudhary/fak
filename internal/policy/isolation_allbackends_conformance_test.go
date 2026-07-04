package policy

// All-registered-backend adjudication-floor conformance (issue #2018,
// acceptance bullet 2: "a single adjudication test suite runs against ALL
// registered backends; a policy-denied action is blocked in every one").
//
// isolation_capfloor_conformance_test.go pins the #2018 floor invariant at four
// hand-picked tiers (goroutine, subprocess, container, gvisor). It silently
// OMITS firecracker and remote — the two untrusted-pole backends where the
// adjudication floor matters most, since they carry the least-trusted work. The
// literal acceptance wording is "all registered backends", so a four-of-six
// witness leaves the strongest tiers unproven. This test closes that gap: it
// drives EVERY backend in the closed isolationLadder vocabulary and pins the
// vocabulary itself, so adding a backend without a floor witness fails loudly
// rather than shrinking coverage silently.

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// TestAdjudicationFloorHoldsAcrossAllRegisteredBackends is the #2018 acceptance
// witness in its literal "all registered backends" form. For each backend in
// the closed isolationLadder it (a) proves the dial actually places a trust
// level on that backend and (b) proves the allow/deny floor is byte-identical
// there — including firecracker and remote, which the four-tier witness omits.
func TestAdjudicationFloorHoldsAcrossAllRegisteredBackends(t *testing.T) {
	// The registered backend vocabulary this floor witness must cover, pinned so
	// that ADDING a backend to isolationLadder trips this test until the author
	// confirms the #2018 floor holds for the new tier (a real drift guard, not a
	// tautology derived from the ladder under test).
	wantBackends := map[string]bool{
		backendGoroutine: true,
		"subprocess":     true,
		"container":      true,
		"gvisor":         true,
		"firecracker":    true,
		"remote":         true,
	}
	if len(isolationLadder) != len(wantBackends) {
		t.Fatalf("isolationLadder has %d backends, floor witness pins %d — a backend was added or removed; extend this conformance witness (#2018 requires ALL registered backends)",
			len(isolationLadder), len(wantBackends))
	}
	for b := range isolationLadder {
		if !wantBackends[b] {
			t.Fatalf("registered backend %q has no adjudication-floor witness; #2018 requires all registered backends to be covered", b)
		}
	}

	// One synthetic trust level per registered backend. goroutine maps to a
	// non-untrusted level (untrusted -> goroutine is refused at load); every
	// other backend gets its own dedicated level.
	backends := isolationBackendNames() // weakest -> strongest, whole vocabulary
	levelFor := func(b string) string { return "trust_" + b }
	trust := make(map[string]string, len(backends))
	for _, b := range backends {
		trust[levelFor(b)] = b
	}
	isoJSON, err := json.Marshal(struct {
		Backends []string          `json:"backends"`
		Trust    map[string]string `json:"trust"`
	}{Backends: backends, Trust: trust})
	if err != nil {
		t.Fatalf("marshal isolation block: %v", err)
	}
	manifest := fmt.Sprintf(`{"version":"fak-policy/v1","allow":["search_kb"],"isolation":%s}`, isoJSON)

	rt, err := ParseRuntime([]byte(manifest))
	if err != nil {
		t.Fatalf("ParseRuntime (all backends): %v", err)
	}
	if rt.Isolation == nil {
		t.Fatal("Runtime.Isolation is nil for a manifest that declares every registered backend")
	}

	a := adjudicator.New(rt.Adjudicator)
	ctx := context.Background()

	covered := make(map[string]bool, len(backends))
	for _, b := range backends {
		level := levelFor(b)

		// (a) The tier is real: the dial resolves this level to backend b.
		got, err := rt.Isolation.BackendFor(level)
		if err != nil || got != b {
			t.Fatalf("BackendFor(%q) = %q, %v; want backend %q", level, got, err, b)
		}

		// (b) The #2018 floor is UNCHANGED at backend b: an allowed tool is
		// allowed and the destructive tool is denied, whether b is the in-process
		// goroutine tier or the untrusted-pole firecracker/remote tier.
		if v := a.Adjudicate(ctx, roCall("search_kb")); v.Kind != abi.VerdictAllow {
			t.Fatalf("backend %s: search_kb Kind=%v, want Allow — isolation must not narrow the floor", b, v.Kind)
		}
		if v := a.Adjudicate(ctx, roCall("refund_payment")); v.Kind != abi.VerdictDeny {
			t.Fatalf("backend %s: refund_payment Kind=%v, want Deny — stronger isolation must not bypass the floor", b, v.Kind)
		}

		covered[b] = true
	}

	// Every pinned backend actually got exercised above.
	for b := range wantBackends {
		if !covered[b] {
			t.Fatalf("backend %q was pinned but never exercised by the floor loop", b)
		}
	}
}
