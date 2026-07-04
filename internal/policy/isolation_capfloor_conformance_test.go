package policy

// Process-isolation capability-floor conformance (issue #2171, layer 5 /
// acceptance bullet 4; the adjudication floor is #2018).
//
// #2171 defines a process-isolation topology for many concurrent fak agents as
// a stack of runtime layers (kernel/supervisor, agent-session, tool-call,
// TUI/client, capability, resource). Its layer 5 is the load-bearing security
// invariant: "every isolation tier still routes effects through fak
// adjudication (#2018), so stronger process isolation does not bypass the trust
// floor." Bullet 4 asks for policy conformance tests that still pass at each
// process-isolation tier, linking back to #2018.
//
// The isolation dial (isolation.go, #2013) is the policy-lane surface that
// picks a *process-isolation tier* (goroutine -> subprocess -> container ->
// gvisor -> firecracker/remote) from a task's trust level. isolation_test.go
// already pins that the dial SELECTS the right backend and fails closed. What
// was unwitnessed is the cross-cutting #2018 invariant these two tests pin: the
// name-level ALLOW/DENY floor (adjudicator.Policy) is *orthogonal to* and
// *invariant across* the isolation tier. A stronger tier can neither bypass the
// floor (a denied tool stays denied at gvisor) nor is a weaker tier granted a
// looser floor (an allowed tool is not narrowed at goroutine), because
// Runtime.Isolation is not an adjudicator.Policy field. These are regression
// witnesses for that structural guarantee.

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// roCall builds an inline read-only tool call. Inline args need no resolver on
// the adjudication read path (see adjudicator's inlineCall helper), so a
// simple allow/deny verdict is decided without a blob backend.
func roCall(tool string) *abi.ToolCall {
	return &abi.ToolCall{
		Tool: tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)},
		Meta: map[string]string{"readOnlyHint": "true"},
	}
}

// TestIsolationTiersDoNotBypassAdjudicationFloor drives the #2018 floor at every
// declared process-isolation tier and proves the allow/deny verdict is
// identical at each — the tier a task dials to never changes whether a tool
// name is admissible. This is the executable form of #2171 layer 5.
func TestIsolationTiersDoNotBypassAdjudicationFloor(t *testing.T) {
	// One manifest: a read-only capability floor (allow: search_kb) AND a dial
	// spanning four DISTINCT tiers, weakest -> strongest. refund_payment is the
	// canonical destructive tool the floor denies (mirrors the 60-second proof's
	// customer-support-readonly policy).
	rt, err := ParseRuntime([]byte(`{
		"version": "fak-policy/v1",
		"allow": ["search_kb"],
		"isolation": {
			"backends": ["goroutine", "subprocess", "container", "gvisor"],
			"trust": {
				"trusted": "goroutine",
				"vetted": "subprocess",
				"third_party": "container",
				"untrusted": "gvisor"
			}
		}
	}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	if rt.Isolation == nil {
		t.Fatal("Runtime.Isolation is nil for a manifest that declares the dial")
	}

	a := adjudicator.New(rt.Adjudicator)
	ctx := context.Background()

	// Weakest -> strongest process-isolation tiers the dial resolves to.
	tiers := []struct{ level, backend string }{
		{"trusted", "goroutine"},
		{"vetted", "subprocess"},
		{"third_party", "container"},
		{"untrusted", "gvisor"},
	}

	for _, tr := range tiers {
		// (a) The tier is real: the dial places this trust level on its backend.
		got, err := rt.Isolation.BackendFor(tr.level)
		if err != nil || got != tr.backend {
			t.Fatalf("BackendFor(%q) = %q, %v; want backend %q", tr.level, got, err, tr.backend)
		}

		// (b) The #2018 floor is UNCHANGED at this tier: an allowed tool is
		// allowed and a denied tool is denied, no matter the isolation backend.
		if v := a.Adjudicate(ctx, roCall("search_kb")); v.Kind != abi.VerdictAllow {
			t.Fatalf("tier %s (%s): search_kb Kind=%v, want Allow — isolation must not narrow the floor",
				tr.level, tr.backend, v.Kind)
		}
		if v := a.Adjudicate(ctx, roCall("refund_payment")); v.Kind != abi.VerdictDeny {
			t.Fatalf("tier %s (%s): refund_payment Kind=%v, want Deny — stronger isolation must not bypass the floor",
				tr.level, tr.backend, v.Kind)
		}

		// (c) The args-independent floor query agrees — belt and suspenders.
		if rt.Adjudicator.NeverAdmits("search_kb") {
			t.Fatalf("tier %s: NeverAdmits(search_kb) = true, want false", tr.level)
		}
		if !rt.Adjudicator.NeverAdmits("refund_payment") {
			t.Fatalf("tier %s: NeverAdmits(refund_payment) = false, want true", tr.level)
		}
	}
}

// TestIsolationDialOrthogonalToAdjudicationFloor is the structural witness: two
// manifests with the SAME allow/deny floor but one with an isolation dial and
// one without produce byte-identical adjudication verdicts for every tool.
// Configuring (or removing) the process-isolation tier cannot move the #2018
// floor, because Runtime.Isolation is not an adjudicator.Policy field.
func TestIsolationDialOrthogonalToAdjudicationFloor(t *testing.T) {
	base := `{"version":"fak-policy/v1","allow":["search_kb"]}`
	withDial := `{
		"version": "fak-policy/v1",
		"allow": ["search_kb"],
		"isolation": {"backends": ["goroutine", "gvisor"], "trust": {"trusted": "goroutine", "untrusted": "gvisor"}}
	}`

	rtBase, err := ParseRuntime([]byte(base))
	if err != nil {
		t.Fatalf("ParseRuntime(base): %v", err)
	}
	rtDial, err := ParseRuntime([]byte(withDial))
	if err != nil {
		t.Fatalf("ParseRuntime(withDial): %v", err)
	}

	// The isolation dial is present in exactly one of the two manifests...
	if rtBase.Isolation != nil {
		t.Fatalf("base manifest declares no isolation block, want nil dial, got %+v", rtBase.Isolation)
	}
	if rtDial.Isolation == nil {
		t.Fatal("dial manifest declares an isolation block, want non-nil dial")
	}

	// ...yet the #2018 floor is identical for every tool, allowed or denied.
	aBase := adjudicator.New(rtBase.Adjudicator)
	aDial := adjudicator.New(rtDial.Adjudicator)
	ctx := context.Background()
	for _, tool := range []string{"search_kb", "refund_payment", "delete_account", "issue_refund"} {
		vb := aBase.Adjudicate(ctx, roCall(tool))
		vd := aDial.Adjudicate(ctx, roCall(tool))
		if vb.Kind != vd.Kind {
			t.Fatalf("tool %q: floor differs with vs without the isolation dial (%v vs %v) — isolation must stay orthogonal to the #2018 floor",
				tool, vb.Kind, vd.Kind)
		}
	}
}
