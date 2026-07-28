package syspromptmmu

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// structural_reason_test.go — the #5442 witness, driven through the two REAL consumers of
// promptmmu.ArraySplicePoints (SpliceSystemOverlay and AuditRealizedPrefix), never through
// the helper itself. A helper-only test would assert that the new reason exists without
// establishing that any consumer benefits from it — the standing gap #5435 tracks. Removing
// the split in promptmmu reds both tests below.

// TestAuditSeparatesUnreadableFromAbsent is the acceptance for #5442 at the Rung-6 auditor:
// a body that could not be READ and a body that is merely not a fak base context used to
// share one verdict (AuditAbsent), so a decode regression showed up as fewer AuditOK rather
// than as an error. AuditAbsent is the expected-large passthrough bucket; a structural
// failure counted there raises no suspicion.
func TestAuditSeparatesUnreadableFromAbsent(t *testing.T) {
	plan := BaseContextPlan()

	// NON-CANDIDATE: a well-formed harness system[] that simply carries no cache_control
	// anchor. The ordinary, high-volume shape. Must stay neutral.
	noAnchor := bodyWith(t, []byte(`[{"type":"text","text":"harness rule"}]`), nil)
	// STRUCTURAL: the body is not a JSON object at all.
	malformed := []byte(`not json at all`)
	// STRUCTURAL: system[] IS an array whose element spans decode (the breakpoint is even
	// found), but its elements are not text blocks. Reachable on live wire bytes.
	badBlocks := bodyWith(t, []byte(`[{"type":"text","text":"a","cache_control":{"type":"ephemeral"}},5]`), nil)
	// NON-CANDIDATE: `system` is present but holds no array. A legitimate wire shape, so it
	// must NOT be reported as a decode failure.
	nonArray := bodyWith(t, []byte(`null`), nil)

	gotNoAnchor := AuditRealizedPrefix(noAnchor, plan)
	gotMalformed := AuditRealizedPrefix(malformed, plan)
	gotBadBlocks := AuditRealizedPrefix(badBlocks, plan)
	gotNonArray := AuditRealizedPrefix(nonArray, plan)

	if gotNoAnchor.Status != AuditAbsent {
		t.Errorf("no-anchor body: status = %q, want %q (the benign non-candidate)", gotNoAnchor.Status, AuditAbsent)
	}
	if gotNonArray.Status != AuditAbsent {
		t.Errorf("non-array system: status = %q, want %q (a legitimate shape, not a decode failure)", gotNonArray.Status, AuditAbsent)
	}
	if gotMalformed.Status != AuditUnreadable {
		t.Errorf("malformed body: status = %q, want %q (structural)", gotMalformed.Status, AuditUnreadable)
	}
	if gotBadBlocks.Status != AuditUnreadable {
		t.Errorf("undecodable system blocks: status = %q, want %q (structural)", gotBadBlocks.Status, AuditUnreadable)
	}

	// THE acceptance: a malformed body and a body with no cache_control anchor must not
	// share one observable outcome at the real call site.
	if gotMalformed.Status == gotNoAnchor.Status {
		t.Fatalf("a malformed body and a no-breakpoint body collapsed into one status %q — "+
			"a decode regression would hide in the benign bucket", gotMalformed.Status)
	}

	// A structural failure is not the spine alarm either: nothing is known to have mutated,
	// so it must not raise Diverged or claim a fak base context is Present.
	for name, a := range map[string]PrefixAudit{"malformed": gotMalformed, "bad-blocks": gotBadBlocks} {
		if a.Present {
			t.Errorf("%s: an unreadable body must not report Present", name)
		}
		if a.Diverged {
			t.Errorf("%s: an unreadable body must not raise the spine-divergence alarm", name)
		}
		if a.BreakIdx != -1 {
			t.Errorf("%s: BreakIdx = %d, want -1", name, a.BreakIdx)
		}
	}
}

// TestSpliceSeparatesNonArrayFromNoBreakpoint is the acceptance for #5442 at the Rung-2
// splicer. Before the split the splicer re-derived the reason itself by unmarshalling the
// `system` value into a []json.RawMessage — and json.Unmarshal of JSON null into a slice
// SUCCEEDS, so `"system": null` fell through that probe and was reported as
// SkipNoBreakpoint: "there was a system array, it just had no anchor." There was no array.
func TestSpliceSeparatesNonArrayFromNoBreakpoint(t *testing.T) {
	plan := BaseContextPlan()
	ov := []cachemeta.PromptSegment{overlaySeg("x")}

	nullSys := SpliceSystemOverlay(bodyWith(t, []byte(`null`), nil), plan, ov, decodeOK)
	noAnchor := SpliceSystemOverlay(bodyWith(t, []byte(`[{"type":"text","text":"foo"}]`), nil), plan, ov, decodeOK)
	malformed := SpliceSystemOverlay([]byte(`not json at all`), plan, ov, decodeOK)
	badBlocks := SpliceSystemOverlay(
		bodyWith(t, []byte(`[{"type":"text","text":"a","cache_control":{"type":"ephemeral"}},5]`), nil), plan, ov, decodeOK)

	for name, res := range map[string]SpliceResult{
		"null-system": nullSys, "no-anchor": noAnchor, "malformed": malformed, "bad-blocks": badBlocks,
	} {
		if res.Changed {
			t.Fatalf("%s: expected fail-safe identity, got a splice", name)
		}
	}

	if nullSys.SkipReason != SkipNoSystemArray {
		t.Errorf("`system: null`: SkipReason = %q, want %q — there is no array, so it cannot be "+
			"an array that merely lacks an anchor", nullSys.SkipReason, SkipNoSystemArray)
	}
	if noAnchor.SkipReason != SkipNoBreakpoint {
		t.Errorf("no-anchor body: SkipReason = %q, want %q", noAnchor.SkipReason, SkipNoBreakpoint)
	}
	if malformed.SkipReason != SkipNotJSONObject {
		t.Errorf("malformed body: SkipReason = %q, want %q", malformed.SkipReason, SkipNotJSONObject)
	}
	if badBlocks.SkipReason != SkipUndecodableSys {
		t.Errorf("undecodable system blocks: SkipReason = %q, want %q (structural)", badBlocks.SkipReason, SkipUndecodableSys)
	}

	// The acceptance: the structural and the not-an-array outcomes each stay distinct from
	// the benign no-anchor idle they used to be folded into.
	if nullSys.SkipReason == noAnchor.SkipReason {
		t.Fatalf("`system: null` and a no-breakpoint array collapsed into one reason %q", nullSys.SkipReason)
	}
	if malformed.SkipReason == noAnchor.SkipReason {
		t.Fatalf("a malformed body and a no-breakpoint array collapsed into one reason %q", malformed.SkipReason)
	}
	if badBlocks.SkipReason == noAnchor.SkipReason {
		t.Fatalf("an undecodable system[] and a no-breakpoint array collapsed into one reason %q", badBlocks.SkipReason)
	}
}
