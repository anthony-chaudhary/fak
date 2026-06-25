package adjudicator

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

// TestMustRunFloorInvariant pins which rungs a RungProfile may elide: the four
// write-shaped refusal rungs are mandatory ONLY for the write class; everything else
// (by-name deny, arg predicates, transform, the terminals) is mandatory for EVERY
// class. This is the floor SetPolicy enforces.
func TestMustRunFloorInvariant(t *testing.T) {
	writeOnly := map[rung]bool{
		rungSelfModify: true, rungCmdSelfModify: true, rungSynthTool: true, rungLintWrite: true,
	}
	for r := rung(0); r < numRungs; r++ {
		// write class: every rung is mandatory.
		if !mustRun(classWrite, r) {
			t.Errorf("mustRun(classWrite, rung %d) = false, want true (write floor is total)", r)
		}
		// read class: only the non-write-only rungs are mandatory.
		want := !writeOnly[r]
		if got := mustRun(classRead, r); got != want {
			t.Errorf("mustRun(classRead, rung %d) = %v, want %v", r, got, want)
		}
	}
}

// TestSanitizeProfileClampsWriteFloorButKeepsReadElision proves the SetPolicy clamp:
// a profile that tries to elide a mandatory write-class refusal rung has that drop
// REJECTED (the rung still runs), while a legal read-class elision survives. The
// floor narrows only, never widens.
func TestSanitizeProfileClampsWriteFloorButKeepsReadElision(t *testing.T) {
	// Hand-construct an over-reaching profile: drop self-modify for BOTH classes.
	pr := (&RungProfile{}).
		elide(classWrite, rungSelfModify, rungCmdSelfModify). // illegal: write floor
		elide(classRead, rungSelfModify, rungCmdSelfModify, rungSynthTool, rungLintWrite)

	a := New(Policy{Profile: pr})
	got := a.policy.Profile

	// Write class: the mandatory refusal rungs were clamped back ON.
	for _, r := range []rung{rungSelfModify, rungCmdSelfModify, rungSynthTool, rungLintWrite} {
		if !got.runs(classWrite, r) {
			t.Errorf("write class: rung %d was elided but must run after sanitize", r)
		}
	}
	// Read class: the legal write-only elisions SURVIVE.
	for _, r := range []rung{rungSelfModify, rungCmdSelfModify, rungSynthTool, rungLintWrite} {
		if got.runs(classRead, r) {
			t.Errorf("read class: rung %d should stay elided after sanitize", r)
		}
	}
}

// TestSanitizeProfileNilStaysNil keeps the drop-in guarantee: a nil profile is left
// nil (run everything), so the zero Policy and DefaultPolicy are unchanged.
func TestSanitizeProfileNilStaysNil(t *testing.T) {
	if sanitizeProfile(nil) != nil {
		t.Fatal("sanitizeProfile(nil) must stay nil")
	}
	a := New(DefaultPolicy())
	if a.policy.Profile != nil {
		t.Fatalf("DefaultPolicy must carry a nil Profile, got %+v", a.policy.Profile)
	}
}

// TestRunsNilProfileRunsEveryRung is the byte-identity anchor for #666: a nil profile
// runs every rung for every class.
func TestRunsNilProfileRunsEveryRung(t *testing.T) {
	var pr *RungProfile // nil
	for cl := class(0); int(cl) < numClasses; cl++ {
		for r := rung(0); r < numRungs; r++ {
			if !pr.runs(cl, r) {
				t.Fatalf("nil profile must run rung %d for class %d", r, cl)
			}
		}
	}
}

// TestSetPolicyAppliesProfileSanitize confirms SetPolicy (not just New) clamps an
// over-reaching profile on the live floor.
func TestSetPolicyAppliesProfileSanitize(t *testing.T) {
	a := New(Policy{Allow: map[string]bool{"x": true}})
	a.SetPolicy(Policy{
		Allow:   map[string]bool{"x": true},
		Profile: (&RungProfile{}).elide(classWrite, rungLintWrite),
	})
	if !a.policy.Profile.runs(classWrite, rungLintWrite) {
		t.Fatal("SetPolicy must clamp an illegal write-class elision")
	}
	// And the clamped profile changes no verdict: an allowed tool still allows.
	if v := a.Adjudicate(context.Background(), inlineCall("x", `{}`)); v.Kind != abi.VerdictAllow {
		t.Fatalf("clamped profile changed a verdict: got %v", v.Kind)
	}
}
