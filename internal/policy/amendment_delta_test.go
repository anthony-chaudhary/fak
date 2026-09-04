package policy

import (
	"regexp"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// A pure-RATCHET delta — added deny, added arg rule, added secret pattern,
// added block host, narrowed allow, stricter posture — classifies tighten with
// nothing in the gated or frozen buckets (#5177).
func TestDiffAmendmentPureTightenClassifiesTighten(t *testing.T) {
	old := adjudicator.Policy{
		Posture:     adjudicator.PostureAdmitAndLog,
		Allow:       map[string]bool{"kept": true, "dropped": true},
		AllowPrefix: []string{"read_", "dropped_"},
		Deny:        map[string]abi.ReasonCode{"kept-deny": abi.ReasonPolicyBlock},
	}
	next := adjudicator.Policy{
		Posture:     adjudicator.PostureFailClosed,
		Allow:       map[string]bool{"kept": true},
		AllowPrefix: []string{"read_"},
		Deny: map[string]abi.ReasonCode{
			"kept-deny": abi.ReasonPolicyBlock,
			"danger":    abi.ReasonPolicyBlock,
		},
		ArgPredicates:    []adjudicator.ArgPredicate{{Tool: "write_file", Arg: "path", Glob: "./out/**", Reason: abi.ReasonPolicyBlock}},
		SecretPatterns:   []*regexp.Regexp{regexp.MustCompile(`sk-[a-z0-9]{20}`)},
		EgressBlockHosts: []string{"ads.example.com"},
	}
	d := DiffAmendment(old, next)
	if got := d.Class(); got != AmendmentTighten {
		t.Fatalf("Class() = %q, want %q (delta=%+v)", got, AmendmentTighten, d)
	}
	if len(d.Widen) != 0 || len(d.Frozen) != 0 {
		t.Fatalf("pure tighten leaked into gated buckets: widen=%v frozen=%v", d.Widen, d.Frozen)
	}
	got := FormatAmendmentChanges(d.Tighten)
	want := "removed_allow=dropped; removed_allow_prefix=dropped_; added_deny=danger; posture=admit_and_log->fail_closed; added_arg_rules=write_file:path; added_secret_patterns=sk-[a-z0-9]{20}; added_block_hosts=ads.example.com"
	if got != want {
		t.Fatalf("tighten rendering = %q, want %q", got, want)
	}
}

// The widen bucket carries exactly the legacy widening set with the legacy
// labels, so the reload gate's refusal message stays recognizable.
func TestDiffAmendmentWidenBucketMatchesLegacyRendering(t *testing.T) {
	old := adjudicator.Policy{
		Posture:         adjudicator.PostureFailClosed,
		Allow:           map[string]bool{"kept": true},
		AllowPrefix:     []string{"read_"},
		Deny:            map[string]abi.ReasonCode{"kept-deny": abi.ReasonPolicyBlock, "removed-deny": abi.ReasonPolicyBlock},
		SelfModifyGlobs: []string{"kept/**", "removed/**"},
	}
	next := adjudicator.Policy{
		Posture:         adjudicator.PostureAdmitAndLog,
		Allow:           map[string]bool{"kept": true, "z-new": true, "a-new": true},
		AllowPrefix:     []string{"read_", "write_"},
		Deny:            map[string]abi.ReasonCode{"kept-deny": abi.ReasonDefaultDeny},
		SelfModifyGlobs: []string{"kept/**"},
	}
	d := DiffAmendment(old, next)
	if got := d.Class(); got != AmendmentWiden {
		t.Fatalf("Class() = %q, want %q", got, AmendmentWiden)
	}
	got := FormatAmendmentChanges(d.Widen)
	want := "added_allow=a-new,z-new; added_allow_prefix=write_; removed_deny=removed-deny; removed_self_modify_globs=removed/**; posture=fail_closed->admit_and_log"
	if got != want {
		t.Fatalf("widen rendering = %q, want %q", got, want)
	}
}

// Removing a tighten-side element (a secret pattern, an arg rule, a block
// host) is a widen, not a free pass — the incidental hole #5177 closes.
func TestDiffAmendmentRemovedRatchetElementsAreWiden(t *testing.T) {
	old := adjudicator.Policy{
		ArgPredicates:    []adjudicator.ArgPredicate{{Tool: "write_file", Arg: "path", Glob: "./out/**"}},
		SecretPatterns:   []*regexp.Regexp{regexp.MustCompile(`sk-[a-z0-9]{20}`)},
		EgressBlockHosts: []string{"ads.example.com"},
	}
	d := DiffAmendment(old, adjudicator.Policy{})
	if got := d.Class(); got != AmendmentWiden {
		t.Fatalf("Class() = %q, want %q (delta=%+v)", got, AmendmentWiden, d)
	}
	got := FormatAmendmentChanges(d.Widen)
	want := "removed_arg_rules=write_file:path; removed_secret_patterns=sk-[a-z0-9]{20}; removed_block_hosts=ads.example.com"
	if got != want {
		t.Fatalf("widen rendering = %q, want %q", got, want)
	}
}

// An edited arg rule diffs as removed+added, so the removal keeps the delta
// gated instead of letting a loosened matcher ride an "addition".
func TestDiffAmendmentEditedArgRuleStaysGated(t *testing.T) {
	old := adjudicator.Policy{ArgPredicates: []adjudicator.ArgPredicate{{Tool: "write_file", Arg: "path", Glob: "./out/**"}}}
	next := adjudicator.Policy{ArgPredicates: []adjudicator.ArgPredicate{{Tool: "write_file", Arg: "path", Glob: "./**"}}}
	d := DiffAmendment(old, next)
	if got := d.Class(); got != AmendmentWiden {
		t.Fatalf("edited rule Class() = %q, want %q (delta=%+v)", got, AmendmentWiden, d)
	}
	if len(d.Widen) != 1 || len(d.Tighten) != 1 {
		t.Fatalf("edited rule should read removed+added: widen=%v tighten=%v", d.Widen, d.Tighten)
	}
}

// A field the registry does not classify routes to the frozen bucket — the
// gate fails closed on an unlabeled surface — and any frozen violation
// dominates the folded class.
func TestDiffAmendmentUnclassifiedFieldFailsClosedFrozen(t *testing.T) {
	var d AmendmentDelta
	d.route(false, AmendmentChange{Field: "NotARegisteredKnob", Label: "mystery", New: "x"})
	if len(d.Frozen) != 1 || d.Class() != AmendmentFrozenViolation {
		t.Fatalf("unclassified field not fail-closed: %+v class=%q", d, d.Class())
	}
	d.Widen = append(d.Widen, AmendmentChange{Field: "Allow", Label: "added_allow", New: "y"})
	d.Tighten = append(d.Tighten, AmendmentChange{Field: "Deny", Label: "added_deny", New: "z"})
	if got := d.Class(); got != AmendmentFrozenViolation {
		t.Fatalf("frozen violation must dominate, got %q", got)
	}
}

// The folded class ranks widen over tighten, and an identical policy is none.
func TestAmendmentDeltaClassSeverity(t *testing.T) {
	old := adjudicator.Policy{Deny: map[string]abi.ReasonCode{"kept": abi.ReasonPolicyBlock}}
	if got := DiffAmendment(old, old).Class(); got != AmendmentNone {
		t.Fatalf("identical policies Class() = %q, want %q", got, AmendmentNone)
	}
	mixed := DiffAmendment(
		adjudicator.Policy{},
		adjudicator.Policy{
			Allow: map[string]bool{"new-allow": true},
			Deny:  map[string]abi.ReasonCode{"new-deny": abi.ReasonPolicyBlock},
		},
	)
	if got := mixed.Class(); got != AmendmentWiden {
		t.Fatalf("mixed delta Class() = %q, want %q", got, AmendmentWiden)
	}
}

func TestDiffAmendmentPostureStrictnessTransitions(t *testing.T) {
	cases := []struct {
		name       string
		old        adjudicator.Posture
		next       adjudicator.Posture
		wantClass  string
		wantChange string
	}{
		{
			name:       "fail_closed to admit_and_log widens",
			old:        adjudicator.PostureFailClosed,
			next:       adjudicator.PostureAdmitAndLog,
			wantClass:  AmendmentWiden,
			wantChange: "posture=fail_closed->admit_and_log",
		},
		{
			name:       "admit_and_log to default_open widens",
			old:        adjudicator.PostureAdmitAndLog,
			next:       adjudicator.PostureDefaultOpen,
			wantClass:  AmendmentWiden,
			wantChange: "posture=admit_and_log->default_open",
		},
		{
			name:       "fail_closed to default_open widens",
			old:        adjudicator.PostureFailClosed,
			next:       adjudicator.PostureDefaultOpen,
			wantClass:  AmendmentWiden,
			wantChange: "posture=fail_closed->default_open",
		},
		{
			name:       "default_open to admit_and_log tightens",
			old:        adjudicator.PostureDefaultOpen,
			next:       adjudicator.PostureAdmitAndLog,
			wantClass:  AmendmentTighten,
			wantChange: "posture=default_open->admit_and_log",
		},
		{
			name:       "admit_and_log to fail_closed tightens",
			old:        adjudicator.PostureAdmitAndLog,
			next:       adjudicator.PostureFailClosed,
			wantClass:  AmendmentTighten,
			wantChange: "posture=admit_and_log->fail_closed",
		},
		{
			name:       "default_open to fail_closed tightens",
			old:        adjudicator.PostureDefaultOpen,
			next:       adjudicator.PostureFailClosed,
			wantClass:  AmendmentTighten,
			wantChange: "posture=default_open->fail_closed",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := DiffAmendment(adjudicator.Policy{Posture: tc.old}, adjudicator.Policy{Posture: tc.next})
			if d.Class() != tc.wantClass {
				t.Fatalf("Class() = %q, want %q", d.Class(), tc.wantClass)
			}
			var changes []AmendmentChange
			if tc.wantClass == AmendmentWiden {
				changes = d.Widen
			} else {
				changes = d.Tighten
			}
			got := FormatAmendmentChanges(changes)
			if got != tc.wantChange {
				t.Fatalf("FormatAmendmentChanges = %q, want %q", got, tc.wantChange)
			}
		})
	}
}
