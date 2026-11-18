package guard

import (
	"testing"
)

func TestNomosReview(t *testing.T) {
	t.Run("allow on benign read action", func(t *testing.T) {
		v := ReviewAction(ProposedAction{Tool: "read_file", Category: CategoryRead, Scope: "repo"})
		if v.Verdict != VerdictAllow {
			t.Fatalf("expected ALLOW for benign read, got %q", v.Verdict)
		}
		if v.Reason != ReasonBenign {
			t.Errorf("expected reason %s, got %q", ReasonBenign, v.Reason)
		}
		if v.Category != CategoryRead {
			t.Errorf("expected category READ, got %q", v.Category)
		}
		if v.Render() != "ALLOW:BENIGN" {
			t.Errorf("expected grammar %q, got %q", "ALLOW:BENIGN", v.Render())
		}
	})

	t.Run("allow on in-scope write and execute", func(t *testing.T) {
		for _, c := range []ActionCategory{CategoryWrite, CategoryExecute} {
			v := ReviewAction(ProposedAction{Tool: "edit", Category: c})
			if v.Verdict != VerdictAllow || v.Reason != ReasonBenign {
				t.Errorf("category %s: expected ALLOW:BENIGN, got %s", c, v.Render())
			}
		}
	})

	t.Run("deny on overeager destructive action", func(t *testing.T) {
		v := ReviewAction(ProposedAction{Tool: "rm", Category: CategoryDestructive, Scope: "rm -rf /"})
		if v.Verdict != VerdictDeny {
			t.Fatalf("expected DENY for destructive action, got %q", v.Verdict)
		}
		if v.Reason != ReasonOvereager {
			t.Errorf("expected reason %s, got %q", ReasonOvereager, v.Reason)
		}
		if v.Render() != "DENY:OVEREAGER" {
			t.Errorf("expected grammar %q, got %q", "DENY:OVEREAGER", v.Render())
		}
	})

	t.Run("deny on overeager privilege and exfiltration", func(t *testing.T) {
		for _, c := range []ActionCategory{CategoryPrivilege, CategoryExfiltration} {
			v := ReviewAction(ProposedAction{Tool: "shell", Category: c})
			if v.Verdict != VerdictDeny || v.Reason != ReasonOvereager {
				t.Errorf("category %s: expected DENY:OVEREAGER, got %s", c, v.Render())
			}
		}
	})

	t.Run("fail-closed on unknown category", func(t *testing.T) {
		weird := ActionCategory("TELEPORT")
		if weird.IsKnown() {
			t.Fatalf("TELEPORT must not be a known category")
		}
		v := ReviewAction(ProposedAction{Tool: "mystery", Category: weird})
		if v.Verdict != VerdictDeny {
			t.Fatalf("expected fail-closed DENY for unknown category, got %q", v.Verdict)
		}
		if v.Reason != ReasonUnknownCategory {
			t.Errorf("expected reason %s, got %q", ReasonUnknownCategory, v.Reason)
		}
		if v.Render() != "DENY:UNKNOWN_CATEGORY" {
			t.Errorf("expected grammar %q, got %q", "DENY:UNKNOWN_CATEGORY", v.Render())
		}
	})

	t.Run("verdict grammar round-trips", func(t *testing.T) {
		grammar := []string{"ALLOW:BENIGN", "DENY:OVEREAGER", "DENY:NOT_ALLOWED", "DENY:UNKNOWN_CATEGORY"}
		for _, g := range grammar {
			got, err := ParseVerdict(g)
			if err != nil {
				t.Fatalf("ParseVerdict(%q) error: %v", g, err)
			}
			if got.Render() != g {
				t.Errorf("round-trip %q -> %q", g, got.Render())
			}
		}
	})

	t.Run("verdict grammar rejects malformed strings", func(t *testing.T) {
		bad := []string{
			"ALLOW",              // missing separator
			"ALLOW:BENIGN:EXTRA", // extra segment is tolerated by SplitN but reason "BENIGN:EXTRA" is unknown
			"MAYBE:BENIGN",       // unknown verdict
			"ALLOW:OVEREAGER",    // ALLOW cannot pair with OVEREAGER
			"DENY:BENIGN",        // DENY cannot pair with BENIGN
			"ALLOW:WHATEVER",     // unknown reason
		}
		for _, s := range bad {
			if _, err := ParseVerdict(s); err == nil {
				t.Errorf("ParseVerdict(%q) should have errored", s)
			}
		}
	})

	t.Run("policy can tighten to deny a normally-allowed category", func(t *testing.T) {
		strict := ReviewPolicy{Allowed: map[ActionCategory]struct{}{CategoryRead: {}}}
		v := strict.Review(ProposedAction{Tool: "edit", Category: CategoryWrite})
		if v.Verdict != VerdictDeny || v.Reason != ReasonNotAllowed {
			t.Errorf("expected DENY:NOT_ALLOWED under strict policy, got %s", v.Render())
		}
	})

	t.Run("batch review summarizes per-action verdicts", func(t *testing.T) {
		actions := []ProposedAction{
			{Tool: "read_file", Category: CategoryRead},
			{Tool: "rm", Category: CategoryDestructive},
			{Tool: "mystery", Category: "TELEPORT"},
		}
		res := DefaultReviewPolicy().ReviewBatch(actions)
		if len(res.Verdicts) != 3 {
			t.Fatalf("expected 3 verdicts, got %d", len(res.Verdicts))
		}
		if res.Allowed != 1 || res.Denied != 2 {
			t.Errorf("expected 1 allowed / 2 denied, got %d/%d", res.Allowed, res.Denied)
		}
		if res.AllAllowed {
			t.Errorf("AllAllowed must be false when any action is denied")
		}
		if res.Verdicts[0].Render() != "ALLOW:BENIGN" {
			t.Errorf("verdict[0] = %q, want ALLOW:BENIGN", res.Verdicts[0].Render())
		}
		if res.Verdicts[1].Render() != "DENY:OVEREAGER" {
			t.Errorf("verdict[1] = %q, want DENY:OVEREAGER", res.Verdicts[1].Render())
		}
		if res.Verdicts[2].Render() != "DENY:UNKNOWN_CATEGORY" {
			t.Errorf("verdict[2] = %q, want DENY:UNKNOWN_CATEGORY", res.Verdicts[2].Render())
		}
	})
}
