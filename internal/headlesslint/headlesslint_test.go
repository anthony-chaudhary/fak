package headlesslint

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
)

// TestFixture is the corpus witness: every labeled sample must scan to its
// declared verdict and Classes. This is what `fak headless-lint --self-test`
// re-derives at the CLI.
func TestFixture(t *testing.T) {
	cases, passed := RunFixture()
	if passed != len(cases) {
		for _, c := range cases {
			if !c.OK {
				t.Errorf("fixture %q: expect_clean=%v got=%s classes=%v", c.Name, c.Clean, c.Got, c.Classes)
			}
		}
	}
}

func TestScanClassifies(t *testing.T) {
	tests := []struct {
		name string
		text string
		want Class
	}{
		{"permission", "Do you want me to push it?", PermissionAsk},
		{"shall-i", "Shall I open the PR?", PermissionAsk},
		{"preference", "Which would you prefer, A or B?", PreferenceAsk},
		{"clarify", "Could you clarify what you meant?", ClarificationRequest},
		{"review", "Please review the changes.", ReviewRequest},
		{"wait", "I'll wait for your confirmation.", ConfirmationWait},
		{"todo", "TODO: handle the timeout later.", DeferredWork},
		{"suggest", "You may want to add caching.", SuggestionPunt},
		{"offer", "Let me know if you want more.", OpenOffer},
		{"surrender-giving-up", "I am giving up on this task.", PrematureSurrender},
		{"surrender-cannot-complete", "I cannot complete the goal.", PrematureSurrender},
		{"surrender-unable-proceed", "I am unable to proceed.", PrematureSurrender},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rep := Scan(tc.text)
			if rep.Classes[tc.want] == 0 {
				t.Fatalf("text %q: want class %s, got %+v", tc.text, tc.want, rep.Classes)
			}
		})
	}
}

// TestAuthorityEarnsHumanResidual: an authority-bearing permission ask folds to
// HUMAN_RESIDUAL (a routed escalation), while an ordinary one is TAKE_OBVIOUS —
// the "earned" override is inherited from choicetriage, not reinvented.
func TestAuthorityEarnsHumanResidual(t *testing.T) {
	auth := Scan("Should I publish the release?")
	if len(auth.Findings) == 0 {
		t.Fatal("expected a finding for an authority-bearing permission ask")
	}
	f := auth.Findings[0]
	if f.Disposition != choicetriage.HumanResidual || !f.NeedsHuman {
		t.Errorf("authority ask: want HUMAN_RESIDUAL/needs-human, got %s needsHuman=%v", f.Disposition, f.NeedsHuman)
	}

	ord := Scan("Should I push the branch?")
	if len(ord.Findings) == 0 {
		t.Fatal("expected a finding for an ordinary permission ask")
	}
	if g := ord.Findings[0]; g.Disposition != choicetriage.TakeObvious || g.NeedsHuman {
		t.Errorf("ordinary ask: want TAKE_OBVIOUS/no-human, got %s needsHuman=%v", g.Disposition, g.NeedsHuman)
	}
}

// TestTicketedDeferralSuppressed: a punt that already cites a ticket is scoping,
// not a bare deferral, so it does not flag.
func TestTicketedDeferralSuppressed(t *testing.T) {
	if rep := Scan("TODO(#901): handle the timeout in the follow-up."); rep.Verdict != Clean {
		t.Errorf("ticketed TODO should be clean, got %s (%+v)", rep.Verdict, rep.Classes)
	}
	if rep := Scan("TODO: handle the timeout."); rep.Verdict != OperatorDirected {
		t.Errorf("bare TODO should flag, got %s", rep.Verdict)
	}
}

func TestCleanOutputHasNoFindings(t *testing.T) {
	rep := Scan("Implemented the parser, committed abc123, tests pass, pushed.")
	if rep.Verdict != Clean || rep.Count != 0 {
		t.Errorf("clean output: want clean/0, got %s/%d %+v", rep.Verdict, rep.Count, rep.Findings)
	}
}
