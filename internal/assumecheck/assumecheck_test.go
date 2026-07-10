package assumecheck

import (
	"errors"
	"strings"
	"testing"
)

// declared is a well-formed assumption for the table: ledger-read witnessed, so a
// matching-Evidence row exercises the mainline and mismatches exercise rule 1.
var declared = Assumption{
	ID:            "test-assumption",
	Owner:         "test",
	Statement:     "the tested condition holds",
	Level:         LevelSubsystem,
	WitnessKind:   WitnessLedgerRead,
	RefusalReason: "TEST_REFUSAL",
}

// selfReported declares session-report as its witness so the self-report rungs
// (rule 4: cannot confirm; rule 5: admission refutes) are reachable without a kind
// mismatch shadowing them.
var selfReported = Assumption{
	ID:          "test-self-report",
	Owner:       "test",
	Statement:   "the agent says it is fine",
	Level:       LevelSession,
	WitnessKind: WitnessSessionReport,
}

// TestCheckAllOutcomeBranches proves Check produces EXACTLY the documented outcome on
// every branch of its decision order, including the explicit UNVERIFIABLE and STALE
// branches — never a silent no-decision, and every result a member of the closed set.
func TestCheckAllOutcomeBranches(t *testing.T) {
	cases := []struct {
		name       string
		a          Assumption
		ev         Evidence
		want       Outcome
		wantReason string // substring the self-describing reason must carry
	}{
		{
			name: "holds: declared witness ran, fresh, confirms",
			a:    declared,
			ev:   Evidence{Kind: WitnessLedgerRead, Witnessed: true, Holds: true, Detail: "seat in pool"},
			want: OutcomeHolds, wantReason: "confirmed",
		},
		{
			name: "holds: within a declared freshness bound",
			a:    declared,
			ev:   Evidence{Kind: WitnessLedgerRead, Witnessed: true, Holds: true, AgeSeconds: 5, MaxAgeSeconds: 60},
			want: OutcomeHolds, wantReason: "confirmed",
		},
		{
			name: "violated: declared witness ran and refuted",
			a:    declared,
			ev:   Evidence{Kind: WitnessLedgerRead, Witnessed: true, Holds: false, Detail: "seat excluded: unservable"},
			want: OutcomeViolated, wantReason: "refuted",
		},
		{
			name: "violated: a session-report ADMITTING violation refutes (credible against interest)",
			a:    selfReported,
			ev:   Evidence{Kind: WitnessSessionReport, Witnessed: true, Holds: false},
			want: OutcomeViolated, wantReason: "refuted",
		},
		{
			name: "unverifiable: witness kind mismatch is never judged",
			a:    declared,
			ev:   Evidence{Kind: WitnessCommandProbe, Witnessed: true, Holds: true},
			want: OutcomeUnverifiable, wantReason: "cross-witness",
		},
		{
			name: "unverifiable: unknown evidence kind fails closed",
			a:    declared,
			ev:   Evidence{Kind: WitnessKind("vibes"), Witnessed: true, Holds: true},
			want: OutcomeUnverifiable, wantReason: "cross-witness",
		},
		{
			name: "unverifiable: assumption with an unset witness kind fails closed",
			a:    Assumption{ID: "no-witness", Level: LevelLoop},
			ev:   Evidence{Kind: WitnessLedgerRead, Witnessed: true, Holds: true},
			want: OutcomeUnverifiable, wantReason: "cross-witness",
		},
		{
			name: "unverifiable: the witness could not produce a decision",
			a:    declared,
			ev:   Evidence{Kind: WitnessLedgerRead, Witnessed: false, Detail: "registry unreadable"},
			want: OutcomeUnverifiable, wantReason: "could not produce a decision",
		},
		{
			name: "unverifiable: a session self-report cannot positively confirm",
			a:    selfReported,
			ev:   Evidence{Kind: WitnessSessionReport, Witnessed: true, Holds: true},
			want: OutcomeUnverifiable, wantReason: "self-report",
		},
		{
			name: "stale: evidence aged past its freshness bound, even when it confirms",
			a:    declared,
			ev:   Evidence{Kind: WitnessLedgerRead, Witnessed: true, Holds: true, AgeSeconds: 120, MaxAgeSeconds: 60},
			want: OutcomeStale, wantReason: "freshness bound",
		},
		{
			name: "stale: a lapsed refutation is also stale, not violated (re-witness first)",
			a:    declared,
			ev:   Evidence{Kind: WitnessLedgerRead, Witnessed: true, Holds: false, AgeSeconds: 120, MaxAgeSeconds: 60},
			want: OutcomeStale, wantReason: "freshness bound",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := Check(tc.a, tc.ev)
			if v.Outcome != tc.want {
				t.Fatalf("Check outcome = %s, want %s (reason: %s)", v.Outcome, tc.want, v.Reason)
			}
			if !ValidOutcome(v.Outcome) {
				t.Fatalf("Check produced an outcome outside the closed set: %q", v.Outcome)
			}
			if v.Reason == "" {
				t.Fatal("Check produced an empty reason: verdicts must be self-describing")
			}
			if !strings.Contains(v.Reason, tc.wantReason) {
				t.Fatalf("Check reason %q does not carry %q", v.Reason, tc.wantReason)
			}
			if v.AssumptionID != tc.a.ID || v.Level != tc.a.Level {
				t.Fatalf("verdict identity = (%s,%s), want (%s,%s)", v.AssumptionID, v.Level, tc.a.ID, tc.a.Level)
			}
		})
	}
}

// TestGuardAssumptionFailClosed proves the hard gate: HOLDS is the ONLY outcome with
// a nil error; violated, unverifiable, and stale all return the typed
// *AssumptionViolationError, branchable via errors.Is/errors.As, carrying the closed
// Verdict and the OUTCOME-CLASS refusal token (#3822 C4) — with the per-assumption
// label folded into the verdict's reason so it is not lost.
func TestGuardAssumptionFailClosed(t *testing.T) {
	holds := Evidence{Kind: WitnessLedgerRead, Witnessed: true, Holds: true}
	if v, err := GuardAssumption(declared, holds); err != nil {
		t.Fatalf("GuardAssumption on holding evidence: unexpected error %v", err)
	} else if v.Outcome != OutcomeHolds {
		t.Fatalf("GuardAssumption verdict = %s, want %s", v.Outcome, OutcomeHolds)
	}

	blocking := []struct {
		name string
		ev   Evidence
		want Outcome
	}{
		{"violated", Evidence{Kind: WitnessLedgerRead, Witnessed: true, Holds: false}, OutcomeViolated},
		{"unverifiable", Evidence{Kind: WitnessLedgerRead, Witnessed: false}, OutcomeUnverifiable},
		{"stale", Evidence{Kind: WitnessLedgerRead, Witnessed: true, Holds: true, AgeSeconds: 2, MaxAgeSeconds: 1}, OutcomeStale},
	}
	for _, tc := range blocking {
		t.Run(tc.name, func(t *testing.T) {
			v, err := GuardAssumption(declared, tc.ev)
			if v.Outcome != tc.want {
				t.Fatalf("verdict = %s, want %s", v.Outcome, tc.want)
			}
			if err == nil {
				t.Fatalf("GuardAssumption(%s) returned nil error: the gate must fail closed", tc.want)
			}
			if !errors.Is(err, ErrAssumptionViolated) {
				t.Fatalf("errors.Is(err, ErrAssumptionViolated) = false for %v", err)
			}
			var ave *AssumptionViolationError
			if !errors.As(err, &ave) {
				t.Fatalf("errors.As to *AssumptionViolationError failed for %v", err)
			}
			if ave.Verdict.Outcome != tc.want {
				t.Fatalf("typed error carries outcome %s, want %s", ave.Verdict.Outcome, tc.want)
			}
			if want := tc.want.RefusalReason(); ave.RefusalReason != want {
				t.Fatalf("typed error refusal token = %q, want the outcome-class token %q", ave.RefusalReason, want)
			}
			if !strings.Contains(ave.Verdict.Reason, declared.RefusalReason) {
				t.Fatalf("verdict reason %q lost the per-assumption label %q", ave.Verdict.Reason, declared.RefusalReason)
			}
		})
	}
}

// TestClosedVocabularies proves membership and the fail-closed String() rendering for
// all three enums, so a corrupt/foreign value can never masquerade as a member.
func TestClosedVocabularies(t *testing.T) {
	for l := range validLevels {
		if !ValidLevel(l) || l.String() != string(l) {
			t.Fatalf("level %q must be valid and render as itself", l)
		}
	}
	for k := range validWitnessKinds {
		if !ValidWitnessKind(k) || k.String() != string(k) {
			t.Fatalf("witness kind %q must be valid and render as itself", k)
		}
	}
	for o := range validOutcomes {
		if !ValidOutcome(o) || o.String() != string(o) {
			t.Fatalf("outcome %q must be valid and render as itself", o)
		}
	}
	if ValidLevel("galactic") || ValidWitnessKind("vibes") || ValidOutcome("MAYBE") {
		t.Fatal("foreign values must not be members of the closed vocabularies")
	}
	if got := Level("galactic").String(); got != "unknown(galactic)" {
		t.Fatalf("foreign Level renders %q, want fail-closed unknown(...)", got)
	}
	if got := WitnessKind("").String(); got != "(unset)" {
		t.Fatalf("empty WitnessKind renders %q, want (unset)", got)
	}
	if got := Outcome("MAYBE").String(); got != "unknown(MAYBE)" {
		t.Fatalf("foreign Outcome renders %q, want fail-closed unknown(...)", got)
	}
}

// TestSeatLaunchableDeclaration pins the spine's one real assumption to the closed
// vocabularies, so the C2 registry inherits a valid first row.
func TestSeatLaunchableDeclaration(t *testing.T) {
	a := SeatLaunchable
	if a.ID != "seat-launchable" {
		t.Fatalf("SeatLaunchable.ID = %q", a.ID)
	}
	if !ValidLevel(a.Level) || !ValidWitnessKind(a.WitnessKind) {
		t.Fatalf("SeatLaunchable declares out-of-vocabulary level/witness: %s/%s", a.Level, a.WitnessKind)
	}
	if a.RefusalReason == "" || a.Statement == "" || a.Owner == "" {
		t.Fatal("SeatLaunchable must carry a refusal token, a statement, and an owner")
	}
}
