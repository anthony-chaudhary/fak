package sweep

import (
	"strings"
	"testing"
)

// Ported from tools/resume_relaunch_audit.py's relaunch_verdict / ledger_actions /
// _superset. The load-bearing facts these pin:
//   - RELAUNCHED_OK iff the last real (non-error, non-banner) assistant turn is NEWER
//     than the last error record — strict ISO-8601 string comparison;
//   - a banner-text assistant record counts as an ERROR, never as real progress (a
//     re-capped resume writes the banner as an ordinary assistant turn);
//   - STRANDED is classified by the shared TerminalFailure taxonomy, OTHER when the
//     error text matches no vocabulary;
//   - NEVER_WORKED when no timestamped real turn exists at all;
//   - the superset copy is (latest last-ts, then most records), never file order;
//   - the ledger fold keeps every attempted sid with its sorted distinct actions.

func TestRelaunchVerdictOK(t *testing.T) {
	r := RelaunchVerdict([]Record{
		rec("assistant", "API Error: Overloaded (529)", "", "2026-06-23T10:00:00Z", true),
		rec("assistant", "resumed and finished the port", "", "2026-06-23T12:00:00Z", false),
	})
	if r.Verdict != VerdictRelaunchedOK {
		t.Fatalf("verdict = %s, want RELAUNCHED_OK (%+v)", r.Verdict, r)
	}
	if r.LastRealTS != "2026-06-23T12:00:00Z" || r.LastErrTS != "2026-06-23T10:00:00Z" {
		t.Fatalf("timestamps not carried: %+v", r)
	}
	if r.Kind != "" || r.Evidence != "" {
		t.Fatalf("OK verdict must carry no kind/evidence: %+v", r)
	}
}

func TestRelaunchVerdictOKWithoutAnyError(t *testing.T) {
	r := RelaunchVerdict([]Record{
		rec("assistant", "clean run, no error record", "", "2026-06-23T12:00:00Z", false),
	})
	if r.Verdict != VerdictRelaunchedOK || r.LastErrTS != "" {
		t.Fatalf("no-error transcript must be RELAUNCHED_OK: %+v", r)
	}
}

func TestRelaunchVerdictStrandedKinds(t *testing.T) {
	cases := []struct {
		errText, wantKind string
	}{
		{"You've hit your session limit . resets 6am (America/Los_Angeles)", "LIMIT"},
		{"Not logged in . Please run /login", "AUTH"},
		{"API Error: Overloaded (529) server-side issue", "API_ERR"},
		{"something exploded in an entirely unclassifiable way", "OTHER"},
	}
	for _, c := range cases {
		r := RelaunchVerdict([]Record{
			rec("assistant", "real work happened here", "", "2026-06-23T10:00:00Z", false),
			rec("assistant", c.errText, "", "2026-06-23T11:00:00Z", true),
		})
		if r.Verdict != VerdictStranded || r.Kind != c.wantKind {
			t.Fatalf("errText %q: got (%s, %s), want (STRANDED, %s)", c.errText, r.Verdict, r.Kind, c.wantKind)
		}
		if !strings.Contains(r.Evidence, strings.Fields(c.errText)[0]) {
			t.Fatalf("evidence %q does not carry the error text", r.Evidence)
		}
	}
}

func TestRelaunchBannerProseCountsAsError(t *testing.T) {
	// The banner arrives as an ORDINARY assistant turn (no error channel bit) after the
	// real work: it must land in the error branch, not count as fresh progress.
	r := RelaunchVerdict([]Record{
		rec("assistant", "real work happened here", "", "2026-06-23T10:00:00Z", false),
		rec("assistant", "You've hit your session limit . resets 6am (America/Los_Angeles)", "", "2026-06-23T11:00:00Z", false),
	})
	if r.Verdict != VerdictStranded || r.Kind != "LIMIT" {
		t.Fatalf("banner-as-prose must strand on LIMIT: %+v", r)
	}
}

func TestRelaunchEqualTimestampsStayStranded(t *testing.T) {
	// The rule is strictly NEWER: a real turn at the exact error timestamp proves nothing.
	r := RelaunchVerdict([]Record{
		rec("assistant", "API Error 529", "", "2026-06-23T11:00:00Z", true),
		rec("assistant", "same-second turn", "", "2026-06-23T11:00:00Z", false),
	})
	if r.Verdict != VerdictStranded {
		t.Fatalf("equal timestamps must stay STRANDED, got %s", r.Verdict)
	}
}

func TestRelaunchNeverWorked(t *testing.T) {
	r := RelaunchVerdict([]Record{
		rec("assistant", "API Error 529", "", "2026-06-23T11:00:00Z", true),
	})
	if r.Verdict != VerdictNeverWorked {
		t.Fatalf("verdict = %s, want NEVER_WORKED", r.Verdict)
	}
	if !strings.Contains(r.Evidence, "API Error 529") {
		t.Fatalf("NEVER_WORKED must carry the error evidence: %+v", r)
	}
}

func TestRelaunchNeverWorkedWhenRealTurnLacksTimestamp(t *testing.T) {
	// A real turn with no timestamp cannot anchor the comparison — Python parity: it
	// leaves last_real_ts empty, so the session reads as never having provably worked.
	r := RelaunchVerdict([]Record{
		rec("assistant", "work with no timestamp", "", "", false),
		rec("assistant", "API Error 529", "", "2026-06-23T11:00:00Z", true),
	})
	if r.Verdict != VerdictNeverWorked {
		t.Fatalf("timestamp-less real turn must not count: got %s", r.Verdict)
	}
}

func TestRelaunchOrderRanks(t *testing.T) {
	want := []string{VerdictStranded, VerdictNeverWorked, VerdictNoTranscript, VerdictRelaunchedOK}
	for i := 1; i < len(want); i++ {
		if RelaunchOrder(want[i-1]) >= RelaunchOrder(want[i]) {
			t.Fatalf("order %s !< %s", want[i-1], want[i])
		}
	}
	if RelaunchOrder("BOGUS") <= RelaunchOrder(VerdictRelaunchedOK) {
		t.Fatalf("unknown verdicts must sort last")
	}
}

func TestSupersetIndexPrefersLastTSThenCount(t *testing.T) {
	older := Copy{Path: "a", Records: []Record{
		rec("assistant", "x", "u1", "2026-06-23T10:00:00Z", false),
		rec("assistant", "y", "u2", "2026-06-23T10:05:00Z", false),
	}}
	newer := Copy{Path: "b", Records: []Record{
		rec("assistant", "x", "u1", "2026-06-23T11:00:00Z", false),
	}}
	if got := SupersetIndex([]Copy{older, newer}); got != 1 {
		t.Fatalf("latest last-ts must win: got index %d", got)
	}
	tied := Copy{Path: "c", Records: []Record{
		rec("assistant", "x", "u1", "2026-06-23T11:00:00Z", false),
		rec("assistant", "y", "u2", "2026-06-23T11:00:00Z", false),
	}}
	if got := SupersetIndex([]Copy{newer, tied}); got != 1 {
		t.Fatalf("on a last-ts tie the most records must win: got index %d", got)
	}
	if got := SupersetIndex(nil); got != -1 {
		t.Fatalf("no copies must yield -1, got %d", got)
	}
}

func TestLedgerActionsFold(t *testing.T) {
	ledger := strings.Join([]string{
		`{"session":"sid-1","action":"resume"}`,
		`{"session":"sid-1","action":"resume"}`,
		`{"session":"sid-1","action":"consolidate"}`,
		`{"session":"sid-2"}`,
		`{"no_session":"ignored"}`,
		`not json at all`,
		``,
	}, "\n")
	got := LedgerActions(strings.NewReader(ledger))
	if len(got) != 2 {
		t.Fatalf("want 2 sids, got %v", got)
	}
	if strings.Join(got["sid-1"], ",") != "consolidate,resume" {
		t.Fatalf("sid-1 actions must be sorted+distinct: %v", got["sid-1"])
	}
	if strings.Join(got["sid-2"], ",") != "?" {
		t.Fatalf("an action-less attempt row keeps the '?' placeholder: %v", got["sid-2"])
	}
	if len(LedgerActions(nil)) != 0 {
		t.Fatalf("nil reader must fold to empty")
	}
}
