package sweep

import (
	"strings"
	"testing"
)

// Ported from tools/resume_relaunch_audit.py's audit(). The load-bearing facts:
//   - every rostered sid gets a row; a missing transcript is a NO_TRANSCRIPT finding,
//     never a skip;
//   - the verdict comes from the SUPERSET copy (latest last-ts, then most records);
//   - rows sort still-broken-first, then account, then most records, then sid;
//   - the exit signal counts every verdict that is not RELAUNCHED_OK.

func TestAuditRelaunchesFold(t *testing.T) {
	roster := map[string][]string{
		"sid-ok":       {"resume"},
		"sid-stranded": {"resume", "retry"},
		"sid-gone":     {"resume"},
	}
	copies := map[string][]Copy{
		"sid-ok": {
			// A stale prefix copy that would mis-verdict as STRANDED: the superset rule
			// must pick the newer copy that advanced past the error.
			{Path: "stale", Account: ".claude-a", Records: []Record{
				rec("assistant", "API Error 529", "", "2026-06-23T10:00:00Z", true),
			}},
			{Path: "fresh", Account: ".claude-b", Records: []Record{
				rec("assistant", "API Error 529", "", "2026-06-23T10:00:00Z", true),
				rec("assistant", "resumed and kept working", "", "2026-06-23T12:00:00Z", false),
			}},
		},
		"sid-stranded": {
			{Path: "only", Account: ".claude-a", Records: []Record{
				rec("assistant", "real work", "", "2026-06-23T09:00:00Z", false),
				rec("assistant", "Not logged in . Please run /login", "", "2026-06-23T11:00:00Z", true),
			}},
		},
	}
	rows := AuditRelaunches(roster, copies, map[string]bool{"sid-stranded": true})

	if len(rows) != 3 {
		t.Fatalf("want a row per rostered sid, got %d", len(rows))
	}
	order := []string{VerdictStranded, VerdictNoTranscript, VerdictRelaunchedOK}
	for i, want := range order {
		if rows[i].Verdict != want {
			t.Fatalf("row %d verdict = %s, want %s (still-broken-first)", i, rows[i].Verdict, want)
		}
	}
	stranded := rows[0]
	if stranded.SID != "sid-stranded" || stranded.Kind != "AUTH" || !stranded.Live {
		t.Fatalf("stranded row wrong: %+v", stranded)
	}
	if strings.Join(stranded.Actions, ",") != "resume,retry" {
		t.Fatalf("actions not carried: %v", stranded.Actions)
	}
	gone := rows[1]
	if gone.SID != "sid-gone" || gone.Account != "" || gone.N != 0 || gone.SupersetPath != "" {
		t.Fatalf("NO_TRANSCRIPT row must be empty of copy facts: %+v", gone)
	}
	ok := rows[2]
	if ok.SupersetPath != "fresh" || ok.Account != ".claude-b" || ok.N != 2 {
		t.Fatalf("verdict must come from the superset copy: %+v", ok)
	}
	if ok.Live {
		t.Fatalf("sid-ok is not in the live census")
	}
}

func TestAuditRelaunchesDeterministicTieBreak(t *testing.T) {
	roster := map[string][]string{"sid-b": {"?"}, "sid-a": {"?"}, "sid-c": {"?"}}
	rows := AuditRelaunches(roster, nil, nil)
	got := []string{rows[0].SID, rows[1].SID, rows[2].SID}
	if strings.Join(got, ",") != "sid-a,sid-b,sid-c" {
		t.Fatalf("equal-key rows must sort by sid: %v", got)
	}
}

func TestCountNotOK(t *testing.T) {
	rows := []RelaunchRow{
		{RelaunchResult: RelaunchResult{Verdict: VerdictRelaunchedOK}},
		{RelaunchResult: RelaunchResult{Verdict: VerdictStranded}},
		{RelaunchResult: RelaunchResult{Verdict: VerdictNoTranscript}},
	}
	if got := CountNotOK(rows); got != 2 {
		t.Fatalf("CountNotOK = %d, want 2", got)
	}
	if CountNotOK(nil) != 0 {
		t.Fatalf("empty audit is all-OK")
	}
}
