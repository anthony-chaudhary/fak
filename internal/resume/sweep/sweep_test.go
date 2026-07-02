package sweep

import (
	"strings"
	"testing"
	"time"
)

// Ported from tools/resume_sweep_test.py. The load-bearing facts these pin:
//   - a session is bucketed from its NEWEST copy's terminal turn, with the reset
//     past/future verdict deciding LIMIT_RESET_PASSED vs LIMIT_RESET_FUTURE;
//   - the SUPERSET copy is chosen by uuid-set + last-ts, NOT file mtime;
//   - the error channel outranks assistant prose, with the averted false positive
//     surfaced as ProseDiverged;
//   - sessions already in the resume ledger in-window are excluded.

// 2026-06-23T18:00Z == 11:00 PDT, the fixture time the Python tests used.
var now = time.Date(2026, 6, 23, 18, 0, 0, 0, time.UTC)

func rec(role, text, uuid, ts string, isErr bool) Record {
	return Record{UUID: uuid, Timestamp: ts, Role: role, Text: text, IsError: isErr}
}

func one(t *testing.T, sid string, recs []Record) Row {
	t.Helper()
	c := Copy{Path: "p", Account: ".claude-x", Project: "C--work-fak", Records: recs}
	return Classify(sid, []Copy{c}, nil, now)
}

func TestLimitResetPassed(t *testing.T) {
	r := one(t, "s1", []Record{
		rec("assistant", "You've hit your session limit . resets 6am (America/Los_Angeles)", "", "2026-06-23T13:00:00Z", true),
	})
	if r.Bucket != BucketLimitResetPassed {
		t.Fatalf("bucket = %s, want LIMIT_RESET_PASSED (6am < 11am now)", r.Bucket)
	}
}

func TestLimitResetFuture(t *testing.T) {
	r := one(t, "s2", []Record{
		rec("assistant", "You've hit your session limit . resets 11pm (America/Los_Angeles)", "", "2026-06-23T16:00:00Z", true),
	})
	if r.Bucket != BucketLimitResetFuture {
		t.Fatalf("bucket = %s, want LIMIT_RESET_FUTURE (11pm not yet at 11am)", r.Bucket)
	}
}

func TestAPIErrorBucket(t *testing.T) {
	r := one(t, "s3", []Record{
		rec("assistant", "API Error: Overloaded (529) server-side issue", "", "2026-06-23T17:00:00Z", true),
	})
	if r.Bucket != BucketAPIErr {
		t.Fatalf("bucket = %s, want API_ERR", r.Bucket)
	}
}

func TestAuthBucket(t *testing.T) {
	r := one(t, "s4", []Record{
		rec("assistant", "Not logged in . Please run /login", "", "2026-06-23T17:00:00Z", true),
	})
	if r.Bucket != BucketAuth {
		t.Fatalf("bucket = %s, want AUTH", r.Bucket)
	}
	if !strings.Contains(r.Evidence, "Please run /login") {
		t.Fatalf("evidence should carry the error text that drove the bucket, got %q", r.Evidence)
	}
}

func TestProseAboutAuthDoesNotOverrideErrorChannel(t *testing.T) {
	// 2026-06-23 regression (gem7 732edb34): a worker editing the resume tooling
	// narrated an auth wall in its FINAL assistant turn while its real error record was
	// a transient 529. The detector must not flag the session that WROTE the detector.
	r := one(t, "s_prose_auth", []Record{
		rec("assistant", "API Error: Server is temporarily limiting requests (not your usage limit) . Rate limited",
			"", "2026-06-23T17:00:00Z", true),
		rec("assistant", "Remediation for the gem7 wall was to please run /login on smith; gem7-netra is logged back in.",
			"", "2026-06-23T17:59:00Z", false),
	})
	if r.Bucket != BucketAPIErr {
		t.Fatalf("bucket = %s, want API_ERR (error channel, not prose)", r.Bucket)
	}
	if !r.ProseDiverged {
		t.Fatal("ProseDiverged should surface the averted prose-only false positive")
	}
	if !strings.Contains(r.Evidence, "Rate limited") {
		t.Fatalf("evidence = %q", r.Evidence)
	}
}

func TestProseLoginDoesNotOverrideLimitError(t *testing.T) {
	r := one(t, "s_prose_login", []Record{
		rec("assistant", "You've hit your session limit . resets 6am (America/Los_Angeles)", "", "2026-06-23T11:30:00Z", true),
		rec("assistant", "Next I'll please run /login to re-home this seat.", "", "2026-06-23T11:35:00Z", false),
	})
	if r.Bucket != BucketLimitResetPassed {
		t.Fatalf("bucket = %s, want LIMIT_RESET_PASSED", r.Bucket)
	}
	if !r.ProseDiverged {
		t.Fatal("ProseDiverged should be set")
	}
}

func TestProseAPIMentionWithoutErrorRecordIsOther(t *testing.T) {
	r := one(t, "s_prose_api", []Record{
		rec("assistant", "Earlier I hit an API Error 529 but retried and it's green now.", "", "2026-06-23T17:00:00Z", false),
	})
	if r.Bucket != BucketOther {
		t.Fatalf("bucket = %s, want OTHER (no error record means no failure bucket)", r.Bucket)
	}
	if r.Evidence != "" {
		t.Fatalf("evidence should be empty for OTHER, got %q", r.Evidence)
	}
}

func TestCleanSessionIsOther(t *testing.T) {
	r := one(t, "s5", []Record{rec("assistant", "All done, shipped and green.", "", "2026-06-23T17:00:00Z", false)})
	if r.Bucket != BucketOther {
		t.Fatalf("bucket = %s, want OTHER", r.Bucket)
	}
}

func TestLiveOverridesError(t *testing.T) {
	c := Copy{Path: "p", Account: ".claude-x", Project: "C--work-fak", Records: []Record{
		rec("assistant", "API Error 529", "", "2026-06-23T17:00:00Z", true),
	}}
	r := Classify("s6", []Copy{c}, map[string]bool{"s6": true}, now)
	if r.Bucket != BucketLive {
		t.Fatalf("bucket = %s, want LIVE", r.Bucket)
	}
}

func TestSupersetPicksLatestTSNotMtime(t *testing.T) {
	// gem7 copy: a strict PREFIX (u1,u2) with an OLDER last-ts (a re-capped resume
	// rewrote only its banner — on disk it would carry the NEWER mtime, the trap).
	gem7 := Copy{Path: "gem7", Account: ".claude-gem7", Project: "C--work-fak", Records: []Record{
		rec("assistant", "a", "u1", "2026-06-23T10:00:00Z", false),
		rec("assistant", "limit resets 6am (America/Los_Angeles)", "u2", "2026-06-23T10:05:00Z", true),
	}}
	// smith copy: the SUPERSET (u1..u3) with the LATER last-ts, tail still the banner.
	smith := Copy{Path: "smith", Account: ".claude-smith", Project: "C--work-fak", Records: []Record{
		rec("assistant", "a", "u1", "2026-06-23T10:00:00Z", false),
		rec("assistant", "b", "u2", "2026-06-23T10:05:00Z", false),
		rec("assistant", "limit resets 6am (America/Los_Angeles)", "u3", "2026-06-23T10:20:00Z", true),
	}}
	r := Classify("s", []Copy{gem7, smith}, nil, now)
	if r.SupersetAccount != ".claude-smith" {
		t.Fatalf("superset account = %s, want .claude-smith (latest last-ts, not mtime)", r.SupersetAccount)
	}
	if !r.IsSuperset || r.NRecords != 3 {
		t.Fatalf("is_superset=%v n=%d, want true/3", r.IsSuperset, r.NRecords)
	}
	if r.Bucket != BucketLimitResetPassed {
		t.Fatalf("bucket = %s, want LIMIT_RESET_PASSED", r.Bucket)
	}
}

func TestNonSupersetFlagged(t *testing.T) {
	a := Copy{Path: "a", Account: ".claude-a", Project: "P", Records: []Record{
		rec("assistant", "x", "u1", "2026-06-23T10:00:00Z", false),
		rec("assistant", "y", "u9", "2026-06-23T10:30:00Z", false),
	}}
	b := Copy{Path: "b", Account: ".claude-b", Project: "P", Records: []Record{
		rec("assistant", "x", "u1", "2026-06-23T10:00:00Z", false),
		rec("assistant", "z", "u2", "2026-06-23T10:05:00Z", false),
	}}
	r := Classify("s", []Copy{a, b}, nil, now)
	if r.SupersetAccount != ".claude-a" {
		t.Fatalf("best = %s, want .claude-a (later last-ts)", r.SupersetAccount)
	}
	if r.IsSuperset {
		t.Fatal("u2 is not in the best copy — IsSuperset must be false (diverged copies)")
	}
}

func TestRecentlyResumedWindow(t *testing.T) {
	ledger := strings.Join([]string{
		`{"ts": "2026-06-23T17:50:00Z", "session": "fresh"}`, // 10m ago
		`{"ts": "2026-06-23T10:00:00Z", "session": "stale"}`, // 8h ago
		`{"ts": "bad", "session": "x"}`,
		``,
	}, "\n")
	got := RecentlyResumed(strings.NewReader(ledger), 600, now) // 10h window
	if !got["fresh"] || !got["stale"] {
		t.Fatalf("10h window should include both, got %v", got)
	}
	got2 := RecentlyResumed(strings.NewReader(ledger), 60, now) // 1h window
	if !got2["fresh"] || got2["stale"] {
		t.Fatalf("1h window should include only fresh, got %v", got2)
	}
	if got2["x"] {
		t.Fatal("malformed ts must be skipped")
	}
}

func TestSortOrder(t *testing.T) {
	rows := []Row{
		{SID: "a", Bucket: BucketLive},
		{SID: "b", Bucket: BucketAuth},
		{SID: "c", Bucket: BucketAPIErr},
		{SID: "d", Bucket: BucketLimitResetFuture},
		{SID: "e", Bucket: BucketLimitResetPassed, SupersetAccount: "z", NRecords: 5},
		{SID: "f", Bucket: BucketLimitResetPassed, SupersetAccount: "z", NRecords: 9},
	}
	Sort(rows)
	want := []string{"f", "e", "c", "d", "b", "a"}
	for i, sid := range want {
		if rows[i].SID != sid {
			t.Fatalf("rows[%d] = %s, want %s (full order %v)", i, rows[i].SID, sid, rows)
		}
	}
}

func TestCwdForSlug(t *testing.T) {
	real := `C:\work\slack-helpers`
	slug := Slugify(real)
	if strings.ContainsAny(slug, `\/`) {
		t.Fatalf("slug must not contain separators: %q", slug)
	}
	if got := CwdForSlug(slug, []string{`C:\work\other`, real}, "fb"); got != real {
		t.Fatalf("CwdForSlug = %q, want %q", got, real)
	}
	if got := CwdForSlug("C--nonexistent-xyz-123", []string{`C:\work\other`}, "/fallback"); got != "/fallback" {
		t.Fatalf("fallback = %q", got)
	}
}
