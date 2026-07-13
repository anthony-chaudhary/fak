package stopped

import (
	"strings"
	"testing"
)

// The load-bearing facts these pin (from tools/stopped_sessions.py):
//   - a synthetic limit banner is only CURRENT when it is the terminal meaningful turn;
//   - an unmatched tool_use at the tail is STOPPED_MIDTOOL, and a later tool_result
//     clears it;
//   - liveness (age <= LiveMinutes) outranks the tail heuristics but NOT the banner or
//     an auth wall;
//   - Decide defers a resumable session when its ACCOUNT is throttled, and never
//     resumes LIVE / PARKED_WAIT / DONE.

func assistant(text string) Record {
	return Record{Type: "assistant", Role: "assistant", Text: text}
}

// userTurn is a user record carrying a tool_result (the shape of a session that died right
// after a tool landed, with the model about to act) — the residual user-final tail (#3783).
func userTurn(text string) Record {
	return Record{Type: "user", Role: "user", Text: text, HasToolResult: true}
}

func TestThrottleCurrentIsStoppedLimit(t *testing.T) {
	r := Classify([]Record{
		assistant("working on it"),
		{Type: "assistant", Role: "assistant", Synthetic: true,
			Text: "You've hit your session limit · resets 6pm (America/Los_Angeles)"},
	}, 60, 10, "2026-07-01T00:00:00Z", "sid", "p")
	if r.Disp != DispStoppedLimit {
		t.Fatalf("disp = %s, want STOPPED_LIMIT", r.Disp)
	}
	if r.ThrottleReset == "" || !r.ThrottleCurrent {
		t.Fatalf("throttle_reset=%q current=%v", r.ThrottleReset, r.ThrottleCurrent)
	}
}

func TestSupersededBannerIsNotCurrent(t *testing.T) {
	// A banner 2 turns back with a clean later turn: the session recovered — the banner
	// must not read as a current limit.
	r := Classify([]Record{
		{Type: "assistant", Role: "assistant", Synthetic: true,
			Text: "You've hit your session limit · resets 6pm"},
		assistant("recovered; continuing the task"),
	}, 60, 10, "", "sid", "p")
	if r.Disp == DispStoppedLimit {
		t.Fatal("superseded banner must not classify as STOPPED_LIMIT")
	}
	if r.ThrottleCurrent {
		t.Fatal("throttle_current must be false when a later turn superseded the banner")
	}
	if r.ThrottleSeen == "" {
		t.Fatal("throttle_seen should still record the superseded banner for observability")
	}
}

func TestMidtoolAndClearedTool(t *testing.T) {
	r := Classify([]Record{
		{Type: "assistant", Role: "assistant", Text: "running a tool", ToolUseName: "Bash"},
	}, 60, 10, "", "sid", "p")
	if r.Disp != DispStoppedMidtool || r.PendingTool != "Bash" {
		t.Fatalf("disp=%s pending=%q, want STOPPED_MIDTOOL/Bash", r.Disp, r.PendingTool)
	}
	r = Classify([]Record{
		{Type: "assistant", Role: "assistant", Text: "running a tool", ToolUseName: "Bash"},
		{Type: "user", Role: "user", Text: "tool output", HasToolResult: true},
	}, 60, 10, "", "sid", "p")
	if r.Disp == DispStoppedMidtool || r.PendingTool != "" {
		t.Fatalf("tool_result must clear the pending tool_use, got disp=%s pending=%q", r.Disp, r.PendingTool)
	}
}

func TestAuthInterruptParkedDoneQuietLive(t *testing.T) {
	cases := []struct {
		text string
		age  float64
		want Disp
	}{
		{"OAuth token has expired · please run /login", 60, DispStoppedAuth},
		{"[Request interrupted by user", 60, DispStoppedInterrupt},
		{"The workflow is still running; the harness will notify me when it completes.", 60, DispParkedWait},
		{"Done — committed and pushed to origin.", 60, DispDone},
		// An assistant-final residual (no wrap-up phrase) is now STOPPED_DONE, not the QUIET
		// umbrella: the model finished a turn and went idle (#3783). Liveness still outranks it.
		{"thinking about the next step", 60, DispStoppedDone},
		{"thinking about the next step", 2, DispLive},
	}
	for _, c := range cases {
		r := Classify([]Record{assistant(c.text)}, c.age, 10, "", "sid", "p")
		if r.Disp != c.want {
			t.Errorf("Classify(%q, age=%v) = %s, want %s", c.text, c.age, r.Disp, c.want)
		}
	}
}

func TestIdentityAndLastEcho(t *testing.T) {
	r := Classify([]Record{
		{Type: "user", Role: "user", Text: "go", CWD: `C:\work\fak`, GitBranch: "main",
			Version: "2.1.0", SessionID: "abc"},
		assistant("line1\nline2 " + strings.Repeat("x", 400)),
	}, 60, 10, "", "fallback", "p")
	if r.Session != "abc" || r.CWD != `C:\work\fak` || r.Git != "main" || r.Version != "2.1.0" {
		t.Fatalf("identity fields lost: %+v", r)
	}
	if strings.Contains(r.Last, "\n") || len([]rune(r.Last)) > 300 {
		t.Fatalf("last echo must be one line, <=300 runes, got %d", len([]rune(r.Last)))
	}
	r = Classify(nil, 60, 10, "", "fallback", "p")
	if r.Session != "fallback" {
		t.Fatalf("session fallback = %q", r.Session)
	}
}

func TestDecideBuckets(t *testing.T) {
	active := func(reset string) bool { return reset == "6pm" }
	rows := []Row{
		{Disp: DispStoppedMidtool, Account: "a1", AgeMin: 10, Session: "m1"},
		{Disp: DispStoppedLimit, Account: "a1", AgeMin: 5, ThrottleReset: "6pm", Session: "l1"},
		{Disp: DispStoppedQuiet, Account: "a2", AgeMin: 20, Session: "q1"},
		{Disp: DispStoppedAuth, Account: "a3", AgeMin: 30, Session: "au1"},
		{Disp: DispDone, Account: "a2", AgeMin: 40, Session: "d1"},
		{Disp: DispLive, Account: "a2", AgeMin: 1, Session: "v1"},
		{Disp: DispParkedWait, Account: "a2", AgeMin: 50, Session: "pw1"},
	}
	d := Decide(rows, active)
	if thr, ok := d.AccountThrottle["a1"]; !ok || thr.Reset != "6pm" {
		t.Fatalf("a1 should be account-throttled, got %+v", d.AccountThrottle)
	}
	// m1 is on the throttled account a1 -> deferred; q1 on a2 -> resumable.
	if len(d.Resume) != 1 || d.Resume[0].Session != "q1" {
		t.Fatalf("resume = %+v, want just q1", d.Resume)
	}
	if len(d.Defer) != 3 {
		t.Fatalf("defer = %d rows, want 3 (m1 throttled-account, l1 limit, au1 auth)", len(d.Defer))
	}
	for _, r := range d.Defer {
		if r.BlockedBy == "" {
			t.Fatalf("deferred row %s missing blocked_by", r.Session)
		}
	}
	if len(d.Skip) != 3 {
		t.Fatalf("skip = %d rows, want 3 (done, live, parked)", len(d.Skip))
	}
	// Rows are youngest first.
	if d.Rows[0].Session != "v1" {
		t.Fatalf("rows[0] = %s, want the youngest (v1)", d.Rows[0].Session)
	}
	if d.Counts[string(DispStoppedLimit)] != 1 || d.Counts[string(DispDone)] != 1 {
		t.Fatalf("counts = %v", d.Counts)
	}
}

func TestDecideExpiredThrottleFreesAccount(t *testing.T) {
	// The limit row's reset has PASSED: the account is no longer throttled, so the
	// midtool session on the same account resumes. The limit session itself still
	// defers on its own banner (resume-in-place after reset is the launcher's call).
	rows := []Row{
		{Disp: DispStoppedMidtool, Account: "a1", AgeMin: 10, Session: "m1"},
		{Disp: DispStoppedLimit, Account: "a1", AgeMin: 5, ThrottleReset: "6am", Session: "l1"},
	}
	d := Decide(rows, func(string) bool { return false })
	if len(d.AccountThrottle) != 0 {
		t.Fatalf("no account should be throttled, got %v", d.AccountThrottle)
	}
	if len(d.Resume) != 1 || d.Resume[0].Session != "m1" {
		t.Fatalf("resume = %+v, want m1", d.Resume)
	}
}

func TestDecideRefusesOverContextWindowResume(t *testing.T) {
	never := func(string) bool { return false }
	// A midtool session whose transcript is far larger than any model context window would
	// OVERFLOW on replay — a blind resume silently truncates/corrupts. The replay-safety
	// precondition must DEFER it (fail-closed) with a witnessed reason, never route it to
	// Resume. A normally-sized sibling on the same untrottled account still resumes: the
	// gate refuses only the over-window row, not the whole account.
	overKB := DefaultResumeContextWindowTokens/EstimatedTokensPerKB + 100 // estimates just over the window
	rows := []Row{
		{Disp: DispStoppedMidtool, Account: "a1", AgeMin: 10, Session: "over", SizeKB: overKB},
		{Disp: DispStoppedMidtool, Account: "a1", AgeMin: 12, Session: "ok", SizeKB: 10},
	}
	d := Decide(rows, never)

	find := func(b []Row, sid string) *Row {
		for i := range b {
			if b[i].Session == sid {
				return &b[i]
			}
		}
		return nil
	}
	over := find(d.Defer, "over")
	if over == nil {
		t.Fatalf("over-window session must be DEFERRED, not resumed; defer=%+v resume=%+v", d.Defer, d.Resume)
	}
	if find(d.Resume, "over") != nil {
		t.Fatal("over-window session must not also resume")
	}
	if !strings.Contains(over.BlockedBy, "context window") || !strings.Contains(over.BlockedBy, "overflow") {
		t.Fatalf("over-window blocked_by = %q, want it to name the context-window overflow", over.BlockedBy)
	}
	if find(d.Resume, "ok") == nil {
		t.Fatalf("normally-sized session should still resume; resume=%+v", d.Resume)
	}
}

func TestReplaySafetyHonorsPerRowWindowAndFitting(t *testing.T) {
	never := func(string) bool { return false }
	// A small transcript that fits the default window still overflows a TINY per-row target
	// window — the MaxContextTokens override is honored. A transcript that fits the tiny
	// window resumes. This pins the override path and the no-false-positive path.
	rows := []Row{
		{Disp: DispStoppedQuiet, Account: "a1", AgeMin: 10, Session: "tiny-window", SizeKB: 100, MaxContextTokens: 1000},
		{Disp: DispStoppedQuiet, Account: "a1", AgeMin: 12, Session: "fits", SizeKB: 1, MaxContextTokens: 1000},
	}
	d := Decide(rows, never)

	find := func(b []Row, sid string) bool {
		for _, r := range b {
			if r.Session == sid {
				return true
			}
		}
		return false
	}
	// 100 KB * 256 = 25600 est tokens > 1000 window -> deferred.
	if !find(d.Defer, "tiny-window") {
		t.Fatalf("per-row window override not honored; defer=%+v resume=%+v", d.Defer, d.Resume)
	}
	if find(d.Resume, "tiny-window") {
		t.Fatal("over-window row must not resume")
	}
	// 1 KB * 256 = 256 est tokens < 1000 window -> resumes.
	if !find(d.Resume, "fits") {
		t.Fatalf("fitting row should resume; resume=%+v", d.Resume)
	}
}

func TestDecideDupLiveSkipsCrashedDuplicate(t *testing.T) {
	never := func(string) bool { return false }
	// A live dispatch-loop owns work-key "loop:--lane ci" in project P. A crashed session
	// (midtool) in the SAME project with the SAME work-key is a duplicate -> SKIP (flagged
	// DupOfLive, real cause preserved), never resumed. A stopped session with a DIFFERENT key
	// resumes normally. A stopped
	// session with an EMPTY key resumes (fail-open). A same-key stopped session in a
	// DIFFERENT project resumes (dedup is per-project).
	rows := []Row{
		{Disp: DispLive, Account: "a1", AgeMin: 1, Session: "live", Project: "P", WorkKey: "loop:--lane ci"},
		{Disp: DispStoppedMidtool, Account: "a2", AgeMin: 10, Session: "dup", Project: "P", WorkKey: "loop:--lane ci"},
		{Disp: DispStoppedMidtool, Account: "a2", AgeMin: 12, Session: "other", Project: "P", WorkKey: "issue:#1538"},
		{Disp: DispStoppedMidtool, Account: "a2", AgeMin: 14, Session: "nokey", Project: "P", WorkKey: ""},
		{Disp: DispStoppedMidtool, Account: "a2", AgeMin: 16, Session: "otherproj", Project: "Q", WorkKey: "loop:--lane ci"},
	}
	d := Decide(rows, never)

	inBucket := func(b []Row, sid string) bool {
		for _, r := range b {
			if r.Session == sid {
				return true
			}
		}
		return false
	}
	if !inBucket(d.Skip, "dup") {
		t.Fatalf("dup should be SKIP (duplicate-of-live); skip=%+v", d.Skip)
	}
	for _, r := range d.Skip {
		if r.Session == "dup" {
			if !r.DupOfLive {
				t.Fatalf("dup must be flagged DupOfLive; got %+v", r)
			}
			// #3800: the dedup rides its own axis — the row KEEPS its real stop-cause.
			if r.Disp != DispStoppedMidtool {
				t.Fatalf("dup must keep its real cause; disp = %s, want %s", r.Disp, DispStoppedMidtool)
			}
			if r.LiveSibling != "loop:--lane ci" {
				t.Fatalf("dup live_sibling = %q, want the owning work-key", r.LiveSibling)
			}
			if r.BlockedBy == "" || !strings.Contains(r.BlockedBy, "loop:--lane ci") {
				t.Fatalf("dup blocked_by = %q, want the work-key named", r.BlockedBy)
			}
		}
	}
	if inBucket(d.Skip, "dup") && inBucket(d.Resume, "dup") {
		t.Fatalf("dup must not also resume")
	}
	for _, sid := range []string{"other", "nokey", "otherproj"} {
		if !inBucket(d.Resume, sid) {
			t.Fatalf("%s should resume (not a live-owned duplicate); resume=%+v", sid, d.Resume)
		}
	}
	dupCount := 0
	for _, r := range d.Skip {
		if r.DupOfLive {
			dupCount++
		}
	}
	if dupCount != 1 {
		t.Fatalf("want exactly 1 dup-of-live row in skip, got %d (skip=%+v)", dupCount, d.Skip)
	}
	// The dedup no longer masks the cause: the duplicate is still tallied under its real
	// STOPPED_MIDTOOL disposition, so Counts stays a per-cause histogram over every row
	// (midtool: dup + other + nokey + otherproj = 4; live = 1).
	if d.Counts[string(DispStoppedMidtool)] != 4 || d.Counts[string(DispLive)] != 1 {
		t.Fatalf("counts must tally preserved causes (midtool=4 live=1), got %v", d.Counts)
	}
}

// TestResidualSplitByLastRole pins #3783: the residual STOPPED_QUIET is split by the
// terminal record's role — assistant-final => STOPPED_DONE (idle, leave alone), user-final
// => STOPPED_MIDTURN (stranded work, resume), empty tail => the STOPPED_QUIET umbrella. The
// explicit stop signals (auth/limit/interrupt/midtool/live) still win over the split.
func TestResidualSplitByLastRole(t *testing.T) {
	cases := []struct {
		name string
		recs []Record
		age  float64
		want Disp
	}{
		{"assistant-final residual is DONE", []Record{assistant("thinking about the next step")}, 60, DispStoppedDone},
		{"user-final residual is MIDTURN", []Record{userTurn("tool output: 3 files changed")}, 60, DispStoppedMidturn},
		{"empty tail stays QUIET umbrella", nil, 60, DispStoppedQuiet},
		{"assistant-final still LIVE when fresh", []Record{assistant("thinking about the next step")}, 2, DispLive},
		// Precedence: the specific signals outrank the residual role split.
		{"pending tool wins over assistant-final DONE",
			[]Record{{Type: "assistant", Role: "assistant", Text: "running a tool", ToolUseName: "Bash"}}, 60, DispStoppedMidtool},
		{"auth wall wins over user-final MIDTURN",
			[]Record{{Type: "user", Role: "user", Text: "OAuth token has expired · please run /login"}}, 60, DispStoppedAuth},
		{"interrupt wins over user-final MIDTURN",
			[]Record{{Type: "user", Role: "user", Text: "[Request interrupted by user"}}, 60, DispStoppedInterrupt},
	}
	for _, c := range cases {
		r := Classify(c.recs, c.age, 10, "", "sid", "p")
		if r.Disp != c.want {
			t.Errorf("%s: disp = %s, want %s", c.name, r.Disp, c.want)
		}
	}
}

// TestMidturnFixtureFromEvidenceRows reproduces the #3783 census evidence: substantial fak
// sessions whose LAST transcript record is a user turn (a tool_result / file input) — the
// model was mid-turn when it died. These were painted the same QUIET as an idle session and
// so never surfaced as recoverable; each must now classify STOPPED_MIDTURN.
func TestMidturnFixtureFromEvidenceRows(t *testing.T) {
	tails := []string{
		"## Notes & research (docs/notes/) ...",
		"TRUNK_WOULD_NOT_COMPILE (advisory): the pushed tip would not build ...",
		"--- PASS: TestDispatchProductWorkerCountIncludesRepair... (3.73s)",
		"https://github.com/anthony-chaudhary/fak/issues/3775",
	}
	for _, tail := range tails {
		r := Classify([]Record{
			assistant("kicking off the next step"),
			userTurn(tail),
		}, 120, 2600, "", "sid", "p")
		if r.Disp != DispStoppedMidturn {
			t.Errorf("evidence tail %q => %s, want STOPPED_MIDTURN (stranded mid-turn work)", tail, r.Disp)
		}
		if r.LastRole != "user" {
			t.Errorf("evidence tail %q last_role = %q, want user", tail, r.LastRole)
		}
	}
}

// TestDecideRoutesMidturnAndDone pins the Decide-side wiring of the #3783 split: a
// STOPPED_MIDTURN row is resume-eligible (subject to the same throttle/replay-safety gates
// as other resumable dispositions), while a STOPPED_DONE row is leave-alone (Skip). An
// over-window MIDTURN still defers on the replay-safety precondition.
func TestDecideRoutesMidturnAndDone(t *testing.T) {
	never := func(string) bool { return false }
	overKB := DefaultResumeContextWindowTokens/EstimatedTokensPerKB + 100
	rows := []Row{
		{Disp: DispStoppedMidturn, Account: "a1", AgeMin: 10, Session: "mt", SizeKB: 10},
		{Disp: DispStoppedDone, Account: "a1", AgeMin: 12, Session: "done"},
		{Disp: DispStoppedMidturn, Account: "a1", AgeMin: 14, Session: "big", SizeKB: overKB},
	}
	d := Decide(rows, never)

	in := func(b []Row, sid string) bool {
		for _, r := range b {
			if r.Session == sid {
				return true
			}
		}
		return false
	}
	if !in(d.Resume, "mt") {
		t.Fatalf("STOPPED_MIDTURN should resume; resume=%+v", d.Resume)
	}
	if !in(d.Skip, "done") || in(d.Resume, "done") {
		t.Fatalf("STOPPED_DONE should skip (leave alone), not resume; skip=%+v resume=%+v", d.Skip, d.Resume)
	}
	if !in(d.Defer, "big") || in(d.Resume, "big") {
		t.Fatalf("over-window STOPPED_MIDTURN should defer on replay-safety; defer=%+v resume=%+v", d.Defer, d.Resume)
	}
	if d.Counts[string(DispStoppedMidturn)] != 2 || d.Counts[string(DispStoppedDone)] != 1 {
		t.Fatalf("counts must tabulate the split separately, got %v", d.Counts)
	}
}

// TestDecideDoneWithLiveSiblingKeepsCause pins that STOPPED_DONE is excluded from the dedup
// flag (like DONE): a finished session that happens to duplicate a live sibling keeps its
// cause (STOPPED_DONE) and is NOT marked DupOfLive — it is not a resume candidate, so the flag
// would be noise (#3783 / cf. #3800). A STOPPED_MIDTURN duplicate — a genuine would-be resume
// candidate — is flagged DupOfLive while keeping its real cause.
func TestDecideDoneWithLiveSiblingKeepsCause(t *testing.T) {
	never := func(string) bool { return false }
	rows := []Row{
		{Disp: DispLive, Account: "a1", AgeMin: 1, Session: "live", Project: "P", WorkKey: "issue:#42"},
		{Disp: DispStoppedDone, Account: "a2", AgeMin: 10, Session: "done", Project: "P", WorkKey: "issue:#42"},
		{Disp: DispStoppedMidturn, Account: "a2", AgeMin: 12, Session: "mt", Project: "P", WorkKey: "issue:#42"},
	}
	d := Decide(rows, never)

	find := func(b []Row, sid string) *Row {
		for i := range b {
			if b[i].Session == sid {
				return &b[i]
			}
		}
		return nil
	}
	done := find(d.Skip, "done")
	if done == nil || done.Disp != DispStoppedDone || done.DupOfLive {
		t.Fatalf("finished duplicate must stay STOPPED_DONE in Skip, not flagged DupOfLive (cause preserved), got %+v", d.Skip)
	}
	mt := find(d.Skip, "mt")
	if mt == nil || !mt.DupOfLive || mt.Disp != DispStoppedMidturn {
		t.Fatalf("mid-turn duplicate should be flagged DupOfLive with its cause preserved; skip=%+v", d.Skip)
	}
}
