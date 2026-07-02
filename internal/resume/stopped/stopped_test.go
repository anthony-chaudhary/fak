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
		want string
	}{
		{"OAuth token has expired · please run /login", 60, DispStoppedAuth},
		{"[Request interrupted by user", 60, DispStoppedInterrupt},
		{"The workflow is still running; the harness will notify me when it completes.", 60, DispParkedWait},
		{"Done — committed and pushed to origin.", 60, DispDone},
		{"thinking about the next step", 60, DispStoppedQuiet},
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
	if d.Counts[DispStoppedLimit] != 1 || d.Counts[DispDone] != 1 {
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
