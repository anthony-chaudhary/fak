package stopped

import "testing"

// The defect these pin (#5386): a session that CRASHED mid-tool-call and a session that is
// merely still inside a SLOW tool call leave byte-identical evidence in the transcript - an
// assistant tool_use with no matching tool_result, on a file nobody has appended to since -
// and the classifier reported both as STOPPED_MIDTOOL, which Decide routes straight into the
// Resume bucket. Those two situations want OPPOSITE operator actions (recover vs. wait), and
// resuming the slow one launches a second driver onto a transcript the first one is still
// writing. The whole point of the fix is the DISCRIMINATION, so these tests hold the
// transcript tail constant and vary only the driver-liveness evidence.

// midtoolTail is the issue's repro tail, verbatim in shape: an assistant turn that opened a
// Bash tool_use and never got its tool_result. Identical bytes for a crash and for a slow
// call - that identity is the bug.
func midtoolTail() []Record {
	return []Record{
		{Type: "user", Role: "user", Text: "please run the build"},
		{Type: "assistant", Role: "assistant", Text: "Running the build now.", ToolUseName: "Bash"},
	}
}

// TestMidtoolCrashedAndSlowDoNotShareOneVerdict is the load-bearing witness: ONE tail, THREE
// liveness readings, THREE distinct verdicts, and three distinct Decide buckets. Case 1 is the
// crashed session (must still be STOPPED_MIDTOOL and still resume). Case 2 is the live-but-slow
// session (must NOT be STOPPED_MIDTOOL and must NOT resume). Case 3 is the unwitnessed session,
// which is what the production entry point Classify produces today: an explicit unknown that
// defers, never either extreme.
func TestMidtoolCrashedAndSlowDoNotShareOneVerdict(t *testing.T) {
	never := func(string) bool { return false }
	// 30 minutes of silence - the issue's backdated mtime. Every case is well past LiveMinutes,
	// so transcript freshness cannot be doing any of the work below.
	const ageMin = 30.0

	crashed := ClassifyWithLiveness(midtoolTail(), ageMin, 10, "", "crashed", "p", LivenessGone)
	slow := ClassifyWithLiveness(midtoolTail(), ageMin, 10, "", "slow", "p", LivenessLive)
	unwitnessed := Classify(midtoolTail(), ageMin, 10, "", "unwitnessed", "p")

	// 1. The genuinely dead mid-tool session is still reported as crashed.
	if crashed.Disp != DispStoppedMidtool {
		t.Fatalf("crashed mid-tool disp = %s, want %s", crashed.Disp, DispStoppedMidtool)
	}
	// 2. The merely-slow session must NOT wear the crash label.
	if slow.Disp == DispStoppedMidtool {
		t.Fatalf("a driver still running a slow tool call must not classify as %s", DispStoppedMidtool)
	}
	if slow.Disp != DispLive {
		t.Fatalf("slow-but-live mid-tool disp = %s, want %s (leave it alone)", slow.Disp, DispLive)
	}
	// 3. No evidence at all lands on the explicit unknown - not on either extreme.
	if unwitnessed.Disp == DispStoppedMidtool || unwitnessed.Disp == DispLive {
		t.Fatalf("unwitnessed mid-tool disp = %s: absence of evidence must not be read as death OR life", unwitnessed.Disp)
	}
	if unwitnessed.Disp != DispMidtoolUnknown {
		t.Fatalf("unwitnessed mid-tool disp = %s, want %s", unwitnessed.Disp, DispMidtoolUnknown)
	}
	// The three verdicts must be pairwise distinct: the bug was two situations sharing one label.
	if crashed.Disp == slow.Disp || crashed.Disp == unwitnessed.Disp || slow.Disp == unwitnessed.Disp {
		t.Fatalf("verdicts collapsed: crashed=%s slow=%s unwitnessed=%s", crashed.Disp, slow.Disp, unwitnessed.Disp)
	}
	// The pending tool is still named in every case - the discrimination adds a fact, it does
	// not cost the operator the one they already had.
	for _, r := range []Row{crashed, slow, unwitnessed} {
		if r.PendingTool != "Bash" {
			t.Fatalf("%s: pending_tool = %q, want Bash", r.Session, r.PendingTool)
		}
	}
	// The evidence that drove each verdict is echoed on the row, so an unwitnessed row is
	// visibly unwitnessed rather than indistinguishable from a measured one.
	if crashed.Liveness != LivenessGone || slow.Liveness != LivenessLive || unwitnessed.Liveness != LivenessUnknown {
		t.Fatalf("liveness echo lost: crashed=%q slow=%q unwitnessed=%q",
			crashed.Liveness, slow.Liveness, unwitnessed.Liveness)
	}

	// Decide must act on the discrimination, not just report it.
	d := Decide([]Row{crashed, slow, unwitnessed}, never)
	in := func(b []Row, sid string) bool {
		for _, r := range b {
			if r.Session == sid {
				return true
			}
		}
		return false
	}
	if !in(d.Resume, "crashed") {
		t.Fatalf("a crashed mid-tool session must still be resumable; resume=%+v", d.Resume)
	}
	if in(d.Resume, "slow") {
		t.Fatal("a live-but-slow session must never be resumed - that is two drivers on one transcript")
	}
	if !in(d.Skip, "slow") {
		t.Fatalf("a live-but-slow session belongs in SKIP (leave alone); skip=%+v", d.Skip)
	}
	if in(d.Resume, "unwitnessed") {
		t.Fatal("an unwitnessed mid-tool session must not be resumed on a guess")
	}
	if !in(d.Defer, "unwitnessed") {
		t.Fatalf("an unwitnessed mid-tool session must DEFER with a reason; defer=%+v skip=%+v", d.Defer, d.Skip)
	}
	// Counts stays a per-cause histogram over the new member too.
	if d.Counts[string(DispMidtoolUnknown)] != 1 || d.Counts[string(DispStoppedMidtool)] != 1 || d.Counts[string(DispLive)] != 1 {
		t.Fatalf("counts must tally each verdict separately, got %v", d.Counts)
	}
}

// TestMidtoolUnknownDefersWithAnHonestReason pins the deferred row's witnessed reason: it must
// name the ambiguity rather than assert a state nobody measured, and it must not page a person
// (the wall lifts by itself when the slow call returns and the session appends again).
func TestMidtoolUnknownDefersWithAnHonestReason(t *testing.T) {
	d := Decide([]Row{Classify(midtoolTail(), 30, 10, "", "unwitnessed", "p")}, func(string) bool { return false })
	if len(d.Defer) != 1 {
		t.Fatalf("want the unwitnessed row deferred, got defer=%+v resume=%+v skip=%+v", d.Defer, d.Resume, d.Skip)
	}
	got := d.Defer[0]
	if got.BlockedBy != MidtoolUnknownBlockedBy {
		t.Fatalf("blocked_by = %q, want the named mid-tool ambiguity reason", got.BlockedBy)
	}
	if got.BlockedBy == "" {
		t.Fatal("a deferred row must always carry a reason")
	}
	// The wall clears on its own, so it belongs on the fleet-wait side of the triage split -
	// an operator must not be paged to babysit a session that may simply be slow.
	if DeferNeedsHuman(got) {
		t.Fatalf("an unresolved mid-tool row must not page a person: %q", got.BlockedBy)
	}
	need, wait := PartitionDefer(d)
	if len(need) != 0 || len(wait) != 1 {
		t.Fatalf("want the unresolved mid-tool row as a fleet-wait, got need=%+v wait=%+v", need, wait)
	}
}

// TestLivenessOnlyDecidesTheAmbiguousBranch pins the fix's blast radius. Liveness evidence is
// consulted ONLY where the transcript is genuinely ambiguous - the mid-tool tail. Every other
// terminal signal reads its own unambiguous evidence and is unchanged: a driver that is alive
// and parked at a login wall is STOPPED_AUTH, not LIVE, and a dead driver whose last turn was a
// current limit banner is STOPPED_LIMIT, not STOPPED_MIDTOOL.
func TestLivenessOnlyDecidesTheAmbiguousBranch(t *testing.T) {
	authWall := []Record{{Type: "assistant", Role: "assistant", Text: "OAuth token has expired · please run /login"}}
	for _, live := range []Liveness{LivenessUnknown, LivenessLive, LivenessGone} {
		if r := ClassifyWithLiveness(authWall, 30, 10, "", "sid", "p", live); r.Disp != DispStoppedAuth {
			t.Errorf("auth wall with liveness %q = %s, want %s", live, r.Disp, DispStoppedAuth)
		}
	}
	banner := []Record{{Type: "assistant", Role: "assistant", Synthetic: true,
		Text: "You've hit your session limit · resets 6pm (America/Los_Angeles)"}}
	for _, live := range []Liveness{LivenessUnknown, LivenessLive, LivenessGone} {
		if r := ClassifyWithLiveness(banner, 30, 10, "", "sid", "p", live); r.Disp != DispStoppedLimit {
			t.Errorf("limit banner with liveness %q = %s, want %s", live, r.Disp, DispStoppedLimit)
		}
	}
	// A tail with NO pending tool is untouched by liveness evidence either way.
	idle := []Record{{Type: "assistant", Role: "assistant", Text: "thinking about the next step"}}
	for _, live := range []Liveness{LivenessUnknown, LivenessLive, LivenessGone} {
		if r := ClassifyWithLiveness(idle, 30, 10, "", "sid", "p", live); r.Disp != DispStoppedDone {
			t.Errorf("assistant-final residual with liveness %q = %s, want %s", live, r.Disp, DispStoppedDone)
		}
	}
	// A fresh transcript is LIVE on freshness alone, even with no liveness evidence: the new
	// fact ADDS a way to prove life, it never removes the old one.
	if r := Classify(midtoolTail(), 1, 10, "", "sid", "p"); r.Disp != DispLive {
		t.Errorf("fresh mid-tool tail = %s, want %s (freshness still proves life)", r.Disp, DispLive)
	}
}

// TestMidtoolDispIsTotalOverLiveness pins the resolver's default arm: an unrecognized liveness
// value is NOT a liveness claim, so it resolves like no evidence at all rather than silently
// falling through to the crash verdict.
func TestMidtoolDispIsTotalOverLiveness(t *testing.T) {
	cases := map[Liveness]Disp{
		LivenessGone:     DispStoppedMidtool,
		LivenessLive:     DispLive,
		LivenessUnknown:  DispMidtoolUnknown,
		Liveness("junk"): DispMidtoolUnknown,
	}
	for live, want := range cases {
		if got := midtoolDisp(live); got != want {
			t.Errorf("midtoolDisp(%q) = %s, want %s", live, got, want)
		}
	}
}
