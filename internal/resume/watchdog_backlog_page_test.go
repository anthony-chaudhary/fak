package resume

import "testing"

// #3582 — the post-reset backlog SLO gate. BOTTLENECK-MAP §7's manual decision ("if
// auto_resume is still >= N with 0 throttled accounts, the cap is the real limiter") as a
// standing detector, so the condition no longer needs a human eyeballing every throttle reset.

// backlogTicks builds M consecutive depth samples at the given depths.
func backlogTicks(depths ...int) []WatchdogStatusEvent {
	out := make([]WatchdogStatusEvent, 0, len(depths))
	for i, d := range depths {
		out = append(out, WatchdogStatusEvent{
			UnixSeconds:     int64(1000 + i*300),
			Phase:           "status",
			Mode:            "LIVE",
			AutoResumeDepth: d,
		})
	}
	return out
}

func backlogInput(events []WatchdogStatusEvent, throttled int, known bool) WatchdogStatusInput {
	return WatchdogStatusInput{
		Mode:                   "LIVE",
		NowUnix:                9000,
		BacklogThreshold:       20,
		BacklogTicks:           3,
		ThrottledAccounts:      throttled,
		ThrottledAccountsKnown: known,
		Events:                 events,
	}
}

// Acceptance 1: depth > threshold for M ticks with 0 throttled accounts pages exactly once.
func TestBacklogGatePagesWhenBacklogOutlivesThrottleReset(t *testing.T) {
	rep := FoldWatchdogStatus(backlogInput(backlogTicks(24, 23, 25), 0, true))

	if rep.Verdict != WatchdogDrainRed {
		t.Fatalf("verdict = %q, want %q", rep.Verdict, WatchdogDrainRed)
	}
	if rep.Page == nil {
		t.Fatal("no page emitted; want one post-reset backlog page")
	}
	if rep.Page.Reason != WatchdogPageBacklogPersists {
		t.Errorf("reason = %q, want %q", rep.Page.Reason, WatchdogPageBacklogPersists)
	}
	if rep.Page.Depth != 25 || rep.Page.Threshold != 20 || rep.Page.Ticks != 3 {
		t.Errorf("page facts = depth %d threshold %d ticks %d, want 25/20/3",
			rep.Page.Depth, rep.Page.Threshold, rep.Page.Ticks)
	}
	if rep.Page.ThrottledAccounts != 0 {
		t.Errorf("throttled = %d, want 0 (the whole point of the gate)", rep.Page.ThrottledAccounts)
	}
	// The page's Detail leads the reasons so it is the headline an operator reads first.
	if len(rep.Reasons) == 0 || rep.Reasons[0] != rep.Page.Detail {
		t.Errorf("page detail must lead reasons; got %v", rep.Reasons)
	}
}

// Acceptance 2: the SAME depth WITH throttled accounts is transient pressure (§4), not a page.
func TestBacklogGateSilentWhileAccountsAreThrottled(t *testing.T) {
	rep := FoldWatchdogStatus(backlogInput(backlogTicks(24, 23, 25), 2, true))

	if rep.Page != nil {
		t.Fatalf("paged with 2 throttled accounts; want silence (transient throttle, §4): %+v", rep.Page)
	}
	for _, r := range rep.Reasons {
		if r == WatchdogPageBacklogPersists {
			t.Errorf("backlog reason leaked into reasons while throttled: %v", rep.Reasons)
		}
	}
}

// Acceptance 3: re-firing across ticks keeps ONE dedup signature — the shell refreshes a
// single occurrence-counted issue/toast instead of filing one per tick.
func TestBacklogGateSignatureIsStableAcrossTicks(t *testing.T) {
	first := FoldWatchdogStatus(backlogInput(backlogTicks(24, 23, 25), 0, true))
	// A later tick: different depths, different clock — same gate.
	later := backlogInput(backlogTicks(31, 40, 37), 0, true)
	later.NowUnix = 99000
	second := FoldWatchdogStatus(later)

	if first.Page == nil || second.Page == nil {
		t.Fatal("both ticks must page")
	}
	if first.Page.Signature != second.Page.Signature {
		t.Errorf("signature drifted across ticks: %q vs %q — would file one issue per tick",
			first.Page.Signature, second.Page.Signature)
	}
	if first.Page.Depth == second.Page.Depth {
		t.Fatal("fixture bug: depths must differ to prove the signature ignores live depth")
	}
}

func TestBacklogGateSilentBelowThresholdAndBeforeEnoughTicks(t *testing.T) {
	cases := []struct {
		name   string
		events []WatchdogStatusEvent
	}{
		// One dip at or below threshold breaks the streak: the backlog did drain.
		{"one tick dips to threshold", backlogTicks(24, 20, 25)},
		{"one tick below threshold", backlogTicks(24, 3, 25)},
		// Not enough history yet — never page off a partial streak.
		{"fewer samples than ticks", backlogTicks(24, 25)},
		{"all below threshold", backlogTicks(1, 2, 3)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if page := FoldWatchdogStatus(backlogInput(tc.events, 0, true)).Page; page != nil {
				t.Errorf("paged, want silence: %+v", page)
			}
		})
	}
}

// An unreadable roster reads as "0 throttled" but proves nothing — fail closed, never page.
func TestBacklogGateFailsClosedWhenThrottledCountUnknown(t *testing.T) {
	if page := FoldWatchdogStatus(backlogInput(backlogTicks(24, 23, 25), 0, false)).Page; page != nil {
		t.Fatalf("paged on an unreadable roster; want fail-closed silence: %+v", page)
	}
}

// The gate is off unless armed, so an existing caller that never sets the thresholds keeps
// its exact prior verdict.
func TestBacklogGateDisarmedByDefault(t *testing.T) {
	in := backlogInput(backlogTicks(99, 99, 99), 0, true)
	in.BacklogThreshold, in.BacklogTicks = 0, 0

	rep := FoldWatchdogStatus(in)
	if rep.Page != nil {
		t.Fatalf("disarmed gate paged: %+v", rep.Page)
	}
	if rep.Verdict != WatchdogDrainGreen {
		t.Errorf("verdict = %q, want %q when nothing else is wrong", rep.Verdict, WatchdogDrainGreen)
	}
}
