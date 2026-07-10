package sessionaudit

import (
	"testing"
	"time"
)

// withText sets an assistant record's content to one or more "text" blocks — the
// committed narration the Confusion lens reads.
func withText(texts ...string) func(map[string]any) {
	return func(rec map[string]any) {
		msg := rec["message"].(map[string]any)
		blocks := make([]any, 0, len(texts))
		for _, t := range texts {
			blocks = append(blocks, map[string]any{"type": "text", "text": t})
		}
		msg["content"] = blocks
	}
}

// TestConfusionLensDetectsAndGuards exercises every category plus the false-positive
// guards and the harness-injected "API Error" exclusion on one crafted transcript. The
// expected counts are the turn/score math the summary() emits.
func TestConfusionLensDetectsAndGuards(t *testing.T) {
	recs := []map[string]any{
		// self_correction: misread + reconsider (2 markers, 1 turn)
		assistantRecord("t1", 100, 0, 0, withText("I misread the config; let me reconsider the approach.")),
		// dead_end: still-broken + same-again (2 markers, 1 turn)
		assistantRecord("t2", 100, 0, 0, withText("The build is still failing with the same error again.")),
		// confusion: no-sense + confused (2 markers, 1 turn)
		assistantRecord("t3", 100, 0, 0, withText("That doesn't make sense — I'm confused why the test passes locally.")),
		// FALSE POSITIVES — none of these may fire: literal "wait", intensifier
		// "actually", in-progress "still running", uncataloged "sorry".
		assistantRecord("t4", 100, 0, 0, withText("Let me wait for the background task. What this actually does is important. The waiter is still running. Sorry, that is a long output.")),
		// harness-injected API error: excluded from numerator AND denominator.
		assistantRecord("t5", 100, 0, 0, withText("API Error: 500 overloaded")),
	}
	c := Analyze(writeTranscript(t, recs)).Confusion

	if c.TextTurns != 4 {
		t.Fatalf("text_turns = %d, want 4 (t5 API-error excluded)", c.TextTurns)
	}
	if c.TurnsWithConfusion != 3 {
		t.Fatalf("turns_with_confusion = %d, want 3", c.TurnsWithConfusion)
	}
	if c.SelfCorrectionTurns != 1 || c.DeadEndTurns != 1 || c.ConfusionTurns != 1 {
		t.Fatalf("category turns = sc:%d de:%d cf:%d, want 1/1/1", c.SelfCorrectionTurns, c.DeadEndTurns, c.ConfusionTurns)
	}
	if c.TotalMarkers != 6 {
		t.Fatalf("total_markers = %d, want 6", c.TotalMarkers)
	}
	if c.Score != 0.75 {
		t.Fatalf("score = %.3f, want 0.750", c.Score)
	}
	got := map[string]int64{}
	for _, m := range c.Markers {
		got[m.Label] = m.Count
	}
	for _, label := range []string{"misread", "reconsider", "still-broken", "same-again", "no-sense", "confused"} {
		if got[label] != 1 {
			t.Fatalf("marker %q count = %d, want 1; markers=%+v", label, got[label], c.Markers)
		}
	}
	// The false-positive turn must contribute no markers.
	for _, bad := range []string{"no-wait", "unexpected", "didnt-work"} {
		if got[bad] != 0 {
			t.Fatalf("false-positive marker %q fired (%d)", bad, got[bad])
		}
	}
}

// TestConfusionLensClean proves a transcript with only benign prose (and tool activity)
// produces a zero Confusion with an empty marker slice.
func TestConfusionLensClean(t *testing.T) {
	recs := []map[string]any{
		assistantRecord("a", 100, 0, 0, withText("I'll implement the fix and run the tests.")),
		assistantRecord("b", 100, 0, 0, withTool("Bash")),
		assistantRecord("c", 100, 0, 0, withText("Done — the tests pass and the tree is green.")),
	}
	c := Analyze(writeTranscript(t, recs)).Confusion
	if c.TotalMarkers != 0 || c.TurnsWithConfusion != 0 {
		t.Fatalf("clean transcript has markers: %+v", c)
	}
	if c.TextTurns != 2 {
		t.Fatalf("text_turns = %d, want 2", c.TextTurns)
	}
	if len(c.Markers) != 0 {
		t.Fatalf("expected empty markers, got %+v", c.Markers)
	}
	if c.Score != 0 {
		t.Fatalf("score = %.3f, want 0", c.Score)
	}
}

// TestCompactConfusionAggregatesAndRecommends folds the per-session lens across two
// sessions that share a recurring marker, and proves the window raises a
// confusion_pressure recommendation that lowers to an action pointed at the worst session.
func TestCompactConfusionAggregatesAndRecommends(t *testing.T) {
	root := t.TempDir()
	ns := "C--work-fak"
	// Two sessions, each confused (>= 3 markers), sharing the "still-broken" marker.
	writeTranscriptIn(t, root, ns, "sess-a.jsonl", []map[string]any{
		assistantRecord("a1", 100, 0, 0, withText("I misread the failure; it is still broken.")),
		assistantRecord("a2", 100, 0, 0, withText("Still failing, and I'm confused why.")),
	})
	writeTranscriptIn(t, root, ns, "sess-b.jsonl", []map[string]any{
		assistantRecord("b1", 100, 0, 0, withText("Let me reconsider; that still fails the same way.")),
		assistantRecord("b2", 100, 0, 0, withText("Still broken. I got that wrong earlier.")),
	})

	rep, err := BuildCompactReportFromDiscovery(DiscoverOptions{Roots: []string{root}, NamespacePrefix: ns}, false, 0, time.Now())
	if err != nil {
		t.Fatalf("build report: %v", err)
	}
	cc := rep.Confusion
	if cc == nil {
		t.Fatal("expected non-nil CompactConfusion")
	}
	if cc.ConfusedSessions != 2 {
		t.Fatalf("confused_sessions = %d, want 2", cc.ConfusedSessions)
	}
	var recurring *RecurringMarkerRow
	for i := range cc.RecurringMarkers {
		if cc.RecurringMarkers[i].Label == "still-broken" {
			recurring = &cc.RecurringMarkers[i]
		}
	}
	if recurring == nil || recurring.Sessions != 2 {
		t.Fatalf("expected recurring still-broken across 2 sessions, got %+v", cc.RecurringMarkers)
	}
	if len(cc.TopSessions) == 0 {
		t.Fatal("expected at least one confused top session")
	}

	var found bool
	for _, rec := range rep.Recommendations {
		if rec.Kind == "confusion_pressure" {
			found = true
			if rec.Severity == "" {
				t.Fatal("confusion_pressure recommendation has empty severity")
			}
		}
	}
	if !found {
		t.Fatalf("expected a confusion_pressure recommendation, got %+v", rep.Recommendations)
	}

	plan := BuildCompactActionPlan(rep)
	var action *CompactAction
	for i := range plan.Actions {
		if plan.Actions[i].Kind == "confusion_pressure" {
			action = &plan.Actions[i]
		}
	}
	if action == nil {
		t.Fatalf("expected a confusion_pressure action, got %+v", plan.Actions)
	}
	if action.Session == "" {
		t.Fatalf("confusion_pressure action has no target session: %+v", action)
	}
}

// confusedSess builds a Session whose prose is confused (shares the recurring
// "still-broken" dead-end marker). stuck toggles a Behavior tool-loop signature, which
// is what makes the confusion Behavior-VISIBLE (not silent).
func confusedSess(id string, stuck bool) Session {
	s := Session{
		Session: id,
		Confusion: Confusion{
			TextTurns:           5,
			TurnsWithConfusion:  4,
			SelfCorrectionTurns: 1,
			DeadEndTurns:        2,
			ConfusionTurns:      1,
			TotalMarkers:        4,
			Score:               0.8,
			Markers: []ConfusionMarkerRow{
				{Category: catDeadEnd, Label: "still-broken", Count: 2, Example: "still broken"},
				{Category: catSelfCorrection, Label: "misread", Count: 1, Example: "I misread it"},
			},
		},
	}
	if stuck {
		s.Behavior = Behavior{RepeatFailures: []RepeatFailureRow{{Tool: "Bash", Sig: "FAIL: boom", Count: 3}}}
	}
	return s
}

// TestConfusionPressureSilentSliceGate is the complementarity guard: two sessions that
// are prose-confused AND already tool-stuck (Behavior owns them) must NOT raise a
// confusion_pressure recommendation — even though they share a recurring marker across
// two sessions — because there is no Behavior-silent confused session. Flipping the
// sessions silent (no tool-loop) makes the SAME confusion actionable. This is what keeps
// the lens from restating a process_issue_pressure finding.
func TestConfusionPressureSilentSliceGate(t *testing.T) {
	// Both confused sessions are ALSO tool-stuck -> Behavior already flags them.
	stuck := []Session{confusedSess("sess-a", true), confusedSess("sess-b", true)}
	cc := aggregateCompactConfusion(stuck)
	if cc == nil {
		t.Fatal("expected non-nil CompactConfusion (markers present)")
	}
	if cc.ConfusedSessions != 2 {
		t.Fatalf("confused_sessions = %d, want 2", cc.ConfusedSessions)
	}
	if cc.SilentConfusedSessions != 0 {
		t.Fatalf("silent_confused_sessions = %d, want 0 (both Behavior-stuck)", cc.SilentConfusedSessions)
	}
	// The recurring "still-broken" marker spans 2 sessions, yet the recommendation must
	// stay silent because Behavior already owns every confused session.
	if _, ok := compactConfusionPressure(cc); ok {
		t.Fatal("confusion_pressure fired on a window with zero Behavior-silent confused sessions")
	}

	// Flip both silent: identical prose confusion, but now Behavior is blind to it.
	silent := []Session{confusedSess("sess-a", false), confusedSess("sess-b", false)}
	cc2 := aggregateCompactConfusion(silent)
	if cc2.SilentConfusedSessions != 2 {
		t.Fatalf("silent_confused_sessions = %d, want 2", cc2.SilentConfusedSessions)
	}
	rec, ok := compactConfusionPressure(cc2)
	if !ok {
		t.Fatal("expected confusion_pressure on 2 Behavior-silent confused sessions")
	}
	if rec.Severity == "" {
		t.Fatal("confusion_pressure has empty severity")
	}
	// Behavior-silent confused rows rank ahead of corroborated ones for triage.
	if len(cc2.TopSessions) == 0 || !cc2.TopSessions[0].Silent {
		t.Fatalf("expected a silent confused session ranked first, got %+v", cc2.TopSessions)
	}
}
