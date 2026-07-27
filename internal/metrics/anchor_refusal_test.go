package metrics

import (
	"strings"
	"testing"
)

// observeAll folds a synthetic run of placement outcomes through the monitor and
// returns every verdict, so a test can assert on WHICH turn the finding fired
// rather than only that it fired.
func observeAll(m *AnchorRefusalMonitor, outcomes []string) []AnchorRefusalVerdict {
	out := make([]AnchorRefusalVerdict, 0, len(outcomes))
	for _, o := range outcomes {
		out = append(out, m.Observe(o))
	}
	return out
}

// findings collects the turns that raised a finding.
func findings(vs []AnchorRefusalVerdict) []AnchorRefusalVerdict {
	var out []AnchorRefusalVerdict
	for _, v := range vs {
		if v.Finding != "" {
			out = append(out, v)
		}
	}
	return out
}

func repeat(outcome string, n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = outcome
	}
	return out
}

// TestAnchorOutcomeClassification pins the closed reduction. The outcome strings
// mirror internal/agent's BreakpointReason* vocabulary verbatim; metrics is a
// lower tier than agent and must not import it, so the vocabulary is restated
// here and this test is the drift alarm.
func TestAnchorOutcomeClassification(t *testing.T) {
	cases := map[string]AnchorPlacementClass{
		"":                                     AnchorEarned, // agent.BreakpointReasonNone
		"placed":                               AnchorEarned, // the metric's spelling
		"volatile_head":                        AnchorRefused,
		"no_stable_head":                       AnchorRefused,
		"splice_failed":                        AnchorRefused,
		"redecode_failed":                      AnchorRefused,
		"already_set":                          AnchorDeferred,
		"non_json":                             AnchorInapplicable,
		"a_reason_this_monitor_has_never_seen": AnchorUnknown,
	}
	for outcome, want := range cases {
		if got := ClassifyAnchorOutcome(outcome); got != want {
			t.Errorf("ClassifyAnchorOutcome(%q) = %q, want %q", outcome, got, want)
		}
	}
	if !AnchorEarned.Decisive() || !AnchorRefused.Decisive() {
		t.Error("earned and refused must both be decisive — they are the fraction's terms")
	}
	for _, c := range []AnchorPlacementClass{AnchorDeferred, AnchorInapplicable, AnchorUnknown} {
		if c.Decisive() {
			t.Errorf("class %q must not be decisive: it is not evidence about whether the anchor earned caching", c)
		}
	}
}

// TestVolatileHeadSessionRaisesFinding is the issue's done condition, first half:
// a session whose head becomes volatile across turns surfaces
// ANCHOR_REFUSED_RISING.
func TestVolatileHeadSessionRaisesFinding(t *testing.T) {
	m := NewAnchorRefusalMonitor(AnchorRefusalThresholds{})
	// A healthy opening, then the head turns volatile and stays that way.
	vs := observeAll(m, append(repeat("placed", 4), repeat("volatile_head", 6)...))

	raised := findings(vs)
	if len(raised) != 1 {
		t.Fatalf("want exactly one ANCHOR_REFUSED_RISING crossing, got %d", len(raised))
	}
	v := raised[0]
	if v.Finding != AnchorFindingRefusedRising {
		t.Errorf("finding = %q, want %q", v.Finding, AnchorFindingRefusedRising)
	}
	if v.RefusedFraction < DefaultAnchorThreshold {
		t.Errorf("raised at fraction %.2f, below threshold %.2f", v.RefusedFraction, DefaultAnchorThreshold)
	}
	if v.TopRefusal != "volatile_head" {
		t.Errorf("top refusal = %q, want volatile_head", v.TopRefusal)
	}
	if !strings.Contains(v.Banner, AnchorFindingRefusedRising) || !strings.Contains(v.Banner, "volatile_head") {
		t.Errorf("banner does not name the finding and its cause: %q", v.Banner)
	}

	r := m.Report()
	if r.Findings != 1 || !r.Alarmed {
		t.Errorf("report findings=%d alarmed=%v, want 1/true", r.Findings, r.Alarmed)
	}
	if r.Refused != 6 || r.Earned != 4 {
		t.Errorf("report earned/refused = %d/%d, want 4/6", r.Earned, r.Refused)
	}
	if !strings.Contains(r.BannerRow(), AnchorFindingRefusedRising) {
		t.Errorf("banner row does not carry the finding: %q", r.BannerRow())
	}
}

// TestStableAnchorSessionStaysQuiet is the done condition's second half: a
// session whose anchor keeps placing raises nothing.
func TestStableAnchorSessionStaysQuiet(t *testing.T) {
	m := NewAnchorRefusalMonitor(AnchorRefusalThresholds{})
	vs := observeAll(m, repeat("placed", 20))
	if raised := findings(vs); len(raised) != 0 {
		t.Fatalf("a stable-anchor session must raise nothing, got %d finding(s)", len(raised))
	}
	r := m.Report()
	if r.Alarmed || r.Findings != 0 {
		t.Errorf("report alarmed=%v findings=%d, want false/0", r.Alarmed, r.Findings)
	}
	if r.RefusedFraction != 0 {
		t.Errorf("refused fraction = %.2f, want 0", r.RefusedFraction)
	}
	if !strings.Contains(r.BannerRow(), "earning") {
		t.Errorf("banner row should read as earning: %q", r.BannerRow())
	}
}

// TestAlreadySetSessionNeverAlarms is the false-positive guard the classification
// exists for: a Claude-Code-shaped session is ~100% already_set, fak defers to the
// client's own breakpoint, and that must never be priced as a refusal.
func TestAlreadySetSessionNeverAlarms(t *testing.T) {
	m := NewAnchorRefusalMonitor(AnchorRefusalThresholds{})
	vs := observeAll(m, repeat("already_set", 30))
	if raised := findings(vs); len(raised) != 0 {
		t.Fatalf("an already_set session must never alarm, got %d finding(s)", len(raised))
	}
	r := m.Report()
	if r.Deferred != 30 || r.Refused != 0 || r.WindowDecisive != 0 {
		t.Errorf("deferred=%d refused=%d decisive=%d, want 30/0/0", r.Deferred, r.Refused, r.WindowDecisive)
	}
	if !strings.Contains(r.BannerRow(), "nothing to judge") {
		t.Errorf("a session with no decisive turn must say so, not report a measured 0%%: %q", r.BannerRow())
	}
}

// TestNonDecisiveTurnsDoNotMoveTheWindow proves deferred/inapplicable/unknown
// turns are recorded but cannot dilute a volatile stretch out of alarming.
func TestNonDecisiveTurnsDoNotMoveTheWindow(t *testing.T) {
	m := NewAnchorRefusalMonitor(AnchorRefusalThresholds{})
	// Four refusals interleaved with noise that must not count.
	vs := observeAll(m, []string{
		"volatile_head", "already_set", "non_json",
		"volatile_head", "mystery_reason", "already_set",
		"volatile_head", "non_json",
		"volatile_head",
	})
	raised := findings(vs)
	if len(raised) != 1 {
		t.Fatalf("want one crossing once the 4-sample floor of refusals is met, got %d", len(raised))
	}
	if raised[0].WindowDecisive != DefaultAnchorMinSamples {
		t.Errorf("crossed with %d decisive turns, want the %d-sample floor", raised[0].WindowDecisive, DefaultAnchorMinSamples)
	}
	r := m.Report()
	if r.Unknown != 1 || r.Deferred != 2 || r.Inapplicable != 2 {
		t.Errorf("unknown/deferred/inapplicable = %d/%d/%d, want 1/2/2", r.Unknown, r.Deferred, r.Inapplicable)
	}
}

// TestUnknownOutcomeNeverArmsTheAlarm proves an out-of-vocabulary outcome is
// recorded for drift visibility but cannot be counted as a refusal — alarming on
// a string this monitor has never seen would be a fabricated verdict.
func TestUnknownOutcomeNeverArmsTheAlarm(t *testing.T) {
	m := NewAnchorRefusalMonitor(AnchorRefusalThresholds{})
	vs := observeAll(m, repeat("some_future_bail_reason", 40))
	if raised := findings(vs); len(raised) != 0 {
		t.Fatalf("unknown outcomes must never arm the alarm, got %d finding(s)", len(raised))
	}
	r := m.Report()
	if r.Unknown != 40 {
		t.Errorf("unknown = %d, want 40 (recorded, not dropped)", r.Unknown)
	}
	if r.WindowDecisive != 0 || r.RefusedFraction != 0 {
		t.Errorf("unknown turns must not enter the window: decisive=%d fraction=%.2f", r.WindowDecisive, r.RefusedFraction)
	}
}

// TestFindingIsEdgeTriggered proves a long volatile stretch raises the finding
// ONCE — RISING is a transition, and a monitor that re-raised every turn would be
// a stuck horn rather than an alarm.
func TestFindingIsEdgeTriggered(t *testing.T) {
	m := NewAnchorRefusalMonitor(AnchorRefusalThresholds{})
	vs := observeAll(m, repeat("volatile_head", 25))
	raised := findings(vs)
	if len(raised) != 1 {
		t.Fatalf("want exactly one crossing across a long volatile stretch, got %d", len(raised))
	}
	if raised[0].ObservedTurn != DefaultAnchorMinSamples {
		t.Errorf("crossed on turn %d, want the first turn that meets the %d-sample floor",
			raised[0].ObservedTurn, DefaultAnchorMinSamples)
	}
	// Every later turn stays alarmed but silent.
	for _, v := range vs[DefaultAnchorMinSamples:] {
		if v.Finding != "" {
			t.Fatalf("turn %d re-raised the finding while already alarming", v.ObservedTurn)
		}
		if !v.Alarmed {
			t.Fatalf("turn %d dropped the alarmed state mid-stretch", v.ObservedTurn)
		}
	}
}

// TestAlarmRearmsAfterRecovery proves the alarm clears when the anchor starts
// earning again, and can fire a SECOND time if the head turns volatile once more.
func TestAlarmRearmsAfterRecovery(t *testing.T) {
	m := NewAnchorRefusalMonitor(AnchorRefusalThresholds{})
	run := append(repeat("volatile_head", 5), repeat("placed", 8)...)
	run = append(run, repeat("volatile_head", 8)...)
	vs := observeAll(m, run)

	raised := findings(vs)
	if len(raised) != 2 {
		t.Fatalf("want two crossings (volatile -> recovered -> volatile again), got %d", len(raised))
	}
	// After the all-placed stretch the monitor must have cleared.
	recovered := vs[5+8-1]
	if recovered.Alarmed || recovered.RefusedFraction != 0 {
		t.Errorf("after a full window of placements: alarmed=%v fraction=%.2f, want false/0",
			recovered.Alarmed, recovered.RefusedFraction)
	}
	if r := m.Report(); r.Findings != 2 || !r.Alarmed {
		t.Errorf("report findings=%d alarmed=%v, want 2/true", r.Findings, r.Alarmed)
	}
}

// TestMinSampleFloorBlocksEarlyAlarm proves one unlucky bail on a cold session
// cannot fire a 100%% fraction.
func TestMinSampleFloorBlocksEarlyAlarm(t *testing.T) {
	m := NewAnchorRefusalMonitor(AnchorRefusalThresholds{})
	vs := observeAll(m, repeat("volatile_head", DefaultAnchorMinSamples-1))
	if raised := findings(vs); len(raised) != 0 {
		t.Fatalf("below the %d-sample floor nothing may be raised, got %d finding(s)",
			DefaultAnchorMinSamples, len(raised))
	}
	last := vs[len(vs)-1]
	if last.RefusedFraction != 1 {
		t.Errorf("fraction = %.2f, want 1 (measured but not actionable yet)", last.RefusedFraction)
	}
	if !strings.Contains(last.Reason, "floor") {
		t.Errorf("reason should name the sample floor: %q", last.Reason)
	}
}

// TestThresholdsNormalization proves a zero or nonsensical tuning folds to the
// documented defaults rather than arming a horn that is always on.
func TestThresholdsNormalization(t *testing.T) {
	got := NewAnchorRefusalMonitor(AnchorRefusalThresholds{}).Thresholds()
	want := AnchorRefusalThresholds{Window: DefaultAnchorWindow, MinSamples: DefaultAnchorMinSamples, Threshold: DefaultAnchorThreshold}
	if got != want {
		t.Errorf("zero tuning normalized to %+v, want %+v", got, want)
	}
	// A zero threshold would alarm on a clean session; a >1 threshold could never
	// fire. Both fold.
	for _, bad := range []float64{0, -1, 1.5} {
		if p := NewAnchorRefusalMonitor(AnchorRefusalThresholds{Threshold: bad}).Thresholds(); p.Threshold != DefaultAnchorThreshold {
			t.Errorf("threshold %v normalized to %v, want the default", bad, p.Threshold)
		}
	}
	// MinSamples can never exceed the window, or the alarm could never fire.
	if p := (AnchorRefusalThresholds{Window: 3, MinSamples: 99}).normalize(); p.MinSamples != 3 {
		t.Errorf("min samples clamped to %d, want the window size 3", p.MinSamples)
	}
}

// TestTunedThresholdsAreHonored proves the rolling threshold is a real lever: a
// stricter tuning alarms on a mix the default tolerates.
//
// The mix is chosen by its ROLLING peak, not its final average. Its fraction
// peaks at 0.25 (one refusal among the first four decisive turns) and decays from
// there, so it crosses a 0.25 threshold exactly once and never approaches 0.5. An
// alternating placed/volatile mix would NOT work here: it averages 33% but
// transiently sits at exactly 2-of-4 == 0.5, which is a genuine crossing of the
// default threshold.
func TestTunedThresholdsAreHonored(t *testing.T) {
	mix := []string{"placed", "placed", "placed", "volatile_head", "placed", "placed"}

	lenient := NewAnchorRefusalMonitor(AnchorRefusalThresholds{})
	if raised := findings(observeAll(lenient, mix)); len(raised) != 0 {
		t.Fatalf("a mix peaking at 25%% must not cross the default 50%% threshold, got %d finding(s)", len(raised))
	}

	strict := NewAnchorRefusalMonitor(AnchorRefusalThresholds{Window: 6, MinSamples: 4, Threshold: 0.25})
	if raised := findings(observeAll(strict, mix)); len(raised) != 1 {
		t.Fatalf("a 25%%-threshold monitor must cross on the same mix, got %d finding(s)", len(raised))
	}
}

// TestThresholdIsInclusive pins the boundary the tuned-thresholds fixture turns on: a
// fraction EXACTLY at the threshold crosses. Half the anchor's turns being
// refused is a refusal rate worth raising, and the docs say ">=".
func TestThresholdIsInclusive(t *testing.T) {
	m := NewAnchorRefusalMonitor(AnchorRefusalThresholds{})
	vs := observeAll(m, []string{"placed", "volatile_head", "placed", "volatile_head"})
	raised := findings(vs)
	if len(raised) != 1 {
		t.Fatalf("an exactly-at-threshold fraction must cross, got %d finding(s)", len(raised))
	}
	if raised[0].RefusedFraction != DefaultAnchorThreshold {
		t.Errorf("crossed at fraction %.2f, want exactly the %.2f threshold",
			raised[0].RefusedFraction, DefaultAnchorThreshold)
	}
}

// TestReportByOutcomeIsDeterministic proves the mix folds in a stable order, so
// the banner and any journalled report are byte-reproducible.
func TestReportByOutcomeIsDeterministic(t *testing.T) {
	build := func() AnchorRefusalReport {
		m := NewAnchorRefusalMonitor(AnchorRefusalThresholds{})
		observeAll(m, []string{"placed", "volatile_head", "placed", "already_set", "non_json", "placed", "volatile_head"})
		return m.Report()
	}
	first := build()
	for i := 0; i < 5; i++ {
		got := build()
		if len(got.ByOutcome) != len(first.ByOutcome) {
			t.Fatalf("fold length drifted: %d vs %d", len(got.ByOutcome), len(first.ByOutcome))
		}
		for j := range got.ByOutcome {
			if got.ByOutcome[j] != first.ByOutcome[j] {
				t.Fatalf("fold order drifted at %d: %+v vs %+v", j, got.ByOutcome[j], first.ByOutcome[j])
			}
		}
	}
	if first.ByOutcome[0].Outcome != "placed" || first.ByOutcome[0].Turns != 3 {
		t.Errorf("most-frequent outcome should lead the fold, got %+v", first.ByOutcome[0])
	}
	if first.Turns != 7 {
		t.Errorf("turns = %d, want 7", first.Turns)
	}
}

// TestNilMonitorIsSafe mirrors the nil-receiver posture the gateway's metrics
// sinks carry, so wiring the monitor into a path with metrics disabled cannot
// panic.
func TestNilMonitorIsSafe(t *testing.T) {
	var m *AnchorRefusalMonitor
	if v := m.Observe("volatile_head"); v.Finding != "" || v.ObservedTurn != 0 {
		t.Errorf("nil monitor produced a verdict: %+v", v)
	}
	if r := m.Report(); r.Turns != 0 {
		t.Errorf("nil monitor produced a report: %+v", r)
	}
	if got := (AnchorRefusalReport{}).BannerRow(); !strings.Contains(got, "no placement attempts") {
		t.Errorf("empty report banner = %q", got)
	}
}
