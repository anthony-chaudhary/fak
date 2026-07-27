package sessionaudit

import (
	"strings"
	"testing"
)

// withSidechain marks a fixture record as a delegated (sub-agent / background) turn, the
// way the harness stamps one.
func withSidechain() func(map[string]any) {
	return func(rec map[string]any) { rec["isSidechain"] = true }
}

func mustShare(t *testing.T, d DelegationShare) float64 {
	t.Helper()
	if d.OutputShare == nil {
		t.Fatalf("delegation share is UNKNOWN, want a number: %+v", d)
	}
	return *d.OutputShare
}

// The point of the whole file: the lever gets a size.
func TestDelegationShareSizesTheLever(t *testing.T) {
	d := FoldDelegationShare(map[string]ModelCounts{
		TrackDelegation: {Turns: 8, Output: 750},
		TrackMain:       {Turns: 4, Output: 250},
	}, map[string]int64{"Task": 3})

	if got := mustShare(t, d); got != 0.75 {
		t.Fatalf("delegated share = %v, want 0.75 (750 of 1000 generated tokens were delegated work)", got)
	}
	if d.UnderInstrumented {
		t.Fatal("a corpus whose delegated turns ARE marked must not be called under-instrumented")
	}
	if d.SpawnCalls != 3 {
		t.Fatalf("spawn calls = %d, want 3", d.SpawnCalls)
	}
}

// THE HEADLINE REFUSAL. A corpus that spawned work but marked none of it must not be able
// to report that delegation costs nothing — that zero is the strongest available argument
// against track E and it would be manufactured entirely out of a missing field.
func TestUninstrumentedCorpusWithSpawnsReportsUnknownNotZero(t *testing.T) {
	d := FoldDelegationShare(map[string]ModelCounts{
		TrackMain: {Turns: 40, Output: 900_000},
	}, map[string]int64{"Task": 12})

	if !d.UnderInstrumented {
		t.Fatal("12 spawn calls and zero marked delegated turns is under-instrumentation, and must be reported as such")
	}
	if d.OutputShare != nil {
		t.Fatalf("delegated share = %v, want UNKNOWN (nil): a 0%% here reads as \"delegation generates no volume\", which this corpus did not measure", *d.OutputShare)
	}
	// Coverage still stands: the tracked/untracked split is a different question, and it
	// IS answerable here.
	if d.Coverage == nil || *d.Coverage != 1 {
		t.Fatalf("coverage = %v, want 1 (every token had a track key, they were just all main)", d.Coverage)
	}
}

// The other side of the same rule: a corpus that never spawned anything genuinely has no
// delegated volume, and that zero IS earned. Refusing to answer here would make the
// measure useless.
func TestCorpusThatNeverSpawnedReportsAnEarnedZero(t *testing.T) {
	d := FoldDelegationShare(map[string]ModelCounts{
		TrackMain: {Turns: 10, Output: 5_000},
	}, map[string]int64{"Bash": 30, "Read": 90})

	if d.UnderInstrumented {
		t.Fatal("no spawn calls means nothing was delegated; that is a measurement, not a gap")
	}
	if got := mustShare(t, d); got != 0 {
		t.Fatalf("delegated share = %v, want 0", got)
	}
}

// The contradiction is checked on TURNS, not output. A delegated turn that generated
// nothing still proves the marker is being written, which is the only thing in question.
func TestAMarkedButSilentDelegatedTurnStillProvesInstrumentation(t *testing.T) {
	d := FoldDelegationShare(map[string]ModelCounts{
		TrackDelegation: {Turns: 1, Output: 0},
		TrackMain:       {Turns: 9, Output: 400},
	}, map[string]int64{"Agent": 1})

	if d.UnderInstrumented {
		t.Fatal("a marked delegated turn proves the marker is written even when it generated no output")
	}
	if got := mustShare(t, d); got != 0 {
		t.Fatalf("delegated share = %v, want 0 (the marker exists; the volume really is zero)", got)
	}
}

// Totality over an OPEN map: a key outside the closed vocabulary may not be quietly folded
// into either side of the fraction, because either choice moves the headline number.
func TestUnknownTrackIsExcludedFromBothSides(t *testing.T) {
	d := FoldDelegationShare(map[string]ModelCounts{
		TrackDelegation: {Turns: 1, Output: 100},
		TrackMain:       {Turns: 1, Output: 100},
		"sideloaded":    {Turns: 5, Output: 800},
	}, map[string]int64{"Task": 1})

	if d.Untracked.Output != 800 {
		t.Fatalf("untracked output = %d, want 800 (an unrecognized key lands in the excluded bucket)", d.Untracked.Output)
	}
	if got := mustShare(t, d); got != 0.5 {
		t.Fatalf("delegated share = %v, want 0.5 — the unknown key must not move the fraction", got)
	}
	if d.Coverage == nil || *d.Coverage != 0.2 {
		t.Fatalf("coverage = %v, want 0.2 (200 of 1000 generated tokens carry a track)", d.Coverage)
	}
	if d.TotalOutput() != 1000 {
		t.Fatalf("total output = %d, want 1000 — no volume may be dropped", d.TotalOutput())
	}
}

func TestNoVolumeAtAllHasNoAnswer(t *testing.T) {
	d := FoldDelegationShare(nil, nil)
	if d.OutputShare != nil || d.Coverage != nil {
		t.Fatalf("an empty corpus must report no answer, got share=%v coverage=%v", d.OutputShare, d.Coverage)
	}
	if d.UnderInstrumented {
		t.Fatal("an empty corpus spawned nothing, so there is no contradiction to report")
	}
}

func TestTrackForSidechain(t *testing.T) {
	if got := TrackForSidechain(true); got != TrackDelegation {
		t.Fatalf("TrackForSidechain(true) = %q, want %q", got, TrackDelegation)
	}
	if got := TrackForSidechain(false); got != TrackMain {
		t.Fatalf("TrackForSidechain(false) = %q, want %q", got, TrackMain)
	}
}

// The tool names are matched EXACTLY. TaskCreate and friends are todo-list bookkeeping
// and spawn nothing, but they are frequent, and a prefix match on "Task" would make almost
// every corpus look like it delegated — which would fire the under-instrumentation warning
// on corpora that simply kept a todo list.
func TestSpawnCallsCountsOnlySpawnTools(t *testing.T) {
	got := SpawnCalls(map[string]int64{
		"Task": 2, "Agent": 3, "Workflow": 1,
		"TaskCreate": 353, "TaskUpdate": 69, "TaskList": 3, "TaskStop": 4, "TaskOutput": 4,
		"Bash": 99, "Read": 40,
	})
	if got != 6 {
		t.Fatalf("spawn calls = %d, want 6 (Task+Agent+Workflow only; the Task* todo tools spawn nothing)", got)
	}
	if n := SpawnCalls(nil); n != 0 {
		t.Fatalf("SpawnCalls(nil) = %d, want 0", n)
	}
}

// END TO END from a transcript: the marker actually reaches the fold.
func TestAnalyzeSplitsDelegatedFromMainTurns(t *testing.T) {
	s := Analyze(writeTranscript(t, []map[string]any{
		assistantRecord("main-1", 100, 0, 0, withTool("Task")),
		assistantRecord("sub-1", 300, 0, 0, withSidechain()),
		assistantRecord("sub-2", 200, 0, 0, withSidechain()),
	}))
	if s.Error != "" {
		t.Fatal(s.Error)
	}
	if got := s.PerTrack[TrackDelegation]; got.Turns != 2 || got.Output != 500 {
		t.Fatalf("delegated track = %+v, want 2 turns / 500 output", got)
	}
	if got := s.PerTrack[TrackMain]; got.Turns != 1 || got.Output != 100 {
		t.Fatalf("main track = %+v, want 1 turn / 100 output", got)
	}

	agg := AggregateSessions([]Session{s})
	d := FoldDelegationShare(agg.PerTrack, agg.ToolMix)
	if got := mustShare(t, d); got < 0.833 || got > 0.834 {
		t.Fatalf("delegated share = %v, want ~0.8333 (500 of 600)", got)
	}
	if d.SpawnCalls != 1 {
		t.Fatalf("spawn calls = %d, want 1 (the Task call)", d.SpawnCalls)
	}
}

// The delegation split must inherit the SAME dedup the model split gets: Claude Code emits
// one assistant response as several records under one message id, so a delegated turn
// would otherwise count two or three times and inflate exactly the number this file exists
// to report honestly.
func TestSplitDelegatedRecordsAreCountedOnce(t *testing.T) {
	s := Analyze(writeTranscript(t, []map[string]any{
		assistantRecord("sub-1", 300, 0, 0, withSidechain()),
		assistantRecord("sub-1", 300, 0, 0, withSidechain()),
		assistantRecord("sub-1", 300, 0, 0, withSidechain()),
		assistantRecord("main-1", 100, 0, 0),
	}))
	if got := s.PerTrack[TrackDelegation]; got.Turns != 1 || got.Output != 300 {
		t.Fatalf("delegated track = %+v, want 1 turn / 300 output — the split records are ONE turn", got)
	}
}

// The two folds of the same volume must agree. If they ever diverge, one of the two
// headline numbers in this report is wrong and nothing else would say so.
func TestPerTrackAndPerModelAccountForTheSameVolume(t *testing.T) {
	s := Analyze(writeTranscript(t, []map[string]any{
		assistantRecord("a", 100, 10, 1, withModel("claude-opus-4-8")),
		assistantRecord("b", 250, 20, 2, withModel("qwen/qwen3.6-4b"), withSidechain()),
		assistantRecord("c", 40, 30, 3, withModel("claude-opus-4-8"), withSidechain()),
	}))
	agg := AggregateSessions([]Session{s})

	var byModel, byTrack ModelCounts
	for _, c := range agg.PerModel {
		addCounts(&byModel, c)
	}
	for _, c := range agg.PerTrack {
		addCounts(&byTrack, c)
	}
	if byModel != byTrack {
		t.Fatalf("the same turns folded two ways disagree:\n  per-model = %+v\n  per-track = %+v", byModel, byTrack)
	}
	if byTrack.Output != 390 {
		t.Fatalf("total output = %d, want 390", byTrack.Output)
	}
	// And delegation is not a property of the model: one id served both tracks here.
	if agg.PerTrack[TrackDelegation].Turns != 2 {
		t.Fatalf("delegated turns = %d, want 2 (one qwen, one opus)", agg.PerTrack[TrackDelegation].Turns)
	}
}

// The refusal has to survive all the way to the rendered page, because the page is what an
// operator actually reads.
func TestRenderRefusesToPrintZeroForAnUninstrumentedCorpus(t *testing.T) {
	var b strings.Builder
	writeDelegationShare(&b, Aggregate{
		PerTrack: map[string]ModelCounts{TrackMain: {Turns: 40, Output: 900_000}},
		ToolMix:  map[string]int64{"Task": 12},
	})
	out := b.String()
	if !strings.Contains(out, "UNKNOWN") {
		t.Fatalf("rendered page must say UNKNOWN for an uninstrumented corpus:\n%s", out)
	}
	if strings.Contains(out, "share of tracked output = 0") {
		t.Fatalf("rendered page printed an unearned 0%%:\n%s", out)
	}
	if !strings.Contains(out, "instrumentation gap") {
		t.Fatalf("the page must name WHY it cannot answer:\n%s", out)
	}
}

func TestRenderPrintsTheShareWhenItIsEarned(t *testing.T) {
	var b strings.Builder
	writeDelegationShare(&b, Aggregate{
		PerTrack: map[string]ModelCounts{
			TrackDelegation: {Turns: 8, Output: 750},
			TrackMain:       {Turns: 4, Output: 250},
		},
		ToolMix: map[string]int64{"Task": 3},
	})
	out := b.String()
	if !strings.Contains(out, "75") {
		t.Fatalf("rendered page must carry the share:\n%s", out)
	}
	if strings.Contains(out, "UNKNOWN") {
		t.Fatalf("an instrumented corpus must not render UNKNOWN:\n%s", out)
	}
}
