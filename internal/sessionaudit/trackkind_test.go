package sessionaudit

import "testing"

// THE REGRESSION THIS FILE EXISTS FOR, and it is measured rather than hypothetical.
//
// On the corpus described in delegated.go's header, this harness wrote every delegated
// turn into its OWN transcript: 1,524 of 1,524 turns in the 59 spawned transcripts carried
// the `isSidechain` marker, and 0 of 5,755 turns in the 111 top-level transcripts did. A
// rollup that folds only top-level transcripts therefore sees spawn-tool calls and not one
// marked turn — the exact under-instrumented shape — and reports UNKNOWN while the answer
// sits in the transcripts it declined to fold.
//
// So the transcript KIND is a second attribution signal. These tests pin both halves: it
// rescues an unmarked spawned transcript, and it never overrides a marker.

func TestTrackForKind(t *testing.T) {
	for _, tc := range []struct{ name, kind, track, want string }{
		// The fallback: the file itself says this was delegated work.
		{"spawned transcript, records unmarked", KindSpawned, TrackMain, TrackDelegation},
		{"spawned transcript, records marked", KindSpawned, TrackDelegation, TrackDelegation},
		// ...and it only ever runs that one way. A per-record marker is a claim; a
		// per-file kind is an inference, and the claim outranks it.
		{"top-level transcript, records unmarked", KindTop, TrackMain, TrackMain},
		{"top-level transcript, marked record stays delegated", KindTop, TrackDelegation, TrackDelegation},
		// Analyze does not set Kind — only a caller holding the Discover record does — so
		// an unset kind must be inert rather than guessed at.
		{"unset kind changes nothing", "", TrackMain, TrackMain},
		{"unknown track is passed through untouched", KindSpawned, "future-track", "future-track"},
	} {
		if got := TrackForKind(tc.kind, tc.track); got != tc.want {
			t.Errorf("%s: TrackForKind(%q, %q) = %q, want %q", tc.name, tc.kind, tc.track, got, tc.want)
		}
	}
}

// The end-to-end shape of the live defect: a top-level session that SPAWNED work, plus the
// spawned transcript itself, whose records carry no marker. Folding only the marker gives
// UNKNOWN; folding the kind as well gives the real fraction.
func TestAnUnmarkedSpawnedTranscriptIsStillDelegatedVolume(t *testing.T) {
	parent := Analyze(writeTranscript(t, []map[string]any{
		assistantRecord("p1", 900, 0, 0, withToolID("Agent", "t1")),
	}))
	parent.Kind = KindTop
	child := Analyze(writeTranscript(t, []map[string]any{
		assistantRecord("c1", 100, 0, 0), // no withSidechain: the marker is absent
	}))
	child.Kind = KindSpawned

	// Without the kind signal this is precisely the corpus that reports UNKNOWN.
	blind := FoldDelegationShare(AggregateSessions([]Session{parent, unkinded(child)}).PerTrack, map[string]int64{"Agent": 1})
	if !blind.UnderInstrumented || blind.OutputShare != nil {
		t.Fatalf("marker-only fold should be UNKNOWN here, got under_instrumented=%v share=%v",
			blind.UnderInstrumented, blind.OutputShare)
	}

	agg := AggregateSessions([]Session{parent, child})
	d := FoldDelegationShare(agg.PerTrack, agg.ToolMix)
	if d.UnderInstrumented {
		t.Fatal("a corpus whose spawned transcripts are folded in is not under-instrumented")
	}
	if d.Delegation.Turns != 1 || d.Delegation.Output != 100 {
		t.Fatalf("delegated = %d turns / %d output, want 1 / 100 — the spawned transcript IS the delegated work",
			d.Delegation.Turns, d.Delegation.Output)
	}
	if d.Main.Turns != 1 || d.Main.Output != 900 {
		t.Fatalf("main = %d turns / %d output, want 1 / 900", d.Main.Turns, d.Main.Output)
	}
	if got := mustShare(t, d); got != 0.1 {
		t.Fatalf("delegated share = %v, want 0.1", got)
	}
}

// unkinded is the before picture: the same analysis with the transcript kind dropped, which
// is what every caller had before Kind reached the rollup.
func unkinded(s Session) Session {
	s.Kind = ""
	return s
}

// The fallback must not manufacture delegation out of an ordinary session. A top-level
// transcript with no marker is main-thread volume and stays that way.
func TestATopLevelTranscriptIsNotReattributed(t *testing.T) {
	s := Analyze(writeTranscript(t, []map[string]any{assistantRecord("m1", 400, 0, 0)}))
	s.Kind = KindTop
	agg := AggregateSessions([]Session{s})
	d := FoldDelegationShare(agg.PerTrack, agg.ToolMix)
	if d.Delegation.Turns != 0 || d.Main.Output != 400 {
		t.Fatalf("delegated=%d main_out=%d, want 0 / 400", d.Delegation.Turns, d.Main.Output)
	}
	// No spawn call was made, so a zero here is EARNED and must print as a zero.
	if d.OutputShare == nil || *d.OutputShare != 0 {
		t.Fatalf("share = %v, want an earned 0", d.OutputShare)
	}
}

// A spawned transcript whose records ARE marked must count once, not twice: the kind
// fallback moves main-track volume, and there is none to move.
func TestAMarkedSpawnedTranscriptIsNotDoubleCounted(t *testing.T) {
	s := Analyze(writeTranscript(t, []map[string]any{
		assistantRecord("c1", 250, 0, 0, withSidechain()),
	}))
	s.Kind = KindSpawned
	agg := AggregateSessions([]Session{s})
	d := FoldDelegationShare(agg.PerTrack, agg.ToolMix)
	if d.Delegation.Turns != 1 || d.Delegation.Output != 250 {
		t.Fatalf("delegated = %d turns / %d output, want 1 / 250", d.Delegation.Turns, d.Delegation.Output)
	}
	if d.TotalOutput() != 250 {
		t.Fatalf("total output = %d, want 250 — the same turn must not be counted on both signals", d.TotalOutput())
	}
}
