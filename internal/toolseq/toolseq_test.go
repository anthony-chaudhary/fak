package toolseq

import (
	"math"
	"reflect"
	"testing"
)

// corpus is two sessions sharing a common read->edit opening, used across the
// transition and n-gram tests so the expected counts are easy to trace by hand.
//
//	s1: read edit read bash
//	s2: read edit test
var corpus = [][]string{
	{"read", "edit", "read", "bash"},
	{"read", "edit", "test"},
}

func TestTransitions(t *testing.T) {
	got := Transitions(corpus)

	// Adjacencies: read->edit x2 (once per session), edit->read, read->bash,
	// edit->test each x1. Outgoing totals: read=3, edit=2.
	want := []Edge{
		{From: "read", To: "edit", Count: 2, Prob: 2.0 / 3.0},
		{From: "edit", To: "read", Count: 1, Prob: 1.0 / 2.0},
		{From: "edit", To: "test", Count: 1, Prob: 1.0 / 2.0},
		{From: "read", To: "bash", Count: 1, Prob: 1.0 / 3.0},
	}
	if len(got) != len(want) {
		t.Fatalf("edge count = %d, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		if got[i].From != want[i].From || got[i].To != want[i].To || got[i].Count != want[i].Count {
			t.Errorf("edge[%d] = %+v, want %+v", i, got[i], want[i])
		}
		if math.Abs(got[i].Prob-want[i].Prob) > 1e-9 {
			t.Errorf("edge[%d] Prob = %v, want %v", i, got[i].Prob, want[i].Prob)
		}
	}
}

// TestTransitionsProbSumsToOne pins the "proper per-source distribution"
// invariant: the outgoing probabilities of every From sum to 1.
func TestTransitionsProbSumsToOne(t *testing.T) {
	sums := map[string]float64{}
	for _, e := range Transitions(corpus) {
		sums[e.From] += e.Prob
	}
	for from, s := range sums {
		if math.Abs(s-1.0) > 1e-9 {
			t.Errorf("outgoing Prob for %q sums to %v, want 1.0", from, s)
		}
	}
}

func TestTransitionsBoundary(t *testing.T) {
	// A transition must not span a session boundary: bash (end of s1) is never
	// followed by read (start of s2).
	for _, e := range Transitions(corpus) {
		if e.From == "bash" && e.To == "read" {
			t.Fatalf("cross-session edge bash->read leaked: %+v", e)
		}
	}
	// Empty input yields an empty, non-nil slice.
	if got := Transitions(nil); got == nil || len(got) != 0 {
		t.Errorf("Transitions(nil) = %+v, want empty non-nil", got)
	}
}

func TestTopSequencesBigrams(t *testing.T) {
	got := TopSequences(corpus, 2, 3)
	want := []SeqCount{
		{Seq: []string{"read", "edit"}, Count: 2},
		{Seq: []string{"edit", "read"}, Count: 1}, // count-1 group broken lexically
		{Seq: []string{"edit", "test"}, Count: 1},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("top bigrams =\n  %+v\nwant\n  %+v", got, want)
	}
}

func TestTopSequencesTrigramsAll(t *testing.T) {
	// k<=0 => no limit; every distinct trigram, ranked. All occur once, so the
	// order is purely lexical.
	got := TopSequences(corpus, 3, 0)
	want := []SeqCount{
		{Seq: []string{"edit", "read", "bash"}, Count: 1},
		{Seq: []string{"read", "edit", "read"}, Count: 1},
		{Seq: []string{"read", "edit", "test"}, Count: 1},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("all trigrams =\n  %+v\nwant\n  %+v", got, want)
	}
}

func TestTopSequencesEdgeCases(t *testing.T) {
	if got := TopSequences(corpus, 0, 3); got != nil {
		t.Errorf("n=0 => want nil, got %+v", got)
	}
	// A window longer than every session contributes nothing.
	if got := TopSequences(corpus, 4, 0); len(got) != 1 {
		// s1 has length 4 -> exactly one length-4 window; s2 (len 3) none.
		t.Errorf("n=4 => want 1 window (from s1), got %+v", got)
	}
	if got := TopSequences(nil, 2, 3); got != nil && len(got) != 0 {
		t.Errorf("empty corpus => want no sequences, got %+v", got)
	}
}
