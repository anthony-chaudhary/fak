package model

import "testing"

func TestNgramDrafterDefaultOff(t *testing.T) {
	if got := (NgramDrafter{}).Draft([]int{1, 2, 3, 4, 1, 2, 3}); got != nil {
		t.Fatalf("default-off draft = %v, want nil", got)
	}
}

func TestNgramDrafterLongestSuffixAndBoundedCopy(t *testing.T) {
	history := []int{1, 2, 3, 4, 5, 1, 2, 3}
	d := NgramDrafter{Enabled: true, MinMatch: 2, MaxMatch: 4, MaxDraft: 2}
	got := d.Draft(history)
	want := []int{4, 5}
	if !ngramEqualInts(got, want) {
		t.Fatalf("draft = %v, want %v", got, want)
	}
	got[0] = 99
	if history[3] != 4 {
		t.Fatal("draft aliases committed history")
	}
}

func TestNgramDrafterNoRepeatedSuffixIsInert(t *testing.T) {
	d := NgramDrafter{Enabled: true, MinMatch: 3, MaxMatch: 8, MaxDraft: 4}
	if got := d.Draft([]int{1, 2, 3, 4, 5, 6}); got != nil {
		t.Fatalf("non-repeating history draft = %v, want nil", got)
	}
}

func TestFirstTokenSubsequenceHandlesOverlap(t *testing.T) {
	if got := firstTokenSubsequence([]int{1, 1, 1, 1, 2}, []int{1, 1, 2}); got != 2 {
		t.Fatalf("overlap match = %d, want 2", got)
	}
}

func TestSpecDecodeGreedyWithNgramDrafterMatchesPlainGreedy(t *testing.T) {
	m := NewSynthetic(cfgV(64, 4, 4, 2, 16, 128))
	prompt := []int{1, 2, 3, 4, 1, 2, 3}
	d := NgramDrafter{Enabled: true, MinMatch: 3, MaxMatch: 3, MaxDraft: 4}
	want := m.NewSession().Generate(prompt, 20)
	run, err := SpecDecodeGreedyWithDrafter(m.NewSession(), prompt, 20, d.MaxDraft, d.Draft)
	if err != nil {
		t.Fatalf("SpecDecodeGreedyWithDrafter: %v", err)
	}
	if !ngramEqualInts(run.Output, want) {
		t.Fatalf("prompt-lookup output = %v, plain greedy = %v", run.Output, want)
	}
	if run.DraftedTokens == 0 {
		t.Fatal("prompt-lookup proposed no tokens; verify path was not exercised")
	}
}

func ngramEqualInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
