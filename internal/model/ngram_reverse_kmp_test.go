package model

import (
	"math/rand"
	"reflect"
	"testing"
)

// Keep the prior reverse-naive selection as an independent behavioral oracle.
func reverseKMPDraftOracle(d NgramDrafter, history []int) []int {
	if !d.Enabled || len(history) < 2 {
		return nil
	}
	minMatch, maxMatch, maxDraft := d.limits(len(history))
	if maxDraft == 0 {
		return nil
	}
	for n := maxMatch; n >= minMatch; n-- {
		for start := len(history) - n - 1; start >= 0; start-- {
			match := true
			for j := 0; j < n; j++ {
				if history[start+j] != history[len(history)-n+j] {
					match = false
					break
				}
			}
			if match {
				end := start + n + min(maxDraft, len(history)-start-n)
				return append([]int(nil), history[start+n:end]...)
			}
		}
	}
	return nil
}

func TestNgramReverseKMPParity(t *testing.T) {
	// Three token values exercise both repeated-prefix fallback and overlap.
	for size, count := 0, 1; size <= 8; size, count = size+1, count*3 {
		history := make([]int, size)
		for code := 0; code < count; code++ {
			value := code
			for i := range history {
				history[i], value = value%3, value/3
			}
			for maxMatch := 1; maxMatch <= size+1; maxMatch++ {
				d := NgramDrafter{Enabled: true, MinMatch: 1, MaxMatch: maxMatch, MaxDraft: 4}
				if got, want := d.Draft(history), reverseKMPDraftOracle(d, history); !reflect.DeepEqual(got, want) {
					t.Fatalf("history=%v max=%d got=%v want=%v", history, maxMatch, got, want)
				}
			}
		}
	}

	rng := rand.New(rand.NewSource(5261))
	for trial := 0; trial < 2000; trial++ {
		history := make([]int, rng.Intn(160))
		for i := range history {
			history[i] = rng.Intn(7) - 3
		}
		d := NgramDrafter{
			Enabled:  trial%9 != 0,
			MinMatch: rng.Intn(12) - 2, MaxMatch: rng.Intn(65) - 2,
			MinN: rng.Intn(12) - 2, MaxN: rng.Intn(65) - 2,
			MaxDraft:          rng.Intn(10) - 2,
			AdaptiveExpansion: trial%2 == 0,
			MaxAdaptiveDraft:  rng.Intn(20), CurrentDraft: rng.Intn(20),
		}
		before := append([]int(nil), history...)
		got, want := d.Draft(history), reverseKMPDraftOracle(d, history)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("trial=%d config=%+v history=%v got=%v want=%v", trial, d, history, got, want)
		}
		if len(got) > 0 {
			got[0]++
		}
		if !ngramEqualInts(history, before) {
			t.Fatalf("trial=%d draft aliases or mutates history", trial)
		}
	}

	d := NgramDrafter{Enabled: true, MinMatch: 2, MaxMatch: 2, MaxDraft: 1}
	if got := d.Draft([]int{1, 2, 10, 1, 2, 20, 1, 2}); !ngramEqualInts(got, []int{20}) {
		t.Fatalf("most-recent continuation = %v, want [20]", got)
	}
	history := make([]int, 80)
	for i := range history {
		history[i] = 7
	}
	d.MaxMatch = 64 // exercises the non-default LPS storage and overlapping match
	if got := d.Draft(history); !ngramEqualInts(got, []int{7}) {
		t.Fatalf("overlapping continuation = %v, want [7]", got)
	}
}

func BenchmarkNgramReverseKMPNearMiss(b *testing.B) {
	history := make([]int, 16384)
	history[len(history)-1] = 1
	for _, maxMatch := range []int{8, 64} {
		d := NgramDrafter{Enabled: true, MinMatch: 2, MaxMatch: maxMatch, MaxDraft: 4}
		name := "default8"
		if maxMatch == 64 {
			name = "configured64"
		}
		b.Run(name+"/prior", func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if got := reverseKMPDraftOracle(d, history); got != nil {
					b.Fatal(got)
				}
			}
		})
		b.Run(name+"/candidate", func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if got := d.Draft(history); got != nil {
					b.Fatal(got)
				}
			}
		})
	}
}
