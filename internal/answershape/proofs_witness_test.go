package answershape

import (
	"reflect"
	"strings"
	"testing"
)

// Witness tests for internal/answershape. See docs/proofs/00-METHOD.md. These are
// deterministic, non-vacuous metamorphic assertions over the shape metric:
//
//   (1) [measure-deterministic] Measure is pure: a fixed (text, Limits) yields a
//       byte-identical Report across many calls, with no map-iteration-order leak
//       in the n-gram / line-block tallies.
//   (2) [threshold-load-bearing] The MaxRepeat knob is real: a text with an
//       intermediate RepeatFraction is degenerate under a threshold below it and
//       NOT degenerate (by repeat) under a threshold above it.
//   (3) [floor-load-bearing] The repeatFloorChars floor is real: the SAME
//       repetition is clean below the floor and degenerate above it.
//   (4) [repetition-monotone] Repeating a clean block more times never lowers the
//       headline RepeatFraction, and enough repetition flips clean → degenerate.

// witnessFixtures spans a clean text and one of each degeneration mode, so the
// determinism witness is non-vacuous on both verdict branches.
var witnessFixtures = []string{
	coherent,
	strings.Repeat("A", 200),
	strings.Repeat("abc", 80),
	strings.Repeat("ALERT: disk full\n", 20),
	strings.Repeat("the kernel adjudicates every tool call before it dispatches anything. ", 8),
	"yes, yes, yes, yes, yes, yes, yes, yes, yes, yes!",
	strings.Repeat("https://example.com/api/v1/resource/12345", 25), // flate/compression path
	strings.Repeat("=", 64), // structural fill — clean
}

// TestMeasureIsDeterministic witnesses (1). Map iteration order in ngramRepeat /
// lineBlockRepeat must not leak into the Report (the TopNGram tie-break and the
// fractions must be stable), so 64 repeated calls produce an EQUAL Report.
func TestMeasureIsDeterministic(t *testing.T) {
	sawDegen, sawClean := false, false
	for _, fx := range witnessFixtures {
		first := Measure([]byte(fx), def())
		if first.Degenerate {
			sawDegen = true
		} else {
			sawClean = true
		}
		for i := 0; i < 64; i++ {
			got := Measure([]byte(fx), def())
			if !reflect.DeepEqual(got, first) {
				t.Fatalf("Measure not deterministic for %q on call %d:\n first=%+v\n got=%+v", short(fx), i, first, got)
			}
		}
	}
	// Non-vacuity: the determinism check exercised BOTH verdict branches.
	if !sawDegen || !sawClean {
		t.Fatalf("vacuous: fixtures must include both a clean and a degenerate text (clean=%v degen=%v)", sawClean, sawDegen)
	}
}

// TestThresholdIsLoadBearing witnesses (2): MaxRepeat is a real input, not a
// coincidental constant. A text whose RepeatFraction sits strictly inside (0,1)
// must be degenerate just below its fraction and clean just above it.
func TestThresholdIsLoadBearing(t *testing.T) {
	// A long repeated sentence has an intermediate n-gram-driven RepeatFraction.
	text := []byte(strings.Repeat("the kernel adjudicates every tool call before it dispatches anything. ", 8))
	f := Measure(text, Limits{MaxRepeat: 1.1, NGram: DefaultNGram}).RepeatFraction
	if f <= 0 || f >= 1 {
		t.Fatalf("need an intermediate RepeatFraction in (0,1) to witness the knob; got %.3f", f)
	}
	below := Measure(text, Limits{MaxRepeat: f - 0.05, NGram: DefaultNGram})
	above := Measure(text, Limits{MaxRepeat: f + 0.05, NGram: DefaultNGram})
	if !below.Degenerate {
		t.Fatalf("RepeatFraction=%.3f should be degenerate under max-repeat %.3f", f, f-0.05)
	}
	if above.Degenerate {
		t.Fatalf("RepeatFraction=%.3f should be clean under max-repeat %.3f, reasons=%v", f, f+0.05, above.Reasons)
	}
}

// TestFloorIsLoadBearing witnesses (3): the same repetition is clean below the
// rune floor and degenerate above it, so the floor is a real guard and not dead.
func TestFloorIsLoadBearing(t *testing.T) {
	belowFloor := "ab ab ab"                // 8 runes < repeatFloorChars
	aboveFloor := strings.Repeat("ab ", 12) // 36 runes >= floor, same kind of repetition
	if r := Measure([]byte(belowFloor), def()); r.Degenerate {
		t.Fatalf("repetition below the %d-rune floor must be clean, got %v (chars=%d)", repeatFloorChars, r.Reasons, r.Chars)
	}
	if r := Measure([]byte(aboveFloor), def()); !r.Degenerate {
		t.Fatalf("the same repetition above the floor must be degenerate (chars=%d frac=%.3f)", r.Chars, r.RepeatFraction)
	}
}

// TestRepetitionIsMonotone witnesses (4): repeating a clean block more times never
// lowers the headline RepeatFraction, and enough repetition flips a clean text to
// degenerate — the loop-amplification property the guard exists to catch.
func TestRepetitionIsMonotone(t *testing.T) {
	block := coherent + "\n"
	prev := -1.0
	flipped := false
	base := Measure([]byte(block), def())
	if base.Degenerate {
		t.Fatalf("a single clean block must not be degenerate to witness the flip: %v", base.Reasons)
	}
	for _, k := range []int{1, 2, 4, 8, 16} {
		r := Measure([]byte(strings.Repeat(block, k)), def())
		if r.RepeatFraction < prev-1e-9 {
			t.Fatalf("RepeatFraction fell with more repetition: k=%d frac=%.4f < prev %.4f", k, r.RepeatFraction, prev)
		}
		prev = r.RepeatFraction
		if r.Degenerate {
			flipped = true
		}
	}
	if !flipped {
		t.Fatalf("repeating a clean block up to 16× never tripped the guard (final frac=%.4f)", prev)
	}
}

// short renders a long fixture compactly in a failure message.
func short(s string) string {
	s = strings.ReplaceAll(s, "\n", "\\n")
	if len(s) > 48 {
		return s[:48] + "…"
	}
	return s
}
