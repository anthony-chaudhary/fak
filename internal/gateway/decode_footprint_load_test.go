package gateway

import "testing"

// TestAnticipatedDecodeBlocksMonotone pins the core property: a larger expected
// output length yields a strictly larger anticipated decode-block term (a request
// expected to grow big raises its host worker's load).
func TestAnticipatedDecodeBlocksMonotone(t *testing.T) {
	base := DecodeFootprintInputs{BlockTokens: 16, Scale: 1}

	small := base
	small.ExpectedOutputTokens = 64 // 4 blocks

	big := base
	big.ExpectedOutputTokens = 20000 // 1250 blocks

	sTerm := AnticipatedDecodeBlocks(small)
	bTerm := AnticipatedDecodeBlocks(big)
	if !(bTerm > sTerm) {
		t.Fatalf("expected larger output to yield larger term: small=%d big=%d", sTerm, bTerm)
	}

	// Non-decreasing across a sweep of increasing expected output lengths.
	prev := -1
	for _, osl := range []int{0, 1, 16, 17, 32, 100, 1000, 5000, 20000} {
		in := base
		in.ExpectedOutputTokens = osl
		got := AnticipatedDecodeBlocks(in)
		if got < prev {
			t.Fatalf("term must be monotone non-decreasing: osl=%d got=%d prev=%d", osl, got, prev)
		}
		prev = got
	}
}

// TestAnticipatedDecodeBlocksZeroBase pins that a zero (or already fully emitted)
// expected output contributes nothing — a short request is costed exactly as the
// count-only body costs it.
func TestAnticipatedDecodeBlocksZeroBase(t *testing.T) {
	if got := AnticipatedDecodeBlocks(DecodeFootprintInputs{ExpectedOutputTokens: 0, BlockTokens: 16, Scale: 1}); got != 0 {
		t.Fatalf("zero expected output must contribute 0, got %d", got)
	}
	// Fully generated: remaining <= 0 → nothing left to anticipate.
	fully := DecodeFootprintInputs{ExpectedOutputTokens: 100, GeneratedTokens: 100, BlockTokens: 16, Scale: 1}
	if got := AnticipatedDecodeBlocks(fully); got != 0 {
		t.Fatalf("fully-generated turn must contribute 0, got %d", got)
	}
	// Decay: as generation reveals the true length, the anticipated remainder
	// shrinks toward zero.
	early := DecodeFootprintInputs{ExpectedOutputTokens: 320, GeneratedTokens: 0, BlockTokens: 16, Scale: 1}
	late := early
	late.GeneratedTokens = 256
	if !(AnticipatedDecodeBlocks(late) < AnticipatedDecodeBlocks(early)) {
		t.Fatalf("term must decay as output is generated: early=%d late=%d",
			AnticipatedDecodeBlocks(early), AnticipatedDecodeBlocks(late))
	}
}

// TestEffectiveLoadComposesAdditively pins that the term folds onto a base load
// additively without double-counting: with a zero expected output the composed
// load equals the base exactly, and otherwise it equals base + term.
func TestEffectiveLoadComposesAdditively(t *testing.T) {
	const base = 7

	// Zero expected output: no change to the base (no double-count of a short req).
	if got := EffectiveLoadWithDecodeFootprint(base, DecodeFootprintInputs{BlockTokens: 16, Scale: 1}); got != base {
		t.Fatalf("zero-output request must leave base load unchanged: got %d want %d", got, base)
	}

	in := DecodeFootprintInputs{ExpectedOutputTokens: 8000, BlockTokens: 16, Scale: 1}
	term := AnticipatedDecodeBlocks(in)
	if term <= 0 {
		t.Fatalf("expected a positive anticipatory term, got %d", term)
	}
	got := EffectiveLoadWithDecodeFootprint(base, in)
	if got != base+term {
		t.Fatalf("composition must be exactly additive: got %d want %d (base=%d term=%d)", got, base+term, base, term)
	}
	// Idempotent/pure: composing again with the same inputs yields the same total
	// (the term does not compound across evaluations).
	if again := EffectiveLoadWithDecodeFootprint(base, in); again != got {
		t.Fatalf("term must be pure: got %d then %d", got, again)
	}

	// A worker near capacity (high base) plus a large-output request outscores an
	// idle worker taking a short request — the steering the issue asks for.
	busyLong := EffectiveLoadWithDecodeFootprint(10, in)
	idleShort := EffectiveLoadWithDecodeFootprint(10, DecodeFootprintInputs{ExpectedOutputTokens: 32, BlockTokens: 16, Scale: 1})
	if !(busyLong > idleShort) {
		t.Fatalf("large-output request must look more expensive: busyLong=%d idleShort=%d", busyLong, idleShort)
	}
}

// TestDecodeFootprintFailsClosed pins that degenerate inputs contribute nothing
// rather than corrupting the load.
func TestDecodeFootprintFailsClosed(t *testing.T) {
	cases := []struct {
		name string
		in   DecodeFootprintInputs
	}{
		{"non-positive block size", DecodeFootprintInputs{ExpectedOutputTokens: 1000, BlockTokens: 0, Scale: 1}},
		{"negative block size", DecodeFootprintInputs{ExpectedOutputTokens: 1000, BlockTokens: -8, Scale: 1}},
		{"negative expected output", DecodeFootprintInputs{ExpectedOutputTokens: -1000, BlockTokens: 16, Scale: 1}},
		{"zero scale disables", DecodeFootprintInputs{ExpectedOutputTokens: 1000, BlockTokens: 16, Scale: 0}},
		{"negative scale disables", DecodeFootprintInputs{ExpectedOutputTokens: 1000, BlockTokens: 16, Scale: -0.5}},
	}
	for _, c := range cases {
		if got := AnticipatedDecodeBlocks(c.in); got != 0 {
			t.Fatalf("%s: expected 0 contribution, got %d", c.name, got)
		}
		if got := EffectiveLoadWithDecodeFootprint(5, c.in); got != 5 {
			t.Fatalf("%s: fail-closed term must leave base unchanged, got %d", c.name, got)
		}
	}

	// Scale above 1 is clamped to 1 (bounded, not amplified).
	unclamped := AnticipatedDecodeBlocks(DecodeFootprintInputs{ExpectedOutputTokens: 1600, BlockTokens: 16, Scale: 1})
	clampedHigh := AnticipatedDecodeBlocks(DecodeFootprintInputs{ExpectedOutputTokens: 1600, BlockTokens: 16, Scale: 5})
	if clampedHigh != unclamped {
		t.Fatalf("scale > 1 must clamp to 1: got %d want %d", clampedHigh, unclamped)
	}

	// Negative base composes closed to just the term.
	if got := EffectiveLoadWithDecodeFootprint(-3, DecodeFootprintInputs{ExpectedOutputTokens: 320, BlockTokens: 16, Scale: 1}); got != 20 {
		t.Fatalf("negative base must clamp to 0: got %d want %d", got, 20)
	}
}

// TestAnticipatedDecodeBytes pins the companion byte projection: bytes = blocks ×
// BlockBytes, monotone in expected output, zero when BlockBytes is non-positive.
func TestAnticipatedDecodeBytes(t *testing.T) {
	in := DecodeFootprintInputs{ExpectedOutputTokens: 320, BlockTokens: 16, BlockBytes: 4096, Scale: 1}
	blocks := AnticipatedDecodeBlocks(in) // 20
	wantBytes := blocks * 4096
	if got := AnticipatedDecodeBytes(in); got != wantBytes {
		t.Fatalf("bytes projection: got %d want %d", got, wantBytes)
	}
	noBytes := in
	noBytes.BlockBytes = 0
	if got := AnticipatedDecodeBytes(noBytes); got != 0 {
		t.Fatalf("non-positive BlockBytes must yield 0, got %d", got)
	}
}

func TestDecodeFootprintArithmeticSaturatesInsteadOfWrapping(t *testing.T) {
	in := DecodeFootprintInputs{ExpectedOutputTokens: maxIntValue(), BlockTokens: 16, BlockBytes: maxIntValue(), Scale: 1}
	if blocks := AnticipatedDecodeBlocks(in); blocks <= 0 {
		t.Fatalf("max-sized prediction wrapped to %d blocks", blocks)
	}
	if got := AnticipatedDecodeBytes(in); got != maxIntValue() {
		t.Fatalf("byte projection = %d, want saturated max %d", got, maxIntValue())
	}
	if got := EffectiveLoadWithDecodeFootprint(maxIntValue(), DecodeFootprintInputs{ExpectedOutputTokens: 16, BlockTokens: 16, Scale: 1}); got != maxIntValue() {
		t.Fatalf("effective load wrapped to %d, want saturated max %d", got, maxIntValue())
	}
}
