package model

import (
	"errors"
	"reflect"
	"testing"
)

// TestV41SpecDecodeVerifyWidthIsPureBatchFunction witnesses the per-step verify
// width contract for issue #13148: the width is a PURE FUNCTION OF THE BATCH, so
// the execute path and the async FSM callback agree without communicating. The
// key properties are (a) a fixed draft depth with a strict prefix verified keeps
// the full depth, (b) a multi-sequence shape narrows to the per-sequence depth
// rather than the whole batch's decode total, (c) MaxWidth and DraftLen floor
// bound the result, and (d) repeated calls with identical input return identical
// widths (no hidden clock/counter/acceptance history).
func TestV41SpecDecodeVerifyWidthIsPureBatchFunction(t *testing.T) {
	const fullDepth = 4
	fullShape := SpeculativeShape{NumSequences: 1, TokensPerSeq: []int{4}, MaxTokensPerSeq: 4}

	t.Run("fixed_depth_strict_prefix_keeps_full_depth", func(t *testing.T) {
		got := v41SpecDecodeVerifyWidth(V41SpecDecodeStepInput{DraftLen: fullDepth, Shape: fullShape})
		if got != fullDepth {
			t.Fatalf("verify width = %d, want %d (full draft depth)", got, fullDepth)
		}
	})

	t.Run("multi_sequence_narrows_to_per_sequence_not_batch_total", func(t *testing.T) {
		shape := SpeculativeShape{
			NumSequences:    4,
			TokensPerSeq:    []int{4, 4, 4, 4},
			MaxTokensPerSeq: 4,
		}
		got := v41SpecDecodeVerifyWidth(V41SpecDecodeStepInput{DraftLen: 8, Shape: shape})
		if got != 4 {
			t.Fatalf("verify width = %d, want 4 (per-sequence depth)", got)
		}
		if got == 16 {
			t.Fatalf("verify width grew with the batch total (16): %d", got)
		}
	})

	t.Run("max_width_caps_further", func(t *testing.T) {
		got := v41SpecDecodeVerifyWidth(V41SpecDecodeStepInput{
			DraftLen: 8,
			Shape:    SpeculativeShape{NumSequences: 1, TokensPerSeq: []int{8}, MaxTokensPerSeq: 8},
			MaxWidth: 3,
		})
		if got != 3 {
			t.Fatalf("verify width = %d, want 3 (MaxWidth cap)", got)
		}
	})

	t.Run("zero_and_negative_draft_len_are_zero", func(t *testing.T) {
		for _, draft := range []int{0, -1, -32} {
			got := v41SpecDecodeVerifyWidth(V41SpecDecodeStepInput{DraftLen: draft, Shape: fullShape})
			if got != 0 {
				t.Fatalf("DraftLen=%d verify width = %d, want 0", draft, got)
			}
		}
	})

	t.Run("purity_repeated_calls_and_paths_agree", func(t *testing.T) {
		in := V41SpecDecodeStepInput{DraftLen: 8, Shape: fullShape}
		first := v41SpecDecodeVerifyWidth(in)
		for i := 0; i < 5; i++ {
			if got := v41SpecDecodeVerifyWidth(in); got != first {
				t.Fatalf("call %d width = %d, want %d (impure: hidden state/clock?)", i, got, first)
			}
		}
		// The execute path and the async FSM callback both call the same pure
		// function; model that agreement directly and assert it cannot diverge.
		if got, want := verifyWidthFromBothPaths(in), verifyWidthFromBothPaths(in); got != want || got != first {
			t.Fatalf("execute/async widths = %d and %d, want both %d", got, want, first)
		}
	})

	t.Run("aggregate_only_shape_returns_draft_len_unchanged", func(t *testing.T) {
		// No TokensPerSeq and no MaxTokensPerSeq: the shape carries only
		// aggregate counts, so there is no per-sequence view to narrow with.
		shape := SpeculativeShape{NumSequences: 3, TotalTokens: 24}
		got := v41SpecDecodeVerifyWidth(V41SpecDecodeStepInput{DraftLen: 8, Shape: shape})
		if got != 8 {
			t.Fatalf("aggregate-only shape width = %d, want 8 (DraftLen unchanged, deterministic)", got)
		}
	})

	t.Run("tokens_per_seq_absent_but_max_tokens_narrows", func(t *testing.T) {
		shape := SpeculativeShape{NumSequences: 2, MaxTokensPerSeq: 3}
		got := v41SpecDecodeVerifyWidth(V41SpecDecodeStepInput{DraftLen: 8, Shape: shape})
		if got != 3 {
			t.Fatalf("MaxTokensPerSeq-only width = %d, want 3", got)
		}
	})
}

// verifyWidthFromBothPaths models the two call sites that must agree: both read
// the SAME pure function of the SAME batch input, never a private width.
func verifyWidthFromBothPaths(in V41SpecDecodeStepInput) int {
	executePath := v41SpecDecodeVerifyWidth(in)
	asyncCallback := v41SpecDecodeVerifyWidth(in)
	if executePath != asyncCallback {
		panic("execute path and async callback derived different widths")
	}
	return executePath
}

// TestV41SpecDecodeVerifyWidthMixedPrefillNarrower witnesses that a mixed
// prefill+decode step verifies a NARROWER width than the same DraftLen does
// without prefill, and that the mixed narrowing never undershoots to zero while a
// per-sequence width exists. A step spending budget on prefill cannot verify a
// full-depth block, so the mixed override beats the batch-size schedule.
func TestV41SpecDecodeVerifyWidthMixedPrefillNarrower(t *testing.T) {
	t.Run("mixed_narrows_and_overrides_batch_schedule", func(t *testing.T) {
		// Two sequences whose per-sequence depth is 4 (the shape's
		// MaxTokensPerSeq), so the NON-mixed width is 4. The step's stated decode
		// side (TotalTokens - PrefillTokens) is only 2 across both sequences, so
		// the mixed width is the per-sequence share 2/2 == 1, strictly narrower
		// for the same DraftLen=4. The 512-token prefill remainder is the budget
		// the mixed step must also spend.
		shape := SpeculativeShape{
			NumSequences:    2,
			MaxTokensPerSeq: 4,
			TotalTokens:     2,
			PrefillTokens:   0,
			ContextPosition: 0,
		}
		if got := v41SpecDecodePerSequenceWidth(shape); got != 4 {
			t.Fatalf("fixture per-sequence width = %d, want 4", got)
		}
		if got := shape.DecodeTokens(); got != 2 {
			t.Fatalf("fixture DecodeTokens() = %d, want 2", got)
		}
		nonMixed := v41SpecDecodeVerifyWidth(V41SpecDecodeStepInput{DraftLen: 4, Shape: shape})
		mixed := v41SpecDecodeVerifyWidth(V41SpecDecodeStepInput{DraftLen: 4, Shape: shape, MixedPrefill: true})
		if mixed != 1 {
			t.Fatalf("mixed width = %d, want 1 (decode share 2/2)", mixed)
		}
		if nonMixed != 4 {
			t.Fatalf("non-mixed width = %d, want 4 (per-sequence depth)", nonMixed)
		}
		if !(mixed < nonMixed) {
			t.Fatalf("mixed width %d must be strictly narrower than non-mixed %d", mixed, nonMixed)
		}
	})

	t.Run("mixed_never_exceeds_draft_len", func(t *testing.T) {
		shapes := []SpeculativeShape{
			{NumSequences: 1, TokensPerSeq: []int{4}, MaxTokensPerSeq: 4},
			{NumSequences: 4, TokensPerSeq: []int{4, 4, 4, 4}, MaxTokensPerSeq: 4},
			{NumSequences: 2, TotalTokens: 100, PrefillTokens: 10, MaxTokensPerSeq: 45},
		}
		for _, shape := range shapes {
			for _, draft := range []int{1, 2, 4, 8} {
				got := v41SpecDecodeVerifyWidth(V41SpecDecodeStepInput{DraftLen: draft, Shape: shape, MixedPrefill: true})
				if got < 0 || got > draft {
					t.Fatalf("mixed DraftLen=%d shape=%+v width = %d, want in [0,%d]", draft, shape, got, draft)
				}
			}
		}
	})

	t.Run("mixed_is_never_wider_than_non_mixed", func(t *testing.T) {
		// Regression for the inverted-narrowing bug: a shape may state an
		// aggregate decode total (TotalTokens) far above its own per-sequence
		// MaxTokensPerSeq. The mixed width must still be capped by the
		// per-sequence width, so it can never come out WIDER than the same step
		// without prefill.
		shapes := []SpeculativeShape{
			{NumSequences: 1, MaxTokensPerSeq: 2, TotalTokens: 100, PrefillTokens: 10},
			{NumSequences: 2, MaxTokensPerSeq: 3, TotalTokens: 200, PrefillTokens: 16},
			{NumSequences: 4, TokensPerSeq: []int{5, 5, 5, 5}, MaxTokensPerSeq: 5, TotalTokens: 4096, PrefillTokens: 512},
			{NumSequences: 1, MaxTokensPerSeq: 8, TotalTokens: 8, PrefillTokens: 64},
		}
		for _, shape := range shapes {
			for _, draft := range []int{1, 2, 4, 8, 16} {
				nonMixed := v41SpecDecodeVerifyWidth(V41SpecDecodeStepInput{DraftLen: draft, Shape: shape})
				mixed := v41SpecDecodeVerifyWidth(V41SpecDecodeStepInput{DraftLen: draft, Shape: shape, MixedPrefill: true})
				if mixed > nonMixed {
					t.Fatalf("mixed width %d exceeds non-mixed %d for DraftLen=%d shape=%+v",
						mixed, nonMixed, draft, shape)
				}
			}
		}
	})

	t.Run("mixed_unspecified_decode_does_not_collapse_to_zero", func(t *testing.T) {
		// The shape states no decode side (DecodeTokens()==0) but DOES carry a
		// per-sequence width (MaxTokensPerSeq). The mixed width must fall back to
		// that width rather than commit zero drafts on a step that has them.
		noDecode := SpeculativeShape{
			NumSequences:    0,
			MaxTokensPerSeq: 3,
			PrefillTokens:   8,
			TotalTokens:     8,
			ContextPosition: 0,
		}
		if got := noDecode.DecodeTokens(); got != 0 {
			t.Fatalf("fixture DecodeTokens() = %d, want 0 (decode side unstated)", got)
		}
		if got := v41SpecDecodePerSequenceWidth(noDecode); got != 3 {
			t.Fatalf("fixture per-sequence width = %d, want 3", got)
		}
		mixed := v41SpecDecodeVerifyWidth(V41SpecDecodeStepInput{DraftLen: 8, Shape: noDecode, MixedPrefill: true})
		if mixed == 0 {
			t.Fatalf("mixed width collapsed to 0 on a shape with per-sequence width 3, want per-sequence fallback")
		}
		if mixed != 3 {
			t.Fatalf("mixed width = %d, want 3 (per-sequence fallback)", mixed)
		}
	})

	t.Run("mixed_decode_width_is_per_sequence_share", func(t *testing.T) {
		shape := SpeculativeShape{NumSequences: 2, TokensPerSeq: []int{2, 2}, MaxTokensPerSeq: 2}
		if got := shape.DecodeTokens(); got != 4 {
			t.Fatalf("fixture DecodeTokens() = %d, want 4", got)
		}
		got := v41SpecDecodeMixedDecodeWidth(shape)
		if want := 2; got != want {
			t.Fatalf("mixed decode width = %d, want %d (decode/NumSequences)", got, want)
		}
	})
}

// TestV41SpecDecodeHostMirrorTrimMatchesDeviceScatter witnesses the host-mirror
// contract for issue #13148: the host mirror trims to EXACTLY the width the
// device scatter committed, so the two never disagree on the next step's shape.
// A trim the step's own width/acceptance cannot explain is refused with the typed
// ErrV41SpecDecodeHostMirrorMismatch rather than silently clamped.
func TestV41SpecDecodeHostMirrorTrimMatchesDeviceScatter(t *testing.T) {
	assertMismatch := func(t *testing.T, err error) {
		t.Helper()
		if !errors.Is(err, ErrV41SpecDecodeHostMirrorMismatch) {
			t.Fatalf("error = %v, want ErrV41SpecDecodeHostMirrorMismatch", err)
		}
	}

	t.Run("strict_prefix_trim_matches_device_scatter", func(t *testing.T) {
		proposals := []int{10, 11, 12, 13}
		mirror, dropped, err := V41SpecDecodeMirrorTrimForStep(proposals, 4, 2)
		if err != nil {
			t.Fatalf("trim refused: %v", err)
		}
		if mirror.Committed() != 2 {
			t.Fatalf("Committed() = %d, want 2 (device-agreed width)", mirror.Committed())
		}
		// The device scatter dropped the verified but unaccepted tail: the two
		// draft tokens at positions [accepted, verifyWidth) = [2, 4).
		if want := []int{12, 13}; !reflect.DeepEqual(dropped, want) {
			t.Fatalf("dropped = %v, want %v", dropped, want)
		}
		if want := []int{10, 11, 12, 13}; !reflect.DeepEqual(mirror.Proposals(), want) {
			t.Fatalf("Proposals() = %v, want full host proposals %v", mirror.Proposals(), want)
		}
		// No broadcast shape error: the committed width equals accepted.
		if mirror.Committed() != 2 {
			t.Fatalf("committed width %d != accepted 2", mirror.Committed())
		}
	})

	t.Run("agreement_over_a_range_of_widths", func(t *testing.T) {
		proposals := []int{1, 2, 3, 4, 5, 6}
		for w := 0; w <= len(proposals); w++ {
			for a := 0; a <= w; a++ {
				mirror, _, err := V41SpecDecodeMirrorTrimForStep(proposals, w, a)
				if err != nil {
					t.Fatalf("(w=%d,a=%d) refused: %v", w, a, err)
				}
				if mirror.Committed() != a {
					t.Fatalf("(w=%d,a=%d) Committed() = %d, want %d", w, a, mirror.Committed(), a)
				}
			}
		}
	})

	t.Run("verify_width_beyond_proposals_refused", func(t *testing.T) {
		proposals := []int{1, 2, 3}
		_, _, err := V41SpecDecodeMirrorTrimForStep(proposals, len(proposals)+1, 0)
		assertMismatch(t, err)
	})

	t.Run("accepted_beyond_verify_width_refused", func(t *testing.T) {
		proposals := []int{1, 2, 3}
		_, _, err := V41SpecDecodeMirrorTrimForStep(proposals, 2, 3)
		assertMismatch(t, err)
	})

	t.Run("negative_width_or_accepted_refused", func(t *testing.T) {
		proposals := []int{1, 2, 3}
		if _, _, err := V41SpecDecodeMirrorTrimForStep(proposals, -1, 0); !errors.Is(err, ErrV41SpecDecodeHostMirrorMismatch) {
			t.Fatalf("negative width error = %v, want ErrV41SpecDecodeHostMirrorMismatch", err)
		}
		if _, _, err := V41SpecDecodeMirrorTrimForStep(proposals, 1, -1); !errors.Is(err, ErrV41SpecDecodeHostMirrorMismatch) {
			t.Fatalf("negative accepted error = %v, want ErrV41SpecDecodeHostMirrorMismatch", err)
		}
	})

	t.Run("trim_to_refuses_out_of_range_widths", func(t *testing.T) {
		mirror := NewV41SpecDecodeHostMirror([]int{1, 2, 3})
		if _, err := mirror.TrimTo(2); err != nil {
			t.Fatalf("TrimTo(2) refused: %v", err)
		}
		// Below the already-committed width.
		if _, err := mirror.TrimTo(1); !errors.Is(err, ErrV41SpecDecodeHostMirrorMismatch) {
			t.Fatalf("TrimTo below committed error = %v, want ErrV41SpecDecodeHostMirrorMismatch", err)
		}
		// Above len(proposals).
		if _, err := mirror.TrimTo(4); !errors.Is(err, ErrV41SpecDecodeHostMirrorMismatch) {
			t.Fatalf("TrimTo above proposals error = %v, want ErrV41SpecDecodeHostMirrorMismatch", err)
		}
	})

	t.Run("trim_to_committed_is_a_noop", func(t *testing.T) {
		mirror := NewV41SpecDecodeHostMirror([]int{1, 2, 3})
		if _, err := mirror.TrimTo(2); err != nil {
			t.Fatalf("TrimTo(2) refused: %v", err)
		}
		dropped, err := mirror.TrimTo(2)
		if err != nil {
			t.Fatalf("TrimTo(committed) refused: %v", err)
		}
		if dropped != nil {
			t.Fatalf("TrimTo(committed) dropped = %v, want nil", dropped)
		}
		if mirror.Committed() != 2 {
			t.Fatalf("Committed() = %d, want 2", mirror.Committed())
		}
	})

	t.Run("nil_mirror_trim_refused", func(t *testing.T) {
		var mirror *V41SpecDecodeHostMirror
		if _, err := mirror.TrimTo(1); !errors.Is(err, ErrV41SpecDecodeHostMirrorMismatch) {
			t.Fatalf("nil mirror TrimTo error = %v, want ErrV41SpecDecodeHostMirrorMismatch", err)
		}
	})
}

// TestV41SpecDecodeDraftCarryForward witnesses the draft carry-forward contract
// for issue #13148: an unverified tail survives a live step but is cleared at a
// DONE boundary, EOS outranks the length cap, and an already-carried head is not
// re-carried. Step is a pure function of prior state plus input.
func TestV41SpecDecodeDraftCarryForward(t *testing.T) {
	t.Run("live_step_carries_unverified_tail", func(t *testing.T) {
		var carry V41SpecDecodeCarry
		next := carry.Step(V41SpecDecodeCarryInput{Drafted: []int{1, 2, 3, 4}, VerifyWidth: 2})
		if done, reason := next.Done(); done || reason != V41SpecDecodeNotDone {
			t.Fatalf("live step Done() = (%v,%s), want (false,not-done)", done, reason)
		}
		if want := []int{3, 4}; !reflect.DeepEqual(next.Pending(), want) {
			t.Fatalf("pending = %v, want unverified tail %v", next.Pending(), want)
		}
	})

	t.Run("length_cap_clears_carry_with_cap_reason", func(t *testing.T) {
		carry := V41SpecDecodeCarry{pending: []int{3, 4}}
		next := carry.Step(V41SpecDecodeCarryInput{
			Drafted:      []int{1, 2},
			VerifyWidth:  2,
			LengthCap:    8,
			EmittedTotal: 8,
		})
		if done, reason := next.Done(); !done || reason != V41SpecDecodeDoneCap {
			t.Fatalf("Done() = (%v,%s), want (true,%s)", done, reason, V41SpecDecodeDoneCap)
		}
		if next.Pending() != nil {
			t.Fatalf("Pending() = %v, want nil (done context clears carry)", next.Pending())
		}
	})

	t.Run("eos_outranks_length_cap", func(t *testing.T) {
		var carry V41SpecDecodeCarry
		next := carry.Step(V41SpecDecodeCarryInput{
			Drafted:       []int{1, 2, 3},
			VerifyWidth:   3,
			LengthCap:     8,
			EmittedTotal:  8,
			StopEnabled:   true,
			StopToken:     2,
			CommittedStop: true,
		})
		done, reason := next.Done()
		if !done || reason != V41SpecDecodeDoneEOS {
			t.Fatalf("Done() = (%v,%s), want (true,%s) when EOS and cap fire together", done, reason, V41SpecDecodeDoneEOS)
		}
		if next.Pending() != nil {
			t.Fatalf("Pending() = %v, want nil", next.Pending())
		}
	})

	t.Run("already_carried_tail_is_not_re_carried", func(t *testing.T) {
		// Step 1 drafted 4, verified 2, so [3,4] carry forward.
		var carry V41SpecDecodeCarry
		carry = carry.Step(V41SpecDecodeCarryInput{Drafted: []int{1, 2, 3, 4}, VerifyWidth: 2})
		if want := []int{3, 4}; !reflect.DeepEqual(carry.Pending(), want) {
			t.Fatalf("prior pending = %v, want %v", carry.Pending(), want)
		}
		// Step 2's draft re-drafts the carried head [3,4] at the head of a longer
		// block [3,4,5,6,7,8], verifying 4 of its 6 drafts. The carried tokens are
		// consumed at the head of this step's draft, so they must NOT be re-carried
		// into the next carry.
		next := carry.Step(V41SpecDecodeCarryInput{Drafted: []int{3, 4, 5, 6, 7, 8}, VerifyWidth: 4})
		got := next.Pending()
		for _, tok := range []int{3, 4} {
			count := 0
			for _, p := range got {
				if p == tok {
					count++
				}
			}
			if count > 1 {
				t.Fatalf("carried token %d appears %d times in %v (re-carried)", tok, count, got)
			}
		}
		// The verified prefix [3,4,5,6] absorbed the carried head, so exactly
		// the step's own unverified tail [7,8] carries — the prior carry is not
		// re-carried and nothing new is silently dropped.
		if want := []int{7, 8}; !reflect.DeepEqual(got, want) {
			t.Fatalf("pending = %v, want own unverified tail %v", got, want)
		}
	})

	t.Run("step_is_pure_and_does_not_mutate_prior_pending", func(t *testing.T) {
		prior := V41SpecDecodeCarry{pending: []int{3, 4}}
		snapshot := prior.Pending()
		in := V41SpecDecodeCarryInput{Drafted: []int{1, 2, 3, 4, 5}, VerifyWidth: 3}
		first := prior.Step(in)
		second := prior.Step(in)
		if !reflect.DeepEqual(first.Pending(), second.Pending()) {
			t.Fatalf("same state+input gave %v then %v", first.Pending(), second.Pending())
		}
		if !reflect.DeepEqual(prior.Pending(), snapshot) {
			t.Fatalf("Step mutated the receiver: Pending() = %v, want %v", prior.Pending(), snapshot)
		}
	})

	t.Run("live_step_with_nothing_unverified_yields_empty_carry", func(t *testing.T) {
		var carry V41SpecDecodeCarry
		next := carry.Step(V41SpecDecodeCarryInput{Drafted: []int{1, 2}, VerifyWidth: 2})
		if done, reason := next.Done(); done || reason != V41SpecDecodeNotDone {
			t.Fatalf("Done() = (%v,%s), want (false,not-done)", done, reason)
		}
		if len(next.Pending()) != 0 {
			t.Fatalf("Pending() = %v, want empty", next.Pending())
		}
	})
}
