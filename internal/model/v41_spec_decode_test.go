package model

import (
	"errors"
	"reflect"
	"testing"
)

// TestV41SpecDecodeParity is the single witness for the V4.1 Flash spec-decode
// adapter. It proves, in order: (a) the zero-value config does not engage, so
// the base path is unchanged; (b) the default layout is the anchored 1+N bonus
// layout; (c) an anchor-first layout is refused with the typed
// ErrV41SpecDecodeAnchorCorruption via errors.Is; (d) the accepted prefix
// under the default layout is bit-exact with the base greedy oracle; and (e)
// enabled-but-unqualified is refused with ErrV41SpecDecodeUnqualified.
func TestV41SpecDecodeParity(t *testing.T) {
	t.Run("a_zero_value_disabled_inert", func(t *testing.T) {
		var zero V41SpecDecodeConfig
		if err := zero.Validate(); err != nil {
			t.Fatalf("zero-value config refused: %v", err)
		}
		textOnly := Config{ModelType: "deepseek_v41_text"}
		if err := textOnly.RefuseDeepSeekV41SpecDecodeUnqualified(zero); err != nil {
			t.Fatalf("disabled spec-decode hook not inert: %v", err)
		}
		for _, mt := range []string{"", "llama", "deepseek_v4", "qwen3"} {
			c := Config{ModelType: mt}
			enabled := V41SpecDecodeConfig{Enabled: true, DraftLen: 4, Layout: V41SpecDecodeLayoutBonus1N}
			if err := c.RefuseDeepSeekV41SpecDecodeUnqualified(enabled); err != nil {
				t.Fatalf("non-V4.1 family %q refused: %v", mt, err)
			}
		}
	})

	t.Run("b_default_layout_is_bonus_1N", func(t *testing.T) {
		if !v41SpecDecodeLayoutIsBonus1N(V41SpecDecodeLayoutBonus1N) {
			t.Fatal("default bonus 1+N layout not reported as default")
		}
		if v41SpecDecodeLayoutIsBonus1N(V41SpecDecodeLayoutInvalid) {
			t.Fatal("zero-value layout must not be the default")
		}
		if V41SpecDecodeLayoutBonus1N == V41SpecDecodeLayoutAnchorFirst {
			t.Fatal("anchor-first layout must be distinct from the bonus 1+N layout")
		}
		if V41SpecDecodeMinDraftLen != 1 || V41SpecDecodeMaxDraftLen != 32 {
			t.Fatalf("draft bound = [%d,%d], want [1,32]", V41SpecDecodeMinDraftLen, V41SpecDecodeMaxDraftLen)
		}
	})

	t.Run("c_anchor_first_refused", func(t *testing.T) {
		anchorFirst := V41SpecDecodeConfig{Enabled: true, DraftLen: 4, Layout: V41SpecDecodeLayoutAnchorFirst}
		if err := anchorFirst.Validate(); !errors.Is(err, ErrV41SpecDecodeAnchorCorruption) {
			t.Fatalf("anchor-first layout error = %v, want ErrV41SpecDecodeAnchorCorruption", err)
		}
		unset := V41SpecDecodeConfig{Enabled: true, DraftLen: 4}
		if err := unset.Validate(); !errors.Is(err, ErrV41SpecDecodeAnchorCorruption) {
			t.Fatalf("unset layout error = %v, want ErrV41SpecDecodeAnchorCorruption", err)
		}
	})

	t.Run("d_default_layout_bit_exact_with_greedy_oracle", func(t *testing.T) {
		// Independent base greedy oracle: derive the base stream by argmax over
		// logits rows, NOT by copying the draft input. The rows below are shaped
		// so the base greedy stream is [1, 2, 3, 4, 5, 6, 7, 8]; base[0] is the
		// anchor and base[i+1] is the token emitted after accepting the i-th
		// draft.
		logits := [][]float64{
			rowMaxAt(1, 9),
			rowMaxAt(2, 9),
			rowMaxAt(3, 9),
			rowMaxAt(4, 9),
			rowMaxAt(5, 9),
			rowMaxAt(6, 9),
			rowMaxAt(7, 9),
			rowMaxAt(8, 9),
		}
		base := v41SpecDecodeOracleFromLogits(logits)
		if want := []int{1, 2, 3, 4, 5, 6, 7, 8}; !reflect.DeepEqual(base, want) {
			t.Fatalf("independent argmax oracle = %v, want %v", base, want)
		}

		// (i) Perfect/consistent draft: the draft mirrors the base stream, so
		// every draft is accepted and the committed prefix must equal the
		// independently-computed base greedy prefix.
		draft := base[1:5]
		accept := []bool{true, true, true, true}
		got := v41SpecDecodeAcceptedUnderBonus1N(base[0], draft, accept, base)
		want := v41SpecDecodeOracleFromLogits(logits)[:len(got)]
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("perfect-draft committed prefix = %v, independent greedy prefix = %v", got, want)
		}

		// (ii) Mismatching draft: the draft disagrees with the base stream at
		// position 0, so the target's own token wins and the rejected off-stream
		// drafts are dropped. The committed output must still be a prefix of the
		// independent base greedy stream, not the draft.
		badDraft := []int{99, 2, 3, 4}
		bad := v41SpecDecodeAcceptedUnderBonus1N(base[0], badDraft, []bool{false, true, true, true}, base)
		badWant := v41SpecDecodeOracleFromLogits(logits)[:len(bad)]
		if !reflect.DeepEqual(bad, badWant) {
			t.Fatalf("mismatching-draft committed output = %v, independent greedy prefix = %v", bad, badWant)
		}
		if got, want := bad, base[:len(bad)]; !reflect.DeepEqual(got, want) {
			t.Fatalf("mismatching-draft output = %v, want base greedy prefix %v", got, want)
		}
		for _, tok := range bad {
			if tok == 99 {
				t.Fatalf("rejected draft token 99 leaked into committed output %v", bad)
			}
		}
	})

	t.Run("e_enabled_unqualified_refused", func(t *testing.T) {
		bad := []V41SpecDecodeConfig{
			{Enabled: true, DraftLen: 0, Layout: V41SpecDecodeLayoutBonus1N},
			{Enabled: true, DraftLen: 33, Layout: V41SpecDecodeLayoutBonus1N},
		}
		for _, sc := range bad {
			if err := sc.Validate(); !errors.Is(err, ErrV41SpecDecodeUnqualified) {
				t.Fatalf("DraftLen=%d error = %v, want ErrV41SpecDecodeUnqualified", sc.DraftLen, err)
			}
		}
		c := Config{ModelType: "deepseek_v41_text"}
		enabledUnqualified := V41SpecDecodeConfig{Enabled: true, DraftLen: 33, Layout: V41SpecDecodeLayoutBonus1N}
		if err := c.RefuseDeepSeekV41SpecDecodeUnqualified(enabledUnqualified); !errors.Is(err, ErrV41SpecDecodeUnqualified) {
			t.Fatalf("hook error = %v, want ErrV41SpecDecodeUnqualified", err)
		}
		qualified := V41SpecDecodeConfig{Enabled: true, DraftLen: 8, Layout: V41SpecDecodeLayoutBonus1N}
		if err := c.RefuseDeepSeekV41SpecDecodeUnqualified(qualified); err != nil {
			t.Fatalf("qualified spec-decode admitted? got %v, want nil", err)
		}
	})
}

// rowMaxAt builds a logits row of width 9 whose sole maximum is index tok, so
// argmax over the row yields tok. Width 9 admits tokens 0..8 for the oracle
// fixture.
func rowMaxAt(tok int, peak float64) []float64 {
	row := make([]float64, 9)
	for i := range row {
		row[i] = 0.1
	}
	row[tok] = peak
	return row
}

// TestV41SpecDecodeComposesWithDSparkRefusal proves the new adapter composes
// with, and does not replace, the existing DSpark refusal semantics.
func TestV41SpecDecodeComposesWithDSparkRefusal(t *testing.T) {
	_, cfg := readDeepSeekV41Config(t)
	if err := cfg.RefuseDeepSeekV41DSpark(); !errors.Is(err, ErrV41DSparkUnsupported) {
		t.Fatalf("DSpark refusal regressed: %v", err)
	}
	qualified := V41SpecDecodeConfig{Enabled: true, DraftLen: 4, Layout: V41SpecDecodeLayoutBonus1N}
	if err := cfg.RefuseDeepSeekV41SpecDecodeUnqualified(qualified); err != nil {
		t.Fatalf("spec-decode adapter refused an admitted request: %v", err)
	}
}
