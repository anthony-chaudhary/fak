package compute

import (
	"bytes"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/grammar"
)

// fak-test:runtime fast est=10ms lane=default

// gbByteVocab is the test TokenDecoder: ids 0..255 decode to their byte.
type gbByteVocab struct{}

func (gbByteVocab) TokenBytes(id int) []byte {
	if id >= 0 && id < 256 {
		return []byte{byte(id)}
	}
	return nil
}

func gbByteIDs(s string) []int {
	ids := make([]int, len(s))
	for i := 0; i < len(s); i++ {
		ids[i] = int(s[i])
	}
	return ids
}

// TestGrammarBitmaskMatchesHostMaskOracle pins the vLLM grammar-bitmask borrow
// from vllm/v1/worker/gpu/structured_outputs.py @ ae065918640a198a56f6f17eb45e251c3229fe41
// (Apache-2.0; lineage xgrammar apply_token_bitmask_inplace_triton):
//
//	(a) a dense allowed-set row produces byte-identical -inf placement vs
//	    CallMask.MaskLogits on the f32 oracle, with allowed values preserved
//	    exactly and untouched rows left alone under an explicit row-index
//	    vector;
//	(b) an empty allowed set leaves only the forced stop token finite
//	    (collapse-to-stop), and stays all-forbidden with no forced stop;
//	(c) a vocabulary not divisible by 32 masks the tail word without spilling
//	    past vocab_size.
//
// CPU/in-memory results only: [SW-VERIFIED], no GPU or throughput claim.
func TestGrammarBitmaskMatchesHostMaskOracle(t *testing.T) {
	t.Parallel()

	t.Run("dense_allowed_matches_oracle", func(t *testing.T) {
		t.Parallel()
		const (
			vocab   = 128
			eos     = 127 // DEL: in-vocabulary EOS contract
			rowsTot = 8
		)
		mask, err := grammar.Compile(
			[]grammar.ToolSpec{{Name: "lookup",
				Schema: []byte(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}`)}},
			gbByteVocab{}, grammar.CompileOptions{EOS: eos})
		if err != nil {
			t.Fatalf("grammar.Compile: %v", err)
		}
		// Emitted prefix ends mid free-string value ("ab", no closing quote):
		// the next byte may be any printable ASCII except backslash, plus the
		// closing quote — a dense allowed set spanning two packed words.
		prefix := `{"name":"lookup","arguments":{"q":"ab`
		history := gbByteIDs(prefix)

		oracle := make([]float32, vocab)
		for i := range oracle {
			oracle[i] = float32(i*13 - 7)
		}
		mask.MaskLogits(history, oracle)

		allowed := make([]bool, vocab)
		nAllowed := 0
		for i := range allowed {
			allowed[i] = !math.IsInf(float64(oracle[i]), -1)
			if allowed[i] {
				nAllowed++
			}
		}
		// 0x20..0x7E minus backslash: 95 - 1 = 94 tokens.
		if nAllowed != 94 {
			t.Fatalf("oracle admits %d tokens, want 94 (printable ASCII minus backslash)", nAllowed)
		}
		for _, c := range []int{0x20, 0x22, 0x7e} {
			if !allowed[c] {
				t.Errorf("oracle should admit token 0x%02X", c)
			}
		}
		for _, c := range []int{0x1f, 0x5c, 0x7f} {
			if allowed[c] {
				t.Errorf("oracle should forbid token 0x%02X (control / backslash / EOS)", c)
			}
		}

		logits := make([]float32, rowsTot*vocab)
		for r := 0; r < rowsTot; r++ {
			for c := 0; c < vocab; c++ {
				logits[r*vocab+c] = float32(r*1000 + c*7 + 1)
			}
		}
		orig := append([]float32(nil), logits...)

		words := make([]int32, GrammarBitmaskWordCount(vocab))
		if err := PackGrammarBitmask(words, allowed, vocab, -1); err != nil {
			t.Fatalf("PackGrammarBitmask returned %v, want nil", err)
		}
		// One packed row per entry of the explicit row-index vector (the
		// pinned-slab layout): the same per-step allowed set repeats for each
		// addressed row.
		rows := []int32{5, 1, 3} // out-of-order subset of the batch rows
		slab := make([]int32, len(rows)*GrammarBitmaskWordCount(vocab))
		for i := range rows {
			copy(slab[i*GrammarBitmaskWordCount(vocab):], words)
		}
		if err := ApplyGrammarBitmask(logits, rows, slab, vocab); err != nil {
			t.Fatalf("ApplyGrammarBitmask returned %v, want nil", err)
		}

		neg := float32(math.Inf(-1))
		for _, r := range rows {
			rr := int(r)
			row := logits[rr*vocab : (rr+1)*vocab]
			for c := 0; c < vocab; c++ {
				wantInf := math.IsInf(float64(oracle[c]), -1)
				gotInf := math.IsInf(float64(row[c]), -1)
				if wantInf != gotInf {
					t.Errorf("row %d col 0x%02X: -inf placement diverges from the host mask oracle (want inf=%v)", rr, c, wantInf)
					continue
				}
				if wantInf {
					if row[c] != neg {
						t.Errorf("row %d col 0x%02X: forbidden slot holds %v, want -inf", rr, c, row[c])
					}
				} else if row[c] != orig[rr*vocab+c] {
					t.Errorf("row %d col 0x%02X: allowed value changed to %v, want %v", rr, c, row[c], orig[rr*vocab+c])
				}
			}
		}
		for _, r := range []int{0, 2, 4, 6, 7} {
			if !bytes.Equal(
				floatBitsSlice(logits[r*vocab:(r+1)*vocab]),
				floatBitsSlice(orig[r*vocab:(r+1)*vocab])) {
				t.Errorf("row %d: unaddressed row was modified", r)
			}
		}
	})

	t.Run("empty_allowed_forced_stop", func(t *testing.T) {
		t.Parallel()
		const vocab = 128
		allowed := make([]bool, vocab) // all false: empty allowed set
		words := make([]int32, GrammarBitmaskWordCount(vocab))
		if err := PackGrammarBitmask(words, allowed, vocab, 42); err != nil {
			t.Fatalf("PackGrammarBitmask(forcedStop=42) returned %v, want nil", err)
		}
		// 42 = 32 + 10: word 1, bit 10, and nothing else.
		want := []int32{0, 1 << 10, 0, 0}
		if !equalInt32s(words, want) {
			t.Fatalf("packed words = %v, want %v", words, want)
		}

		row := make([]float32, vocab)
		for c := range row {
			row[c] = float32(c + 1)
		}
		if err := ApplyGrammarBitmask(row, []int32{0}, words, vocab); err != nil {
			t.Fatalf("ApplyGrammarBitmask returned %v, want nil", err)
		}
		for c := range row {
			switch {
			case c == 42:
				if row[c] != 43 {
					t.Errorf("col 42: forced stop value changed to %v, want 43", row[c])
				}
			case math.IsInf(float64(row[c]), -1):
				// forbidden: correct
			default:
				t.Errorf("col %d: finite value %v survived an empty allowed set", c, row[c])
			}
		}

		// No forced stop: the raw upstream semantics keep every token
		// forbidden.
		raw := make([]int32, GrammarBitmaskWordCount(vocab))
		if err := PackGrammarBitmask(raw, allowed, vocab, -1); err != nil {
			t.Fatalf("PackGrammarBitmask(forcedStop=-1) returned %v, want nil", err)
		}
		if !equalInt32s(raw, []int32{0, 0, 0, 0}) {
			t.Fatalf("raw-packed words = %v, want all zero", raw)
		}
		row2 := make([]float32, vocab)
		for c := range row2 {
			row2[c] = 1
		}
		if err := ApplyGrammarBitmask(row2, []int32{0}, raw, vocab); err != nil {
			t.Fatalf("ApplyGrammarBitmask(raw) returned %v, want nil", err)
		}
		for c := range row2 {
			if !math.IsInf(float64(row2[c]), -1) {
				t.Errorf("col %d: finite value survived an all-forbidden row", c)
			}
		}
	})

	t.Run("tail_word_no_spill", func(t *testing.T) {
		t.Parallel()
		const vocab = 97 // 3 full words + 1 tail token

		// Full allowed set: the tail word carries only its first bit.
		allowed := make([]bool, vocab)
		for i := range allowed {
			allowed[i] = true
		}
		words := make([]int32, GrammarBitmaskWordCount(vocab))
		if err := PackGrammarBitmask(words, allowed, vocab, -1); err != nil {
			t.Fatalf("PackGrammarBitmask returned %v, want nil", err)
		}
		want := []int32{^int32(0), ^int32(0), ^int32(0), 1}
		if !equalInt32s(words, want) {
			t.Fatalf("packed words = %v, want %v (tail bits must clear)", words, want)
		}

		// Two rows: the tail word's clear bits 1..31 sit over the NEXT row's
		// memory in the flat buffer; the column guard must keep them there
		// untouched.
		sentinel := float32(999)
		logits := make([]float32, 2*vocab)
		for c := 0; c < vocab; c++ {
			logits[c] = 1.5
			logits[vocab+c] = sentinel
		}
		if err := ApplyGrammarBitmask(logits, []int32{0}, words, vocab); err != nil {
			t.Fatalf("ApplyGrammarBitmask returned %v, want nil", err)
		}
		for c := 0; c < vocab; c++ {
			if logits[c] != 1.5 {
				t.Errorf("row 0 col %d: allowed value changed to %v", c, logits[c])
			}
			if logits[vocab+c] != sentinel {
				t.Errorf("row 1 col %d: tail-word spill wrote %v, want sentinel %v", c, logits[vocab+c], sentinel)
			}
		}

		// Last token forbidden: the tail word is entirely clear, so the whole
		// tail chunk is forbidden without touching row 1.
		allowed[96] = false
		words2 := make([]int32, GrammarBitmaskWordCount(vocab))
		if err := PackGrammarBitmask(words2, allowed, vocab, -1); err != nil {
			t.Fatalf("PackGrammarBitmask(last forbidden) returned %v, want nil", err)
		}
		if words2[3] != 0 {
			t.Fatalf("tail word = %d, want 0", words2[3])
		}
		logits2 := make([]float32, 2*vocab)
		for c := 0; c < vocab; c++ {
			logits2[c] = 2.5
			logits2[vocab+c] = sentinel
		}
		if err := ApplyGrammarBitmask(logits2, []int32{0}, words2, vocab); err != nil {
			t.Fatalf("ApplyGrammarBitmask(last forbidden) returned %v, want nil", err)
		}
		if !math.IsInf(float64(logits2[96]), -1) {
			t.Errorf("col 96: last token stayed finite %v, want -inf", logits2[96])
		}
		for c := 0; c < 96; c++ {
			if logits2[c] != 2.5 {
				t.Errorf("row 0 col %d: allowed value changed to %v", c, logits2[c])
			}
		}
		for c := 0; c < vocab; c++ {
			if logits2[vocab+c] != sentinel {
				t.Errorf("row 1 col %d: spill wrote %v, want sentinel %v", c, logits2[vocab+c], sentinel)
			}
		}
	})

	t.Run("sentinel_errors", func(t *testing.T) {
		t.Parallel()
		const vocab = 128
		words := make([]int32, GrammarBitmaskWordCount(vocab))
		allowed := make([]bool, vocab)
		row := make([]float32, 4*vocab)
		checks := []struct {
			name     string
			err      error
			sentinel error
		}{
			{"pack_short_words", packErr(make([]int32, 3), allowed, vocab, -1), ErrGrammarBitmaskShape},
			{"pack_long_words", packErr(make([]int32, 5), allowed, vocab, -1), ErrGrammarBitmaskShape},
			{"pack_short_allowed", packErr(words, allowed[:127], vocab, -1), ErrGrammarBitmaskShape},
			{"pack_zero_vocab", packErr(make([]int32, 0), nil, 0, -1), ErrGrammarBitmaskShape},
			{"pack_stop_high", packErr(words, allowed, vocab, vocab), ErrGrammarBitmaskStop},
			{"pack_stop_negative", packErr(words, allowed, vocab, -5), ErrGrammarBitmaskStop},
			{"apply_negative_row", applyErr(row, []int32{-1}, words, vocab), ErrGrammarBitmaskRow},
			{"apply_row_out_of_range", applyErr(row, []int32{4}, words, vocab), ErrGrammarBitmaskRow},
			{"apply_bitmask_short", applyErr(row, []int32{0, 1}, words[:3], vocab), ErrGrammarBitmaskShape},
			{"apply_bitmask_long", applyErr(row, []int32{0}, append(append([]int32(nil), words...), 0), vocab), ErrGrammarBitmaskShape},
			{"apply_logits_not_multiple", applyErr(row[:vocab-1], []int32{0}, words, vocab), ErrGrammarBitmaskShape},
			{"apply_empty_rows_nonempty_mask", applyErr(row, nil, words, vocab), ErrGrammarBitmaskShape},
			{"apply_zero_vocab", applyErr(row, nil, nil, 0), ErrGrammarBitmaskShape},
		}
		for _, c := range checks {
			if !errors.Is(c.err, c.sentinel) {
				t.Errorf("%s: errors.Is(%v, %v) = false, want true", c.name, c.err, c.sentinel)
			}
		}
	})

	t.Run("source_contract", func(t *testing.T) {
		t.Parallel()
		cu, err := os.ReadFile(filepath.FromSlash("grammar_bitmask.cu"))
		if err != nil {
			t.Fatal(err)
		}
		cuText := string(bytes.ReplaceAll(cu, []byte("\r\n"), []byte("\n")))
		cuCode := stripCComments(cuText)
		for token, want := range map[string]int{
			"__global__ void k_grammar_bitmask_":    1,
			`extern "C" int fcuda_grammar_bitmask_`: 1,
			"#include <cuda_runtime.h>":             1,
		} {
			if got := strings.Count(cuCode, token); got != want {
				t.Errorf("grammar_bitmask.cu has %d occurrences of %q, want %d", got, token, want)
			}
		}
		if !strings.Contains(cuText, "ae065918640a198a56f6f17eb45e251c3229fe41") {
			t.Error("grammar_bitmask.cu lacks the upstream vLLM pin citation")
		}
	})
}

func packErr(words []int32, allowed []bool, vocabSize, forcedStop int) error {
	return PackGrammarBitmask(words, allowed, vocabSize, forcedStop)
}

func applyErr(logits []float32, rows []int32, bitmask []int32, vocabSize int) error {
	return ApplyGrammarBitmask(logits, rows, bitmask, vocabSize)
}

func equalInt32s(a, b []int32) bool {
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

// floatBitsSlice renders a float32 slice as its exact IEEE-754 bytes for a
// byte-identity comparison that treats -0.0 and +0.0 distinctly.
func floatBitsSlice(f []float32) []byte {
	out := make([]byte, len(f)*4)
	for i, v := range f {
		bits := uint32(math.Float32bits(v))
		out[i*4] = byte(bits)
		out[i*4+1] = byte(bits >> 8)
		out[i*4+2] = byte(bits >> 16)
		out[i*4+3] = byte(bits >> 24)
	}
	return out
}
