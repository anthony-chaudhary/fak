// grammar_bitmask.go — grammar bitmask packing + CPU reference of the device
// apply kernel.
//
// ADAPTED from vllm-project/vllm,
// vllm/v1/worker/gpu/structured_outputs.py:133 (_apply_grammar_bitmask_kernel)
// at ae065918640a198a56f6f17eb45e251c3229fe41 (Apache-2.0; attribution in
// THIRD_PARTY_NOTICES.md), lineage xgrammar
// apply_token_bitmask_inplace_triton (mlc-ai/xgrammar
// python/xgrammar/kernels/apply_token_bitmask_inplace_triton.py): a packed
// int32 bitmask (32 tokens per word, bit=0 => forbidden) applied to
// device-resident f32 logits rows addressed by an explicit row-index vector.
// Upstream H2D-copies the bitmask and the mapping on a side stream, then
// launches one kernel that stores -inf at every forbidden token; this
// candidate re-expresses that contract as the Go oracle that pins it
// (GrammarBitmaskWordCount / PackGrammarBitmask / ApplyGrammarBitmask), with
// the CUDA kernel in grammar_bitmask.cu. No upstream code copied.
// Ref oss-port-vllmtop-grammar-bitmask-device-20261104.
//
// SEMANTICS (pinned by grammar_bitmask_test.go):
//   - One packed word covers 32 tokens: bit i of word w addresses token
//     w*32+i; bit=1 means allowed, bit=0 means forbidden.
//   - PackGrammarBitmask writes one row into a pre-sized slab: bits at or
//     above vocabSize are always cleared, so the tail word can never
//     authorize past the vocabulary. When the allowed set is empty and
//     forcedStop >= 0, the forced-stop bit alone is set instead
//     (collapse-to-stop: mirrors the host complete-envelope state where only
//     the EOS/stop token survives, internal/agent maskLogits). With
//     forcedStop == -1 an empty allowed set stays all-zero (every token
//     forbidden, the raw upstream semantics).
//   - ApplyGrammarBitmask is the CPU mirror of fcuda_grammar_bitmask_apply:
//     for each entry of the explicit row-index vector, every token whose bit
//     is 0 becomes -inf; allowed tokens keep their exact value; columns at or
//     above vocabSize are never written. Duplicate row entries are harmless
//     (the operation is idempotent).
//
// The host mask path (internal/agent nativeStructuredConstraint.maskLogits)
// remains the correctness oracle and fallback; no GPU or throughput claim
// attaches to this candidate.

package compute

import (
	"errors"
	"math"
)

const grammarBitmaskTokensPerWord = 32

var (
	// ErrGrammarBitmaskShape rejects geometry that does not match the
	// declared vocabulary: a non-positive vocabulary, a wrong word count, an
	// allowed-set length, or a logits extent.
	ErrGrammarBitmaskShape = errors.New("compute: grammar bitmask: shape mismatch")

	// ErrGrammarBitmaskRow rejects a row-index vector entry that addresses
	// outside the logits buffer.
	ErrGrammarBitmaskRow = errors.New("compute: grammar bitmask: invalid row index")

	// ErrGrammarBitmaskStop rejects a forced stop token that is neither
	// "none" (-1) nor inside the vocabulary.
	ErrGrammarBitmaskStop = errors.New("compute: grammar bitmask: forced stop out of range")
)

// GrammarBitmaskWordCount returns the number of packed int32 words covering
// vocabSize tokens: 32 tokens per word, last word partial.
func GrammarBitmaskWordCount(vocabSize int) int {
	return (vocabSize + grammarBitmaskTokensPerWord - 1) / grammarBitmaskTokensPerWord
}

// PackGrammarBitmask packs one per-step allowed set into words: bit i of word
// w is 1 iff token w*32+i is in allowed and below vocabSize. Bits at or above
// vocabSize are cleared so the tail word can never authorize past the
// vocabulary. When allowed is empty and forcedStop >= 0, the forced-stop bit
// alone is set instead (collapse-to-stop); with forcedStop == -1 an empty
// allowed set stays all-zero (every token forbidden, the raw upstream
// semantics).
func PackGrammarBitmask(words []int32, allowed []bool, vocabSize, forcedStop int) error {
	if vocabSize <= 0 || len(allowed) != vocabSize {
		return ErrGrammarBitmaskShape
	}
	if len(words) != GrammarBitmaskWordCount(vocabSize) {
		return ErrGrammarBitmaskShape
	}
	if forcedStop < -1 || forcedStop >= vocabSize {
		return ErrGrammarBitmaskStop
	}
	for i := range words {
		words[i] = 0
	}
	for tok, ok := range allowed {
		if ok {
			words[tok/grammarBitmaskTokensPerWord] |= 1 << uint(tok%grammarBitmaskTokensPerWord)
		}
	}
	if !anyAllowed(allowed) && forcedStop >= 0 {
		words[forcedStop/grammarBitmaskTokensPerWord] |= 1 << uint(forcedStop%grammarBitmaskTokensPerWord)
	}
	return nil
}

func anyAllowed(allowed []bool) bool {
	for _, ok := range allowed {
		if ok {
			return true
		}
	}
	return false
}

// ApplyGrammarBitmask applies packed bitmask rows to f32 logits rows
// addressed by an explicit row-index vector: the CPU reference of
// fcuda_grammar_bitmask_apply. For each entry r of rows, every token whose
// bit is 0 in its word becomes -inf; allowed tokens keep their exact value;
// columns at or above vocabSize are never written. Duplicate row entries are
// harmless (the operation is idempotent).
func ApplyGrammarBitmask(logits []float32, rows []int32, bitmask []int32, vocabSize int) error {
	if vocabSize <= 0 || len(logits)%vocabSize != 0 {
		return ErrGrammarBitmaskShape
	}
	numRows := len(logits) / vocabSize
	words := GrammarBitmaskWordCount(vocabSize)
	if len(bitmask) != len(rows)*words {
		return ErrGrammarBitmaskShape
	}
	neg := float32(math.Inf(-1))
	for i, r := range rows {
		ri := int(r)
		if ri < 0 || ri >= numRows {
			return ErrGrammarBitmaskRow
		}
		rowBase := ri * vocabSize
		base := i * words
		for w := 0; w < words; w++ {
			col0 := w * grammarBitmaskTokensPerWord
			if col0 >= vocabSize {
				break
			}
			word := bitmask[base+w]
			for b := 0; b < grammarBitmaskTokensPerWord; b++ {
				col := col0 + b
				if col >= vocabSize {
					break
				}
				if (word>>uint(b))&1 == 0 {
					logits[rowBase+col] = neg
				}
			}
		}
	}
	return nil
}
