package tokenizer

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

// GGML token_type values, mirroring llama.cpp's LLAMA_TOKEN_TYPE_*. Only the two
// that mark a token as "special" (matched literally, never byte-level decoded)
// matter here: CONTROL (e.g. <|endoftext|>) and USER_DEFINED (e.g. the ChatML
// <|im_start|>/<|im_end|> markers on some conversions).
const (
	ggmlTypeControl     int32 = 3
	ggmlTypeUserDefined int32 = 4
)

// FromGGML builds a Tokenizer from the byte-level BPE arrays embedded directly in
// a GGUF checkpoint (tokenizer.ggml.{tokens,merges,token_type,pre}). It is the
// offline, always-matches-the-model path: the GGUF already carries its own vocab
// and merges, so no separate tokenizer.json download is needed.
//
//   - tokens[i] is the vocab string for id i, in byte-level-encoded form (e.g.
//     "Ġhello", "Ċ", "<|im_end|>").
//   - merges[r] is the rank-r BPE merge "left right" (byte-level encoded; the
//     0x20 space byte itself encodes to Ġ, so the single separating space is
//     unambiguous).
//   - tokenTypes[i], when present, marks CONTROL(3)/USER_DEFINED(4) ids special.
//     An empty/short tokenTypes falls back to recognizing "<|...|>" markers so a
//     conversion that omitted token_type still stops on ChatML control tokens.
//   - pre selects the pre-tokenizer: the Qwen explicit-Split family (any hint
//     containing "qwen") vs the GPT-2 ByteLevel default. The caller may pass the
//     GGUF general.architecture when tokenizer.ggml.pre is absent.
//
// The result is byte-exact with the tokenizer.json path for the same model (gated
// by oracle_ggml_test.go against the llama.cpp golden corpus).
func FromGGML(tokens, merges []string, tokenTypes []int32, pre string) (*Tokenizer, error) {
	if len(tokens) == 0 {
		return nil, errors.New("tokenizer: empty GGML vocab")
	}
	if len(tokenTypes) != 0 && len(tokenTypes) != len(tokens) {
		return nil, fmt.Errorf("tokenizer: %d GGML token_type entries for %d tokens", len(tokenTypes), len(tokens))
	}

	idToToken := make([]string, len(tokens))
	tokenToID := make(map[string]int, len(tokens))
	special := make(map[int]string)
	specialByContent := make([]specialToken, 0)
	for id, content := range tokens {
		if content == "" {
			// A GGUF should never store an empty vocab string, but tolerate it: a
			// placeholder keeps Decode from erroring if the id is ever emitted, and
			// leaving it out of tokenToID keeps Encode from ever selecting it.
			idToToken[id] = fmt.Sprintf("<unused_%d>", id)
			continue
		}
		idToToken[id] = content
		tokenToID[content] = id
		if isGGMLSpecial(content, tokenTypes, id) {
			special[id] = content
			specialByContent = append(specialByContent, specialToken{content: content, id: id})
		}
	}
	sort.Slice(specialByContent, func(i, j int) bool {
		return len(specialByContent[i].content) > len(specialByContent[j].content)
	})

	mergeRank := make(map[tokenPair]int, len(merges))
	for rank, merge := range merges {
		left, right, ok := strings.Cut(merge, " ")
		if !ok || left == "" || right == "" {
			return nil, fmt.Errorf("tokenizer: unsupported GGML merge %q", merge)
		}
		pair := tokenPair{left: left, right: right}
		if _, exists := mergeRank[pair]; exists {
			continue // first (lowest-rank) occurrence wins, as in BPE
		}
		mergeRank[pair] = rank
	}
	if len(mergeRank) == 0 {
		return nil, errors.New("tokenizer: GGML checkpoint has no BPE merges (non-BPE tokenizer?)")
	}

	split := preTokenizeByteLevel
	if isQwenPreTokenizer(pre) {
		split = preTokenizeQwen
	}

	return &Tokenizer{
		idToToken:        idToToken,
		tokenToID:        tokenToID,
		special:          special,
		specialByContent: specialByContent,
		mergeRank:        mergeRank,
		split:            split,
		metaspace:        isMetaspaceTokenizer(pre),
	}, nil
}

// isMetaspaceTokenizer reports whether the GGUF tokenizer hint is a SentencePiece-style
// metaspace BPE (Gemma 4: tokenizer.ggml.model "gemma4"), whose vocab/merges use the ▁
// marker for spaces and store literal UTF-8 rather than the GPT-2 byte-level alphabet.
func isMetaspaceTokenizer(pre string) bool {
	return strings.Contains(strings.ToLower(pre), "gemma")
}

// isGGMLSpecial reports whether token id should be treated as a literal special
// token. It trusts tokenTypes when present; otherwise it falls back to the
// "<|...|>" added-token shape that Qwen/SmolLM control tokens use.
func isGGMLSpecial(content string, tokenTypes []int32, id int) bool {
	if id < len(tokenTypes) {
		t := tokenTypes[id]
		return t == ggmlTypeControl || t == ggmlTypeUserDefined
	}
	return isWrappedSpecial(content)
}

// isWrappedSpecial matches the "<|...|>" control-token shape (e.g. <|im_end|>,
// <|endoftext|>) used as the fallback when a GGUF omits token_type.
func isWrappedSpecial(s string) bool {
	return len(s) >= 4 && strings.HasPrefix(s, "<|") && strings.HasSuffix(s, "|>")
}

// isQwenPreTokenizer reports whether the GGUF pre-tokenizer/arch hint belongs to
// the Qwen family, which uses the explicit Split regex (preTokenizeQwen) rather
// than the GPT-2 ByteLevel default. Qwen2.5 reports "qwen2", Qwen3.6 "qwen35",
// and the MoE variants "qwen2moe"/"qwen3moe" — all contain "qwen". The same hint
// drives llama.cpp's own dispatch, so trusting it keeps us byte-exact with it.
func isQwenPreTokenizer(pre string) bool {
	return strings.Contains(strings.ToLower(pre), "qwen")
}
