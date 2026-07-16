package tokenizer

import "strings"

// preTokKind is the CLOSED set of pre-tokenizer/metaspace backends this leaf
// dispatches on. Both loaders — ParseJSON (tokenizer.json) and FromGGML (a GGUF
// checkpoint) — map their own input to one of these kinds, then resolvePreTokenizer
// turns the kind into the concrete split function and metaspace flag. Keeping the
// dispatch in one exhaustive switch is the point: before this, the two loaders each
// carried their own inline dispatch and silently diverged — a GLM-4 tokenizer loaded
// through the JSON path missed the GLM-4 branch the GGML path had and got the GPT-2
// ByteLevel splitter instead, a wrong-tokens bug (#4265). Adding a family now means
// one new kind + one case here, and both loaders pick it up.
type preTokKind int

const (
	// preTokByteLevel is the GPT-2 ByteLevel/Digits default (GPT-2, SmolLM2, bare-BPE).
	preTokByteLevel preTokKind = iota
	// preTokQwen is the explicit Split-regex family (Qwen2.5/Qwen3.6): one digit at a time.
	preTokQwen
	// preTokGLM4 is GLM-4/ChatGLM4: Qwen's split with digits grouped in runs of 1-3.
	preTokGLM4
	// preTokMetaspace is the SentencePiece metaspace BPE (Gemma 4): ▁-marked literal text.
	preTokMetaspace
)

// resolvePreTokenizer maps a closed pre-tokenizer kind to the split function and the
// metaspace flag both loaders install on the Tokenizer. This is the SINGLE dispatch
// seam: it is exhaustive over preTokKind, so a call site can never silently miss a
// family. The default arm is preTokByteLevel, the GPT-2 fallback both loaders already used.
func resolvePreTokenizer(kind preTokKind) (split func(string) []string, metaspace bool) {
	switch kind {
	case preTokQwen:
		return preTokenizeQwen, false
	case preTokGLM4:
		return preTokenizeGLM4, false
	case preTokMetaspace:
		// Gemma BPEs the ▁-marked literal chunk directly; Encode keys off metaspace and
		// does not regex-split, so the ByteLevel splitter is the inert placeholder here
		// (matching the pre-unification FromGGML behavior exactly).
		return preTokenizeByteLevel, true
	default: // preTokByteLevel
		return preTokenizeByteLevel, false
	}
}

// jsonPreTokKind maps a tokenizer.json pre_tokenizer config to the closed kind. A Split
// stage marks the explicit-regex family (Qwen/GLM-4); within it, GLM-4 is the variant
// whose regex groups digits in runs of 1-3. Anything without a Split is the GPT-2
// ByteLevel default. (Gemma's Metaspace never reaches here — ParseJSON requires a
// ByteLevel decoder and rejects the Gemma tokenizer.json upstream.)
func jsonPreTokKind(c preTokConfig) preTokKind {
	if !c.hasSplit() {
		return preTokByteLevel
	}
	if c.hasGLM4DigitGrouping() {
		return preTokGLM4
	}
	return preTokQwen
}

// ggmlPreTokKind maps a GGUF pre-tokenizer/arch hint (tokenizer.ggml.pre, or the
// general.architecture fallback) to the closed kind, keeping FromGGML byte-exact with
// llama.cpp's own hint-driven dispatch. Order mirrors the pre-unification FromGGML:
// Qwen and GLM-4 select their explicit splitters; the Gemma hint selects metaspace;
// everything else is the GPT-2 ByteLevel default.
func ggmlPreTokKind(pre string) preTokKind {
	switch {
	case isQwenPreTokenizer(pre):
		return preTokQwen
	case isGLM4PreTokenizer(pre):
		return preTokGLM4
	case isMetaspaceTokenizer(pre):
		return preTokMetaspace
	default:
		return preTokByteLevel
	}
}

// hasGLM4DigitGrouping reports whether any Split stage's regex carries GLM-4's
// digit-triplet grouping marker (\p{N}{1,3}) — the ONE feature that distinguishes a
// GLM-4/ChatGLM4 tokenizer.json from a Qwen one. Both families use an explicit Split,
// so hasSplit alone cannot tell them apart; before this the JSON path routed every
// Split tokenizer to the Qwen splitter and mis-grouped GLM-4 digits (#4265). It walks
// a Sequence recursively, matching the shape hasSplit already trusts.
func (c preTokConfig) hasGLM4DigitGrouping() bool {
	if c.Type == "Split" && strings.Contains(c.Pattern.Regex, `\p{N}{1,3}`) {
		return true
	}
	for _, p := range c.Pretokenizers {
		if p.hasGLM4DigitGrouping() {
			return true
		}
	}
	return false
}
