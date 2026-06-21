package ggufload

// GGMLTokenizer is the byte-level BPE tokenizer that a GGUF checkpoint embeds in
// its tokenizer.ggml.* metadata. It is plain data so the loader stays decoupled
// from internal/tokenizer: a caller feeds Tokens/Merges/TokenTypes/Pre to
// tokenizer.FromGGML to get a working tokenizer with no separate tokenizer.json.
type GGMLTokenizer struct {
	Model      string // tokenizer.ggml.model, e.g. "gpt2" (byte-level BPE) or "llama" (SPM)
	Pre        string // tokenizer.ggml.pre, falling back to general.architecture when absent
	Tokens     []string
	Merges     []string
	TokenTypes []int32
	BOS        int // -1 when unset
	EOS        int // -1 when unset
}

// GGMLTokenizer extracts the embedded byte-level BPE tokenizer arrays. ok is true
// only when both a vocab and merges are present — i.e. the checkpoint carries a
// usable BPE tokenizer (Qwen/GPT-2/SmolLM family). SPM-only checkpoints (llama,
// no merges) report ok=false so the caller falls back to a tokenizer.json.
func (f *File) GGMLTokenizer() (*GGMLTokenizer, bool) {
	tokens, ok := f.StringArray("tokenizer.ggml.tokens")
	if !ok || len(tokens) == 0 {
		return nil, false
	}
	merges, ok := f.StringArray("tokenizer.ggml.merges")
	if !ok || len(merges) == 0 {
		return nil, false
	}
	gt := &GGMLTokenizer{
		Tokens: tokens,
		Merges: merges,
		BOS:    -1,
		EOS:    -1,
	}
	gt.Model, _ = f.String("tokenizer.ggml.model")
	if pre, ok := f.String("tokenizer.ggml.pre"); ok && pre != "" {
		gt.Pre = pre
	} else if arch, ok := f.String("general.architecture"); ok {
		// Older conversions omit tokenizer.ggml.pre; the architecture still tells
		// us the family well enough to pick the Qwen-vs-GPT2 pre-tokenizer.
		gt.Pre = arch
	}
	if tt, ok := f.Int32Array("tokenizer.ggml.token_type"); ok {
		gt.TokenTypes = tt
	}
	if v, ok := f.Uint64("tokenizer.ggml.bos_token_id"); ok && v <= uint64(int(^uint(0)>>1)) {
		gt.BOS = int(v)
	}
	if v, ok := f.Uint64("tokenizer.ggml.eos_token_id"); ok && v <= uint64(int(^uint(0)>>1)) {
		gt.EOS = int(v)
	}
	return gt, true
}

// Int32Array reads a GGUF array of integers as []int32. It accepts any signed or
// unsigned integer element type (token_type is typically stored as int32, but
// some writers use uint32); values outside int32 range fail the read.
func (f *File) Int32Array(key string) ([]int32, bool) {
	v, ok := f.Metadata[key]
	if !ok || v.Type != TypeArray {
		return nil, false
	}
	items, ok := v.Value.([]Value)
	if !ok {
		return nil, false
	}
	out := make([]int32, len(items))
	for i, item := range items {
		u, ok := valueUint64(item)
		if !ok || u > uint64(^uint32(0)>>1) {
			return nil, false
		}
		out[i] = int32(u)
	}
	return out, true
}
