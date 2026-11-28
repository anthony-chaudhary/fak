package agent

import (
	"encoding/json"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/tokenizer"
	"github.com/anthony-chaudhary/fak/internal/toolgrammar"
)

// ForcedDrafter supplies deterministic forced-token spans derived from a
// compiled grammar (toolgrammar.ForcedDraft) during a decode turn.
//
// NextForced observes the most recently emitted token when hasToken is true (or
// queries initial forced tokens before prediction when hasToken is false) and
// returns deterministic continuation token IDs to emit and commit in one batched
// Prefill pass instead of serial Step forward passes. Returned tokens must be
// valid non-stop token IDs.
type ForcedDrafter interface {
	NextForced(token int, hasToken bool) []int
}

func forcedDraftTokens(tok *tokenizer.Tokenizer, stops map[int]bool) []toolgrammar.VocabToken {
	if tok == nil {
		return nil
	}
	specials := tok.SpecialTokens()
	vocabSize := tok.Vocab()
	tokens := make([]toolgrammar.VocabToken, 0, vocabSize)
	for id := 0; id < vocabSize; id++ {
		if stops[id] {
			continue
		}
		if _, isSpecial := specials[id]; isSpecial {
			continue
		}
		b := tok.TokenBytes(id)
		if len(b) == 0 || !utf8.Valid(b) {
			continue
		}
		piece, err := tok.Decode([]int{id})
		if err != nil || piece != string(b) {
			continue
		}
		tokens = append(tokens, toolgrammar.VocabToken{ID: int32(id), Text: piece})
	}
	return tokens
}

// BuildForcedDraftVocab constructs an immutable tokenizer prefix trie excluding
// stop and special tokens for deterministic grammar-constrained jump-forward
// drafting.
func BuildForcedDraftVocab(tok *tokenizer.Tokenizer, stops map[int]bool) *toolgrammar.PrefixVocab {
	tokens := forcedDraftTokens(tok, stops)
	if len(tokens) == 0 {
		return nil
	}
	return toolgrammar.NewPrefixVocab(tokens)
}

// BuildForcedDraftVocabAndMaskCache constructs both the immutable prefix trie
// and matching MaskCache in a single vocabulary scan.
func BuildForcedDraftVocabAndMaskCache(tok *tokenizer.Tokenizer, stops map[int]bool) (*toolgrammar.PrefixVocab, *toolgrammar.MaskCache) {
	tokens := forcedDraftTokens(tok, stops)
	if len(tokens) == 0 {
		return nil, nil
	}
	return toolgrammar.NewPrefixVocab(tokens), toolgrammar.NewMaskCache(tokens)
}

// ForcedDraftToolSpec returns the parameter schema and tool name when the
// request forces a concrete non-Write JSON tool call, or (nil, "") otherwise.
// toolgrammar.Compile wraps {"name":"<tool>","arguments":<parameters>}, matching
// inKernelForcedToolInstruction.
func ForcedDraftToolSpec(sp SampleParams, tools []ToolDef) (params json.RawMessage, toolName string) {
	name := inKernelEffectiveToolName(sp.ToolChoice, tools)
	if name == "" || name == "Write" {
		return nil, ""
	}
	for _, tool := range tools {
		if tool.Function.Name != name {
			continue
		}
		if len(tool.Function.Parameters) == 0 {
			return nil, ""
		}
		return tool.Function.Parameters, name
	}
	return nil, ""
}

// HasSamplingSideEffects reports whether token-level sampling overrides that do
// not compose with bypassed deterministic spans (penalties / logit_bias) are set.
func HasSamplingSideEffects(sp SampleParams) bool {
	if len(sp.LogitBias) > 0 {
		return true
	}
	if sp.FrequencyPenalty != nil && *sp.FrequencyPenalty != 0 {
		return true
	}
	return sp.PresencePenalty != nil && *sp.PresencePenalty != 0
}

type forcedDrafter struct {
	vocab     *toolgrammar.PrefixVocab
	maskCache *toolgrammar.MaskCache
	state     *toolgrammar.JumpForwardState
	decode    func([]int) (string, error)
	stops     map[int]bool
	dst       []int32
	disabled  bool
}

// NewForcedDrafter builds a per-turn deterministic jump-forward drafter for
// forced-tool requests when token penalties are unset and a compiled grammar
// state can be initialized. Returns nil when forced drafting is ineligible.
func NewForcedDrafter(decode func([]int) (string, error), vocab *toolgrammar.PrefixVocab, stops map[int]bool, sp SampleParams, tools []ToolDef) *forcedDrafter {
	return NewForcedDrafterWithMask(decode, vocab, nil, stops, sp, tools)
}

// NewForcedDrafterWithMask builds a per-turn deterministic jump-forward drafter
// paired with a reusable MaskCache for grammar-constrained logit masking.
func NewForcedDrafterWithMask(decode func([]int) (string, error), vocab *toolgrammar.PrefixVocab, maskCache *toolgrammar.MaskCache, stops map[int]bool, sp SampleParams, tools []ToolDef) *forcedDrafter {
	if decode == nil || vocab == nil || HasSamplingSideEffects(sp) {
		return nil
	}
	params, toolName := ForcedDraftToolSpec(sp, tools)
	if len(params) == 0 || toolName == "" {
		return nil
	}
	cg, err := toolgrammar.Compile(params, toolgrammar.GrammarOptions{
		Dialect:        toolgrammar.DialectJSON,
		ToolName:       toolName,
		IncludeToolTag: true,
	})
	if err != nil {
		return nil
	}
	state, err := toolgrammar.NewJumpForwardState(cg)
	if err != nil {
		return nil
	}
	return &forcedDrafter{
		vocab:     vocab,
		maskCache: maskCache,
		state:     state,
		decode:    decode,
		stops:     stops,
	}
}

// MaskLogits masks invalid token logits to -Inf according to the current
// grammar state while preserving stop/EOS tokens once complete.
func (d *forcedDrafter) MaskLogits(logits []float32) {
	if d == nil || d.disabled || d.state == nil || d.maskCache == nil || len(logits) == 0 {
		return
	}
	if d.state.IsComplete() {
		type savedLogit struct {
			id  int
			val float32
		}
		var stackSaved [8]savedLogit
		var extra []savedLogit
		n := 0
		for id := range d.stops {
			if id < 0 || id >= len(logits) {
				continue
			}
			if n < len(stackSaved) {
				stackSaved[n] = savedLogit{id: id, val: logits[id]}
				n++
			} else {
				extra = append(extra, savedLogit{id: id, val: logits[id]})
			}
		}
		d.maskCache.MaskLogits(d.state, logits)
		for i := 0; i < n; i++ {
			logits[stackSaved[i].id] = stackSaved[i].val
		}
		for _, s := range extra {
			logits[s.id] = s.val
		}
		return
	}
	d.maskCache.MaskLogits(d.state, logits)
}

func (d *forcedDrafter) NextForced(token int, hasToken bool) []int {
	if d == nil || d.disabled || d.state.IsComplete() {
		return nil
	}
	if hasToken {
		piece, err := d.decode([]int{token})
		if err != nil || piece == "" {
			d.disabled = true
			return nil
		}
		if err := d.state.AdvanceText(piece); err != nil {
			d.disabled = true
			return nil
		}
		if d.state.IsComplete() {
			return nil
		}
	}
	toks, txt := toolgrammar.ForcedDraftInto(d.state, d.vocab, d.dst[:0])
	d.dst = toks
	if len(toks) == 0 || txt == "" {
		return nil
	}
	out := make([]int, 0, len(toks))
	for _, id32 := range toks {
		id := int(id32)
		if d.stops[id] {
			break
		}
		out = append(out, id)
	}
	if len(out) == 0 {
		return nil
	}
	if len(out) < len(toks) {
		piece, err := d.decode(out)
		if err != nil || d.state.AdvanceText(piece) != nil {
			d.disabled = true
			return nil
		}
		return out
	}
	if err := d.state.AdvanceText(txt); err != nil {
		d.disabled = true
		return nil
	}
	return out
}
