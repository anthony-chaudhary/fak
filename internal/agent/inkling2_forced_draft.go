package agent

import (
	"encoding/json"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/toolgrammar"
)

// Inkling2LoopDrafter supplies deterministic forced-token spans derived from a
// compiled grammar (toolgrammar.ForcedDraft) during an Inkling-2 decode turn.
//
// NextForced observes the most recently emitted token when hasToken is true (or
// queries initial forced tokens before prediction when hasToken is false) and
// returns deterministic continuation token IDs to emit and commit in one batched
// Prefill pass instead of serial Step forward passes. Returned tokens must be
// valid non-stop token IDs.
type Inkling2LoopDrafter interface {
	NextForced(token int, hasToken bool) []int
}

// forcedDraftVocab builds and caches the immutable tokenizer prefix trie once
// per Inkling2LoopPlanner so repeated turns pay zero trie construction cost.
func (p *Inkling2LoopPlanner) forcedDraftVocab() *toolgrammar.PrefixVocab {
	if p == nil || p.tok == nil {
		return nil
	}
	p.vocabOnce.Do(func() {
		specials := p.tok.SpecialTokens()
		vocabSize := p.tok.Vocab()
		tokens := make([]toolgrammar.VocabToken, 0, vocabSize)
		for id := 0; id < vocabSize; id++ {
			if p.stops[id] {
				continue
			}
			if _, isSpecial := specials[id]; isSpecial {
				continue
			}
			b := p.tok.TokenBytes(id)
			if len(b) == 0 || !utf8.Valid(b) {
				continue
			}
			piece, err := p.tok.Decode([]int{id})
			if err != nil || piece != string(b) {
				continue
			}
			tokens = append(tokens, toolgrammar.VocabToken{ID: int32(id), Text: piece})
		}
		if len(tokens) > 0 {
			p.prefixVocab = toolgrammar.NewPrefixVocab(tokens)
		}
	})
	return p.prefixVocab
}

// forcedDraftToolSpec returns the parameter schema and tool name when the
// request forces a concrete non-Write JSON tool call, or (nil, "") otherwise.
// toolgrammar.Compile wraps {"name":"<tool>","arguments":<parameters>}, matching
// inKernelForcedToolInstruction.
func forcedDraftToolSpec(sp SampleParams, tools []ToolDef) (params json.RawMessage, toolName string) {
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

// hasSamplingSideEffects reports whether token-level sampling overrides that do
// not compose with bypassed deterministic spans (penalties / logit_bias) are set.
func hasSamplingSideEffects(sp SampleParams) bool {
	if len(sp.LogitBias) > 0 {
		return true
	}
	if sp.FrequencyPenalty != nil && *sp.FrequencyPenalty != 0 {
		return true
	}
	return sp.PresencePenalty != nil && *sp.PresencePenalty != 0
}

type inkling2ForcedDrafter struct {
	vocab    *toolgrammar.PrefixVocab
	state    *toolgrammar.JumpForwardState
	decode   func([]int) (string, error)
	stops    map[int]bool
	dst      []int32
	disabled bool
}

// newForcedDrafter builds a per-turn deterministic jump-forward drafter for
// forced-tool requests when reasoning effort is 0 and token penalties are unset.
func (p *Inkling2LoopPlanner) newForcedDrafter(sp SampleParams, tools []ToolDef) *inkling2ForcedDrafter {
	if p == nil || p.tok == nil || !inkling2PrefillsAnswerHeader(nil, sp) || hasSamplingSideEffects(sp) {
		return nil
	}
	params, toolName := forcedDraftToolSpec(sp, tools)
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
	vocab := p.forcedDraftVocab()
	if vocab == nil {
		return nil
	}
	return &inkling2ForcedDrafter{
		vocab:  vocab,
		state:  state,
		decode: p.tok.Decode,
		stops:  p.stops,
	}
}

func (d *inkling2ForcedDrafter) NextForced(token int, hasToken bool) []int {
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
