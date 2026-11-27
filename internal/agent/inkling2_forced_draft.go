package agent

import (
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/toolgrammar"
)

// Inkling2LoopDrafter aliases the model-agnostic ForcedDrafter interface.
type Inkling2LoopDrafter = ForcedDrafter

type inkling2ForcedDrafter = forcedDrafter

// forcedDraftVocab builds and caches the immutable tokenizer prefix trie once
// per Inkling2LoopPlanner so repeated turns pay zero trie construction cost.
func (p *Inkling2LoopPlanner) forcedDraftVocab() *toolgrammar.PrefixVocab {
	if p == nil || p.tok == nil {
		return nil
	}
	p.vocabOnce.Do(func() {
		p.prefixVocab = BuildForcedDraftVocab(p.tok, p.stops)
	})
	return p.prefixVocab
}

func forcedDraftToolSpec(sp SampleParams, tools []ToolDef) (params json.RawMessage, toolName string) {
	return ForcedDraftToolSpec(sp, tools)
}

func hasSamplingSideEffects(sp SampleParams) bool {
	return HasSamplingSideEffects(sp)
}

// newForcedDrafter builds a per-turn deterministic jump-forward drafter for
// forced-tool requests when reasoning effort is 0 and token penalties are unset.
func (p *Inkling2LoopPlanner) newForcedDrafter(sp SampleParams, tools []ToolDef) *inkling2ForcedDrafter {
	if p == nil || p.tok == nil || !inkling2PrefillsAnswerHeader(nil, sp) {
		return nil
	}
	vocab := p.forcedDraftVocab()
	if vocab == nil {
		return nil
	}
	return NewForcedDrafter(p.tok.Decode, vocab, p.stops, sp, tools)
}
