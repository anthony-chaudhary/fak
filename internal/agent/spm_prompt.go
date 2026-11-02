package agent

import (
	"context"
	"fmt"
)

// prepareSPMRawPrompt owns special-token insertion only for the explicitly
// selected SPM raw-completion route. It does not guess a chat template.
func (p *InKernelPlanner) prepareSPMRawPrompt(ctx context.Context, messages []Message, tools []ToolDef, sp SampleParams) (preparedPrompt, error) {
	if err := ctx.Err(); err != nil {
		return preparedPrompt{}, err
	}
	if !sp.RawPrompt || len(messages) != 1 || len(tools) != 0 || messages[0].Role != RoleUser ||
		len(messages[0].ToolCalls) != 0 || messages[0].FunctionCall != nil || messages[0].ToolCallID != "" || messages[0].Name != "" ||
		messages[0].ContentWasStructured || len(messages[0].VisionImages) != 0 ||
		messages[0].Thinking != "" || messages[0].ThinkingSignature != "" || len(messages[0].RedactedThinking) != 0 || messages[0].ReasoningContent != "" {
		return preparedPrompt{}, fmt.Errorf("native SPM: only one plain raw-completion user input is supported; chat/tools/vision require a separately admitted template")
	}
	policy, ok := p.tok.SPMPolicy()
	if !ok || p.m == nil || policy.BOS < 0 || policy.EOS < 0 || p.m.Cfg.BOSTokenID == nil || *p.m.Cfg.BOSTokenID != policy.BOS {
		return preparedPrompt{}, fmt.Errorf("native SPM: tokenizer/model BOS identity is not consistently declared")
	}
	eos := p.m.Cfg.EOSTokenID
	if len(p.m.Cfg.EOSTokenIDs) > 0 {
		if len(p.m.Cfg.EOSTokenIDs) != 1 {
			return preparedPrompt{}, fmt.Errorf("native SPM: multiple model stop identities require a separate contract")
		}
		eos = p.m.Cfg.EOSTokenIDs[0]
	}
	if eos != policy.EOS {
		return preparedPrompt{}, fmt.Errorf("native SPM: tokenizer/model EOS identity differs")
	}
	text := messages[0].Content
	ids, err := p.tok.EncodeSPM(text, true, false)
	if err != nil {
		return preparedPrompt{}, err
	}
	if err := ctx.Err(); err != nil {
		return preparedPrompt{}, err
	}
	maxNew := p.maxNew
	if sp.MaxTokens != nil && *sp.MaxTokens > 0 {
		maxNew = *sp.MaxTokens
	}
	// Raw input is neither prompt-shrunk nor silently wrapped in ChatML.
	// EncodePrompt/Complete keep their existing context/output admission.
	return preparedPrompt{messages: messages, rendered: text, ids: ids, maxNew: maxNew}, nil
}

// spmRawStreamPrompt is only the initial stream-projection seed. Complete still
// performs the single authoritative prepare/encode/admission before observers
// can run. Do not re-tokenize, shrink or render ChatML at this seam.
func (p *InKernelPlanner) spmRawStreamPrompt(messages []Message, sp SampleParams) (string, bool) {
	if p == nil || p.tok == nil || !p.tok.IsSPM() || !sp.RawPrompt || len(messages) != 1 || messages[0].Role != RoleUser {
		return "", false
	}
	return messages[0].Content, true
}
