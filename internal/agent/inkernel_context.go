package agent

import "fmt"

// InKernelContextLengthError reports that an exactly tokenized native request
// cannot fit inside the planner's effective context window. The request is
// refused before any prefill, KV growth, or device allocation begins.
type InKernelContextLengthError struct {
	PromptTokens int
	MaxNewTokens int
	MaxContext   int
}

func (e *InKernelContextLengthError) Error() string {
	if e == nil {
		return "in-kernel context length exceeded"
	}
	return fmt.Sprintf("in-kernel context length exceeded: prompt tokens %d plus max new tokens %d exceed context window %d",
		e.PromptTokens, e.MaxNewTokens, e.MaxContext)
}

// ContextWindow returns the exact total-token ceiling enforced by this planner.
// A positive configured cap narrows a known model window and supplies a bound
// when model metadata is absent. Zero means neither source declares a limit.
func (p *InKernelPlanner) ContextWindow() int {
	if p == nil {
		return 0
	}
	configured := p.contextTokens
	declared := 0
	if p.m != nil {
		declared = p.m.Cfg.MaxPositionEmbeddings
	}
	switch {
	case configured > 0 && declared > 0 && configured < declared:
		return configured
	case declared > 0:
		return declared
	case configured > 0:
		return configured
	default:
		return 0
	}
}

// refuseContextLength validates prompt plus planned decode without adding the
// two values, so an adversarial max-token value cannot overflow an int and wrap
// back inside the window.
func (p *InKernelPlanner) refuseContextLength(promptTokens, maxNewTokens int) error {
	limit := p.ContextWindow()
	if limit <= 0 {
		return nil
	}
	if promptTokens < 0 {
		promptTokens = 0
	}
	if maxNewTokens < 0 {
		maxNewTokens = 0
	}
	if promptTokens > limit || maxNewTokens > limit-promptTokens {
		return &InKernelContextLengthError{
			PromptTokens: promptTokens,
			MaxNewTokens: maxNewTokens,
			MaxContext:   limit,
		}
	}
	return nil
}
