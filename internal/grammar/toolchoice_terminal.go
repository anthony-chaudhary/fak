package grammar

import "errors"

// ToolchoiceMode selects the closed structural grammar contract. Named-tool
// selection uses Required after a separate tool-name filter. Auto mode is
// deliberately outside this bounded contract.
type ToolchoiceMode uint8

const (
	ToolchoiceRequired ToolchoiceMode = iota
	ToolchoiceNone
)

// ToolchoiceEvent represents a validated structural event, not raw model text.
// The caller validates arguments, tool names and response-channel syntax.
// ResponseComplete means a complete, closed response channel, not incremental
// text. In None mode the caller must enforce the upstream nonempty response
// contract (including its non-whitespace prefix).
type ToolchoiceEvent uint8

const (
	ToolchoiceResponseComplete ToolchoiceEvent = iota
	ToolchoiceToolsOpen
	ToolchoiceCallComplete
	ToolchoiceToolsClose
	ToolchoiceMessageClose
)

// ErrToolchoiceTerminal marks a refused event or invalid configuration.
var ErrToolchoiceTerminal = errors.New("grammar: tool-choice structural refusal")

// ToolchoiceTerminal is a host contract adapted from SGLang Kimi-K3
// structural tags at ebeed893722150a29303d9eb9bd975abec4072a5 (Apache-2.0).
// Required mode permits EOS only after a complete nonempty tools section.
// Parallel mode does not terminate after the first individual call. None
// forbids tools and requires response content. This is not sampler wiring.
// Construct with NewToolchoiceTerminal; the zero value refuses all events.
type ToolchoiceTerminal struct {
	mode          ToolchoiceMode
	parallel      bool
	valid         bool
	response      bool
	inTools       bool
	calls         int
	toolsClosed   bool
	messageClosed bool
}

func NewToolchoiceTerminal(mode ToolchoiceMode, parallel bool) (*ToolchoiceTerminal, error) {
	if mode > ToolchoiceNone {
		return nil, ErrToolchoiceTerminal
	}
	return &ToolchoiceTerminal{mode: mode, parallel: parallel, valid: true}, nil
}

// Accept consumes one valid structural event. Refusal leaves state unchanged.
func (s *ToolchoiceTerminal) Accept(event ToolchoiceEvent) error {
	if s == nil || !s.valid || s.messageClosed {
		return ErrToolchoiceTerminal
	}
	switch event {
	case ToolchoiceResponseComplete:
		if s.response || s.inTools || s.toolsClosed {
			return ErrToolchoiceTerminal
		}
		s.response = true
	case ToolchoiceToolsOpen:
		if s.mode == ToolchoiceNone || s.inTools || s.toolsClosed {
			return ErrToolchoiceTerminal
		}
		s.inTools = true
	case ToolchoiceCallComplete:
		if !s.inTools || (!s.parallel && s.calls != 0) {
			return ErrToolchoiceTerminal
		}
		// Only presence matters; saturate to avoid an unbounded counter.
		s.calls = 1
	case ToolchoiceToolsClose:
		if !s.inTools || s.calls == 0 {
			return ErrToolchoiceTerminal
		}
		s.inTools = false
		s.toolsClosed = true
	case ToolchoiceMessageClose:
		if !s.Terminal() {
			return ErrToolchoiceTerminal
		}
		s.messageClosed = true
	default:
		return ErrToolchoiceTerminal
	}
	return nil
}

// Terminal reports whether the structural suffix can end now. A terminal
// tools section may still consume its optional message close, but never
// another call or response. EOS availability is distinct from hard closure.
func (s *ToolchoiceTerminal) Terminal() bool {
	if s == nil || !s.valid || s.inTools {
		return false
	}
	if s.toolsClosed {
		return true
	}
	return s.mode != ToolchoiceRequired && s.response
}
