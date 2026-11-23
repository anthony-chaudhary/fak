// Adapted from open-telemetry/semantic-conventions-genai, docs/gen-ai/gen-ai-agent-spans.md,
// docs/gen-ai/execute-tool.md and docs/gen-ai/gen-ai-spans.md at commit
// e6658f14c9d9ea0760825f029cb82e7673f48fab (Apache-2.0). This is an adaptation of the
// span-name and attribute conventions, not a copy of upstream code.

package stepspan

// GenAISemconvVersion identifies the pinned semantic-conventions-genai revision; every
// gen_ai string below has Development stability there and may change on a re-pin.
const GenAISemconvVersion = "semantic-conventions-genai@e6658f14c9d9ea0760825f029cb82e7673f48fab"

const (
	GenAIOperationName  = "gen_ai.operation.name"
	GenAIProviderName   = "gen_ai.provider.name"
	GenAIAgentName      = "gen_ai.agent.name"
	GenAIConversationID = "gen_ai.conversation.id"
	GenAIRequestModel   = "gen_ai.request.model"
	GenAIToolName       = "gen_ai.tool.name"
	GenAIToolCallID     = "gen_ai.tool.call.id"
	GenAIErrorType      = "error.type"
)

const (
	GenAIOpInvokeAgent     = "invoke_agent"
	GenAIOpExecuteTool     = "execute_tool"
	GenAIOpGenerateContent = "generate_content"
	GenAIOpPlan            = "plan"
)

// genAIContentKeys are the opt-in attributes that carry prompt, completion, tool-argument
// or tool-result content; the mapping never produces them and exporters drop them.
var genAIContentKeys = map[string]struct{}{
	"gen_ai.input.messages":          {},
	"gen_ai.output.messages":         {},
	"gen_ai.system_instructions":     {},
	"gen_ai.tool.definitions":        {},
	"gen_ai.tool.call.arguments":     {},
	"gen_ai.tool.call.result":        {},
	"gen_ai.input.messages_ref":      {},
	"gen_ai.output.messages_ref":     {},
	"gen_ai.system_instructions_ref": {},
	"gen_ai.tool.definitions_ref":    {},
}

// IsGenAIContentKey reports whether key names a content-bearing GenAI attribute.
func IsGenAIContentKey(key string) bool {
	_, ok := genAIContentKeys[key]
	return ok
}

// GenAIAttr is one string-valued span attribute.
type GenAIAttr struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

// GenAISpan is the GenAI view of a step span: a span name plus ordered attributes.
type GenAISpan struct {
	Name  string      `json:"name"`
	Attrs []GenAIAttr `json:"attrs"`
}

// GenAIOptions carries caller-known context that a step span does not record.
// AgentName must be low cardinality; leave it empty rather than use a framework default.
// ProviderName is required for generate_content; without it LLM calls do not map.
type GenAIOptions struct {
	AgentName    string
	ProviderName string
}

// GenAI maps s onto the pinned GenAI conventions; ok is false when the span kind has no
// GenAI meaning or a required attribute is unavailable.
func GenAI(s Span, opt GenAIOptions) (g GenAISpan, ok bool) {
	var op, target string
	switch {
	case s.Plane == Agent && s.Kind == Turn:
		op, target = GenAIOpInvokeAgent, opt.AgentName
	case s.Plane == Agent && s.Kind == ToolCall:
		if s.Attrs.Tool == "" {
			return GenAISpan{}, false
		}
		op, target = GenAIOpExecuteTool, s.Attrs.Tool
	case s.Plane == Agent && s.Kind == LLMCall:
		if opt.ProviderName == "" {
			return GenAISpan{}, false
		}
		op, target = GenAIOpGenerateContent, s.Attrs.Model
	case s.Plane == Planner && s.Kind == Plan:
		op, target = GenAIOpPlan, opt.AgentName
	default:
		return GenAISpan{}, false
	}
	g.Name = op
	if target != "" {
		g.Name = op + " " + target
	}
	str := func(k, v string) {
		if v != "" {
			g.Attrs = append(g.Attrs, GenAIAttr{Key: k, Value: v})
		}
	}
	str(GenAIOperationName, op)
	switch op {
	case GenAIOpExecuteTool:
		str(GenAIToolName, s.Attrs.Tool)
		str(GenAIToolCallID, s.IDs.CallID)
	case GenAIOpGenerateContent:
		str(GenAIProviderName, opt.ProviderName)
		str(GenAIRequestModel, s.Attrs.Model)
	}
	str(GenAIAgentName, opt.AgentName)
	if op != GenAIOpPlan {
		str(GenAIConversationID, s.IDs.SessionID)
	}
	if s.Status != OK && s.Status != "" {
		str(GenAIErrorType, genAIErrorType(s.Status))
	}
	return g, true
}

// genAIErrorType keeps the specific non-OK statuses as custom low-cardinality classes;
// the generic Error status carries no class, so it takes the upstream fallback.
func genAIErrorType(st Status) string {
	if st == Error {
		return "_OTHER"
	}
	return string(st)
}
