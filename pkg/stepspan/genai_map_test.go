package stepspan

import (
	"reflect"
	"strings"
	"testing"
)

func TestGenAIMapSpanNamesAndAttributes(t *testing.T) {
	opt := GenAIOptions{AgentName: "Math Tutor", ProviderName: "anthropic"}
	cases := []struct {
		name  string
		span  Span
		opt   GenAIOptions
		want  string
		attrs []GenAIAttr
	}{
		{"turn", Span{Plane: Agent, Kind: Turn, Status: OK, IDs: IDs{SessionID: "conv_1"}}, opt, "invoke_agent Math Tutor", []GenAIAttr{
			{"gen_ai.operation.name", "invoke_agent"}, {"gen_ai.agent.name", "Math Tutor"}, {"gen_ai.conversation.id", "conv_1"}}},
		{"turn without agent name", Span{Plane: Agent, Kind: Turn, Status: OK}, GenAIOptions{}, "invoke_agent", []GenAIAttr{
			{"gen_ai.operation.name", "invoke_agent"}}},
		{"tool", Span{Plane: Agent, Kind: ToolCall, Status: Error, Attrs: Attrs{Tool: "Flights"}, IDs: IDs{CallID: "call_9", SessionID: "conv_1"}}, GenAIOptions{}, "execute_tool Flights", []GenAIAttr{
			{"gen_ai.operation.name", "execute_tool"}, {"gen_ai.tool.name", "Flights"}, {"gen_ai.tool.call.id", "call_9"}, {"gen_ai.conversation.id", "conv_1"}, {"error.type", "_OTHER"}}},
		{"llm", Span{Plane: Agent, Kind: LLMCall, Status: OK, Attrs: Attrs{Model: "gpt-4"}}, opt, "generate_content gpt-4", []GenAIAttr{
			{"gen_ai.operation.name", "generate_content"}, {"gen_ai.provider.name", "anthropic"}, {"gen_ai.request.model", "gpt-4"}, {"gen_ai.agent.name", "Math Tutor"}}},
		{"plan", Span{Plane: Planner, Kind: Plan, Status: Cancelled, IDs: IDs{SessionID: "conv_1"}}, opt, "plan Math Tutor", []GenAIAttr{
			{"gen_ai.operation.name", "plan"}, {"gen_ai.agent.name", "Math Tutor"}, {"error.type", "cancelled"}}},
	}
	for _, c := range cases {
		g, ok := GenAI(c.span, c.opt)
		if !ok || g.Name != c.want || !reflect.DeepEqual(g.Attrs, c.attrs) {
			t.Fatalf("%s: ok=%v got=%+v want name=%q attrs=%+v", c.name, ok, g, c.want, c.attrs)
		}
	}
}

func TestGenAIMapNotApplicable(t *testing.T) {
	for _, s := range []Span{
		{Plane: Engine, Kind: DecodeStep},
		{Plane: Gateway, Kind: Request},
		{Plane: Router, Kind: Request},
		{Plane: Agent, Kind: Adjudicate},
		{Plane: Agent, Kind: ToolCall},
		{Plane: Agent, Kind: LLMCall, Attrs: Attrs{Model: "gpt-4"}},
	} {
		if g, ok := GenAI(s, GenAIOptions{AgentName: "a"}); ok {
			t.Fatalf("%s/%s mapped to %+v", s.Plane, s.Kind, g)
		}
	}
}

func TestGenAIMapNeverEmitsContent(t *testing.T) {
	for _, k := range []Kind{Turn, ToolCall, LLMCall} {
		s := Span{Plane: Agent, Kind: k, Status: Error, Name: "secret prompt", Attrs: Attrs{Tool: "t", Model: "m", Path: "secret"}, IDs: IDs{SessionID: "s", CallID: "c"}}
		g, ok := GenAI(s, GenAIOptions{AgentName: "a", ProviderName: "p"})
		if !ok {
			t.Fatalf("%s unmapped", k)
		}
		for _, a := range g.Attrs {
			if IsGenAIContentKey(a.Key) || !strings.HasPrefix(a.Key, "gen_ai.") && a.Key != GenAIErrorType {
				t.Fatalf("%s emitted %q", k, a.Key)
			}
			if strings.Contains(a.Value, "secret") {
				t.Fatalf("%s leaked span content into %q", k, a.Key)
			}
		}
	}
	for _, k := range []string{"gen_ai.input.messages", "gen_ai.output.messages", "gen_ai.tool.call.arguments", "gen_ai.tool.call.result", "gen_ai.system_instructions"} {
		if !IsGenAIContentKey(k) {
			t.Fatalf("%s not classified as content", k)
		}
	}
	if IsGenAIContentKey(GenAIToolName) {
		t.Fatal("tool name classified as content")
	}
}
