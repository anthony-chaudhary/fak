package agent

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/abi"
)

type droppedCallPlanner struct {
	completion Completion
	requests   [][]Message
}

func (p *droppedCallPlanner) Model() string { return "dropped-call-test" }

func (p *droppedCallPlanner) Complete(_ context.Context, messages []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	p.requests = append(p.requests, append([]Message(nil), messages...))
	comp := p.completion
	return &comp, nil
}

// fak-test:runtime fast est=500ms
func TestArmToolCallDropNamesReason(t *testing.T) {
	for _, reason := range []abi.ReasonCode{abi.ReasonOversize, abi.ReasonMalformed, abi.ReasonNone} {
		for _, budget := range []int{-1, 1} {
			t.Run(fmt.Sprintf("%s/budget=%d", abi.ReasonName(reason), budget), func(t *testing.T) {
				const reply = "REPLY-SECRET-13562"
				const argument = "ARGUMENT-SECRET-13562"
				const raw = "RAW-SECRET-13562"
				p := &droppedCallPlanner{completion: Completion{
					Message: Message{Role: RoleAssistant, Content: reply + `<tool_call>{"name":"Bash","arguments":{"command":"` + argument},
					Raw:     []byte(raw), FinishReason: "length", Usage: Usage{CompletionTokens: 24},
					ToolCallsDropped: true, ToolCallsDroppedReason: reason,
				}}
				var trace []traceEvent
				m, err := RunArm(context.Background(), p, "task", true, 5, &trace, WithInfraRepromptBudget(budget))
				wantCalls := 1
				if budget > 0 {
					wantCalls += budget
				}
				if len(p.requests) != wantCalls || m.Turns != wantCalls || m.InfraReprompts != wantCalls-1 || m.CompletionTokens != 24*wantCalls {
					t.Fatalf("retry/accounting changed: requests=%d metrics=%+v", len(p.requests), m)
				}
				if m.ToolCalls != 0 || m.EngineCalls != 0 || len(trace) != 0 || m.FinalAnswer != "" || m.HitTurnCap {
					t.Fatalf("dropped call escaped the refusal: trace=%+v metrics=%+v", trace, m)
				}
				want := fmt.Sprintf("fak arm turn %d: upstream announced tool_calls but none parsed; refusing to skip adjudication", wantCalls)
				if reason != abi.ReasonNone {
					want += " (" + abi.ReasonName(reason) + ")"
				}
				if err == nil || err.Error() != want {
					t.Errorf("terminal error = %v, want %q", err, want)
				}
				for i, request := range p.requests {
					if i > 0 {
						last := request[len(request)-1]
						if last.Role != RoleUser || last.Content != infraContinuationText("TRANSPORT_EXHAUSTED") {
							t.Errorf("continuation changed: %+v", last)
						}
					}
					for _, message := range request {
						for _, secret := range []string{reply, argument, raw} {
							if strings.Contains(message.Content, secret) {
								t.Errorf("model request leaked %s", secret)
							}
						}
					}
				}
				if err != nil {
					for _, secret := range []string{reply, argument, raw} {
						if strings.Contains(err.Error(), secret) {
							t.Errorf("terminal error leaked %s", secret)
						}
					}
				}
			})
		}
	}
}
