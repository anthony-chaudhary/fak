package gateway

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/session"
)

type stopGatePlanner struct {
	calls int
	seen  []agent.Message
}

func (p *stopGatePlanner) Model() string { return "stop-gate-test" }
func (p *stopGatePlanner) Complete(_ context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls++
	p.seen = append([]agent.Message(nil), messages...)
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "done"},
		FinishReason: "stop",
		Usage:        agent.Usage{PromptTokens: 1, CompletionTokens: 1},
	}, nil
}

func TestStopGate_HoldsUntilWitness(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterReason(ReasonStopUnwitnessed, ReasonStopUnwitnessedName)
	j := journal.OpenMemory()
	abi.RegisterEmitter(j)

	planner := &stopGatePlanner{}
	checks := 0
	srv := &Server{
		planner:        planner,
		nativeMaxTurns: 3,
		decideSession:  func(context.Context, string) SessionVerdict { return SessionVerdict{Proceed: true} },
		stopGate: func(context.Context, string) StopGateResult {
			checks++
			return StopGateResult{Satisfied: checks > 1, Witness: "file:proof.sha256"}
		},
	}
	arm, err := srv.runNativeArm(context.Background(), &agent.AnthropicMessagesRequest{
		Messages: []agent.Message{{Role: agent.RoleUser, Content: "finish"}},
	}, "trace-stop-gate")
	if err != nil {
		t.Fatal(err)
	}
	if arm.FinalAnswer != "done" || planner.calls != 2 {
		t.Fatalf("final=%q calls=%d, want held once then completed", arm.FinalAnswer, planner.calls)
	}
	foundFeedback := false
	for _, m := range planner.seen {
		if strings.Contains(m.Content, "STOP_UNWITNESSED") && strings.Contains(m.Content, "file:proof.sha256") {
			foundFeedback = true
		}
	}
	if !foundFeedback {
		t.Fatalf("second turn did not receive named missing witness: %+v", planner.seen)
	}
	rows := j.Recent(0)
	if len(rows) != 1 || rows[0].Reason != ReasonStopUnwitnessedName || rows[0].Witness != "file:proof.sha256" {
		t.Fatalf("journal rows=%+v, want one STOP_UNWITNESSED with named witness", rows)
	}
}

func TestStopGate_BudgetDrainBeatsHold(t *testing.T) {
	planner := &stopGatePlanner{}
	decisions := 0
	checks := 0
	srv := &Server{
		planner:        planner,
		nativeMaxTurns: 3,
		decideSession: func(context.Context, string) SessionVerdict {
			decisions++
			if decisions == 1 {
				return SessionVerdict{Proceed: true}
			}
			return SessionVerdict{Proceed: false, Stop: true, Reason: session.ReasonBudgetTurns}
		},
		stopGate: func(context.Context, string) StopGateResult {
			checks++
			return StopGateResult{Witness: "dos verify plan/phase"}
		},
	}
	arm, err := srv.runNativeArm(context.Background(), &agent.AnthropicMessagesRequest{
		Messages: []agent.Message{{Role: agent.RoleUser, Content: "finish"}},
	}, "trace-budget")
	if err != nil {
		t.Fatal(err)
	}
	if arm.FinalAnswer != "" || arm.StoppedBySession != session.ReasonBudgetTurns {
		t.Fatalf("final=%q stopped=%q, want budget drain", arm.FinalAnswer, arm.StoppedBySession)
	}
	if planner.calls != 1 || checks != 1 || decisions != 2 {
		t.Fatalf("calls=%d checks=%d decisions=%d, want 1/1/2", planner.calls, checks, decisions)
	}
}

func TestStopGate_StreamDoesNotLeakRejectedFinal(t *testing.T) {
	planner := &stopGatePlanner{}
	checks := 0
	srv := &Server{
		planner:        planner,
		nativeMaxTurns: 3,
		stopGate: func(context.Context, string) StopGateResult {
			checks++
			return StopGateResult{Satisfied: checks > 1, Witness: "file:proof.sha256"}
		},
	}
	var streamed strings.Builder
	arm, err := srv.runNativeArmStream(context.Background(), &agent.AnthropicMessagesRequest{
		Messages: []agent.Message{{Role: agent.RoleUser, Content: "finish"}},
	}, "trace-stream", func(delta string) error {
		streamed.WriteString(delta)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if planner.calls != 2 || arm.FinalAnswer != "done" || streamed.String() != "done" {
		t.Fatalf("calls=%d final=%q streamed=%q, want held answer hidden and one witnessed final", planner.calls, arm.FinalAnswer, streamed.String())
	}
}
