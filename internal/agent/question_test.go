package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

func TestQuestionInteractiveResolver(t *testing.T) {
	DisarmQuestionTool()
	t.Cleanup(DisarmQuestionTool)

	calledPrompt := false
	termResolver := &TerminalResolver{
		Prompt: func(ctx context.Context, item QuestionItem) (AnswerItem, error) {
			calledPrompt = true
			if item.Question != "Deploy to production?" {
				return AnswerItem{}, fmt.Errorf("unexpected question: %s", item.Question)
			}
			return AnswerItem{
				Question: item.Question,
				Selected: []string{"Canary 10%"},
			}, nil
		},
	}

	qDef := ArmQuestionTool(termResolver)
	if qDef.Function.Name != ToolQuestion {
		t.Fatalf("qDef.Function.Name = %q, want %q", qDef.Function.Name, ToolQuestion)
	}

	ctx := context.Background()
	argsJSON := `{"questions":[{"question":"Deploy to production?","options":[{"label":"Canary 10%"},{"label":"Full 100%"}]}]}`
	ref, err := abi.ActiveResolver().Put(ctx, []byte(argsJSON))
	if err != nil {
		t.Fatalf("Put error: %v", err)
	}

	call := &abi.ToolCall{
		Tool:   ToolQuestion,
		Args:   ref,
		Engine: EngineQuestion,
	}

	res, err := activeQuestionEngine.Complete(ctx, call)
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if !calledPrompt {
		t.Fatalf("expected TerminalResolver.Prompt to be called")
	}

	var qr QuestionResult
	if err := json.Unmarshal(res.Payload.Inline, &qr); err != nil {
		t.Fatalf("unmarshal QuestionResult: %v; payload: %s", err, string(res.Payload.Inline))
	}

	if qr.Mode != "interactive" {
		t.Errorf("qr.Mode = %q, want %q", qr.Mode, "interactive")
	}
	if qr.Resolved != "RESOLVED" {
		t.Errorf("qr.Resolved = %q, want %q", qr.Resolved, "RESOLVED")
	}
	if len(qr.Answers) != 1 {
		t.Fatalf("len(qr.Answers) = %d, want 1", len(qr.Answers))
	}
	if len(qr.Answers[0].Selected) != 1 || qr.Answers[0].Selected[0] != "Canary 10%" {
		t.Errorf("qr.Answers[0].Selected = %v, want [\"Canary 10%%%%\"]", qr.Answers[0].Selected)
	}

	// Test TerminalResolver with In and Out reader/writer
	inBuf := strings.NewReader("2\n")
	outBuf := &bytes.Buffer{}
	ioResolver := &TerminalResolver{
		In:  inBuf,
		Out: outBuf,
	}
	ans, err := ioResolver.Resolve(ctx, QuestionItem{
		Question: "Select region",
		Options: []QuestionOption{
			{Label: "us-east-1"},
			{Label: "us-west-2"},
		},
	})
	if err != nil {
		t.Fatalf("ioResolver.Resolve error: %v", err)
	}
	if len(ans.Selected) != 1 || ans.Selected[0] != "us-west-2" {
		t.Errorf("ans.Selected = %v, want [\"us-west-2\"]", ans.Selected)
	}
}

func TestQuestionSupervisorResolver(t *testing.T) {
	DisarmQuestionTool()
	t.Cleanup(DisarmQuestionTool)

	forwarded := false
	supResolver := &SupervisorResolver{
		Forward: func(ctx context.Context, item QuestionItem) (AnswerItem, error) {
			forwarded = true
			if item.Question != "Confirm supervisor routing?" {
				return AnswerItem{}, fmt.Errorf("unexpected question: %s", item.Question)
			}
			return AnswerItem{
				Question: item.Question,
				Selected: []string{"A2A Forward Confirmed"},
			}, nil
		},
	}

	ArmQuestionTool(supResolver)

	ctx := context.Background()
	argsJSON := `{"questions":[{"question":"Confirm supervisor routing?","options":[{"label":"A2A Forward Confirmed"},{"label":"Reject"}]}]}`
	ref, err := abi.ActiveResolver().Put(ctx, []byte(argsJSON))
	if err != nil {
		t.Fatalf("Put error: %v", err)
	}

	call := &abi.ToolCall{
		Tool:   ToolQuestion,
		Args:   ref,
		Engine: EngineQuestion,
	}

	res, err := activeQuestionEngine.Complete(ctx, call)
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if !forwarded {
		t.Fatalf("expected SupervisorResolver.Forward to be called")
	}

	var qr QuestionResult
	if err := json.Unmarshal(res.Payload.Inline, &qr); err != nil {
		t.Fatalf("unmarshal QuestionResult: %v", err)
	}

	if qr.Mode != "supervisor" {
		t.Errorf("qr.Mode = %q, want %q", qr.Mode, "supervisor")
	}
	if qr.Resolved != "RESOLVED" {
		t.Errorf("qr.Resolved = %q, want %q", qr.Resolved, "RESOLVED")
	}
	if len(qr.Answers) != 1 || len(qr.Answers[0].Selected) != 1 || qr.Answers[0].Selected[0] != "A2A Forward Confirmed" {
		t.Errorf("qr.Answers = %+v, want Selected=[\"A2A Forward Confirmed\"]", qr.Answers)
	}
}

func TestQuestionModelEscalationResolver(t *testing.T) {
	DisarmQuestionTool()
	t.Cleanup(DisarmQuestionTool)

	escalated := false
	modelResolver := &ModelEscalationResolver{
		Escalate: func(ctx context.Context, item QuestionItem) (AnswerItem, error) {
			escalated = true
			if item.Question != "Reasoning ambiguity resolution" {
				return AnswerItem{}, fmt.Errorf("unexpected question: %s", item.Question)
			}
			return AnswerItem{
				Question: item.Question,
				Selected: []string{"Secondary Reasoning Consensus"},
				Custom:   "Deep chain-of-thought verification succeeded",
			}, nil
		},
	}

	ArmQuestionTool(modelResolver)

	ctx := context.Background()
	argsJSON := `{"questions":[{"question":"Reasoning ambiguity resolution","options":[{"label":"Secondary Reasoning Consensus"},{"label":"Fallback"}]}]}`
	ref, err := abi.ActiveResolver().Put(ctx, []byte(argsJSON))
	if err != nil {
		t.Fatalf("Put error: %v", err)
	}

	call := &abi.ToolCall{
		Tool:   ToolQuestion,
		Args:   ref,
		Engine: EngineQuestion,
	}

	res, err := activeQuestionEngine.Complete(ctx, call)
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if !escalated {
		t.Fatalf("expected ModelEscalationResolver.Escalate to be called")
	}

	var qr QuestionResult
	if err := json.Unmarshal(res.Payload.Inline, &qr); err != nil {
		t.Fatalf("unmarshal QuestionResult: %v", err)
	}

	if qr.Mode != "escalation" {
		t.Errorf("qr.Mode = %q, want %q", qr.Mode, "escalation")
	}
	if qr.Resolved != "RESOLVED" {
		t.Errorf("qr.Resolved = %q, want %q", qr.Resolved, "RESOLVED")
	}
	if len(qr.Answers) != 1 || len(qr.Answers[0].Selected) != 1 || qr.Answers[0].Selected[0] != "Secondary Reasoning Consensus" {
		t.Errorf("qr.Answers = %+v, want Selected=[\"Secondary Reasoning Consensus\"]", qr.Answers)
	}
	if qr.Answers[0].Custom != "Deep chain-of-thought verification succeeded" {
		t.Errorf("qr.Answers[0].Custom = %q, want deep reasoning detail", qr.Answers[0].Custom)
	}
}

func TestQuestionFallbackResolver(t *testing.T) {
	DisarmQuestionTool()
	t.Cleanup(DisarmQuestionTool)

	// Slow underlying resolver that times out past deadline
	slowResolver := &TerminalResolver{
		Prompt: func(ctx context.Context, item QuestionItem) (AnswerItem, error) {
			select {
			case <-time.After(500 * time.Millisecond):
				return AnswerItem{Question: item.Question, Selected: []string{"Too late"}}, nil
			case <-ctx.Done():
				return AnswerItem{}, ctx.Err()
			}
		},
	}

	fbResolver := &FallbackResolver{
		Underlying: slowResolver,
		Timeout:    25 * time.Millisecond,
	}

	ArmQuestionTool(fbResolver)

	ctx := context.Background()
	argsJSON := `{
		"questions": [{
			"question": "Proceed with default choice?",
			"options": [
				{"label": "Option 0 Default", "description": "Safe default choice"},
				{"label": "Option 1 Alternate", "description": "Secondary choice"}
			]
		}]
	}`
	ref, err := abi.ActiveResolver().Put(ctx, []byte(argsJSON))
	if err != nil {
		t.Fatalf("Put error: %v", err)
	}

	call := &abi.ToolCall{
		Tool:   ToolQuestion,
		Args:   ref,
		Engine: EngineQuestion,
	}

	res, err := activeQuestionEngine.Complete(ctx, call)
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	var qr QuestionResult
	if err := json.Unmarshal(res.Payload.Inline, &qr); err != nil {
		t.Fatalf("unmarshal QuestionResult: %v", err)
	}

	if qr.Mode != "fallback" {
		t.Errorf("qr.Mode = %q, want %q", qr.Mode, "fallback")
	}
	if qr.Resolved != AdvisoryResolvedByFallback {
		t.Errorf("qr.Resolved = %q, want %q", qr.Resolved, AdvisoryResolvedByFallback)
	}
	if len(qr.Answers) != 1 {
		t.Fatalf("len(qr.Answers) = %d, want 1", len(qr.Answers))
	}
	ans := qr.Answers[0]
	if ans.Advisory != AdvisoryResolvedByFallback {
		t.Errorf("ans.Advisory = %q, want %q", ans.Advisory, AdvisoryResolvedByFallback)
	}
	if len(ans.Selected) != 1 || ans.Selected[0] != "Option 0 Default" {
		t.Errorf("ans.Selected = %v, want [\"Option 0 Default\"]", ans.Selected)
	}

	// Also verify nil underlying immediately selects option 0 with fallback advisory
	nilResolver := &FallbackResolver{Underlying: nil}
	item := QuestionItem{
		Question: "Headless question?",
		Options: []QuestionOption{
			{Label: "First Option"},
			{Label: "Second Option"},
		},
	}
	directAns, err := nilResolver.Resolve(ctx, item)
	if err != nil {
		t.Fatalf("nilResolver.Resolve error: %v", err)
	}
	if directAns.Advisory != AdvisoryResolvedByFallback {
		t.Errorf("directAns.Advisory = %q, want %q", directAns.Advisory, AdvisoryResolvedByFallback)
	}
	if len(directAns.Selected) != 1 || directAns.Selected[0] != "First Option" {
		t.Errorf("directAns.Selected = %v, want [\"First Option\"]", directAns.Selected)
	}
}

func TestQuestionArmCodeToolsWithOptions(t *testing.T) {
	DisarmCodeTools()
	t.Cleanup(DisarmCodeTools)

	defs, err := ArmCodeToolsWithOptions(CodeToolsOptions{
		Root:           t.TempDir(),
		EnableQuestion: true,
	})
	if err != nil {
		t.Fatalf("ArmCodeToolsWithOptions error: %v", err)
	}

	foundInDefs := false
	for _, d := range defs {
		if d.Function.Name == ToolQuestion {
			foundInDefs = true
			break
		}
	}
	if !foundInDefs {
		t.Fatalf("ToolQuestion %q not found in returned defs", ToolQuestion)
	}

	catalog := CodeToolCatalog()
	foundInCatalog := false
	for _, d := range catalog {
		if d.Function.Name == ToolQuestion {
			foundInCatalog = true
			break
		}
	}
	if !foundInCatalog {
		t.Fatalf("ToolQuestion %q not found in CodeToolCatalog()", ToolQuestion)
	}

	// Verify DisarmCodeTools disarms question
	DisarmCodeTools()
	if qCatalog := QuestionToolCatalog(); qCatalog != nil {
		t.Errorf("QuestionToolCatalog() = %v, want nil after DisarmCodeTools", qCatalog)
	}
}

func TestQuestionKernelDispatch(t *testing.T) {
	DisarmCodeTools()
	t.Cleanup(DisarmCodeTools)

	fbResolver := &FallbackResolver{
		Underlying: nil,
	}
	ArmQuestionTool(fbResolver)
	Configure()

	k := kernel.New("localtools")
	ctx := context.Background()
	argsJSON := `{"questions":[{"question":"Dispatch question?","options":[{"label":"Default Choice"},{"label":"Other"}]}]}`

	content, ev := execViaKernel(ctx, k, ToolQuestion, argsJSON, EngineQuestion, traceEvent{})
	if ev.Verdict != "ALLOW" {
		t.Errorf("ev.Verdict = %q, want ALLOW", ev.Verdict)
	}

	var qr QuestionResult
	if err := json.Unmarshal([]byte(content), &qr); err != nil {
		t.Fatalf("unmarshal QuestionResult: %v; raw content: %s", err, content)
	}

	if qr.Resolved != AdvisoryResolvedByFallback {
		t.Errorf("qr.Resolved = %q, want %q", qr.Resolved, AdvisoryResolvedByFallback)
	}
	if len(qr.Answers) != 1 || len(qr.Answers[0].Selected) != 1 || qr.Answers[0].Selected[0] != "Default Choice" {
		t.Errorf("qr.Answers = %+v, want Selected=[\"Default Choice\"]", qr.Answers)
	}
}

func TestQuestionTerminalResolver_BufferedStream(t *testing.T) {
	inBuf := strings.NewReader("1\n2\n")
	res := &TerminalResolver{
		In:  inBuf,
		Out: &bytes.Buffer{},
	}
	ctx := context.Background()

	ans1, err := res.Resolve(ctx, QuestionItem{
		Question: "Question 1",
		Options: []QuestionOption{
			{Label: "First"},
			{Label: "Second"},
		},
	})
	if err != nil {
		t.Fatalf("Resolve q1 error: %v", err)
	}
	if len(ans1.Selected) != 1 || ans1.Selected[0] != "First" {
		t.Errorf("ans1.Selected = %v, want [\"First\"]", ans1.Selected)
	}

	ans2, err := res.Resolve(ctx, QuestionItem{
		Question: "Question 2",
		Options: []QuestionOption{
			{Label: "Choice A"},
			{Label: "Choice B"},
		},
	})
	if err != nil {
		t.Fatalf("Resolve q2 error: %v", err)
	}
	if len(ans2.Selected) != 1 || ans2.Selected[0] != "Choice B" {
		t.Errorf("ans2.Selected = %v, want [\"Choice B\"]", ans2.Selected)
	}
}

func TestQuestionTerminalResolver_CanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	res := &TerminalResolver{
		In:  strings.NewReader("1\n"),
		Out: &bytes.Buffer{},
	}
	_, err := res.Resolve(ctx, QuestionItem{
		Question: "Cancelled?",
		Options:  []QuestionOption{{Label: "Yes"}},
	})
	if err == nil || !strings.Contains(err.Error(), "context canceled") {
		t.Fatalf("expected context canceled error, got: %v", err)
	}
}

func TestQuestionFallbackResolver_ParentContextCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	fb := &FallbackResolver{
		Underlying: &TerminalResolver{
			Prompt: func(ctx context.Context, item QuestionItem) (AnswerItem, error) {
				return AnswerItem{Question: item.Question, Selected: []string{"Underlying"}}, nil
			},
		},
		Timeout: 50 * time.Millisecond,
	}

	_, err := fb.Resolve(ctx, QuestionItem{
		Question: "Parent cancelled?",
		Options:  []QuestionOption{{Label: "Option 1"}},
	})
	if err == nil || !strings.Contains(err.Error(), "context canceled") {
		t.Fatalf("expected context canceled error, got: %v", err)
	}
}

func TestQuestionFallbackResolver_ZeroTimeout(t *testing.T) {
	called := false
	fb := &FallbackResolver{
		Underlying: &TerminalResolver{
			Prompt: func(ctx context.Context, item QuestionItem) (AnswerItem, error) {
				called = true
				return AnswerItem{Question: item.Question, Selected: []string{"Immediate"}}, nil
			},
		},
		Timeout: 0,
	}

	ans, err := fb.Resolve(context.Background(), QuestionItem{
		Question: "Zero timeout?",
		Options:  []QuestionOption{{Label: "Immediate"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !called {
		t.Fatalf("expected underlying resolver to be called")
	}
	if len(ans.Selected) != 1 || ans.Selected[0] != "Immediate" {
		t.Errorf("ans.Selected = %v, want [\"Immediate\"]", ans.Selected)
	}
}

func TestQuestionCodeToolCatalog_QuestionOnly(t *testing.T) {
	DisarmCodeTools()
	t.Cleanup(DisarmCodeTools)

	ArmQuestionTool(nil)

	cat := CodeToolCatalog()
	if len(cat) != 1 || cat[0].Function.Name != ToolQuestion {
		t.Fatalf("CodeToolCatalog() = %+v, want 1 entry with %q", cat, ToolQuestion)
	}

	meta := codeToolMeta(ToolQuestion)
	if meta == nil || meta["destructive"] != "false" {
		t.Fatalf("codeToolMeta(ToolQuestion) = %+v, want destructive=false", meta)
	}
}
