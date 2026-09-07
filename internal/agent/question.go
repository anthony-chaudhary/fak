package agent

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// question.go — interactive clarify/question tool with multi-target resolution
// (terminal, supervisor A2A, model escalation, and headless fallback) (#11422).

const (
	// ToolQuestion is the planner/model-facing tool name.
	ToolQuestion = "question"

	// EngineQuestion is the registered kernel engine ID.
	EngineQuestion = "agent.question"

	// RungNameQuestion is the adjudicator link name.
	RungNameQuestion = "questiontool"

	// questionToolRank places the gate alongside context_control before the monitor.
	questionToolRank = 23

	// AdvisoryResolvedByFallback indicates the question answer was selected by fallback.
	AdvisoryResolvedByFallback = "RESOLVED_BY_FALLBACK"
)

// QuestionOption represents a single selectable choice for a question.
type QuestionOption struct {
	Label       string `json:"label"`
	Description string `json:"description,omitempty"`
}

// QuestionItem defines a single clarify or selection prompt.
type QuestionItem struct {
	Question string           `json:"question"`
	Header   string           `json:"header,omitempty"`
	Options  []QuestionOption `json:"options,omitempty"`
	Multiple bool             `json:"multiple,omitempty"`
	Custom   bool             `json:"custom,omitempty"`
}

// QuestionArgs contains the tool arguments for question.
type QuestionArgs struct {
	Questions []QuestionItem `json:"questions"`
}

// AnswerItem represents the answer provided for a single question.
type AnswerItem struct {
	Question string   `json:"question"`
	Selected []string `json:"selected,omitempty"`
	Custom   string   `json:"custom,omitempty"`
	Advisory string   `json:"advisory,omitempty"`
}

// QuestionResult represents the outcome of resolving all questions.
type QuestionResult struct {
	Answers  []AnswerItem `json:"answers"`
	Resolved string       `json:"resolved"`
	Mode     string       `json:"mode"`
}

// QuestionResolver resolves a single question item into an answer item.
type QuestionResolver interface {
	Resolve(ctx context.Context, item QuestionItem) (AnswerItem, error)
}

// ---------------------------------------------------------------------------
// 1. TerminalResolver
// ---------------------------------------------------------------------------

// TerminalResolver is an interactive/terminal prompter with a custom prompt function or reader/writer.
type TerminalResolver struct {
	Prompt   func(ctx context.Context, item QuestionItem) (AnswerItem, error)
	In       io.Reader
	Out      io.Writer
	ModeName string

	mu  sync.Mutex
	rdr *bufio.Reader
}

func (t *TerminalResolver) Mode() string {
	if t != nil && t.ModeName != "" {
		return t.ModeName
	}
	return "interactive"
}

func (t *TerminalResolver) Resolve(ctx context.Context, item QuestionItem) (AnswerItem, error) {
	if t == nil {
		return AnswerItem{}, fmt.Errorf("terminal resolver is nil")
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	if err := ctx.Err(); err != nil {
		return AnswerItem{}, err
	}

	if t.Prompt != nil {
		return t.Prompt(ctx, item)
	}
	if t.In != nil {
		if t.rdr == nil {
			t.rdr = bufio.NewReader(t.In)
		}
		out := t.Out
		if out == nil {
			out = io.Discard
		}
		if item.Header != "" {
			fmt.Fprintf(out, "[%s] %s\n", item.Header, item.Question)
		} else {
			fmt.Fprintf(out, "%s\n", item.Question)
		}
		for i, opt := range item.Options {
			if opt.Description != "" {
				fmt.Fprintf(out, "  %d) %s: %s\n", i+1, opt.Label, opt.Description)
			} else {
				fmt.Fprintf(out, "  %d) %s\n", i+1, opt.Label)
			}
		}
		if item.Custom {
			fmt.Fprintf(out, "  c) Type custom answer\n")
		}
		fmt.Fprintf(out, "> ")

		rawLine, err := t.rdr.ReadString('\n')
		if err != nil && !errors.Is(err, io.EOF) {
			return AnswerItem{}, err
		}
		if len(rawLine) == 0 && errors.Is(err, io.EOF) {
			return AnswerItem{
				Question: item.Question,
			}, nil
		}
		line := strings.TrimSpace(rawLine)
		if line == "" && len(item.Options) > 0 {
			return AnswerItem{
				Question: item.Question,
				Selected: []string{item.Options[0].Label},
			}, nil
		}
		if item.Multiple && strings.Contains(line, ",") {
			parts := strings.Split(line, ",")
			var selected []string
			for _, p := range parts {
				p = strings.TrimSpace(p)
				if p == "" {
					continue
				}
				if num, err := strconv.Atoi(p); err == nil && num >= 1 && num <= len(item.Options) {
					selected = append(selected, item.Options[num-1].Label)
					continue
				}
				matched := false
				for _, opt := range item.Options {
					if strings.EqualFold(opt.Label, p) {
						selected = append(selected, opt.Label)
						matched = true
						break
					}
				}
				if !matched {
					selected = append(selected, p)
				}
			}
			if len(selected) > 0 {
				return AnswerItem{
					Question: item.Question,
					Selected: selected,
				}, nil
			}
		}
		if num, err := strconv.Atoi(line); err == nil && num >= 1 && num <= len(item.Options) {
			return AnswerItem{
				Question: item.Question,
				Selected: []string{item.Options[num-1].Label},
			}, nil
		}
		for _, opt := range item.Options {
			if strings.EqualFold(opt.Label, line) {
				return AnswerItem{
					Question: item.Question,
					Selected: []string{opt.Label},
				}, nil
			}
		}
		return AnswerItem{
			Question: item.Question,
			Custom:   line,
		}, nil
	}
	return AnswerItem{}, fmt.Errorf("terminal resolver has no prompt function or input reader")
}

// ---------------------------------------------------------------------------
// 2. SupervisorResolver
// ---------------------------------------------------------------------------

// SupervisorResolver forwards questions to a supervisor A2A message routing callback.
type SupervisorResolver struct {
	Forward  func(ctx context.Context, item QuestionItem) (AnswerItem, error)
	ModeName string
}

func (s *SupervisorResolver) Mode() string {
	if s != nil && s.ModeName != "" {
		return s.ModeName
	}
	return "supervisor"
}

func (s *SupervisorResolver) Resolve(ctx context.Context, item QuestionItem) (AnswerItem, error) {
	if s == nil || s.Forward == nil {
		return AnswerItem{}, fmt.Errorf("supervisor resolver has no forward handler")
	}
	return s.Forward(ctx, item)
}

// ---------------------------------------------------------------------------
// 3. ModelEscalationResolver
// ---------------------------------------------------------------------------

// ModelEscalationResolver forwards questions to a secondary reasoning model completion callback.
type ModelEscalationResolver struct {
	Escalate func(ctx context.Context, item QuestionItem) (AnswerItem, error)
	ModeName string
}

func (m *ModelEscalationResolver) Mode() string {
	if m != nil && m.ModeName != "" {
		return m.ModeName
	}
	return "escalation"
}

func (m *ModelEscalationResolver) Resolve(ctx context.Context, item QuestionItem) (AnswerItem, error) {
	if m == nil || m.Escalate == nil {
		return AnswerItem{}, fmt.Errorf("model escalation resolver has no escalate handler")
	}
	return m.Escalate(ctx, item)
}

// ---------------------------------------------------------------------------
// 4. FallbackResolver
// ---------------------------------------------------------------------------

// FallbackResolver wraps an underlying resolver with a timeout; if timeout expires (or if underlying is nil),
// selects option 0 with Advisory: AdvisoryResolvedByFallback, Selected: []string{item.Options[0].Label}.
type FallbackResolver struct {
	Underlying QuestionResolver
	Timeout    time.Duration
	ModeName   string
}

func (f *FallbackResolver) Mode() string {
	if f != nil && f.ModeName != "" {
		return f.ModeName
	}
	if f != nil && f.Underlying != nil {
		if mr, ok := f.Underlying.(interface{ Mode() string }); ok {
			return mr.Mode()
		}
	}
	return "fallback"
}

func (f *FallbackResolver) fallbackAnswer(item QuestionItem) AnswerItem {
	ans := AnswerItem{
		Question: item.Question,
		Advisory: AdvisoryResolvedByFallback,
	}
	if len(item.Options) > 0 {
		ans.Selected = []string{item.Options[0].Label}
	}
	return ans
}

func (f *FallbackResolver) Resolve(ctx context.Context, item QuestionItem) (AnswerItem, error) {
	if err := ctx.Err(); err != nil {
		return AnswerItem{}, err
	}
	if f == nil || f.Underlying == nil {
		var dummy FallbackResolver
		return dummy.fallbackAnswer(item), nil
	}
	if f.Timeout <= 0 {
		return f.Underlying.Resolve(ctx, item)
	}

	resolveCtx, cancel := context.WithTimeout(ctx, f.Timeout)
	defer cancel()

	type resolveRes struct {
		ans AnswerItem
		err error
	}

	ch := make(chan resolveRes, 1)
	go func() {
		ans, err := f.Underlying.Resolve(resolveCtx, item)
		ch <- resolveRes{ans: ans, err: err}
	}()

	select {
	case <-ctx.Done():
		return AnswerItem{}, ctx.Err()
	case <-resolveCtx.Done():
		if err := ctx.Err(); err != nil {
			return AnswerItem{}, err
		}
		return f.fallbackAnswer(item), nil
	case r := <-ch:
		if err := ctx.Err(); err != nil {
			return AnswerItem{}, err
		}
		if r.err != nil && errors.Is(r.err, context.DeadlineExceeded) {
			return f.fallbackAnswer(item), nil
		}
		return r.ans, r.err
	}
}

// ---------------------------------------------------------------------------
// Lifecycle & Kernel Integration
// ---------------------------------------------------------------------------

type questionState struct {
	resolver QuestionResolver
}

var (
	armedQuestion        atomic.Pointer[questionState]
	questionGateOnce     sync.Once
	questionEngineOnce   sync.Once
	activeQuestionEngine *questionEngine
)

// ArmQuestionTool arms the question tool with the given resolver, registers its engine,
// installs the questionGate adjudicator at rank 23, and returns the tool definition.
func ArmQuestionTool(resolver QuestionResolver) ToolDef {
	if resolver == nil {
		resolver = &FallbackResolver{}
	}
	st := &questionState{resolver: resolver}
	armedQuestion.Store(st)

	questionEngineOnce.Do(func() {
		activeQuestionEngine = &questionEngine{}
		abi.RegisterEngine(EngineQuestion, activeQuestionEngine)
	})

	questionGateOnce.Do(func() {
		abi.RegisterAdjudicator(questionToolRank, questionGate{})
	})

	return QuestionToolDef()
}

// DisarmQuestionTool unarms the question tool, restoring the unarmed state.
func DisarmQuestionTool() {
	armedQuestion.Store(nil)
}

// QuestionToolCatalog returns the tool definitions for the question tool when armed, or nil.
func QuestionToolCatalog() []ToolDef {
	if armedQuestion.Load() == nil {
		return nil
	}
	return []ToolDef{QuestionToolDef()}
}

// QuestionToolDef returns the tool definition and JSON schema for the question tool.
func QuestionToolDef() ToolDef {
	return ToolDef{
		Type: "function",
		Function: ToolDefFunction{
			Name: ToolQuestion,
			Description: "Ask interactive clarify or selection questions with multi-target resolution " +
				"(terminal, supervisor A2A, model escalation, or fallback).",
			Parameters: rawSchema(`{
  "type": "object",
  "properties": {
    "questions": {
      "type": "array",
      "description": "Questions to ask the user, supervisor, or reasoning model",
      "items": {
        "type": "object",
        "properties": {
          "question": {
            "type": "string",
            "description": "The question to ask"
          },
          "header": {
            "type": "string",
            "description": "Short topic header or label"
          },
          "options": {
            "type": "array",
            "description": "Available choices for selection",
            "items": {
              "type": "object",
              "properties": {
                "label": {
                  "type": "string",
                  "description": "Option display text"
                },
                "description": {
                  "type": "string",
                  "description": "Explanation of option choice"
                }
              },
              "required": ["label"]
            }
          },
          "multiple": {
            "type": "boolean",
            "description": "Allow selecting multiple choices"
          },
          "custom": {
            "type": "boolean",
            "description": "Allow typing a custom answer"
          }
        },
        "required": ["question"]
      }
    }
  },
  "required": ["questions"]
}`),
		},
	}
}

// ---------------------------------------------------------------------------
// Adjudicator Gate & Engine Implementation
// ---------------------------------------------------------------------------

type questionGate struct{}

func (questionGate) Caps() []abi.Capability { return nil }

func (questionGate) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	st := armedQuestion.Load()
	if st == nil {
		return abi.Verdict{Kind: abi.VerdictDefer, By: RungNameQuestion}
	}
	if c.Tool == ToolQuestion {
		c.Engine = EngineQuestion
		return abi.Verdict{Kind: abi.VerdictAllow, By: RungNameQuestion}
	}
	return abi.Verdict{Kind: abi.VerdictDefer, By: RungNameQuestion}
}

type questionEngine struct{}

func (e *questionEngine) Caps() []abi.Capability { return nil }
func (e *questionEngine) WeightBearing() bool    { return false }

func (e *questionEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	body, _ := decodeCallArgs(ctx, c.Args)
	st := armedQuestion.Load()
	if st == nil {
		errResp, _ := json.Marshal(map[string]any{
			"error": "question tool is unarmed",
		})
		return engineResult(ctx, c, body, errResp, true, EngineQuestion), nil
	}

	var args QuestionArgs
	if len(body) > 0 {
		if err := json.Unmarshal(body, &args); err != nil || len(args.Questions) == 0 {
			var items []QuestionItem
			if err2 := json.Unmarshal(body, &items); err2 == nil && len(items) > 0 {
				args.Questions = items
			} else {
				var single QuestionItem
				if err3 := json.Unmarshal(body, &single); err3 == nil && single.Question != "" {
					args.Questions = []QuestionItem{single}
				} else if err != nil {
					errResp, _ := json.Marshal(map[string]any{
						"error": fmt.Sprintf("invalid arguments JSON: %v", err),
					})
					return engineResult(ctx, c, body, errResp, true, EngineQuestion), nil
				}
			}
		}
	}

	if len(args.Questions) == 0 {
		errResp, _ := json.Marshal(map[string]any{
			"error": "no questions provided",
		})
		return engineResult(ctx, c, body, errResp, true, EngineQuestion), nil
	}

	answers := make([]AnswerItem, 0, len(args.Questions))
	hasFallback := false
	for _, q := range args.Questions {
		ans, err := st.resolver.Resolve(ctx, q)
		if err != nil {
			errResp, _ := json.Marshal(map[string]any{
				"error": fmt.Sprintf("failed to resolve question %q: %v", q.Question, err),
			})
			return engineResult(ctx, c, body, errResp, true, EngineQuestion), nil
		}
		if ans.Question == "" {
			ans.Question = q.Question
		}
		if ans.Advisory == AdvisoryResolvedByFallback {
			hasFallback = true
		}
		answers = append(answers, ans)
	}

	mode := "interactive"
	if mr, ok := st.resolver.(interface{ Mode() string }); ok {
		mode = mr.Mode()
	}
	resolved := "RESOLVED"
	if hasFallback {
		resolved = AdvisoryResolvedByFallback
		mode = "fallback"
	}

	res := QuestionResult{
		Answers:  answers,
		Resolved: resolved,
		Mode:     mode,
	}
	respBytes, _ := json.Marshal(res)
	return engineResult(ctx, c, body, respBytes, false, EngineQuestion), nil
}

func questionMeta(tool string) (map[string]string, bool) {
	if armedQuestion.Load() == nil {
		return nil, false
	}
	if tool == ToolQuestion {
		return map[string]string{
			"readOnlyHint":   "false",
			"idempotentHint": "false",
			"destructive":    "false",
			"consistency":    "BEST_EFFORT",
		}, true
	}
	return nil, false
}
