package gym

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// StressProfile configures the stress parameters for a closed-loop scenario simulation.
type StressProfile struct {
	MaxTurns         int           `json:"max_turns"`
	TargetToolCalls  int           `json:"target_tool_calls"`
	PayloadSizeBytes int           `json:"payload_size_bytes"`
	InduceYield      bool          `json:"induce_yield"`
	SimulateRunaway  bool          `json:"simulate_runaway"`
	ExpectRestore    bool          `json:"expect_restore"`
	TurnTimeout      time.Duration `json:"turn_timeout"`
}

// ScriptEngine specifies the programmable model mock interface used during simulation.
type ScriptEngine interface {
	agent.Planner
}

type scriptEngineFunc struct {
	model string
	fn    func(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error)
}

func (s *scriptEngineFunc) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	return s.fn(ctx, messages, tools, opts...)
}

func (s *scriptEngineFunc) Model() string {
	if s.model != "" {
		return s.model
	}
	return "script-engine-mock"
}

// NewScriptEngineFunc creates a ScriptEngine from a functional completion handler.
func NewScriptEngineFunc(modelName string, fn func(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error)) ScriptEngine {
	return &scriptEngineFunc{
		model: modelName,
		fn:    fn,
	}
}

// profileScriptEngine synthesizes deterministic mock completions based on the StressProfile.
type profileScriptEngine struct {
	profile         StressProfile
	mu              sync.Mutex
	stepsDispatched int
}

func parseElidedMarker(content string) (toolName string, hexID string, found bool) {
	idx := strings.Index(content, "recover original via ")
	if idx == -1 {
		return "", "", false
	}
	sub := content[idx+len("recover original via "):]
	parts := strings.SplitN(sub, " id=sha256:", 2)
	if len(parts) != 2 {
		return "", "", false
	}
	toolName = strings.TrimSpace(parts[0])
	rest := parts[1]
	endIdx := strings.IndexAny(rest, "] \t\n")
	if endIdx != -1 {
		hexID = rest[:endIdx]
	} else {
		hexID = rest
	}
	return toolName, strings.TrimSpace(hexID), true
}

func (p *profileScriptEngine) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// 1. Simulate runaway loop for livelock testing
	if p.profile.SimulateRunaway {
		return &agent.Completion{
			Message: agent.Message{
				Role:    agent.RoleAssistant,
				Content: "Polling resource status...",
				ToolCalls: []agent.ToolCall{
					{
						ID:   "call_runaway_repeat",
						Type: "function",
						Function: agent.Func{
							Name:      "read_status",
							Arguments: `{"query":"status","retry":true}`,
						},
					},
				},
			},
			FinishReason: "tool_calls",
			Usage:        agent.Usage{PromptTokens: 120, CompletionTokens: 25, TotalTokens: 145},
		}, nil
	}

	// 2. Expect restore testing
	if p.profile.ExpectRestore {
		var elidedTool, elidedHex string
		var hasElided bool
		for _, m := range messages {
			if tn, hx, ok := parseElidedMarker(m.Content); ok {
				elidedTool = tn
				elidedHex = hx
				hasElided = true
				break
			}
		}

		var alreadyRestored bool
		for _, m := range messages {
			if strings.Contains(m.Content, "[fak: restored context") ||
				strings.Contains(m.Content, "Successfully restored context") {
				alreadyRestored = true
				break
			}
		}

		if hasElided && !alreadyRestored {
			if elidedTool == "" || elidedTool == "fak_context_restore" {
				elidedTool = "mcp__fak__fak_context_restore"
			}
			return &agent.Completion{
				Message: agent.Message{
					Role:    agent.RoleAssistant,
					Content: "Detected elided context, restoring original payload...",
					ToolCalls: []agent.ToolCall{
						{
							ID:   "call_restore_elided",
							Type: "function",
							Function: agent.Func{
								Name:      elidedTool,
								Arguments: fmt.Sprintf(`{"id":"sha256:%s"}`, elidedHex),
							},
						},
					},
				},
				FinishReason: "tool_calls",
				Usage:        agent.Usage{PromptTokens: 250, CompletionTokens: 30, TotalTokens: 280},
			}, nil
		}

		if alreadyRestored {
			return &agent.Completion{
				Message: agent.Message{
					Role:    agent.RoleAssistant,
					Content: "Successfully restored context. Task complete.",
				},
				FinishReason: "stop",
				Usage:        agent.Usage{PromptTokens: 280, CompletionTokens: 15, TotalTokens: 295},
			}, nil
		}
	}

	// 3. Sequential subturn tool calls
	if p.profile.TargetToolCalls > 0 && p.stepsDispatched < p.profile.TargetToolCalls {
		p.stepsDispatched++
		step := p.stepsDispatched
		toolName := "read_file"
		for _, t := range tools {
			if !strings.Contains(t.Function.Name, "restore") {
				toolName = t.Function.Name
				break
			}
		}
		return &agent.Completion{
			Message: agent.Message{
				Role:    agent.RoleAssistant,
				Content: fmt.Sprintf("Executing step %d...", step),
				ToolCalls: []agent.ToolCall{
					{
						ID:   fmt.Sprintf("call_step_%d", step),
						Type: "function",
						Function: agent.Func{
							Name:      toolName,
							Arguments: fmt.Sprintf(`{"path":"file_%d.txt"}`, step),
						},
					},
				},
			},
			FinishReason: "tool_calls",
			Usage:        agent.Usage{PromptTokens: 100 + step*40, CompletionTokens: 20, TotalTokens: 120 + step*40},
		}, nil
	}

	// 4. Conclude scenario
	return &agent.Completion{
		Message: agent.Message{
			Role:    agent.RoleAssistant,
			Content: fmt.Sprintf("All steps completed successfully (%d steps).", p.stepsDispatched),
		},
		FinishReason: "stop",
		Usage:        agent.Usage{PromptTokens: 100 + p.stepsDispatched*40, CompletionTokens: 15, TotalTokens: 115 + p.stepsDispatched*40},
	}, nil
}

func (p *profileScriptEngine) Model() string {
	return "gym-profile-model"
}

// NewProfileScriptEngine returns a ScriptEngine conforming to the behavior of a StressProfile.
func NewProfileScriptEngine(profile StressProfile) ScriptEngine {
	return &profileScriptEngine{
		profile: profile,
	}
}

// Scenario describes a closed-loop multi-turn stress test scenario.
type Scenario struct {
	ID          string          `json:"id"`
	Name        string          `json:"name"`
	Dialect     string          `json:"dialect"`
	Profile     StressProfile   `json:"profile"`
	MockScript  ScriptEngine    `json:"-"`
	ClientTools []agent.ToolDef `json:"client_tools,omitempty"`
}

// ScenarioRunner pairs an EphemeralGateway with a HeadlessHarness to execute closed-loop simulation.
type ScenarioRunner struct{}

// NewScenarioRunner constructs a new ScenarioRunner.
func NewScenarioRunner() *ScenarioRunner {
	return &ScenarioRunner{}
}

// Run executes the scenario closed loop, collects telemetry, and renders a verified GymReceipt.
func (r *ScenarioRunner) Run(ctx context.Context, s Scenario) (*GymReceipt, error) {
	if s.ID == "" {
		s.ID = fmt.Sprintf("gym-scenario-%d", time.Now().UnixNano())
	}
	if s.Profile.MaxTurns <= 0 {
		s.Profile.MaxTurns = 10
	}
	if s.Profile.TurnTimeout <= 0 {
		s.Profile.TurnTimeout = 20 * time.Second
	}

	if len(s.ClientTools) == 0 {
		s.ClientTools = []agent.ToolDef{
			{
				Type: "function",
				Function: agent.ToolDefFunction{
					Name:        "read_file",
					Description: "Read file contents",
					Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`),
				},
			},
			{
				Type: "function",
				Function: agent.ToolDefFunction{
					Name:        "read_status",
					Description: "Check resource status",
					Parameters:  json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"},"retry":{"type":"boolean"}},"required":["query"]}`),
				},
			},
			{
				Type: "function",
				Function: agent.ToolDefFunction{
					Name:        "mcp__fak__fak_context_restore",
					Description: "Restore dropped context by content-addressed sha256 id",
					Parameters:  json.RawMessage(`{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}`),
				},
			},
		}
	}

	planner := s.MockScript
	if planner == nil {
		planner = NewProfileScriptEngine(s.Profile)
	}

	gwOpts := EphemeralGatewayOptions{
		DeferColdTools: true,
		CustomPlanner:  planner,
	}

	if s.Profile.InduceYield {
		gwOpts.MaxSubturnToolCalls = 2
		gwOpts.MaxSubturnTokens = 100
	}

	eg, err := NewEphemeralGateway(gwOpts)
	if err != nil {
		return nil, fmt.Errorf("failed to create ephemeral gateway: %w", err)
	}
	defer func() { _ = eg.Close() }()

	toolRunner := func(ctx context.Context, call agent.ToolCall) (string, error) {
		if strings.Contains(call.Function.Name, "restore") || call.Function.Name == "fak_context_restore" {
			return "[fak: restored context for " + call.Function.Arguments + "]", nil
		}
		if s.Profile.PayloadSizeBytes > 0 {
			var b strings.Builder
			b.Grow(s.Profile.PayloadSizeBytes)
			for i := 0; b.Len() < s.Profile.PayloadSizeBytes; i++ {
				fmt.Fprintf(&b, "timestamp=%d level=info component=worker-%05d operation=%s verified=true\n", 1700000000+i, i, call.Function.Name)
			}
			res := b.String()
			if len(res) > s.Profile.PayloadSizeBytes {
				res = res[:s.Profile.PayloadSizeBytes]
			}
			return res, nil
		}
		return fmt.Sprintf("result for tool %s", call.Function.Name), nil
	}

	harnessOpts := HeadlessHarnessOptions{
		GatewayURL:    eg.URL(),
		Dialect:       s.Dialect,
		ScenarioID:    s.ID,
		MaxTurns:      s.Profile.MaxTurns,
		TurnTimeout:   s.Profile.TurnTimeout,
		ClientTools:   s.ClientTools,
		ToolRunner:    toolRunner,
		InitialPrompt: fmt.Sprintf("Start scenario %s (ID=%s)", s.Name, s.ID),
	}

	harness := NewHeadlessHarness(harnessOpts)
	receipt, err := harness.Execute(ctx)
	if err != nil {
		return nil, fmt.Errorf("harness execution error: %w", err)
	}

	return receipt, nil
}
