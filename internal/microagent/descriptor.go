package microagent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

const DescriptorSchema = "fak-micro-context-descriptor/1"

// Descriptor is the process-free contract for one logical agent context. It
// carries only semantic execution state; terminal, cwd, process, credential
// store, and approval UI remain adapter concerns instead of per-context state.
type Descriptor struct {
	Schema         string           `json:"schema"`
	ID             string           `json:"id"`
	BaseID         string           `json:"base_id"`
	TaskDelta      string           `json:"task_delta"`
	Tools          []string         `json:"tools,omitempty"`
	Budget         DescriptorBudget `json:"budget"`
	Continuation   []agent.Message  `json:"continuation,omitempty"`
	OutputContract OutputContract   `json:"output_contract"`
}

type DescriptorBudget struct {
	MaxTurns        int `json:"max_turns"`
	MaxOutputTokens int `json:"max_output_tokens"`
}

type OutputContract struct {
	Kind     string `json:"kind"` // exact, contains, or nonempty
	Expected string `json:"expected,omitempty"`
}

func (d Descriptor) Validate() error {
	if d.Schema != DescriptorSchema {
		return fmt.Errorf("microagent: descriptor schema %q", d.Schema)
	}
	if strings.TrimSpace(d.ID) == "" || strings.TrimSpace(d.BaseID) == "" || strings.TrimSpace(d.TaskDelta) == "" {
		return errors.New("microagent: descriptor requires id, base_id, and task_delta")
	}
	switch d.OutputContract.Kind {
	case "exact", "contains", "nonempty":
	default:
		return fmt.Errorf("microagent: unsupported output contract %q", d.OutputContract.Kind)
	}
	if d.Budget.MaxTurns != 1 || d.Budget.MaxOutputTokens <= 0 {
		return errors.New("microagent: descriptor v1 requires max_turns=1 and positive max_output_tokens")
	}
	if d.OutputContract.Kind != "nonempty" && d.OutputContract.Expected == "" {
		return errors.New("microagent: output contract requires expected value")
	}
	seen := map[string]bool{}
	for _, c := range d.Tools {
		if strings.TrimSpace(c) == "" || seen[c] {
			return errors.New("microagent: tools entries must be nonempty and unique")
		}
		seen[c] = true
	}
	return nil
}

// DescriptorAgent adapts Descriptor into the existing Host/Gateway seam. One
// descriptor is one model turn for the initial lightweight spine.
type DescriptorAgent struct {
	Descriptor Descriptor
	Base       []agent.Message
	Result     string
}

func (a *DescriptorAgent) Step(ctx context.Context, gw Gateway) (bool, error) {
	if err := a.Descriptor.Validate(); err != nil {
		return false, err
	}
	msgs := make([]agent.Message, 0, len(a.Base)+len(a.Descriptor.Continuation)+1)
	msgs = append(msgs, a.Base...)
	msgs = append(msgs, a.Descriptor.Continuation...)
	msgs = append(msgs, agent.Message{Role: agent.RoleUser, Content: a.Descriptor.TaskDelta})
	tools := make([]agent.ToolDef, 0, len(a.Descriptor.Tools))
	for _, name := range a.Descriptor.Tools {
		tools = append(tools, agent.ToolDef{Type: "function", Function: agent.ToolDefFunction{Name: name, Parameters: json.RawMessage(`{"type":"object"}`)}})
	}
	opts := []agent.SampleOpt{}
	if a.Descriptor.Budget.MaxOutputTokens > 0 {
		opts = append(opts, agent.WithMaxTokens(a.Descriptor.Budget.MaxOutputTokens))
	}
	completion, err := gw.Complete(WithTrace(ctx, a.Descriptor.ID), msgs, tools, opts...)
	if err != nil {
		return false, err
	}
	a.Result = completion.Message.Content
	if !a.Descriptor.OutputContract.Match(a.Result) {
		return false, fmt.Errorf("microagent: output contract refused %q", a.Result)
	}
	return true, nil
}

func (c OutputContract) Match(got string) bool {
	switch c.Kind {
	case "exact":
		return got == c.Expected
	case "contains":
		return strings.Contains(got, c.Expected)
	case "nonempty":
		return strings.TrimSpace(got) != ""
	}
	return false
}

// SpawnDescriptor wires the descriptor's budget into the Host's existing
// session table before spawning; it does not create a second scheduler.
func SpawnDescriptor(h *Host, d Descriptor, base []agent.Message) (*DescriptorAgent, error) {
	if h == nil {
		return nil, errors.New("microagent: nil host")
	}
	if err := d.Validate(); err != nil {
		return nil, err
	}
	a := &DescriptorAgent{Descriptor: d, Base: append([]agent.Message(nil), base...)}
	if err := h.Spawn(d.ID, a); err != nil {
		return nil, err
	}
	return a, nil
}

// DescriptorSize is the stable serialized bytes carried per logical context.
func DescriptorSize(d Descriptor) (int, error) { b, err := json.Marshal(d); return len(b), err }
