package microagent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

const (
	DescriptorSchema   = "fak-micro-context-descriptor/1"
	DescriptorSchemaV2 = "fak-micro-context-descriptor/2"
)

// Descriptor is the process-free contract for one logical agent context. It
// carries only semantic execution state; terminal, cwd, process, credential
// store, and approval UI remain adapter concerns instead of per-context state.
type Descriptor struct {
	Schema            string           `json:"schema"`
	ID                string           `json:"id"`
	BaseID            string           `json:"base_id"`
	TaskDelta         string           `json:"task_delta"`
	Tools             []string         `json:"tools,omitempty"`
	Budget            DescriptorBudget `json:"budget"`
	Continuation      []agent.Message  `json:"continuation,omitempty"`
	OutputContract    OutputContract   `json:"output_contract"`
	ContinuationToken string           `json:"continuation_token,omitempty"`
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
	if d.Schema != DescriptorSchema && d.Schema != DescriptorSchemaV2 {
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
	if d.Budget.MaxOutputTokens <= 0 {
		return errors.New("microagent: descriptor requires positive max_output_tokens")
	}
	if d.Schema == DescriptorSchema && d.Budget.MaxTurns != 1 {
		return errors.New("microagent: descriptor v1 requires max_turns=1")
	}
	if d.Schema == DescriptorSchemaV2 && d.Budget.MaxTurns <= 0 {
		return errors.New("microagent: descriptor v2 requires positive max_turns")
	}
	if d.Schema == DescriptorSchemaV2 && strings.TrimSpace(d.ContinuationToken) == "" {
		return errors.New("microagent: descriptor v2 requires continuation_token")
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
	TurnsUsed  int
	History    []agent.Message
}

func (a *DescriptorAgent) Step(ctx context.Context, gw Gateway) (bool, error) {
	if err := a.Descriptor.Validate(); err != nil {
		return false, err
	}
	if a.TurnsUsed >= a.Descriptor.Budget.MaxTurns {
		return false, errors.New("microagent: descriptor turn budget exhausted")
	}
	msgs := make([]agent.Message, 0, len(a.Base)+len(a.Descriptor.Continuation)+len(a.History)+1)
	msgs = append(msgs, a.Base...)
	msgs = append(msgs, a.Descriptor.Continuation...)
	msgs = append(msgs, agent.Message{Role: agent.RoleUser, Content: a.Descriptor.TaskDelta})
	msgs = append(msgs, a.History...)
	tools := make([]agent.ToolDef, 0, len(a.Descriptor.Tools))
	for _, name := range a.Descriptor.Tools {
		tools = append(tools, agent.ToolDef{Type: "function", Function: agent.ToolDefFunction{Name: name, Parameters: json.RawMessage(`{"type":"object"}`)}})
	}
	opts := []agent.SampleOpt{}
	if a.Descriptor.Budget.MaxOutputTokens > 0 {
		opts = append(opts, agent.WithMaxTokens(a.Descriptor.Budget.MaxOutputTokens))
	}
	trace := a.Descriptor.ID
	if a.Descriptor.ContinuationToken != "" {
		trace = a.Descriptor.ContinuationToken
	}
	completion, err := gw.Complete(WithTrace(ctx, trace), msgs, tools, opts...)
	if err != nil {
		return false, err
	}
	a.TurnsUsed++
	a.Result = completion.Message.Content
	a.History = append(a.History, completion.Message)
	if a.Descriptor.OutputContract.Match(a.Result) {
		return true, nil
	}
	if a.TurnsUsed >= a.Descriptor.Budget.MaxTurns {
		return false, fmt.Errorf("microagent: output contract refused %q at turn budget %d", a.Result, a.Descriptor.Budget.MaxTurns)
	}
	return false, nil
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

type descriptorFrozenState struct {
	Schema            string          `json:"schema"`
	ID                string          `json:"id"`
	ContinuationToken string          `json:"continuation_token"`
	TurnsUsed         int             `json:"turns_used"`
	Result            string          `json:"result"`
	History           []agent.Message `json:"history"`
}

const descriptorFrozenSchema = "fak-micro-context-descriptor-state/1"

// Freeze captures only mutable continuation state. The immutable base and the
// validated descriptor remain outside the parked payload.
func (a *DescriptorAgent) Freeze() ([]byte, error) {
	if err := a.Descriptor.Validate(); err != nil {
		return nil, err
	}
	return json.Marshal(descriptorFrozenState{Schema: descriptorFrozenSchema, ID: a.Descriptor.ID, ContinuationToken: a.Descriptor.ContinuationToken, TurnsUsed: a.TurnsUsed, Result: a.Result, History: a.History})
}

// Thaw restores a descriptor's exact between-turn continuation state and
// refuses state belonging to another logical context.
func (a *DescriptorAgent) Thaw(b []byte) error {
	var st descriptorFrozenState
	if err := json.Unmarshal(b, &st); err != nil {
		return fmt.Errorf("microagent: thaw descriptor: %w", err)
	}
	if st.Schema != descriptorFrozenSchema || st.ID != a.Descriptor.ID || st.ContinuationToken != a.Descriptor.ContinuationToken {
		return errors.New("microagent: descriptor frozen-state identity mismatch")
	}
	if st.TurnsUsed < 0 || st.TurnsUsed > a.Descriptor.Budget.MaxTurns {
		return errors.New("microagent: descriptor frozen-state turn count outside budget")
	}
	a.TurnsUsed, a.Result = st.TurnsUsed, st.Result
	a.History = append(a.History[:0], st.History...)
	return nil
}
