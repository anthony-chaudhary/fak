// Package plancfi implements control-flow integrity for agent plans, refusing
// or escalating tool calls that deviate from approved execution plans.
// An unplanned tool call outside the declared plan triggers deviation handling.
package plancfi

import (
	"context"
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// VerdictRequireApproval indicates an unconfirmed action requiring human escalation.
const VerdictRequireApproval abi.VerdictKind = 1024

const requireApprovalFoldRank = 50

// Mode specifies the enforcement strategy for an approved plan.
type Mode uint8

const (
	// AllowedSet requires every tool call to belong to the approved set without ordering constraints.
	AllowedSet Mode = iota
	// Sequence requires tool calls to follow the defined plan order.
	Sequence
)

// Plan configures the approved tool set and enforcement mode for a trace.
type Plan struct {
	Tools []string
	Mode  Mode
}

func (p Plan) has(tool string) bool {
	for _, t := range p.Tools {
		if t == tool {
			return true
		}
	}
	return false
}

// Ledger tracks approved plans and execution progress per trace.
type Ledger struct {
	mu    sync.RWMutex
	plans map[string]*state
}

type state struct {
	plan Plan
	pos  int
}

// NewLedger constructs an empty plan ledger without active trace plans.
func NewLedger() *Ledger { return &Ledger{plans: map[string]*state{}} }

// Declare registers an approved plan for the specified trace.
func (l *Ledger) Declare(trace string, p Plan) {
	l.mu.Lock()
	l.plans[trace] = &state{plan: p}
	l.mu.Unlock()
}

// Clear removes the active plan associated with the trace.
func (l *Ledger) Clear(trace string) {
	l.mu.Lock()
	delete(l.plans, trace)
	l.mu.Unlock()
}

// Declared reports whether an approved plan is active for the trace.
func (l *Ledger) Declared(trace string) bool {
	l.mu.RLock()
	defer l.mu.RUnlock()
	_, ok := l.plans[trace]
	return ok
}

// conforms reports whether the tool call is permitted under the active trace plan.
func (l *Ledger) conforms(trace, tool string) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	st, ok := l.plans[trace]
	if !ok {
		return true
	}
	if st.plan.Mode == AllowedSet {
		return st.plan.has(tool)
	}
	for i := 0; i <= st.pos+1 && i < len(st.plan.Tools); i++ {
		if st.plan.Tools[i] == tool {
			if i > st.pos {
				st.pos = i
			}
			return true
		}
	}
	return false
}

// Default provides the process-wide plan ledger instance.
var Default = NewLedger()

// Adjudicator evaluates tool calls against approved plans in the ledger.
type Adjudicator struct {
	ledger      *Ledger
	OnDeviation abi.VerdictKind
}

// New constructs a plan Adjudicator backed by the given ledger.
func New(l *Ledger) *Adjudicator {
	return &Adjudicator{ledger: l, OnDeviation: VerdictRequireApproval}
}

// Caps reports required capabilities for the plan adjudicator.
func (a *Adjudicator) Caps() []abi.Capability { return nil }

// Adjudicate evaluates whether the tool call conforms to the approved plan for its trace.
func (a *Adjudicator) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	if c == nil || !a.ledger.Declared(c.TraceID) {
		return abi.Verdict{Kind: abi.VerdictDefer, By: "plancfi"}
	}
	if a.ledger.conforms(c.TraceID, c.Tool) {
		return abi.Verdict{Kind: abi.VerdictDefer, By: "plancfi"}
	}
	return abi.Verdict{
		Kind:        a.OnDeviation,
		Reason:      abi.ReasonPolicyBlock,
		Disposition: "ESCALATE",
		By:          "plancfi",
		Payload:     abi.WitnessPayload{Claim: fmt.Sprintf("call %q deviates from the approved plan", c.Tool)},
		Meta:        map[string]string{"plancfi": "deviation", "tool": c.Tool, "disposition": "ESCALATE"},
	}
}

// DefaultAdjudicator provides the registered singleton adjudicator instance.
var DefaultAdjudicator = New(Default)

func init() {
	abi.RegisterVerdictKind(VerdictRequireApproval, "RequireApproval", requireApprovalFoldRank, abi.FallbackDeny)
	abi.RegisterAdjudicator(25, DefaultAdjudicator)
	abi.RegisterCapability("plancfi.v1")
}
