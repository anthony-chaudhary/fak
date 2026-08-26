package generationctl

import (
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/streamrules"
)

// DirectiveKind is the provider-neutral action at a steering point.
type DirectiveKind string

const (
	Continue DirectiveKind = "continue"
	Redirect DirectiveKind = "redirect"
	Fork     DirectiveKind = "fork"
	Yield    DirectiveKind = "yield"
	Stop     DirectiveKind = "stop"
)

// Compute names the replaceable execution placement for an epoch.
type Compute struct {
	Worker string `json:"worker"`
	Model  string `json:"model"`
	Device string `json:"device,omitempty"`
}

// Epoch identifies one contiguous generation span. Owner is deliberately
// separate from Compute: a micro-agent can retain responsibility while its
// decoding placement changes, or hand responsibility to another micro-agent.
type Epoch struct {
	TrajectoryID string  `json:"trajectory_id"`
	Number       uint64  `json:"number"`
	Owner        string  `json:"owner"`
	Compute      Compute `json:"compute"`
}

// Checkpoint is the committed trajectory prefix at an epoch boundary. Partial
// tool arguments are intentionally absent: they are speculative until admitted.
type Checkpoint struct {
	TrajectoryID string `json:"trajectory_id"`
	AfterEpoch   uint64 `json:"after_epoch"`
	Accepted     string `json:"accepted"`
}

// Directive requests a transition at the next safe steering point.
type Directive struct {
	Kind   DirectiveKind `json:"kind"`
	Reason string        `json:"reason,omitempty"`
	Action string        `json:"action,omitempty"`
}

// Transition records an epoch boundary. For Continue, Checkpoint is nil.
type Transition struct {
	Directive  Directive   `json:"directive"`
	Checkpoint *Checkpoint `json:"checkpoint,omitempty"`
	Rule       string      `json:"rule,omitempty"`
}

// Controller owns the minimal provider-neutral continuous-generation state.
// It is not a decoder: adapters feed accepted output and speculative tool-call
// argument deltas into it.
type Controller struct {
	trajectory string
	epoch      Epoch
	accepted   strings.Builder
	matcher    *streamrules.Matcher
	closed     bool
}

func New(trajectoryID, owner string, compute Compute, rules []streamrules.Rule) (*Controller, error) {
	if strings.TrimSpace(trajectoryID) == "" {
		return nil, errors.New("trajectory id is required")
	}
	if strings.TrimSpace(owner) == "" {
		return nil, errors.New("epoch owner is required")
	}
	matcher, diagnostics := streamrules.Compile(rules)
	if len(diagnostics) != 0 {
		return nil, fmt.Errorf("invalid stream rules: %v", diagnostics)
	}
	return &Controller{
		trajectory: trajectoryID,
		epoch:      Epoch{TrajectoryID: trajectoryID, Number: 1, Owner: owner, Compute: compute},
		matcher:    matcher,
	}, nil
}

func (c *Controller) Epoch() Epoch { return c.epoch }

// Accept commits provider output to the durable trajectory prefix. Adapters
// must not call Accept for partial tool arguments before tool admission.
func (c *Controller) Accept(text string) error {
	if c.closed {
		return errors.New("generation epoch is closed")
	}
	c.accepted.WriteString(text)
	return nil
}

// ObserveText commits an output delta to the durable prefix and checks it
// against text-scope rules in one step. The order is load-bearing: the bytes
// join the prefix BEFORE the check, so a checkpoint taken by a rule that
// matched them still contains the text the provider already emitted. A redirect
// steers what happens next; it does not retroactively unsay the turn.
func (c *Controller) ObserveText(key streamrules.StreamKey, delta string) (Transition, error) {
	if err := c.Accept(delta); err != nil {
		return Transition{}, err
	}
	return c.observe(key, delta)
}

// ObserveThinking checks a reasoning delta against thinking-scope rules. Unlike
// ObserveText it never accepts: reasoning is steerable but is not part of the
// durable trajectory prefix, so a checkpoint taken here carries the output
// committed before it and nothing from the reasoning channel.
func (c *Controller) ObserveThinking(key streamrules.StreamKey, delta string) (Transition, error) {
	return c.observeSpeculative(key, delta)
}

// ObserveToolDelta checks speculative tool arguments as they stream. A matched
// substitution closes the epoch before those arguments become an effect. The
// bytes are never accepted, which is what keeps a refused call's arguments out
// of the prefix the next epoch resumes from.
func (c *Controller) ObserveToolDelta(key streamrules.StreamKey, delta string) (Transition, error) {
	return c.observeSpeculative(key, delta)
}

// observeSpeculative is the shared gate for stream scopes whose bytes must never join
// the durable accepted prefix. Both reasoning and partial tool arguments stop at the
// same closed-epoch boundary before the matcher sees another delta.
func (c *Controller) observeSpeculative(key streamrules.StreamKey, delta string) (Transition, error) {
	if c.closed {
		return Transition{}, errors.New("generation epoch is closed")
	}
	return c.observe(key, delta)
}

// observe is the one steering step every stream scope shares: feed the delta to
// the matcher, and on the first interrupting match close the epoch at a
// checkpoint carrying the substitute action. What differs per scope is only
// whether the caller accepted the bytes first, not how a match is handled.
func (c *Controller) observe(key streamrules.StreamKey, delta string) (Transition, error) {
	var match streamrules.Match
	for _, candidate := range c.matcher.CheckDelta(key, delta) {
		if candidate.Interrupt {
			match = candidate
			break
		}
	}
	if !match.Interrupt {
		return Transition{Directive: Directive{Kind: Continue}}, nil
	}
	if strings.TrimSpace(match.SubstituteAction) == "" {
		return Transition{}, fmt.Errorf("rule %q interrupted without a substitute action", match.Rule)
	}
	c.closed = true
	cp := c.checkpoint()
	return Transition{
		Directive:  Directive{Kind: Redirect, Reason: "stream-rule:" + match.Rule, Action: match.SubstituteAction},
		Checkpoint: &cp,
		Rule:       match.Rule,
	}, nil
}

// Steer closes the current epoch for an external live directive. Continue is a
// no-op; redirect, fork, yield, and stop all create a durable steering point.
func (c *Controller) Steer(d Directive) (Transition, error) {
	if c.closed {
		return Transition{}, errors.New("generation epoch is closed")
	}
	switch d.Kind {
	case Continue:
		return Transition{Directive: d}, nil
	case Redirect, Fork, Yield, Stop:
		c.closed = true
		cp := c.checkpoint()
		return Transition{Directive: d, Checkpoint: &cp}, nil
	default:
		return Transition{}, fmt.Errorf("unknown directive kind %q", d.Kind)
	}
}

// Resume opens the next epoch from a checkpoint, potentially under a different
// micro-agent and on different compute. The trajectory identity and accepted
// prefix are invariant across the handoff.
func Resume(cp Checkpoint, owner string, compute Compute, rules []streamrules.Rule) (*Controller, error) {
	if cp.TrajectoryID == "" || cp.AfterEpoch == 0 {
		return nil, errors.New("valid checkpoint is required")
	}
	c, err := New(cp.TrajectoryID, owner, compute, rules)
	if err != nil {
		return nil, err
	}
	c.epoch.Number = cp.AfterEpoch + 1
	c.accepted.WriteString(cp.Accepted)
	return c, nil
}

func (c *Controller) Checkpoint() Checkpoint { return c.checkpoint() }

func (c *Controller) checkpoint() Checkpoint {
	return Checkpoint{TrajectoryID: c.trajectory, AfterEpoch: c.epoch.Number, Accepted: c.accepted.String()}
}
