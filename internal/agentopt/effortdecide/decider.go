// Package effortdecide lets a decision model choose the reasoning effort for
// each agent turn by reading its preference over declared options (the
// Jev/SemIf pattern: score declared labels, softmax over only those labels),
// instead of the regex heuristic in agentopt.IntraModelEffortRouter.
//
// It is opt-in: nothing constructs a Decider unless FromEnv is configured or a
// caller builds one. Every failure path returns the heuristic classification,
// so enabling it can change effort choices but never block a turn.
package effortdecide

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/agentopt"
)

// Reader scores declared answer labels for a prompt. Scores are
// log-probabilities normalized over the model's full vocabulary, so their
// exp-sum is the mass the model put on any declared answer; labels the model
// did not score are omitted.
type Reader interface {
	ReadOptions(ctx context.Context, prompt string, labels []string) (map[string]float64, error)
}

// ReaderFunc adapts a function to Reader.
type ReaderFunc func(ctx context.Context, prompt string, labels []string) (map[string]float64, error)

// ReadOptions calls f.
func (f ReaderFunc) ReadOptions(ctx context.Context, prompt string, labels []string) (map[string]float64, error) {
	return f(ctx, prompt, labels)
}

// Single-letter labels keep each option one token on common tokenizers.
var options = []struct {
	label string
	tier  agentopt.EffortTier
	desc  string
}{
	{"A", agentopt.EffortNone, "none: mechanical step, the next action is obvious"},
	{"B", agentopt.EffortLow, "low: a quick check or a simple verdict"},
	{"C", agentopt.EffortMedium, "medium: multi-step diagnosis, diff review or synthesis"},
	{"D", agentopt.EffortHigh, "high: planning, decomposition, replanning or recovering from a failure"},
}

// Labels returns the declared answer labels in option order.
func Labels() []string {
	out := make([]string, len(options))
	for i, o := range options {
		out[i] = o.label
	}
	return out
}

const (
	// SourceModel marks an effort chosen by the decision model.
	SourceModel = "decision-model"
	// SourceHeuristic marks an effort taken from the heuristic router.
	SourceHeuristic = "heuristic"

	defaultMinConfidence   = 0.5
	defaultMinDeclaredMass = 0.5
	defaultTimeout         = 750 * time.Millisecond
	maxFieldBytes          = 1500
)

// Config configures a Decider. Reader is required.
type Config struct {
	Reader Reader
	// Fallback supplies the operational category, budgets and the fallback
	// decision. Nil uses a fresh heuristic router.
	Fallback *agentopt.IntraModelEffortRouter
	// MinConfidence is the top-option probability below which the heuristic
	// wins. Zero means 0.5.
	MinConfidence float64
	// MinDeclaredMass is the total probability the model must put on the
	// declared labels before its choice counts. Without it, stray letters in
	// the tail of a distribution dominated by "<think>" or "**" renormalize
	// into a confident-looking answer. Zero means 0.5.
	MinDeclaredMass float64
	// Timeout bounds one read. Zero means 750ms; the read sits on the turn's
	// critical path, so a slow decider must cost less than the thinking it saves.
	Timeout time.Duration
	// AllowBelowHeuristicOnError lets the model pick less effort than the
	// heuristic on error-recovery turns. Off by default: under-thinking a
	// failure is the expensive mistake.
	AllowBelowHeuristicOnError bool
}

// Decider picks a per-turn effort tier with a decision model.
type Decider struct {
	cfg Config
}

// New returns a Decider, or an error when cfg has no Reader.
func New(cfg Config) (*Decider, error) {
	if cfg.Reader == nil {
		return nil, errors.New("effortdecide: Reader is required")
	}
	if cfg.Fallback == nil {
		cfg.Fallback = agentopt.NewIntraModelEffortRouter()
	}
	if cfg.MinConfidence <= 0 {
		cfg.MinConfidence = defaultMinConfidence
	}
	if cfg.MinDeclaredMass <= 0 {
		cfg.MinDeclaredMass = defaultMinDeclaredMass
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = defaultTimeout
	}
	return &Decider{cfg: cfg}, nil
}

// Decision is a turn classification plus how it was reached.
type Decision struct {
	agentopt.TurnClassification
	Source         string                          `json:"source"`
	Confidence     float64                         `json:"confidence,omitempty"`
	DeclaredMass   float64                         `json:"declared_mass,omitempty"`
	Probabilities  map[agentopt.EffortTier]float64 `json:"probabilities,omitempty"`
	HeuristicTier  agentopt.EffortTier             `json:"heuristic_tier"`
	FallbackReason string                          `json:"fallback_reason,omitempty"`
}

// Decide classifies one turn. It always returns a usable decision.
func (d *Decider) Decide(ctx context.Context, turn agentopt.TurnContext) Decision {
	heur := d.cfg.Fallback.Classify(turn)
	fallback := func(reason string) Decision {
		return Decision{TurnClassification: heur, Source: SourceHeuristic, HeuristicTier: heur.Effort, FallbackReason: reason}
	}

	scores, err := d.read(ctx, BuildPrompt(turn))
	if err != nil {
		switch {
		case ctx.Err() != nil:
			return fallback("canceled")
		case errors.Is(err, context.DeadlineExceeded):
			return fallback("timeout")
		}
		return fallback("read_error: " + err.Error())
	}
	probs, mass, err := declaredSoftmax(scores)
	if err != nil {
		return fallback(err.Error())
	}
	if mass < d.cfg.MinDeclaredMass {
		dec := fallback(fmt.Sprintf("low_declared_mass %.2g < %.2g", mass, d.cfg.MinDeclaredMass))
		dec.Probabilities, dec.DeclaredMass = probs, mass
		return dec
	}

	best, conf := agentopt.EffortTier(""), -1.0
	for _, o := range options {
		if p, ok := probs[o.tier]; ok && p > conf {
			best, conf = o.tier, p
		}
	}
	if conf < d.cfg.MinConfidence {
		dec := fallback(fmt.Sprintf("low_confidence %.2f < %.2f", conf, d.cfg.MinConfidence))
		dec.Probabilities, dec.Confidence, dec.DeclaredMass = probs, conf, mass
		return dec
	}

	tier, reason := best, fmt.Sprintf("decision model chose %s (p=%.2f)", best, conf)
	if heur.Category == agentopt.CategoryErrorRecovery && !d.cfg.AllowBelowHeuristicOnError && rank(tier) < rank(heur.Effort) {
		tier = heur.Effort
		reason += fmt.Sprintf("; raised to heuristic %s on error recovery", heur.Effort)
	}
	budget := d.cfg.Fallback.ThinkingBudget(tier)
	return Decision{
		TurnClassification: agentopt.TurnClassification{
			Effort:          tier,
			Category:        heur.Category,
			ThinkingBudget:  budget,
			AllocatedBudget: budget,
			Reason:          reason,
		},
		Source:        SourceModel,
		Confidence:    conf,
		DeclaredMass:  mass,
		Probabilities: probs,
		HeuristicTier: heur.Effort,
	}
}

// read bounds the Reader by Timeout even when it ignores ctx; such a reader's
// goroutine outlives the turn, but the turn does not wait for it.
func (d *Decider) read(ctx context.Context, prompt string) (map[string]float64, error) {
	rctx, cancel := context.WithTimeout(ctx, d.cfg.Timeout)
	defer cancel()
	type result struct {
		scores map[string]float64
		err    error
	}
	ch := make(chan result, 1)
	go func() {
		s, err := d.cfg.Reader.ReadOptions(rctx, prompt, Labels())
		ch <- result{s, err}
	}()
	select {
	case r := <-ch:
		return r.scores, r.err
	case <-rctx.Done():
		return nil, rctx.Err()
	}
}

func rank(t agentopt.EffortTier) int {
	for i, o := range options {
		if o.tier == t {
			return i
		}
	}
	return -1
}

// declaredSoftmax normalizes over declared labels only and also returns their
// raw probability mass. Fewer than two finite scores cannot express a
// preference, so it fails closed.
func declaredSoftmax(scores map[string]float64) (map[agentopt.EffortTier]float64, float64, error) {
	type kv struct {
		tier agentopt.EffortTier
		s    float64
	}
	var vals []kv
	for _, o := range options {
		if s, ok := scores[o.label]; ok && !math.IsNaN(s) && !math.IsInf(s, 0) {
			vals = append(vals, kv{o.tier, s})
		}
	}
	if len(vals) < 2 {
		return nil, 0, fmt.Errorf("insufficient_options: %d finite of %d declared", len(vals), len(options))
	}
	maxS := math.Inf(-1)
	for _, v := range vals {
		maxS = math.Max(maxS, v.s)
	}
	var sum, mass float64
	for _, v := range vals {
		sum += math.Exp(v.s - maxS)
		mass += math.Exp(v.s)
	}
	out := make(map[agentopt.EffortTier]float64, len(vals))
	for _, v := range vals {
		out[v.tier] = math.Exp(v.s-maxS) / sum
	}
	return out, mass, nil
}

// BuildPrompt renders the declared-option question for one turn. Fields are
// truncated so the read stays cheap relative to the turn it prices.
func BuildPrompt(turn agentopt.TurnContext) string {
	var b strings.Builder
	b.WriteString("You are choosing how much reasoning an AI coding agent should spend on its next step.\n\n")
	b.WriteString("Current turn:\n")
	tool := turn.ToolName
	if tool == "" {
		tool = turn.TargetToolName
	}
	field := func(name, v string) {
		v = strings.TrimSpace(v)
		if v == "" {
			return
		}
		if len(v) > maxFieldBytes {
			cut := maxFieldBytes
			for cut > 0 && !utf8.RuneStart(v[cut]) {
				cut--
			}
			v = v[:cut] + " ...[truncated]"
		}
		fmt.Fprintf(&b, "- %s: %s\n", name, v)
	}
	field("last message role", turn.LastRole)
	field("tool", tool)
	if len(turn.ToolCalls) > 0 {
		names := make([]string, 0, len(turn.ToolCalls))
		for _, tc := range turn.ToolCalls {
			names = append(names, tc.Name)
		}
		field("planned tool calls", strings.Join(names, ", "))
	}
	field("task prompt", turn.Prompt)
	field("tool output", turn.ToolOutput)
	field("tool error", turn.ToolError)
	field("error", turn.ErrorMessage)
	var flags []string
	for name, on := range map[string]bool{
		"initial_prompt": turn.IsInitial, "planning": turn.IsPlanning, "replan": turn.IsReplan,
		"test_failure": turn.TestFailure, "compiler_error": turn.CompilerError,
		"panic": turn.ExecutionPanic, "policy_refusal": turn.PolicyRefusal,
	} {
		if on {
			flags = append(flags, name)
		}
	}
	sort.Strings(flags)
	field("signals", strings.Join(flags, ", "))
	b.WriteString("\nHow much reasoning should the next step get?\n")
	for _, o := range options {
		fmt.Fprintf(&b, "%s) %s\n", o.label, o.desc)
	}
	b.WriteString("Answer with one letter.")
	return b.String()
}
