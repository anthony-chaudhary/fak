package agent

// inject.go adds an ERROR-INJECTION COMPOSITION HARNESS over the existing live
// A/B loop (loop.go). The turn-tax benchmark MODELS the baseline as "+1 turn per
// recoverable tool error"; this harness exists to MEASURE whether a real model
// actually fires that retry turn.
//
// The mechanism is a Planner DECORATOR: it wraps the inner planner (the live
// HTTPPlanner or the offline MockPlanner) and, with a seeded probability,
// CORRUPTS the tool-call arguments the inner planner returns BEFORE the loop
// executes them — renaming a canonical arg to a known-bad grammar alias
// (from_currency -> from) or dropping a required field. The corruption is the
// SAME on both arms (the loop runs the same decorated planner twice), so the
// arms differ ONLY by the kernel:
//
//   - fak arm:      the grammar rung repairs the alias in-syscall -> NO tool
//                   error, NO retry turn (Repairs++).
//   - baseline arm: execNaive passes the malformed args to the tool, which
//                   rejects them -> a tool ERROR (ToolErrors++). A REAL model
//                   then sees the error and must spend +1 turn to retry. THAT
//                   delta is the measurement.
//
// IMPORTANT (and why the live run matters): the offline MockPlanner is SCRIPTED.
// It does NOT adaptively retry on an arbitrary injected error — it only retries
// the one convert_currency malform its own script anticipates (s.convertErrored).
// So the OFFLINE test (inject_test.go) validates PLUMBING ONLY: that the
// decorator corrupts args and that the kernel/grammar path behaves (fak repairs,
// baseline errors). The empirical "+1 retry turn per recoverable error" can only
// be WITNESSED with a LIVE model that genuinely re-plans after seeing the error.

import (
	"context"
	"encoding/json"
	"hash/fnv"
	"math/rand"
	"sort"
)

// InjectKind names the corruption applied to a tool call's args.
type InjectKind string

const (
	// InjectNone leaves the call untouched.
	InjectNone InjectKind = "none"
	// InjectAlias renames a canonical arg to a known grammar alias
	// (from_currency -> from). The fak grammar rung repairs it; the baseline
	// tool rejects it.
	InjectAlias InjectKind = "alias"
	// InjectDrop deletes a required field. Neither arm can repair a true
	// missing field, so this is a HARD error on BOTH arms — used to confirm the
	// loop counts a tool error the model must recover from. (Default off; the
	// alias path is the one that isolates the kernel delta.)
	InjectDrop InjectKind = "drop"
)

// aliasFor maps a canonical arg name to the grammar alias the kernel repairs.
// These mirror grammar aliases registered in tools.go's Configure() for
// convert_currency, so an aliased call is repairable on the fak arm and only the
// fak arm.
var aliasFor = map[string]string{
	"from_currency": "from",
	"to_currency":   "to",
}

// InjectingPlanner is an error-injecting Planner DECORATOR. It forwards each turn
// to Inner, then with probability Prob corrupts the args of the FIRST corruptible
// tool call in the response. Corruption is deterministic under a fixed Seed.
type InjectingPlanner struct {
	Inner Planner // the wrapped planner (live HTTPPlanner or offline MockPlanner)
	Prob  float64 // probability in [0,1] of corrupting a corruptible call this turn
	Seed  int64   // seed for the deterministic per-call RNG
	Kind  InjectKind

	// Target restricts injection to a specific tool (empty = convert_currency,
	// the alias-prone tool the grammar rung covers). We deliberately default to
	// convert_currency because it is the ONLY tool whose alias the kernel can
	// repair, so it is the only corruption that isolates the fak/baseline delta.
	Target string

	// Once, when true, injects at most ONE corruption per arm-run: it corrupts the
	// first matching call, then lets every later call (the model's RETRY) through
	// untouched. This is what isolates a CLEAN "+1 retry turn per error"
	// measurement — if we re-corrupt every retry (Once=false, Prob=1) the baseline
	// can never recover and instead spins to the turn cap, which is a derailment,
	// not a measurable single retry. Defaults to true via NewInjectingPlanner.
	Once bool

	// done is the per-arm latch for Once. Run() reuses the SAME decorator across
	// both arms, so the latch must reset at the start of each arm — RunArm has no
	// hook for that, so instead we detect an arm boundary by the message list
	// shrinking back to the 2-message seed (system+user). See Complete.
	done bool

	// Injected counts the corruptions actually applied (the denominator for the
	// empirical retry-turns-per-injected-error). It is shared by reference so the
	// runner can read it after a run.
	Injected *int
}

// NewInjectingPlanner wraps inner with deterministic alias-corruption of every
// convert_currency call (Prob=1.0). A counter pointer is allocated so the runner
// can read how many injections fired.
func NewInjectingPlanner(inner Planner, seed int64) *InjectingPlanner {
	n := 0
	return &InjectingPlanner{
		Inner:    inner,
		Prob:     1.0,
		Seed:     seed,
		Kind:     InjectAlias,
		Target:   toolConvert,
		Once:     true, // one injected error per arm — let the model's retry recover
		Injected: &n,
	}
}

func (p *InjectingPlanner) Model() string { return p.Inner.Model() + "+inject" }

// target returns the configured target tool, defaulting to convert_currency.
func (p *InjectingPlanner) target() string {
	if p.Target == "" {
		return toolConvert
	}
	return p.Target
}

// Complete forwards to the inner planner, then corrupts the args of the first
// matching tool call with the configured probability. The decision is keyed off a
// per-turn hash of (seed, turn-index, tool, raw-args) so it is reproducible: the
// SAME inner response yields the SAME corruption on BOTH arms, which is exactly
// what makes the fak/baseline delta the kernel's doing and nothing else.
func (p *InjectingPlanner) Complete(ctx context.Context, messages []Message, tools []ToolDef, opts ...SampleOpt) (*Completion, error) {
	comp, err := p.Inner.Complete(ctx, messages, tools, opts...)
	if err != nil {
		return nil, err
	}
	if comp == nil || len(comp.Message.ToolCalls) == 0 || p.Kind == InjectNone {
		return comp, nil
	}
	target := p.target()
	// Arm-boundary detection: Run() reuses this decorator across both arms, and a
	// fresh arm starts with exactly the 2-message seed (system + user). Seeing that
	// resets the Once latch so each arm gets its own single injection.
	if len(messages) <= 2 {
		p.done = false
	}
	if p.Once && p.done {
		return comp, nil // already injected this arm; let the retry through
	}
	// Count assistant turns so far for a stable per-turn seed component (the inner
	// planner has not yet appended this response to messages).
	turnIdx := 0
	for _, m := range messages {
		if m.Role == RoleAssistant {
			turnIdx++
		}
	}
	for i := range comp.Message.ToolCalls {
		tc := &comp.Message.ToolCalls[i]
		if tc.Function.Name != target {
			continue
		}
		if !p.fire(turnIdx, tc.Function.Name, tc.Function.Arguments) {
			continue
		}
		corrupted, ok := corruptArgs(tc.Function.Arguments, p.Kind)
		if ok {
			tc.Function.Arguments = corrupted
			if p.Injected != nil {
				*p.Injected++
			}
			p.done = true // latch for Once: the model's later retry is left clean
		}
		break // corrupt at most one call per turn — one injected error per turn
	}
	return comp, nil
}

// fire decides whether to inject for this call, deterministically. A hash of the
// seed and the call's identity drives a seeded RNG so the same seed + same inner
// behaviour always make the same choice (across both arms and across reruns).
func (p *InjectingPlanner) fire(turnIdx int, tool, rawArgs string) bool {
	if p.Prob >= 1.0 {
		return true
	}
	if p.Prob <= 0.0 {
		return false
	}
	h := fnv.New64a()
	var seedBuf [8]byte
	for i := 0; i < 8; i++ {
		seedBuf[i] = byte(p.Seed >> (8 * i))
	}
	_, _ = h.Write(seedBuf[:])
	_, _ = h.Write([]byte{byte(turnIdx)})
	_, _ = h.Write([]byte(tool))
	_, _ = h.Write([]byte(rawArgs))
	r := rand.New(rand.NewSource(int64(h.Sum64()))) //nolint:gosec // deterministic test harness, not crypto
	return r.Float64() < p.Prob
}

// corruptArgs applies the corruption to a raw JSON args string and returns the new
// raw string + whether any corruption was actually applied. Keys are emitted in
// sorted order so the output is deterministic regardless of Go's map iteration.
func corruptArgs(rawArgs string, kind InjectKind) (string, bool) {
	var m map[string]any
	if rawArgs == "" {
		rawArgs = "{}"
	}
	if err := json.Unmarshal([]byte(rawArgs), &m); err != nil || m == nil {
		return rawArgs, false
	}
	changed := false
	switch kind {
	case InjectAlias:
		// Rename each canonical key present to its grammar alias.
		for canon, alias := range aliasFor {
			if v, ok := m[canon]; ok {
				delete(m, canon)
				m[alias] = v
				changed = true
			}
		}
	case InjectDrop:
		// Drop the first required canonical field present (deterministic order).
		for _, canon := range []string{"from_currency", "to_currency", "amount"} {
			if _, ok := m[canon]; ok {
				delete(m, canon)
				changed = true
				break
			}
		}
	default:
		return rawArgs, false
	}
	if !changed {
		return rawArgs, false
	}
	return marshalSorted(m), true
}

// marshalSorted marshals a map with keys in sorted order for a stable raw-args
// string (Go's encoding/json already sorts map keys, but we re-marshal explicitly
// so the harness's output is obviously deterministic).
func marshalSorted(m map[string]any) string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	b := []byte{'{'}
	for i, k := range keys {
		if i > 0 {
			b = append(b, ',')
		}
		kb, _ := json.Marshal(k)
		vb, _ := json.Marshal(m[k])
		b = append(b, kb...)
		b = append(b, ':')
		b = append(b, vb...)
	}
	b = append(b, '}')
	return string(b)
}

// ---------------------------------------------------------------------------
// Composition runner — drive the live A/B with the injecting decorator and
// compute the EMPIRICAL retry-turns-per-injected-error.
// ---------------------------------------------------------------------------

// InjectionResult is the harness's measurement for one error-injecting A/B run.
type InjectionResult struct {
	// AB is the full underlying A/B result (per-arm metrics, both_completed, etc.).
	AB *RunResult `json:"ab"`

	// Injected is how many tool-call args the decorator corrupted across BOTH arms'
	// runs combined (the loop runs the decorated planner once per arm).
	Injected int `json:"injected"`

	// FakRepairs is the kernel-measured count of in-syscall grammar repairs on the
	// fak arm — the alias corruptions the kernel absorbed with NO retry turn.
	FakRepairs int `json:"fak_repairs"`

	// BaselineToolErrors is the harness-measured count of tool errors the baseline
	// arm hit — the corruptions that became errors a real model must recover from.
	BaselineToolErrors int `json:"baseline_tool_errors"`

	// RetryTurnsPerError is the EMPIRICAL retry-turns-per-injected-error on the
	// BASELINE arm: (baseline.Turns - fak.Turns) / baseline.ToolErrors, the extra
	// model round-trips the baseline spent per tool error it had to recover from.
	// This is meaningful ONLY if BothCompleted (a derailed arm "saves" turns by
	// failing). It is the number the turn-tax benchmark MODELS as 1.0.
	RetryTurnsPerError float64 `json:"retry_turns_per_error"`

	// RetrySupported is true iff the measurement supports the benchmark's
	// "+1 turn per recoverable error" model: both arms completed AND the baseline
	// spent at least ~1 extra turn per tool error it hit.
	RetrySupported bool `json:"retry_supported"`

	// Live is true if a real network model drove the run.
	Live bool `json:"live"`

	// Note carries the honest read for a degraded / pending live attempt.
	Note string `json:"note,omitempty"`
}

// RunInjection drives BOTH arms of agent.Run over the same task using the supplied
// inner planner wrapped in an InjectingPlanner (deterministic under seed), then
// computes the empirical retry-turns-per-injected-error. The wrapped planner is
// reused across both arms inside Run, so the SAME corruption decisions apply to
// each arm and the only difference is the kernel.
func RunInjection(ctx context.Context, inner Planner, task string, maxTurns int, seed int64) (*InjectionResult, []traceEvent, error) {
	ip := NewInjectingPlanner(inner, seed)
	res, trace, err := Run(ctx, ip, task, maxTurns)
	if err != nil {
		return nil, nil, err
	}
	// Run() detects live by asserting the planner is *HTTPPlanner, but our
	// decorator hides the inner type, so it cannot. Re-derive Live from the inner
	// planner here so a wrapped live run is still flagged live.
	if _, isLive := inner.(*HTTPPlanner); isLive {
		res.Live = true
	}
	out := summarizeInjection(res, *ip.Injected)
	return out, trace, nil
}

// summarizeInjection computes the empirical measurement from an A/B result. Split
// out so a test can exercise it on a synthetic RunResult without a planner.
func summarizeInjection(res *RunResult, injected int) *InjectionResult {
	out := &InjectionResult{
		AB:                 res,
		Injected:           injected,
		FakRepairs:         res.Fak.Repairs,
		BaselineToolErrors: res.Baseline.ToolErrors,
		Live:               res.Live,
	}
	// The retry-turns-per-error is the baseline's EXTRA turns (vs the fak arm,
	// which repaired the same corruptions for free) divided by the baseline tool
	// errors that drove them. Comparable only when both arms did the same work.
	if res.BothCompleted && res.Baseline.ToolErrors > 0 {
		out.RetryTurnsPerError = float64(res.TurnsSaved) / float64(res.Baseline.ToolErrors)
		// The benchmark models +1 turn/error; treat >= ~0.5 as supporting it
		// (a real model needs at least a partial extra round-trip per error).
		out.RetrySupported = out.RetryTurnsPerError >= 0.5
	}
	if !res.BothCompleted {
		out.Note = "turn delta NOT comparable: arms completed different work (a derailed arm 'saves' turns by failing)"
	} else if res.Baseline.ToolErrors == 0 {
		out.Note = "baseline hit 0 injected tool errors — model emitted canonical args or injection did not fire; retry behaviour unwitnessed this run"
	}
	return out
}
