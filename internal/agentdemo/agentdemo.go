// Package agentdemo is the shared spine for fak's agentic "try-it" demos: a
// deterministic, no-key, tool-using agent loop that drives the REAL kernel one
// call at a time.
//
// The existing on-box demos (cmd/guarddemo, cmd/turntaxdemo) replay a frozen,
// class-labeled tool-call TRACE through the kernel. That is the right shape for a
// security/efficiency proof, but an *agentic* demo wants the live loop: a prompt,
// a planner that decides which tools to call, the kernel adjudicating each call,
// and the agent answering from the results it was allowed to get. This package is
// that loop, factored out so a new agentic demo is a toolset + a planner + a
// scenario (~30 lines) instead of a fresh main with its own kernel wiring.
//
// It is the agent-loop dual of internal/turnbench's trace replay: the SAME real
// adjudication path fak preflight uses —
//
//	adjudicator.Default.SetPolicy(floor)        // install the capability floor
//	v := kernel.Fold(ctx, abi.AdjudicatorsFor(tc), tc)   // one real verdict per call
//
// so the safety floor falls out for free: an allowed get_time runs; an injected
// delete_calendar / wipe_disk is refused at the floor with a closed reason code,
// exactly as guarddemo shows, but inside a live agent loop rather than a replay.
//
// Determinism is a hard requirement (these demos ship a -selfcheck the CI dog-foods
// cross-platform): a Tool handler must be a pure function of its args — no
// time.Now, no network, no randomness. A demo that needs "the current time" injects
// a frozen clock (see cmd/timewolfdemo), so the same scenario reproduces
// bit-identically on any box.
//
// The caller MUST blank-import internal/registrations so the full adjudicator chain
// (resolver, vDSO, adjudicator, ctx-MMU, normgate, IFC, witness, engines) is wired
// before Run folds it — the same one-line requirement every other on-box demo main
// carries.
package agentdemo

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

// Tool is one capability the demo agent can call. Handler produces the result
// string for an ALLOWED call and MUST be a pure function of args (deterministic —
// no time.Now, no network, no randomness) so the demo's -selfcheck reproduces
// identically everywhere. A nil Handler is allowed (the call is adjudicated and
// recorded, but yields no result text).
type Tool struct {
	Name    string
	Summary string
	Handler func(args json.RawMessage) string
}

// Floor is a toolset's capability floor in the demo's own vocabulary. It compiles
// to an adjudicator.Policy: Allow / AllowPrefix affirmatively permit, Deny provably
// refuses with POLICY_BLOCK, and anything else falls to fail-closed DEFAULT_DENY.
type Floor struct {
	// Allow lists exact tool names that are affirmatively permitted.
	Allow []string
	// AllowPrefix permits any tool whose name starts with one of these (the
	// read-only family idiom: "get_", "read_", "search_", "list_").
	AllowPrefix []string
	// Deny lists tool names that are provably refused (POLICY_BLOCK) — the
	// destructive sinks an injection would try to ride.
	Deny []string
}

func (f Floor) policy() adjudicator.Policy {
	allow := make(map[string]bool, len(f.Allow))
	for _, t := range f.Allow {
		allow[t] = true
	}
	deny := make(map[string]abi.ReasonCode, len(f.Deny))
	for _, t := range f.Deny {
		deny[t] = abi.ReasonPolicyBlock
	}
	return adjudicator.Policy{
		Posture:     adjudicator.PostureFailClosed,
		Allow:       allow,
		AllowPrefix: append([]string(nil), f.AllowPrefix...),
		Deny:        deny,
	}
}

// Toolset binds a demo's tools to its capability floor.
type Toolset struct {
	tools map[string]Tool
	floor Floor
}

// NewToolset builds a toolset from a floor and its tools. A tool not named in the
// floor's Allow/AllowPrefix still falls to DEFAULT_DENY at the kernel — the floor,
// not the tool registry, is what admits a call.
func NewToolset(floor Floor, tools ...Tool) *Toolset {
	m := make(map[string]Tool, len(tools))
	for _, t := range tools {
		m[t.Name] = t
	}
	return &Toolset{tools: m, floor: floor}
}

// Tools returns the registered tools sorted by name (for a stable gallery render).
func (ts *Toolset) Tools() []Tool {
	out := make([]Tool, 0, len(ts.tools))
	for _, t := range ts.tools {
		out = append(out, t)
	}
	// insertion order is non-deterministic over a map; sort by name.
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j-1].Name > out[j].Name; j-- {
			out[j-1], out[j] = out[j], out[j-1]
		}
	}
	return out
}

// Step is one planned tool call — a planner's output (the agent's decision to call
// Tool with Args). Note is an optional human label (why the agent made this call,
// e.g. "the injection's payload").
type Step struct {
	Tool string          `json:"tool"`
	Args json.RawMessage `json:"args,omitempty"`
	Note string          `json:"note,omitempty"`
}

// Planner maps a prompt to a sequence of tool-call steps. The lowest-common-
// denominator demo path uses a deterministic, rule-based planner (no model); a live
// latest-model planner satisfies the SAME type later, so the model arm is a clean
// upgrade rather than a fork.
type Planner func(prompt string) []Step

// Turn is one executed step: the real kernel verdict plus the tool result (only for
// an allowed call).
type Turn struct {
	Index   int    `json:"index"`
	Tool    string `json:"tool"`
	Verdict string `json:"verdict"`          // ALLOW|DENY|TRANSFORM|QUARANTINE|WITNESS|DEFER|INDETERMINATE
	Reason  string `json:"reason"`           // closed refusal-vocabulary name (NONE on allow)
	By      string `json:"by,omitempty"`     // which adjudicator decided (forensics)
	Allowed bool   `json:"allowed"`          // the call dispatched (ALLOW or TRANSFORM)
	Result  string `json:"result,omitempty"` // tool output for an allowed call
	Note    string `json:"note,omitempty"`
}

// Transcript is one full agent-loop run: the prompt, every adjudicated turn, the
// agent's final answer (joined from the results it was allowed to get), and the
// allow/deny tally that a -selfcheck asserts against.
type Transcript struct {
	Scenario string `json:"scenario"`
	Prompt   string `json:"prompt"`
	Turns    []Turn `json:"turns"`
	Answer   string `json:"answer"`
	Allowed  int    `json:"allowed"`
	Denied   int    `json:"denied"`
}

// Run replays a planned step sequence through the REAL kernel and returns the
// transcript. It installs the toolset's floor on adjudicator.Default for the run
// (process-global — these are single-purpose demo binaries) and folds the live
// adjudicator chain per call, so every verdict is a real kernel decision, not a
// script. The caller MUST have blank-imported internal/registrations.
func (ts *Toolset) Run(ctx context.Context, scenario, prompt string, plan []Step) (Transcript, error) {
	adjudicator.Default.SetPolicy(ts.floor.policy())
	res := abi.ActiveResolver()
	tr := Transcript{Scenario: scenario, Prompt: prompt}
	var answers []string
	for i, st := range plan {
		args := st.Args
		if len(args) == 0 {
			args = json.RawMessage("{}")
		}
		ref, err := res.Put(ctx, []byte(args))
		if err != nil {
			return Transcript{}, fmt.Errorf("agentdemo: put args for %q: %w", st.Tool, err)
		}
		tc := &abi.ToolCall{Tool: st.Tool, Args: ref}
		v := kernel.Fold(ctx, abi.AdjudicatorsFor(tc), tc)
		allowed := v.Kind == abi.VerdictAllow || v.Kind == abi.VerdictTransform
		turn := Turn{
			Index:   i,
			Tool:    st.Tool,
			Verdict: VerdictName(v.Kind),
			Reason:  abi.ReasonName(v.Reason),
			By:      v.By,
			Allowed: allowed,
			Note:    st.Note,
		}
		if allowed {
			if t, ok := ts.tools[st.Tool]; ok && t.Handler != nil {
				turn.Result = t.Handler(args)
				if turn.Result != "" {
					answers = append(answers, turn.Result)
				}
			}
			tr.Allowed++
		} else {
			tr.Denied++
		}
		tr.Turns = append(tr.Turns, turn)
	}
	tr.Answer = strings.Join(answers, " ")
	return tr, nil
}

// Plan runs planner(prompt) and feeds the result to Run — the one-call convenience
// for a demo that owns a deterministic planner.
func (ts *Toolset) Plan(ctx context.Context, scenario, prompt string, planner Planner) (Transcript, error) {
	return ts.Run(ctx, scenario, prompt, planner(prompt))
}

// VerdictName renders a kernel verdict kind as its stable uppercase name.
func VerdictName(k abi.VerdictKind) string {
	switch k {
	case abi.VerdictAllow:
		return "ALLOW"
	case abi.VerdictDeny:
		return "DENY"
	case abi.VerdictTransform:
		return "TRANSFORM"
	case abi.VerdictQuarantine:
		return "QUARANTINE"
	case abi.VerdictRequireWitness:
		return "WITNESS"
	case abi.VerdictDefer:
		return "DEFER"
	default:
		return "INDETERMINATE"
	}
}

// JSON renders the transcript as indented JSON (safe to log: tool results are the
// demo's own deterministic strings, never raw secrets).
func (tr Transcript) JSON() string {
	b, _ := json.MarshalIndent(tr, "", "  ")
	return string(b)
}

// RenderText writes a plain-text walkthrough of the loop to w: the prompt, one line
// per turn (verdict · tool · result-or-reason), and the agent's final answer plus
// the allow/deny tally. No ANSI, so it reads the same on a plain Windows console.
func (tr Transcript) RenderText(w io.Writer) {
	fmt.Fprintf(w, "agent loop · %s\n", tr.Scenario)
	fmt.Fprintf(w, "  prompt: %s\n\n", tr.Prompt)
	for _, t := range tr.Turns {
		mark := "x"
		if t.Allowed {
			mark = "."
		}
		detail := t.Result
		if !t.Allowed {
			detail = "REFUSED (" + t.Reason + ")"
		}
		fmt.Fprintf(w, "  %s %-9s %-16s %s\n", mark, t.Verdict, t.Tool, detail)
		if t.Note != "" {
			fmt.Fprintf(w, "        ↳ %s\n", t.Note)
		}
	}
	fmt.Fprintf(w, "\n  answer: %s\n", emptyDash(tr.Answer))
	fmt.Fprintf(w, "  floor:  %d allowed · %d refused\n", tr.Allowed, tr.Denied)
}

func emptyDash(s string) string {
	if strings.TrimSpace(s) == "" {
		return "(no answer — every call was refused)"
	}
	return s
}
