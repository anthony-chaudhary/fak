// Package guardtrace is the end-to-end test/replay harness for `fak guard`.
//
// `fak guard` is the productized form of fak's whole reason for existing: it stands
// up the gateway on loopback, points a child agent at it, and adjudicates EVERY tool
// call the agent proposes against a capability floor — allowing benign calls, denying
// the danger classes (rm -rf, sudo, curl-pipe-sh, writes into .ssh/.git), recording
// each verdict to a hash-chained decision journal, and surfacing the per-turn token /
// cache economy. Until now that whole path had no end-to-end exercise: the unit tests
// drove an in-process stub planner one turn at a time and never fired the floor while
// also asserting on the journal and the token numbers.
//
// guardtrace closes that gap with a hermetic, no-API-key, no-GPU loop:
//
//   - A FIXTURE TRACE (Fixture) is an ordered script of turns. Each turn is one
//     upstream model response carrying tool_use / tool_calls blocks AND a usage block;
//     each call names the verdict the floor MUST reach (allow / deny + reason).
//   - A FAKE UPSTREAM (FakeUpstream) is an httptest.Server that replays the fixture
//     turn-by-turn in a provider-correct wire shape (Anthropic /v1/messages or
//     OpenAI /v1/chat/completions). Pointing the gateway's Config.BaseURL at it runs
//     the REAL proxy planner + parse path — not a stub.
//
// The gateway end-to-end test (internal/gateway) and the `fak guard --replay-trace`
// operator surface (cmd/fak) both build on this package, so the thing the operator
// watches in the terminal is the SAME loop the test asserts on.
package guardtrace

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

// Fixture is a parsed guard trace: an ordered list of turns the fake upstream replays.
type Fixture struct {
	SliceID string `json:"slice_id"`
	Turns   []Turn `json:"turns"`
}

// Turn is one upstream model response: the tool calls it proposes plus the token usage
// the provider would report for that turn.
type Turn struct {
	Note string `json:"note,omitempty"`
	// Messages, when present, is the client history the replay posts into the gateway for
	// this turn. Older fixtures omit it and get the compact default request; context
	// fixtures use it to drive ctx-view over the same HTTP path as guard/serve.
	Messages []RequestMessage `json:"messages,omitempty"`
	Usage    Usage            `json:"usage"`
	Calls    []Call           `json:"calls"`
}

// RequestMessage is one client-side history span posted into the gateway during replay.
type RequestMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Call is one proposed tool call in a turn, with the verdict the floor must reach.
type Call struct {
	ID   string          `json:"id"`
	Tool string          `json:"tool"`
	Args json.RawMessage `json:"args"`
	// Class is the disposition the guard floor MUST reach for this call: "allow" (the
	// call survives to the caller) or "deny" (the floor drops it). Reason, when set,
	// is the closed-vocabulary refusal code the deny must carry (e.g. POLICY_BLOCK,
	// SELF_MODIFY) — asserted so a deny for the WRONG reason is caught, not just any deny.
	Class  string `json:"class"`
	Reason string `json:"reason,omitempty"`
}

// Usage is the provider-reported token accounting for one turn. Field names match BOTH
// the Anthropic Messages usage object and what the OpenAI usage object is mapped from,
// so one fixture drives both wires.
type Usage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
}

// ExpectAllow reports whether this call must survive the floor.
func (c Call) ExpectAllow() bool { return strings.EqualFold(strings.TrimSpace(c.Class), "allow") }

// ArgString renders the call's args as the compact JSON the model would emit — the form
// the upstream wire carries (Anthropic tool_use.input object; OpenAI tool_calls
// function.arguments string).
func (c Call) ArgString() string {
	if len(c.Args) == 0 {
		return "{}"
	}
	// Re-marshal to drop any fixture whitespace, so the emitted wire bytes are tight.
	var v any
	if err := json.Unmarshal(c.Args, &v); err != nil {
		return string(c.Args)
	}
	b, err := json.Marshal(v)
	if err != nil {
		return string(c.Args)
	}
	return string(b)
}

// ArgPreview is a short, single-line form of the call's most salient argument for a
// human report line (the command for Bash, the file_path for a write). It never returns
// the whole args blob, so a replay report line stays one row.
func (c Call) ArgPreview() string {
	var m map[string]any
	if json.Unmarshal(c.Args, &m) != nil {
		return ""
	}
	for _, k := range []string{"command", "file_path", "filePath", "path", "query"} {
		if v, ok := m[k]; ok {
			if s, ok := v.(string); ok && s != "" {
				return truncate(s, 48)
			}
		}
	}
	return ""
}

// LoadFixture reads and parses a guard trace fixture from disk, validating that every
// call names a known class so a typo fails loud at load rather than silently passing the
// assertions. It is the one parse path the test and the CLI share.
func LoadFixture(path string) (*Fixture, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("guardtrace: read fixture %s: %w", path, err)
	}
	return ParseFixture(raw)
}

// ParseFixture parses fixture bytes (the path-free core of LoadFixture).
func ParseFixture(raw []byte) (*Fixture, error) {
	var f Fixture
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, fmt.Errorf("guardtrace: decode fixture: %w", err)
	}
	if len(f.Turns) == 0 {
		return nil, fmt.Errorf("guardtrace: fixture has no turns")
	}
	for ti, t := range f.Turns {
		if len(t.Calls) == 0 {
			return nil, fmt.Errorf("guardtrace: turn %d has no calls", ti)
		}
		for mi, m := range t.Messages {
			switch strings.ToLower(strings.TrimSpace(m.Role)) {
			case "system", "user", "assistant":
			default:
				return nil, fmt.Errorf("guardtrace: turn %d message %d has unknown role %q (want system|user|assistant)", ti, mi, m.Role)
			}
			if strings.TrimSpace(m.Content) == "" {
				return nil, fmt.Errorf("guardtrace: turn %d message %d has empty content", ti, mi)
			}
		}
		for ci, c := range t.Calls {
			if c.Tool == "" {
				return nil, fmt.Errorf("guardtrace: turn %d call %d has no tool", ti, ci)
			}
			cls := strings.ToLower(strings.TrimSpace(c.Class))
			if cls != "allow" && cls != "deny" {
				return nil, fmt.Errorf("guardtrace: turn %d call %q has unknown class %q (want allow|deny)", ti, c.Tool, c.Class)
			}
		}
	}
	return &f, nil
}

// Expectation folds a fixture into the aggregate counts the gateway's AdjudicationSummary
// must report after the whole trace runs, so a caller can assert the roll-up the exit
// banner prints in one comparison.
type Expectation struct {
	TotalCalls int
	Allowed    int
	Denied     int
	// ByReason counts the deny reasons the fixture declares (e.g. POLICY_BLOCK -> 2).
	ByReason map[string]int
	// CacheReadTokens / CacheCreationTokens / InputTokens are the summed provider usage
	// axes across every turn — the token economy the summary surfaces.
	CacheReadTokens     int
	CacheCreationTokens int
	InputTokens         int
}

// Expect computes the aggregate expectation over the whole fixture.
func (f *Fixture) Expect() Expectation {
	e := Expectation{ByReason: map[string]int{}}
	for _, t := range f.Turns {
		e.InputTokens += t.Usage.InputTokens
		e.CacheReadTokens += t.Usage.CacheReadInputTokens
		e.CacheCreationTokens += t.Usage.CacheCreationInputTokens
		for _, c := range t.Calls {
			e.TotalCalls++
			if c.ExpectAllow() {
				e.Allowed++
				continue
			}
			e.Denied++
			if c.Reason != "" {
				e.ByReason[c.Reason]++
			}
		}
	}
	return e
}

// Reasons returns the declared deny reasons in stable order (for a deterministic report).
func (e Expectation) Reasons() []string {
	out := make([]string, 0, len(e.ByReason))
	for r := range e.ByReason {
		out = append(out, r)
	}
	sort.Strings(out)
	return out
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	if n <= 1 {
		return s[:n]
	}
	return s[:n-1] + "…"
}
