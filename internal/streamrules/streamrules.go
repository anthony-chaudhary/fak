// Package streamrules matches regular-expression rules against incrementally
// received text, thinking, and tool-argument streams. It is deliberately a
// pure detection primitive: callers decide what, if anything, a match means.
package streamrules

import (
	"fmt"
	"path"
	"regexp"
	"strings"
)

// Scope selects the kind of stream a rule can inspect.
type Scope string

const (
	// ScopeText is also the zero-value scope. In particular, the zero value
	// never inspects thinking.
	ScopeText      Scope = "text"
	ScopeThinking  Scope = "thinking"
	ScopeAnyTool   Scope = "any-tool"
	ScopeNamedTool Scope = "named-tool"
)

// ParseScope parses the stable textual scope form used by recorded streams and
// configuration: text, thinking, any-tool, or named-tool:<name>.
func ParseScope(spec string) (Scope, string, error) {
	switch spec {
	case string(ScopeText):
		return ScopeText, "", nil
	case string(ScopeThinking):
		return ScopeThinking, "", nil
	case string(ScopeAnyTool):
		return ScopeAnyTool, "", nil
	}
	const prefix = "named-tool:"
	if strings.HasPrefix(spec, prefix) && strings.TrimPrefix(spec, prefix) != "" {
		return ScopeNamedTool, strings.TrimPrefix(spec, prefix), nil
	}
	return "", "", fmt.Errorf("unknown scope %q", spec)
}

// Rule is an immutable matcher declaration. Tool is required for
// ScopeNamedTool. PathGlob optionally narrows a tool rule using path.Match
// syntax against StreamKey.Path.
type Rule struct {
	Name     string
	Pattern  string
	Scope    Scope
	Tool     string
	PathGlob string
}

// Diagnostic reports a rule dropped during compilation.
type Diagnostic struct {
	Rule  string
	Error string
}

// StreamKey identifies one independently buffered stream. ToolCallID isolates
// sibling tool calls; ToolName and Path carry the metadata used by rule scope.
type StreamKey struct {
	ToolCallID string
	ToolName   string
	Path       string
	Scope      Scope
}

// Match records the complete buffer at the first point a rule matched it.
type Match struct {
	Rule     string
	Key      StreamKey
	Snapshot string
}

type compiledRule struct {
	rule Rule
	re   *regexp.Regexp
}

type streamState struct {
	value        string
	lastSnapshot string
	matched      map[int]bool
}

// Matcher owns the per-turn stream buffers for a compiled rule set.
type Matcher struct {
	rules   []compiledRule
	streams map[StreamKey]*streamState
}

// Compile drops invalid rules, reports each drop, and always returns a usable
// matcher. One malformed declaration therefore cannot wedge registration of
// the remaining rules.
func Compile(rules []Rule) (*Matcher, []Diagnostic) {
	m := &Matcher{streams: make(map[StreamKey]*streamState)}
	var diagnostics []Diagnostic
	for _, rule := range rules {
		if rule.Scope == "" {
			rule.Scope = ScopeText
		}
		if err := validateScope(rule); err != nil {
			diagnostics = append(diagnostics, Diagnostic{Rule: rule.Name, Error: err.Error()})
			continue
		}
		re, err := regexp.Compile(rule.Pattern)
		if err != nil {
			diagnostics = append(diagnostics, Diagnostic{Rule: rule.Name, Error: fmt.Sprintf("compile pattern: %v", err)})
			continue
		}
		if rule.PathGlob != "" {
			if _, err := path.Match(rule.PathGlob, ""); err != nil {
				diagnostics = append(diagnostics, Diagnostic{Rule: rule.Name, Error: fmt.Sprintf("compile path glob: %v", err)})
				continue
			}
		}
		m.rules = append(m.rules, compiledRule{rule: rule, re: re})
	}
	return m, diagnostics
}

func validateScope(rule Rule) error {
	switch rule.Scope {
	case ScopeText, ScopeThinking, ScopeAnyTool:
		return nil
	case ScopeNamedTool:
		if rule.Tool == "" {
			return fmt.Errorf("named-tool scope requires a tool name")
		}
		return nil
	default:
		return fmt.Errorf("unknown scope %q", rule.Scope)
	}
}

// ResetTurn discards every buffered stream and match latch.
func (m *Matcher) ResetTurn() {
	m.streams = make(map[StreamKey]*streamState)
}

// CheckDelta appends delta to key's independent buffer and returns rules that
// match for the first time on that stream during this turn.
func (m *Matcher) CheckDelta(key StreamKey, delta string) []Match {
	state := m.state(key)
	state.value += delta
	state.lastSnapshot = state.value
	return m.check(key, state)
}

// CheckSnapshot replaces key's buffer with snapshot. Consecutive identical
// snapshots are ignored. Like CheckDelta, each rule is reported at most once
// per stream per turn.
func (m *Matcher) CheckSnapshot(key StreamKey, snapshot string) []Match {
	state := m.state(key)
	if snapshot == state.lastSnapshot {
		return nil
	}
	state.value = snapshot
	state.lastSnapshot = snapshot
	return m.check(key, state)
}

func (m *Matcher) state(key StreamKey) *streamState {
	if key.Scope == "" {
		key.Scope = ScopeText
	}
	state := m.streams[key]
	if state == nil {
		state = &streamState{matched: make(map[int]bool)}
		m.streams[key] = state
	}
	return state
}

func (m *Matcher) check(key StreamKey, state *streamState) []Match {
	var matches []Match
	for i, rule := range m.rules {
		if state.matched[i] || !inScope(rule.rule, key) || !rule.re.MatchString(state.value) {
			continue
		}
		state.matched[i] = true
		matches = append(matches, Match{Rule: rule.rule.Name, Key: key, Snapshot: state.value})
	}
	return matches
}

func inScope(rule Rule, key StreamKey) bool {
	scope := key.Scope
	if scope == "" {
		scope = ScopeText
	}
	switch rule.Scope {
	case ScopeText:
		if scope != ScopeText {
			return false
		}
	case ScopeThinking:
		if scope != ScopeThinking {
			return false
		}
	case ScopeAnyTool:
		if scope != ScopeAnyTool && scope != ScopeNamedTool {
			return false
		}
	case ScopeNamedTool:
		if (scope != ScopeAnyTool && scope != ScopeNamedTool) || key.ToolName != rule.Tool {
			return false
		}
	}
	if rule.PathGlob != "" {
		matched, _ := path.Match(rule.PathGlob, key.Path)
		return matched
	}
	return true
}
