package agent

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/codetools"
)

// CodeToolsSelfcheckResult is the behavioral acceptance result for the default
// owned-loop repository-tool catalog in an arbitrary workspace.
type CodeToolsSelfcheckResult struct {
	Catalog       []string `json:"catalog"`
	EngineCalls   int      `json:"engine_calls"`
	Denies        int      `json:"denies"`
	ReadOK        bool     `json:"read_ok"`
	WriteOK       bool     `json:"write_ok"`
	EditOK        bool     `json:"edit_ok"`
	BashOK        bool     `json:"bash_ok"`
	GrepOK        bool     `json:"grep_ok"`
	GlobOK        bool     `json:"glob_ok"`
	TraversalDeny bool     `json:"traversal_deny"`
}

type defaultsCheckPlanner struct {
	turns    []*Completion
	n        int
	messages []Message
}

func (p *defaultsCheckPlanner) Complete(_ context.Context, messages []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	p.messages = append([]Message(nil), messages...)
	c := bindLatestCodeToolVersion(p.turns[p.n], messages)
	if p.n < len(p.turns)-1 {
		p.n++
	}
	return c, nil
}
func (*defaultsCheckPlanner) Model() string { return "defaults-selfcheck" }

func bindLatestCodeToolVersion(c *Completion, messages []Message) *Completion {
	if c == nil || len(c.Message.ToolCalls) == 0 {
		return c
	}
	clone := *c
	clone.Message.ToolCalls = append([]ToolCall(nil), c.Message.ToolCalls...)
	changed := false
	for i := range clone.Message.ToolCalls {
		call := &clone.Message.ToolCalls[i]
		if call.Function.Name != codetools.ToolEdit && call.Function.Name != codetools.ToolWrite {
			continue
		}
		var args map[string]any
		if json.Unmarshal([]byte(call.Function.Arguments), &args) != nil {
			continue
		}
		if call.Function.Name == codetools.ToolWrite && args["mode"] == "create" {
			continue
		}
		if expected, _ := args["expected_version"].(string); expected != "" {
			continue
		}
		if version := latestCodeToolVersion(messages); version != "" {
			args["expected_version"] = version
			call.Function.Arguments = mustSelfcheckArgs(args)
			changed = true
		}
	}
	if !changed {
		return c
	}
	return &clone
}

func latestCodeToolVersion(messages []Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role != "tool" {
			continue
		}
		var receipt struct {
			Version string `json:"version"`
		}
		if json.Unmarshal([]byte(messages[i].Content), &receipt) == nil && receipt.Version != "" {
			return receipt.Version
		}
	}
	return ""
}

// RunCodeToolsSelfcheck drives all six default tools and one escaping read through
// the real owned agent loop and kernel engines. The caller owns workspace cleanup.
func RunCodeToolsSelfcheck(ctx context.Context, workspace, outside string) (CodeToolsSelfcheckResult, error) {
	if err := os.WriteFile(filepath.Join(workspace, "fixture.txt"), []byte("needle before\n"), 0o644); err != nil {
		return CodeToolsSelfcheckResult{}, err
	}
	if err := os.WriteFile(filepath.Join(outside, "secret.txt"), []byte("must-not-leak"), 0o600); err != nil {
		return CodeToolsSelfcheckResult{}, err
	}
	catalog, err := ArmFocusedCodeTools(workspace)
	if err != nil {
		return CodeToolsSelfcheckResult{}, err
	}
	defer DisarmCodeTools()
	bash := "git status --short"
	if runtime.GOOS == "windows" {
		bash = "git status --short"
	}
	calls := []struct{ tool, args string }{
		{codetools.ToolRead, `{"file_path":"fixture.txt"}`},
		{codetools.ToolWrite, `{"file_path":"created.txt","content":"alpha\n","mode":"create"}`},
		{codetools.ToolEdit, `{"file_path":"created.txt","old_string":"alpha","new_string":"beta"}`},
		{codetools.ToolBash, mustSelfcheckArgs(map[string]any{"command": bash})},
		{codetools.ToolGrep, `{"pattern":"needle","glob":"*.txt"}`},
		{codetools.ToolGlob, `{"pattern":"**/*.txt"}`},
		{codetools.ToolRead, mustSelfcheckArgs(map[string]any{"file_path": filepath.Join(outside, "secret.txt")})},
	}
	turns := make([]*Completion, 0, len(calls)+1)
	for i, c := range calls {
		turns = append(turns, &Completion{Message: Message{ToolCalls: []ToolCall{{ID: string(rune('a' + i)), Function: Func{Name: c.tool, Arguments: c.args}}}}})
	}
	turns = append(turns, &Completion{Message: Message{Content: "done"}})
	planner := &defaultsCheckPlanner{turns: turns}
	var log []traceEvent
	metrics, err := RunArm(ctx, planner, "exercise bounded repository tools", true, len(turns)+1, &log, WithToolCatalog(catalog))
	if err != nil {
		return CodeToolsSelfcheckResult{}, err
	}
	result := CodeToolsSelfcheckResult{EngineCalls: metrics.EngineCalls, Denies: metrics.Denies, TraversalDeny: metrics.Denies == 1}
	for _, d := range catalog {
		result.Catalog = append(result.Catalog, d.Function.Name)
	}
	for _, m := range planner.messages {
		if m.Role != "tool" {
			continue
		}
		switch m.Name {
		case codetools.ToolRead:
			result.ReadOK = result.ReadOK || strings.Contains(m.Content, "needle before")
		case codetools.ToolWrite:
			result.WriteOK = true
		case codetools.ToolEdit:
			result.EditOK = true
		case codetools.ToolBash:
			result.BashOK = true
		case codetools.ToolGrep:
			result.GrepOK = strings.Contains(m.Content, "fixture.txt")
		case codetools.ToolGlob:
			result.GlobOK = strings.Contains(m.Content, "fixture.txt") && strings.Contains(m.Content, "created.txt")
		}
	}
	return result, nil
}

func mustSelfcheckArgs(v any) string { b, _ := json.Marshal(v); return string(b) }
