// The per-tool runtime envelope (seam 5 of the tool-process table,
// docs/notes/CONCEPT-TOOL-PROCESS-TABLE-2026-07-02.md): deadline and heartbeat
// cadence belong in the capability manifest, per tool — the runtime envelope
// granted at admission alongside the capability itself. "You may run this
// tool" and "you may run it for this long, reporting at this cadence" are one
// grant.
//
// This is the FILE layer only: the manifest declares the grant, load-time
// compilation validates it, and an embedder of the toolproc supervisor
// (the gateway proxy, `fak guard`, the hook adapter) resolves each spawn's
// envelope via EnvelopeFor and passes it to Spawn. The field names are the
// toolproc journal's own (`deadline_ms`, `heartbeat_every_ms`) so the grant
// and the event stream share one vocabulary; zero keeps the journal meaning
// "not declared" (the fold's Config defaults apply).
package policy

import (
	"fmt"
	"sort"
	"strings"
)

// ToolRuntimeCatchAll is the reserved tool name for the default row: a tool
// with no exact row resolves to this one when declared.
const ToolRuntimeCatchAll = "*"

// ToolRuntimeRule is one per-tool runtime-envelope grant in the manifest's
// `tool_runtime` block. Tool is the exact tool name (or ToolRuntimeCatchAll);
// at least one of DeadlineMS / HeartbeatEveryMS must be positive — an all-zero
// row grants nothing and is refused at load. Like RateLimit and Isolation,
// this is manifest/runtime-only, NOT an adjudicator.Policy field (how long a
// call may run is separate from the name-level allow/deny floor).
type ToolRuntimeRule struct {
	Tool             string `json:"tool"`
	DeadlineMS       int64  `json:"deadline_ms,omitempty"`
	HeartbeatEveryMS int64  `json:"heartbeat_every_ms,omitempty"`
}

// ToolRuntimeTable is the compiled per-tool envelope lookup. A nil table
// (no tool_runtime block in the manifest) resolves nothing — absent config
// grants no envelope, it does not invent one.
type ToolRuntimeTable struct {
	rules map[string]ToolRuntimeRule
}

// compileToolRuntime validates a declared tool_runtime block (absent => nil,
// no envelopes) at policy LOAD, so an empty tool name, a negative value, an
// all-zero row, or a duplicate tool fails loud here, never at spawn time.
func compileToolRuntime(rules []ToolRuntimeRule) (*ToolRuntimeTable, error) {
	if len(rules) == 0 {
		return nil, nil // absent => no envelopes declared
	}
	t := &ToolRuntimeTable{rules: make(map[string]ToolRuntimeRule, len(rules))}
	for i, r := range rules {
		tool := strings.TrimSpace(r.Tool)
		if tool == "" {
			return nil, fmt.Errorf("tool_runtime[%d]: tool is required", i)
		}
		if r.DeadlineMS < 0 || r.HeartbeatEveryMS < 0 {
			return nil, fmt.Errorf("tool_runtime[%d] %s: deadline_ms/heartbeat_every_ms must be non-negative (got deadline=%d heartbeat=%d)",
				i, tool, r.DeadlineMS, r.HeartbeatEveryMS)
		}
		if r.DeadlineMS == 0 && r.HeartbeatEveryMS == 0 {
			return nil, fmt.Errorf("tool_runtime[%d] %s: declare at least one of deadline_ms / heartbeat_every_ms (an all-zero row grants nothing)", i, tool)
		}
		if _, dup := t.rules[tool]; dup {
			return nil, fmt.Errorf("tool_runtime[%d]: duplicate row for tool %q", i, tool)
		}
		r.Tool = tool
		t.rules[tool] = r
	}
	return t, nil
}

// EnvelopeFor resolves the runtime envelope granted to tool: the exact row
// wins, else the "*" catch-all row, else ok=false — the embedder then spawns
// with a zero envelope and the toolproc fold's Config defaults apply. A nil
// table resolves nothing (no block, no grant).
func (t *ToolRuntimeTable) EnvelopeFor(tool string) (ToolRuntimeRule, bool) {
	if t == nil {
		return ToolRuntimeRule{}, false
	}
	if r, ok := t.rules[strings.TrimSpace(tool)]; ok {
		return r, true
	}
	if r, ok := t.rules[ToolRuntimeCatchAll]; ok {
		return r, true
	}
	return ToolRuntimeRule{}, false
}

// Rules returns the compiled rows sorted by tool name ("*" first by ASCII),
// for the operator summary and tests. Nil-safe.
func (t *ToolRuntimeTable) Rules() []ToolRuntimeRule {
	if t == nil || len(t.rules) == 0 {
		return nil
	}
	out := make([]ToolRuntimeRule, 0, len(t.rules))
	for _, r := range t.rules {
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Tool < out[j].Tool })
	return out
}
