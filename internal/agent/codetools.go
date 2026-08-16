package agent

import (
	"context"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/codetools"
)

// codetools.go — arming the kernel-mediated coding filesystem tools (Read/Write/Edit/Grep/Glob) on the
// owned loop (#6703, child of #6658).
//
// The loop's built-in catalog is the airline-support fixture and its only real
// filesystem engine is readengine.go's read-only `fak_read` MCP miss path. So an operator
// asking the native harness to perform a coding task had nothing to dispatch. This file
// is the seam that changes: it binds internal/codetools' engines + adjudicator rung into
// the process registries and teaches Configure()'s policy to admit the configured coding
// tools, so a coding tool proposed by the model crosses the SAME k.Syscall boundary as
// every other tool call — adjudicated, counted, journaled, and dispatched to a registered
// engine.
//
// WHY A GATE RATHER THAN A DIRECT REGISTRATION. abi.RegisterAdjudicator APPENDS, so
// arming twice would stack two rungs that both decide every call. codeToolGate is
// registered exactly ONCE and holds the live toolset behind an atomic pointer, so
// re-arming (a second run, a test, a re-Configure) swaps the toolset instead of growing
// the chain. Unarmed, the gate defers on every call and the loop is byte-for-byte the
// historical loop.

// codeToolRank places the gate before the rank-100 monitor (adjudicator.Default) so it
// can pin abi.ToolCall.Engine while that rung still has the final say on the call, and
// after the cheap shape rungs (grammar 5, ratelimit 8, preflight 10) that should get to
// refuse a malformed call before a filesystem path is ever canonicalized.
const codeToolRank = 20

// codeToolGate is the single registered adjudicator link for the coding tools. It
// owns no policy itself: it forwards to whichever Toolset is currently armed.
type codeToolGate struct{}

var (
	// armedCodeTools is the live toolset, or nil when the coding tools are not armed.
	armedCodeTools atomic.Pointer[codetools.Toolset]
	// codeToolGateOnce guards the one-time chain registration.
	codeToolGateOnce sync.Once
)

// Caps advertises no optional capabilities.
func (codeToolGate) Caps() []abi.Capability { return nil }

// Adjudicate forwards to the armed toolset's rung, or defers when nothing is armed —
// which is what keeps installing this gate from changing any verdict on a loop that never
// asked for the coding tools.
func (codeToolGate) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	ts := armedCodeTools.Load()
	if ts == nil {
		return abi.Verdict{Kind: abi.VerdictDefer, By: codetools.RungName}
	}
	return ts.Adjudicate(ctx, c)
}

// ArmCodeTools builds a codetools Toolset confined to root (empty => the process cwd),
// registers its engines, installs the gate once, and returns the planner-facing catalog
// in the loop's ToolDef shape.
//
// Call it BEFORE the run: Configure() reads the armed state to widen the loop's
// adjudicator policy, and Configure runs at the start of every fak-arm RunArm.
func ArmCodeTools(root string) ([]ToolDef, error) {
	return armCodeTools(root, false)
}

// ArmFocusedCodeTools uses the same kernel catalog with Bash narrowed to focused tests
// and diff/status inspection for browser-operated coding sessions.
func ArmFocusedCodeTools(root string) ([]ToolDef, error) {
	return armCodeTools(root, true)
}

func armCodeTools(root string, focused bool) ([]ToolDef, error) {
	ts, err := codetools.New(codetools.Config{Root: root, FocusedCommands: focused})
	if err != nil {
		return nil, err
	}
	ts.RegisterEngines()
	armedCodeTools.Store(ts)
	codeToolGateOnce.Do(func() { abi.RegisterAdjudicator(codeToolRank, codeToolGate{}) })
	return CodeToolCatalog(), nil
}

// DisarmCodeTools drops the armed toolset, restoring the historical loop. The gate stays
// registered but defers, so nothing has to be unregistered from a frozen registry.
func DisarmCodeTools() { armedCodeTools.Store(nil) }

// CodeToolCatalog renders the coding tools as loop ToolDefs. Empty when unarmed, so
// a caller can splice it into a catalog unconditionally.
func CodeToolCatalog() []ToolDef {
	if armedCodeTools.Load() == nil {
		return nil
	}
	defs := codetools.Catalog()
	out := make([]ToolDef, 0, len(defs))
	for _, d := range defs {
		out = append(out, ToolDef{Type: "function", Function: ToolDefFunction{
			Name:        d.Name,
			Description: d.Description,
			Parameters:  d.Parameters,
		}})
	}
	return out
}

// codeToolMeta returns the vDSO scope meta for a coding tool, or nil when the tool is not
// one. Sourcing it from codetools.CallMeta rather than the loop's own readOnlyTools table
// is what keeps the catalog's read-only bit and the cache key from drifting apart: one
// source of truth for "does this tool mutate", consulted by both.
func codeToolMeta(tool string) map[string]string {
	if armedCodeTools.Load() == nil {
		return nil
	}
	for _, d := range codetools.Catalog() {
		if d.Name == tool {
			return codetools.CallMeta(tool, "")
		}
	}
	return nil
}

// codeToolAllow returns the tool names the armed toolset admits, for Configure() to fold
// into the loop's adjudicator policy. Empty when unarmed.
func codeToolAllow() []string {
	if armedCodeTools.Load() == nil {
		return nil
	}
	names := make([]string, 0, len(codetools.Catalog()))
	for _, d := range codetools.Catalog() {
		names = append(names, d.Name)
	}
	return names
}
