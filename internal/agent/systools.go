package agent

import (
	"context"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/systools"
)

// sysToolRank places the gate before the rank-100 monitor (adjudicator.Default) so it
// can pin abi.ToolCall.Engine while that rung still has the final say on the call.
const sysToolRank = 21

// sysToolGate is the single registered adjudicator link for the system tools.
type sysToolGate struct{}

var (
	// armedSysTools is the live toolset, or nil when systools are not armed.
	armedSysTools atomic.Pointer[systools.Toolset]
	// sysToolGateOnce guards the one-time chain registration.
	sysToolGateOnce sync.Once
)

// Caps advertises no optional capabilities.
func (sysToolGate) Caps() []abi.Capability { return nil }

// Adjudicate forwards to the armed toolset's rung, or defers when nothing is armed.
func (sysToolGate) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	ts := armedSysTools.Load()
	if ts == nil {
		return abi.Verdict{Kind: abi.VerdictDefer, By: systools.RungName}
	}
	return ts.Adjudicate(ctx, c)
}

// ArmSysTools builds a systools Toolset configured with cfg, registers its engines,
// installs the gate once, and returns the planner-facing catalog in the loop's ToolDef shape.
func ArmSysTools(cfg systools.Config) ([]ToolDef, error) {
	ts, err := systools.New(cfg)
	if err != nil {
		return nil, err
	}
	ts.RegisterEngines()
	armedSysTools.Store(ts)
	sysToolGateOnce.Do(func() { abi.RegisterAdjudicator(sysToolRank, sysToolGate{}) })
	return SysToolCatalog(), nil
}

// DisarmSysTools drops the armed toolset, restoring the unarmed state.
func DisarmSysTools() {
	armedSysTools.Store(nil)
}

// SysToolCatalog renders the system tools as loop ToolDefs. Empty when unarmed.
func SysToolCatalog() []ToolDef {
	if armedSysTools.Load() == nil {
		return nil
	}
	defs := systools.Catalog()
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

// sysToolMeta returns the vDSO scope meta for a system tool, or (nil, false) when unarmed or not a systool.
func sysToolMeta(tool string) (map[string]string, bool) {
	if armedSysTools.Load() == nil {
		return nil, false
	}
	for _, d := range systools.Catalog() {
		if d.Name == tool {
			return systools.CallMeta(tool, ""), true
		}
	}
	return nil, false
}

// sysToolAllow returns the tool names the armed toolset admits, for Configure() to fold
// into the loop's adjudicator policy. Empty when unarmed.
func sysToolAllow() []string {
	if armedSysTools.Load() == nil {
		return nil
	}
	names := make([]string, 0, len(systools.Catalog()))
	for _, d := range systools.Catalog() {
		names = append(names, d.Name)
	}
	return names
}
