package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// turnkeyAgentWarmGate is the turnkey (package-main) equivalent of the gateway's
// agent-warm readiness gate (internal/gateway/readiness_warmup.go, CW-07 #13333).
// The turnkey `fak up` path owns its own readinessGate rather than embedding a
// gateway.Server, so it needs the same second-half gate here: the #3051 warmup
// answers "is the backend LOADED?", while this answers "is a warm PREFIX resident
// and reusable?". A configured profile is admitted ONLY against a live receipt
// (matching identity, restored to the stable boundary); a cold/partial/mismatched
// warm DEGRADES with a closed reason. The zero value is unconfigured (silent), so
// a bare &turnkeyServer{} stays ready — existing tests that construct one remain
// byte-for-byte unaffected. Guarded by its own mutex; safe on a nil receiver.
type turnkeyAgentWarmGate struct {
	mu         sync.Mutex
	configured bool
	spec       agent.WarmPrefixSpec
	status     string
	reason     string
	receipt    *agent.WarmReceipt
}

// configure installs a derived descriptor and moves the gate to pending.
// Reconfiguring clears any prior receipt, so a generation change can never reuse
// the previous profile's warm readiness.
func (g *turnkeyAgentWarmGate) configure(spec agent.WarmPrefixSpec) {
	if g == nil {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	g.configured = true
	g.spec = spec
	g.status = gateway.AgentWarmPending
	g.reason = ""
	g.receipt = nil
}

// observe records a native warm attempt and adjudicates readiness from the
// receipt, never from the attempt's mere completion. A receipt that is not Ready,
// whose identity does not match, or whose restored prefix does not reach the
// stable boundary is DEGRADED with a closed reason; unsupported is reported
// independently and never holds readiness.
func (g *turnkeyAgentWarmGate) observe(receipt agent.WarmReceipt, unsupported bool, err error) {
	if g == nil {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.configured {
		g.status = gateway.AgentWarmUnconfigured
		g.receipt = nil
		return
	}
	g.receipt = nil
	switch {
	case unsupported:
		g.status = gateway.AgentWarmUnsupported
		g.reason = "unsupported"
		return
	case err != nil:
		g.status = gateway.AgentWarmDegraded
		g.reason = receipt.Reason
		if g.reason == "" {
			g.reason = "warm_error"
		}
		return
	}
	switch {
	case receipt.Identity == "" || receipt.Identity != g.spec.Identity:
		g.status = gateway.AgentWarmDegraded
		g.reason = "identity_mismatch"
		return
	case !receipt.Ready:
		g.status = gateway.AgentWarmDegraded
		g.reason = receipt.Reason
		if g.reason == "" {
			g.reason = "not_ready"
		}
		return
	case receipt.RestoredTokens < receipt.RequestedTokens || receipt.RestoredTokens <= 0:
		g.status = gateway.AgentWarmDegraded
		g.reason = "partial_restore"
		return
	case receipt.Status != agent.WarmStatusReady:
		g.status = gateway.AgentWarmDegraded
		g.reason = "status_not_ready"
		return
	}
	g.status = gateway.AgentWarmReady
	g.reason = ""
	r := receipt
	g.receipt = &r
}

// admit reports whether readiness must be HELD for a configured agent-warm
// profile, with the closed blocking status/reason. A pending or degraded gate
// blocks; an unconfigured/unsupported/ready gate does not.
func (g *turnkeyAgentWarmGate) admit() (blocked bool, status, reason string) {
	if g == nil {
		return false, gateway.AgentWarmUnconfigured, ""
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	switch g.status {
	case gateway.AgentWarmPending, gateway.AgentWarmDegraded:
		return true, g.status, g.reason
	default:
		return false, g.status, g.reason
	}
}

// agentWarmBlock returns the read-only /healthz projection of the gate, or nil
// when no profile was ever configured (the key is then absent, never a
// fabricated status).
func (g *turnkeyAgentWarmGate) agentWarmBlock() map[string]any {
	if g == nil {
		return nil
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.configured {
		return nil
	}
	block := map[string]any{"status": g.status}
	if g.reason != "" {
		block["reason"] = g.reason
	}
	if g.receipt != nil {
		block["identity"] = g.receipt.Identity
		block["restored_tokens"] = g.receipt.RestoredTokens
	}
	return block
}

func (g *turnkeyAgentWarmGate) configuration() (agent.WarmPrefixSpec, bool) {
	if g == nil {
		return agent.WarmPrefixSpec{}, false
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.spec, g.configured
}

// turnkeyAgentWarmWorkspaceTenant is the cache scope the turnkey-owned agent warm
// is bounded to; a local turnkey serve owns exactly one tenant on the appliance.
const turnkeyAgentWarmWorkspaceTenant = "turnkey-local"

// turnkeyRequestContext binds the turnkey-owned prefix-cache identity to a request
// context (CW-18, #13328). The turnkey path owns exactly one tenant on the
// appliance, and CW-09 (#13332) materializes its startup warm under that tenant's
// SCOPED cache tree. Without this binding a real turnkey request reaches the
// planner UNSCOPED: its lookup consults the shared tree, never sees the warmed
// prefix, and pays a full prefill the warm promised to avoid. Binding the same
// tenant on both the warm ctx and the demand ctx makes the demand lookup consult
// the scoped tree the warm restored into. A tenant-free context preserves the
// legacy single-user namespace.
func turnkeyRequestContext(ctx context.Context) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return agent.WithPrefixCacheIdentity(ctx, turnkeyAgentWarmWorkspaceTenant, "")
}

// resolveUpAgentWarmWorkspace resolves the workspace whose AGENTS.md seeds the
// turnkey startup agent warm: an explicit --code-workspace wins, otherwise
// FAK_UP_CODE_WORKSPACE, otherwise the current directory. It mirrors
// resolveNativeCodeWorkspace (serve parity) so the two entrypoints agree.
func resolveUpAgentWarmWorkspace(configured string) string {
	if workspace := strings.TrimSpace(configured); workspace != "" {
		return workspace
	}
	if workspace := strings.TrimSpace(os.Getenv("FAK_UP_CODE_WORKSPACE")); workspace != "" {
		return workspace
	}
	workspace, err := os.Getwd()
	if err != nil {
		return ""
	}
	return workspace
}

// turnkeyAgentWarmInputsInstaller is the narrow seam the turnkey startup path
// uses to hand the warmer the REAL stable inputs (workspace instructions +
// ordered tool schemas) its later WarmPrefix re-encodes. *agent.InKernelPlanner
// satisfies it.
type turnkeyAgentWarmInputsInstaller interface {
	SetWarmPrefixInputs(agent.WarmPrefixInputs)
}

// turnkeyAgentWarmReleaser is the shutdown-owner hook the turnkey lifecycle
// invokes exactly once when it installed a startup agent warm. A planner with no
// startup warm is left untouched.
type turnkeyAgentWarmReleaser interface {
	ReleaseStartupWarm()
}

// turnkeyAgentWarmWarmer is the narrow derive/materialize seam the turnkey warm
// path calls. *agent.InKernelPlanner satisfies it.
type turnkeyAgentWarmWarmer interface {
	DeriveWarmPrefix(tenant, agent string, in agent.WarmPrefixInputs) (agent.WarmPrefixSpec, error)
	WarmPrefix(ctx context.Context, spec agent.WarmPrefixSpec) (agent.WarmReceipt, error)
}

// installTurnkeyAgentWarm installs the effective agent KV-cache warm profile on
// the turnkey server BEFORE readiness is bound (CW-09, #13332). It resolves the
// workspace instruction snapshot (AGENTS.md) and the ordered kernel coding-tool
// catalog — the SAME bytes/order the forward path resolves — hands them to the
// warmer via SetWarmPrefixInputs (the warmer re-encodes the boundary from them),
// then derives and installs the profile on the turnkey agent-warm gate.
//
// It installs but does NOT execute the warm: the profile is armed before
// readiness (agent_warm_pending from the first probe) and the caller materializes
// it (runTurnkeyAgentWarmup) on the serve's existing background startup path.
//
// It is deliberately fail-open for readiness: a workspace with no readable
// instruction snapshot, or a planner that cannot derive a bounded descriptor, is
// reported and left UNCONFIGURED so readiness is never held on a profile that
// cannot be realized and no synthetic warm is invented. It returns true only when
// a warm profile was actually installed. Never fatal: a warm is an optimization.
func installTurnkeyAgentWarm(ts *turnkeyServer, workspace string, log io.Writer) bool {
	if ts == nil {
		return false
	}
	return installTurnkeyAgentWarmForPlanner(ts.planner, ts.agentWarm, workspace, log)
}

// installTurnkeyAgentWarmForPlanner is the planner+gate half of the turnkey
// install, split out so the server can arm the profile BEFORE the turnkeyServer
// value exists (readiness must be held from the first probe). It returns true
// only when a bounded profile was actually installed on the gate.
func installTurnkeyAgentWarmForPlanner(planner agent.Planner, gate *turnkeyAgentWarmGate, workspace string, log io.Writer) bool {
	if planner == nil || gate == nil {
		return false
	}
	installer, ok := planner.(turnkeyAgentWarmInputsInstaller)
	if !ok {
		return false
	}
	workspace = strings.TrimSpace(workspace)
	if workspace == "" {
		if log != nil {
			fmt.Fprintf(log, "fak up: agent cache warm unconfigured (no workspace resolved)\n")
		}
		return false
	}
	instructions, err := os.ReadFile(filepath.Join(workspace, "AGENTS.md"))
	if err != nil || len(instructions) == 0 {
		// No effective agent instruction snapshot: the warm cannot be bounded to a
		// real, identity-bearing prefix. Leave the gate unconfigured — readiness is
		// not held and no synthetic profile is invented.
		if log != nil {
			fmt.Fprintf(log, "fak up: agent cache warm unconfigured (no readable AGENTS.md under %q): %v\n", workspace, err)
		}
		return false
	}
	inputs := agent.WarmPrefixInputs{
		Instructions: instructions,
		Tools:        agent.ToolCatalog(),
	}
	installer.SetWarmPrefixInputs(inputs)
	spec, err := deriveTurnkeyAgentWarmPrefix(planner, inputs)
	if err != nil {
		// The planner could not derive a bounded descriptor (nil/unconfigured
		// planner, missing model/tokenizer identity). The gate is left unconfigured
		// so readiness is unaffected.
		if log != nil {
			fmt.Fprintf(log, "fak up: agent cache warm unavailable: %v\n", err)
		}
		return false
	}
	gate.configure(spec)
	if log != nil {
		fmt.Fprintf(log, "fak up: agent cache warm armed identity=%s (materialized on the background startup path)\n", spec.Identity)
	}
	return true
}

// deriveTurnkeyAgentWarmPrefix derives the warm descriptor through the planner's
// warmer seam, so the gate binds the receipt against exactly the descriptor the
// later WarmPrefix targets. The turnkey path installs the gate itself (it owns the
// readinessGate), so it derives here rather than through a gateway.Server.
func deriveTurnkeyAgentWarmPrefix(planner agent.Planner, in agent.WarmPrefixInputs) (agent.WarmPrefixSpec, error) {
	warmer, ok := planner.(turnkeyAgentWarmWarmer)
	if !ok {
		return agent.WarmPrefixSpec{}, gateway.ErrAgentWarmUnconfigured
	}
	spec, err := warmer.DeriveWarmPrefix(turnkeyAgentWarmWorkspaceTenant, "", in)
	if err != nil {
		return agent.WarmPrefixSpec{}, err
	}
	if !spec.Bounded() {
		return agent.WarmPrefixSpec{}, gateway.ErrAgentWarmUnconfigured
	}
	return spec, nil
}

// runTurnkeyAgentWarmup materializes the profile configured by
// installTurnkeyAgentWarm through the planner's warmer and adjudicates readiness
// from the returned receipt. It is the execution half (CW-09): the host calls it
// at boot alongside the #3051 backend warmup. A planner that is not a warmer, or
// a gate never configured, is an explicit no-op.
func (ts *turnkeyServer) runTurnkeyAgentWarmup(ctx context.Context) {
	if ts == nil || ts.planner == nil {
		return
	}
	warmer, ok := ts.planner.(turnkeyAgentWarmWarmer)
	if !ok {
		ts.agentWarm.observe(agent.WarmReceipt{}, true, gateway.ErrAgentWarmUnconfigured)
		return
	}
	spec, configured := ts.agentWarm.configuration()
	if !configured {
		return
	}
	// CW-18 (#13328): materialize the warm under the SAME turnkey tenant the demand
	// path binds, so the prime admits into the scoped tree the readback consults
	// (an unscoped prime would admit to the shared tree and read back as a miss).
	receipt, err := warmer.WarmPrefix(turnkeyRequestContext(ctx), spec)
	unsupported := errors.Is(err, agent.ErrWarmPrefixUnsupported)
	ts.agentWarm.observe(receipt, unsupported, err)
}

// releaseTurnkeyAgentWarm releases the planner's startup warm ownership exactly
// once. It is the shutdown half: Close, Shutdown, and the bounded idle/stop path
// all converge here, and the planner's own Release is idempotent. A planner
// without the seam is never touched.
func (ts *turnkeyServer) releaseTurnkeyAgentWarm() {
	if ts == nil {
		return
	}
	ts.agentWarmReleaseOnce.Do(func() {
		if releaser, ok := ts.planner.(turnkeyAgentWarmReleaser); ok {
			releaser.ReleaseStartupWarm()
		}
	})
}
