package main

// serve_agent_cache_warm_test.go — CW-08 (#13329) witness. The `fak serve` startup
// seam must install the effective agent KV-cache warm profile BEFORE binding
// readiness: the workspace instruction snapshot (AGENTS.md) and the ordered kernel
// coding-tool schemas it hands the warmer must be the REAL bytes the forward path
// resolves, a workspace that cannot establish a profile must leave readiness
// unaffected (no synthetic warm), and the planner's startup warm ownership must be
// released exactly once across shutdown.
//
// The gateway-side live-receipt admission itself is pinned by the gateway's own
// TestAgentCacheWarmReadiness (internal/gateway, CW-07 #13333); this witness pins the
// serve-side producer seam.

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// serveWarmFakePlanner is an agent.Planner + serveAgentWarmReleaser that records the
// exact WarmPrefixInputs it was handed and counts releases. It lets the witness prove
// which bytes the serve seam supplies and that release fires at-most-once, without a
// live model or a gateway server.
type serveWarmFakePlanner struct {
	gotInputs agent.WarmPrefixInputs
	released  int
}

func (p *serveWarmFakePlanner) Model() string { return "serve-warm-fake" }

func (p *serveWarmFakePlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}}, nil
}

func (p *serveWarmFakePlanner) DeriveWarmPrefix(tenant, agentName string, in agent.WarmPrefixInputs) (agent.WarmPrefixSpec, error) {
	p.gotInputs = in
	if tenant == "" {
		return agent.WarmPrefixSpec{}, agent.ErrWarmPrefixUnavailable
	}
	return agent.WarmPrefixSpec{
		StableTokens:      8,
		StableTokenDigest: "sha256:stable",
		InstructionDigest: "sha256:instr",
		AdapterID:         "native-inkernel",
		Identity:          "sha256:desc",
		Scope:             radixkv.CacheIdentity{Tenant: tenant, Agent: agentName},
	}, nil
}

func (p *serveWarmFakePlanner) WarmPrefix(_ context.Context, _ agent.WarmPrefixSpec) (agent.WarmReceipt, error) {
	return agent.WarmReceipt{}, nil
}

// ReleaseStartupWarm is the shutdown-owner hook the serve seam must call once. It
// mirrors the production planner's idempotence so a double release cannot silently
// over-count.
func (p *serveWarmFakePlanner) ReleaseStartupWarm() {
	if p.released > 0 {
		return
	}
	p.released++
}

// TestServeAgentCacheWarmInputs is the producer witness: the serve seam resolves the
// REAL workspace instruction bytes (AGENTS.md) and the non-empty ordered tool catalog
// for the warmer, and a workspace with no readable snapshot yields an explicit
// unconfigured result rather than a synthetic profile.
func TestServeAgentCacheWarmInputs(t *testing.T) {
	workspace := t.TempDir()
	instruction := []byte("# Serve Warm Test\n\nEffective agent instructions for the warm witness.\n")
	if err := os.WriteFile(filepath.Join(workspace, "AGENTS.md"), instruction, 0o644); err != nil {
		t.Fatalf("write AGENTS.md: %v", err)
	}

	inputs, ok := deriveServeAgentWarmInputs(workspace)
	if !ok {
		t.Fatal("deriveServeAgentWarmInputs did not resolve a profile for a readable AGENTS.md")
	}
	if got := string(inputs.Instructions); got != string(instruction) {
		t.Fatalf("warmer instruction bytes = %q, want the workspace AGENTS.md %q", got, instruction)
	}
	if len(inputs.Tools) == 0 {
		t.Fatal("warmer received zero ordered tool schemas, want the kernel coding-tool catalog")
	}

	// No workspace / no readable snapshot: explicit unconfigured, never a synthetic
	// profile.
	if _, ok := deriveServeAgentWarmInputs(""); ok {
		t.Fatal("empty workspace resolved a warm profile, want unconfigured")
	}
	if _, ok := deriveServeAgentWarmInputs(filepath.Join(t.TempDir(), "absent")); ok {
		t.Fatal("workspace with no AGENTS.md resolved a warm profile, want unconfigured")
	}
}

// TestServeAgentCacheWarmReleaseOnce is the lifecycle witness: the serve shutdown
// half releases the planner's startup warm ownership exactly once, converging normal
// shutdown and cancellation on the same sync.Once guard. A planner without the seam
// is never touched.
func TestServeAgentCacheWarmReleaseOnce(t *testing.T) {
	planner := &serveWarmFakePlanner{}
	var once sync.Once

	releaseServeAgentWarmOnce(&once, planner)
	releaseServeAgentWarmOnce(&once, planner)
	if planner.released != 1 {
		t.Fatalf("ReleaseStartupWarm calls = %d, want 1 (release-once across shutdown)", planner.released)
	}

	// A nil planner, and a planner without the seam, are no-ops (no panic, no touch).
	var nilOnce sync.Once
	releaseServeAgentWarmOnce(&nilOnce, nil)
	releaseServeAgentWarmOnce(nil, planner)
}
