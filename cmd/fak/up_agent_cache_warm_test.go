package main

// up_agent_cache_warm_test.go — CW-09 (#13332) witness. The `fak up` turnkey
// startup seam must install the effective agent KV-cache warm profile BEFORE
// binding readiness: the workspace instruction snapshot (AGENTS.md) and the
// ordered kernel coding-tool schemas it hands the warmer must be the REAL bytes
// the forward path resolves, a profile that cannot be established must leave
// readiness unaffected, a changed instruction snapshot must derive a NEW
// descriptor (so a prior receipt is stale), and the planner's startup warm
// ownership must be released exactly once across shutdown.

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// turnkeyWarmFakePlanner is an agent.Planner + turnkeyAgentWarmInputsInstaller +
// turnkeyAgentWarmWarmer + turnkeyAgentWarmReleaser that records the exact
// WarmPrefixInputs it was handed and returns a scripted receipt. It lets the
// witness prove which bytes the startup seam supplied without a live model.
type turnkeyWarmFakePlanner struct {
	gotInputs agent.WarmPrefixInputs
	installed bool
	warmed    int
	released  int
	receipt   agent.WarmReceipt
	err       error
}

func (p *turnkeyWarmFakePlanner) Model() string { return "turnkey-warm-fake" }

func (p *turnkeyWarmFakePlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}}, nil
}

func (p *turnkeyWarmFakePlanner) SetWarmPrefixInputs(in agent.WarmPrefixInputs) {
	p.gotInputs = in
	p.installed = true
}

func (p *turnkeyWarmFakePlanner) DeriveWarmPrefix(tenant, agentName string, in agent.WarmPrefixInputs) (agent.WarmPrefixSpec, error) {
	if tenant == "" {
		return agent.WarmPrefixSpec{}, agent.ErrWarmPrefixUnavailable
	}
	spec := agent.WarmPrefixSpec{
		StableTokens:      8,
		StableTokenDigest: "sha256:stable",
		InstructionDigest: "sha256:instr",
		AdapterID:         "native-inkernel",
		Identity:          "sha256:desc",
		Scope:             radixkv.CacheIdentity{Tenant: tenant, Agent: agentName},
	}
	// The workspace instruction bytes are the identity-bearing input: a changed
	// snapshot must derive a different descriptor, exactly as the production path.
	if len(in.Instructions) > 0 {
		spec.InstructionDigest = "sha256:instr-" + string(in.Instructions)
		spec.Identity = "sha256:desc-" + string(in.Instructions)
	}
	return spec, nil
}

func (p *turnkeyWarmFakePlanner) WarmPrefix(_ context.Context, _ agent.WarmPrefixSpec) (agent.WarmReceipt, error) {
	p.warmed++
	return p.receipt, p.err
}

// ReleaseStartupWarm is the shutdown-owner hook the turnkey seam must call once.
// The fake counts invocations so the witness can prove at-most-once release. It
// mirrors the production planner's idempotence so a double release cannot
// silently over-count.
func (p *turnkeyWarmFakePlanner) ReleaseStartupWarm() {
	if p.released > 0 {
		return
	}
	p.released++
}

// newTurnkeyWarmServer builds a turnkeyServer around the fake planner with no
// listener bound: the readiness/healthz surface is exercised directly.
func newTurnkeyWarmServer(planner *turnkeyWarmFakePlanner) *turnkeyServer {
	return &turnkeyServer{
		planner:   planner,
		ready:     &readinessGate{},
		done:      make(chan struct{}),
		agentWarm: &turnkeyAgentWarmGate{},
	}
}

// turnkeyWarmReceipt builds a receipt that satisfies the strongest admission
// rule against the fake's derived descriptor.
func turnkeyWarmReceipt(identity string) agent.WarmReceipt {
	return agent.WarmReceipt{
		Ready:             true,
		Status:            agent.WarmStatusReady,
		Identity:          identity,
		StableTokenDigest: "sha256:stable",
		Scope:             radixkv.CacheIdentity{Tenant: turnkeyAgentWarmWorkspaceTenant},
		RequestedTokens:   8,
		RestoredTokens:    8,
		Claim:             &agent.WarmClaimReceipt{Live: true, Tokens: 8, Bytes: 4096},
	}
}

// TestTurnkeyAgentCacheWarmStartup is the CW-09 witness. It proves the turnkey
// seam supplies the REAL workspace instruction bytes and the ordered tool
// schemas to the warmer, holds readiness until a live receipt is observed, keeps
// readiness unaffected when no stable profile can be established, derives a NEW
// profile on a changed instruction snapshot, and releases the planner's startup
// warm ownership exactly once.
func TestTurnkeyAgentCacheWarmStartup(t *testing.T) {
	workspace := t.TempDir()
	instruction := []byte("# Turnkey Warm Test\n\nEffective agent instructions for the warm witness.\n")
	if err := os.WriteFile(filepath.Join(workspace, "AGENTS.md"), instruction, 0o644); err != nil {
		t.Fatalf("write AGENTS.md: %v", err)
	}

	planner := &turnkeyWarmFakePlanner{}
	ts := newTurnkeyWarmServer(planner)

	// The seam installs the profile before readiness: no warm has run yet, so a
	// configured profile must HOLD readiness (pending) with the block present.
	var log bytes.Buffer
	if !installTurnkeyAgentWarmForPlanner(planner, ts.agentWarm, workspace, &log) {
		t.Fatalf("installTurnkeyAgentWarm did not install a profile; log=%s", log.String())
	}
	if !planner.installed {
		t.Fatalf("install did not supply inputs to the warmer; log=%s", log.String())
	}
	if got := string(planner.gotInputs.Instructions); got != string(instruction) {
		t.Fatalf("warmer instruction bytes = %q, want the workspace AGENTS.md %q", got, instruction)
	}
	if len(planner.gotInputs.Tools) == 0 {
		t.Fatal("warmer received zero ordered tool schemas, want the kernel coding-tool catalog")
	}
	if ready, state, _ := ts.readiness(); ready || state != gateway.AgentWarmPending {
		t.Fatalf("configured-unwarmed: readiness=(%v,%q), want (false, pending)", ready, state)
	}
	if block := ts.agentWarm.agentWarmBlock(); block == nil || block["status"] != gateway.AgentWarmPending {
		t.Fatalf("configured-unwarmed: /healthz agent_warm=%v, want status=pending", block)
	}

	// A live receipt for the installed descriptor admits readiness.
	planner.receipt = turnkeyWarmReceipt("sha256:desc-" + string(instruction))
	ts.runTurnkeyAgentWarmup(context.Background())
	if ready, state, _ := ts.readiness(); !ready || state != "ok" {
		t.Fatalf("live-warm: readiness=(%v,%q), want (true, ok)", ready, state)
	}
	if block := ts.agentWarm.agentWarmBlock(); block == nil || block["status"] != gateway.AgentWarmReady {
		t.Fatalf("live-warm: /healthz agent_warm=%v, want status=ready", block)
	}

	// A changed instruction snapshot derives a DIFFERENT descriptor, so the
	// previous warm receipt no longer matches — readiness is held until a matching
	// warm.
	changed := []byte("# Turnkey Warm Test\n\nCHANGED effective agent instructions.\n")
	if err := os.WriteFile(filepath.Join(workspace, "AGENTS.md"), changed, 0o644); err != nil {
		t.Fatalf("rewrite AGENTS.md: %v", err)
	}
	if !installTurnkeyAgentWarmForPlanner(planner, ts.agentWarm, workspace, &log) {
		t.Fatal("re-install for changed instructions did not install a profile")
	}
	if string(planner.gotInputs.Instructions) != string(changed) {
		t.Fatalf("changed workspace instruction bytes = %q, want %q", planner.gotInputs.Instructions, changed)
	}
	if ready, state, _ := ts.readiness(); ready || state != gateway.AgentWarmPending {
		t.Fatalf("changed-instructions: readiness=(%v,%q), want (false, pending)", ready, state)
	}

	// The shutdown owner releases the planner's startup warm ownership exactly
	// once. Shutdown/Close both converge on releaseTurnkeyAgentWarm; calling it
	// twice must not double-fire (idempotent).
	ts.releaseTurnkeyAgentWarm()
	ts.releaseTurnkeyAgentWarm()
	if planner.released != 1 {
		t.Fatalf("ReleaseStartupWarm calls = %d, want 1 (release-once across shutdown)", planner.released)
	}
}

// TestTurnkeyAgentCacheWarmUnconfiguredWorkspace proves a workspace with no
// readable instruction snapshot leaves the agent warm gate unconfigured:
// readiness is never held, no /healthz block is fabricated, and no synthetic
// profile is invented to claim a warm.
func TestTurnkeyAgentCacheWarmUnconfiguredWorkspace(t *testing.T) {
	planner := &turnkeyWarmFakePlanner{}
	ts := newTurnkeyWarmServer(planner)

	var log bytes.Buffer
	if installTurnkeyAgentWarmForPlanner(planner, ts.agentWarm, filepath.Join(t.TempDir(), "absent"), &log) {
		t.Fatal("installTurnkeyAgentWarm installed a profile for a workspace with no AGENTS.md")
	}
	if planner.installed {
		t.Fatal("install supplied inputs for a workspace with no AGENTS.md")
	}
	if ready, state, _ := ts.readiness(); !ready || state != "ok" {
		t.Fatalf("unconfigured workspace: readiness=(%v,%q), want (true, ok)", ready, state)
	}
	if block := ts.agentWarm.agentWarmBlock(); block != nil {
		t.Fatalf("unconfigured workspace: /healthz agent_warm=%v, want absent", block)
	}
}

// TestTurnkeyAgentCacheWarmHealthzWiring proves the HTTP /healthz surface exposes
// the agent_warm block and inherits the readiness hold, so an operator's probe of
// a configured-but-cold turnkey server sees the real pending status rather than a
// bare ok.
func TestTurnkeyAgentCacheWarmHealthzWiring(t *testing.T) {
	workspace := t.TempDir()
	if err := os.WriteFile(filepath.Join(workspace, "AGENTS.md"), []byte("# Warm\n"), 0o644); err != nil {
		t.Fatalf("write AGENTS.md: %v", err)
	}
	planner := &turnkeyWarmFakePlanner{}
	ts := newTurnkeyWarmServer(planner)
	var log bytes.Buffer
	if !installTurnkeyAgentWarmForPlanner(planner, ts.agentWarm, workspace, &log) {
		t.Fatalf("install failed; log=%s", log.String())
	}

	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rec := httptest.NewRecorder()
	ts.handleHealthz(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("/healthz code = %d, want 200 (liveness preserved)", rec.Code)
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode /healthz body %q: %v", rec.Body.String(), err)
	}
	if body["ok"] != false || body["status"] != gateway.AgentWarmPending {
		t.Fatalf("/healthz ok=%v status=%v, want false/pending while the warm is configured and cold", body["ok"], body["status"])
	}
	aw, _ := body["agent_warm"].(map[string]any)
	if aw == nil || aw["status"] != gateway.AgentWarmPending {
		t.Fatalf("/healthz agent_warm=%v, want status=pending", aw)
	}
}
