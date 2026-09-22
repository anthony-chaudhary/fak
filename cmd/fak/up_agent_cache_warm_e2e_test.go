package main

// up_agent_cache_warm_e2e_test.go — CW-18 (#13328) witness. CW-09 (#13332) proved
// the `fak up` turnkey seam INSTALLS and MATERIALIZES the effective agent KV-cache
// warm profile before readiness. This leaf proves the OTHER half the turnkey path
// owed: a REAL request through the REAL turnkey HTTP handler actually REUSES that
// prepared prefix — the demand turn matches the stable boundary the warm restored,
// its output equals a cold reference, and once the warm profile is invalidated the
// next request truthfully MISSES rather than reusing a stale warm-ready claim.
//
// The fixture is deliberately reduced (a 2-layer synthetic model + the package's
// byte-level probe tokenizer) and drives the production entrypoints end-to-end:
// installTurnkeyAgentWarmForPlanner -> runTurnkeyAgentWarmup -> handleChatCompletions.
// It adds no benchmark harness and touches no cache backend.

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
	"github.com/anthony-chaudhary/fak/internal/macfit"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// turnkeyWarmE2ECfg is the reduced model config the e2e witness runs the served
// forward on. Small but forward-runnable, with a vocab that covers the probe
// tokenizer's ~259 byte-level ids so the real EncodePrompt boundary can be prefilled.
func turnkeyWarmE2ECfg() fakmodel.Config {
	return fakmodel.Config{
		HiddenSize:       32,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		HeadDim:          8,
		IntermediateSize: 64,
		VocabSize:        320,
		RMSNormEps:       1e-5,
		RopeTheta:        10000,
		EOSTokenID:       -1,
	}
}

// newTurnkeyWarmE2EPlanner builds a reuse-enabled in-kernel planner over a reduced
// synthetic model. The planner owns the radix tree + scoped tree the demand path
// consults, so a scoped warm is visible to a scoped demand request exactly as in
// production.
func newTurnkeyWarmE2EPlanner(t *testing.T, modelID string) *agent.InKernelPlanner {
	t.Helper()
	m := fakmodel.NewSynthetic(turnkeyWarmE2ECfg())
	m.Quantize()
	p := agent.NewInKernelPlannerWithConfig(m, testProbeTokenizer(t), modelID, false, nil, false, agent.InKernelPlannerConfig{})
	if p == nil {
		t.Fatal("planner is nil")
	}
	return p
}

// newTurnkeyWarmE2EServer binds the reduced planner to the real turnkey chat handler.
func newTurnkeyWarmE2EServer(p agent.Planner) *turnkeyServer {
	return &turnkeyServer{
		planner:   p,
		plan:      macfit.TurnkeyProfile{ContextBudgetTokens: 4096, Tier: macfit.ModelTier{ModelID: "synthetic-warm-e2e"}},
		ready:     &readinessGate{},
		done:      make(chan struct{}),
		agentWarm: &turnkeyAgentWarmGate{},
	}
}

// postTurnkeyWarmChat posts one buffered /v1/chat/completions turn through the real
// handler and returns the decoded response. It attaches the SAME ordered tool
// catalog the startup warm derived its stable prefix from, so the demand prompt
// reproduces the stable boundary rather than a tools-less suffix.
func postTurnkeyWarmChat(t *testing.T, srv *turnkeyServer, messages []agent.Message) *gateway.ChatResponse {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"model":      "synthetic-warm-e2e",
		"messages":   messages,
		"tools":      agent.ToolCatalog(),
		"max_tokens": 3,
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	srv.handleChatCompletions(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("/v1/chat/completions code = %d, want 200 (body=%s)", rec.Code, rec.Body.String())
	}
	var out gateway.ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode completion body %q: %v", rec.Body.String(), err)
	}
	return &out
}

// lastMatchedPrefix returns the most recent demand turn's pre-gate radix match — how
// many leading prompt tokens the served turn reused. This is the public turn-tax
// readback (inkernel_turntax.go) and the honest witness that a warm was CONSUMED.
func lastMatchedPrefix(t *testing.T, p *agent.InKernelPlanner) int {
	t.Helper()
	entries := p.TurnTaxDecisions()
	if len(entries) == 0 {
		t.Fatal("planner recorded no turn-tax decision for the demand request")
	}
	return entries[len(entries)-1].Decision.MatchedPrefix
}

// sharedPromptPrefix returns the length of the longest common leading token run
// between the warm's stable prompt and a demand prompt, computed through the SAME
// production encoder the planner uses. It is the honest expected match for a demand
// turn that shares the stable instruction+tool prefix and then diverges on its user
// suffix.
func sharedPromptPrefix(t *testing.T, p *agent.InKernelPlanner, stable, demand []agent.Message) int {
	t.Helper()
	encStable, err := p.EncodePrompt(context.Background(), stable, agent.ToolCatalog())
	if err != nil {
		t.Fatalf("EncodePrompt(stable): %v", err)
	}
	encDemand, err := p.EncodePrompt(context.Background(), demand, agent.ToolCatalog())
	if err != nil {
		t.Fatalf("EncodePrompt(demand): %v", err)
	}
	n := 0
	for n < len(encStable.TokenIDs) && n < len(encDemand.TokenIDs) && encStable.TokenIDs[n] == encDemand.TokenIDs[n] {
		n++
	}
	return n
}

// TestTurnkeyCacheWarmFirstRealRequest is the leaf's named witness (fak#13328). It
// proves the turnkey startup warm CW-09 installed is actually CONSUMED by the first
// real request through the real handler: the demand turn matches the restored stable
// boundary with cold-reference parity, and an invalidated profile yields a truthful
// miss instead of a stale warm-ready claim.
func TestTurnkeyCacheWarmFirstRealRequest(t *testing.T) {
	workspace := t.TempDir()
	write := func(s string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(workspace, "AGENTS.md"), []byte(s), 0o644); err != nil {
			t.Fatalf("write AGENTS.md: %v", err)
		}
	}
	const instruction = "You are the turnkey agent. Follow the repo conventions exactly."
	write(instruction)

	// The real CW-09 install path: resolve the workspace AGENTS.md, hand the warmer
	// the REAL instruction bytes + ordered tool catalog, derive and install the
	// profile on the turnkey gate.
	planner := newTurnkeyWarmE2EPlanner(t, "synthetic-warm-e2e")
	srv := newTurnkeyWarmE2EServer(planner)
	var log bytes.Buffer
	if !installTurnkeyAgentWarmForPlanner(planner, srv.agentWarm, workspace, &log) {
		t.Fatalf("installTurnkeyAgentWarmForPlanner did not install a profile; log=%s", log.String())
	}
	spec, configured := srv.agentWarm.configuration()
	if !configured || spec.StableTokens <= 0 {
		t.Fatalf("agent warm gate not configured with a bounded profile: configured=%v spec=%+v", configured, spec)
	}

	// Readiness is held until the materialized warm produces a live receipt.
	if ready, state, _ := srv.readiness(); ready || state != gateway.AgentWarmPending {
		t.Fatalf("configured-unwarmed readiness = (%v,%q), want (false, pending)", ready, state)
	}
	srv.runTurnkeyAgentWarmup(context.Background())
	if ready, state, _ := srv.readiness(); !ready || state != "ok" {
		t.Fatalf("warmed readiness = (%v,%q), want (true, ok); agent_warm=%v spec=%+v", ready, state, srv.agentWarm.agentWarmBlock(), spec)
	}
	if block := srv.agentWarm.agentWarmBlock(); block == nil || block["status"] != gateway.AgentWarmReady {
		t.Fatalf("/healthz agent_warm=%v, want status=ready after a live receipt", block)
	}

	// The first real request carries the SAME stable instructions as a system turn,
	// then a user suffix. The demand path must MATCH the restored stable boundary.
	stableMsg := agent.Message{Role: agent.RoleSystem, Content: instruction}
	suffix := agent.Message{Role: agent.RoleUser, Content: "continue the task with one short step"}
	got := postTurnkeyWarmChat(t, srv, []agent.Message{stableMsg, suffix})
	if len(got.Choices) == 0 {
		t.Fatal("first real request returned no choices")
	}

	// The warm restore is real only if the demand turn visibly matched the stable
	// boundary the warm materialized. A zero-length match means the request went out
	// UNSCOPED and never saw the turnkey warm. The exact boundary is the longest
	// common prefix of the demand prompt and the warm's stable prompt (the user turn
	// diverges after the shared system+tool prefix), so compute it with the SAME
	// encoder the planner used rather than asserting a guessed constant.
	stableMsgs := []agent.Message{{Role: agent.RoleSystem, Content: instruction}}
	demandMsgs := []agent.Message{stableMsg, suffix}
	wantMatched := sharedPromptPrefix(t, planner, stableMsgs, demandMsgs)
	warmMatched := lastMatchedPrefix(t, planner)
	if warmMatched < wantMatched {
		t.Fatalf("first real request matched %d tokens, want >= the shared stable prefix %d (turnkey warm not consumed)",
			warmMatched, wantMatched)
	}
	if warmMatched == 0 {
		t.Fatal("first real request matched 0 tokens: the turnkey warm was not consumed")
	}

	// Cold-reference parity: a fresh planner (no warm) continuing the same suffix
	// must produce the SAME output, so the warm changed cost, not behavior.
	coldPlanner := newTurnkeyWarmE2EPlanner(t, "synthetic-warm-e2e")
	coldSrv := newTurnkeyWarmE2EServer(coldPlanner)
	coldOut := postTurnkeyWarmChat(t, coldSrv, []agent.Message{stableMsg, suffix})
	if len(coldOut.Choices) == 0 {
		t.Fatal("cold reference returned no choices")
	}
	if coldMatched := lastMatchedPrefix(t, coldPlanner); coldMatched != 0 {
		t.Fatalf("cold reference matched %d tokens, want 0", coldMatched)
	}
	if got.Choices[0].Message.Content != coldOut.Choices[0].Message.Content {
		t.Fatalf("warmed output %q != cold reference %q", got.Choices[0].Message.Content, coldOut.Choices[0].Message.Content)
	}

	// Invalidate the warm profile: the stable instructions change, so the installed
	// profile derives a NEW descriptor and the prior warm receipt is stale. Readiness
	// must return to pending rather than reusing the stale warm-ready claim.
	write(instruction + "\nCHANGED: a new instruction snapshot invalidates the warm.")
	if !installTurnkeyAgentWarmForPlanner(planner, srv.agentWarm, workspace, &log) {
		t.Fatal("re-install for changed instructions did not install a profile")
	}
	if ready, state, _ := srv.readiness(); ready || state != gateway.AgentWarmPending {
		t.Fatalf("changed-profile readiness = (%v,%q), want (false, pending)", ready, state)
	}
}

// TestTurnkeyCacheWarmInvalidatedProfileMisses proves the negative half of the
// contract: once the profile changes, the gate rejects the stale warm-ready claim
// (readiness returns to pending) and a demand request for the NEW, not-yet-warmed
// prefix truthfully MISSES rather than reusing the prior profile's warm.
func TestTurnkeyCacheWarmInvalidatedProfileMisses(t *testing.T) {
	workspace := t.TempDir()
	const original = "Original stable snapshot for the miss witness."
	writeAgents := func(s string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(workspace, "AGENTS.md"), []byte(s), 0o644); err != nil {
			t.Fatalf("write AGENTS.md: %v", err)
		}
	}
	writeAgents(original)

	planner := newTurnkeyWarmE2EPlanner(t, "synthetic-warm-e2e")
	srv := newTurnkeyWarmE2EServer(planner)
	var log bytes.Buffer
	if !installTurnkeyAgentWarmForPlanner(planner, srv.agentWarm, workspace, &log) {
		t.Fatalf("install failed; log=%s", log.String())
	}
	originalSpec, _ := srv.agentWarm.configuration()
	srv.runTurnkeyAgentWarmup(context.Background())

	// The original warm admits readiness.
	if ready, _, _ := srv.readiness(); !ready {
		t.Fatal("original warm did not admit readiness")
	}

	// Change the profile to a completely DIFFERENT stable snapshot. The new
	// descriptor differs, so the prior warm receipt is stale.
	const changed = "A completely different instruction snapshot now."
	writeAgents(changed)
	if !installTurnkeyAgentWarmForPlanner(planner, srv.agentWarm, workspace, &log) {
		t.Fatal("re-install after change did not install a profile")
	}
	newSpec, _ := srv.agentWarm.configuration()
	if newSpec.Identity == originalSpec.Identity {
		t.Fatal("changed instructions derived the same identity, want a new descriptor")
	}
	// The stale warm-ready claim is rejected: readiness is held pending until the new
	// profile's own warm is observed.
	if ready, state, _ := srv.readiness(); ready || state != gateway.AgentWarmPending {
		t.Fatalf("changed-profile readiness = (%v,%q), want (false, pending): a stale warm-ready claim was reused", ready, state)
	}

	// Before materializing the new profile's warm, a demand request for the NEW prefix
	// must NOT reuse the prior profile's warm. The original profile's prefix is still
	// resident, so the only admissible overlap is the trivial ChatML framing the two
	// snapshots share — compute that envelope with the production encoder and require
	// the served match not to exceed it (no instruction-bearing token is reused).
	newDemand := []agent.Message{
		{Role: agent.RoleSystem, Content: changed},
		{Role: agent.RoleUser, Content: "one short step"},
	}
	originalStable := []agent.Message{{Role: agent.RoleSystem, Content: original}}
	staleEnvelope := sharedPromptPrefix(t, planner, originalStable, newDemand)
	postTurnkeyWarmChat(t, srv, newDemand)
	if matched := lastMatchedPrefix(t, planner); matched > staleEnvelope {
		t.Fatalf("un-warmed new profile reused a stale warm: new-prefix match = %d, want <= the framing-only envelope %d (truthful miss)",
			matched, staleEnvelope)
	}
}
