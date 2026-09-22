package main

// serve_agent_cache_warm_e2e_test.go — CW-14 (#13327) witness. The CW-08 (#13329)
// producer witness pins the serve seam's INPUTS and release-once lifecycle with a
// fake planner and no server; the gateway's CW-07 (#13333) witness pins the readiness
// ADMISSION rule. Neither proves the two halves meet: that a `fak serve` which
// configured and materialized an agent warm admits its FIRST REAL REQUEST through the
// real handler against a warm prefix (and holds it, with a closed reason, after a
// profile change invalidates the warm).
//
// This witness closes that gap end to end through the PRODUCTION seam: it builds a
// real gateway.Server with a real HTTP handler, installs the warm profile through
// installServeAgentWarm (the exact call `fak serve` makes before binding its
// listener), materializes it through runServeAgentWarmup (the exact call the serve
// startup goroutine makes alongside RunWarmup), and then sends the first natural
// request through the real /v1/chat/completions handler. It distinguishes PRIME work
// (DeriveWarmPrefix + WarmPrefix, done once before the first request) from DEMAND
// work (Complete, done by the served turn), so a "warm" that is actually a request-
// time re-prime cannot pass.
//
// [SW-VERIFIED] software logic only: the warmer is an in-process fake, so this proves
// the serve/gateway wiring and the cold-vs-warm admission shape, NOT physical cache
// residency or a throughput number. The live Strix first-request witness remains with
// fak#13326 and physical criteria stay unchecked.

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// serveWarmE2EPlanner is a real agent.Planner AND a gateway.AgentWarmWarmer. It counts
// PRIME work (DeriveWarmPrefix + WarmPrefix) separately from DEMAND work (Complete),
// so the witness can prove the first request is admitted against a prefix primed
// BEFORE it, never re-primed on its own critical path. A scripted receipt drives every
// admission shape (ready+live claim, a stale/mismatched identity) with no live model.
type serveWarmE2EPlanner struct {
	mu sync.Mutex

	deriveCalls int
	warmCalls   int
	complete    int

	// identity is the descriptor identity DeriveWarmPrefix folds. A change forces the
	// gateway's receipt-identity match to miss, the invalidated-profile case.
	identity string
	// receipt is returned by WarmPrefix. Its Identity must match the DERIVED descriptor
	// to admit readiness.
	receipt agent.WarmReceipt
	// warmErr, when set, is returned by WarmPrefix (unsupported / backend error).
	warmErr error
}

func (p *serveWarmE2EPlanner) Model() string { return "serve-e2e-warm" }

func (p *serveWarmE2EPlanner) Complete(_ context.Context, msgs []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.mu.Lock()
	p.complete++
	p.mu.Unlock()
	// Echo a deterministic assistant turn so the served request has a stable,
	// comparable body across the cold and warm cases.
	last := ""
	if n := len(msgs); n > 0 {
		last = msgs[n-1].Content
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "warm:" + last}}, nil
}

func (p *serveWarmE2EPlanner) DeriveWarmPrefix(tenant, agentName string, _ agent.WarmPrefixInputs) (agent.WarmPrefixSpec, error) {
	p.mu.Lock()
	p.deriveCalls++
	identity := p.identity
	p.mu.Unlock()
	if tenant == "" {
		return agent.WarmPrefixSpec{}, agent.ErrWarmPrefixUnavailable
	}
	return agent.WarmPrefixSpec{
		StableTokens:      16,
		StableTokenDigest: "sha256:stable",
		InstructionDigest: "sha256:instr",
		AdapterID:         "native-inkernel",
		Identity:          identity,
		Scope:             radixkv.CacheIdentity{Tenant: tenant, Agent: agentName},
	}, nil
}

func (p *serveWarmE2EPlanner) WarmPrefix(_ context.Context, _ agent.WarmPrefixSpec) (agent.WarmReceipt, error) {
	p.mu.Lock()
	p.warmCalls++
	r, err := p.receipt, p.warmErr
	p.mu.Unlock()
	return r, err
}

func (p *serveWarmE2EPlanner) counts() (derive, warm, demand int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.deriveCalls, p.warmCalls, p.complete
}

// readyE2EWarmReceipt is a receipt that satisfies the strongest admission rule for the
// given descriptor identity: matching identity, ready status, a full restore to the
// stable boundary, and a live residency claim.
func readyE2EWarmReceipt(identity string) agent.WarmReceipt {
	return agent.WarmReceipt{
		Ready:             true,
		Status:            agent.WarmStatusReady,
		Identity:          identity,
		StableTokenDigest: "sha256:stable",
		Scope:             radixkv.CacheIdentity{Tenant: serveAgentWarmWorkspaceTenant},
		RequestedTokens:   16,
		RestoredTokens:    16,
		Claim:             &agent.WarmClaimReceipt{Live: true, Tokens: 16, Bytes: 4096},
	}
}

// postChat sends one natural chat completion through the real server handler and
// returns the status code and decoded body. A non-2xx is returned, not fatal — the
// gated (503) and admitted (200) cases are both assertions the witness makes.
func postChat(t *testing.T, ts *httptest.Server, content string) (int, map[string]any) {
	t.Helper()
	payload := []byte(`{"model":"serve-e2e-warm","messages":[{"role":"user","content":"` + content + `"}]}`)
	res, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(payload))
	if err != nil {
		t.Fatalf("POST /v1/chat/completions: %v", err)
	}
	defer res.Body.Close()
	raw, _ := io.ReadAll(res.Body)
	var body map[string]any
	_ = json.Unmarshal(raw, &body)
	return res.StatusCode, body
}

// getHealthz serves one real /healthz request through the server handler and returns
// the agent_warm block (nil when absent).
func getHealthz(t *testing.T, ts *httptest.Server) (int, map[string]any) {
	t.Helper()
	res, err := http.Get(ts.URL + "/healthz")
	if err != nil {
		t.Fatalf("GET /healthz: %v", err)
	}
	defer res.Body.Close()
	var body map[string]any
	if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
		t.Fatalf("decode /healthz: %v", err)
	}
	aw, _ := body["agent_warm"].(map[string]any)
	return res.StatusCode, aw
}

// TestAgentCacheWarmFirstRealRequest is the CW-14 (#13327) e2e witness. It proves the
// serve startup-to-first-request path end to end on the PRODUCTION seam: a request
// arriving before the materialized agent warm is HELD (503, closed reason); the first
// real request after readiness is ADMITTED (200) against a warm primed BEFORE it (no
// demand-time re-prime); and a changed profile that invalidates the warm is TRUTHFULLY
// missed (readiness held again), never silently served warm.
func TestAgentCacheWarmFirstRealRequest(t *testing.T) {
	// A real workspace whose AGENTS.md seeds the warm inputs — the same bytes the
	// forward path encodes (deriveServeAgentWarmInputs reads this file).
	workspace := t.TempDir()
	if err := os.WriteFile(filepath.Join(workspace, "AGENTS.md"), []byte("# Serve E2E Warm\n\nEffective instructions.\n"), 0o644); err != nil {
		t.Fatalf("write AGENTS.md: %v", err)
	}

	planner := &serveWarmE2EPlanner{identity: "sha256:desc-1"}
	srv, err := gateway.New(gateway.Config{Model: "serve-e2e-warm"})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	srv.SetPlanner(planner)
	srv.MarkWarmupComplete(0) // backend LOADED; the agent-warm half is still the gate.

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// (1) COLD / pre-warm: install the profile through the production seam. Readiness
	// must be HELD from the very first probe (agent_warm_pending), and the first real
	// request must be refused 503 without doing any demand work.
	if !installServeAgentWarm(srv, workspace, io.Discard) {
		t.Fatal("installServeAgentWarm did not arm a profile for a readable AGENTS.md")
	}
	if code, aw := getHealthz(t, ts); code != http.StatusServiceUnavailable || aw == nil || aw["status"] != gateway.AgentWarmPending {
		t.Fatalf("pre-warm /healthz = code %d agent_warm %v, want 503 status=pending", code, aw)
	}
	code, body := postChat(t, ts, "hello")
	if code != http.StatusServiceUnavailable {
		t.Fatalf("pre-warm first request code = %d body %v, want 503 (agent warm holds readiness)", code, body)
	}
	// installServeAgentWarm derived the descriptor (derive>=1) but has NOT materialized
	// it yet (WarmPrefix belongs to runServeAgentWarmup), and the held request did zero
	// demand work — no token was generated against a cold prefix the serve promised to
	// warm.
	if derive, warm, demand := planner.counts(); derive == 0 || warm != 0 || demand != 0 {
		t.Fatalf("pre-warm counts derive=%d warm=%d demand=%d, want descriptor derived, not yet materialized, no demand work", derive, warm, demand)
	}

	// (2) MATCHING PREFIX: materialize the warm through the production startup half,
	// then send the FIRST real request. Readiness admits (200 status=ready) and the
	// request is served against the prefix primed BEFORE it — demand work happens
	// exactly once and no re-prime occurs on the request path.
	planner.receipt = readyE2EWarmReceipt("sha256:desc-1")
	runServeAgentWarmup(context.Background(), srv)
	if code, aw := getHealthz(t, ts); code != http.StatusOK || aw == nil || aw["status"] != gateway.AgentWarmReady {
		t.Fatalf("warm /healthz = code %d agent_warm %v, want 200 status=ready", code, aw)
	}
	_, warmPrime, warmDemand := planner.counts()
	code, body = postChat(t, ts, "hi")
	if code != http.StatusOK {
		t.Fatalf("first real request after warm code = %d body %v, want 200", code, body)
	}
	derive2, prime2, demand2 := planner.counts()
	if prime2 != warmPrime {
		t.Fatalf("first real request re-primed the cache: warm calls %d -> %d (demand must not re-prime)", warmPrime, prime2)
	}
	if demand2 != warmDemand+1 {
		t.Fatalf("demand (Complete) calls = %d, want exactly one served turn (%d)", demand2, warmDemand+1)
	}
	if derive2 != 1 {
		t.Fatalf("DeriveWarmPrefix calls = %d, want 1 (a demand request must not re-derive the descriptor)", derive2)
	}

	// (3) INVALIDATED PROFILE: a changed descriptor (different folded identity) is a
	// TRUTHFUL miss. Reconfiguring returns readiness to pending (the prior warm receipt
	// is cleared, never reused), and a request arriving before the new profile is
	// materialized is held again — a stale warm can never report ready.
	planner.identity = "sha256:desc-2"
	if _, err := srv.SetAgentWarmProfile(gateway.AgentWarmProfile{
		Tenant: serveAgentWarmWorkspaceTenant,
	}); err != nil {
		t.Fatalf("reconfigure SetAgentWarmProfile: %v", err)
	}
	if code, aw := getHealthz(t, ts); code != http.StatusServiceUnavailable || aw == nil || aw["status"] != gateway.AgentWarmPending {
		t.Fatalf("invalidated /healthz = code %d agent_warm %v, want 503 status=pending", code, aw)
	}
	if code, body = postChat(t, ts, "again"); code != http.StatusServiceUnavailable {
		t.Fatalf("invalidated first request code = %d body %v, want 503 (stale warm must not serve)", code, body)
	}

	// A receipt whose identity does NOT match the reconfigured descriptor degrades with
	// a closed reason rather than reporting a readiness the next request cannot use.
	planner.receipt = readyE2EWarmReceipt("sha256:desc-1") // stale identity
	runServeAgentWarmup(context.Background(), srv)
	if code, aw := getHealthz(t, ts); aw == nil || aw["reason"] != "identity_mismatch" {
		t.Fatalf("mismatched warm /healthz = code %d agent_warm %v, want reason=identity_mismatch", code, aw)
	}
	if code, body = postChat(t, ts, "again"); code != http.StatusServiceUnavailable {
		t.Fatalf("mismatched first request code = %d body %v, want 503 (identity mismatch holds readiness)", code, body)
	}
}
