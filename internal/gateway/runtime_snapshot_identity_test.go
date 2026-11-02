package gateway

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type runtimeSnapshotPlanner struct {
	stubPlanner
	onKV              func()
	kvCalls, moeCalls int
}

func (p *runtimeSnapshotPlanner) KVMemoryStats() agent.KVMemoryStats {
	p.kvCalls++
	if p.onKV != nil {
		p.onKV()
	}
	return agent.KVMemoryStats{Enabled: true, ResidentTokens: 7, ResidentBytes: 28, Scope: "host"}
}

func (p *runtimeSnapshotPlanner) MoEResidencyStats() agent.MoEResidencyLedger {
	p.moeCalls++
	return agent.MoEResidencyLedger{Requests: 2, Tokens: 7, Hits: 3}
}

func runtimeSnapshotResponse(t *testing.T, s *Server) (debugVarsResponse, string) {
	t.Helper()
	r := httptest.NewRecorder()
	s.Handler().ServeHTTP(r, httptest.NewRequest(http.MethodGet, "/debug/vars", nil))
	if r.Code != http.StatusOK {
		t.Fatalf("status=%d", r.Code)
	}
	var response debugVarsResponse
	if err := json.Unmarshal(r.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response.RuntimeSnapshot == nil {
		t.Fatal("missing runtime snapshot identity")
	}
	return response, r.Body.String()
}

// fak-test:runtime fast est=20ms lane=default
func TestRuntimeSnapshotIdentityHandler(t *testing.T) {
	s := newTestServer(t)
	p := &runtimeSnapshotPlanner{}
	s.planner = p
	s.SetModelLoadProfile(&ModelLoadProfile{Source: "test", TotalSeconds: 0.1})
	s.agentWarm.mu.Lock()
	s.agentWarm.configured = true
	s.agentWarm.status = "ready"
	s.agentWarm.receipt = &agent.WarmReceipt{Ready: true, Identity: "warm-id", RuntimeGeneration: "generation-a"}
	s.agentWarm.mu.Unlock()
	response, raw := runtimeSnapshotResponse(t, s)
	x := response.RuntimeSnapshot
	if x.State != "OBSERVED" || response.ModelLoad == nil || x.LoadID != response.ModelLoad.LoadID || x.LoadID == "" {
		t.Fatalf("load binding=%+v", x)
	}
	config, err := json.Marshal(s.ScalarConfig())
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(config)
	if x.ConfigSHA256 != hex.EncodeToString(sum[:]) || x.ConfigIdentityMode != "scalar_v1" || x.ConfigEpoch != s.ConfigEpoch() {
		t.Fatalf("config binding=%+v", x)
	}
	if x.Schema != "fak.runtime-snapshot.v1" || x.Source != "/debug/vars" || x.Scope != "current_server_observation" || x.StartedAt.IsZero() || x.CompletedAt.Before(x.StartedAt) {
		t.Fatalf("observation=%+v", x)
	}
	if x.KVState != "OBSERVED" || x.MoEState != "OBSERVED" || x.CounterScope != "process_model_lifetime" || p.kvCalls != 1 || p.moeCalls != 1 {
		t.Fatalf("samples=%+v kv=%d moe=%d", x, p.kvCalls, p.moeCalls)
	}
	if response.KVMemory == nil || response.KVMemory.ResidentTokens != 7 || response.MoEResidency == nil || response.MoEResidency.Hits != 3 {
		t.Fatal("metadata changed existing reporter samples")
	}
	if x.Warm == nil || x.Warm.Identity != "warm-id" || x.Warm.RuntimeGeneration != "generation-a" || !x.Warm.Ready {
		t.Fatalf("warm=%+v", x.Warm)
	}
	if x.Warm.LoadBinding != "UNKNOWN" || x.Warm.LoadBindingReason != "warm_receipt_has_no_load_id" || x.Warm.RuntimeGeneration == x.LoadID {
		t.Fatal("independent warm generation was promoted to load-bound evidence")
	}
	// No request body, prompt, tools, token IDs or complete warm receipt is exported.
	for _, forbidden := range []string{`"stable_tokens"`, `"token_ids"`, `"tool_policy_digest"`, `"requested_tokens"`} {
		if strings.Contains(raw, forbidden) {
			t.Fatalf("unexpected payload field %s", forbidden)
		}
	}
}

// fak-test:runtime fast est=40ms lane=default
func TestRuntimeSnapshotIdentityReplacement(t *testing.T) {
	for _, mode := range []string{"load", "clear", "config", "config_aba"} {
		t.Run(mode, func(t *testing.T) {
			s := newTestServer(t)
			s.SetModelLoadProfile(&ModelLoadProfile{Source: "before"})
			before := s.modelLoadProfile().LoadID
			p := &runtimeSnapshotPlanner{}
			p.onKV = func() {
				switch mode {
				case "load":
					s.SetModelLoadProfile(&ModelLoadProfile{Source: "after"})
				case "clear":
					s.SetModelLoadProfile(nil)
				case "config", "config_aba":
					original := *s.ScalarConfig()
					changed := original
					changed.LogLevel = "debug"
					if _, err := s.SetScalarConfig(changed); err != nil {
						t.Fatal(err)
					}
					if mode == "config_aba" {
						if _, err := s.SetScalarConfig(original); err != nil {
							t.Fatal(err)
						}
					}
				}
			}
			s.planner = p
			response, _ := runtimeSnapshotResponse(t, s)
			x := response.RuntimeSnapshot
			if x.State != "UNKNOWN" || x.LoadID != "" || x.Reason == "" {
				t.Fatalf("replacement accepted=%+v", x)
			}
			if response.ModelLoad == nil || response.ModelLoad.LoadID != before {
				t.Fatal("model-load sample resampled")
			}
		})
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestRuntimeSnapshotIdentityMissingSources(t *testing.T) {
	s := newTestServer(t)
	s.planner = stubPlanner{}
	response, _ := runtimeSnapshotResponse(t, s)
	x := response.RuntimeSnapshot
	if x.State != "UNKNOWN" || x.LoadID != "" || x.Reason != "load_identity_unavailable" {
		t.Fatalf("missing load=%+v", x)
	}
	if x.KVState != "UNKNOWN" || x.MoEState != "UNKNOWN" || x.Warm != nil || response.KVMemory != nil || response.MoEResidency != nil {
		t.Fatalf("fabricated reporters=%+v", x)
	}
	s.SetModelLoadProfile(&ModelLoadProfile{})
	response, _ = runtimeSnapshotResponse(t, s)
	if response.RuntimeSnapshot.State != "OBSERVED" || response.RuntimeSnapshot.KVState != "UNKNOWN" || response.RuntimeSnapshot.MoEState != "UNKNOWN" {
		t.Fatal("binding promoted unsupported reporters")
	}
}
