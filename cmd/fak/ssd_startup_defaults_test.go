package main

import (
	"bytes"
	"context"
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// fak-test:runtime fast est=2s lane=default
func TestTurnkeyStartupRestoresDefaultDiskWarmAndReportsHealth(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	artifact := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(artifact, []byte("stable model artifact fixture"), 0o600); err != nil {
		t.Fatal(err)
	}
	cacheDir := t.TempDir()
	modelID := "synthetic-ssd-startup"
	cfg := fakmodel.Config{
		HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 320, RMSNormEps: 1e-5, RopeTheta: 10000, EOSTokenID: 319,
	}
	inputs := agent.WarmPrefixInputs{
		Instructions: []byte("Follow the repository conventions."),
		SystemBlocks: [][]byte{[]byte("Default-deny capability floor.")},
		KVLayout:     "fp32",
		AdapterID:    "native-inkernel",
	}

	start := func(t *testing.T) (*turnkeyNativeResources, *gateway.Server, agent.WarmReceipt, map[string]any, int) {
		t.Helper()
		before := enginestep.Default.Snapshot(0, "")
		model := fakmodel.NewSynthetic(cfg)
		model.Quantize()
		tok := testProbeTokenizer(t)
		deps := mtpStatusTestDeps()
		deps.resolveMetal = func() (serveMetalDecision, error) { return serveMetalDecision{}, nil }
		deps.loadModel = func(string, compute.Backend, int, *serveFitBudget) (*fakmodel.Model, bool, *gateway.ModelLoadProfile) {
			return model, false, nil
		}
		deps.loadTokenizer = func(string) (*tokenizer.Tokenizer, bool) { return tok, true }
		deps.newPlanner = func(m *fakmodel.Model, tok *tokenizer.Tokenizer, id string, q4k bool, backend compute.Backend, metal bool, contextTokens int) *agent.InKernelPlanner {
			return agent.NewInKernelPlannerWithConfig(m, tok, id, q4k, backend, metal, agent.InKernelPlannerConfig{ContextTokens: contextTokens})
		}
		deps.resolveMTPStatus = func(*turnkeyMTPQualificationResult, *agent.InKernelPlanner) (bool, string) {
			return false, string(turnkeyMTPNoEligibleContext)
		}
		resources, err := loadTurnkeyNativeResourcesWith(context.Background(), artifact, modelID, 2048, deps)
		if err != nil {
			t.Fatalf("loadTurnkeyNativeResourcesWith: %v", err)
		}
		resources.Planner.SetWarmDiskConfig(agent.WarmDiskConfig{Dir: cacheDir, MaxEntries: 4, MaxBytes: 8 << 20})

		srv, err := gateway.New(gateway.Config{Model: modelID})
		if err != nil {
			t.Fatalf("gateway.New: %v", err)
		}
		srv.SetPlanner(resources.Planner)
		srv.MarkWarmupComplete(0)
		if _, err := srv.SetAgentWarmProfile(gateway.AgentWarmProfile{Tenant: serveAgentWarmWorkspaceTenant, Inputs: inputs}); err != nil {
			t.Fatalf("SetAgentWarmProfile: %v", err)
		}
		receipt, err := srv.RunAgentWarmup(context.Background())
		if err != nil {
			t.Fatalf("RunAgentWarmup: %v (status=%s reason=%s requested=%d)", err, receipt.Status, receipt.Reason, receipt.RequestedTokens)
		}
		rec := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("GET /healthz status=%d body=%q", rec.Code, rec.Body.String())
		}
		var health map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &health); err != nil {
			t.Fatalf("decode /healthz: %v", err)
		}
		after := enginestep.Default.Snapshot(0, "")
		return resources, srv, receipt, health, int(after.PrefillTokens - before.PrefillTokens)
	}

	first, _, written, _, coldPrefillTokens := start(t)
	if written.Disk == nil || written.Disk.Outcome != agent.WarmDiskPersisted || written.PrefilledTokens != written.RequestedTokens {
		t.Fatalf("first startup disk=%v prefilled=%d requested=%d, want persisted full prefill", diskOutcome(written), written.PrefilledTokens, written.RequestedTokens)
	}
	if coldPrefillTokens == 0 {
		t.Fatal("first startup engine-step prefill delta=0, want cold prefill execution")
	}
	first.closeModel = func() error { return nil }
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}

	second, restoredServer, restored, health, restoredPrefillTokens := start(t)
	second.closeModel = func() error { return nil }
	t.Cleanup(func() { _ = second.Close() })
	if restored.Disk == nil || restored.Disk.Outcome != agent.WarmDiskRestored || restored.PrefilledTokens != 0 {
		t.Fatalf("second startup disk=%v prefilled=%d, want restored with zero prefill", diskOutcome(restored), restored.PrefilledTokens)
	}
	if restoredPrefillTokens != 0 {
		t.Fatalf("second startup engine-step prefill delta=%d, want 0 after disk restore", restoredPrefillTokens)
	}
	aw, _ := health["agent_warm"].(map[string]any)
	disk, _ := aw["disk"].(map[string]any)
	readBytes, _ := disk["read_bytes"].(float64)
	if aw["status"] != gateway.AgentWarmReady || aw["prefilled_tokens"] != float64(0) ||
		disk["outcome"] != string(agent.WarmDiskRestored) || disk["tier"] != string(agent.WarmDiskTierLocalSSD) || readBytes <= 0 {
		t.Fatalf("/healthz agent_warm=%v, want ready local-SSD restore with zero prefill", aw)
	}

	chatBody, err := json.Marshal(map[string]any{
		"model": modelID,
		"messages": []map[string]string{
			{"role": "system", "content": string(inputs.Instructions)},
			{"role": "system", "content": string(inputs.SystemBlocks[0])},
			{"role": "user", "content": "Continue from the stable startup context."},
		},
		"max_tokens": 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	matchedBefore := enginestep.Default.Snapshot(0, "").PrefixMatched
	chat := httptest.NewRecorder()
	restoredServer.Handler().ServeHTTP(chat, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(chatBody)))
	if chat.Code != http.StatusOK {
		t.Fatalf("anonymous demand status=%d body=%q", chat.Code, chat.Body.String())
	}
	matchedAfter := enginestep.Default.Snapshot(0, "").PrefixMatched
	if matchedAfter <= matchedBefore {
		t.Fatalf("anonymous demand prefix-match delta=%d, want positive reuse from startup warm", matchedAfter-matchedBefore)
	}

	named, err := gateway.New(gateway.Config{Model: modelID, KeyPrincipals: map[string]string{"named-secret": "tenant-other"}})
	if err != nil {
		t.Fatalf("named gateway: %v", err)
	}
	named.SetPlanner(second.Planner)
	named.MarkWarmupComplete(0)
	namedBefore := enginestep.Default.Snapshot(0, "").PrefixMatched
	namedRequest := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(chatBody))
	namedRequest.Header.Set("Authorization", "Bearer named-secret")
	namedResponse := httptest.NewRecorder()
	named.Handler().ServeHTTP(namedResponse, namedRequest)
	if namedResponse.Code != http.StatusOK {
		t.Fatalf("named demand status=%d body=%q", namedResponse.Code, namedResponse.Body.String())
	}
	if delta := enginestep.Default.Snapshot(0, "").PrefixMatched - namedBefore; delta != 0 {
		t.Fatalf("named principal inherited anonymous startup prefix: matched delta=%d", delta)
	}
}

func diskOutcome(receipt agent.WarmReceipt) string {
	if receipt.Disk == nil {
		return ""
	}
	return string(receipt.Disk.Outcome)
}

// fak-test:runtime fast est=100ms lane=default
func TestServeWarmDiskModelIdentityCallerWiring(t *testing.T) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "serve.go", nil, 0)
	if err != nil {
		t.Fatalf("parse serve.go: %v", err)
	}
	var build *ast.FuncDecl
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if ok && fn.Name.Name == "buildGateway" {
			build = fn
			break
		}
	}
	if build == nil {
		t.Fatal("serveRuntime.buildGateway not found")
	}
	assignedIdentity := false
	passedPlannerConfig := false
	ast.Inspect(build.Body, func(node ast.Node) bool {
		switch n := node.(type) {
		case *ast.AssignStmt:
			for i, lhs := range n.Lhs {
				if i < len(n.Rhs) && selectorPath(lhs) == "nativePlannerConfig.WarmDiskModelIdentity" && selectorPath(n.Rhs[i]) == "rt.warmDiskModelIdentity" {
					assignedIdentity = true
				}
			}
		case *ast.KeyValueExpr:
			key, _ := n.Key.(*ast.Ident)
			value, _ := n.Value.(*ast.Ident)
			if key != nil && value != nil && key.Name == "InKernelPlanner" && value.Name == "nativePlannerConfig" {
				passedPlannerConfig = true
			}
		}
		return true
	})
	if !assignedIdentity || !passedPlannerConfig {
		t.Fatalf("buildGateway identity assignment=%t planner config handoff=%t, want both", assignedIdentity, passedPlannerConfig)
	}
}

func selectorPath(expr ast.Expr) string {
	switch n := expr.(type) {
	case *ast.Ident:
		return n.Name
	case *ast.SelectorExpr:
		prefix := selectorPath(n.X)
		if prefix == "" {
			return n.Sel.Name
		}
		return prefix + "." + n.Sel.Name
	default:
		return ""
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestWarmDiskModelArtifactIdentityTracksIncarnation(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(path, []byte("aaaa"), 0o600); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	mtime := info.ModTime()
	firstStamp := stampModelArtifact(path)
	first := firstStamp.identityAfterLoad(path)
	if first == "" || firstStamp.identityAfterLoad(path) != first {
		t.Fatal("unchanged artifact identity is empty or unstable")
	}

	if err := os.WriteFile(path, []byte("bbbb"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, mtime, mtime); err != nil {
		t.Fatal(err)
	}
	if got := firstStamp.identityAfterLoad(path); got != "" {
		t.Fatalf("stale stamp survived same-size overwrite with restored mtime: %q", got)
	}
	secondStamp := stampModelArtifact(path)
	second := secondStamp.identityAfterLoad(path)
	if second == "" || second == first {
		t.Fatalf("same-size overwrite identity=%q, want nonempty and different", second)
	}

	replacement := filepath.Join(dir, "replacement.gguf")
	if err := os.WriteFile(replacement, []byte("cccc"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(replacement, mtime, mtime); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(replacement, path); err != nil {
		t.Fatal(err)
	}
	thirdStamp := stampModelArtifact(path)
	third := thirdStamp.identityAfterLoad(path)
	if third == "" || third == second {
		t.Fatalf("equal-size/equal-mtime replacement identity=%q, want nonempty and different", third)
	}
}
