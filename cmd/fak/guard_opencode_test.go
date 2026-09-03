package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/policy"
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

// TestGuardOpenCodeProfileResolution verifies that OpenCode executables are recognized
// and mapped to the OpenAI wire with environment-based repointing.
func TestGuardOpenCodeProfileResolution(t *testing.T) {
	candidates := []string{
		"opencode",
		"opencode.exe",
		"opencode.cmd",
		`C:\Program Files\nodejs\opencode.cmd`,
		"/usr/local/bin/opencode",
	}

	for _, cand := range candidates {
		prof, ok := harnessprofile.Lookup(cand)
		if !ok {
			t.Fatalf("harnessprofile.Lookup(%q) = not ok, want recognized", cand)
		}
		if prof.Wire != harnessprofile.WireOpenAI {
			t.Errorf("Lookup(%q).Wire = %q, want %q", cand, prof.Wire, harnessprofile.WireOpenAI)
		}
		if !prof.HasRepoint(harnessprofile.RepointEnv) {
			t.Errorf("Lookup(%q) missing RepointEnv mechanism", cand)
		}
	}
}

// TestGuardOpenCodeInjectedEnv verifies that guard sets the expected environment variables
// for OpenCode to reach the in-process gateway on the OpenAI wire.
func TestGuardOpenCodeInjectedEnv(t *testing.T) {
	gwURL := "http://127.0.0.1:8137"
	injected := guardInjectedEnv("openai", "", gwURL)

	envMap := make(map[string]string)
	for _, pair := range injected {
		envMap[pair[0]] = pair[1]
	}

	wantOpenAIBase := gwURL + "/v1"
	if got := envMap["OPENAI_BASE_URL"]; got != wantOpenAIBase {
		t.Errorf("OPENAI_BASE_URL = %q, want %q", got, wantOpenAIBase)
	}
	if got := envMap["OPENAI_API_BASE"]; got != wantOpenAIBase {
		t.Errorf("OPENAI_API_BASE = %q, want %q", got, wantOpenAIBase)
	}
	if got := envMap["ANTHROPIC_BASE_URL"]; got != gwURL {
		t.Errorf("ANTHROPIC_BASE_URL = %q, want %q", got, gwURL)
	}
}

// TestGuardOpenCodeToolDialectAdjudication verifies that OpenCode's lowercase tool names
// and camelCase argument structure (filePath) are properly adjudicated by the capability floor.
func TestGuardOpenCodeToolDialectAdjudication(t *testing.T) {
	rt, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		t.Fatalf("ParseRuntime(guardDefaultPolicyJSON) error: %v", err)
	}
	adj := adjudicator.New(rt.Adjudicator)
	res := abi.ActiveResolver()
	if res == nil {
		t.Fatal("no Ref resolver registered (internal/registrations blank import missing)")
	}

	decide := func(tool, argsJSON string) abi.Verdict {
		ref, err := res.Put(context.Background(), []byte(argsJSON))
		if err != nil {
			t.Fatalf("res.Put error: %v", err)
		}
		return adj.Adjudicate(context.Background(), &abi.ToolCall{
			Tool: tool,
			Args: ref,
		})
	}

	cases := []struct {
		name       string
		tool       string
		args       string
		wantKind   abi.VerdictKind
		wantReason abi.ReasonCode
	}{
		// Allowed OpenCode tool calls
		{"benign bash", "bash", `{"command":"echo fak-opencode-ok"}`, abi.VerdictAllow, abi.ReasonNone},
		{"benign read with filePath", "read", `{"filePath":"README.md"}`, abi.VerdictAllow, abi.ReasonNone},
		{"benign in-tree write", "write", `{"filePath":"notes.txt","content":"hello"}`, abi.VerdictAllow, abi.ReasonNone},
		{"benign in-tree edit", "edit", `{"filePath":"notes.txt","oldString":"a","newString":"b"}`, abi.VerdictAllow, abi.ReasonNone},
		{"benign grep", "grep", `{"pattern":"func main"}`, abi.VerdictAllow, abi.ReasonNone},
		{"benign glob", "glob", `{"pattern":"**/*.go"}`, abi.VerdictAllow, abi.ReasonNone},
		{"benign webfetch", "webfetch", `{"url":"https://github.com/anthony-chaudhary/fak"}`, abi.VerdictAllow, abi.ReasonNone},
		{"benign todowrite", "todowrite", `{"todos":[]}`, abi.VerdictAllow, abi.ReasonNone},
		{"benign skill", "skill", `{"name":"agent-readiness"}`, abi.VerdictAllow, abi.ReasonNone},

		// Dangerous bash commands blocked by POLICY_BLOCK
		{"rm -rf blocked", "bash", `{"command":"rm -rf /tmp/danger"}`, abi.VerdictDeny, abi.ReasonPolicyBlock},
		{"sudo blocked", "bash", `{"command":"sudo rm /etc/hosts"}`, abi.VerdictDeny, abi.ReasonPolicyBlock},
		{"fork bomb blocked", "bash", `{"command":":(){ :|:& };:"}`, abi.VerdictDeny, abi.ReasonPolicyBlock},

		// Sensitive repository structures blocked by SELF_MODIFY via filePath
		{"edit .git/config blocked", "edit", `{"filePath":".git/config","oldString":"a","newString":"b"}`, abi.VerdictDeny, abi.ReasonSelfModify},
		{"write .git/hooks blocked", "write", `{"filePath":".git/hooks/pre-commit","content":"#!/bin/sh"}`, abi.VerdictDeny, abi.ReasonSelfModify},
		{"write .env blocked", "write", `{"filePath":".env","content":"KEY=secret"}`, abi.VerdictDeny, abi.ReasonSelfModify},

		// Unlisted tool fails closed under default deny
		{"unregistered tool fails closed", "arbitrary_execution", `{}`, abi.VerdictDeny, abi.ReasonDefaultDeny},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := decide(tc.tool, tc.args)
			if v.Kind != tc.wantKind {
				t.Errorf("%s: got kind %v, want %v", tc.name, v.Kind, tc.wantKind)
			}
			if tc.wantReason != abi.ReasonNone && v.Reason != tc.wantReason {
				t.Errorf("%s: got reason %v, want %v", tc.name, v.Reason, tc.wantReason)
			}
		})
	}
}

// TestGuardOpenCodeNoBypassCredentialSwap verifies that requests with the child placeholder
// are rejected when hitting upstream directly, but succeed when proxied through the gateway.
func TestGuardOpenCodeNoBypassCredentialSwap(t *testing.T) {
	const realKey = "real-secret-key-12345"
	const placeholder = "fak-guard-placeholder"

	// Mock upstream requiring the real key
	upstreamHits := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth := r.Header.Get("Authorization")
		if auth != "Bearer "+realKey {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":{"message":"Invalid API key","code":"invalid_api_key"}}`))
			return
		}
		upstreamHits++
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"chatcmpl-test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"pong"},"finish_reason":"stop"}]}`))
	}))
	defer upstream.Close()

	// 1. Direct call with placeholder MUST fail with 401
	directReq, err := http.NewRequest(http.MethodPost, upstream.URL+"/v1/chat/completions", strings.NewReader(`{"model":"test","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("http.NewRequest direct: %v", err)
	}
	directReq.Header.Set("Authorization", "Bearer "+placeholder)
	directReq.Header.Set("Content-Type", "application/json")

	directResp, err := http.DefaultClient.Do(directReq)
	if err != nil {
		t.Fatalf("direct Do error: %v", err)
	}
	directResp.Body.Close()
	if directResp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("direct placeholder request got status %d, want 401 Unauthorized", directResp.StatusCode)
	}

	// 2. Proxied call with placeholder through a test reverse-proxy simulating fak gateway
	proxyHits := 0
	gatewayProxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		proxyHits++
		clientAuth := r.Header.Get("Authorization")
		if clientAuth != "Bearer "+placeholder {
			t.Errorf("child presented %q, want Bearer %s", clientAuth, placeholder)
		}

		// Gateway swaps credentials upstream
		upReq, err := http.NewRequest(r.Method, upstream.URL+r.URL.Path, r.Body)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		upReq.Header.Set("Authorization", "Bearer "+realKey)
		upReq.Header.Set("Content-Type", "application/json")

		upResp, err := http.DefaultClient.Do(upReq)
		if err != nil {
			http.Error(w, err.Error(), 502)
			return
		}
		defer upResp.Body.Close()

		w.Header().Set("Content-Type", upResp.Header.Get("Content-Type"))
		w.WriteHeader(upResp.StatusCode)
		buf := new(bytes.Buffer)
		_, _ = buf.ReadFrom(upResp.Body)
		_, _ = w.Write(buf.Bytes())
	}))
	defer gatewayProxy.Close()

	// Call gateway with placeholder
	gwReq, err := http.NewRequest(http.MethodPost, gatewayProxy.URL+"/v1/chat/completions", strings.NewReader(`{"model":"test","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("http.NewRequest gw: %v", err)
	}
	gwReq.Header.Set("Authorization", "Bearer "+placeholder)
	gwReq.Header.Set("Content-Type", "application/json")

	gwResp, err := http.DefaultClient.Do(gwReq)
	if err != nil {
		t.Fatalf("gw Do error: %v", err)
	}
	defer gwResp.Body.Close()

	if gwResp.StatusCode != http.StatusOK {
		t.Fatalf("proxied request got status %d, want 200 OK", gwResp.StatusCode)
	}
	if proxyHits != 1 || upstreamHits != 1 {
		t.Errorf("proxyHits=%d upstreamHits=%d, want 1, 1", proxyHits, upstreamHits)
	}
}

// TestGuardOpenCodeAuditJournalVerification verifies that decision journal entries for OpenCode
// tool calls form a valid, verifiable hash chain that passes journal verification.
func TestGuardOpenCodeAuditJournalVerification(t *testing.T) {
	tempDir := t.TempDir()
	journalPath := filepath.Join(tempDir, "audit.jsonl")
	res := abi.ActiveResolver()
	if res == nil {
		t.Fatal("no Ref resolver registered")
	}

	j, err := journal.Open(journalPath)
	if err != nil {
		t.Fatalf("journal.Open: %v", err)
	}

	// Emit config swap row
	j.AppendConfigSwap(journal.ConfigSwapFloor, "guard-default-policy.json", "sha256:test", journal.ConfigSwapOK, "")

	ref1, _ := res.Put(context.Background(), []byte(`{"command":"echo fak-test"}`))
	j.Emit(abi.Event{
		Kind: abi.EvDecide,
		Call: &abi.ToolCall{
			SeqNo: 1,
			Tool:  "bash",
			Args:  ref1,
		},
		Verdict: &abi.Verdict{
			Kind:   abi.VerdictAllow,
			Reason: abi.ReasonNone,
			By:     "monitor",
		},
	})

	ref2, _ := res.Put(context.Background(), []byte(`{"command":"rm -rf /tmp/danger"}`))
	j.Emit(abi.Event{
		Kind: abi.EvDeny,
		Call: &abi.ToolCall{
			SeqNo: 2,
			Tool:  "bash",
			Args:  ref2,
		},
		Verdict: &abi.Verdict{
			Kind:   abi.VerdictDeny,
			Reason: abi.ReasonPolicyBlock,
			By:     "monitor",
		},
	})

	ref3, _ := res.Put(context.Background(), []byte(`{"filePath":".git/config","oldString":"a","newString":"b"}`))
	j.Emit(abi.Event{
		Kind: abi.EvDeny,
		Call: &abi.ToolCall{
			SeqNo: 3,
			Tool:  "edit",
			Args:  ref3,
		},
		Verdict: &abi.Verdict{
			Kind:   abi.VerdictDeny,
			Reason: abi.ReasonSelfModify,
			By:     "monitor",
		},
	})

	// Close journal
	if err := j.Close(); err != nil {
		t.Fatalf("journal.Close: %v", err)
	}

	// Verify journal integrity
	n, err := journal.Verify(journalPath)
	if err != nil {
		t.Fatalf("Verify sound=false: %v", err)
	}
	if n != 4 {
		t.Fatalf("Verify rows=%d, want 4", n)
	}

	// Verify file content has expected hash chain
	data, err := os.ReadFile(journalPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 4 {
		t.Fatalf("expected 4 lines, got %d", len(lines))
	}

	var firstRow, lastRow struct {
		Seq      uint64 `json:"seq"`
		PrevHash string `json:"prev_hash"`
		Hash     string `json:"hash"`
		Tool     string `json:"tool"`
		Verdict  string `json:"verdict"`
	}

	if err := json.Unmarshal([]byte(lines[0]), &firstRow); err != nil {
		t.Fatalf("Unmarshal line 0: %v", err)
	}
	if firstRow.PrevHash != "" {
		t.Errorf("first row prev_hash = %q, want empty", firstRow.PrevHash)
	}

	if err := json.Unmarshal([]byte(lines[3]), &lastRow); err != nil {
		t.Fatalf("Unmarshal line 3: %v", err)
	}
	if lastRow.Tool != "edit" || lastRow.Verdict != "DENY" {
		t.Errorf("last row: tool=%q verdict=%q, want edit/DENY", lastRow.Tool, lastRow.Verdict)
	}
}
