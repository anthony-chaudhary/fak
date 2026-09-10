package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
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
		// Allowed OpenCode tool calls and transparent kernel transforms (#11150, #11499)
		{"benign bash", "bash", `{"command":"echo fak-opencode-ok"}`, abi.VerdictAllow, abi.ReasonNone},
		{"benign read with filePath", "read", `{"filePath":"README.md"}`, abi.VerdictTransform, abi.ReasonNone},
		{"benign in-tree write", "write", `{"filePath":"notes.txt","content":"hello"}`, abi.VerdictAllow, abi.ReasonNone},
		{"benign in-tree edit", "edit", `{"filePath":"notes.txt","oldString":"a","newString":"b"}`, abi.VerdictAllow, abi.ReasonNone},
		{"benign grep", "grep", `{"pattern":"func main"}`, abi.VerdictTransform, abi.ReasonNone},
		{"benign glob", "glob", `{"pattern":"**/*.go"}`, abi.VerdictTransform, abi.ReasonNone},
		{"benign webfetch", "webfetch", `{"url":"https://github.com/anthony-chaudhary/fak"}`, abi.VerdictAllow, abi.ReasonNone},
		{"benign todowrite", "todowrite", `{"todos":[]}`, abi.VerdictAllow, abi.ReasonNone},
		{"benign skill", "skill", `{"name":"agent-readiness"}`, abi.VerdictAllow, abi.ReasonNone},

		// Dangerous bash commands blocked by POLICY_BLOCK
		{"rm -rf blocked", "bash", `{"command":"rm -rf /tmp/danger"}`, abi.VerdictDeny, abi.ReasonPolicyBlock},
		{"sudo blocked", "bash", `{"command":"sudo rm /etc/hosts"}`, abi.VerdictDeny, abi.ReasonPolicyBlock},
		{"fork bomb blocked", "bash", `{"command":":(){ :|:& };:"}`, abi.VerdictDeny, abi.ReasonPolicyBlock},

		// Sensitive repository structures blocked by SELF_MODIFY or POLICY_BLOCK via filePath
		{"edit .git/config blocked", "edit", `{"filePath":".git/config","oldString":"a","newString":"b"}`, abi.VerdictDeny, abi.ReasonSelfModify},
		{"write .git/hooks blocked", "write", `{"filePath":".git/hooks/pre-commit","content":"#!/bin/sh"}`, abi.VerdictDeny, abi.ReasonSelfModify},
		{"write .env blocked", "write", `{"filePath":".env","content":"KEY=secret"}`, abi.VerdictDeny, abi.ReasonPolicyBlock},

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

func TestGuardOpenCodeConfigInjection(t *testing.T) {
	command := []string{"opencode"}
	gwURL := "http://127.0.0.1:54321"
	modelID := "qwen38"
	getenv := func(string) string { return "" }

	injected, install := installGuardOpenCodeConfig(command, gwURL, modelID, getenv)

	if !install.Applied {
		t.Fatalf("install.Applied = false, want true")
	}
	if install.ProviderID != "fak" {
		t.Errorf("install.ProviderID = %q, want %q", install.ProviderID, "fak")
	}
	if install.Model != "fak/qwen38" {
		t.Errorf("install.Model = %q, want %q", install.Model, "fak/qwen38")
	}
	if install.BaseURL != "http://127.0.0.1:54321/v1" {
		t.Errorf("install.BaseURL = %q, want %q", install.BaseURL, "http://127.0.0.1:54321/v1")
	}

	envMap := make(map[string]string)
	for _, pair := range injected {
		envMap[pair[0]] = pair[1]
	}

	configRaw, ok := envMap["OPENCODE_CONFIG_CONTENT"]
	if !ok || strings.TrimSpace(configRaw) == "" {
		t.Fatalf("injected missing OPENCODE_CONFIG_CONTENT: %v", injected)
	}
	if keyVal, ok := envMap["OPENAI_API_KEY"]; !ok || keyVal != "fak-guard-placeholder" {
		t.Errorf("injected OPENAI_API_KEY = %q, want %q", keyVal, "fak-guard-placeholder")
	}

	var config map[string]any
	if err := json.Unmarshal([]byte(configRaw), &config); err != nil {
		t.Fatalf("failed to unmarshal OPENCODE_CONFIG_CONTENT JSON: %v", err)
	}

	if got := config["model"]; got != "fak/qwen38" {
		t.Errorf("config.model = %v, want %q", got, "fak/qwen38")
	}
	if got := config["small_model"]; got != "fak/qwen38" {
		t.Errorf("config.small_model = %v, want %q", got, "fak/qwen38")
	}

	providerMap, ok := config["provider"].(map[string]any)
	if !ok {
		t.Fatalf("config.provider is not a map: %T", config["provider"])
	}
	fakProvider, ok := providerMap["fak"].(map[string]any)
	if !ok {
		t.Fatalf("config.provider.fak is not a map: %T", providerMap["fak"])
	}
	if got := fakProvider["npm"]; got != "@ai-sdk/openai-compatible" {
		t.Errorf("fakProvider.npm = %v, want %q", got, "@ai-sdk/openai-compatible")
	}
	optionsMap, ok := fakProvider["options"].(map[string]any)
	if !ok {
		t.Fatalf("fakProvider.options is not a map: %T", fakProvider["options"])
	}
	if got := optionsMap["baseURL"]; got != "{env:OPENAI_BASE_URL}" {
		t.Errorf("options.baseURL = %v, want %q", got, "{env:OPENAI_BASE_URL}")
	}
	if got := optionsMap["apiKey"]; got != "{env:OPENAI_API_KEY}" {
		t.Errorf("options.apiKey = %v, want %q", got, "{env:OPENAI_API_KEY}")
	}

	modelsMap, ok := fakProvider["models"].(map[string]any)
	if !ok {
		t.Fatalf("fakProvider.models is not a map: %T", fakProvider["models"])
	}
	if _, ok := modelsMap["qwen38"]; !ok {
		t.Errorf("modelsMap missing qwen38")
	}

	commonAliases := []string{
		"fak-local",
		"qwen38:27b",
		"glm-5.2",
		"glm-5.3-flash",
		"qwen2.5-coder:7b",
		"qwen2.5-coder:32b",
	}
	for _, alias := range commonAliases {
		if _, ok := modelsMap[alias]; !ok {
			t.Errorf("modelsMap missing common alias %q", alias)
		}
	}
}

func TestGuardOpenCodeConfigPreservesExplicitModelLimits(t *testing.T) {
	existing := `{"provider":{"fak":{"options":{"custom":"kept"},"models":{"qwen38":{"name":"Custom Qwen","limit":{"context":8192,"output":512},"custom":"kept"}}}}}`
	injected, install := installGuardOpenCodeConfig([]string{"opencode"}, "http://127.0.0.1:54321", "qwen38", func(key string) string {
		if key == "OPENCODE_CONFIG_CONTENT" {
			return existing
		}
		return ""
	})
	if !install.Applied {
		t.Fatal("OpenCode config injection was not applied")
	}
	var config map[string]any
	if err := json.Unmarshal([]byte(injected[0][1]), &config); err != nil {
		t.Fatal(err)
	}
	provider := config["provider"].(map[string]any)["fak"].(map[string]any)
	model := provider["models"].(map[string]any)["qwen38"].(map[string]any)
	limit := model["limit"].(map[string]any)
	if limit["context"] != float64(8192) || limit["output"] != float64(512) || model["name"] != "Custom Qwen" || model["custom"] != "kept" {
		t.Fatalf("explicit model fields changed: %#v", model)
	}
	if provider["options"].(map[string]any)["custom"] != "kept" {
		t.Fatalf("explicit provider options changed: %#v", provider["options"])
	}
}

func TestDiscoverGuardOpenCodeModelLimits(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"data":[{"id":"local","context_length":8192,"max_output_tokens":512}]}`)
	}))
	defer server.Close()
	got := discoverGuardOpenCodeModelLimits(server.URL, "local")
	if got.ID != "local" || got.Context != 8192 || got.Output != 512 {
		t.Fatalf("discovered limits = %+v", got)
	}
}

func TestGuardOpenCodeExistingConfigContentPreserved(t *testing.T) {
	existingJSON := `{
		"permission": "allow",
		"instructions": ["AGENTS.md"],
		"custom_setting": 42,
		"provider": {
			"anthropic": {
				"name": "Anthropic",
				"npm": "@ai-sdk/anthropic"
			}
		},
		"model": "anthropic/claude-3-5-sonnet",
		"small_model": "anthropic/claude-3-5-haiku"
	}`

	getenv := func(k string) string {
		if k == "OPENCODE_CONFIG_CONTENT" {
			return existingJSON
		}
		return ""
	}

	injected, install := installGuardOpenCodeConfig([]string{"opencode"}, "http://127.0.0.1:54321", "qwen38", getenv)
	if !install.Applied {
		t.Fatalf("install.Applied = false, want true")
	}

	envMap := make(map[string]string)
	for _, pair := range injected {
		envMap[pair[0]] = pair[1]
	}

	configRaw := envMap["OPENCODE_CONFIG_CONTENT"]
	var config map[string]any
	if err := json.Unmarshal([]byte(configRaw), &config); err != nil {
		t.Fatalf("failed to unmarshal config: %v", err)
	}

	if got := config["permission"]; got != "allow" {
		t.Errorf("config.permission = %v, want %q", got, "allow")
	}
	instructions, ok := config["instructions"].([]any)
	if !ok || len(instructions) != 1 || instructions[0] != "AGENTS.md" {
		t.Errorf("config.instructions = %v, want [AGENTS.md]", config["instructions"])
	}
	if got := config["custom_setting"]; got != float64(42) {
		t.Errorf("config.custom_setting = %v, want 42", got)
	}

	if got := config["model"]; got != "fak/qwen38" {
		t.Errorf("config.model = %v, want %q", got, "fak/qwen38")
	}
	if got := config["small_model"]; got != "fak/qwen38" {
		t.Errorf("config.small_model = %v, want %q", got, "fak/qwen38")
	}

	providerMap, ok := config["provider"].(map[string]any)
	if !ok {
		t.Fatalf("provider is not a map")
	}
	if _, ok := providerMap["anthropic"]; !ok {
		t.Errorf("existing provider 'anthropic' was not preserved in provider map")
	}
	if _, ok := providerMap["fak"]; !ok {
		t.Errorf("provider 'fak' was not merged into provider map")
	}
}

func TestGuardOpenCodeModelOverrideFromCommand(t *testing.T) {
	t.Run("-m flag override", func(t *testing.T) {
		command := []string{"opencode", "-m", "my-custom-model"}
		injected, install := installGuardOpenCodeConfig(command, "http://127.0.0.1:54321", "", nil)
		if !install.Applied {
			t.Fatalf("install.Applied = false, want true")
		}
		if install.Model != "fak/my-custom-model" {
			t.Errorf("install.Model = %q, want %q", install.Model, "fak/my-custom-model")
		}

		var config map[string]any
		for _, pair := range injected {
			if pair[0] == "OPENCODE_CONFIG_CONTENT" {
				_ = json.Unmarshal([]byte(pair[1]), &config)
			}
		}
		providerMap := config["provider"].(map[string]any)
		fakProvider := providerMap["fak"].(map[string]any)
		modelsMap := fakProvider["models"].(map[string]any)
		if _, ok := modelsMap["my-custom-model"]; !ok {
			t.Errorf("modelsMap missing my-custom-model")
		}
	})

	t.Run("--model=fak/another-model flag override", func(t *testing.T) {
		command := []string{"opencode", "--model=fak/another-model"}
		injected, install := installGuardOpenCodeConfig(command, "http://127.0.0.1:54321", "", nil)
		if !install.Applied {
			t.Fatalf("install.Applied = false, want true")
		}
		if install.Model != "fak/another-model" {
			t.Errorf("install.Model = %q, want %q", install.Model, "fak/another-model")
		}

		var config map[string]any
		for _, pair := range injected {
			if pair[0] == "OPENCODE_CONFIG_CONTENT" {
				_ = json.Unmarshal([]byte(pair[1]), &config)
			}
		}
		providerMap := config["provider"].(map[string]any)
		fakProvider := providerMap["fak"].(map[string]any)
		modelsMap := fakProvider["models"].(map[string]any)
		if _, ok := modelsMap["another-model"]; !ok {
			t.Errorf("modelsMap missing another-model")
		}
	})

	t.Run("explicit modelID preserves command model in models map", func(t *testing.T) {
		command := []string{"opencode", "-m", "my-custom-model"}
		injected, install := installGuardOpenCodeConfig(command, "http://127.0.0.1:54321", "primary-model", nil)
		if !install.Applied {
			t.Fatalf("install.Applied = false, want true")
		}
		if install.Model != "fak/primary-model" {
			t.Errorf("install.Model = %q, want %q", install.Model, "fak/primary-model")
		}

		var config map[string]any
		for _, pair := range injected {
			if pair[0] == "OPENCODE_CONFIG_CONTENT" {
				_ = json.Unmarshal([]byte(pair[1]), &config)
			}
		}
		providerMap := config["provider"].(map[string]any)
		fakProvider := providerMap["fak"].(map[string]any)
		modelsMap := fakProvider["models"].(map[string]any)
		if _, ok := modelsMap["primary-model"]; !ok {
			t.Errorf("modelsMap missing primary-model")
		}
		if _, ok := modelsMap["my-custom-model"]; !ok {
			t.Errorf("modelsMap missing my-custom-model from command args")
		}
	})
}

func TestGuardOpenCodeNonOpencodeIgnored(t *testing.T) {
	cases := []struct {
		name    string
		command []string
	}{
		{"claude", []string{"claude"}},
		{"bash", []string{"bash"}},
		{"empty", []string{}},
		{"other binary", []string{"python", "script.py"}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			injected, install := installGuardOpenCodeConfig(tc.command, "http://127.0.0.1:54321", "qwen38", nil)
			if install.Applied {
				t.Errorf("%s: install.Applied = true, want false", tc.name)
			}
			if install.Reason != "non-opencode-child" {
				t.Errorf("%s: install.Reason = %q, want %q", tc.name, install.Reason, "non-opencode-child")
			}
			if len(injected) != 0 {
				t.Errorf("%s: expected no injected env, got %v", tc.name, injected)
			}
		})
	}
}

func TestGuardOpenCodeSafetyRootSettingsUntouched(t *testing.T) {
	mockHome := t.TempDir()
	t.Setenv("HOME", mockHome)
	t.Setenv("USERPROFILE", mockHome)

	workDir := t.TempDir()
	t.Chdir(workDir)

	// Pre-create ~/.config/opencode/opencode.json with sentinel content
	userConfigDir := filepath.Join(mockHome, ".config", "opencode")
	if err := os.MkdirAll(userConfigDir, 0755); err != nil {
		t.Fatalf("mkdir userConfigDir: %v", err)
	}
	userConfigFile := filepath.Join(userConfigDir, "opencode.json")
	sentinelUserContent := []byte(`{"sentinel": "user-level"}`)
	if err := os.WriteFile(userConfigFile, sentinelUserContent, 0644); err != nil {
		t.Fatalf("write userConfigFile: %v", err)
	}
	userStatBefore, err := os.Stat(userConfigFile)
	if err != nil {
		t.Fatalf("stat userConfigFile: %v", err)
	}

	// Pre-create ./opencode.json with sentinel content in working directory
	workspaceConfigFile := filepath.Join(workDir, "opencode.json")
	sentinelWorkContent := []byte(`{"sentinel": "workspace-level"}`)
	if err := os.WriteFile(workspaceConfigFile, sentinelWorkContent, 0644); err != nil {
		t.Fatalf("write workspaceConfigFile: %v", err)
	}
	workStatBefore, err := os.Stat(workspaceConfigFile)
	if err != nil {
		t.Fatalf("stat workspaceConfigFile: %v", err)
	}

	// Call installGuardOpenCodeConfig
	injected, install := installGuardOpenCodeConfig([]string{"opencode"}, "http://127.0.0.1:54321", "qwen38", nil)
	if !install.Applied {
		t.Fatalf("install.Applied = false, want true")
	}
	if len(injected) == 0 {
		t.Fatalf("expected injected env pairs")
	}

	// Verify user config file was NOT touched or modified
	userStatAfter, err := os.Stat(userConfigFile)
	if err != nil {
		t.Fatalf("userConfigFile stat after: %v", err)
	}
	if userStatAfter.ModTime() != userStatBefore.ModTime() || userStatAfter.Size() != userStatBefore.Size() {
		t.Errorf("userConfigFile modtime or size changed: was %v / %d, now %v / %d",
			userStatBefore.ModTime(), userStatBefore.Size(), userStatAfter.ModTime(), userStatAfter.Size())
	}
	userContentAfter, err := os.ReadFile(userConfigFile)
	if err != nil {
		t.Fatalf("read userConfigFile after: %v", err)
	}
	if string(userContentAfter) != string(sentinelUserContent) {
		t.Errorf("userConfigFile content modified: got %s, want %s", userContentAfter, sentinelUserContent)
	}

	// Verify workspace config file was NOT touched or modified
	workStatAfter, err := os.Stat(workspaceConfigFile)
	if err != nil {
		t.Fatalf("workspaceConfigFile stat after: %v", err)
	}
	if workStatAfter.ModTime() != workStatBefore.ModTime() || workStatAfter.Size() != workStatBefore.Size() {
		t.Errorf("workspaceConfigFile modtime or size changed: was %v / %d, now %v / %d",
			workStatBefore.ModTime(), workStatBefore.Size(), workStatAfter.ModTime(), workStatAfter.Size())
	}
	workContentAfter, err := os.ReadFile(workspaceConfigFile)
	if err != nil {
		t.Fatalf("read workspaceConfigFile after: %v", err)
	}
	if string(workContentAfter) != string(sentinelWorkContent) {
		t.Errorf("workspaceConfigFile content modified: got %s, want %s", workContentAfter, sentinelWorkContent)
	}

	// Also test the clean-slate case: files do not exist before call -> must not be created
	cleanHome := t.TempDir()
	cleanWorkDir := t.TempDir()
	t.Setenv("HOME", cleanHome)
	t.Setenv("USERPROFILE", cleanHome)
	t.Chdir(cleanWorkDir)

	_, installClean := installGuardOpenCodeConfig([]string{"opencode"}, "http://127.0.0.1:54321", "qwen38", nil)
	if !installClean.Applied {
		t.Fatalf("installClean.Applied = false, want true")
	}

	if _, err := os.Stat(filepath.Join(cleanHome, ".config", "opencode", "opencode.json")); !os.IsNotExist(err) {
		t.Errorf("expected ~/.config/opencode/opencode.json to not exist, got err: %v", err)
	}
	if _, err := os.Stat(filepath.Join(cleanWorkDir, "opencode.json")); !os.IsNotExist(err) {
		t.Errorf("expected ./opencode.json to not exist, got err: %v", err)
	}
}
