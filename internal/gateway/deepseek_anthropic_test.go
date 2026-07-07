package gateway

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// TestDeepSeekAnthropicMessagesURL pins the endpoint composition: the DeepSeek route
// composes the SAME `/anthropic` base + `/v1/messages` suffix the real-Anthropic
// adapter uses, yielding the documented DeepSeek URL.
func TestDeepSeekAnthropicMessagesURL(t *testing.T) {
	if DeepSeekAnthropicMessagesPath != "/v1/messages" {
		t.Fatalf("messages path drifted from the Anthropic adapter suffix: %q", DeepSeekAnthropicMessagesPath)
	}
	prof, err := NewDeepSeekAnthropicProfile("k", modelroute.DeepSeekV4ProModel, "")
	if err != nil {
		t.Fatalf("build profile: %v", err)
	}
	const want = "https://api.deepseek.com/anthropic/v1/messages"
	if got := prof.MessagesURL(); got != want {
		t.Fatalf("MessagesURL = %q, want %q", got, want)
	}
	// A base with a trailing slash must not double the separator.
	slashed := DeepSeekAnthropicProfile{BaseURL: "https://api.deepseek.com/anthropic/", Model: "deepseek-v4-pro"}
	if got := slashed.MessagesURL(); got != want {
		t.Fatalf("trailing-slash base composed %q, want %q", got, want)
	}
}

// TestDeepSeekAnthropicResolveModel covers the explicit model-name policy: direct V4
// ids pass through, Claude family names map by fak's own rule, [1m] is recognized, and
// an unroutable name FAILS LOUD instead of silently defaulting to Flash.
func TestDeepSeekAnthropicResolveModel(t *testing.T) {
	cases := []struct {
		in      string
		model   string
		source  DeepSeekAnthropicModelSource
		has1M   bool
		wantErr bool
	}{
		{in: "deepseek-v4-pro", model: "deepseek-v4-pro", source: DeepSeekAnthropicModelDirect},
		{in: "deepseek-v4-flash", model: "deepseek-v4-flash", source: DeepSeekAnthropicModelDirect},
		{in: "deepseek-v4-pro[1m]", model: "deepseek-v4-pro", source: DeepSeekAnthropicModelDirect, has1M: true},
		{in: "claude-opus-4-8", model: "deepseek-v4-pro", source: DeepSeekAnthropicModelClaudeMapped},
		{in: "claude-sonnet-4-5", model: "deepseek-v4-pro", source: DeepSeekAnthropicModelClaudeMapped},
		{in: "claude-3-5-haiku", model: "deepseek-v4-flash", source: DeepSeekAnthropicModelClaudeMapped},
		{in: "gpt-4o", wantErr: true},
		{in: "", wantErr: true},
	}
	for _, c := range cases {
		model, source, has1M, err := ResolveDeepSeekAnthropicModel(c.in)
		if c.wantErr {
			if err == nil {
				t.Errorf("ResolveDeepSeekAnthropicModel(%q): want error, got model=%q", c.in, model)
			}
			continue
		}
		if err != nil {
			t.Errorf("ResolveDeepSeekAnthropicModel(%q): unexpected error %v", c.in, err)
			continue
		}
		if model != c.model || source != c.source || has1M != c.has1M {
			t.Errorf("ResolveDeepSeekAnthropicModel(%q) = (%q,%q,%v), want (%q,%q,%v)", c.in, model, source, has1M, c.model, c.source, c.has1M)
		}
	}
}

// TestDeepSeekAnthropicUnsupportedContentBlockRefused proves the fence refuses every
// documented unsupported block with a CLOSED reason, and passes the supported set.
func TestDeepSeekAnthropicUnsupportedContentBlockRefused(t *testing.T) {
	unsupported := map[string]string{
		"image":             DeepSeekAnthropicUnsupportedImage,
		"document":          DeepSeekAnthropicUnsupportedDocument,
		"redacted_thinking": DeepSeekAnthropicUnsupportedRedactedThinking,
		"code_execution":    DeepSeekAnthropicUnsupportedCodeExecution,
		"server_tool_use":   DeepSeekAnthropicUnsupportedCodeExecution,
		"mcp_tool_use":      DeepSeekAnthropicUnsupportedMCP,
		"mcp_tool_result":   DeepSeekAnthropicUnsupportedMCP,
	}
	for blk, wantReason := range unsupported {
		reason, bad := DeepSeekAnthropicUnsupportedBlockReason(blk)
		if !bad || reason != wantReason {
			t.Errorf("block %q: got (%q,%v), want (%q,true)", blk, reason, bad, wantReason)
		}
		// Casing must not slip a block past the fence.
		if _, bad := DeepSeekAnthropicUnsupportedBlockReason(strings.ToUpper(blk)); !bad {
			t.Errorf("block %q upper-cased slipped past the fence", blk)
		}
		err := FenceContentBlocks([]DeepSeekAnthropicMessage{{Role: "user", Content: []DeepSeekAnthropicBlock{{Type: blk}}}})
		if err == nil || !strings.Contains(err.Error(), wantReason) {
			t.Errorf("FenceContentBlocks(%q): want error naming %q, got %v", blk, wantReason, err)
		}
	}
	for _, ok := range []string{"text", "tool_use", "tool_result", "thinking"} {
		if _, bad := DeepSeekAnthropicUnsupportedBlockReason(ok); bad {
			t.Errorf("supported block %q wrongly refused", ok)
		}
	}
	// A supported-only conversation passes clean.
	if err := FenceContentBlocks([]DeepSeekAnthropicMessage{{Role: "user", Content: []DeepSeekAnthropicBlock{{Type: "text", Text: "hi"}}}}); err != nil {
		t.Errorf("supported-only conversation refused: %v", err)
	}
	// Multiple bad blocks are all named, once each.
	err := FenceContentBlocks([]DeepSeekAnthropicMessage{{Role: "user", Content: []DeepSeekAnthropicBlock{
		{Type: "image"}, {Type: "image"}, {Type: "document"}, {Type: "text"},
	}}})
	if err == nil || !strings.Contains(err.Error(), DeepSeekAnthropicUnsupportedImage) || !strings.Contains(err.Error(), DeepSeekAnthropicUnsupportedDocument) {
		t.Fatalf("multi-block refusal must name image AND document: %v", err)
	}
	if strings.Count(err.Error(), DeepSeekAnthropicUnsupportedImage) != 1 {
		t.Errorf("duplicate image block reported more than once: %v", err)
	}
}

// TestDeepSeekAnthropicEffort covers effort validation across the two levels, unset,
// and a rejected typo.
func TestDeepSeekAnthropicEffort(t *testing.T) {
	for _, ok := range []string{"", "high", "max", "MAX"} {
		if err := ValidateDeepSeekAnthropicEffort(ok); err != nil {
			t.Errorf("effort %q rejected: %v", ok, err)
		}
	}
	for _, bad := range []string{"maximum", "low", "extreme"} {
		if err := ValidateDeepSeekAnthropicEffort(bad); err == nil {
			t.Errorf("effort %q accepted, want rejection", bad)
		}
	}
}

// TestDeepSeekAnthropicCacheControlNotFakSaving fences the value-accounting risk: an
// ignored cache_control on this route is NEVER a fak-managed saving.
func TestDeepSeekAnthropicCacheControlNotFakSaving(t *testing.T) {
	if IsFakManagedCacheSaving() {
		t.Fatal("DeepSeek Anthropic route must never book cache_control as a fak-managed saving")
	}
	if DeepSeekAnthropicCacheControlProvenance != "provider-ignored" {
		t.Fatalf("cache_control provenance = %q, want provider-ignored", DeepSeekAnthropicCacheControlProvenance)
	}
}

// TestDeepSeekAnthropicOfflinePost drives the full offline round trip: the request
// posts to the expected DeepSeek URL with x-api-key (and leaks the secret nowhere),
// carries output_config.effort=max and thinking WITHOUT budget_tokens, and the
// thinking-mode response blocks decode back.
func TestDeepSeekAnthropicOfflinePost(t *testing.T) {
	const secret = "sk-deepseek-SECRET-should-not-leak"
	var gotPath, gotKey string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotKey = r.Header.Get("x-api-key")
		raw, _ := io.ReadAll(r.Body)
		if strings.Contains(r.URL.String(), secret) {
			t.Errorf("secret leaked into URL: %s", r.URL.String())
		}
		if strings.Contains(string(raw), secret) {
			t.Errorf("secret leaked into request body")
		}
		if r.Header.Get("Authorization") != "" {
			t.Errorf("DeepSeek route must use x-api-key, not Authorization: %q", r.Header.Get("Authorization"))
		}
		_ = json.Unmarshal(raw, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{
			"id":"msg_1","model":"deepseek-v4-pro","role":"assistant",
			"content":[
				{"type":"thinking","thinking":"weighing the options","signature":"sig-abc"},
				{"type":"text","text":"done"}
			],
			"stop_reason":"end_turn",
			"usage":{"input_tokens":10,"output_tokens":5,"prompt_cache_hit_tokens":8,"prompt_cache_miss_tokens":2}
		}`)
	}))
	defer srv.Close()

	prof, err := NewDeepSeekAnthropicProfile(secret, "claude-opus-4-8", DeepSeekAnthropicEffortMax)
	if err != nil {
		t.Fatalf("build profile: %v", err)
	}
	prof.BaseURL = srv.URL + "/anthropic" // point the /anthropic base at the offline server
	if prof.Model != "deepseek-v4-pro" {
		t.Fatalf("claude-opus should map to deepseek-v4-pro, got %q", prof.Model)
	}

	resp, err := prof.PostMessages(context.Background(), srv.Client(),
		"you are a coding agent",
		[]DeepSeekAnthropicMessage{{Role: "user", Content: []DeepSeekAnthropicBlock{{Type: "text", Text: "fix the bug"}}}},
		1024, true)
	if err != nil {
		t.Fatalf("PostMessages: %v", err)
	}

	if gotPath != "/anthropic/v1/messages" {
		t.Errorf("posted to %q, want /anthropic/v1/messages", gotPath)
	}
	if gotKey != secret {
		t.Errorf("x-api-key = %q, want the configured secret", gotKey)
	}
	// output_config.effort=max present.
	oc, _ := gotBody["output_config"].(map[string]any)
	if oc == nil || oc["effort"] != "max" {
		t.Errorf("request output_config.effort != max: %v", gotBody["output_config"])
	}
	// thinking enabled, budget_tokens NOT sent (DeepSeek ignores it).
	th, _ := gotBody["thinking"].(map[string]any)
	if th == nil || th["type"] != "enabled" {
		t.Errorf("request thinking not enabled: %v", gotBody["thinking"])
	}
	if _, has := th["budget_tokens"]; has {
		t.Errorf("budget_tokens must not be sent (DeepSeek ignores it): %v", th)
	}
	// Thinking-mode response blocks decode back.
	var haveThinking, haveText bool
	for _, b := range resp.Content {
		switch b.Type {
		case "thinking":
			haveThinking = b.Thinking == "weighing the options" && b.Signature == "sig-abc"
		case "text":
			haveText = b.Text == "done"
		}
	}
	if !haveThinking || !haveText {
		t.Errorf("response blocks not decoded: thinking=%v text=%v (%+v)", haveThinking, haveText, resp.Content)
	}
	// Provider cache counters are OBSERVED, not fak savings.
	if resp.Usage.PromptCacheHitTokens != 8 || IsFakManagedCacheSaving() {
		t.Errorf("cache hit tokens are provider-observed, never a fak saving: hit=%d", resp.Usage.PromptCacheHitTokens)
	}
}

// TestDeepSeekAnthropicOfflinePostRefusesUnsupportedBlock proves the fence stops an
// image block BEFORE it reaches the wire.
func TestDeepSeekAnthropicOfflinePostRefusesUnsupportedBlock(t *testing.T) {
	var reached bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reached = true
		io.WriteString(w, `{}`)
	}))
	defer srv.Close()
	prof, err := NewDeepSeekAnthropicProfile("k", "deepseek-v4-flash", "")
	if err != nil {
		t.Fatalf("build profile: %v", err)
	}
	prof.BaseURL = srv.URL + "/anthropic"
	_, err = prof.PostMessages(context.Background(), srv.Client(), "",
		[]DeepSeekAnthropicMessage{{Role: "user", Content: []DeepSeekAnthropicBlock{{Type: "image"}}}}, 256, false)
	if err == nil || !strings.Contains(err.Error(), DeepSeekAnthropicUnsupportedImage) {
		t.Fatalf("post with image block must be refused loud: %v", err)
	}
	if reached {
		t.Fatal("unsupported block reached the upstream instead of being fenced")
	}
}

// TestClaudeCodeDeepSeekRunbook proves the Claude Code env preset renders for both
// Linux/macOS and Windows, references the operator-owned DEEPSEEK_API_KEY, and never
// inlines a secret value.
func TestClaudeCodeDeepSeekRunbook(t *testing.T) {
	env, err := ClaudeCodeDeepSeekEnv("deepseek-v4-pro[1m]")
	if err != nil {
		t.Fatalf("env: %v", err)
	}
	if env["ANTHROPIC_BASE_URL"] != modelroute.DeepSeekAnthropicBaseURL {
		t.Errorf("base url = %q", env["ANTHROPIC_BASE_URL"])
	}
	if env["ANTHROPIC_MODEL"] != "deepseek-v4-pro[1m]" {
		t.Errorf("model = %q, want deepseek-v4-pro[1m]", env["ANTHROPIC_MODEL"])
	}
	if !strings.Contains(env["ANTHROPIC_API_KEY"], modelroute.DeepSeekAPIKeyEnv) {
		t.Errorf("api key entry must reference %s, got %q", modelroute.DeepSeekAPIKeyEnv, env["ANTHROPIC_API_KEY"])
	}

	unix, err := ClaudeCodeDeepSeekRunbook("unix", "claude-opus-4-8")
	if err != nil {
		t.Fatalf("unix runbook: %v", err)
	}
	win, err := ClaudeCodeDeepSeekRunbook("windows", "claude-opus-4-8")
	if err != nil {
		t.Fatalf("windows runbook: %v", err)
	}
	joinUnix, joinWin := strings.Join(unix, "\n"), strings.Join(win, "\n")
	if !strings.Contains(joinUnix, "export ANTHROPIC_BASE_URL=") {
		t.Errorf("unix runbook missing export lines:\n%s", joinUnix)
	}
	if !strings.Contains(joinWin, "$env:ANTHROPIC_BASE_URL") {
		t.Errorf("windows runbook missing $env: lines:\n%s", joinWin)
	}
	for _, block := range []string{joinUnix, joinWin} {
		if !strings.Contains(block, modelroute.DeepSeekAPIKeyEnv) {
			t.Errorf("runbook must name the operator secret %s:\n%s", modelroute.DeepSeekAPIKeyEnv, block)
		}
	}
	// Windows form uses PowerShell's own env reference for the secret.
	if !strings.Contains(joinWin, "$env:"+modelroute.DeepSeekAPIKeyEnv) {
		t.Errorf("windows runbook should reference $env:%s:\n%s", modelroute.DeepSeekAPIKeyEnv, joinWin)
	}
	if _, err := ClaudeCodeDeepSeekRunbook("plan9", "deepseek-v4-pro"); err == nil {
		t.Error("unknown platform should error")
	}
}
