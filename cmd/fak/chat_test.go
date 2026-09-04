package main

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/dropin"
)

// chatScript is a deterministic offline planner for the fak chat e2e: it returns a
// fixed sequence of completions, one per Complete call (one per model turn), so a
// multi-turn REPL session is fully reproducible with no upstream. It satisfies
// agent.Planner.
type chatScript struct {
	turns []*agent.Completion
	n     int
}

func (p *chatScript) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	c := p.turns[p.n]
	if p.n < len(p.turns)-1 {
		p.n++
	}
	return c, nil
}
func (p *chatScript) Model() string { return "chat-script" }

func toolTurn(tool, args string) *agent.Completion {
	return &agent.Completion{Message: agent.Message{ToolCalls: []agent.ToolCall{{ID: "c", Function: agent.Func{Name: tool, Arguments: args}}}}}
}
func finalTurn(text string) *agent.Completion {
	return &agent.Completion{Message: agent.Message{Content: text}}
}

// TestChatTwoTurnsWithDeniedDestructive is the acceptance witness for #1320: a
// scripted two-turn `fak chat` session driven entirely through agent.RunArm with
// kernel.Syscall as the sole tool path. Turn 1 is an ordinary read that resolves
// to a final answer; turn 2 emits a destructive delete_account call that the
// capability floor DENIES. The test asserts the denial was returned as a VALUE
// (Denies==1, no crash), the destructive tool never executed
// (DestructiveExecuted==false), and the denied call never reached the engine
// (EngineCalls==0 on that turn) — with no upstream involved (offline planner).
func TestChatTwoTurnsWithDeniedDestructive(t *testing.T) {
	// Two human turns over one stdin stream. Turn 1's model script: one read then a
	// final answer. Turn 2's model script: one delete_account (denied) then a final
	// answer. Because runChat drives ONE RunArm per human line and the scripted
	// planner advances per Complete call, the script must lay the turns end to end.
	planner := &chatScript{turns: []*agent.Completion{
		toolTurn("get_user_details", `{"user_id":"mia_li_3668"}`), // turn 1, model step 1
		finalTurn("Found your account."),                          // turn 1, model step 2 (ends turn 1)
		toolTurn("delete_account", `{"user_id":"mia_li_3668"}`),   // turn 2, model step 1 — DENIED
		finalTurn("I can't delete the account; that's blocked."),  // turn 2, model step 2 (ends turn 2)
	}}

	in := strings.NewReader("look up my account\ndelete my account\n")
	var out strings.Builder
	runChat(in, &out, planner, 10)

	got := out.String()
	if !strings.Contains(got, "Found your account.") {
		t.Fatalf("turn 1 final answer missing from REPL output:\n%s", got)
	}
	if !strings.Contains(got, "1 denied") {
		t.Fatalf("turn 2 should report exactly one denied call in its summary:\n%s", got)
	}
}

// TestRunChatTurnMetrics drives the two scripted turns through runChat by reusing
// the same end-to-end stream, then re-runs RunArm directly on the denied turn so
// the value-not-crash assertions read off ArmMetrics precisely: a denied
// destructive call is a structured value, never an executed effect, and never an
// engine dispatch.
func TestRunChatTurnMetrics(t *testing.T) {
	deny := &chatScript{turns: []*agent.Completion{
		toolTurn("delete_account", `{"user_id":"mia_li_3668"}`),
		finalTurn("blocked, as expected"),
	}}
	m, err := agent.RunArm(context.Background(), deny, "delete my account", true, 10, nil)
	if err != nil {
		t.Fatalf("RunArm returned an error on a denied call (should be a value, not a crash): %v", err)
	}
	if m.Denies != 1 {
		t.Fatalf("expected exactly 1 deny, got %d", m.Denies)
	}
	if m.DestructiveExecuted {
		t.Fatal("destructive delete_account must NOT have executed")
	}
	if m.EngineCalls != 0 {
		t.Fatalf("a denied call must never reach the engine; EngineCalls=%d", m.EngineCalls)
	}
	if !strings.Contains(m.FinalAnswer, "blocked") {
		t.Fatalf("loop should have continued past the deny to a final answer, got %q", m.FinalAnswer)
	}
}

func TestRenderChatTerminationUsesSharedSafeClassification(t *testing.T) {
	var out bytes.Buffer
	renderChatTermination(&out, errors.New("provider status 429: secret body"))
	got := out.String()
	if !strings.Contains(got, "[rate_limited]") || !strings.Contains(got, "provider reported rate limiting") || strings.Contains(got, "secret") {
		t.Fatalf("%q", got)
	}
}

func TestChatWithCodeToolsAllowed(t *testing.T) {
	root := t.TempDir()
	filePath := filepath.Join(root, "sample.txt")
	if err := os.WriteFile(filePath, []byte("kernel-gated workspace contents\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog, err := agent.ArmFocusedCodeTools(root)
	if err != nil {
		t.Fatal(err)
	}
	defer agent.DisarmCodeTools()

	planner := &chatScript{turns: []*agent.Completion{
		toolTurn("Read", `{"file_path":"`+filepath.ToSlash(filePath)+`"}`),
		finalTurn("Found file with contents."),
	}}

	in := strings.NewReader("read sample.txt\n")
	var out strings.Builder
	runChat(in, &out, planner, 10, agent.WithToolCatalog(catalog))

	got := out.String()
	if !strings.Contains(got, "Found file with contents.") {
		t.Fatalf("expected final answer in output, got:\n%s", got)
	}
	if !strings.Contains(got, "[tool] Read") || !strings.Contains(got, "ALLOW") {
		t.Fatalf("expected tool execution receipt in output, got:\n%s", got)
	}
}

func TestChatHeadless(t *testing.T) {
	root := t.TempDir()
	filePath := filepath.Join(root, "note.txt")
	if err := os.WriteFile(filePath, []byte("headless proof\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog, err := agent.ArmFocusedCodeTools(root)
	if err != nil {
		t.Fatal(err)
	}
	defer agent.DisarmCodeTools()

	planner := &chatScript{turns: []*agent.Completion{
		toolTurn("Read", `{"file_path":"`+filepath.ToSlash(filePath)+`"}`),
		finalTurn("Headless read finished."),
	}}

	var out strings.Builder
	err = runChatHeadless(&out, planner, "read note.txt", 10, agent.WithToolCatalog(catalog))
	if err != nil {
		t.Fatalf("runChatHeadless failed: %v", err)
	}

	got := out.String()
	if strings.Contains(got, "fak chat — native REPL") || strings.Contains(got, "you> ") {
		t.Fatalf("headless mode must not emit interactive REPL chrome:\n%s", got)
	}
	if !strings.Contains(got, "[tool] Read") || !strings.Contains(got, "ALLOW") {
		t.Fatalf("expected tool execution receipt in headless output:\n%s", got)
	}
	if !strings.Contains(got, "Headless read finished.") {
		t.Fatalf("expected final answer in headless output, got:\n%s", got)
	}
}

type recordingChatPlanner struct {
	recorded [][]agent.Message
	answers  []string
	idx      int
}

func (p *recordingChatPlanner) Complete(_ context.Context, msgs []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	cp := make([]agent.Message, len(msgs))
	copy(cp, msgs)
	p.recorded = append(p.recorded, cp)
	ans := "default answer"
	if p.idx < len(p.answers) {
		ans = p.answers[p.idx]
		p.idx++
	}
	return finalTurn(ans), nil
}

func (p *recordingChatPlanner) Model() string { return "recording-planner" }

func TestChatMultiTurnContext(t *testing.T) {
	planner := &recordingChatPlanner{
		answers: []string{"Answer one", "Answer two"},
	}

	in := strings.NewReader("hello from turn 1\nwhat did I say earlier?\n")
	var out strings.Builder
	runChat(in, &out, planner, 10)

	if len(planner.recorded) != 2 {
		t.Fatalf("expected 2 turns recorded, got %d", len(planner.recorded))
	}

	turn2Msgs := planner.recorded[1]
	// Should contain: system prompt, turn 1 user, turn 1 assistant, turn 2 user
	foundUser1 := false
	foundAsst1 := false
	foundUser2 := false
	for _, m := range turn2Msgs {
		if m.Role == agent.RoleUser && strings.Contains(m.Content, "hello from turn 1") {
			foundUser1 = true
		}
		if m.Role == agent.RoleAssistant && strings.Contains(m.Content, "Answer one") {
			foundAsst1 = true
		}
		if m.Role == agent.RoleUser && strings.Contains(m.Content, "what did I say earlier?") {
			foundUser2 = true
		}
	}

	if !foundUser1 || !foundAsst1 || !foundUser2 {
		t.Fatalf("turn 2 did not receive multi-turn context (user1=%v, asst1=%v, user2=%v):\n%+v",
			foundUser1, foundAsst1, foundUser2, turn2Msgs)
	}
}

func TestChatClearCommandResetsContext(t *testing.T) {
	planner := &recordingChatPlanner{
		answers: []string{"Answer one", "Answer two"},
	}

	in := strings.NewReader("message before clear\n/clear\nmessage after clear\n")
	var out strings.Builder
	runChat(in, &out, planner, 10)

	got := out.String()
	if !strings.Contains(got, "conversation cleared.") {
		t.Fatalf("expected clear notification in output:\n%s", got)
	}

	if len(planner.recorded) != 2 {
		t.Fatalf("expected 2 turns recorded, got %d", len(planner.recorded))
	}

	turn2Msgs := planner.recorded[1]
	for _, m := range turn2Msgs {
		if strings.Contains(m.Content, "message before clear") {
			t.Fatalf("cleared message leaked into turn 2 context:\n%+v", turn2Msgs)
		}
	}
}

func TestChatPlannerOfflineDefault(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("XAI_API_KEY", "")

	oldProbeURL := localDaemonProbeURL
	localDaemonProbeURL = "http://127.0.0.1:0/readyz"
	defer func() { localDaemonProbeURL = oldProbeURL }()

	p := chatPlanner(false, "", "openai", "gemini-3.8-flash", "OPENAI_API_KEY", "auto")
	if p.Model() != "gemini-3.8-flash" {
		t.Fatalf("expected model gemini-3.8-flash, got %s", p.Model())
	}
	if _, ok := p.(*agent.MockPlanner); !ok {
		t.Fatalf("expected *agent.MockPlanner when baseURL is empty, got %T", p)
	}
}

func TestChatPlannerGeminiWire(t *testing.T) {
	baseURL := dropin.DefaultBaseURL("gemini")
	p := chatPlanner(false, baseURL, "gemini", "gemini-3.8-flash", "GEMINI_API_KEY", "auto")
	hp, ok := p.(*agent.HTTPPlanner)
	if !ok {
		t.Fatalf("expected *agent.HTTPPlanner for gemini wire, got %T", p)
	}
	if hp.Provider != agent.ProviderGemini {
		t.Fatalf("expected ProviderGemini, got %v", hp.Provider)
	}
	if hp.BaseURL != baseURL {
		t.Fatalf("expected BaseURL %q, got %q", baseURL, hp.BaseURL)
	}
}

func TestChatLiveGeminiDogfood(t *testing.T) {
	if testing.Short() {
		t.Skip("-short: skipping live Gemini dogfood witness")
	}
	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		t.Skip("GEMINI_API_KEY unset; skipping live Gemini dogfood witness")
	}

	baseURL := dropin.DefaultBaseURL("gemini")
	planner := chatPlanner(false, baseURL, "gemini", "gemini-3.8-flash", "GEMINI_API_KEY", "auto")

	var out strings.Builder
	err := runChatHeadless(&out, planner, "Say 'GEMINI_DOGFOOD_OK' and nothing else.", 3)
	if err != nil {
		t.Fatalf("live gemini dogfood headless run failed: %v", err)
	}

	got := out.String()
	if !strings.Contains(got, "GEMINI_DOGFOOD_OK") {
		t.Fatalf("expected GEMINI_DOGFOOD_OK in response, got:\n%s", got)
	}
}

func TestAutoDetectInferenceWithGeminiKey(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "dummy-gemini-key")
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("XAI_API_KEY", "")

	oldProbeURL := localDaemonProbeURL
	localDaemonProbeURL = "http://127.0.0.1:0/readyz"
	defer func() { localDaemonProbeURL = oldProbeURL }()

	detected := autoDetectInference()
	if detected == nil {
		t.Fatal("expected non-nil detectedInference when GEMINI_API_KEY is set")
	}
	if detected.provider != "gemini" {
		t.Fatalf("expected provider gemini, got %s", detected.provider)
	}
	if detected.model != "gemini-2.5-flash" {
		t.Fatalf("expected model gemini-2.5-flash, got %s", detected.model)
	}
	if detected.apiKeyEnv != "GEMINI_API_KEY" {
		t.Fatalf("expected apiKeyEnv GEMINI_API_KEY, got %s", detected.apiKeyEnv)
	}
}

func TestChatAutoDetectLocalDaemon(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/readyz" {
			w.WriteHeader(http.StatusOK)
			return
		}
		http.NotFound(w, r)
	}))
	defer ts.Close()

	oldProbeURL := localDaemonProbeURL
	oldBaseURL := localDaemonBaseURL
	localDaemonProbeURL = ts.URL + "/readyz"
	localDaemonBaseURL = ts.URL + "/v1"
	defer func() {
		localDaemonProbeURL = oldProbeURL
		localDaemonBaseURL = oldBaseURL
	}()

	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("XAI_API_KEY", "")

	p := chatPlanner(false, "", "", "", "", "auto")
	hp, ok := p.(*agent.HTTPPlanner)
	if !ok {
		t.Fatalf("expected *agent.HTTPPlanner from local daemon probe, got %T", p)
	}
	if hp.BaseURL != ts.URL+"/v1" {
		t.Fatalf("expected baseURL %s, got %s", ts.URL+"/v1", hp.BaseURL)
	}
	if hp.Provider != agent.ProviderOpenAI {
		t.Fatalf("expected provider %v, got %v", agent.ProviderOpenAI, hp.Provider)
	}
	if p.Model() != "default" {
		t.Fatalf("expected model default, got %s", p.Model())
	}

	pCustom := chatPlanner(false, "", "", "custom-model", "", "auto")
	if pCustom.Model() != "custom-model" {
		t.Fatalf("expected model custom-model when explicitly passed, got %s", pCustom.Model())
	}
}

func TestChatAutoDetectEnvKeys(t *testing.T) {
	oldProbeURL := localDaemonProbeURL
	localDaemonProbeURL = "http://127.0.0.1:0/readyz"
	defer func() { localDaemonProbeURL = oldProbeURL }()

	tests := []struct {
		name         string
		setKey       string
		wantProvider agent.Provider
		wantBaseURL  string
		wantModel    string
	}{
		{
			name:         "gemini",
			setKey:       "GEMINI_API_KEY",
			wantProvider: agent.ProviderGemini,
			wantBaseURL:  "https://generativelanguage.googleapis.com/v1beta",
			wantModel:    "gemini-2.5-flash",
		},
		{
			name:         "anthropic",
			setKey:       "ANTHROPIC_API_KEY",
			wantProvider: agent.ProviderAnthropic,
			wantBaseURL:  "https://api.anthropic.com",
			wantModel:    "claude-3-7-sonnet-20250219",
		},
		{
			name:         "openai",
			setKey:       "OPENAI_API_KEY",
			wantProvider: agent.ProviderOpenAI,
			wantBaseURL:  "https://api.openai.com/v1",
			wantModel:    "gpt-4o",
		},
		{
			name:         "xai",
			setKey:       "XAI_API_KEY",
			wantProvider: agent.ProviderOpenAI,
			wantBaseURL:  "https://api.x.ai/v1",
			wantModel:    "grok-beta",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("GEMINI_API_KEY", "")
			t.Setenv("ANTHROPIC_API_KEY", "")
			t.Setenv("OPENAI_API_KEY", "")
			t.Setenv("XAI_API_KEY", "")
			t.Setenv(tt.setKey, "mock-api-key")

			p := chatPlanner(false, "", "", "", "", "auto")
			hp, ok := p.(*agent.HTTPPlanner)
			if !ok {
				t.Fatalf("expected *agent.HTTPPlanner when %s is set, got %T", tt.setKey, p)
			}
			if hp.Provider != tt.wantProvider {
				t.Fatalf("expected provider %v, got %v", tt.wantProvider, hp.Provider)
			}
			if hp.BaseURL != tt.wantBaseURL {
				t.Fatalf("expected baseURL %s, got %s", tt.wantBaseURL, hp.BaseURL)
			}
			if p.Model() != tt.wantModel {
				t.Fatalf("expected model %s, got %s", tt.wantModel, p.Model())
			}
		})
	}
}

func TestChatAutoDetectEnvPrecedenceAndExplicitModel(t *testing.T) {
	oldProbeURL := localDaemonProbeURL
	localDaemonProbeURL = "http://127.0.0.1:0/readyz"
	defer func() { localDaemonProbeURL = oldProbeURL }()

	t.Setenv("GEMINI_API_KEY", "gemini-key")
	t.Setenv("ANTHROPIC_API_KEY", "anthropic-key")
	t.Setenv("OPENAI_API_KEY", "openai-key")
	t.Setenv("XAI_API_KEY", "xai-key")

	p := chatPlanner(false, "", "", "", "", "auto")
	hp, ok := p.(*agent.HTTPPlanner)
	if !ok {
		t.Fatalf("expected *agent.HTTPPlanner, got %T", p)
	}
	if hp.Provider != agent.ProviderGemini {
		t.Fatalf("expected ProviderGemini precedence, got %v", hp.Provider)
	}
	if p.Model() != "gemini-2.5-flash" {
		t.Fatalf("expected model gemini-2.5-flash, got %s", p.Model())
	}

	pExplicitModel := chatPlanner(false, "", "", "custom-gemini", "", "auto")
	if pExplicitModel.Model() != "custom-gemini" {
		t.Fatalf("expected custom-gemini model, got %s", pExplicitModel.Model())
	}
}

func TestChatAutoDetectOfflineForcesSynthetic(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	oldProbeURL := localDaemonProbeURL
	localDaemonProbeURL = ts.URL + "/readyz"
	defer func() { localDaemonProbeURL = oldProbeURL }()

	t.Setenv("GEMINI_API_KEY", "key")
	t.Setenv("OPENAI_API_KEY", "key")

	p := chatPlanner(true, "", "", "gemini-2.5-flash", "", "auto")
	if _, ok := p.(*agent.SyntheticPlanner); !ok {
		t.Fatalf("expected *agent.SyntheticPlanner when offline=true, got %T", p)
	}
	if p.Model() != "gemini-2.5-flash" {
		t.Fatalf("expected model gemini-2.5-flash, got %s", p.Model())
	}
}

func TestChatAutoDetectExplicitBaseURLOverrides(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	oldProbeURL := localDaemonProbeURL
	localDaemonProbeURL = ts.URL + "/readyz"
	defer func() { localDaemonProbeURL = oldProbeURL }()

	t.Setenv("GEMINI_API_KEY", "key")

	explicitURL := "https://explicit.provider.endpoint/v1"
	p := chatPlanner(false, explicitURL, "openai", "custom-model", "OPENAI_API_KEY", "auto")
	hp, ok := p.(*agent.HTTPPlanner)
	if !ok {
		t.Fatalf("expected *agent.HTTPPlanner, got %T", p)
	}
	if hp.BaseURL != explicitURL {
		t.Fatalf("expected baseURL %s, got %s", explicitURL, hp.BaseURL)
	}
	if p.Model() != "custom-model" {
		t.Fatalf("expected model custom-model, got %s", p.Model())
	}
}

func TestChatExplicitProviderOverridesAutoDetect(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "dummy-gemini-key")
	t.Setenv("ANTHROPIC_API_KEY", "dummy-anthropic-key")

	p := chatPlanner(false, "", "anthropic", "", "", "auto")
	hp, ok := p.(*agent.HTTPPlanner)
	if !ok {
		t.Fatalf("expected *agent.HTTPPlanner, got %T", p)
	}
	if hp.Provider != agent.ProviderAnthropic {
		t.Fatalf("expected ProviderAnthropic, got %v", hp.Provider)
	}
	if hp.BaseURL != "https://api.anthropic.com" {
		t.Fatalf("expected Anthropic BaseURL, got %q", hp.BaseURL)
	}
	if p.Model() != "claude-3-7-sonnet-20250219" {
		t.Fatalf("expected default Anthropic model, got %s", p.Model())
	}
}
