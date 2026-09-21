package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
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

type namedChatPlanner struct {
	*chatScript
	model string
}

func (p *namedChatPlanner) Model() string { return p.model }

type streamingChatScript struct {
	chunks []string
	final  string
}

func (p *streamingChatScript) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	return finalTurn(p.final), nil
}

func (p *streamingChatScript) CompleteStream(_ context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	for _, chunk := range p.chunks {
		if err := sink(chunk); err != nil {
			return nil, err
		}
	}
	return finalTurn(p.final), nil
}

func (p *streamingChatScript) StreamingSupported() bool { return true }
func (p *streamingChatScript) Model() string            { return "streaming-chat-script" }

type providerRecoveryChatPlanner struct {
	failures  int
	successes int
}

func (p *providerRecoveryChatPlanner) Complete(_ context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	for _, message := range messages {
		if strings.Contains(message.Content, "first request") {
			p.failures++
			return nil, errors.New("provider status 429: secret upstream body")
		}
	}
	p.successes++
	return finalTurn("Recovered on the next input."), nil
}

func (p *providerRecoveryChatPlanner) Model() string { return "provider-recovery-chat-planner" }

type turnCapHistoryPlanner struct {
	recorded [][]agent.Message
	calls    int
}

func (p *turnCapHistoryPlanner) Complete(_ context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	cp := append([]agent.Message(nil), messages...)
	p.recorded = append(p.recorded, cp)
	p.calls++
	if p.calls == 1 {
		return toolTurn("get_user_details", `{"user_id":"mia_li_3668"}`), nil
	}
	return finalTurn("Continued after the capped turn."), nil
}

func (p *turnCapHistoryPlanner) Model() string { return "turn-cap-history-planner" }

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

func TestChatDefaultOutputIsCompact(t *testing.T) {
	planner := &chatScript{turns: []*agent.Completion{
		toolTurn("get_user_details", `{"user_id":"private-user-3668"}`),
		finalTurn("Account found."),
	}}

	var out strings.Builder
	runChat(strings.NewReader("find my account\n"), &out, planner, 10)
	got := out.String()
	if strings.Count(got, "Account found.") != 1 {
		t.Fatalf("final answer should appear exactly once:\n%s", got)
	}
	if !strings.Contains(got, "[tool] get_user_details") {
		t.Fatalf("compact output should retain concise tool activity:\n%s", got)
	}
	for _, noisy := range []string{"private-user-3668", "model turns", "engine calls"} {
		if strings.Contains(got, noisy) {
			t.Fatalf("compact output leaked verbose detail %q:\n%s", noisy, got)
		}
	}
}

func TestChatModelPathRendersShortLabel(t *testing.T) {
	planner := &namedChatPlanner{
		chatScript: &chatScript{turns: []*agent.Completion{finalTurn("Ready.")}},
		model:      `/var/lib/fak/models/Qwen3.8-27B-UD-Q2_K_XL.gguf`,
	}

	var out strings.Builder
	runChat(strings.NewReader("hello\n"), &out, planner, 10)
	got := out.String()
	if !strings.Contains(got, "Qwen3.8-27B-UD-Q2_K_XL") {
		t.Fatalf("chat header missing short model label:\n%s", got)
	}
	for _, leaked := range []string{"/var/lib/fak/models", ".gguf"} {
		if strings.Contains(got, leaked) {
			t.Fatalf("chat header leaked model path detail %q:\n%s", leaked, got)
		}
	}
}

func TestChatMultipleToolsRenderOneCompactActivityLine(t *testing.T) {
	planner := &chatScript{turns: []*agent.Completion{
		{Message: agent.Message{ToolCalls: []agent.ToolCall{
			{ID: "a", Function: agent.Func{Name: "get_user_details", Arguments: `{"user_id":"secret-a"}`}},
			{ID: "b", Function: agent.Func{Name: "get_user_details", Arguments: `{"user_id":"secret-b"}`}},
			{ID: "c", Function: agent.Func{Name: "get_user_details", Arguments: `{"user_id":"secret-c"}`}},
		}}},
		finalTurn("Three lookups finished."),
	}}

	var out strings.Builder
	runChat(strings.NewReader("look up three accounts\n"), &out, planner, 10)
	got := out.String()
	if strings.Count(got, "[tools]") != 1 || !strings.Contains(got, "[tools] 3 actions: get_user_details") {
		t.Fatalf("multiple calls should render as one aggregate activity line:\n%s", got)
	}
	for _, leaked := range []string{"secret-a", "secret-b", "secret-c", `{"user_id"`} {
		if strings.Contains(got, leaked) {
			t.Fatalf("aggregate activity leaked raw argument %q:\n%s", leaked, got)
		}
	}
}

func TestChatHelpAndVerboseOptIn(t *testing.T) {
	planner := &recordingChatPlanner{answers: []string{"Verbose answer."}}
	var out strings.Builder
	runChat(strings.NewReader("/help\n/verbose\ninspect this\n"), &out, planner, 10)
	got := out.String()
	if len(planner.recorded) != 1 {
		t.Fatalf("slash commands must not reach the provider; calls=%d output:\n%s", len(planner.recorded), got)
	}
	for _, want := range []string{"/help", "/verbose", "model turns", "engine calls", "Verbose answer."} {
		if !strings.Contains(got, want) {
			t.Fatalf("help/verbose output missing %q:\n%s", want, got)
		}
	}
}

func TestChatStreamingFinalAnswerNotDuplicated(t *testing.T) {
	planner := &streamingChatScript{
		chunks: []string{"Streamed ", "answer."},
		final:  "Streamed answer.",
	}
	var out strings.Builder
	runChat(strings.NewReader("answer me\n"), &out, planner, 10)
	if got := out.String(); strings.Count(got, "Streamed answer.") != 1 {
		t.Fatalf("streamed final answer must render exactly once:\n%s", got)
	}
}

func TestChatTurnCapRendersExplicitIncompleteResponse(t *testing.T) {
	planner := &chatScript{turns: []*agent.Completion{
		toolTurn("get_user_details", `{"user_id":"mia_li_3668"}`),
	}}
	var out strings.Builder
	runChat(strings.NewReader("keep working\n"), &out, planner, 1)
	got := out.String()
	lower := strings.ToLower(got)
	if !strings.Contains(lower, "incomplete") && !strings.Contains(lower, "turn limit") {
		t.Fatalf("turn cap needs an explicit incomplete response:\n%s", got)
	}
	if strings.Contains(got, "fak> \n") {
		t.Fatalf("turn cap must not render a blank assistant response:\n%s", got)
	}
}

func TestChatTurnCapPreservesUserRequestForContinue(t *testing.T) {
	planner := &turnCapHistoryPlanner{}
	var out strings.Builder
	runChat(strings.NewReader("inspect my account\ncontinue\n"), &out, planner, 1)

	if len(planner.recorded) != 2 {
		t.Fatalf("provider calls = %d, want 2; output:\n%s", len(planner.recorded), out.String())
	}
	var sawCappedRequest, sawContinue bool
	for _, message := range planner.recorded[1] {
		if message.Role != agent.RoleUser {
			continue
		}
		sawCappedRequest = sawCappedRequest || strings.Contains(message.Content, "inspect my account")
		sawContinue = sawContinue || strings.Contains(message.Content, "continue")
	}
	if !sawCappedRequest || !sawContinue {
		t.Fatalf("second provider call lost capped-turn context (capped=%v continue=%v):\n%+v", sawCappedRequest, sawContinue, planner.recorded[1])
	}
}

func TestChatProviderFailureIsSanitizedAndNextInputSucceeds(t *testing.T) {
	planner := &providerRecoveryChatPlanner{}
	var out strings.Builder
	runChat(strings.NewReader("first request\nsecond request\n"), &out, planner, 10)
	got := out.String()
	if !strings.Contains(got, "[rate_limited]") || strings.Contains(got, "secret upstream body") {
		t.Fatalf("provider failure was not safely classified:\n%s", got)
	}
	if planner.failures < 2 || planner.successes != 1 || !strings.Contains(got, "Recovered on the next input.") {
		t.Fatalf("REPL did not continue after the failed turn (failures=%d successes=%d):\n%s", planner.failures, planner.successes, got)
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
	if !strings.Contains(got, "[tool] Read") || (!strings.Contains(got, "ALLOW") && !strings.Contains(got, "TRANSFORM")) {
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
	err = runChatHeadless(&out, planner, "read note.txt", 10, false, "", "", agent.WithToolCatalog(catalog))
	if err != nil {
		t.Fatalf("runChatHeadless failed: %v", err)
	}

	got := out.String()
	if strings.Contains(got, "fak chat — native REPL") || strings.Contains(got, "you> ") {
		t.Fatalf("headless mode must not emit interactive REPL chrome:\n%s", got)
	}
	if !strings.Contains(got, "[tool] Read") || (!strings.Contains(got, "ALLOW") && !strings.Contains(got, "TRANSFORM")) {
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

func TestProbeLocalGateway_Success(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"ok":     true,
				"engine": "inkernel",
				"model":  "qwen38:27b-q4",
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer ts.Close()

	model, ok := probeLocalGateway(ts.URL)
	if !ok {
		t.Fatalf("probeLocalGateway(%q) failed, want ok: true", ts.URL)
	}
	if model != "qwen38:27b-q4" {
		t.Fatalf("probeLocalGateway model = %q, want %q", model, "qwen38:27b-q4")
	}
}

func TestProbeLocalGateway_Offline(t *testing.T) {
	// A port that is not listening
	model, ok := probeLocalGateway("http://127.0.0.1:54321")
	if ok || model != "" {
		t.Fatalf("expected probeLocalGateway to return false on closed port, got ok=%v, model=%q", ok, model)
	}
}

// TestProbeGatewayOnce_SuccessAndModelDiscovery exercises the inner probe
// directly: the healthz path yields the served model, and a mock/blank model
// falls through to /v1/models discovery.
func TestProbeGatewayOnce_SuccessAndModelDiscovery(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "model": "qwen38:27b-q4"})
		case "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{{"id": "discovered-model"}},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()

	client := &http.Client{}
	model, ok := probeGatewayOnce(client, ts.URL)
	if !ok {
		t.Fatalf("probeGatewayOnce(%q) failed, want ok: true", ts.URL)
	}
	if model != "qwen38:27b-q4" {
		t.Fatalf("probeGatewayOnce model = %q, want %q", model, "qwen38:27b-q4")
	}
}

func TestProbeGatewayOnce_MockFallsThroughToModels(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "model": "mock"})
		case "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{{"id": "discovered-model"}},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()

	client := &http.Client{}
	model, ok := probeGatewayOnce(client, ts.URL)
	if !ok {
		t.Fatalf("probeGatewayOnce(%q) failed, want ok: true", ts.URL)
	}
	if model != "discovered-model" {
		t.Fatalf("probeGatewayOnce model = %q, want %q", model, "discovered-model")
	}
}

func TestDetectServerModel_FromHealthz(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"ok":     true,
				"engine": "inkernel",
				"model":  "qwen38:27b-q4",
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer ts.Close()

	model := detectServerModel(ts.URL + "/v1")
	if model != "qwen38:27b-q4" {
		t.Fatalf("detectServerModel = %q, want %q", model, "qwen38:27b-q4")
	}
}

func TestDetectServerModel_FromModels(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"ok":     true,
				"engine": "inkernel",
				"model":  "mock",
			})
			return
		}
		if r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{
					{"id": "qwen38:27b-q4", "object": "model"},
				},
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer ts.Close()

	model := detectServerModel(ts.URL + "/v1")
	if model != "qwen38:27b-q4" {
		t.Fatalf("detectServerModel = %q, want %q", model, "qwen38:27b-q4")
	}
}

func TestServeModelDefaultFromGGUFAlias(t *testing.T) {
	rt := &serveRuntime{}
	gguf := "qwen38:27b-q4"
	tok := ""
	baseURL := ""
	model := "mock"
	sf := &serveFlags{
		ggufPath: &gguf,
		tokPath:  &tok,
		baseURL:  &baseURL,
		model:    &model,
	}

	rt.resolveServeModelSources(sf)
	if *sf.model != "qwen38:27b-q4" {
		t.Fatalf("expected sf.model to default to GGUF alias %q, got %q", "qwen38:27b-q4", *sf.model)
	}
}
