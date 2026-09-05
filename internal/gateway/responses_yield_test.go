package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type capturingYieldPlanner struct {
	called   bool
	comp     *agent.Completion
	messages []agent.Message
}

func (p *capturingYieldPlanner) Complete(ctx context.Context, m []agent.Message, t []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.called = true
	p.messages = append([]agent.Message(nil), m...)
	if p.comp != nil {
		return p.comp, nil
	}
	return &agent.Completion{
		FinishReason: "stop",
		Message: agent.Message{
			Role:    agent.RoleAssistant,
			Content: "normal planner output",
		},
	}, nil
}

func (p *capturingYieldPlanner) Model() string {
	return "test-model"
}

// TestResponsesSubturnYieldExceedsThresholds simulates a multi-tool execution turn that
// exceeds token and tool-call thresholds, verifying the gateway activates the yield valve,
// intercepts the turn with the synthetic conclusion and compaction prompt, and sets
// X-Fak-Subturn-Yield: true header.
func TestResponsesSubturnYieldExceedsThresholds(t *testing.T) {
	t.Setenv("FAK_RESPONSES_MAX_SUBTURN_TOKENS", "1000")
	t.Setenv("FAK_RESPONSES_MAX_SUBTURN_TOOL_CALLS", "5")

	srv := newTestServer(t)
	planner := &capturingYieldPlanner{}
	srv.planner = planner

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// 6 tool calls / results with >4000 characters (>1000 tokens).
	inputItems := []any{
		map[string]any{"type": "message", "role": "user", "content": "run long subturn tasks"},
	}
	for i := 0; i < 6; i++ {
		inputItems = append(inputItems,
			map[string]any{
				"type":    "function_call",
				"call_id": "call_" + itoa(uint64(i)),
				"name":    "tool_action",
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_" + itoa(uint64(i)),
				"output":  "tool result data: " + strings.Repeat("A", 800),
			},
		)
	}

	body := map[string]any{
		"model": "test-model",
		"input": inputItems,
	}

	reqBytes, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal body: %v", err)
	}

	httpResp, err := http.Post(ts.URL+"/v1/responses", "application/json", bytes.NewReader(reqBytes))
	if err != nil {
		t.Fatalf("POST /v1/responses: %v", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(httpResp.Body)
		t.Fatalf("expected 200 OK, got %d: %s", httpResp.StatusCode, string(respBody))
	}

	// Verify X-Fak-Subturn-Yield header is set.
	if got := httpResp.Header.Get(SubturnYieldHeader); got != "true" {
		t.Fatalf("expected header %s: true, got %q", SubturnYieldHeader, got)
	}

	// Decode response body.
	var resp responsesResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response JSON: %v", err)
	}

	if resp.Status != "completed" {
		t.Fatalf("expected resp.Status == completed, got %q", resp.Status)
	}
	if resp.FinishReason != "stop" && resp.FinishReason != "yield" {
		t.Fatalf("expected resp.FinishReason stop or yield, got %q", resp.FinishReason)
	}
	if !strings.Contains(resp.OutputText, SubturnYieldMessage) {
		t.Fatalf("expected OutputText to contain %q, got %q", SubturnYieldMessage, resp.OutputText)
	}

	// Verify the upstream planner was NOT called because the turn was intercepted.
	if planner.called {
		t.Fatal("expected upstream planner NOT to be called when yield valve activates")
	}
}

// TestResponsesSubturnYieldBelowThresholds verifies that when context and tool calls are below
// thresholds, normal execution proceeds without yielding.
func TestResponsesSubturnYieldBelowThresholds(t *testing.T) {
	t.Setenv("FAK_RESPONSES_MAX_SUBTURN_TOKENS", "10000")
	t.Setenv("FAK_RESPONSES_MAX_SUBTURN_TOOL_CALLS", "10")

	srv := newTestServer(t)
	planner := &capturingYieldPlanner{
		comp: &agent.Completion{
			FinishReason: "stop",
			Message: agent.Message{
				Role:    agent.RoleAssistant,
				Content: "normal completion below threshold",
			},
		},
	}
	srv.planner = planner

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Only 2 tool calls and modest tokens.
	inputItems := []any{
		map[string]any{"type": "message", "role": "user", "content": "quick task"},
		map[string]any{
			"type":    "function_call",
			"call_id": "call_0",
			"name":    "tool_action",
		},
		map[string]any{
			"type":    "function_call_output",
			"call_id": "call_0",
			"output":  "short output",
		},
	}

	body := map[string]any{
		"model": "test-model",
		"input": inputItems,
	}

	reqBytes, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal body: %v", err)
	}

	httpResp, err := http.Post(ts.URL+"/v1/responses", "application/json", bytes.NewReader(reqBytes))
	if err != nil {
		t.Fatalf("POST /v1/responses: %v", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(httpResp.Body)
		t.Fatalf("expected 200 OK, got %d: %s", httpResp.StatusCode, string(respBody))
	}

	// Verify X-Fak-Subturn-Yield header is NOT set.
	if got := httpResp.Header.Get(SubturnYieldHeader); got != "" {
		t.Fatalf("expected header %s to be empty, got %q", SubturnYieldHeader, got)
	}

	var resp responsesResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response JSON: %v", err)
	}

	if !strings.Contains(resp.OutputText, "normal completion below threshold") {
		t.Fatalf("expected normal output, got %q", resp.OutputText)
	}
	if !planner.called {
		t.Fatal("expected upstream planner to be called for below-threshold request")
	}
}

// TestResponsesSubturnYieldStreaming verifies that streaming Responses requests also
// honor the mid-turn token yield valve.
func TestResponsesSubturnYieldStreaming(t *testing.T) {
	t.Setenv("FAK_RESPONSES_MAX_SUBTURN_TOKENS", "500")
	t.Setenv("FAK_RESPONSES_MAX_SUBTURN_TOOL_CALLS", "3")

	srv := newTestServer(t)
	planner := &capturingYieldPlanner{}
	srv.planner = planner

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	inputItems := []any{
		map[string]any{"type": "message", "role": "user", "content": "stream tasks"},
	}
	for i := 0; i < 4; i++ {
		inputItems = append(inputItems,
			map[string]any{
				"type":    "function_call",
				"call_id": "call_" + itoa(uint64(i)),
				"name":    "tool_action",
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_" + itoa(uint64(i)),
				"output":  "stream output " + strings.Repeat("B", 600),
			},
		)
	}

	body := map[string]any{
		"model":  "test-model",
		"stream": true,
		"input":  inputItems,
	}

	reqBytes, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal body: %v", err)
	}

	httpResp, err := http.Post(ts.URL+"/v1/responses", "application/json", bytes.NewReader(reqBytes))
	if err != nil {
		t.Fatalf("POST /v1/responses: %v", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d", httpResp.StatusCode)
	}

	if got := httpResp.Header.Get(SubturnYieldHeader); got != "true" {
		t.Fatalf("expected header %s: true, got %q", SubturnYieldHeader, got)
	}

	streamBytes, err := io.ReadAll(httpResp.Body)
	if err != nil {
		t.Fatalf("read stream body: %v", err)
	}

	streamStr := string(streamBytes)
	if !strings.Contains(streamStr, SubturnYieldMessage) {
		t.Fatalf("expected stream to contain yield message, got: %s", streamStr)
	}
	if planner.called {
		t.Fatal("expected planner not called on stream yield")
	}
}

// TestSubturnYieldDefaults verifies the default constants and env overrides.
func TestSubturnYieldDefaults(t *testing.T) {
	if DefaultMaxSubturnTokens != 160000 {
		t.Fatalf("expected DefaultMaxSubturnTokens == 160000, got %d", DefaultMaxSubturnTokens)
	}
	if DefaultMaxSubturnToolCalls != 30 {
		t.Fatalf("expected DefaultMaxSubturnToolCalls == 30, got %d", DefaultMaxSubturnToolCalls)
	}

	tokens, calls := resolveSubturnYieldThresholds()
	if tokens != 160000 || calls != 30 {
		t.Fatalf("expected default thresholds (160000, 30), got (%d, %d)", tokens, calls)
	}

	t.Setenv("FAK_RESPONSES_MAX_SUBTURN_TOKENS", "80000")
	t.Setenv("FAK_RESPONSES_MAX_SUBTURN_TOOL_CALLS", "15")
	tokens, calls = resolveSubturnYieldThresholds()
	if tokens != 80000 || calls != 15 {
		t.Fatalf("expected overridden thresholds (80000, 15), got (%d, %d)", tokens, calls)
	}
}

// TestCountSubturnToolCalls verifies counting of tool results and function calls.
func TestCountSubturnToolCalls(t *testing.T) {
	msgs := []agent.Message{
		{Role: agent.RoleUser, Content: "hi"},
		{
			Role: agent.RoleAssistant,
			ToolCalls: []agent.ToolCall{
				{ID: "c1", Function: agent.Func{Name: "fn1"}},
				{ID: "c2", Function: agent.Func{Name: "fn2"}},
			},
		},
		{Role: agent.RoleTool, ToolCallID: "c1", Content: "res1"},
		{Role: agent.RoleTool, ToolCallID: "c2", Content: "res2"},
	}
	count := countSubturnToolCalls(msgs)
	if count != 2 {
		t.Fatalf("expected tool call count 2, got %d", count)
	}
}

// TestSubturnLoopRunawayDetection verifies that runaway sub-turn loops trigger the valve.
func TestSubturnLoopRunawayDetection(t *testing.T) {
	// Case 1: 5 identical consecutive calls.
	var runawayMsgs []agent.Message
	for i := 0; i < 5; i++ {
		runawayMsgs = append(runawayMsgs, agent.Message{
			Role: agent.RoleAssistant,
			ToolCalls: []agent.ToolCall{
				{
					ID: "call_" + itoa(uint64(i)),
					Function: agent.Func{
						Name:      "infinite_loop_tool",
						Arguments: `{"retry":true}`,
					},
				},
			},
		})
	}
	if !detectSubturnRepetitionRunaway(runawayMsgs, 5, 30) {
		t.Fatal("expected runaway loop to be detected for 5 identical consecutive tool calls")
	}

	// Case 2: Diverse calls under runaway threshold should not trigger runaway.
	var normalMsgs []agent.Message
	for i := 0; i < 4; i++ {
		normalMsgs = append(normalMsgs, agent.Message{
			Role: agent.RoleAssistant,
			ToolCalls: []agent.ToolCall{
				{
					ID: "call_" + itoa(uint64(i)),
					Function: agent.Func{
						Name:      "tool_" + itoa(uint64(i)),
						Arguments: `{}`,
					},
				},
			},
		})
	}
	if detectSubturnRepetitionRunaway(normalMsgs, 4, 30) {
		t.Fatal("expected non-repeating calls not to trigger runaway")
	}

	// Case 3: Excessive tool calls exceeding 2x maxToolCalls.
	if !detectSubturnRepetitionRunaway(normalMsgs, 60, 30) {
		t.Fatal("expected tool calls >= 2*maxToolCalls to trigger runaway")
	}
}

// TestShouldYieldSubturnMatrix verifies the threshold combination logic.
func TestShouldYieldSubturnMatrix(t *testing.T) {
	t.Setenv("FAK_RESPONSES_MAX_SUBTURN_TOKENS", "1000")
	t.Setenv("FAK_RESPONSES_MAX_SUBTURN_TOOL_CALLS", "5")

	// 1. High tokens, low calls -> false
	highTokLowCalls := []agent.Message{
		{Role: agent.RoleUser, Content: strings.Repeat("X", 8000)}, // ~2000 tokens
		{
			Role: agent.RoleAssistant,
			ToolCalls: []agent.ToolCall{
				{ID: "c1", Function: agent.Func{Name: "fn1", Arguments: "{}"}},
			},
		},
		{Role: agent.RoleTool, ToolCallID: "c1", Content: "ok"},
	}
	yield, tok, calls := shouldYieldResponsesSubturn(highTokLowCalls, nil, nil)
	if yield {
		t.Fatalf("expected yield=false when tool calls (1) < threshold (5), got tokens=%d calls=%d", tok, calls)
	}

	// 2. Low tokens, high calls (non-runaway) -> false
	var lowTokHighCalls []agent.Message
	for i := 0; i < 6; i++ {
		lowTokHighCalls = append(lowTokHighCalls,
			agent.Message{
				Role: agent.RoleAssistant,
				ToolCalls: []agent.ToolCall{
					{ID: "c" + itoa(uint64(i)), Function: agent.Func{Name: "fn" + itoa(uint64(i))}},
				},
			},
			agent.Message{
				Role:       agent.RoleTool,
				ToolCallID: "c" + itoa(uint64(i)),
				Content:    "ok",
			},
		)
	}
	yield, tok, calls = shouldYieldResponsesSubturn(lowTokHighCalls, nil, nil)
	if yield {
		t.Fatalf("expected yield=false when tokens (%d) < threshold (1000), got calls=%d", tok, calls)
	}

	// 3. High tokens AND high calls -> true
	var highTokHighCalls []agent.Message
	for i := 0; i < 6; i++ {
		highTokHighCalls = append(highTokHighCalls,
			agent.Message{
				Role: agent.RoleAssistant,
				ToolCalls: []agent.ToolCall{
					{ID: "c" + itoa(uint64(i)), Function: agent.Func{Name: "fn" + itoa(uint64(i))}},
				},
			},
			agent.Message{
				Role:       agent.RoleTool,
				ToolCallID: "c" + itoa(uint64(i)),
				Content:    strings.Repeat("Y", 800), // ~200 tokens each => >1200 tokens
			},
		)
	}
	yield, tok, calls = shouldYieldResponsesSubturn(highTokHighCalls, nil, nil)
	if !yield {
		t.Fatalf("expected yield=true when both tokens (%d) >= 1000 and calls (%d) >= 5", tok, calls)
	}
}

// TestEstimateSubturnTokens verifies token estimation.
func TestEstimateSubturnTokens(t *testing.T) {
	msgs := []agent.Message{
		{Role: agent.RoleUser, Content: strings.Repeat("A", 400)}, // 400 chars ~ 101 tokens
	}
	tokens := estimateSubturnTokens(msgs, nil, nil)
	if tokens < 100 || tokens > 110 {
		t.Fatalf("expected ~101 tokens for 400 chars, got %d", tokens)
	}
}

// TestCountSubturnToolCallsScopedToActiveTurn verifies that historical tool calls
// from a previous user turn do not count towards the active sub-turn tool call count.
func TestCountSubturnToolCallsScopedToActiveTurn(t *testing.T) {
	// Turn 1: user prompt, 5 tool calls / results, assistant message concluding turn.
	// Turn 2: user prompt just started (0 tool calls in current turn).
	msgsTurnJustStarted := []agent.Message{
		{Role: agent.RoleUser, Content: "turn 1 start"},
	}
	for i := 0; i < 5; i++ {
		msgsTurnJustStarted = append(msgsTurnJustStarted,
			agent.Message{
				Role: agent.RoleAssistant,
				ToolCalls: []agent.ToolCall{
					{ID: "c" + itoa(uint64(i)), Function: agent.Func{Name: "fn"}},
				},
			},
			agent.Message{
				Role:       agent.RoleTool,
				ToolCallID: "c" + itoa(uint64(i)),
				Content:    "res",
			},
		)
	}
	msgsTurnJustStarted = append(msgsTurnJustStarted,
		agent.Message{Role: agent.RoleAssistant, Content: "turn 1 concluded"},
		agent.Message{Role: agent.RoleUser, Content: "turn 2 start"},
	)

	count := countSubturnToolCalls(msgsTurnJustStarted)
	if count != 0 {
		t.Fatalf("expected active subturn tool call count 0 at turn start, got %d", count)
	}

	// Turn 2 makes 1 tool call.
	msgsOneCall := append(msgsTurnJustStarted,
		agent.Message{
			Role: agent.RoleAssistant,
			ToolCalls: []agent.ToolCall{
				{ID: "t2_c1", Function: agent.Func{Name: "fn_t2"}},
			},
		},
		agent.Message{
			Role:       agent.RoleTool,
			ToolCallID: "t2_c1",
			Content:    "res_t2",
		},
	)
	count = countSubturnToolCalls(msgsOneCall)
	if count != 1 {
		t.Fatalf("expected active subturn tool call count 1, got %d", count)
	}

	// Also test raw input JSON scoping.
	rawInputItems := []any{
		map[string]any{"type": "message", "role": "user", "content": "turn 1"},
	}
	for i := 0; i < 5; i++ {
		rawInputItems = append(rawInputItems,
			map[string]any{"type": "function_call", "call_id": "c" + itoa(uint64(i)), "name": "fn"},
			map[string]any{"type": "function_call_output", "call_id": "c" + itoa(uint64(i)), "output": "res"},
		)
	}
	rawInputItems = append(rawInputItems,
		map[string]any{"type": "message", "role": "assistant", "content": "done turn 1"},
		map[string]any{"type": "message", "role": "user", "content": "turn 2"},
	)
	rawBytes, _ := json.Marshal(rawInputItems)
	rawCount := countSubturnToolCallsFromRaw(rawBytes)
	if rawCount != 0 {
		t.Fatalf("expected raw active subturn tool call count 0 at turn start, got %d", rawCount)
	}
}

// TestResponsesSubturnHistoricalToolCallsDoNotTriggerYield verifies that historical tool calls
// from a previous user turn do not trigger the yield valve on a new turn.
func TestResponsesSubturnHistoricalToolCallsDoNotTriggerYield(t *testing.T) {
	t.Setenv("FAK_RESPONSES_MAX_SUBTURN_TOKENS", "1000")
	t.Setenv("FAK_RESPONSES_MAX_SUBTURN_TOOL_CALLS", "5")

	srv := newTestServer(t)
	planner := &capturingYieldPlanner{
		comp: &agent.Completion{
			FinishReason: "stop",
			Message: agent.Message{
				Role:    agent.RoleAssistant,
				Content: "turn 2 normal output",
			},
		},
	}
	srv.planner = planner

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Turn 1 had 6 tool calls and >1000 tokens of output.
	// Turn 2 just started with a new user message.
	inputItems := []any{
		map[string]any{"type": "message", "role": "user", "content": "turn 1 tasks"},
	}
	for i := 0; i < 6; i++ {
		inputItems = append(inputItems,
			map[string]any{
				"type":    "function_call",
				"call_id": "call_" + itoa(uint64(i)),
				"name":    "tool_action",
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_" + itoa(uint64(i)),
				"output":  "tool result data: " + strings.Repeat("A", 800),
			},
		)
	}
	inputItems = append(inputItems,
		map[string]any{"type": "message", "role": "assistant", "content": "turn 1 complete"},
		map[string]any{"type": "message", "role": "user", "content": "turn 2 task"},
	)

	body := map[string]any{
		"model": "test-model",
		"input": inputItems,
	}

	reqBytes, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal body: %v", err)
	}

	httpResp, err := http.Post(ts.URL+"/v1/responses", "application/json", bytes.NewReader(reqBytes))
	if err != nil {
		t.Fatalf("POST /v1/responses: %v", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(httpResp.Body)
		t.Fatalf("expected 200 OK, got %d: %s", httpResp.StatusCode, string(respBody))
	}

	// Yield header should NOT be set because active sub-turn has 0 tool calls.
	if got := httpResp.Header.Get(SubturnYieldHeader); got != "" {
		t.Fatalf("expected header %s to be empty, got %q", SubturnYieldHeader, got)
	}
	if !planner.called {
		t.Fatal("expected upstream planner to be called because active turn has 0 tool calls")
	}
}

// TestResponsesSubturnSuppressImmediateYieldLoop verifies that when the preceding assistant
// message was SubturnYieldMessage, the next turn is not intercepted by the yield valve.
func TestResponsesSubturnSuppressImmediateYieldLoop(t *testing.T) {
	t.Setenv("FAK_RESPONSES_MAX_SUBTURN_TOKENS", "500")
	t.Setenv("FAK_RESPONSES_MAX_SUBTURN_TOOL_CALLS", "3")

	srv := newTestServer(t)
	planner := &capturingYieldPlanner{
		comp: &agent.Completion{
			FinishReason: "stop",
			Message: agent.Message{
				Role:    agent.RoleAssistant,
				Content: "Here is the requested progress summary after yield.",
			},
		},
	}
	srv.planner = planner

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Turn 1 had 4 tool calls (>3 max) and ended with SubturnYieldMessage.
	// Turn 2 is client's request to summarize.
	inputItems := []any{
		map[string]any{"type": "message", "role": "user", "content": "long task with >500 tokens: " + strings.Repeat("Z", 2500)},
	}
	for i := 0; i < 4; i++ {
		inputItems = append(inputItems,
			map[string]any{"type": "function_call", "call_id": "c" + itoa(uint64(i)), "name": "fn"},
			map[string]any{"type": "function_call_output", "call_id": "c" + itoa(uint64(i)), "output": "res"},
		)
	}
	inputItems = append(inputItems,
		map[string]any{"type": "message", "role": "assistant", "content": SubturnYieldMessage},
		map[string]any{"type": "message", "role": "user", "content": "Please summarize progress so far."},
	)

	body := map[string]any{
		"model": "test-model",
		"input": inputItems,
	}

	reqBytes, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal body: %v", err)
	}

	httpResp, err := http.Post(ts.URL+"/v1/responses", "application/json", bytes.NewReader(reqBytes))
	if err != nil {
		t.Fatalf("POST /v1/responses: %v", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(httpResp.Body)
		t.Fatalf("expected 200 OK, got %d: %s", httpResp.StatusCode, string(respBody))
	}

	// Verify X-Fak-Subturn-Yield header is NOT set on the turn following SubturnYieldMessage.
	if got := httpResp.Header.Get(SubturnYieldHeader); got != "" {
		t.Fatalf("expected header %s to be empty, got %q", SubturnYieldHeader, got)
	}

	var resp responsesResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response JSON: %v", err)
	}

	if !strings.Contains(resp.OutputText, "progress summary after yield") {
		t.Fatalf("expected output to contain summary, got %q", resp.OutputText)
	}
	if !planner.called {
		t.Fatal("expected upstream planner to be called to generate summary")
	}

	// Also verify direct call to shouldYieldResponsesSubturn suppresses yield even if tool calls are present.
	msgs := []agent.Message{
		{Role: agent.RoleUser, Content: strings.Repeat("W", 3000)},
		{Role: agent.RoleAssistant, Content: SubturnYieldMessage},
	}
	for i := 0; i < 5; i++ {
		msgs = append(msgs, agent.Message{
			Role: agent.RoleAssistant,
			ToolCalls: []agent.ToolCall{
				{ID: "c" + itoa(uint64(i)), Function: agent.Func{Name: "fn", Arguments: "{}"}},
			},
		})
	}
	// Here the last assistant messages are tool calls, not SubturnYieldMessage.
	// But if the last assistant message is SubturnYieldMessage (e.g. client sent next user turn or no user turn):
	msgsAfterYield := []agent.Message{
		{Role: agent.RoleUser, Content: strings.Repeat("W", 3000)},
		{Role: agent.RoleAssistant, Content: SubturnYieldMessage},
		{Role: agent.RoleUser, Content: "summarize"},
	}
	shouldYield, _, _ := shouldYieldResponsesSubturn(msgsAfterYield, nil, nil)
	if shouldYield {
		t.Fatal("expected shouldYieldResponsesSubturn to return false when preceding assistant message was SubturnYieldMessage")
	}
}
