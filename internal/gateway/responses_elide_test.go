package gateway

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestResponsesElideOlderToolOutputs proves tool outputs > 1024 bytes in older turns
// are replaced with fak_context_restore markers while the most recent 4 tool results
// remain intact.
func TestResponsesElideOlderToolOutputs(t *testing.T) {
	srv := newTestServer(t)
	const trace = "t-responses-elide-protect"

	bigOutput := strings.Repeat("A", 2000)
	smallOutput := strings.Repeat("B", 500)

	messages := []agent.Message{
		{Role: agent.RoleUser, Content: "turn 0"},
		{Role: agent.RoleTool, ToolCallID: "call_0", Content: bigOutput}, // older, >1024 => elide
		{Role: agent.RoleUser, Content: "turn 1"},
		{Role: agent.RoleTool, ToolCallID: "call_1", Content: smallOutput}, // older, <=1024 => keep
		{Role: agent.RoleUser, Content: "turn 2"},
		{Role: agent.RoleTool, ToolCallID: "call_2", Content: bigOutput}, // recent 4 (4th from end) => keep
		{Role: agent.RoleUser, Content: "turn 3"},
		{Role: agent.RoleTool, ToolCallID: "call_3", Content: bigOutput}, // recent 4 (3rd from end) => keep
		{Role: agent.RoleUser, Content: "turn 4"},
		{Role: agent.RoleTool, ToolCallID: "call_4", Content: bigOutput}, // recent 4 (2nd from end) => keep
		{Role: agent.RoleUser, Content: "turn 5"},
		{Role: agent.RoleTool, ToolCallID: "call_5", Content: bigOutput}, // recent 4 (1st from end) => keep
	}

	elided := srv.maybeElideResponsesToolResults(trace, messages)
	if len(elided) != len(messages) {
		t.Fatalf("expected %d messages, got %d", len(messages), len(elided))
	}

	// Tool 0 should be elided with canonical marker.
	sum := sha256.Sum256([]byte(bigOutput))
	digest := hex.EncodeToString(sum[:])
	wantMarker := fmt.Sprintf("...[fak: tool output elided (len=%d bytes); recover original via fak_context_restore id=sha256:%s]...", len(bigOutput), digest)
	if elided[1].Content != wantMarker {
		t.Fatalf("tool 0 output = %q, want marker %q", elided[1].Content, wantMarker)
	}

	// Tool 1 was <= 1024 bytes, should remain intact.
	if elided[3].Content != smallOutput {
		t.Fatalf("tool 1 output = %q, want %q", elided[3].Content, smallOutput)
	}

	// Tools 2..5 are the recent 4 tool messages, should remain intact even though > 1024 bytes.
	for i, idx := range []int{5, 7, 9, 11} {
		if elided[idx].Content != bigOutput {
			t.Fatalf("recent tool %d (msg index %d) was elided: %q", i, idx, elided[idx].Content)
		}
	}

	// Caller's original messages slice must not be mutated in place.
	if messages[1].Content != bigOutput {
		t.Fatal("caller transcript was mutated in place")
	}
}

// TestResponsesElideRoundTripRestoreStash proves fak_context_restore pages back
// the exact byte-identical tool output from the in-memory stash.
func TestResponsesElideRoundTripRestoreStash(t *testing.T) {
	srv := newTestServer(t)
	const trace = "t-responses-stash-roundtrip"

	origOutput := "tool output payload: " + strings.Repeat("X", 2500)
	messages := []agent.Message{
		{Role: agent.RoleTool, ToolCallID: "call_old", Content: origOutput},
		{Role: agent.RoleTool, ToolCallID: "call_r1", Content: "r1"},
		{Role: agent.RoleTool, ToolCallID: "call_r2", Content: "r2"},
		{Role: agent.RoleTool, ToolCallID: "call_r3", Content: "r3"},
		{Role: agent.RoleTool, ToolCallID: "call_r4", Content: "r4"},
	}

	elided := srv.maybeElideResponsesToolResults(trace, messages)
	marker := elided[0].Content
	if !strings.HasPrefix(marker, "...[fak: tool output elided") {
		t.Fatalf("expected elision marker, got %q", marker)
	}

	re := regexp.MustCompile(`id=(sha256:[0-9a-f]{64})`)
	match := re.FindStringSubmatch(marker)
	if len(match) < 2 {
		t.Fatalf("failed to extract id from marker: %s", marker)
	}
	markerID := match[1]
	cleanDigest := strings.TrimPrefix(markerID, "sha256:")

	// 1. Restore via srv.restoreContext with sha256: prefix.
	res, err := srv.restoreContext("", ContextRestoreRequest{ID: markerID, TraceID: trace})
	if err != nil {
		t.Fatalf("restoreContext with %s failed: %v", markerID, err)
	}
	if res.Bytes != origOutput {
		t.Fatalf("restored bytes mismatch: got %d bytes, want %d bytes", len(res.Bytes), len(origOutput))
	}

	// 2. Restore via srv.restoreContext with bare hex digest.
	resClean, err := srv.restoreContext("", ContextRestoreRequest{ID: cleanDigest, TraceID: trace})
	if err != nil {
		t.Fatalf("restoreContext with %s failed: %v", cleanDigest, err)
	}
	if resClean.Bytes != origOutput {
		t.Fatalf("restored bytes mismatch for clean digest: got %d bytes, want %d bytes", len(resClean.Bytes), len(origOutput))
	}

	// 3. Restore via callMCPTool.
	mcpRes := callMCPTool[CtxRestoreResult](t, srv, "fak_context_restore", map[string]any{"id": markerID, "trace_id": trace})
	if mcpRes.Bytes != origOutput {
		t.Fatalf("MCP restored bytes mismatch: got %d bytes, want %d bytes", len(mcpRes.Bytes), len(origOutput))
	}
}

// TestResponsesElideRoundTripRestoreCAS proves large tool outputs (>= 32 KiB)
// are persisted to durable CAS and can be restored even when the per-trace memory stash misses.
func TestResponsesElideRoundTripRestoreCAS(t *testing.T) {
	srv := newTestServer(t)
	const trace = "t-responses-cas-roundtrip"

	// 35 KiB payload >= 32 KiB threshold.
	origOutput := "LARGE_CAS_TOOL_PAYLOAD: " + strings.Repeat("Z", 35<<10)
	messages := []agent.Message{
		{Role: agent.RoleTool, ToolCallID: "call_big", Content: origOutput},
		{Role: agent.RoleTool, ToolCallID: "call_r1", Content: "r1"},
		{Role: agent.RoleTool, ToolCallID: "call_r2", Content: "r2"},
		{Role: agent.RoleTool, ToolCallID: "call_r3", Content: "r3"},
		{Role: agent.RoleTool, ToolCallID: "call_r4", Content: "r4"},
	}

	elided := srv.maybeElideResponsesToolResults(trace, messages)
	marker := elided[0].Content
	re := regexp.MustCompile(`id=(sha256:[0-9a-f]{64})`)
	match := re.FindStringSubmatch(marker)
	if len(match) < 2 {
		t.Fatalf("failed to extract id from marker: %s", marker)
	}
	markerID := match[1]
	cleanDigest := strings.TrimPrefix(markerID, "sha256:")

	// Restore using an unknown trace ("t-cas-miss-stash") to force a stash miss and CAS resolution.
	res, err := srv.restoreContext("", ContextRestoreRequest{ID: cleanDigest, TraceID: "t-cas-miss-stash"})
	if err != nil {
		t.Fatalf("CAS restore failed on stash miss: %v", err)
	}
	if res.Bytes != origOutput {
		t.Fatalf("CAS restored bytes mismatch: got %d bytes, want %d bytes", len(res.Bytes), len(origOutput))
	}
}

// TestResponsesElideHttpIntegration proves elision runs through POST /v1/responses.
func TestResponsesElideHttpIntegration(t *testing.T) {
	srv := newTestServer(t)
	planner := &capturingResponsesPlanner{
		comp: &agent.Completion{
			Message: agent.Message{Role: agent.RoleAssistant, Content: "done"},
		},
	}
	srv.planner = planner

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	bigOutput := strings.Repeat("W", 1500)
	body := map[string]any{
		"model": "test-model",
		"input": []any{
			map[string]any{"type": "message", "role": "user", "content": "run queries"},
			map[string]any{"type": "function_call_output", "call_id": "c0", "output": bigOutput},
			map[string]any{"type": "function_call_output", "call_id": "c1", "output": "short1"},
			map[string]any{"type": "function_call_output", "call_id": "c2", "output": "short2"},
			map[string]any{"type": "function_call_output", "call_id": "c3", "output": "short3"},
			map[string]any{"type": "function_call_output", "call_id": "c4", "output": "short4"},
		},
	}

	status, resp := postResponses(t, ts.URL, body)
	if status != 200 || resp.Status != "completed" {
		t.Fatalf("expected 200 completed, got status=%d resp=%+v", status, resp)
	}

	// Inspect the messages captured by the planner.
	if len(planner.messages) < 6 {
		t.Fatalf("expected >= 6 messages forwarded to planner, got %d", len(planner.messages))
	}

	// c0 was an older tool result exceeding 1024 bytes; it must have been elided.
	var c0Msg *agent.Message
	for i := range planner.messages {
		if planner.messages[i].ToolCallID == "c0" {
			c0Msg = &planner.messages[i]
			break
		}
	}
	if c0Msg == nil {
		t.Fatal("c0 message not found in forwarded messages")
	}
	if !strings.HasPrefix(c0Msg.Content, "...[fak: tool output elided") {
		t.Fatalf("expected c0 to be elided, got %q", c0Msg.Content)
	}
}

func TestResponsesElideToolAlignment(t *testing.T) {
	srv := newTestServer(t)
	const trace = "t-responses-elide-alignment"

	bigOutput := strings.Repeat("M", 2000)
	messages := []agent.Message{
		{Role: agent.RoleTool, ToolCallID: "c0", Content: bigOutput},
		{Role: agent.RoleTool, ToolCallID: "c1", Content: "r1"},
		{Role: agent.RoleTool, ToolCallID: "c2", Content: "r2"},
		{Role: agent.RoleTool, ToolCallID: "c3", Content: "r3"},
		{Role: agent.RoleTool, ToolCallID: "c4", Content: "r4"},
	}

	elided := srv.maybeElideResponsesToolResults(trace, messages, "mcp__fak__fak_context_restore")
	sum := sha256.Sum256([]byte(bigOutput))
	digest := hex.EncodeToString(sum[:])
	wantMarker := "...[fak: tool output elided (len=2000 bytes); recover original via mcp__fak__fak_context_restore id=sha256:" + digest + "]..."

	if elided[0].Content != wantMarker {
		t.Fatalf("marker mismatch:\ngot:  %q\nwant: %q", elided[0].Content, wantMarker)
	}
}

// TestResponsesSubturnHistoricalToolElisionPreventsPrematureYield verifies that when
// historical tool outputs exceed 160k tokens un-elided, running tool result elision
// compresses context below the threshold so shouldYieldResponsesSubturn does NOT prematurely yield.
func TestResponsesSubturnHistoricalToolElisionPreventsPrematureYield(t *testing.T) {
	srv := newTestServer(t)
	const trace = "t-responses-subturn-elide-prevent-yield"
	t.Setenv("FAK_RESPONSES_SUBTURN_YIELD", "true")

	// 1. Build a multi-step conversation: 1 user turn followed by 32 tool calls and outputs.
	// 28 historical tool calls have oversized outputs (24,000 chars ~ 6,000 tokens each),
	// totaling > 672,000 chars (> 168,000 tokens), which exceeds the 160,000 token default threshold.
	// 4 recent tool calls have small outputs (400 chars each) and represent the active working set.
	const (
		historicalCount = 28
		recentCount     = 4
		totalToolCalls  = historicalCount + recentCount // 32 >= 30 threshold
		largeOutputLen  = 24000
		smallOutputLen  = 400
	)

	var (
		rawInputItems []any
		unelidedMsgs  []agent.Message
		toolDefs      []agent.ToolDef
	)

	rawInputItems = append(rawInputItems, map[string]any{
		"type":    "message",
		"role":    "user",
		"content": "run long multi-step data processing analysis",
	})
	unelidedMsgs = append(unelidedMsgs, agent.Message{
		Role:    agent.RoleUser,
		Content: "run long multi-step data processing analysis",
	})

	for i := 0; i < totalToolCalls; i++ {
		callID := fmt.Sprintf("call_%d", i)
		fnName := fmt.Sprintf("action_step_%d", i)

		var outputText string
		if i < historicalCount {
			outputText = fmt.Sprintf("step %d payload: ", i) + strings.Repeat("D", largeOutputLen)
		} else {
			outputText = fmt.Sprintf("recent step %d: ", i) + strings.Repeat("R", smallOutputLen)
		}

		rawInputItems = append(rawInputItems,
			map[string]any{
				"type":      "function_call",
				"call_id":   callID,
				"name":      fnName,
				"arguments": fmt.Sprintf(`{"step":%d}`, i),
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": callID,
				"output":  outputText,
			},
		)

		unelidedMsgs = append(unelidedMsgs,
			agent.Message{
				Role: agent.RoleAssistant,
				ToolCalls: []agent.ToolCall{{
					ID:       callID,
					Type:     "function",
					Function: agent.Func{Name: fnName, Arguments: fmt.Sprintf(`{"step":%d}`, i)},
				}},
			},
			agent.Message{
				Role:       agent.RoleTool,
				ToolCallID: callID,
				Content:    outputText,
			},
		)

		toolDefs = append(toolDefs, agent.ToolDef{
			Type: "function",
			Function: agent.ToolDefFunction{
				Name:        fnName,
				Description: fmt.Sprintf("step action %d", i),
			},
		})
	}

	rawInputBytes, err := json.Marshal(rawInputItems)
	if err != nil {
		t.Fatalf("marshal rawInputItems: %v", err)
	}

	// 2. Witness the defect: before elision, shouldYieldResponsesSubturn trips prematurely.
	// Both token count (>168k tokens >= 160k threshold) and tool call count (32 >= 30) are met.
	shouldYieldUnelided, estTokensUnelided, callsUnelided := shouldYieldResponsesSubturn(unelidedMsgs, toolDefs, nil)
	if !shouldYieldUnelided {
		t.Fatalf("expected un-elided messages to trip yield valve (est_tokens=%d, tool_calls=%d)", estTokensUnelided, callsUnelided)
	}
	if estTokensUnelided < DefaultMaxSubturnTokens {
		t.Fatalf("expected un-elided estTokens >= %d, got %d", DefaultMaxSubturnTokens, estTokensUnelided)
	}

	// Also verify that passing un-elided rawInput trips the valve even if messages were somehow elided.
	shouldYieldWithRaw, estWithRaw, _ := shouldYieldResponsesSubturn(nil, toolDefs, rawInputBytes)
	if !shouldYieldWithRaw || estWithRaw < DefaultMaxSubturnTokens {
		t.Fatalf("expected rawInput with length %d to trip threshold, got yield=%v estTokens=%d", len(rawInputBytes), shouldYieldWithRaw, estWithRaw)
	}

	// 3. Apply tool result elision: older tool results are stashed and replaced with CAS stubs,
	// while the recent 4 tool results remain intact.
	elidedMsgs := srv.maybeElideResponsesToolResults(trace, unelidedMsgs)
	if len(elidedMsgs) != len(unelidedMsgs) {
		t.Fatalf("expected %d messages after elision, got %d", len(unelidedMsgs), len(elidedMsgs))
	}

	// Verify historical tool outputs were elided to small CAS reference markers.
	historicalElidedCount := 0
	for _, m := range elidedMsgs {
		if m.Role == agent.RoleTool && strings.HasPrefix(m.Content, "...[fak: tool output elided") {
			historicalElidedCount++
		}
	}
	if historicalElidedCount != historicalCount {
		t.Fatalf("expected %d historical tool outputs elided, got %d", historicalCount, historicalElidedCount)
	}

	// 4. Witness the fix: with tool result elision run first and nil rawInput,
	// context tokens compress from >168k to <3k tokens, so shouldYieldResponsesSubturn does NOT yield.
	shouldYieldElided, estTokensElided, callsElided := shouldYieldResponsesSubturn(elidedMsgs, toolDefs, nil)
	if shouldYieldElided {
		t.Fatalf("expected elided messages NOT to yield: est_tokens=%d tool_calls=%d", estTokensElided, callsElided)
	}
	if estTokensElided >= DefaultMaxSubturnTokens {
		t.Fatalf("expected elided estTokens < %d, got %d", DefaultMaxSubturnTokens, estTokensElided)
	}
	if callsElided != totalToolCalls {
		t.Fatalf("expected %d tool calls counted, got %d", totalToolCalls, callsElided)
	}

	// 5. Integration witness via HTTP POST /v1/responses:
	// The real gateway route must run maybeElideResponsesToolResults before shouldYieldResponsesSubturn
	// and pass nil rawInput, allowing the request to proceed to the upstream planner rather than yielding.
	planner := &capturingResponsesPlanner{
		comp: &agent.Completion{
			FinishReason: "stop",
			Message: agent.Message{
				Role:    agent.RoleAssistant,
				Content: "analysis completed successfully without premature yield",
			},
		},
	}
	srv.planner = planner

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := map[string]any{
		"model": "test-model",
		"input": rawInputItems,
	}

	reqBytes, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal body: %v", err)
	}

	httpResp, err := http.Post(ts.URL+"/v1/responses", "application/json", strings.NewReader(string(reqBytes)))
	if err != nil {
		t.Fatalf("POST /v1/responses: %v", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", httpResp.StatusCode)
	}

	// Must NOT set the sub-turn yield header.
	if got := httpResp.Header.Get(SubturnYieldHeader); got == "true" {
		t.Fatalf("premature yield valve activated on HTTP wire: header %s = %q", SubturnYieldHeader, got)
	}

	var resp responsesResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		t.Fatalf("decode responsesResponse: %v", err)
	}

	// Must NOT return the synthetic compaction prompt.
	if resp.OutputText == SubturnYieldMessage {
		t.Fatalf("expected planner completion, but got SubturnYieldMessage: %q", resp.OutputText)
	}
	if !planner.called {
		t.Fatal("expected planner to be called, but it was skipped due to premature yield")
	}
	if resp.OutputText != "analysis completed successfully without premature yield" {
		t.Fatalf("output text mismatch: got %q", resp.OutputText)
	}
}

type capturingResponsesPlanner struct {
	called   bool
	comp     *agent.Completion
	messages []agent.Message
	tools    []agent.ToolDef
}

func (p *capturingResponsesPlanner) Complete(ctx context.Context, m []agent.Message, t []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.called = true
	p.messages = append([]agent.Message(nil), m...)
	p.tools = append([]agent.ToolDef(nil), t...)
	return p.comp, nil
}

func (p *capturingResponsesPlanner) Model() string { return "capturing" }
