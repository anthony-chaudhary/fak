package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/agentopt"
)

func TestDynamicEffort(t *testing.T) {
	sessionID := "sess-dyn-effort-42"

	// Turn 1: User Prompt (initial planning / decomposition) -> budget 2048 / high
	msgsTurn1 := []agent.Message{
		{
			Role:    agent.RoleUser,
			Content: "Plan and decompose the architecture for user authentication service.",
		},
	}
	reqTurn1 := &ChatRequest{
		Model:       "gemini-2.5-flash",
		Messages:    msgsTurn1,
		AffinityKey: sessionID,
	}

	streamBefore1, err := PromptPrefixStreamBytes(reqTurn1.Messages, "", nil)
	if err != nil {
		t.Fatalf("turn 1 prefix stream: %v", err)
	}

	dec1 := ApplyDynamicEffort(reqTurn1)
	if dec1 == nil {
		t.Fatalf("turn 1: expected non-nil decision")
	}
	if dec1.AllocatedBudget != 2048 {
		t.Errorf("turn 1 budget: got %d, want 2048", dec1.AllocatedBudget)
	}
	if dec1.ThinkingBudget != 2048 {
		t.Errorf("turn 1 thinking_budget: got %d, want 2048", dec1.ThinkingBudget)
	}
	if dec1.Effort != EffortHigh {
		t.Errorf("turn 1 effort: got %s, want %s", dec1.Effort, EffortHigh)
	}
	if dec1.Category != CategoryPlanAndDecompose {
		t.Errorf("turn 1 category: got %s, want %s", dec1.Category, CategoryPlanAndDecompose)
	}
	if dec1.VolumeMultiplier() != 1.6 {
		t.Errorf("turn 1 volume multiplier: got %f, want 1.6", dec1.VolumeMultiplier())
	}
	if reqTurn1.ReasoningEffort != "high" {
		t.Errorf("turn 1 reasoning_effort: got %s, want high", reqTurn1.ReasoningEffort)
	}
	if reqTurn1.AffinityKey != sessionID {
		t.Errorf("turn 1 session identity: got %s, want %s", reqTurn1.AffinityKey, sessionID)
	}

	var thinkCfg1 map[string]int
	if err := json.Unmarshal(reqTurn1.ThinkingConfig, &thinkCfg1); err != nil || thinkCfg1["thinkingBudget"] != 2048 {
		t.Errorf("turn 1 thinkingConfig: got %v, want thinkingBudget=2048", thinkCfg1)
	}

	streamAfter1, _ := PromptPrefixStreamBytes(reqTurn1.Messages, "", nil)
	if !bytes.Equal(streamBefore1, streamAfter1) {
		t.Fatalf("turn 1: prompt prefix mutated during dynamic effort modulation")
	}

	// Turn 2: Tool Emission happened, now Tool Result (routine file read output) -> budget 0 / none
	toolCall1 := agent.ToolCall{
		ID: "call_read_1",
		Function: agent.Func{
			Name:      "read_file",
			Arguments: `{"path":"auth.go"}`,
		},
	}
	toolResult1 := agent.Message{
		Role:       agent.RoleTool,
		ToolCallID: "call_read_1",
		Name:       "read_file",
		Content:    "package auth\n\ntype UserService struct{}\n",
	}
	msgsTurn2 := []agent.Message{
		msgsTurn1[0],
		{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{toolCall1}},
		toolResult1,
	}
	reqTurn2 := &ChatRequest{
		Model:       "gemini-2.5-flash",
		Messages:    msgsTurn2,
		AffinityKey: sessionID,
	}

	streamBefore2, err := PromptPrefixStreamBytes(reqTurn2.Messages, "", nil)
	if err != nil {
		t.Fatalf("turn 2 prefix stream: %v", err)
	}
	if !ValidatePrefixCachePreservation(streamAfter1, streamBefore2) {
		t.Fatalf("turn 2: prompt prefix does not contain turn 1 prefix as bit-identical leading slice")
	}

	dec2 := ApplyDynamicEffort(reqTurn2)
	if dec2 == nil {
		t.Fatalf("turn 2: expected non-nil decision")
	}
	if dec2.AllocatedBudget != 0 {
		t.Errorf("turn 2 budget: got %d, want 0", dec2.AllocatedBudget)
	}
	if dec2.ThinkingBudget != 0 {
		t.Errorf("turn 2 thinking_budget: got %d, want 0", dec2.ThinkingBudget)
	}
	if dec2.Effort != EffortNone {
		t.Errorf("turn 2 effort: got %s, want %s", dec2.Effort, EffortNone)
	}
	if dec2.Category != CategoryRoutineTool {
		t.Errorf("turn 2 category: got %s, want %s", dec2.Category, CategoryRoutineTool)
	}
	if dec2.VolumeMultiplier() != 0.4 {
		t.Errorf("turn 2 volume multiplier: got %f, want 0.4", dec2.VolumeMultiplier())
	}
	if reqTurn2.ReasoningEffort != "none" {
		t.Errorf("turn 2 reasoning_effort: got %s, want none", reqTurn2.ReasoningEffort)
	}
	if reqTurn2.AffinityKey != sessionID {
		t.Errorf("turn 2 session identity: got %s, want %s", reqTurn2.AffinityKey, sessionID)
	}

	var thinkCfg2 map[string]int
	if err := json.Unmarshal(reqTurn2.ThinkingConfig, &thinkCfg2); err != nil || thinkCfg2["thinkingBudget"] != 0 {
		t.Errorf("turn 2 thinkingConfig: got %v, want thinkingBudget=0", thinkCfg2)
	}

	streamAfter2, _ := PromptPrefixStreamBytes(reqTurn2.Messages, "", nil)
	if !bytes.Equal(streamBefore2, streamAfter2) {
		t.Fatalf("turn 2: prompt prefix mutated during dynamic effort modulation")
	}

	// Turn 3: Next Tool Call emitted (subsequent tool output: routine list_dir) -> budget 0 / none
	toolCall2 := agent.ToolCall{
		ID: "call_list_2",
		Function: agent.Func{
			Name:      "list_dir",
			Arguments: `{"path":"."}`,
		},
	}
	toolResult2 := agent.Message{
		Role:       agent.RoleTool,
		ToolCallID: "call_list_2",
		Name:       "list_dir",
		Content:    "auth.go\nauth_test.go\nconfig.go\n",
	}
	msgsTurn3 := []agent.Message{
		msgsTurn2[0],
		msgsTurn2[1],
		msgsTurn2[2],
		{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{toolCall2}},
		toolResult2,
	}
	reqTurn3 := &ChatRequest{
		Model:       "gemini-2.5-flash",
		Messages:    msgsTurn3,
		AffinityKey: sessionID,
	}

	streamBefore3, err := PromptPrefixStreamBytes(reqTurn3.Messages, "", nil)
	if err != nil {
		t.Fatalf("turn 3 prefix stream: %v", err)
	}
	if !ValidatePrefixCachePreservation(streamAfter2, streamBefore3) {
		t.Fatalf("turn 3: prompt prefix does not contain turn 2 prefix as bit-identical leading slice")
	}

	dec3 := ApplyDynamicEffort(reqTurn3)
	if dec3 == nil {
		t.Fatalf("turn 3: expected non-nil decision")
	}
	if dec3.AllocatedBudget != 0 {
		t.Errorf("turn 3 budget: got %d, want 0", dec3.AllocatedBudget)
	}
	if dec3.ThinkingBudget != 0 {
		t.Errorf("turn 3 thinking_budget: got %d, want 0", dec3.ThinkingBudget)
	}
	if dec3.Effort != EffortNone {
		t.Errorf("turn 3 effort: got %s, want %s", dec3.Effort, EffortNone)
	}
	if dec3.Category != CategoryRoutineTool {
		t.Errorf("turn 3 category: got %s, want %s", dec3.Category, CategoryRoutineTool)
	}
	if dec3.VolumeMultiplier() != 0.4 {
		t.Errorf("turn 3 volume multiplier: got %f, want 0.4", dec3.VolumeMultiplier())
	}
	if reqTurn3.ReasoningEffort != "none" {
		t.Errorf("turn 3 reasoning_effort: got %s, want none", reqTurn3.ReasoningEffort)
	}
	if reqTurn3.AffinityKey != sessionID {
		t.Errorf("turn 3 session identity: got %s, want %s", reqTurn3.AffinityKey, sessionID)
	}

	var thinkCfg3 map[string]int
	if err := json.Unmarshal(reqTurn3.ThinkingConfig, &thinkCfg3); err != nil || thinkCfg3["thinkingBudget"] != 0 {
		t.Errorf("turn 3 thinkingConfig: got %v, want thinkingBudget=0", thinkCfg3)
	}

	streamAfter3, _ := PromptPrefixStreamBytes(reqTurn3.Messages, "", nil)
	if !bytes.Equal(streamBefore3, streamAfter3) {
		t.Fatalf("turn 3: prompt prefix mutated during dynamic effort modulation")
	}

	// Turn 4: Test failure in tool output -> budget 2048 / high (error recovery)
	toolCall3 := agent.ToolCall{
		ID: "call_test_3",
		Function: agent.Func{
			Name:      "run_tests",
			Arguments: `{"pkg":"./auth"}`,
		},
	}
	toolResult3 := agent.Message{
		Role:       agent.RoleTool,
		ToolCallID: "call_test_3",
		Name:       "run_tests",
		Content:    "--- FAIL: TestUserAuth (0.02s)\n    auth_test.go:42: assertion failed: invalid token\nFAIL\nexit status 1",
	}
	msgsTurn4 := []agent.Message{
		msgsTurn3[0],
		msgsTurn3[1],
		msgsTurn3[2],
		msgsTurn3[3],
		msgsTurn3[4],
		{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{toolCall3}},
		toolResult3,
	}
	reqTurn4 := &ChatRequest{
		Model:       "gemini-2.5-flash",
		Messages:    msgsTurn4,
		AffinityKey: sessionID,
	}

	streamBefore4, err := PromptPrefixStreamBytes(reqTurn4.Messages, "", nil)
	if err != nil {
		t.Fatalf("turn 4 prefix stream: %v", err)
	}
	if !ValidatePrefixCachePreservation(streamAfter3, streamBefore4) {
		t.Fatalf("turn 4: prompt prefix does not contain turn 3 prefix as bit-identical leading slice")
	}

	dec4 := ApplyDynamicEffort(reqTurn4)
	if dec4 == nil {
		t.Fatalf("turn 4: expected non-nil decision")
	}
	if dec4.AllocatedBudget != 2048 {
		t.Errorf("turn 4 budget: got %d, want 2048", dec4.AllocatedBudget)
	}
	if dec4.ThinkingBudget != 2048 {
		t.Errorf("turn 4 thinking_budget: got %d, want 2048", dec4.ThinkingBudget)
	}
	if dec4.Effort != EffortHigh {
		t.Errorf("turn 4 effort: got %s, want %s", dec4.Effort, EffortHigh)
	}
	if dec4.Category != CategoryErrorRecovery {
		t.Errorf("turn 4 category: got %s, want %s", dec4.Category, CategoryErrorRecovery)
	}
	if dec4.VolumeMultiplier() != 1.6 {
		t.Errorf("turn 4 volume multiplier: got %f, want 1.6", dec4.VolumeMultiplier())
	}
	if reqTurn4.ReasoningEffort != "high" {
		t.Errorf("turn 4 reasoning_effort: got %s, want high", reqTurn4.ReasoningEffort)
	}
	if reqTurn4.AffinityKey != sessionID {
		t.Errorf("turn 4 session identity: got %s, want %s", reqTurn4.AffinityKey, sessionID)
	}

	var thinkCfg4 map[string]int
	if err := json.Unmarshal(reqTurn4.ThinkingConfig, &thinkCfg4); err != nil || thinkCfg4["thinkingBudget"] != 2048 {
		t.Errorf("turn 4 thinkingConfig: got %v, want thinkingBudget=2048", thinkCfg4)
	}

	streamAfter4, _ := PromptPrefixStreamBytes(reqTurn4.Messages, "", nil)
	if !bytes.Equal(streamBefore4, streamAfter4) {
		t.Fatalf("turn 4: prompt prefix mutated during dynamic effort modulation")
	}

	// Turn 5: Final Synthesis prompt -> budget 1024 / medium (CategoryDiagnostic)
	msgsTurn5 := []agent.Message{
		msgsTurn4[0],
		msgsTurn4[1],
		msgsTurn4[2],
		msgsTurn4[3],
		msgsTurn4[4],
		msgsTurn4[5],
		msgsTurn4[6],
		{
			Role:    agent.RoleAssistant,
			Content: "Resolved the auth validation error and verified all assertions pass.",
		},
		{
			Role:    agent.RoleUser,
			Content: "Synthesize all findings and provide a final report on implementation.",
		},
	}
	reqTurn5 := &ChatRequest{
		Model:       "gemini-2.5-flash",
		Messages:    msgsTurn5,
		AffinityKey: sessionID,
	}

	streamBefore5, err := PromptPrefixStreamBytes(reqTurn5.Messages, "", nil)
	if err != nil {
		t.Fatalf("turn 5 prefix stream: %v", err)
	}
	if !ValidatePrefixCachePreservation(streamAfter4, streamBefore5) {
		t.Fatalf("turn 5: prompt prefix does not contain turn 4 prefix as bit-identical leading slice")
	}

	dec5 := ApplyDynamicEffort(reqTurn5)
	if dec5 == nil {
		t.Fatalf("turn 5: expected non-nil decision")
	}
	if dec5.AllocatedBudget != 1024 {
		t.Errorf("turn 5 budget: got %d, want 1024", dec5.AllocatedBudget)
	}
	if dec5.ThinkingBudget != 1024 {
		t.Errorf("turn 5 thinking_budget: got %d, want 1024", dec5.ThinkingBudget)
	}
	if dec5.Effort != EffortMedium {
		t.Errorf("turn 5 effort: got %s, want %s", dec5.Effort, EffortMedium)
	}
	if dec5.Category != CategoryDiagnostic {
		t.Errorf("turn 5 category: got %s, want %s", dec5.Category, CategoryDiagnostic)
	}
	if dec5.VolumeMultiplier() != 1.0 {
		t.Errorf("turn 5 volume multiplier: got %f, want 1.0", dec5.VolumeMultiplier())
	}
	if reqTurn5.ReasoningEffort != "medium" {
		t.Errorf("turn 5 reasoning_effort: got %s, want medium", reqTurn5.ReasoningEffort)
	}
	if reqTurn5.AffinityKey != sessionID {
		t.Errorf("turn 5 session identity: got %s, want %s", reqTurn5.AffinityKey, sessionID)
	}

	var thinkCfg5 map[string]int
	if err := json.Unmarshal(reqTurn5.ThinkingConfig, &thinkCfg5); err != nil || thinkCfg5["thinkingBudget"] != 1024 {
		t.Errorf("turn 5 thinkingConfig: got %v, want thinkingBudget=1024", thinkCfg5)
	}

	streamAfter5, _ := PromptPrefixStreamBytes(reqTurn5.Messages, "", nil)
	if !bytes.Equal(streamBefore5, streamAfter5) {
		t.Fatalf("turn 5: prompt prefix mutated during dynamic effort modulation")
	}

	// Verify the exact budget sequence: [2048, 0, 0, 2048, 1024]
	gotBudgets := []int{
		dec1.AllocatedBudget,
		dec2.AllocatedBudget,
		dec3.AllocatedBudget,
		dec4.AllocatedBudget,
		dec5.AllocatedBudget,
	}
	wantBudgets := []int{2048, 0, 0, 2048, 1024}
	if !reflect.DeepEqual(gotBudgets, wantBudgets) {
		t.Fatalf("thinkingBudget trajectory mismatch:\n  got:  %v\n  want: %v", gotBudgets, wantBudgets)
	}

	// Verify the exact effort sequence: [high, none, none, high, medium]
	gotEfforts := []string{
		string(dec1.Effort),
		string(dec2.Effort),
		string(dec3.Effort),
		string(dec4.Effort),
		string(dec5.Effort),
	}
	wantEfforts := []string{"high", "none", "none", "high", "medium"}
	if !reflect.DeepEqual(gotEfforts, wantEfforts) {
		t.Fatalf("effort trajectory mismatch:\n  got:  %v\n  want: %v", gotEfforts, wantEfforts)
	}

	// Verify session continuity (AffinityKey preserved across all turns)
	turns := []*ChatRequest{reqTurn1, reqTurn2, reqTurn3, reqTurn4, reqTurn5}
	for i, req := range turns {
		if req.AffinityKey != sessionID {
			t.Errorf("turn %d affinity key: got %s, want %s", i+1, req.AffinityKey, sessionID)
		}
	}

	// Verify DynamicEffortDecision metrics across all decisions
	decisions := []*DynamicEffortDecision{dec1, dec2, dec3, dec4, dec5}
	for i, dec := range decisions {
		if dec.Reason == "" {
			t.Errorf("turn %d decision reason is empty", i+1)
		}
		if dec.AllocatedBudget != wantBudgets[i] {
			t.Errorf("turn %d decision allocated budget: got %d, want %d", i+1, dec.AllocatedBudget, wantBudgets[i])
		}
		if dec.ThinkingBudget != wantBudgets[i] {
			t.Errorf("turn %d decision thinking budget: got %d, want %d", i+1, dec.ThinkingBudget, wantBudgets[i])
		}
	}

	// Verify reasoning effort overrides when caller sets an initial divergent effort
	reqOverride := &ChatRequest{
		Model:           "gemini-2.5-flash",
		ReasoningEffort: "low",
		Messages:        msgsTurn4,
		AffinityKey:     sessionID,
	}
	decOverride := ApplyDynamicEffort(reqOverride)
	if decOverride == nil {
		t.Fatalf("override check: expected non-nil decision")
	}
	if reqOverride.ReasoningEffort != "high" {
		t.Errorf("reasoning effort override: expected 'high', got %s", reqOverride.ReasoningEffort)
	}
	if decOverride.AllocatedBudget != 2048 {
		t.Errorf("reasoning effort override budget: got %d, want 2048", decOverride.AllocatedBudget)
	}

	// Verify message preservation
	if !ValidateMessagesPreserved(msgsTurn1, msgsTurn5) {
		t.Errorf("turn 1 messages not byte-preserved in turn 5")
	}
	if !ValidateMessagesPreserved(msgsTurn2, msgsTurn5) {
		t.Errorf("turn 2 messages not byte-preserved in turn 5")
	}
	if !ValidateMessagesPreserved(msgsTurn3, msgsTurn5) {
		t.Errorf("turn 3 messages not byte-preserved in turn 5")
	}
	if !ValidateMessagesPreserved(msgsTurn4, msgsTurn5) {
		t.Errorf("turn 4 messages not byte-preserved in turn 5")
	}
}

func TestDynamicEffort_TurnLevelSequence(t *testing.T) {
	TestDynamicEffort(t)
}

func TestDynamicEffort_DiagnosticVerification(t *testing.T) {
	msgs := []agent.Message{
		{Role: agent.RoleUser, Content: "Run tests and verify output"},
		{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "call_t", Function: agent.Func{Name: "test_runner"}}}},
		{Role: agent.RoleTool, ToolCallID: "call_t", Content: "PASS: all 42 tests passed in 0.12s\nok  fak/pkg 0.15s"},
	}
	req := &ChatRequest{
		Model:    "o3-mini",
		Messages: msgs,
	}
	dec := ApplyDynamicEffort(req)
	if dec == nil {
		t.Fatalf("expected non-nil decision")
	}
	if dec.Category != agentopt.CategoryDiagnosticVerify {
		t.Errorf("category: got %s, want %s", dec.Category, agentopt.CategoryDiagnosticVerify)
	}
	if dec.Effort != agentopt.EffortLow {
		t.Errorf("effort: got %s, want %s", dec.Effort, agentopt.EffortLow)
	}
	if dec.AllocatedBudget != 256 {
		t.Errorf("budget: got %d, want 256", dec.AllocatedBudget)
	}
}

func TestDynamicEffort_GeminiJSONModulation(t *testing.T) {
	raw := []byte(`{
		"contents": [
			{"role": "user", "parts": [{"text": "Plan the database migration."}]}
		],
		"generationConfig": {
			"temperature": 0.2
		},
		"session_id": "gemini-session-999"
	}`)

	prefixBefore, err := ExtractPromptPrefixFromJSON(raw)
	if err != nil {
		t.Fatalf("extract prefix: %v", err)
	}

	modulated, dec, err := ApplyDynamicEffortToGeminiJSON(raw, "gemini-2.5-flash")
	if err != nil {
		t.Fatalf("modulate gemini json: %v", err)
	}
	if dec.AllocatedBudget != 2048 {
		t.Errorf("budget: got %d, want 2048", dec.AllocatedBudget)
	}

	// Verify prefix bytes remain bit-identical
	prefixAfter, err := ExtractPromptPrefixFromJSON(modulated)
	if err != nil {
		t.Fatalf("extract prefix after: %v", err)
	}
	if !bytes.Equal(prefixBefore, prefixAfter) {
		t.Fatalf("gemini json: prompt prefix mutated during modulation")
	}

	// Verify generationConfig.thinkingConfig injected
	var root map[string]json.RawMessage
	if err := json.Unmarshal(modulated, &root); err != nil {
		t.Fatalf("unmarshal modulated: %v", err)
	}
	var genCfg map[string]json.RawMessage
	if err := json.Unmarshal(root["generationConfig"], &genCfg); err != nil {
		t.Fatalf("unmarshal generationConfig: %v", err)
	}
	var thinkCfg map[string]int
	if err := json.Unmarshal(genCfg["thinkingConfig"], &thinkCfg); err != nil {
		t.Fatalf("unmarshal thinkingConfig: %v", err)
	}
	if thinkCfg["thinkingBudget"] != 2048 {
		t.Errorf("thinkingBudget: got %d, want 2048", thinkCfg["thinkingBudget"])
	}

	// Verify session identity preserved
	var sessID string
	_ = json.Unmarshal(root["session_id"], &sessID)
	if sessID != "gemini-session-999" {
		t.Errorf("session_id: got %s, want gemini-session-999", sessID)
	}

	// Modulate with zero budget -> thinkingBudget must be explicitly 0
	zeroDec := agentopt.TurnEffortDecision{
		Effort:          agentopt.EffortNone,
		Category:        agentopt.CategoryRoutineTool,
		AllocatedBudget: 0,
	}
	zeroMod, err := ModulateGeminiJSON(modulated, zeroDec)
	if err != nil {
		t.Fatalf("modulate zero: %v", err)
	}
	var zeroRoot map[string]json.RawMessage
	_ = json.Unmarshal(zeroMod, &zeroRoot)
	var zeroGenCfg map[string]json.RawMessage
	_ = json.Unmarshal(zeroRoot["generationConfig"], &zeroGenCfg)
	var zeroThinkCfg map[string]int
	_ = json.Unmarshal(zeroGenCfg["thinkingConfig"], &zeroThinkCfg)
	if zeroThinkCfg["thinkingBudget"] != 0 {
		t.Errorf("zero budget: got %d, want 0", zeroThinkCfg["thinkingBudget"])
	}
}

func TestDynamicEffort_OpenAIJSONModulation(t *testing.T) {
	raw := []byte(`{
		"model": "o3-mini",
		"messages": [{"role": "user", "content": "Plan and decompose the task"}],
		"session_id": "sess-openai-88"
	}`)

	prefixBefore, _ := ExtractPromptPrefixFromJSON(raw)
	modulated, dec, err := ApplyDynamicEffortToOpenAIJSON(raw, "o3-mini")
	if err != nil {
		t.Fatalf("apply openai: %v", err)
	}
	if dec.Effort != agentopt.EffortHigh {
		t.Errorf("effort: got %s, want high", dec.Effort)
	}

	prefixAfter, _ := ExtractPromptPrefixFromJSON(modulated)
	if !bytes.Equal(prefixBefore, prefixAfter) {
		t.Fatalf("openai json: prompt prefix was mutated")
	}

	var root map[string]json.RawMessage
	_ = json.Unmarshal(modulated, &root)
	var effort string
	_ = json.Unmarshal(root["reasoning_effort"], &effort)
	if effort != "high" {
		t.Errorf("reasoning_effort: got %s, want high", effort)
	}

	var sess string
	_ = json.Unmarshal(root["session_id"], &sess)
	if sess != "sess-openai-88" {
		t.Errorf("session_id: got %s, want sess-openai-88", sess)
	}
}

func TestDynamicEffort_AnthropicJSONModulation(t *testing.T) {
	raw := []byte(`{
		"model": "claude-3-7-sonnet-20250219",
		"messages": [{"role": "user", "content": "Plan and break down the project."}]
	}`)

	prefixBefore, _ := ExtractPromptPrefixFromJSON(raw)
	modulated, dec, err := ApplyDynamicEffortToAnthropicJSON(raw, "claude-3-7-sonnet-20250219")
	if err != nil {
		t.Fatalf("anthropic modulate: %v", err)
	}
	if dec.AllocatedBudget != 2048 {
		t.Errorf("budget: got %d, want 2048", dec.AllocatedBudget)
	}

	prefixAfter, _ := ExtractPromptPrefixFromJSON(modulated)
	if !bytes.Equal(prefixBefore, prefixAfter) {
		t.Fatalf("anthropic json: prompt prefix was mutated")
	}

	var root map[string]json.RawMessage
	_ = json.Unmarshal(modulated, &root)
	var thinkCfg struct {
		Type   string `json:"type"`
		Budget int    `json:"budget_tokens"`
	}
	_ = json.Unmarshal(root["thinking"], &thinkCfg)
	if thinkCfg.Type != "enabled" || thinkCfg.Budget != 2048 {
		t.Errorf("thinking: got %+v, want enabled/2048", thinkCfg)
	}

	// Disable thinking (zero budget)
	zeroDec := agentopt.TurnEffortDecision{
		Effort:          agentopt.EffortNone,
		AllocatedBudget: 0,
	}
	zeroMod, _ := ModulateAnthropicJSON(raw, zeroDec)
	var zeroRoot map[string]json.RawMessage
	_ = json.Unmarshal(zeroMod, &zeroRoot)
	var zeroThinkCfg struct {
		Type string `json:"type"`
	}
	_ = json.Unmarshal(zeroRoot["thinking"], &zeroThinkCfg)
	if zeroThinkCfg.Type != "disabled" {
		t.Errorf("zero thinking: got %s, want disabled", zeroThinkCfg.Type)
	}
}

func TestDynamicEffort_ModelSupportDetection(t *testing.T) {
	supported := []string{
		"gemini-2.0-flash-thinking",
		"gemini-2.5-pro",
		"gemini-3.8-flash",
		"o1",
		"o1-mini",
		"o1-preview",
		"o3",
		"o3-mini",
		"o4",
		"o4-mini",
		"o5",
		"o5-mini",
		"o6",
		"gpt-6-astra",
		"gpt-7-astra",
		"openai/o5",
		"openai/gpt-6-astra",
		"claude-3-7-sonnet",
		"claude-3.7-sonnet-20250219",
		"claude-4-sonnet",
		"custom-reasoning-model",
	}
	for _, m := range supported {
		if !ModelSupportsThinking(m) {
			t.Errorf("expected %q to support thinking", m)
		}
	}

	unsupported := []string{
		"gpt-4o",
		"gpt-4o-mini",
		"claude-3-5-sonnet-20241022",
		"llama-3.3-70b",
		"mistral-large",
	}
	for _, m := range unsupported {
		if ModelSupportsThinking(m) {
			t.Errorf("expected %q to NOT support thinking natively", m)
		}
	}

	// Explicit override
	if !RequestSupportsThinking("gpt-4o", true) {
		t.Errorf("explicit thinking config must be supported regardless of model name")
	}
}

func TestDynamicEffort_ServerHookIntegration(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, err := json.Marshal(ChatRequest{
		Model: "o3-mini",
		Messages: []agent.Message{
			{Role: agent.RoleUser, Content: "Plan the implementation of cache manager"},
		},
		AffinityKey: "sess-integration-1",
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post /v1/chat/completions: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d", resp.StatusCode)
	}
}
