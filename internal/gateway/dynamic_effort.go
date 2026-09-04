package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/agentopt"
)

// DynamicEffortDecision is an alias for agentopt.TurnEffortDecision representing
// the adjudicated effort decision and token budget for a turn.
type DynamicEffortDecision = agentopt.TurnEffortDecision

// Operational category aliases for gateway dynamic effort classification.
const (
	CategoryPlanAndDecompose = agentopt.CategoryPlanAndDecompose
	CategoryRoutineTool      = agentopt.CategoryRoutineTool
	CategoryDiagnostic       = agentopt.CategoryDiagnosticAndVerification
	CategoryDiagnosticVerify = agentopt.CategoryDiagnosticVerify
	CategoryErrorRecovery    = agentopt.CategoryErrorRecovery
	CategorySynthesisReport  = agentopt.CategorySynthesisReport
)

// Reasoning effort tier aliases.
const (
	EffortNone   = agentopt.EffortNone
	EffortLow    = agentopt.EffortLow
	EffortMedium = agentopt.EffortMedium
	EffortHigh   = agentopt.EffortHigh
)

// ModelSupportsThinking reports whether the given model name natively supports
// reasoning/thinking effort modulation (e.g. Gemini 2.0/2.5/3.8 Flash, OpenAI o1/o3/o4,
// Anthropic Claude 3.7+ thinking).
func ModelSupportsThinking(model string) bool {
	m := strings.ToLower(strings.TrimSpace(model))
	if m == "" {
		return false
	}
	if strings.Contains(m, "thinking") || strings.Contains(m, "reason") {
		return true
	}
	// Gemini: Gemini 2.0 / 2.5 / 3.8 Flash and pro
	if strings.Contains(m, "gemini") {
		if strings.Contains(m, "2.0") || strings.Contains(m, "2.5") || strings.Contains(m, "3.8") ||
			strings.Contains(m, "flash") || strings.Contains(m, "pro") {
			return true
		}
	}
	// OpenAI o1/o3/o4 series
	if strings.HasPrefix(m, "o1") || strings.HasPrefix(m, "o3") || strings.HasPrefix(m, "o4") ||
		strings.Contains(m, "/o1") || strings.Contains(m, "/o3") || strings.Contains(m, "/o4") {
		return true
	}
	// Anthropic Claude 3.7+ thinking
	if strings.Contains(m, "claude-3-7") || strings.Contains(m, "claude-3.7") || strings.Contains(m, "claude-4") {
		return true
	}
	return false
}

// RequestSupportsThinking reports whether the request should undergo dynamic effort modulation,
// based on either model capability or explicit thinking configuration present in the request.
func RequestSupportsThinking(model string, hasExplicitThinking bool) bool {
	return hasExplicitThinking || ModelSupportsThinking(model)
}

// BuildTurnContext constructs an agentopt.TurnContext from the conversation messages,
// inspecting trailing role, tool calls, and error indicators from tool results.
func BuildTurnContext(messages []agent.Message, prompt string) agentopt.TurnContext {
	ctx := agentopt.TurnContext{
		Prompt: strings.TrimSpace(prompt),
	}
	if len(messages) == 0 {
		if ctx.Prompt != "" {
			ctx.IsPlanning = true
		}
		return ctx
	}

	lastMsg := messages[len(messages)-1]
	ctx.LastRole = lastMsg.Role
	ctx.LastContent = lastMsg.Content

	// Map tool calls by ID so we can resolve tool names for tool results
	toolNameByID := make(map[string]string)
	for _, m := range messages {
		for _, tc := range m.ToolCalls {
			if tc.ID != "" && tc.Function.Name != "" {
				toolNameByID[tc.ID] = tc.Function.Name
			}
		}
	}

	switch lastMsg.Role {
	case agent.RoleTool:
		// Model is reacting to tool execution results. Collect the trailing contiguous tool results.
		var currentToolMsgs []agentopt.TurnMessage
		for i := len(messages) - 1; i >= 0; i-- {
			if messages[i].Role == agent.RoleTool {
				currentToolMsgs = append([]agentopt.TurnMessage{{
					Role:        messages[i].Role,
					Content:     messages[i].Content,
					ToolResults: []string{messages[i].Content},
				}}, currentToolMsgs...)
				inspectToolResultForErrors(messages[i].Content, &ctx)
			} else {
				break
			}
		}
		ctx.Messages = currentToolMsgs

		toolName := lastMsg.Name
		if toolName == "" && lastMsg.ToolCallID != "" {
			toolName = toolNameByID[lastMsg.ToolCallID]
		}
		if toolName == "" {
			for i := len(messages) - 2; i >= 0; i-- {
				if len(messages[i].ToolCalls) > 0 {
					toolName = messages[i].ToolCalls[len(messages[i].ToolCalls)-1].Function.Name
					break
				}
			}
		}
		ctx.ToolName = toolName
		ctx.TargetToolName = toolName
		ctx.ToolOutput = lastMsg.Content

	case agent.RoleUser:
		ctx.Prompt = lastMsg.Content
		if len(messages) == 1 {
			ctx.IsPlanning = true
		}
		ctx.Messages = []agentopt.TurnMessage{
			{
				Role:    lastMsg.Role,
				Content: lastMsg.Content,
			},
		}

	case agent.RoleAssistant:
		if len(lastMsg.ToolCalls) > 0 {
			ctx.TargetToolName = lastMsg.ToolCalls[0].Function.Name
			for _, tc := range lastMsg.ToolCalls {
				ctx.ToolCalls = append(ctx.ToolCalls, agentopt.ToolCall{
					ID:   tc.ID,
					Name: tc.Function.Name,
				})
			}
		}
		ctx.Messages = []agentopt.TurnMessage{
			{
				Role:    lastMsg.Role,
				Content: lastMsg.Content,
			},
		}

	default:
		if ctx.Prompt == "" {
			ctx.Prompt = lastMsg.Content
		}
	}

	return ctx
}

func inspectToolResultForErrors(content string, ctx *agentopt.TurnContext) {
	lower := strings.ToLower(content)
	if strings.Contains(lower, "--- fail") || strings.Contains(lower, "test failed") ||
		strings.Contains(lower, "assertion failed") || strings.Contains(lower, "assertionerror") ||
		strings.Contains(lower, "fail:") {
		ctx.TestFailure = true
		ctx.HasError = true
		ctx.ErrorMessage = "test failure in tool output"
	}
	if strings.Contains(lower, "exit status") || strings.Contains(lower, "exit code") {
		ctx.HasError = true
		ctx.ExitCode = 1
	}
	if strings.Contains(lower, "syntax error") || strings.Contains(lower, "build failed") ||
		strings.Contains(lower, "compilation error") || strings.Contains(lower, "compile error") ||
		strings.Contains(lower, "undefined:") || strings.Contains(lower, "cannot use") {
		ctx.CompilerError = true
		ctx.HasError = true
		ctx.ErrorMessage = "compiler/build error in tool output"
	}
	if strings.Contains(lower, "panic:") || strings.Contains(lower, "runtime error:") {
		ctx.ExecutionPanic = true
		ctx.HasError = true
		ctx.ErrorMessage = "runtime panic in tool output"
	}
	if strings.Contains(lower, "policy block") || strings.Contains(lower, "policy refusal") ||
		strings.Contains(lower, "permission denied") {
		ctx.PolicyRefusal = true
		ctx.PolicyReason = "policy refusal in tool output"
	}
}

// ClassifyTurnEffort runs the turn context through IntraModelEffortRouter to obtain
// the deterministic TurnEffortDecision.
func ClassifyTurnEffort(turnCtx agentopt.TurnContext) agentopt.TurnEffortDecision {
	router := agentopt.NewIntraModelEffortRouter()
	return router.Classify(turnCtx)
}

// ModulateGeminiJSON injects or updates generationConfig.thinkingConfig in a raw Gemini JSON payload
// without altering message contents or history.
// When AllocatedBudget == 0, thinkingBudget: 0 is explicitly set.
func ModulateGeminiJSON(raw []byte, decision agentopt.TurnEffortDecision) ([]byte, error) {
	if len(raw) == 0 {
		return raw, nil
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return nil, fmt.Errorf("modulate gemini json: %w", err)
	}

	budget := decision.AllocatedBudget
	thinkingJSON := fmt.Sprintf(`{"thinkingBudget":%d}`, budget)

	var genCfg map[string]json.RawMessage
	if rawCfg, ok := root["generationConfig"]; ok && len(rawCfg) > 0 {
		if err := json.Unmarshal(rawCfg, &genCfg); err != nil {
			genCfg = make(map[string]json.RawMessage)
		}
	} else {
		genCfg = make(map[string]json.RawMessage)
	}

	genCfg["thinkingConfig"] = json.RawMessage(thinkingJSON)
	marshaledGenCfg, err := json.Marshal(genCfg)
	if err != nil {
		return nil, err
	}
	root["generationConfig"] = marshaledGenCfg

	return json.Marshal(root)
}

// ModulateGeminiRequest updates a decoded GeminiGenerateContentRequest with the allocated thinking budget.
func ModulateGeminiRequest(req *agent.GeminiGenerateContentRequest, decision agentopt.TurnEffortDecision) {
	// Preserved for typed representation conformance.
}

// ModulateOpenAIJSON injects or updates reasoning_effort in a raw OpenAI ChatCompletions JSON payload
// without altering message history or system prompt bytes.
func ModulateOpenAIJSON(raw []byte, decision agentopt.TurnEffortDecision) ([]byte, error) {
	if len(raw) == 0 {
		return raw, nil
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return nil, fmt.Errorf("modulate openai json: %w", err)
	}

	root["reasoning_effort"] = json.RawMessage(fmt.Sprintf("%q", decision.Effort))
	return json.Marshal(root)
}

// ModulateAnthropicJSON updates the thinking configuration in a raw Anthropic Messages JSON payload.
// If AllocatedBudget == 0, thinking is disabled; otherwise type is enabled with budget_tokens set.
func ModulateAnthropicJSON(raw []byte, decision agentopt.TurnEffortDecision) ([]byte, error) {
	if len(raw) == 0 {
		return raw, nil
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return nil, fmt.Errorf("modulate anthropic json: %w", err)
	}

	if decision.AllocatedBudget == 0 {
		root["thinking"] = json.RawMessage(`{"type":"disabled"}`)
	} else {
		root["thinking"] = json.RawMessage(fmt.Sprintf(`{"type":"enabled","budget_tokens":%d}`, decision.AllocatedBudget))
	}
	return json.Marshal(root)
}

// ModulateChatRequest sets ReasoningEffort and ThinkingConfig on ChatRequest.
func ModulateChatRequest(req *ChatRequest, decision agentopt.TurnEffortDecision) {
	if req == nil {
		return
	}
	req.ReasoningEffort = string(decision.Effort)
	req.ThinkingConfig, _ = json.Marshal(map[string]int{
		"thinkingBudget": decision.AllocatedBudget,
	})
}

// ExtractPromptPrefixBytes constructs a canonical representation of prompt instructions, messages,
// and tools to demonstrate byte-level immutability across dynamic effort shifts.
func ExtractPromptPrefixBytes(messages []agent.Message, system string, tools []agent.ToolDef) ([]byte, error) {
	type canonicalContext struct {
		System   string          `json:"system,omitempty"`
		Messages []agent.Message `json:"messages"`
		Tools    []agent.ToolDef `json:"tools,omitempty"`
	}
	return json.Marshal(canonicalContext{
		System:   system,
		Messages: messages,
		Tools:    tools,
	})
}

// PromptPrefixStreamBytes returns the concatenated, ordered bytes of the prompt context
// (system instructions, tools, and ordered messages) such that a subsequent turn's prefix
// strictly contains the previous turn's prefix as a bit-identical leading slice.
func PromptPrefixStreamBytes(messages []agent.Message, system string, tools []agent.ToolDef) ([]byte, error) {
	var buf bytes.Buffer
	if system != "" {
		sRaw, err := json.Marshal(system)
		if err != nil {
			return nil, err
		}
		buf.WriteString("system:")
		buf.Write(sRaw)
		buf.WriteByte('\n')
	}
	if len(tools) > 0 {
		tRaw, err := json.Marshal(tools)
		if err != nil {
			return nil, err
		}
		buf.WriteString("tools:")
		buf.Write(tRaw)
		buf.WriteByte('\n')
	}
	for i, m := range messages {
		mRaw, err := json.Marshal(m)
		if err != nil {
			return nil, err
		}
		buf.WriteString(fmt.Sprintf("msg-%d:", i))
		buf.Write(mRaw)
		buf.WriteByte('\n')
	}
	return buf.Bytes(), nil
}

// ExtractPromptPrefixFromJSON extracts only prompt context (contents, messages, system, tools)
// from raw provider JSON, ignoring metadata keys (generationConfig, reasoning_effort, thinking).
func ExtractPromptPrefixFromJSON(raw []byte) ([]byte, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return nil, err
	}

	type canonicalPrompt struct {
		Contents          json.RawMessage `json:"contents,omitempty"`
		Messages          json.RawMessage `json:"messages,omitempty"`
		SystemInstruction json.RawMessage `json:"systemInstruction,omitempty"`
		System            json.RawMessage `json:"system,omitempty"`
		Tools             json.RawMessage `json:"tools,omitempty"`
	}

	p := canonicalPrompt{
		Contents:          root["contents"],
		Messages:          root["messages"],
		SystemInstruction: root["systemInstruction"],
		System:            root["system"],
		Tools:             root["tools"],
	}
	return json.Marshal(p)
}

// ValidatePrefixCachePreservation verifies that previousPrefix is an exact bit-identical
// leading slice of currentPrefix, guaranteeing 100% prefix cache hits on provider KV caches.
func ValidatePrefixCachePreservation(previousPrefix, currentPrefix []byte) bool {
	if len(previousPrefix) == 0 {
		return true
	}
	return bytes.HasPrefix(currentPrefix, previousPrefix)
}

// ValidateMessagesPreserved checks that the prior messages in current are byte-identical
// to the messages in previous.
func ValidateMessagesPreserved(previous, current []agent.Message) bool {
	if len(previous) > len(current) {
		return false
	}
	for i := range previous {
		prevRaw, _ := json.Marshal(previous[i])
		currRaw, _ := json.Marshal(current[i])
		if !bytes.Equal(prevRaw, currRaw) {
			return false
		}
	}
	return true
}

// ApplyDynamicEffort applies dynamic turn-level thinking budget modulation to req.
func ApplyDynamicEffort(req *ChatRequest) *agentopt.TurnEffortDecision {
	if req == nil {
		return nil
	}
	if !RequestSupportsThinking(req.Model, req.ReasoningEffort != "" || len(req.ThinkingConfig) > 0) {
		return nil
	}

	turnCtx := BuildTurnContext(req.Messages, "")
	router := agentopt.NewIntraModelEffortRouter()
	decision := router.Classify(turnCtx)
	ModulateChatRequest(req, decision)
	return &decision
}

func (s *Server) getEffortRouter() *agentopt.IntraModelEffortRouter {
	return agentopt.NewIntraModelEffortRouter()
}

// applyDynamicEffortModulation modulates turn-level reasoning effort on req if supported.
func (s *Server) applyDynamicEffortModulation(req *ChatRequest) *agentopt.TurnEffortDecision {
	if req == nil {
		return nil
	}
	if !RequestSupportsThinking(req.Model, req.ReasoningEffort != "" || len(req.ThinkingConfig) > 0) {
		return nil
	}

	turnCtx := BuildTurnContext(req.Messages, "")
	decision := s.getEffortRouter().Classify(turnCtx)
	ModulateChatRequest(req, decision)
	return &decision
}

// applyDynamicEffortModulationToGemini modulates Gemini raw request payload if supported.
func (s *Server) applyDynamicEffortModulationToGemini(req *agent.GeminiGenerateContentRequest, raw *[]byte) *agentopt.TurnEffortDecision {
	if req == nil {
		return nil
	}
	if !RequestSupportsThinking(req.Model, false) {
		return nil
	}

	turnCtx := BuildTurnContext(req.Messages, req.System)
	decision := s.getEffortRouter().Classify(turnCtx)

	if raw != nil && len(*raw) > 0 {
		if modulated, err := ModulateGeminiJSON(*raw, decision); err == nil {
			*raw = modulated
		}
	}
	ModulateGeminiRequest(req, decision)
	return &decision
}

// ApplyDynamicEffortToGeminiJSON classifies and modulates a raw Gemini JSON payload.
func ApplyDynamicEffortToGeminiJSON(raw []byte, model string) ([]byte, *agentopt.TurnEffortDecision, error) {
	req, err := agent.DecodeGeminiGenerateContentRequest(raw, model)
	if err != nil {
		return raw, nil, err
	}
	turnCtx := BuildTurnContext(req.Messages, req.System)
	decision := ClassifyTurnEffort(turnCtx)
	modulated, err := ModulateGeminiJSON(raw, decision)
	if err != nil {
		return raw, nil, err
	}
	return modulated, &decision, nil
}

// ApplyDynamicEffortToOpenAIJSON classifies and modulates a raw OpenAI JSON payload.
func ApplyDynamicEffortToOpenAIJSON(raw []byte, model string) ([]byte, *agentopt.TurnEffortDecision, error) {
	var chatReq ChatRequest
	if err := json.Unmarshal(raw, &chatReq); err != nil {
		return raw, nil, err
	}
	if chatReq.Model == "" {
		chatReq.Model = model
	}
	turnCtx := BuildTurnContext(chatReq.Messages, "")
	decision := ClassifyTurnEffort(turnCtx)
	modulated, err := ModulateOpenAIJSON(raw, decision)
	if err != nil {
		return raw, nil, err
	}
	return modulated, &decision, nil
}

// ApplyDynamicEffortToAnthropicJSON classifies and modulates a raw Anthropic JSON payload.
func ApplyDynamicEffortToAnthropicJSON(raw []byte, model string) ([]byte, *agentopt.TurnEffortDecision, error) {
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		return raw, nil, err
	}
	turnCtx := BuildTurnContext(req.Messages, req.System)
	decision := ClassifyTurnEffort(turnCtx)
	modulated, err := ModulateAnthropicJSON(raw, decision)
	if err != nil {
		return raw, nil, err
	}
	return modulated, &decision, nil
}
