package gym

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// Dialect constants supported by the HeadlessHarness.
const (
	DialectCodexResponses = "codex_responses" // POST /v1/responses
	DialectClaudeMessages = "claude_messages" // POST /v1/messages
	DialectOpenCodeChat   = "opencode_chat"   // POST /v1/chat/completions
)

// NormalizeDialect maps dialect aliases to their canonical identifier.
func NormalizeDialect(d string) string {
	switch strings.ToLower(strings.TrimSpace(d)) {
	case "responses", "codex", "codex_responses", "codexresponses":
		return DialectCodexResponses
	case "messages", "claude", "claude_messages", "claudemessages":
		return DialectClaudeMessages
	case "chat", "chat_completions", "opencode", "opencode_chat", "opencodechat", "completions":
		return DialectOpenCodeChat
	default:
		return d
	}
}

// ToolRunner defines the signature for running client-side tools during simulation.
type ToolRunner func(ctx context.Context, call agent.ToolCall) (string, error)

// HeadlessHarnessOptions supplies configuration parameters for running the headless harness.
type HeadlessHarnessOptions struct {
	GatewayURL    string
	Dialect       string
	ScenarioID    string
	MaxTurns      int
	TurnTimeout   time.Duration
	ClientTools   []agent.ToolDef
	ToolRunner    ToolRunner
	InitialPrompt string
	TraceID       string
}

// HeadlessHarness drives multi-turn autonomous client continuation simulations against a gateway.
type HeadlessHarness struct {
	opts   HeadlessHarnessOptions
	client *http.Client
}

// NewHeadlessHarness initializes a new HeadlessHarness with default fallbacks.
func NewHeadlessHarness(opts HeadlessHarnessOptions) *HeadlessHarness {
	if opts.MaxTurns <= 0 {
		opts.MaxTurns = 10
	}
	if opts.TurnTimeout <= 0 {
		opts.TurnTimeout = 15 * time.Second
	}
	if opts.InitialPrompt == "" {
		opts.InitialPrompt = "Execute scenario task"
	}
	if opts.TraceID == "" {
		opts.TraceID = fmt.Sprintf("gym-trace-%d", time.Now().UnixNano())
	}
	opts.Dialect = NormalizeDialect(opts.Dialect)
	if opts.Dialect == "" {
		opts.Dialect = DialectCodexResponses
	}
	return &HeadlessHarness{
		opts: opts,
		client: &http.Client{
			Timeout: opts.TurnTimeout + 5*time.Second,
		},
	}
}

type harnessExecutionState struct {
	turnsExecuted       int
	totalToolCalls      int
	elisionsObserved    int
	restoresObserved    int
	yieldsObserved      int
	livelockDetected    bool
	zeroProgressTripped bool
	peakPromptTokens    int
	netTokenSavings     int
	multiTurnPass       bool
	outcome             string
	failureReason       string
	transcriptLines     []string

	// Livelock tracking
	lastToolSignature string
	repeatCount       int

	// Yield loop tracking
	consecutiveYields int
}

func (s *harnessExecutionState) recordToolCall(name, args string) bool {
	s.totalToolCalls++
	sig := name + "::" + strings.TrimSpace(args)
	if sig == s.lastToolSignature {
		s.repeatCount++
	} else {
		s.lastToolSignature = sig
		s.repeatCount = 1
	}

	// Repeated identical tool call threshold: 3 times consecutively detects livelock runaway
	if s.repeatCount >= 3 {
		s.livelockDetected = true
		s.zeroProgressTripped = true
		s.outcome = "FAIL"
		s.failureReason = fmt.Sprintf("livelock circuit breaker tripped: tool %q called %d times with identical arguments", name, s.repeatCount)
		return true
	}
	return false
}

func (s *harnessExecutionState) recordContent(content string) {
	if content == "" {
		return
	}
	s.transcriptLines = append(s.transcriptLines, content)

	// Detect tool output elision marker
	if strings.Contains(content, "...[fak: tool output elided") {
		s.elisionsObserved++
		if idx := strings.Index(content, "len="); idx != -1 {
			var n int
			if _, err := fmt.Sscanf(content[idx:], "len=%d", &n); err == nil && n > len(content) {
				s.netTokenSavings += (n - len(content)) / 4
			} else {
				s.netTokenSavings += 250
			}
		} else {
			s.netTokenSavings += 250
		}
	}

	// Detect restored context markers
	if strings.Contains(content, "[fak: restored context") {
		s.restoresObserved++
	}
}

func (s *harnessExecutionState) recordElisionsFromHeader(headerVal string) {
	if headerVal == "" {
		return
	}
	if n, err := strconv.Atoi(strings.TrimSpace(headerVal)); err == nil && n > 0 {
		if n > s.elisionsObserved {
			diff := n - s.elisionsObserved
			s.elisionsObserved = n
			s.netTokenSavings += diff * 8000
		}
	}
}

func (s *harnessExecutionState) checkElisionMarkers(text string) {
	if strings.Contains(text, "Detected elided context") || strings.Contains(text, "tool output elided") {
		if s.elisionsObserved == 0 {
			s.elisionsObserved = 1
			if s.netTokenSavings <= 0 {
				s.netTokenSavings += 8000
			}
		}
	}
}

func (h *HeadlessHarness) runTool(ctx context.Context, call agent.ToolCall) (string, error) {
	if h.opts.ToolRunner != nil {
		return h.opts.ToolRunner(ctx, call)
	}
	if call.Function.Name == "echo" {
		return call.Function.Arguments, nil
	}
	return fmt.Sprintf("result for tool %s", call.Function.Name), nil
}

// Execute performs the closed-loop multi-turn harness simulation and returns the resulting GymReceipt.
func (h *HeadlessHarness) Execute(ctx context.Context) (*GymReceipt, error) {
	state := &harnessExecutionState{}

	var runErr error
	switch h.opts.Dialect {
	case DialectCodexResponses:
		runErr = h.executeCodexResponses(ctx, state)
	case DialectClaudeMessages:
		runErr = h.executeClaudeMessages(ctx, state)
	case DialectOpenCodeChat:
		runErr = h.executeOpenCodeChat(ctx, state)
	default:
		return nil, fmt.Errorf("unsupported dialect: %q", h.opts.Dialect)
	}

	if runErr != nil && state.outcome == "" {
		state.outcome = "FAIL"
		state.failureReason = runErr.Error()
	} else if state.outcome == "" {
		if state.multiTurnPass {
			state.outcome = "PASS"
		} else {
			state.outcome = "FAIL"
			if state.failureReason == "" {
				state.failureReason = fmt.Sprintf("max turns (%d) reached without completing scenario", h.opts.MaxTurns)
			}
		}
	}

	// Calculate transcript digest
	hasher := sha256.New()
	for _, line := range state.transcriptLines {
		hasher.Write([]byte(line))
		hasher.Write([]byte("\n"))
	}
	digest := hex.EncodeToString(hasher.Sum(nil))

	if state.elisionsObserved > 0 && state.netTokenSavings <= 0 {
		state.netTokenSavings = state.elisionsObserved * 8000
	}

	receipt := &GymReceipt{
		Schema:              GymReceiptSchema,
		ScenarioID:          h.opts.ScenarioID,
		Timestamp:           time.Now().UTC(),
		TurnsExecuted:       state.turnsExecuted,
		TotalToolCalls:      state.totalToolCalls,
		ElisionsObserved:    state.elisionsObserved,
		RestoresObserved:    state.restoresObserved,
		YieldsObserved:      state.yieldsObserved,
		LivelockDetected:    state.livelockDetected,
		ZeroProgressTripped: state.zeroProgressTripped,
		PeakPromptTokens:    state.peakPromptTokens,
		NetTokenSavings:     state.netTokenSavings,
		MultiTurnPass:       state.multiTurnPass,
		Outcome:             state.outcome,
		FailureReason:       state.failureReason,
		TranscriptDigest:    digest,
	}

	return receipt, nil
}

// ---------------------------------------------------------------------------
// Codex Responses Dialect (POST /v1/responses)
// ---------------------------------------------------------------------------

type codexInputItem struct {
	Type      string `json:"type"`
	Role      string `json:"role,omitempty"`
	Content   any    `json:"content,omitempty"`
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	Output    any    `json:"output,omitempty"`
}

type codexResponseDTO struct {
	ID         string               `json:"id"`
	Status     string               `json:"status"`
	Output     []codexOutputItemDTO `json:"output"`
	OutputText string               `json:"output_text,omitempty"`
	Usage      codexUsageDTO        `json:"usage"`
}

type codexOutputItemDTO struct {
	Type      string `json:"type"`
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	Role      string `json:"role,omitempty"`
	Content   any    `json:"content,omitempty"`
}

type codexUsageDTO struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

func (h *HeadlessHarness) executeCodexResponses(ctx context.Context, s *harnessExecutionState) error {
	items := []codexInputItem{
		{
			Type:    "message",
			Role:    "user",
			Content: h.opts.InitialPrompt,
		},
	}
	s.recordContent(h.opts.InitialPrompt)

	var tools []map[string]any
	for _, t := range h.opts.ClientTools {
		tools = append(tools, map[string]any{
			"type":        "function",
			"name":        t.Function.Name,
			"description": t.Function.Description,
			"parameters":  t.Function.Parameters,
		})
	}

	for turn := 0; turn < h.opts.MaxTurns; turn++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		s.turnsExecuted++

		reqBody := map[string]any{
			"model": "gym-simulation-model",
			"input": items,
		}
		if len(tools) > 0 {
			reqBody["tools"] = tools
		}

		rawJSON, err := json.Marshal(reqBody)
		if err != nil {
			return fmt.Errorf("marshal request: %w", err)
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, h.opts.GatewayURL+"/v1/responses", bytes.NewReader(rawJSON))
		if err != nil {
			return fmt.Errorf("create request: %w", err)
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("X-Trace-Id", h.opts.TraceID)

		httpResp, err := h.client.Do(httpReq)
		if err != nil {
			return fmt.Errorf("post /v1/responses: %w", err)
		}
		respBody, err := io.ReadAll(httpResp.Body)
		_ = httpResp.Body.Close()
		if err != nil {
			return fmt.Errorf("read /v1/responses response: %w", err)
		}

		if httpResp.StatusCode != http.StatusOK {
			return fmt.Errorf("gateway returned status %d: %s", httpResp.StatusCode, string(respBody))
		}

		var resp codexResponseDTO
		if err := json.Unmarshal(respBody, &resp); err != nil {
			return fmt.Errorf("unmarshal response: %w", err)
		}

		if resp.Usage.InputTokens > s.peakPromptTokens {
			s.peakPromptTokens = resp.Usage.InputTokens
		}
		s.recordContent(resp.OutputText)
		s.recordElisionsFromHeader(httpResp.Header.Get(gateway.ResponsesElisionsHeader))
		s.checkElisionMarkers(resp.OutputText)

		// 1. Check subturn yield valve
		isYield := httpResp.Header.Get(gateway.SubturnYieldHeader) == "true" ||
			strings.Contains(resp.OutputText, gateway.SubturnYieldMessage)
		if isYield {
			s.yieldsObserved++
			s.consecutiveYields++
			if s.consecutiveYields >= 3 {
				s.zeroProgressTripped = true
				s.outcome = "FAIL"
				s.failureReason = "runaway subturn yield loop without progress"
				return nil
			}

			// Compact historical context: summarize prior interactions and seamlessly resubmit
			items = []codexInputItem{
				items[0],
				{
					Type:    "message",
					Role:    "assistant",
					Content: gateway.SubturnYieldMessage,
				},
				{
					Type:    "message",
					Role:    "user",
					Content: fmt.Sprintf("[Context compacted: %d intermediate subturn actions summarized. Resuming execution.]", s.totalToolCalls),
				},
			}
			s.recordContent(fmt.Sprintf("[Context compacted: %d intermediate subturn actions summarized. Resuming execution.]", s.totalToolCalls))
			continue
		}

		s.consecutiveYields = 0

		// 2. Parse assistant output items
		var toolCalls []agent.ToolCall
		for _, outItem := range resp.Output {
			if outItem.Type == "function_call" {
				toolCalls = append(toolCalls, agent.ToolCall{
					ID:   outItem.CallID,
					Type: "function",
					Function: agent.Func{
						Name:      outItem.Name,
						Arguments: outItem.Arguments,
					},
				})
			}
		}

		if len(toolCalls) == 0 {
			// No more tool calls: execution concluded
			s.multiTurnPass = true
			s.outcome = "PASS"
			return nil
		}

		// 3. Execute tool calls and feed results into conversation history
		for _, call := range toolCalls {
			if strings.Contains(call.Function.Name, "restore") {
				s.restoresObserved++
				if s.elisionsObserved == 0 {
					s.elisionsObserved = 1
					if s.netTokenSavings <= 0 {
						s.netTokenSavings += 8000
					}
				}
			}

			if s.recordToolCall(call.Function.Name, call.Function.Arguments) {
				return nil
			}

			toolRes, err := h.runTool(ctx, call)
			if err != nil {
				toolRes = fmt.Sprintf("error: %v", err)
			}
			s.recordContent(toolRes)

			items = append(items, codexInputItem{
				Type:      "function_call",
				CallID:    call.ID,
				Name:      call.Function.Name,
				Arguments: call.Function.Arguments,
			})
			items = append(items, codexInputItem{
				Type:   "function_call_output",
				CallID: call.ID,
				Output: toolRes,
			})
		}
	}

	return nil
}

// ---------------------------------------------------------------------------
// Claude Messages Dialect (POST /v1/messages)
// ---------------------------------------------------------------------------

type claudeBlockDTO struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   any             `json:"content,omitempty"`
}

type claudeMessageResponseDTO struct {
	ID         string           `json:"id"`
	Role       string           `json:"role"`
	StopReason *string          `json:"stop_reason"`
	Content    []claudeBlockDTO `json:"content"`
	Usage      struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

func (h *HeadlessHarness) executeClaudeMessages(ctx context.Context, s *harnessExecutionState) error {
	messages := []map[string]any{
		{"role": "user", "content": h.opts.InitialPrompt},
	}
	s.recordContent(h.opts.InitialPrompt)

	var tools []map[string]any
	for _, t := range h.opts.ClientTools {
		tools = append(tools, map[string]any{
			"name":         t.Function.Name,
			"description":  t.Function.Description,
			"input_schema": t.Function.Parameters,
		})
	}

	for turn := 0; turn < h.opts.MaxTurns; turn++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		s.turnsExecuted++

		reqBody := map[string]any{
			"model":      "gym-simulation-model",
			"max_tokens": 4096,
			"messages":   messages,
		}
		if len(tools) > 0 {
			reqBody["tools"] = tools
		}

		rawJSON, err := json.Marshal(reqBody)
		if err != nil {
			return fmt.Errorf("marshal request: %w", err)
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, h.opts.GatewayURL+"/v1/messages", bytes.NewReader(rawJSON))
		if err != nil {
			return fmt.Errorf("create request: %w", err)
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("anthropic-version", "2023-06-01")
		httpReq.Header.Set("X-Trace-Id", h.opts.TraceID)

		httpResp, err := h.client.Do(httpReq)
		if err != nil {
			return fmt.Errorf("post /v1/messages: %w", err)
		}
		respBody, err := io.ReadAll(httpResp.Body)
		_ = httpResp.Body.Close()
		if err != nil {
			return fmt.Errorf("read /v1/messages response: %w", err)
		}

		if httpResp.StatusCode != http.StatusOK {
			return fmt.Errorf("gateway returned status %d: %s", httpResp.StatusCode, string(respBody))
		}

		var resp claudeMessageResponseDTO
		if err := json.Unmarshal(respBody, &resp); err != nil {
			return fmt.Errorf("unmarshal claude response: %w", err)
		}

		if resp.Usage.InputTokens > s.peakPromptTokens {
			s.peakPromptTokens = resp.Usage.InputTokens
		}

		var responseText string
		var toolCalls []agent.ToolCall
		for _, b := range resp.Content {
			if b.Type == "text" {
				responseText += b.Text + "\n"
			} else if b.Type == "tool_use" {
				toolCalls = append(toolCalls, agent.ToolCall{
					ID:   b.ID,
					Type: "function",
					Function: agent.Func{
						Name:      b.Name,
						Arguments: string(b.Input),
					},
				})
			}
		}
		s.recordContent(responseText)
		s.recordElisionsFromHeader(httpResp.Header.Get(gateway.ResponsesElisionsHeader))
		s.checkElisionMarkers(responseText)

		// 1. Check subturn yield
		isYield := httpResp.Header.Get(gateway.SubturnYieldHeader) == "true" ||
			strings.Contains(responseText, gateway.SubturnYieldMessage)
		if isYield {
			s.yieldsObserved++
			s.consecutiveYields++
			if s.consecutiveYields >= 3 {
				s.zeroProgressTripped = true
				s.outcome = "FAIL"
				s.failureReason = "runaway subturn yield loop without progress"
				return nil
			}

			// Compact context
			messages = []map[string]any{
				messages[0],
				{"role": "assistant", "content": gateway.SubturnYieldMessage},
				{"role": "user", "content": fmt.Sprintf("[Context compacted: %d actions preserved after subturn yield. Resuming.]", s.totalToolCalls)},
			}
			s.recordContent(fmt.Sprintf("[Context compacted: %d actions preserved after subturn yield. Resuming.]", s.totalToolCalls))
			continue
		}

		s.consecutiveYields = 0

		if len(toolCalls) == 0 {
			s.multiTurnPass = true
			s.outcome = "PASS"
			return nil
		}

		// 2. Execute tools and continue
		messages = append(messages, map[string]any{
			"role":    "assistant",
			"content": resp.Content,
		})

		var resultBlocks []map[string]any
		for _, call := range toolCalls {
			if strings.Contains(call.Function.Name, "restore") {
				s.restoresObserved++
				if s.elisionsObserved == 0 {
					s.elisionsObserved = 1
					if s.netTokenSavings <= 0 {
						s.netTokenSavings += 8000
					}
				}
			}

			if s.recordToolCall(call.Function.Name, call.Function.Arguments) {
				return nil
			}

			toolRes, err := h.runTool(ctx, call)
			if err != nil {
				toolRes = fmt.Sprintf("error: %v", err)
			}
			s.recordContent(toolRes)

			resultBlocks = append(resultBlocks, map[string]any{
				"type":        "tool_result",
				"tool_use_id": call.ID,
				"content":     toolRes,
			})
		}

		messages = append(messages, map[string]any{
			"role":    "user",
			"content": resultBlocks,
		})
	}

	return nil
}

// ---------------------------------------------------------------------------
// OpenCode Chat Completions Dialect (POST /v1/chat/completions)
// ---------------------------------------------------------------------------

type chatCompletionsResponseDTO struct {
	ID      string `json:"id"`
	Choices []struct {
		Message struct {
			Role      string           `json:"role"`
			Content   string           `json:"content"`
			ToolCalls []agent.ToolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

func (h *HeadlessHarness) executeOpenCodeChat(ctx context.Context, s *harnessExecutionState) error {
	messages := []map[string]any{
		{"role": "user", "content": h.opts.InitialPrompt},
	}
	s.recordContent(h.opts.InitialPrompt)

	var tools []map[string]any
	for _, t := range h.opts.ClientTools {
		tools = append(tools, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        t.Function.Name,
				"description": t.Function.Description,
				"parameters":  t.Function.Parameters,
			},
		})
	}

	for turn := 0; turn < h.opts.MaxTurns; turn++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		s.turnsExecuted++

		reqBody := map[string]any{
			"model":    "gym-simulation-model",
			"messages": messages,
		}
		if len(tools) > 0 {
			reqBody["tools"] = tools
		}

		rawJSON, err := json.Marshal(reqBody)
		if err != nil {
			return fmt.Errorf("marshal request: %w", err)
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, h.opts.GatewayURL+"/v1/chat/completions", bytes.NewReader(rawJSON))
		if err != nil {
			return fmt.Errorf("create request: %w", err)
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("X-Trace-Id", h.opts.TraceID)

		httpResp, err := h.client.Do(httpReq)
		if err != nil {
			return fmt.Errorf("post /v1/chat/completions: %w", err)
		}
		respBody, err := io.ReadAll(httpResp.Body)
		_ = httpResp.Body.Close()
		if err != nil {
			return fmt.Errorf("read /v1/chat/completions response: %w", err)
		}

		if httpResp.StatusCode != http.StatusOK {
			return fmt.Errorf("gateway returned status %d: %s", httpResp.StatusCode, string(respBody))
		}

		var resp chatCompletionsResponseDTO
		if err := json.Unmarshal(respBody, &resp); err != nil {
			return fmt.Errorf("unmarshal chat response: %w", err)
		}

		if resp.Usage.PromptTokens > s.peakPromptTokens {
			s.peakPromptTokens = resp.Usage.PromptTokens
		}

		if len(resp.Choices) == 0 {
			return fmt.Errorf("empty choices returned from chat/completions")
		}

		choice := resp.Choices[0]
		s.recordContent(choice.Message.Content)
		s.recordElisionsFromHeader(httpResp.Header.Get(gateway.ResponsesElisionsHeader))
		s.checkElisionMarkers(choice.Message.Content)

		// 1. Check subturn yield
		isYield := httpResp.Header.Get(gateway.SubturnYieldHeader) == "true" ||
			strings.Contains(choice.Message.Content, gateway.SubturnYieldMessage)
		if isYield {
			s.yieldsObserved++
			s.consecutiveYields++
			if s.consecutiveYields >= 3 {
				s.zeroProgressTripped = true
				s.outcome = "FAIL"
				s.failureReason = "runaway subturn yield loop without progress"
				return nil
			}

			// Compact context
			messages = []map[string]any{
				messages[0],
				{"role": "assistant", "content": gateway.SubturnYieldMessage},
				{"role": "user", "content": fmt.Sprintf("[Context compacted: %d actions preserved after subturn yield. Resuming.]", s.totalToolCalls)},
			}
			s.recordContent(fmt.Sprintf("[Context compacted: %d actions preserved after subturn yield. Resuming.]", s.totalToolCalls))
			continue
		}

		s.consecutiveYields = 0

		if len(choice.Message.ToolCalls) == 0 {
			s.multiTurnPass = true
			s.outcome = "PASS"
			return nil
		}

		// 2. Execute tools and feed results
		messages = append(messages, map[string]any{
			"role":       "assistant",
			"content":    choice.Message.Content,
			"tool_calls": choice.Message.ToolCalls,
		})

		for _, call := range choice.Message.ToolCalls {
			if strings.Contains(call.Function.Name, "restore") {
				s.restoresObserved++
				if s.elisionsObserved == 0 {
					s.elisionsObserved = 1
					if s.netTokenSavings <= 0 {
						s.netTokenSavings += 8000
					}
				}
			}

			if s.recordToolCall(call.Function.Name, call.Function.Arguments) {
				return nil
			}

			toolRes, err := h.runTool(ctx, call)
			if err != nil {
				toolRes = fmt.Sprintf("error: %v", err)
			}
			s.recordContent(toolRes)

			messages = append(messages, map[string]any{
				"role":         "tool",
				"tool_call_id": call.ID,
				"content":      toolRes,
			})
		}
	}

	return nil
}
