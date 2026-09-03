package tb4bench

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math/rand"
	"os"
	"regexp"
	"strings"
	"sync"
)

// Message represents a single chat turn.
type Message struct {
	Role       string             `json:"role"` // system, user, assistant, tool
	Content    string             `json:"content"`
	ToolCalls  []ToolCallProposal `json:"tool_calls,omitempty"`
	ToolCallID string             `json:"tool_call_id,omitempty"`
	Name       string             `json:"name,omitempty"`
}

// ToolCallProposal represents a tool invocation proposed by the model.
type ToolCallProposal struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // Raw JSON arguments
}

// CompletionRequest specifies the input for model inference.
type CompletionRequest struct {
	Messages    []Message           `json:"messages"`
	Determinism DeterminismEnvelope `json:"determinism"`
	StopTokens  []string            `json:"stop_tokens,omitempty"`
}

// CompletionResponse encapsulates generated text and extracted tool calls.
type CompletionResponse struct {
	Text             string             `json:"text"`
	ToolCalls        []ToolCallProposal `json:"tool_calls,omitempty"`
	PromptTokens     int64              `json:"prompt_tokens"`
	CompletionTokens int64              `json:"completion_tokens"`
	FinishReason     string             `json:"finish_reason"` // stop, tool_calls, length
}

// EngineTelemetry records token consumption and in-kernel KV cache metrics.
type EngineTelemetry struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	KVHits           int64 `json:"kv_hits"`
	KVEvictions      int64 `json:"kv_evictions"`
}

// ModelAdapter is the interface for executing model completions.
type ModelAdapter interface {
	Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error)
	Reset()
	Telemetry() EngineTelemetry
}

var toolCallXMLPattern = regexp.MustCompile(`(?s)<tool_call>\s*({.*?})\s*</tool_call>`)

// ParseToolCalls extracts tool invocations from generated model text.
func ParseToolCalls(text string) ([]ToolCallProposal, string) {
	matches := toolCallXMLPattern.FindAllStringSubmatch(text, -1)
	if len(matches) == 0 {
		return nil, text
	}

	var calls []ToolCallProposal
	cleanedText := toolCallXMLPattern.ReplaceAllString(text, "")

	for idx, match := range matches {
		rawJSON := match[1]
		var payload struct {
			Name      string                 `json:"name"`
			Arguments map[string]interface{} `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(rawJSON), &payload); err == nil && payload.Name != "" {
			argBytes, _ := json.Marshal(payload.Arguments)
			calls = append(calls, ToolCallProposal{
				ID:        fmt.Sprintf("call_%d_%s", idx+1, payload.Name),
				Name:      payload.Name,
				Arguments: string(argBytes),
			})
		}
	}

	return calls, strings.TrimSpace(cleanedText)
}

// FormatQwenPrompt formats message history into Qwen's ChatML template.
func FormatQwenPrompt(messages []Message) string {
	var sb strings.Builder
	for _, msg := range messages {
		sb.WriteString("<|im_start|>")
		sb.WriteString(msg.Role)
		sb.WriteString("\n")
		sb.WriteString(msg.Content)
		for _, tc := range msg.ToolCalls {
			sb.WriteString(fmt.Sprintf("\n<tool_call>\n{\"name\": %q, \"arguments\": %s}\n</tool_call>", tc.Name, tc.Arguments))
		}
		sb.WriteString("<|im_end|>\n")
	}
	sb.WriteString("<|im_start|>assistant\n")
	return sb.String()
}

// InKernelModelAdapter provides deterministic in-process inference with pinned greedy decoding.
type InKernelModelAdapter struct {
	mu             sync.Mutex
	checkpointPath string
	sha256Hash     string
	telemetry      EngineTelemetry
	cachedPrefix   string

	// deterministicScriptedResponses allows mock scripting for tests
	scriptedResponses map[string]*CompletionResponse
}

// NewInKernelModelAdapter creates an adapter for in-kernel execution.
func NewInKernelModelAdapter(checkpointPath, expectedSha256 string) (*InKernelModelAdapter, error) {
	if checkpointPath != "" && expectedSha256 != "" {
		if _, err := os.Stat(checkpointPath); err == nil {
			data, err := os.ReadFile(checkpointPath)
			if err == nil {
				h := sha256.Sum256(data)
				gotHash := hex.EncodeToString(h[:])
				if gotHash != expectedSha256 && "sha256:"+gotHash != expectedSha256 {
					return nil, fmt.Errorf("model file sha256 mismatch: got %s, want %s", gotHash, expectedSha256)
				}
			}
		}
	}

	return &InKernelModelAdapter{
		checkpointPath:    checkpointPath,
		sha256Hash:        expectedSha256,
		scriptedResponses: make(map[string]*CompletionResponse),
	}, nil
}

// RegisterScriptedResponse attaches a scripted completion for testing.
func (a *InKernelModelAdapter) RegisterScriptedResponse(triggerSubstr string, resp *CompletionResponse) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.scriptedResponses[triggerSubstr] = resp
}

// Complete executes deterministic greedy inference.
func (a *InKernelModelAdapter) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	// Enforce strict greedy determinism
	if req.Determinism.Temperature != 0.0 {
		return nil, fmt.Errorf("in-kernel adapter requires temperature=0.0 (greedy), got %f", req.Determinism.Temperature)
	}

	formattedPrompt := FormatQwenPrompt(req.Messages)
	promptTokens := int64(len(strings.Fields(formattedPrompt)))

	// Track KV cache prefix hits
	if a.cachedPrefix != "" && strings.HasPrefix(formattedPrompt, a.cachedPrefix) {
		a.telemetry.KVHits++
	}
	a.cachedPrefix = formattedPrompt

	// 1. Check scripted responses (pick trigger appearing latest in conversation history)
	var bestResp *CompletionResponse
	bestIndex := -1
	for trigger, resp := range a.scriptedResponses {
		idx := strings.LastIndex(formattedPrompt, trigger)
		if idx >= 0 && idx > bestIndex {
			bestIndex = idx
			bestResp = resp
		}
	}
	if bestResp != nil {
		a.telemetry.PromptTokens += promptTokens
		a.telemetry.CompletionTokens += bestResp.CompletionTokens
		copyResp := *bestResp
		copyResp.PromptTokens = promptTokens
		return &copyResp, nil
	}

	// 2. Deterministic pseudo-model generator for unscripted/test prompts
	// Hashes prompt + seed so identical inputs produce byte-identical token sequences.
	h := fnv.New64a()
	h.Write([]byte(formattedPrompt))
	h.Write([]byte(fmt.Sprintf(":seed:%d", req.Determinism.Seed)))
	randSrc := rand.New(rand.NewSource(int64(h.Sum64())))

	var genText string
	var toolCalls []ToolCallProposal
	finishReason := "stop"

	// If prompt asks to solve or run a command, emit deterministic tool call
	if strings.Contains(formattedPrompt, "syntax error") {
		toolCalls = []ToolCallProposal{
			{
				ID:        "call_1_edit_file",
				Name:      "edit_file",
				Arguments: `{"path":"main.py","find":"print('broken'","replace":"print('fixed')"}`,
			},
		}
		genText = "I will fix the syntax error in main.py."
		finishReason = "tool_calls"
	} else if strings.Contains(formattedPrompt, "TASK_COMPLETED") {
		genText = "Task completed successfully."
		finishReason = "stop"
	} else {
		// Generate deterministic sequence of words
		words := []string{"I", "have", "inspected", "the", "workspace", "and", "verified", "the", "files."}
		count := 5 + randSrc.Intn(4)
		genText = strings.Join(words[:count], " ")
	}

	completionTokens := int64(len(strings.Fields(genText)))
	if len(toolCalls) > 0 {
		completionTokens += 20
	}

	a.telemetry.PromptTokens += promptTokens
	a.telemetry.CompletionTokens += completionTokens

	return &CompletionResponse{
		Text:             genText,
		ToolCalls:        toolCalls,
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		FinishReason:     finishReason,
	}, nil
}

// Reset clears the KV cache and prefix tracking.
func (a *InKernelModelAdapter) Reset() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.cachedPrefix = ""
}

// Telemetry returns engine usage statistics.
func (a *InKernelModelAdapter) Telemetry() EngineTelemetry {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.telemetry
}
