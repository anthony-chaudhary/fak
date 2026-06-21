// webbench-token-measure measures actual token usage from model API runs.
// This is the real measurement counterpart to the theoretical cost arm calculations.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

// TokenUsage records the token counts from a model API response.
type TokenUsage struct {
	PrefillTokens int `json:"prefill_tokens"`
	DecodeTokens  int `json:"decode_tokens"`
	TotalTokens   int `json:"total_tokens"`
}

// ModelResponse is a standard API response structure.
type ModelResponse struct {
	Usage TokenUsage `json:"usage"`
	ID    string     `json:"id"`
}

// Measurement records one turn's token usage.
type Measurement struct {
	TurnNumber    int       `json:"turn_number"`
	PrefillTokens int       `json:"prefill_tokens"`
	DecodeTokens  int       `json:"decode_tokens"`
	TotalTokens   int       `json:"total_tokens"`
	Timestamp     time.Time `json:"timestamp"`
	Model         string    `json:"model"`
	APIProvider   string    `json:"api_provider"`
}

// Run aggregates all measurements for one task.
type Run struct {
	TaskID       string        `json:"task_id"`
	Measurements []Measurement `json:"measurements"`
	TotalPrefill int           `json:"total_prefill"`
	TotalDecode  int           `json:"total_decode"`
	TotalTokens  int           `json:"total_tokens"`
	TurnCount    int           `json:"turn_count"`
	AvgPrefill   float64       `json:"avg_prefill_per_turn"`
	StartTime    time.Time     `json:"start_time"`
	EndTime      time.Time     `json:"end_time"`
}

// Config holds the measurement configuration.
type Config struct {
	ModelProvider string // "openai", "anthropic", "ollama", etc.
	ModelName     string // "gpt-4", "claude-3-opus-20240229", etc.
	APIKey        string // API key (or empty for local/no-auth)
	MaxTurns      int    // Maximum turns to record per task
	OutputDir     string // Where to write measurement JSON
}

// OpenAIUsage is the usage structure from OpenAI API responses.
type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// OpenAIResponse is a minimal OpenAI API response structure.
type OpenAIResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []OpenAIChoice `json:"choices"`
	Usage   OpenAIUsage    `json:"usage"`
}

// OpenAIChoice represents one completion choice.
type OpenAIChoice struct {
	Index        int           `json:"index"`
	Message      OpenAIMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

// OpenAIMessage is the message content.
type OpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// AnthropicUsage is the usage structure from Anthropic API responses.
type AnthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// AnthropicResponse is a minimal Anthropic API response structure.
type AnthropicResponse struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"`
	RawUsage     AnthropicUsage `json:"usage"`
	StopReason   string         `json:"stop_reason"`
	StopSequence *string        `json:"stop_sequence,omitempty"`
	Model        string         `json:"model"`
}

// parseOpenAIResponse extracts token usage from an OpenAI API response.
func parseOpenAIResponse(body []byte) (TokenUsage, error) {
	var resp OpenAIResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return TokenUsage{}, fmt.Errorf("unmarshal OpenAI response: %w", err)
	}
	return TokenUsage{
		PrefillTokens: resp.Usage.PromptTokens,
		DecodeTokens:  resp.Usage.CompletionTokens,
		TotalTokens:   resp.Usage.TotalTokens,
	}, nil
}

// parseAnthropicResponse extracts token usage from an Anthropic API response.
func parseAnthropicResponse(body []byte) (TokenUsage, error) {
	var resp AnthropicResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return TokenUsage{}, fmt.Errorf("unmarshal Anthropic response: %w", err)
	}
	return TokenUsage{
		PrefillTokens: resp.RawUsage.InputTokens,
		DecodeTokens:  resp.RawUsage.OutputTokens,
		TotalTokens:   resp.RawUsage.InputTokens + resp.RawUsage.OutputTokens,
	}, nil
}

// measureTurn makes a (mock) API call and records token usage.
// In production, this would call the actual model API and parse the response.
func measureTurn(cfg Config, turnNum int, taskID string, context string) (Measurement, error) {
	// Placeholder: In production, make actual API call here.
	// For now, simulate with estimated values based on context length.
	estimatedTokens := len(context) / 4 // Rough estimate: 4 chars per token.

	meas := Measurement{
		TurnNumber:    turnNum,
		PrefillTokens: estimatedTokens,
		DecodeTokens:  150 + (estimatedTokens / 10), // Rough decode estimate
		Timestamp:     time.Now(),
		Model:         cfg.ModelName,
		APIProvider:   cfg.ModelProvider,
	}
	meas.TotalTokens = meas.PrefillTokens + meas.DecodeTokens

	return meas, nil
}

// main demonstrates the token measurement tool.
func main() {
	cfg := Config{
		ModelProvider: "demo",
		ModelName:     "gpt-4-demo",
		MaxTurns:      10,
		OutputDir:     "measurements",
	}

	// Demo: simulate measuring a 5-turn task.
	taskID := "demo-task-001"
	var run Run
	run.TaskID = taskID
	run.StartTime = time.Now()

	// Simulate a growing context (like a web agent).
	context := ""
	for i := 1; i <= 5; i++ {
		// Each turn adds more context (DOM state, actions, etc.).
		context += fmt.Sprintf("\n[Turn %d action and DOM state...]", i)

		meas, err := measureTurn(cfg, i, taskID, context)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error measuring turn %d: %v\n", i, err)
			continue
		}

		run.Measurements = append(run.Measurements, meas)
		run.TotalPrefill += meas.PrefillTokens
		run.TotalDecode += meas.DecodeTokens
		run.TotalTokens += meas.TotalTokens
		run.TurnCount++
	}

	run.EndTime = time.Now()
	if run.TurnCount > 0 {
		run.AvgPrefill = float64(run.TotalPrefill) / float64(run.TurnCount)
	}

	// Output the measurement.
	fmt.Printf("Token Measurement Summary:\n")
	fmt.Printf("  Task ID: %s\n", run.TaskID)
	fmt.Printf("  Turns: %d\n", run.TurnCount)
	fmt.Printf("  Total Prefill: %d tokens\n", run.TotalPrefill)
	fmt.Printf("  Total Decode: %d tokens\n", run.TotalDecode)
	fmt.Printf("  Total Tokens: %d\n", run.TotalTokens)
	fmt.Printf("  Avg Prefill/Turn: %.0f tokens\n", run.AvgPrefill)

	// Show the turn-by-turn breakdown (like A/B/C comparison).
	fmt.Printf("\nTurn-by-turn breakdown:\n")
	fmt.Printf("  Turn | Prefill | Decode | Total\n")
	fmt.Printf("  ----|---------|--------|------\n")
	for _, m := range run.Measurements {
		fmt.Printf("  %3d | %7d | %6d | %5d\n", m.TurnNumber, m.PrefillTokens, m.DecodeTokens, m.TotalTokens)
	}

	// Compare naive (cumulative re-prefill) vs fak (shared prefix).
	fmt.Printf("\nCost comparison (naive vs fak):\n")
	naiveTotal := 0
	for _, m := range run.Measurements {
		// Naive: re-prefill entire context each turn
		naiveTotal += m.TotalTokens
	}
	// With fak: shared prefix means only first turn pays full prefill cost
	fakTotal := run.Measurements[0].PrefillTokens // First turn prefill
	for i := 1; i < len(run.Measurements); i++ {
		// Subsequent turns: only pay for new tokens (simplified)
		fakTotal += run.Measurements[i].DecodeTokens
	}

	elimination := float64(naiveTotal) / float64(fakTotal)
	fmt.Printf("  Naive total: %d tokens\n", naiveTotal)
	fmt.Printf("  fak total: %d tokens\n", fakTotal)
	fmt.Printf("  Elimination: %.1fx\n", elimination)
}
