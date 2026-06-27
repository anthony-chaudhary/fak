// webbench-token-measure measures actual token usage from model API runs.
// This is the real measurement counterpart to the theoretical cost arm calculations.
//
// Usage:
//   webbench-token-measure --responses responses.jsonl
//   webbench-token-measure --demo
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
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
	ResponseFile string // Path to JSONL file containing API responses
	Demo         bool   // Run demo mode with simulated data
}

var config Config

func init() {
	flag.StringVar(&config.ResponseFile, "responses", "", "Path to JSONL file with API responses (one per line)")
	flag.BoolVar(&config.Demo, "demo", false, "Run demo mode with simulated measurements")
}

func main() {
	flag.Parse()

	if config.ResponseFile == "" && !config.Demo {
		fmt.Fprintln(os.Stderr, "Error: --responses is required (or use --demo)")
		flag.Usage()
		os.Exit(1)
	}

	if config.Demo {
		runDemo()
		return
	}

	if err := processResponses(config.ResponseFile); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
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
	estimatedTokens := len(context) / 4

	meas := Measurement{
		TurnNumber:    turnNum,
		PrefillTokens: estimatedTokens,
		DecodeTokens:  150 + (estimatedTokens / 10),
		Timestamp:     time.Now(),
		Model:         "demo-model",
		APIProvider:   "demo",
	}
	meas.TotalTokens = meas.PrefillTokens + meas.DecodeTokens

	return meas, nil
}

// APIResponseRaw is a raw API response that may be from OpenAI or Anthropic.
type APIResponseRaw struct {
	ID     string          `json:"id"`
	Object string          `json:"object,omitempty"`
	Type   string          `json:"type,omitempty"`
	Model  string          `json:"model"`
	Usage  json.RawMessage `json:"usage"`
}

// processResponses reads a JSONL file of API responses and measures token usage.
func processResponses(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open responses file: %w", err)
	}
	defer file.Close()

	var run Run
	run.TaskID = "real-measurements-" + time.Now().Format("20060102-150405")
	run.StartTime = time.Now()

	scanner := bufio.NewScanner(file)
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var raw APIResponseRaw
		if err := json.Unmarshal([]byte(line), &raw); err != nil {
			return fmt.Errorf("parse response at line %d: %w", lineNum, err)
		}

		var usage TokenUsage
		var err error

		switch {
		case raw.Object == "chat.completion":
			usage, err = parseOpenAIResponse([]byte(line))
		case raw.Type == "message":
			usage, err = parseAnthropicResponse([]byte(line))
		default:
			return fmt.Errorf("line %d: unknown response format (object=%q, type=%q)", lineNum, raw.Object, raw.Type)
		}

		if err != nil {
			return fmt.Errorf("parse usage at line %d: %w", lineNum, err)
		}

		meas := Measurement{
			TurnNumber:    lineNum,
			PrefillTokens: usage.PrefillTokens,
			DecodeTokens:  usage.DecodeTokens,
			TotalTokens:   usage.TotalTokens,
			Timestamp:     time.Now(),
			Model:         raw.Model,
			APIProvider:   inferProvider(raw.Object, raw.Type),
		}

		run.Measurements = append(run.Measurements, meas)
		run.TotalPrefill += meas.PrefillTokens
		run.TotalDecode += meas.DecodeTokens
		run.TotalTokens += meas.TotalTokens
		run.TurnCount++
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read responses file: %w", err)
	}

	run.EndTime = time.Now()
	if run.TurnCount > 0 {
		run.AvgPrefill = float64(run.TotalPrefill) / float64(run.TurnCount)
	}

	return printReport(&run)
}

// inferProvider guesses the provider from response fields.
func inferProvider(obj, typ string) string {
	switch {
	case obj == "chat.completion":
		return "openai"
	case typ == "message":
		return "anthropic"
	default:
		return "unknown"
	}
}

// runDemo runs the demo mode with simulated measurements.
func runDemo() {
	var run Run
	run.TaskID = "demo-task-001"
	run.StartTime = time.Now()

	context := ""
	for i := 1; i <= 5; i++ {
		context += fmt.Sprintf("\n[Turn %d action and DOM state...]", i)

		meas, err := measureTurn(config, i, run.TaskID, context)
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

	printReport(&run)
}

// printReport prints the measurement report.
func printReport(run *Run) error {
	fmt.Printf("Token Measurement Summary:\n")
	fmt.Printf("  Task ID: %s\n", run.TaskID)
	fmt.Printf("  Turns: %d\n", run.TurnCount)
	fmt.Printf("  Total Prefill: %d tokens\n", run.TotalPrefill)
	fmt.Printf("  Total Decode: %d tokens\n", run.TotalDecode)
	fmt.Printf("  Total Tokens: %d\n", run.TotalTokens)
	fmt.Printf("  Avg Prefill/Turn: %.0f tokens\n", run.AvgPrefill)

	fmt.Printf("\nTurn-by-turn breakdown:\n")
	fmt.Printf("  Turn | Prefill | Decode | Total | Model\n")
	fmt.Printf("  ----|---------|--------|------|-------\n")
	for _, m := range run.Measurements {
		fmt.Printf("  %3d | %7d | %6d | %5d | %s\n", m.TurnNumber, m.PrefillTokens, m.DecodeTokens, m.TotalTokens, m.Model)
	}

	if run.TurnCount > 0 {
		fmt.Printf("\nCost comparison (naive vs fak):\n")
		naiveTotal := 0
		for _, m := range run.Measurements {
			naiveTotal += m.TotalTokens
		}
		fakTotal := run.Measurements[0].PrefillTokens
		for i := 1; i < len(run.Measurements); i++ {
			fakTotal += run.Measurements[i].DecodeTokens
		}

		elimination := float64(naiveTotal) / float64(fakTotal)
		fmt.Printf("  Naive total: %d tokens\n", naiveTotal)
		fmt.Printf("  fak total: %d tokens\n", fakTotal)
		fmt.Printf("  Elimination: %.1fx\n", elimination)
	}

	return nil
}
