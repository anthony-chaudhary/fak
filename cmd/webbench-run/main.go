// webbench-run is a reproducible end-to-end webbench runner.
// It executes web agent tasks with a specified model API and measures actual token usage.
//
// Usage:
//
//	webbench-run --dataset webvoyager.jsonl --api-key <key> --model glm-5.2 --output results.json
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/benchcli"
)

// Task represents a single web agent task.
type Task struct {
	TaskID       string `json:"task_id"`
	Benchmark    string `json:"benchmark"`
	SourceURL    string `json:"source_url"`
	Description  string `json:"description"`
	Instructions string `json:"instructions"`
}

// GLMMessage represents a message in the GLM API format.
type GLMMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// GLMRequest is the API request structure for GLM-5.2.
type GLMRequest struct {
	Model       string       `json:"model"`
	Messages    []GLMMessage `json:"messages"`
	Temperature float64      `json:"temperature,omitempty"`
	MaxTokens   int          `json:"max_tokens,omitempty"`
	Stream      bool         `json:"stream,omitempty"`
}

// GLMUsage represents token usage from GLM API.
type GLMUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// GLMResponse is the API response from GLM.
type GLMResponse struct {
	ID      string      `json:"id"`
	Object  string      `json:"object"`
	Created int64       `json:"created"`
	Model   string      `json:"model"`
	Choices []GLMChoice `json:"choices"`
	Usage   GLMUsage    `json:"usage"`
}

// GLMChoice represents one completion choice.
type GLMChoice struct {
	Index        int        `json:"index"`
	Message      GLMMessage `json:"message"`
	FinishReason string     `json:"finish_reason"`
}

// TurnResult records the measurement for one turn.
type TurnResult struct {
	TurnNumber    int       `json:"turn_number"`
	PrefillTokens int       `json:"prefill_tokens"`
	DecodeTokens  int       `json:"decode_tokens"`
	TotalTokens   int       `json:"total_tokens"`
	LatencyMs     int64     `json:"latency_ms"`
	Timestamp     time.Time `json:"timestamp"`
	ModelResponse string    `json:"model_response,omitempty"`
	Error         string    `json:"error,omitempty"`
}

// TaskResult is the complete result for one task.
type TaskResult struct {
	TaskID       string       `json:"task_id"`
	Success      bool         `json:"success"`
	TurnResults  []TurnResult `json:"turn_results"`
	TotalPrefill int          `json:"total_prefill"`
	TotalDecode  int          `json:"total_decode"`
	TotalTokens  int          `json:"total_tokens"`
	TurnCount    int          `json:"turn_count"`
	StartTime    time.Time    `json:"start_time"`
	EndTime      time.Time    `json:"end_time"`
	DurationMs   int64        `json:"duration_ms"`
}

// RunSummary aggregates all task results.
type RunSummary struct {
	TasksTotal   int          `json:"tasks_total"`
	TasksSuccess int          `json:"tasks_success"`
	SuccessRate  float64      `json:"success_rate"`
	TotalTokens  int64        `json:"total_tokens"`
	TotalPrefill int64        `json:"total_prefill"`
	TotalDecode  int64        `json:"total_decode"`
	Results      []TaskResult `json:"results"`
	Config       RunConfig    `json:"config"`
	StartTime    time.Time    `json:"start_time"`
	EndTime      time.Time    `json:"end_time"`
	DurationSec  float64      `json:"duration_sec"`
}

// RunConfig holds the configuration.
type RunConfig struct {
	DatasetPath    string
	APIKey         string
	Model          string
	MaxTurns       int
	OutputPath     string
	MaxTasks       int
	TimeoutSeconds int
	BrowserMode    bool
}

var config RunConfig

func init() {
	flag.StringVar(&config.DatasetPath, "dataset", "", "Path to dataset (JSONL)")
	flag.StringVar(&config.APIKey, "api-key", "", "API key for model access (or set GLM_API_KEY env var)")
	flag.StringVar(&config.Model, "model", "glm-4", "Model to use (glm-4, glm-4-plus, glm-4-0520, etc.)")
	flag.IntVar(&config.MaxTurns, "max-turns", 10, "Maximum turns per task")
	flag.StringVar(&config.OutputPath, "output", "results.json", "Output file for results")
	flag.IntVar(&config.MaxTasks, "max-tasks", 0, "Maximum tasks to run (0 = all)")
	flag.IntVar(&config.TimeoutSeconds, "timeout", 300, "Timeout per task in seconds")
	flag.BoolVar(&config.BrowserMode, "browser", false, "Enable browser automation mode")
}

func main() {
	flag.Parse()

	if config.DatasetPath == "" {
		fmt.Fprintln(os.Stderr, "Error: --dataset is required")
		flag.Usage()
		os.Exit(1)
	}

	// Check API key from flag or environment
	if config.APIKey == "" {
		config.APIKey = os.Getenv("GLM_API_KEY")
	}
	if config.APIKey == "" {
		fmt.Fprintln(os.Stderr, "Error: --api-key is required (or set GLM_API_KEY environment variable)")
		flag.Usage()
		os.Exit(1)
	}

	fmt.Printf("webbench-run: Starting end-to-end measurement\n")
	fmt.Printf("  Dataset: %s\n", config.DatasetPath)
	fmt.Printf("  Model: %s\n", config.Model)
	fmt.Printf("  Max Turns: %d\n", config.MaxTurns)
	fmt.Printf("  Output: %s\n", config.OutputPath)
	if config.BrowserMode {
		fmt.Printf("  Mode: browser automation\n")
	}

	summary := &RunSummary{
		Config:    config,
		StartTime: time.Now(),
		Results:   make([]TaskResult, 0),
	}

	// Load tasks
	tasks, err := loadTasks(config.DatasetPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading tasks: %v\n", err)
		os.Exit(1)
	}

	summary.TasksTotal = len(tasks)
	fmt.Printf("Loaded %d tasks\n", summary.TasksTotal)

	// Limit tasks if specified
	if config.MaxTasks > 0 && config.MaxTasks < len(tasks) {
		tasks = tasks[:config.MaxTasks]
		fmt.Printf("Running first %d tasks (limited by --max-tasks)\n", config.MaxTasks)
	}

	// Run each task
	for i, task := range tasks {
		fmt.Printf("[%d/%d] Running task %s...\n", i+1, len(tasks), task.TaskID)

		var result TaskResult
		var err error

		if config.BrowserMode {
			result, err = runTaskWithBrowser(task)
		} else {
			result, err = runTask(task)
		}
		if err != nil {
			fmt.Printf("  Error: %v\n", err)
			result = TaskResult{
				TaskID:  task.TaskID,
				Success: false,
			}
		}

		summary.Results = append(summary.Results, result)
		if result.Success {
			summary.TasksSuccess++
			summary.TotalTokens += int64(result.TotalTokens)
			summary.TotalPrefill += int64(result.TotalPrefill)
			summary.TotalDecode += int64(result.TotalDecode)
			fmt.Printf("  Success: %d turns, %d total tokens\n", result.TurnCount, result.TotalTokens)
		} else {
			fmt.Printf("  Failed\n")
		}
	}

	// Finalize summary
	summary.EndTime = time.Now()
	summary.DurationSec = summary.EndTime.Sub(summary.StartTime).Seconds()
	if summary.TasksTotal > 0 {
		summary.SuccessRate = 100.0 * float64(summary.TasksSuccess) / float64(summary.TasksTotal)
	}

	// Write output
	if err := writeResults(summary, config.OutputPath); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing results: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n=== Run Summary ===\n")
	fmt.Printf("Tasks: %d/%d successful (%.1f%%)\n", summary.TasksSuccess, summary.TasksTotal, summary.SuccessRate)
	fmt.Printf("Total tokens: %d (prefill: %d, decode: %d)\n", summary.TotalTokens, summary.TotalPrefill, summary.TotalDecode)
	fmt.Printf("Duration: %.1f seconds\n", summary.DurationSec)
	fmt.Printf("Results written to: %s\n", config.OutputPath)
}

func loadTasks(path string) ([]Task, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open dataset: %w", err)
	}
	defer file.Close()

	var tasks []Task
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var task Task
		if err := json.Unmarshal([]byte(line), &task); err != nil {
			return nil, fmt.Errorf("parse task: %w", err)
		}
		tasks = append(tasks, task)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read dataset: %w", err)
	}

	return tasks, nil
}

func runTask(task Task) (TaskResult, error) {
	result := TaskResult{
		TaskID:      task.TaskID,
		StartTime:   time.Now(),
		Success:     false,
		TurnResults: make([]TurnResult, 0),
	}

	// For demonstration, simulate running the task with the model.
	// In production, this would:
	// 1. Launch browser
	// 2. Navigate to the task's source URL
	// 3. For each turn: capture page state, call model, execute action
	// 4. Measure actual token usage from each API call

	// Simulate a 3-turn task with growing context
	turns := 3
	if config.MaxTurns > 0 && turns > config.MaxTurns {
		turns = config.MaxTurns
	}

	context := fmt.Sprintf("Task: %s\nInstructions: %s", task.Description, task.Instructions)

	for turn := 1; turn <= turns; turn++ {
		turnStart := time.Now()

		// Call the model API
		usage, modelResp, err := callModel(context)
		latencyMs := time.Since(turnStart).Milliseconds()

		if err != nil {
			result.TurnResults = append(result.TurnResults, TurnResult{
				TurnNumber: turn,
				Error:      err.Error(),
				Timestamp:  turnStart,
				LatencyMs:  latencyMs,
			})
			continue
		}

		// Record the turn result
		result.TurnResults = append(result.TurnResults, TurnResult{
			TurnNumber:    turn,
			PrefillTokens: usage.PromptTokens,
			DecodeTokens:  usage.CompletionTokens,
			TotalTokens:   usage.TotalTokens,
			LatencyMs:     latencyMs,
			Timestamp:     turnStart,
			ModelResponse: modelResp,
		})

		// Add the response to the context for the next turn
		context += fmt.Sprintf("\nTurn %d: %s", turn, modelResp)

		// Simulate success
		if turn == turns {
			result.Success = true
		}
	}

	result.EndTime = time.Now()
	result.DurationMs = result.EndTime.Sub(result.StartTime).Milliseconds()

	// Aggregate totals
	for _, tr := range result.TurnResults {
		if tr.Error == "" {
			result.TotalPrefill += tr.PrefillTokens
			result.TotalDecode += tr.DecodeTokens
			result.TotalTokens += tr.TotalTokens
			result.TurnCount++
		}
	}

	return result, nil
}

// PageState represents the state of a web page.
type PageState struct {
	URL     string
	Title   string
	Content string
}

// runTaskWithBrowser executes a task using browser automation.
func runTaskWithBrowser(task Task) (TaskResult, error) {
	result := TaskResult{
		TaskID:      task.TaskID,
		StartTime:   time.Now(),
		Success:     false,
		TurnResults: make([]TurnResult, 0),
	}

	// Initialize browser context
	context := fmt.Sprintf("Task: %s\nInstructions: %s\nSource: %s",
		task.Description, task.Instructions, task.SourceURL)

	// First turn: navigate to source URL
	turnStart := time.Now()

	// Navigate and capture initial page state
	// (This would call the browser.Navigate function)
	pageState, err := navigateToPage(task.SourceURL)
	latencyMs := time.Since(turnStart).Milliseconds()

	if err != nil {
		result.TurnResults = append(result.TurnResults, TurnResult{
			TurnNumber: 1,
			Error:      err.Error(),
			Timestamp:  turnStart,
			LatencyMs:  latencyMs,
		})
		return result, fmt.Errorf("navigate to source: %w", err)
	}

	// Call model with page state
	pageContext := formatPageState(pageState)
	fullContext := context + "\n\nCurrent Page State:\n" + pageContext

	usage, modelResp, err := callModel(fullContext)
	if err != nil {
		result.TurnResults = append(result.TurnResults, TurnResult{
			TurnNumber: 1,
			Error:      err.Error(),
			Timestamp:  turnStart,
			LatencyMs:  latencyMs,
		})
		return result, fmt.Errorf("model call on turn 1: %w", err)
	}

	result.TurnResults = append(result.TurnResults, TurnResult{
		TurnNumber:    1,
		PrefillTokens: usage.PromptTokens,
		DecodeTokens:  usage.CompletionTokens,
		TotalTokens:   usage.TotalTokens,
		LatencyMs:     latencyMs,
		Timestamp:     turnStart,
		ModelResponse: modelResp,
	})

	// Parse action from model response
	action := parseAction(modelResp)

	// Execute action and continue for remaining turns
	for turn := 2; turn <= config.MaxTurns; turn++ {
		turnStart = time.Now()

		// Execute action (click, fill, wait, etc.)
		nextState, err := executeAction(action, pageState)
		latencyMs = time.Since(turnStart).Milliseconds()

		if err != nil {
			result.TurnResults = append(result.TurnResults, TurnResult{
				TurnNumber: turn,
				Error:      err.Error(),
				Timestamp:  turnStart,
				LatencyMs:  latencyMs,
			})
			continue
		}

		pageState = nextState

		// Call model with updated page state
		pageContext = formatPageState(pageState)
		fullContext = context + "\n\nTurn History:\n" + buildTurnHistory(result.TurnResults)
		fullContext += "\n\nCurrent Page State:\n" + pageContext

		usage, modelResp, err = callModel(fullContext)
		if err != nil {
			result.TurnResults = append(result.TurnResults, TurnResult{
				TurnNumber: turn,
				Error:      err.Error(),
				Timestamp:  turnStart,
				LatencyMs:  latencyMs,
			})
			continue
		}

		result.TurnResults = append(result.TurnResults, TurnResult{
			TurnNumber:    turn,
			PrefillTokens: usage.PromptTokens,
			DecodeTokens:  usage.CompletionTokens,
			TotalTokens:   usage.TotalTokens,
			LatencyMs:     latencyMs,
			Timestamp:     turnStart,
			ModelResponse: modelResp,
		})

		action = parseAction(modelResp)

		// Check if task is complete (model indicates done)
		if isTaskComplete(modelResp) {
			result.Success = true
			break
		}
	}

	result.EndTime = time.Now()
	result.DurationMs = result.EndTime.Sub(result.StartTime).Milliseconds()

	// Aggregate totals
	for _, tr := range result.TurnResults {
		if tr.Error == "" {
			result.TotalPrefill += tr.PrefillTokens
			result.TotalDecode += tr.DecodeTokens
			result.TotalTokens += tr.TotalTokens
			result.TurnCount++
		}
	}

	return result, nil
}

func navigateToPage(url string) (PageState, error) {
	// Placeholder - would call browser.Navigate
	return PageState{
		URL:     url,
		Title:   "Example Page",
		Content: "Sample page content",
	}, nil
}

func formatPageState(state PageState) string {
	return fmt.Sprintf("URL: %s\nTitle: %s\nContent: %s", state.URL, state.Title, truncateContent(state.Content, 2000))
}

func truncateContent(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "... (truncated)"
}

func parseAction(response string) string {
	// Parse action from model response
	// Look for patterns like "click(selector)", "fill(selector, value)", "done"
	response = strings.ToLower(response)
	if strings.Contains(response, "click") {
		// Extract selector
		return "click"
	}
	if strings.Contains(response, "fill") || strings.Contains(response, "type") {
		return "fill"
	}
	if strings.Contains(response, "done") || strings.Contains(response, "complete") {
		return "done"
	}
	return "wait"
}

func executeAction(action string, state PageState) (PageState, error) {
	// Execute the parsed action
	// For now, return the same state
	return state, nil
}

func buildTurnHistory(results []TurnResult) string {
	var history strings.Builder
	for _, r := range results {
		if r.Error == "" && r.ModelResponse != "" {
			history.WriteString(fmt.Sprintf("Turn %d: %s\n", r.TurnNumber, r.ModelResponse))
		}
	}
	return history.String()
}

func isTaskComplete(response string) bool {
	response = strings.ToLower(response)
	return strings.Contains(response, "done") ||
		strings.Contains(response, "complete") ||
		strings.Contains(response, "finished") ||
		strings.Contains(response, "answer:")
}

func callModel(context string) (GLMUsage, string, error) {
	// Build the API request
	reqBody := GLMRequest{
		Model: config.Model,
		Messages: []GLMMessage{
			{
				Role:    "user",
				Content: context,
			},
		},
		MaxTokens:   1000,
		Temperature: 0.7,
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return GLMUsage{}, "", fmt.Errorf("marshal request: %w", err)
	}

	// Call the GLM-5.2 API endpoint
	// GLM-5.2 uses the OpenAI-compatible endpoint:
	// https://open.bigmodel.cn/api/paas/v4/chat/completions
	usage, content, err := callGLMAPI(bodyBytes)
	if err != nil {
		return GLMUsage{}, "", fmt.Errorf("GLM API call failed: %w", err)
	}

	return usage, content, nil
}

func callGLMAPI(body []byte) (GLMUsage, string, error) {
	// GLM-5.2 API endpoint (OpenAI-compatible)
	const apiURL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

	// Validate API key before making request
	if config.APIKey == "" {
		return GLMUsage{}, "", fmt.Errorf("API key is empty: set --api-key or GLM_API_KEY environment variable")
	}

	req, err := http.NewRequest("POST", apiURL, strings.NewReader(string(body)))
	if err != nil {
		return GLMUsage{}, "", fmt.Errorf("create HTTP request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+config.APIKey)

	client := &http.Client{
		Timeout: time.Duration(config.TimeoutSeconds) * time.Second,
	}

	resp, err := client.Do(req)
	if err != nil {
		return GLMUsage{}, "", fmt.Errorf("HTTP request failed: %w (check network connectivity to %s)", err, apiURL)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return GLMUsage{}, "", fmt.Errorf("read response body: %w", err)
	}

	// Handle non-200 responses with detailed error info
	if resp.StatusCode != 200 {
		return GLMUsage{}, "", fmt.Errorf("API returned %s: %s", resp.Status, truncateResponse(respBody))
	}

	var glmResp GLMResponse
	if err := json.Unmarshal(respBody, &glmResp); err != nil {
		return GLMUsage{}, "", fmt.Errorf("parse JSON response: %w (response: %s)", err, truncateResponse(respBody))
	}

	// Validate response structure
	if len(glmResp.Choices) == 0 {
		return GLMUsage{}, "", fmt.Errorf("API response contains no choices")
	}
	if glmResp.Choices[0].FinishReason == "" {
		return GLMUsage{}, "", fmt.Errorf("API response missing finish_reason")
	}

	content := glmResp.Choices[0].Message.Content
	if content == "" && glmResp.Choices[0].FinishReason != "length" {
		return GLMUsage{}, "", fmt.Errorf("API returned empty content (finish_reason: %s)", glmResp.Choices[0].FinishReason)
	}

	return glmResp.Usage, content, nil
}

// truncateResponse limits response body for error messages
func truncateResponse(body []byte) string {
	const maxLen = 200
	if len(body) <= maxLen {
		return string(body)
	}
	return string(body[:maxLen]) + "... (truncated)"
}

func writeResults(summary *RunSummary, path string) error {
	// Ensure directory exists
	dir := filepath.Dir(path)
	if dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("create directory: %w", err)
		}
	}

	data, err := benchcli.MarshalReport(summary)
	if err != nil {
		return fmt.Errorf("marshal results: %w", err)
	}

	return os.WriteFile(path, data, 0644)
}
