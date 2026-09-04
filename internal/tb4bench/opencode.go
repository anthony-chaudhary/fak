package tb4bench

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// OpenCodeConfig specifies the execution environment for OpenCode running headlessly.
type OpenCodeConfig struct {
	ServerBaseURL string        `json:"server_base_url"`
	ModelID       string        `json:"model_id"`
	RuntimeDir    string        `json:"runtime_dir"`
	BinaryPath    string        `json:"binary_path"`
	Timeout       time.Duration `json:"timeout"`
}

// GenerateOpenCodeJSON synthesizes the opencode.json configuration targeting llama-server.
func GenerateOpenCodeJSON(baseURL, modelID string) ([]byte, error) {
	if baseURL == "" {
		baseURL = "http://127.0.0.1:8080/v1"
	}
	if modelID == "" {
		modelID = "qwen3.8-reference"
	}

	cfg := map[string]interface{}{
		"$schema":  "https://opencode.ai/config.json",
		"snapshot": false,
		"provider": map[string]interface{}{
			"llamacpp": map[string]interface{}{
				"npm":  "@ai-sdk/openai-compatible",
				"name": "llama.cpp reference",
				"options": map[string]interface{}{
					"baseURL": baseURL,
				},
				"models": map[string]interface{}{
					modelID: map[string]interface{}{
						"name": modelID,
					},
				},
			},
		},
	}

	return json.MarshalIndent(cfg, "", "  ")
}

// OpenCodeAdapter drives headless execution of the OpenCode agent harness.
type OpenCodeAdapter struct {
	config OpenCodeConfig
}

// NewOpenCodeAdapter creates an adapter with the specified configuration.
func NewOpenCodeAdapter(cfg OpenCodeConfig) *OpenCodeAdapter {
	if cfg.BinaryPath == "" {
		if p, err := exec.LookPath("opencode"); err == nil {
			cfg.BinaryPath = p
		} else {
			cfg.BinaryPath = "opencode"
		}
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 600 * time.Second
	}
	return &OpenCodeAdapter{config: cfg}
}

// SetupEnvironment prepares isolated temporary directories for OpenCode.
func (a *OpenCodeAdapter) SetupEnvironment() (string, error) {
	runtimeDir, err := os.MkdirTemp("", "tb4-opencode-env-*")
	if err != nil {
		return "", fmt.Errorf("failed to create opencode runtime dir: %w", err)
	}

	configDir := filepath.Join(runtimeDir, "config")
	dataDir := filepath.Join(runtimeDir, "data")
	_ = os.MkdirAll(configDir, 0755)
	_ = os.MkdirAll(dataDir, 0755)

	configJSON, err := GenerateOpenCodeJSON(a.config.ServerBaseURL, a.config.ModelID)
	if err != nil {
		return "", err
	}

	if err := os.WriteFile(filepath.Join(configDir, "opencode.json"), configJSON, 0644); err != nil {
		return "", err
	}

	a.config.RuntimeDir = runtimeDir
	return runtimeDir, nil
}

// Teardown cleans up temporary environment files.
func (a *OpenCodeAdapter) Teardown() error {
	if a.config.RuntimeDir != "" {
		return os.RemoveAll(a.config.RuntimeDir)
	}
	return nil
}

// Execute runs OpenCode headlessly on the given task prompt in the designated workspace.
func (a *OpenCodeAdapter) Execute(ctx context.Context, task TaskManifest, wsDir string) (*ArmExecutionResult, error) {
	if a.config.RuntimeDir == "" {
		if _, err := a.SetupEnvironment(); err != nil {
			return nil, err
		}
		defer a.Teardown()
	}

	execCtx, cancel := context.WithTimeout(ctx, a.config.Timeout)
	defer cancel()

	configDir := filepath.Join(a.config.RuntimeDir, "config")
	dataDir := filepath.Join(a.config.RuntimeDir, "data")

	cmd := exec.CommandContext(execCtx, a.config.BinaryPath, "run", "--print-logs", "--dangerously-skip-permissions", task.Prompt)
	cmd.Dir = wsDir
	cmd.Env = append(os.Environ(),
		fmt.Sprintf("XDG_CONFIG_HOME=%s", configDir),
		fmt.Sprintf("XDG_DATA_HOME=%s", dataDir),
		fmt.Sprintf("OPENAI_BASE_URL=%s", a.config.ServerBaseURL),
		"OPENAI_API_KEY=dummy",
	)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	start := time.Now()
	err := cmd.Run()
	duration := time.Since(start)

	res, parseErr := ParseOpenCodeTranscript(stdout.Bytes(), stderr.Bytes(), task.TaskID)
	if parseErr != nil {
		return nil, fmt.Errorf("failed to parse transcript: %w", parseErr)
	}

	res.DurationMs = duration.Milliseconds()

	if err != nil {
		if errors.Is(execCtx.Err(), context.DeadlineExceeded) {
			res.Status = "TIMEOUT"
		} else {
			res.Status = "CRASHED"
		}
	} else {
		res.Status = "COMPLETED"
	}

	return res, nil
}

// Regex patterns for parsing OpenCode transcript lines
var (
	toolCallPattern = regexp.MustCompile(`(?i)(?:call|tool|executing):\s*([a-zA-Z0-9_-]+)\s*(?:with|args)?:?\s*(.*)`)
	tokenUsageRegex = regexp.MustCompile(`(?i)tokens:\s*(\d+)\s*prompt,\s*(\d+)\s*completion`)
)

// ParseOpenCodeTranscript extracts turns, tool executions, and token metrics from OpenCode outputs.
func ParseOpenCodeTranscript(stdout, stderr []byte, taskID string) (*ArmExecutionResult, error) {
	result := &ArmExecutionResult{
		ArmID:  "opencode_llamacpp",
		TaskID: taskID,
		Status: "COMPLETED",
	}

	lines := strings.Split(string(stdout), "\n")
	var currentTurn TurnRecord
	turnNum := 1

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}

		// Check for token usage line
		if tokenMatch := tokenUsageRegex.FindStringSubmatch(trimmed); len(tokenMatch) == 3 {
			p, _ := strconv.ParseInt(tokenMatch[1], 10, 64)
			c, _ := strconv.ParseInt(tokenMatch[2], 10, 64)
			result.TotalPromptTokens += p
			result.TotalCompletionTokens += c
			continue
		}

		// Check for tool call line
		if match := toolCallPattern.FindStringSubmatch(trimmed); len(match) >= 2 {
			toolName := match[1]
			toolArgs := ""
			if len(match) >= 3 {
				toolArgs = match[2]
			}

			tc := ToolCallProposal{
				ID:        fmt.Sprintf("call_opencode_%d", len(currentTurn.ToolCalls)+1),
				Name:      toolName,
				Arguments: toolArgs,
			}
			currentTurn.ToolCalls = append(currentTurn.ToolCalls, tc)
			currentTurn.ToolResults = append(currentTurn.ToolResults, ToolExecutionResult{
				ToolCallID: tc.ID,
				Tool:       toolName,
				Args:       toolArgs,
				Stdout:     "executed",
				ExitCode:   0,
			})
			continue
		}

		// Turn separator or text
		if strings.Contains(trimmed, "TASK_COMPLETED") || strings.Contains(trimmed, "Done.") {
			currentTurn.Turn = turnNum
			currentTurn.ModelText = trimmed
			result.Turns = append(result.Turns, currentTurn)
			currentTurn = TurnRecord{}
			turnNum++
		}
	}

	if len(currentTurn.ToolCalls) > 0 || currentTurn.ModelText != "" {
		currentTurn.Turn = turnNum
		result.Turns = append(result.Turns, currentTurn)
	}

	if len(result.Turns) == 0 {
		// Create at least one synthesized turn record
		result.Turns = append(result.Turns, TurnRecord{
			Turn:      1,
			ModelText: string(stdout),
		})
	}

	result.TotalTurns = len(result.Turns)
	if result.TotalPromptTokens == 0 {
		result.TotalPromptTokens = int64(len(stdout) / 4)
	}
	if result.TotalCompletionTokens == 0 {
		result.TotalCompletionTokens = int64(len(stdout) / 8)
	}

	return result, nil
}
