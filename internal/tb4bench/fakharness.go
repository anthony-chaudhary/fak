package tb4bench

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

const TB4SystemPrompt = "You are an autonomous terminal coding agent running in Terminal-Bench 4. You have access to bash, read_file, write_file, and edit_file tools. Execute operations in the workspace to fulfill the user prompt. When you are confident the task is solved, output TASK_COMPLETED."

// TurnRecord documents one full turn of the agent harness loop.
type TurnRecord struct {
	Turn                int                   `json:"turn"`
	ModelText           string                `json:"model_text"`
	ToolCalls           []ToolCallProposal    `json:"tool_calls,omitempty"`
	AdjudicationVerdict string                `json:"adjudication_verdict,omitempty"`
	RefusalReason       string                `json:"refusal_reason,omitempty"`
	ToolResults         []ToolExecutionResult `json:"tool_results,omitempty"`
	PromptTokens        int64                 `json:"prompt_tokens"`
	CompletionTokens    int64                 `json:"completion_tokens"`
	DurationMs          int64                 `json:"duration_ms"`
	Timestamp           time.Time             `json:"timestamp"`
}

// ToolExecutionResult records stdout, stderr, and exit status of a tool invocation.
type ToolExecutionResult struct {
	ToolCallID string `json:"tool_call_id"`
	Tool       string `json:"tool"`
	Args       string `json:"args"`
	Stdout     string `json:"stdout"`
	Stderr     string `json:"stderr"`
	ExitCode   int    `json:"exit_code"`
	DurationMs int64  `json:"duration_ms"`
}

// ArmExecutionResult records the complete trajectory and kernel telemetry of a harness run.
type ArmExecutionResult struct {
	ArmID                 string       `json:"arm_id"`
	TaskID                string       `json:"task_id"`
	Status                string       `json:"status"` // COMPLETED, TIMEOUT, EXHAUSTED, CRASHED
	Turns                 []TurnRecord `json:"turns"`
	TotalTurns            int          `json:"total_turns"`
	TotalPromptTokens     int64        `json:"total_prompt_tokens"`
	TotalCompletionTokens int64        `json:"total_completion_tokens"`
	PolicyBlocks          int64        `json:"policy_blocks"`
	VDSOHits              int64        `json:"vdso_hits"`
	DurationMs            int64        `json:"duration_ms"`
	JournalHash           string       `json:"journal_hash"`
}

// FakHarness coordinates the native agent loop with in-kernel inference and policy adjudication.
type FakHarness struct {
	adapter     ModelAdapter
	adjudicator *adjudicator.Adjudicator
	wsMgr       *WorkspaceManager
	journal     []string
	prevHash    string
}

// NewFakHarness creates a new native fak harness instance with an optional adjudicator.
func NewFakHarness(adapter ModelAdapter, adj *adjudicator.Adjudicator, wsMgr *WorkspaceManager) *FakHarness {
	return &FakHarness{
		adapter:     adapter,
		adjudicator: adj,
		wsMgr:       wsMgr,
		prevHash:    "0000000000000000000000000000000000000000000000000000000000000000",
	}
}

// NewFakHarnessWithPolicy creates a fak harness wrapping a concrete adjudicator.Policy.
func NewFakHarnessWithPolicy(adapter ModelAdapter, pol adjudicator.Policy, wsMgr *WorkspaceManager) *FakHarness {
	return NewFakHarness(adapter, adjudicator.New(pol), wsMgr)
}

// appendJournal records a hash-chained audit journal entry.
func (h *FakHarness) appendJournal(action string, details map[string]interface{}) {
	entry := map[string]interface{}{
		"action":    action,
		"prev_hash": h.prevHash,
		"details":   details,
		"timestamp": time.Now().UTC().Format(time.RFC3339Nano),
	}
	data, _ := json.Marshal(entry)
	hasher := sha256.Sum256(data)
	h.prevHash = hex.EncodeToString(hasher[:])
	h.journal = append(h.journal, string(data))
}

// ExecuteTask runs the agent loop against a seeded workspace until completion or turn budget.
func (h *FakHarness) ExecuteTask(ctx context.Context, task TaskManifest, determinism DeterminismEnvelope) (*ArmExecutionResult, error) {
	startTime := time.Now()
	maxTurns := task.BudgetTurns
	if maxTurns <= 0 {
		maxTurns = determinism.MaxTurns
	}
	if maxTurns <= 0 {
		maxTurns = DefaultMaxTurns
	}

	messages := []Message{
		{Role: "system", Content: TB4SystemPrompt},
		{Role: "user", Content: task.Prompt},
	}

	result := &ArmExecutionResult{
		ArmID:  "fak_inkernel",
		TaskID: task.TaskID,
		Status: "RUNNING",
	}

	h.appendJournal("TASK_START", map[string]interface{}{
		"task_id": task.TaskID,
		"prompt":  task.Prompt,
	})

	for turn := 1; turn <= maxTurns; turn++ {
		turnStart := time.Now()

		// 1. Model completion
		compResp, err := h.adapter.Complete(ctx, CompletionRequest{
			Messages:    messages,
			Determinism: determinism,
		})
		if err != nil {
			result.Status = "CRASHED"
			return result, fmt.Errorf("in-kernel model error on turn %d: %w", turn, err)
		}

		result.TotalPromptTokens += compResp.PromptTokens
		result.TotalCompletionTokens += compResp.CompletionTokens

		// Parse any tool calls in response
		extractedCalls, cleanedText := ParseToolCalls(compResp.Text)
		proposedCalls := append(compResp.ToolCalls, extractedCalls...)

		turnRecord := TurnRecord{
			Turn:             turn,
			ModelText:        cleanedText,
			ToolCalls:        proposedCalls,
			PromptTokens:     compResp.PromptTokens,
			CompletionTokens: compResp.CompletionTokens,
			Timestamp:        turnStart,
		}

		// Check for completion signal
		if strings.Contains(compResp.Text, "TASK_COMPLETED") && len(proposedCalls) == 0 {
			turnRecord.DurationMs = time.Since(turnStart).Milliseconds()
			result.Turns = append(result.Turns, turnRecord)
			result.Status = "COMPLETED"
			h.appendJournal("TASK_COMPLETED", map[string]interface{}{"turn": turn})
			break
		}

		// 2. Execute and adjudicate each proposed tool call
		assistantMsg := Message{
			Role:      "assistant",
			Content:   cleanedText,
			ToolCalls: proposedCalls,
		}
		messages = append(messages, assistantMsg)

		var toolOutputs []string
		for _, tc := range proposedCalls {
			toolStart := time.Now()

			// Adjudicate tool call against kernel policy if configured
			verdictKind := abi.VerdictAllow
			refusalReason := ""

			if h.adjudicator != nil {
				call := &abi.ToolCall{
					Tool: tc.Name,
					Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(tc.Arguments)},
				}
				verdict := h.adjudicator.Adjudicate(ctx, call)
				verdictKind = verdict.Kind
				if verdictKind == abi.VerdictDeny {
					refusalReason = "POLICY_BLOCK"
				}
			}

			if verdictKind == abi.VerdictDeny {
				turnRecord.AdjudicationVerdict = "DENIED (POLICY_BLOCK)"
				turnRecord.RefusalReason = refusalReason
				result.PolicyBlocks++
				h.appendJournal("POLICY_DENIAL", map[string]interface{}{
					"tool": tc.Name,
					"args": tc.Arguments,
				})

				denyMsg := fmt.Sprintf("Error: action blocked by kernel security policy (%s)", refusalReason)
				turnRecord.ToolResults = append(turnRecord.ToolResults, ToolExecutionResult{
					ToolCallID: tc.ID,
					Tool:       tc.Name,
					Args:       tc.Arguments,
					Stderr:     denyMsg,
					ExitCode:   1,
					DurationMs: time.Since(toolStart).Milliseconds(),
				})
				toolOutputs = append(toolOutputs, denyMsg)
				continue
			}

			turnRecord.AdjudicationVerdict = "ALLOWED"

			// Execute allowed tool call inside container workspace
			res := h.executeTool(ctx, tc)
			turnRecord.ToolResults = append(turnRecord.ToolResults, res)
			h.appendJournal("TOOL_EXECUTED", map[string]interface{}{
				"tool":      tc.Name,
				"exit_code": res.ExitCode,
			})

			outText := res.Stdout
			if res.Stderr != "" {
				outText += "\n" + res.Stderr
			}
			toolOutputs = append(toolOutputs, outText)
		}

		// Append tool outputs to conversation
		if len(toolOutputs) > 0 {
			messages = append(messages, Message{
				Role:    "tool",
				Content: strings.Join(toolOutputs, "\n---\n"),
			})
		}

		turnRecord.DurationMs = time.Since(turnStart).Milliseconds()
		result.Turns = append(result.Turns, turnRecord)

		if turn == maxTurns && result.Status == "RUNNING" {
			result.Status = "TIMEOUT"
			h.appendJournal("TIMEOUT_AGENT", map[string]interface{}{"max_turns": maxTurns})
		}
	}

	result.TotalTurns = len(result.Turns)
	result.DurationMs = time.Since(startTime).Milliseconds()
	result.JournalHash = h.prevHash

	telemetry := h.adapter.Telemetry()
	result.VDSOHits = telemetry.KVHits

	return result, nil
}

// executeTool routes tool execution to the container workspace.
func (h *FakHarness) executeTool(ctx context.Context, tc ToolCallProposal) ToolExecutionResult {
	start := time.Now()
	res := ToolExecutionResult{
		ToolCallID: tc.ID,
		Tool:       tc.Name,
		Args:       tc.Arguments,
	}

	var argsMap map[string]interface{}
	_ = json.Unmarshal([]byte(tc.Arguments), &argsMap)

	switch tc.Name {
	case "bash":
		cmdStr, _ := argsMap["cmd"].(string)
		if cmdStr == "" {
			cmdStr, _ = argsMap["command"].(string)
		}
		if cmdStr == "" {
			res.ExitCode = 1
			res.Stderr = "missing 'cmd' argument"
			res.DurationMs = time.Since(start).Milliseconds()
			return res
		}
		execRes, err := h.wsMgr.Exec(ctx, []string{"sh", "-c", cmdStr}, 60*time.Second)
		if err != nil {
			res.ExitCode = 1
			res.Stderr = err.Error()
		} else {
			res.ExitCode = execRes.ExitCode
			res.Stdout = string(execRes.Stdout)
			res.Stderr = string(execRes.Stderr)
		}

	case "read_file":
		path, _ := argsMap["path"].(string)
		if path == "" {
			res.ExitCode = 1
			res.Stderr = "missing 'path' argument"
			res.DurationMs = time.Since(start).Milliseconds()
			return res
		}
		cmd := fmt.Sprintf("cat %s", path)
		execRes, err := h.wsMgr.Exec(ctx, []string{"sh", "-c", cmd}, 30*time.Second)
		if err != nil {
			res.ExitCode = 1
			res.Stderr = err.Error()
		} else {
			res.ExitCode = execRes.ExitCode
			res.Stdout = string(execRes.Stdout)
			res.Stderr = string(execRes.Stderr)
		}

	case "write_file":
		path, _ := argsMap["path"].(string)
		content, _ := argsMap["content"].(string)
		if path == "" {
			res.ExitCode = 1
			res.Stderr = "missing 'path' argument"
			res.DurationMs = time.Since(start).Milliseconds()
			return res
		}
		dir := filepath.Dir(path)
		cmd := fmt.Sprintf("mkdir -p %s && cat << 'EOF' > %s\n%s\nEOF", dir, path, content)
		execRes, err := h.wsMgr.Exec(ctx, []string{"sh", "-c", cmd}, 30*time.Second)
		if err != nil {
			res.ExitCode = 1
			res.Stderr = err.Error()
		} else {
			res.ExitCode = execRes.ExitCode
			res.Stdout = fmt.Sprintf("Wrote %d bytes to %s", len(content), path)
		}

	case "edit_file":
		path, _ := argsMap["path"].(string)
		findStr, _ := argsMap["find"].(string)
		replaceStr, _ := argsMap["replace"].(string)
		if path == "" || findStr == "" {
			res.ExitCode = 1
			res.Stderr = "missing required arguments for edit_file"
			res.DurationMs = time.Since(start).Milliseconds()
			return res
		}
		// Read file, replace string, write back
		readCmd := fmt.Sprintf("cat %s", path)
		readRes, err := h.wsMgr.Exec(ctx, []string{"sh", "-c", readCmd}, 30*time.Second)
		if err != nil || readRes.ExitCode != 0 {
			res.ExitCode = 1
			res.Stderr = fmt.Sprintf("failed to read file for editing: %s", path)
			res.DurationMs = time.Since(start).Milliseconds()
			return res
		}
		updated := strings.Replace(string(readRes.Stdout), findStr, replaceStr, 1)
		dir := filepath.Dir(path)
		writeCmd := fmt.Sprintf("mkdir -p %s && cat << 'EOF' > %s\n%s\nEOF", dir, path, updated)
		_, _ = h.wsMgr.Exec(ctx, []string{"sh", "-c", writeCmd}, 30*time.Second)
		res.ExitCode = 0
		res.Stdout = fmt.Sprintf("Successfully edited %s", path)

	default:
		res.ExitCode = 1
		res.Stderr = fmt.Sprintf("unknown tool %s", tc.Name)
	}

	res.DurationMs = time.Since(start).Milliseconds()
	return res
}

// WriteAuditLog writes the hash-chained audit journal to disk.
func (h *FakHarness) WriteAuditLog(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	for _, entry := range h.journal {
		if _, err := f.WriteString(entry + "\n"); err != nil {
			return err
		}
	}
	return nil
}
