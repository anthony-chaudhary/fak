package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/canon"
)

const nativeAgentReceiptSchema = "fak.agent.native.v1"
const nativeAgentCallsSchema = "fak.agent.native.calls.v1"

const nativeAgentCallPreviewLimit = 160

type nativeAgentCalls struct {
	Schema  string            `json:"schema"`
	Entries []agent.CallTrace `json:"entries"`
}

type nativeAgentReceipt struct {
	Schema       string            `json:"schema"`
	Task         string            `json:"task"`
	Model        string            `json:"model"`
	Status       string            `json:"status,omitempty"`
	TouchedPaths []string          `json:"touched_paths,omitempty"`
	GitDiffHash  string            `json:"git_diff_hash,omitempty"`
	FinalAnswer  string            `json:"final_answer,omitempty"`
	Metrics      agent.ArmMetrics  `json:"metrics"`
	Calls        *nativeAgentCalls `json:"calls,omitempty"`
}

func newNativeAgentReceipt(task, model string, metrics agent.ArmMetrics) nativeAgentReceipt {
	return nativeAgentReceipt{
		Schema:  nativeAgentReceiptSchema,
		Task:    task,
		Model:   model,
		Metrics: metrics,
	}
}

func nativeAgentCallPreview(value string) string {
	value = strings.Join(strings.Fields(value), " ")
	if value == "" {
		return ""
	}
	raw := []byte(value)
	if !canon.RawSecretComplete(raw) {
		return "[redacted:secret]"
	}
	redacted, _ := canon.RedactSecrets(raw)
	if canon.Scan(redacted).Secret {
		return "[redacted:secret]"
	}
	value = string(redacted)
	if len(value) <= nativeAgentCallPreviewLimit {
		return value
	}
	const suffix = "..."
	limit := nativeAgentCallPreviewLimit - len(suffix)
	for limit > 0 && !utf8.RuneStart(value[limit]) {
		limit--
	}
	return value[:limit] + suffix
}

func nativeAgentCallsFromTrace(calls []agent.CallTrace) *nativeAgentCalls {
	if len(calls) == 0 {
		return nil
	}
	entries := make([]agent.CallTrace, len(calls))
	for i, call := range calls {
		entries[i] = call
		entries[i].Args = nativeAgentCallPreview(call.Args)
		entries[i].Note = nativeAgentCallPreview(call.Note)
	}
	return &nativeAgentCalls{Schema: nativeAgentCallsSchema, Entries: entries}
}

func extractTouchedPaths(calls []agent.CallTrace) []string {
	seen := make(map[string]struct{})
	pathKeys := []string{"file_path", "path", "filePath", "filepath"}
	listKeys := []string{"file_paths", "paths"}

	for _, c := range calls {
		argsStr := strings.TrimSpace(c.Args)
		if argsStr == "" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(argsStr), &m); err == nil {
			for _, k := range pathKeys {
				if v, ok := m[k]; ok {
					if s, ok := v.(string); ok && strings.TrimSpace(s) != "" {
						seen[strings.TrimSpace(s)] = struct{}{}
					}
				}
			}
			for _, k := range listKeys {
				if v, ok := m[k]; ok {
					if arr, ok := v.([]any); ok {
						for _, item := range arr {
							if s, ok := item.(string); ok && strings.TrimSpace(s) != "" {
								seen[strings.TrimSpace(s)] = struct{}{}
							}
						}
					}
				}
			}
		} else {
			re := regexp.MustCompile(`"(?:file_path|path|filePath|filepath)"\s*:\s*"([^"]+)"`)
			for _, match := range re.FindAllStringSubmatch(argsStr, -1) {
				if len(match) > 1 && strings.TrimSpace(match[1]) != "" {
					seen[strings.TrimSpace(match[1])] = struct{}{}
				}
			}
		}
	}
	if len(seen) == 0 {
		return nil
	}
	paths := make([]string, 0, len(seen))
	for p := range seen {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	return paths
}

func computeGitDiffHash(workspace string) string {
	ws := strings.TrimSpace(workspace)
	if ws == "" {
		var err error
		ws, err = os.Getwd()
		if err != nil {
			return ""
		}
	}
	checkCmd := exec.Command("git", "rev-parse", "--is-inside-work-tree")
	checkCmd.Dir = ws
	configureDispatchHelperCommand(checkCmd)
	if err := checkCmd.Run(); err != nil {
		return ""
	}
	cmd := exec.Command("git", "diff", "HEAD")
	cmd.Dir = ws
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		cmd = exec.Command("git", "diff")
		cmd.Dir = ws
		configureDispatchHelperCommand(cmd)
		out, err = cmd.Output()
		if err != nil {
			return ""
		}
	}
	if len(bytes.TrimSpace(out)) == 0 {
		return ""
	}
	h := sha256.Sum256(out)
	return fmt.Sprintf("sha256:%x", h)
}

func newHeadlessAgentReceipt(task, model string, metrics agent.ArmMetrics, calls []agent.CallTrace, workspace string, runErr error) nativeAgentReceipt {
	var status string
	if runErr != nil {
		status = "failed"
	} else if metrics.CircuitBreakerTripped {
		status = "stopped_circuit_breaker"
	} else if metrics.HitTurnCap {
		status = "turn_cap_exceeded"
	} else {
		status = "completed"
	}

	return nativeAgentReceipt{
		Schema:       nativeAgentReceiptSchema,
		Task:         task,
		Model:        model,
		Status:       status,
		TouchedPaths: extractTouchedPaths(calls),
		GitDiffHash:  computeGitDiffHash(workspace),
		FinalAnswer:  metrics.FinalAnswer,
		Metrics:      metrics,
		Calls:        nativeAgentCallsFromTrace(calls),
	}
}

const rawAgentReceiptSchema = "agent.raw-receipt.v1"

type rawAgentReceipt struct {
	Schema        string           `json:"schema"`
	Mode          string           `json:"mode"`
	FakMediated   bool             `json:"fak_mediated"`
	Adjudications int              `json:"adjudications"`
	Task          string           `json:"task"`
	Model         string           `json:"model"`
	Metrics       agent.ArmMetrics `json:"metrics"`
}

func newRawAgentReceipt(task, model string, metrics agent.ArmMetrics) rawAgentReceipt {
	return rawAgentReceipt{
		Schema:        rawAgentReceiptSchema,
		Mode:          "raw",
		FakMediated:   false,
		Adjudications: 0,
		Task:          task,
		Model:         model,
		Metrics:       metrics,
	}
}
