package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/flock"
	"github.com/anthony-chaudhary/fak/internal/toolcallcontrol"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

type toolcallHookOutput struct {
	HookSpecificOutput struct {
		HookEventName            string `json:"hookEventName"`
		PermissionDecision       string `json:"permissionDecision"`
		PermissionDecisionReason string `json:"permissionDecisionReason"`
	} `json:"hookSpecificOutput"`
}

type toolcallTraceRow struct {
	Schema      string                         `json:"schema"`
	SessionID   string                         `json:"session_id"`
	CallID      string                         `json:"call_id,omitempty"`
	Tool        string                         `json:"tool"`
	Verdict     toolcallcontrol.RuntimeVerdict `json:"verdict"`
	Disposition string                         `json:"disposition"`
}

func toolcallControlHook(stdout io.Writer, stdin io.Reader, kind string, mode toolcallcontrol.Mode, dir string) error {
	if strings.TrimSpace(dir) == "" {
		dir = filepath.Join(os.TempDir(), "fak-toolcall-control")
	}
	payload, err := io.ReadAll(io.LimitReader(stdin, 4<<20))
	if err != nil {
		return err
	}
	var p toolproc.HookPayload
	if err := json.Unmarshal(payload, &p); err != nil {
		return fmt.Errorf("parse payload: %w", err)
	}
	if strings.TrimSpace(p.SessionID) == "" {
		return fmt.Errorf("missing session_id")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	lockPath := filepath.Join(dir, toolcallFileStem(p.SessionID)+".lock")
	lock, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return err
	}
	defer lock.Close()
	if err := flock.TryLock(lock); err != nil {
		return fmt.Errorf("state lock: %w", err)
	}
	defer flock.Unlock(lock)

	statePath := filepath.Join(dir, toolcallFileStem(p.SessionID)+".json")
	state, err := readToolcallState(statePath)
	if err != nil {
		return err
	}
	readOnly := toolcallReadOnly(p.ToolName)
	input := toolcallcontrol.RuntimeInput{
		SessionID:   p.SessionID,
		CallID:      p.ToolUseID,
		Tool:        p.ToolName,
		Args:        toolcallSemanticArgs(p.ToolInput),
		ReadOnly:    readOnly,
		Succeeded:   toolcallSucceeded(p.ToolResponse),
		PromptUnits: toolcallPromptUnits(payload),
	}
	switch kind {
	case "pre":
		verdict := toolcallcontrol.Before(mode, state, input)
		disposition := "executed"
		if verdict.Action == toolcallcontrol.Reuse {
			disposition = "counterfactual_reuse"
			if verdict.Applied {
				disposition = "suppressed"
			}
		}
		if err := appendToolcallTrace(dir, toolcallTraceRow{Schema: "fak-toolcall-control-decision/1", SessionID: p.SessionID, CallID: p.ToolUseID, Tool: p.ToolName, Verdict: verdict, Disposition: disposition}); err != nil {
			return err
		}
		if verdict.Applied && verdict.Action == toolcallcontrol.Reuse {
			var out toolcallHookOutput
			out.HookSpecificOutput.HookEventName = "PreToolUse"
			out.HookSpecificOutput.PermissionDecision = "deny"
			out.HookSpecificOutput.PermissionDecisionReason = "FAK_TOOLCALL_REUSE: exact fresh result already observed at this mutation epoch; reuse " + verdict.ResultRef
			return json.NewEncoder(stdout).Encode(out)
		}
		return nil
	case "post":
		input.ResultRef = p.ToolUseID
		state = toolcallcontrol.After(state, input, 128)
		return writeToolcallState(statePath, state)
	default:
		return nil
	}
}

func toolcallReadOnly(tool string) bool {
	t := strings.ToLower(strings.TrimSpace(tool))
	if i := strings.LastIndex(t, "__"); i >= 0 {
		t = t[i+2:]
	}
	switch t {
	case "read", "grep", "glob", "websearch", "webfetch", "search", "list", "view", "get", "query":
		return true
	default:
		return false
	}
}

func toolcallSemanticArgs(raw json.RawMessage) json.RawMessage {
	var input map[string]any
	if json.Unmarshal(raw, &input) != nil {
		return raw
	}
	for _, key := range []string{"fak_prompt_units", "prompt_units", "prompt_tokens"} {
		delete(input, key)
	}
	data, err := json.Marshal(input)
	if err != nil {
		return raw
	}
	return data
}

func toolcallPromptUnits(raw json.RawMessage) int64 {
	if len(raw) == 0 {
		return 0
	}
	var input map[string]any
	if json.Unmarshal(raw, &input) != nil {
		return 0
	}
	for _, key := range []string{"fak_prompt_units", "prompt_units", "prompt_tokens"} {
		if value, ok := input[key].(float64); ok && value > 0 {
			return int64(value)
		}
	}
	if usage, ok := input["usage"].(map[string]any); ok {
		for _, key := range []string{"input_tokens", "prompt_tokens"} {
			if value, ok := usage[key].(float64); ok && value > 0 {
				return int64(value)
			}
		}
	}
	return 0
}

func toolcallSucceeded(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return true
	}
	text := strings.ToLower(string(raw))
	return !strings.Contains(text, `"is_error":true`) && !strings.Contains(text, `"error":true`)
}

func toolcallFileStem(session string) string {
	var b strings.Builder
	for _, r := range session {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			b.WriteRune(r)
		}
	}
	if b.Len() == 0 {
		return "session"
	}
	sum := sha256.Sum256([]byte(session))
	return b.String() + "-" + hex.EncodeToString(sum[:4])
}

func readToolcallState(path string) (toolcallcontrol.RuntimeState, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return toolcallcontrol.RuntimeState{}, nil
	}
	if err != nil {
		return toolcallcontrol.RuntimeState{}, err
	}
	var state toolcallcontrol.RuntimeState
	if err := json.Unmarshal(data, &state); err != nil {
		return toolcallcontrol.RuntimeState{}, fmt.Errorf("parse state: %w", err)
	}
	return state, nil
}

func writeToolcallState(path string, state toolcallcontrol.RuntimeState) error {
	data, err := json.Marshal(state)
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(data, '\n'), 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func appendToolcallTrace(dir string, row toolcallTraceRow) error {
	data, err := json.Marshal(row)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(filepath.Join(dir, "decisions.jsonl"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(append(data, '\n'))
	return err
}
