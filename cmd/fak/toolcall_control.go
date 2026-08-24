package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
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
	Schema      string                          `json:"schema"`
	SessionID   string                          `json:"session_id"`
	CallID      string                          `json:"call_id,omitempty"`
	Tool        string                          `json:"tool"`
	Verdict     *toolcallcontrol.RuntimeVerdict `json:"verdict,omitempty"`
	Outcome     *toolcallcontrol.RuntimeReceipt `json:"outcome,omitempty"`
	Disposition string                          `json:"disposition"`
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
	exitCode := toolcallExitCode(p.ToolResponse)
	input := toolcallcontrol.RuntimeInput{
		SessionID:   p.SessionID,
		CallID:      p.ToolUseID,
		Tool:        p.ToolName,
		Args:        toolcallSemanticArgs(p.ToolInput),
		ReadOnly:    readOnly,
		Succeeded:   toolcallSucceeded(p.ToolResponse, exitCode),
		PromptUnits: toolcallPromptUnits(payload),
		Declaration: toolcallOutcomeDeclaration(payload, p.ToolInput),
		ExitCode:    exitCode,
		Output:      append(json.RawMessage(nil), p.ToolResponse...),
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
		if err := appendToolcallTrace(dir, toolcallTraceRow{Schema: "fak-toolcall-control-decision/1", SessionID: p.SessionID, CallID: p.ToolUseID, Tool: p.ToolName, Verdict: &verdict, Disposition: disposition}); err != nil {
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
		if err := writeToolcallState(statePath, state); err != nil {
			return err
		}
		outcome := toolcallcontrol.ClassifyOutcome(input)
		return appendToolcallTrace(dir, toolcallTraceRow{
			Schema:      "fak-toolcall-control-outcome/1",
			SessionID:   p.SessionID,
			CallID:      p.ToolUseID,
			Tool:        p.ToolName,
			Outcome:     &outcome,
			Disposition: string(outcome.Projection),
		})
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
	for _, key := range []string{"fak_prompt_units", "prompt_units", "prompt_tokens", "fak_expected_negative", "fak_outcome_class", "fak_outcome"} {
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

func toolcallSucceeded(raw json.RawMessage, exitCode *int) bool {
	if exitCode != nil && *exitCode != 0 {
		return false
	}
	if len(raw) == 0 {
		return true
	}
	var response map[string]any
	if json.Unmarshal(raw, &response) == nil {
		for _, key := range []string{"is_error", "error", "failed", "timed_out", "interrupted"} {
			if value, ok := response[key].(bool); ok && value {
				return false
			}
		}
		if status, ok := response["status"].(string); ok {
			switch strings.ToLower(strings.TrimSpace(status)) {
			case "error", "failed", "failure", "timed_out", "timeout", "interrupted", "cancelled", "canceled":
				return false
			}
		}
	}
	text := strings.ToLower(string(raw))
	return !strings.Contains(text, `"is_error":true`) && !strings.Contains(text, `"error":true`)
}

func toolcallExitCode(raw json.RawMessage) *int {
	if len(raw) == 0 {
		return nil
	}
	var response map[string]any
	if json.Unmarshal(raw, &response) == nil {
		for _, key := range []string{"exit_code", "exitCode"} {
			if code, ok := outcomeInt(response[key]); ok {
				return &code
			}
		}
	}
	lower := strings.ToLower(string(raw))
	const marker = "exit code:"
	if at := strings.Index(lower, marker); at >= 0 {
		rest := strings.TrimSpace(lower[at+len(marker):])
		end := 0
		for end < len(rest) && (rest[end] == '-' || rest[end] >= '0' && rest[end] <= '9') {
			end++
		}
		if end > 0 {
			if code, err := strconv.Atoi(rest[:end]); err == nil {
				return &code
			}
		}
	}
	return nil
}

func outcomeInt(value any) (int, bool) {
	switch value := value.(type) {
	case float64:
		converted := int(value)
		return converted, float64(converted) == value
	case string:
		converted, err := strconv.Atoi(strings.TrimSpace(value))
		return converted, err == nil
	default:
		return 0, false
	}
}

func toolcallOutcomeDeclaration(payload, toolInput json.RawMessage) toolcallcontrol.OutcomeDeclaration {
	var expected []bool
	var classes []toolcallcontrol.OutcomeClass
	invalid := false
	collect := func(raw json.RawMessage) {
		var object map[string]any
		if len(raw) == 0 || json.Unmarshal(raw, &object) != nil {
			return
		}
		expectedValue, hasExpected := object["fak_expected_negative"]
		classValue, hasClass := object["fak_outcome_class"]
		invalid = appendToolcallOutcomeFields(&expected, &classes, expectedValue, hasExpected, classValue, hasClass) || invalid
		if value, ok := object["fak_outcome"]; ok {
			declaration, typed := value.(map[string]any)
			if !typed {
				invalid = true
				return
			}
			expectedValue, hasExpected := declaration["expected_negative"]
			classValue, hasClass := declaration["class"]
			invalid = appendToolcallOutcomeFields(&expected, &classes, expectedValue, hasExpected, classValue, hasClass) || invalid
		}
	}
	collect(payload)
	collect(toolInput)
	declaration := toolcallcontrol.OutcomeDeclaration{Invalid: invalid}
	for _, marker := range expected {
		if declaration.ExpectedNegativeSet && marker != declaration.ExpectedNegative {
			declaration.Invalid = true
		}
		declaration.ExpectedNegative = marker
		declaration.ExpectedNegativeSet = true
	}
	for _, class := range classes {
		if declaration.Class != "" && class != declaration.Class {
			declaration.Invalid = true
		}
		declaration.Class = class
	}
	return declaration
}

func toolcallExpectedNegative(value any) (bool, bool) {
	marker, valid := value.(bool)
	return marker, valid
}

func toolcallOutcomeClass(value any) (toolcallcontrol.OutcomeClass, bool) {
	class, valid := value.(string)
	class = strings.TrimSpace(class)
	return toolcallcontrol.OutcomeClass(strings.ToLower(class)), valid && class != ""
}

func appendToolcallOutcomeFields(expected *[]bool, classes *[]toolcallcontrol.OutcomeClass, expectedValue any, hasExpected bool, classValue any, hasClass bool) bool {
	invalid := false
	if hasExpected {
		marker, valid := toolcallExpectedNegative(expectedValue)
		if valid {
			*expected = append(*expected, marker)
		} else {
			invalid = true
		}
	}
	if hasClass {
		class, valid := toolcallOutcomeClass(classValue)
		if valid {
			*classes = append(*classes, class)
		} else {
			invalid = true
		}
	}
	return invalid
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
