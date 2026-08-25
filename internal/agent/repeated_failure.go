package agent

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/attemptbudget"
)

// canonicalFailureKey identifies semantically identical failed tool calls. It
// returns no key for successful, skipped, malformed, or otherwise unrecognized
// tool results.
func canonicalFailureKey(tool, rawArgs, content string) (key string, failed bool) {
	var receipt ToolReceipt
	if err := json.Unmarshal([]byte(content), &receipt); err == nil && receipt.Status == ToolResultError {
		normalized, err := json.Marshal(receipt)
		if err != nil {
			return "", false
		}
		return joinFailureKey(tool, canonicalArgs(rawArgs), string(normalized)), true
	}

	var result map[string]any
	if err := json.Unmarshal([]byte(content), &result); err != nil {
		return "", false
	}
	errorText, ok := result["error"].(string)
	errorText = strings.TrimSpace(errorText)
	if !ok || errorText == "" {
		return "", false
	}
	return joinFailureKey(tool, canonicalArgs(rawArgs), errorText), true
}

func canonicalArgs(rawArgs string) string {
	var args any
	if err := json.Unmarshal([]byte(rawArgs), &args); err != nil {
		return strings.TrimSpace(rawArgs)
	}
	canonical, err := json.Marshal(args)
	if err != nil {
		return strings.TrimSpace(rawArgs)
	}
	return string(canonical)
}

func joinFailureKey(tool, args, normalizedError string) string {
	key, _ := json.Marshal([]string{tool, args, normalizedError})
	return string(key)
}

func recordRepeatedFailure(t *attemptbudget.RepeatedFailureTracker, tool, rawArgs, content string) bool {
	var rc ToolReceipt
	if json.Unmarshal([]byte(content), &rc) != nil || rc.Status != ToolResultError {
		return t.Record("", true)
	}
	var b bytes.Buffer
	if json.Compact(&b, []byte(rawArgs)) == nil {
		rawArgs = b.String()
	}
	key, _ := canonicalFailureKey(tool, rawArgs, content)
	return t.Record(key, false)
}
