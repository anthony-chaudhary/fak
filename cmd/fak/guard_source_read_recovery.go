package main

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

const guardSourceReadRecoverySchema = "fak-guard-source-read-recovery/1"

type guardSourceReadRecovery struct {
	Schema     string `json:"schema"`
	Tool       string `json:"tool"`
	Path       string `json:"path"`
	Repeat     int    `json:"repeat"`
	Checkpoint string `json:"checkpoint"`
	NextAction string `json:"next_action"`
}

// guardQuarantinedReadRecovery recognizes the #6658/#6703 failure shape from
// the canonical transcript: the same Read call is followed repeatedly by a
// TRUST_VIOLATION quarantine. It returns one bounded restart instruction that
// rotates discovery to a repository-native, read-only command instead of
// replaying the quarantined tool call in another fresh context.
func guardQuarantinedReadRecovery(messages []agent.Message) (string, guardSourceReadRecovery, bool) {
	for _, message := range messages {
		if strings.Contains(message.Content, guardSourceReadRecoverySchema) {
			return "", guardSourceReadRecovery{}, false
		}
	}
	type readFailure struct {
		count int
		path  string
	}
	failures := map[string]readFailure{}
	lastReadID, lastReadKey := "", ""
	for _, message := range messages {
		for _, call := range message.ToolCalls {
			if !strings.EqualFold(strings.TrimSpace(call.Function.Name), "Read") {
				continue
			}
			path := guardReadPath(call.Function.Arguments)
			if path == "" {
				continue
			}
			lastReadID, lastReadKey = call.ID, strings.ToLower(filepath.Clean(path))
			failure := failures[lastReadKey]
			failure.path = path
			failures[lastReadKey] = failure
		}
		if message.Role != agent.RoleTool || !strings.Contains(strings.ToUpper(message.Content), "TRUST_VIOLATION") {
			continue
		}
		if message.ToolCallID != "" && lastReadID != "" && message.ToolCallID != lastReadID {
			continue
		}
		if lastReadKey != "" {
			failure := failures[lastReadKey]
			failure.count++
			failures[lastReadKey] = failure
		}
	}
	var chosen readFailure
	for _, failure := range failures {
		if failure.count >= 2 && failure.count > chosen.count {
			chosen = failure
		}
	}
	if chosen.path == "" {
		return "", guardSourceReadRecovery{}, false
	}
	quoted := strings.ReplaceAll(chosen.path, "'", "''")
	recovery := guardSourceReadRecovery{
		Schema: guardSourceReadRecoverySchema, Tool: "Read", Path: chosen.path, Repeat: chosen.count,
		Checkpoint: "parked-repeated-quarantine", NextAction: "rotate-to-bounded-repository-read",
	}
	text := fmt.Sprintf(`SOURCE_READ_RECOVERY
schema=%s
The Read tool for %q was quarantined with TRUST_VIOLATION %d times before this managed-context restart. Do not replay that Read call. Inspect the same repository source through this bounded read-only alternative, then continue the named task:
PowerShell: Get-Content -LiteralPath '%s' | Select-Object -First 240
If more lines are required, request one explicit bounded range with Select-Object -Skip N -First 240. Before another restart, emit the task's intent/progress claim or a typed park artifact.`, guardSourceReadRecoverySchema, chosen.path, chosen.count, quoted)
	return text, recovery, true
}

func guardReadPath(arguments string) string {
	var obj map[string]any
	if json.Unmarshal([]byte(arguments), &obj) != nil {
		return ""
	}
	for _, key := range []string{"file_path", "path", "filename"} {
		if value, ok := obj[key].(string); ok {
			if value = strings.TrimSpace(value); value != "" {
				return value
			}
		}
	}
	return ""
}
