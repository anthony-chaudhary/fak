package journal

import "strings"

// KindToolDefinitionPruned marks an advisory row for one tool definition removed
// from an inbound tools[] surface before the model could call it.
const KindToolDefinitionPruned = "TOOL_DEFINITION_PRUNED"

// AppendToolDefinitionPruned durably records a concrete never-advertised tool
// name. The row intentionally carries no arguments or schema bytes.
func (j *Journal) AppendToolDefinitionPruned(traceID, tool string) Row {
	if j == nil {
		return Row{}
	}
	traceID = strings.TrimSpace(traceID)
	tool = strings.TrimSpace(tool)
	if traceID == "" || tool == "" {
		return Row{}
	}
	row := Row{
		Kind:    KindToolDefinitionPruned,
		Tool:    tool,
		TraceID: traceID,
		Verdict: "ADVISORY",
		Reason:  "DEFAULT_DENY",
		By:      "tool-definition-pruner",
	}
	j.mu.Lock()
	j.appendLocked(row)
	committed := j.recent[len(j.recent)-1]
	j.mu.Unlock()
	return committed
}
