package agent

import (
	"fmt"
	"strings"
)

// ChildPreamble is the parent's stable request head: its leading system message
// and the tool schemas it declared. A child that sends both unchanged renders the
// same leading system/tools block, so backend prefix caches reuse it; a child with
// its own system text diverges a few tokens in and reuses almost nothing.
type ChildPreamble struct {
	System string
	Tools  []ToolDef
}

func (s *TaskState) setChildPreamble(messages []Message, tools []ToolDef) {
	if s == nil || len(messages) == 0 || messages[0].Role != RoleSystem || messages[0].Content == "" {
		return
	}
	p := &ChildPreamble{System: messages[0].Content, Tools: append([]ToolDef(nil), tools...)}
	s.mu.Lock()
	s.preamble = p
	s.mu.Unlock()
}

// nativeChildBrief carries the child's role-specific instructions in the user turn,
// after the shared preamble. Renderers fold every system message into the leading
// block ahead of the tools, so a second system message would break the shared prefix.
func nativeChildBrief(req ChildTaskRunRequest, catalog []ToolDef) string {
	names := make([]string, 0, len(catalog))
	for _, def := range catalog {
		names = append(names, def.Function.Name)
	}
	allowed := "none"
	if len(names) > 0 {
		allowed = strings.Join(names, ", ")
	}
	var b strings.Builder
	b.WriteString("You are running as a delegated subtask")
	if role := strings.TrimSpace(req.SubagentType); role != "" {
		fmt.Fprintf(&b, " (role: %s)", role)
	}
	fmt.Fprintf(&b, ". Tools you may call: %s. A call to any other declared tool ends this subtask with an error.", allowed)
	if desc := strings.TrimSpace(req.Description); desc != "" {
		fmt.Fprintf(&b, "\nDescription: %s", desc)
	}
	b.WriteString("\n\nTask:\n")
	b.WriteString(req.Prompt)
	return b.String()
}
