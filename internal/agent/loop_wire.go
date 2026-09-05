package agent

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
)

// loop_wire.go — the two RunOptions a WIRE caller needs to hand the owned loop a real
// served request instead of a reconstruction (#6657).
//
// runArm has always seeded itself with a fixed two-message transcript (its system prompt
// plus one task string) and the built-in ToolCatalog(). That is right for the A/B
// benchmark arms it was written for, where the task IS a string and the catalog IS the
// fixture. It is wrong for `fak serve --native`, where a client posted an ordered
// conversation and its own tools[]: flattening that to lastUserText() throws away every
// prior turn, and ignoring tools[] advertises a fixture the caller never asked for.
//
// Both options are strictly additive: unset, resolveRunConfig leaves them zero and the
// loop is byte-for-byte the historical loop.

// WithConversation seeds the owned loop with the caller's ORDERED transcript instead of
// the single task string. The loop's own system prompt still leads; msgs is spliced
// directly after it with roles and content preserved, so prior user/assistant turns and
// tool results reach the model exactly as the caller sent them.
//
// An empty (or nil) msgs leaves the historical task-only seed, so this is a no-op for
// every existing caller.
func WithConversation(msgs []Message) RunOption {
	return func(c *runConfig) { c.conversation = msgs }
}

// WithToolCatalog replaces the built-in ToolCatalog() with a REQUEST-SCOPED catalog for
// this run. It is the caller's declared tool surface, advertised to the model verbatim;
// nothing from the fixture is blended in, because a blended catalog would let the model
// call a tool the caller never declared.
//
// An empty (or nil) tools leaves ToolCatalog() standing — the existing no-tools run.
func WithToolCatalog(tools []ToolDef) RunOption {
	return func(c *runConfig) { c.toolCatalog = tools }
}

// CodeAgentSystemPrompt is the default standing instruction when repository coding tools are armed.
const CodeAgentSystemPrompt = "You are a software engineering agent working in this repository. Use the provided tools (Read, Write, Edit, Grep, Glob, Bash) to inspect, analyze, and modify code to complete the user's request. Always verify changes and report accurate, concise outcomes."

// WithSystemPrompt overrides the loop's standing system prompt for this run.
func WithSystemPrompt(prompt string) RunOption {
	return func(c *runConfig) { c.systemPrompt = prompt }
}

// WithMemoryDigest seeds the owned loop with a verified workspace memory digest
// block. It is rendered into the leading system context right after the loop's
// standing instruction, preserving cache stability while orienting the agent on
// verified workspace knowledge.
func WithMemoryDigest(digest string) RunOption {
	return func(c *runConfig) { c.memoryDigest = digest }
}

func (c runConfig) hasCodingTools() bool {
	if armedCodeTools.Load() != nil {
		return true
	}
	for _, t := range c.seedTools() {
		switch strings.ToLower(t.Function.Name) {
		case "read", "write", "edit", "grep", "glob", "bash":
			return true
		}
	}
	return false
}

func (c runConfig) seedSystemPrompt() string {
	base := SystemPrompt
	if c.systemPrompt != "" {
		base = c.systemPrompt
	} else if c.hasCodingTools() {
		base = CodeAgentSystemPrompt
	}
	block := BuildOwnedSystemBlock([][]byte{[]byte(base)}, func(syspromptmmu.BaseEdit) bool { return true })
	if len(block.Value) == 0 {
		return base
	}
	return string(block.Value)
}

// seedMessages builds the transcript the loop opens with: the loop's system prompt,
// optional memory digest, then either the wired conversation or the single task message.
func (c runConfig) seedMessages(task string) []Message {
	if len(c.conversation) > 0 && c.conversation[0].Role == RoleSystem {
		return append([]Message(nil), c.conversation...)
	}
	msgs := make([]Message, 0, len(c.conversation)+3)
	msgs = append(msgs, Message{Role: RoleSystem, Content: c.seedSystemPrompt()})
	if c.memoryDigest != "" {
		msgs = append(msgs, Message{Role: RoleSystem, Content: c.memoryDigest})
	}
	if len(c.conversation) > 0 {
		return append(msgs, c.conversation...)
	}
	return append(msgs, Message{Role: RoleUser, Content: task})
}

// WithTodoTools arms the kernel-mediated planning and todo tools (todowrite, todoread)
// and merges them into the run's tool catalog.
func WithTodoTools() RunOption {
	return func(c *runConfig) {
		_, _ = ArmTodoTools()
		c.todoTools = true
	}
}

// WithContextControl arms the context_control tool and merges it into the run's tool catalog.
func WithContextControl(opts ...ContextControlOption) RunOption {
	return func(c *runConfig) {
		_, _ = ArmContextControl(opts...)
		c.contextControl = true
	}
}

// seedTools resolves this run's tool catalog: the request-scoped one when wired,
// otherwise the kernel-owned built-in catalog.
func (c runConfig) seedTools() []ToolDef {
	base := c.toolCatalog
	if len(base) == 0 {
		base = ToolCatalog()
	}
	if c.todoTools {
		hasTodo := false
		for _, t := range base {
			if t.Function.Name == ToolTodoWrite {
				hasTodo = true
				break
			}
		}
		if !hasTodo {
			todoDefs := TodoToolCatalog()
			if len(todoDefs) == 0 {
				todoDefs = todoToolDefs()
			}
			base = append(append([]ToolDef(nil), base...), todoDefs...)
		}
	}
	if c.contextControl {
		hasCC := false
		for _, t := range base {
			if t.Function.Name == ToolContextControl {
				hasCC = true
				break
			}
		}
		if !hasCC {
			ccDefs := ContextControlCatalog()
			if len(ccDefs) == 0 {
				ccDefs = contextControlToolDefs()
			}
			base = append(append([]ToolDef(nil), base...), ccDefs...)
		}
	}
	return base
}

func ownedAgentSystemPrompt() string {
	prompt := SystemPrompt
	if armedCodeTools.Load() != nil {
		prompt = CodeAgentSystemPrompt
	}
	block := BuildOwnedSystemBlock([][]byte{[]byte(prompt)}, func(syspromptmmu.BaseEdit) bool { return true })
	if len(block.Value) == 0 {
		return prompt
	}
	return string(block.Value)
}
