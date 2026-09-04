package agent

import "github.com/anthony-chaudhary/fak/internal/syspromptmmu"

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

func (c runConfig) seedSystemPrompt() string {
	base := SystemPrompt
	if c.systemPrompt != "" {
		base = c.systemPrompt
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

// seedTools resolves this run's tool catalog: the request-scoped one when wired,
// otherwise the kernel-owned built-in catalog.
func (c runConfig) seedTools() []ToolDef {
	if len(c.toolCatalog) > 0 {
		return c.toolCatalog
	}
	return ToolCatalog()
}

func ownedAgentSystemPrompt() string {
	block := BuildOwnedSystemBlock([][]byte{[]byte(SystemPrompt)}, func(syspromptmmu.BaseEdit) bool { return true })
	if len(block.Value) == 0 {
		return SystemPrompt
	}
	return string(block.Value)
}
