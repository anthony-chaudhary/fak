package policy_test

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// bareCall builds a benign, read-shaped tool call — enough to exercise the floor's
// name-level allow/deny without tripping an unrelated write-class rung.
func bareCall(tool string) *abi.ToolCall {
	return &abi.ToolCall{
		Tool: tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)},
		Meta: map[string]string{"readOnlyHint": "true"},
	}
}

// TestSkillSpan_ToolOutsideFrontmatterRefused is the #2442 kernel-policy witness: a
// skill's frontmatter compiles (through the manifest path) into a span-scoped
// envelope, and mid-span a tool OUTSIDE the frontmatter-declared allowed set is
// refused by the kernel with DEFAULT_DENY — not by hoping the model reads the
// frontmatter. The narrowing is span-scoped: it applies to the skill's activation
// window only and does not leak into the session's base tool envelope.
func TestSkillSpan_ToolOutsideFrontmatterRefused(t *testing.T) {
	ctx := context.Background()

	// The session's base floor admits BOTH tools.
	base := adjudicator.Policy{
		Posture: adjudicator.PostureFailClosed,
		Allow:   map[string]bool{"read_file": true, "run_shell": true},
	}

	// The skill's frontmatter declares only read_file as allowed. Compile it through
	// the manifest compile path into a span-scoped envelope.
	span, err := policy.CompileSkillSpan(policy.SkillFrontmatter{
		Name:         "summarize",
		Description:  "summarize a file",
		AllowedTools: []string{"read_file"},
		Model:        "claude-haiku",
	})
	if err != nil {
		t.Fatalf("CompileSkillSpan: %v", err)
	}

	// Resident cost is name+description bytes only (progressive disclosure).
	if got, want := span.ResidentBytes(), len("summarize")+len("summarize a file"); got != want {
		t.Fatalf("ResidentBytes = %d, want %d", got, want)
	}

	// Mid-span: the effective floor is the span floor.
	spanFloor := span.SpanFloor(base)
	midSpan := adjudicator.New(spanFloor)

	// A tool INSIDE the frontmatter set is admitted mid-span.
	if v := midSpan.Adjudicate(ctx, bareCall("read_file")); v.Kind == abi.VerdictDeny {
		t.Fatalf("read_file mid-span: want admitted, got Deny (reason %s)", abi.ReasonName(v.Reason))
	}

	// A tool OUTSIDE the frontmatter set is REFUSED by the kernel mid-span, even
	// though the base session would have allowed it.
	v := midSpan.Adjudicate(ctx, bareCall("run_shell"))
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("run_shell mid-span: want VerdictDeny, got %v", v.Kind)
	}
	if v.Reason != abi.ReasonDefaultDeny {
		t.Fatalf("run_shell mid-span: want DEFAULT_DENY, got %s", abi.ReasonName(v.Reason))
	}

	// Span-scoping: the narrowing does NOT leak. Restoring the base floor re-admits
	// run_shell — the skill's envelope applied only for its activation window.
	afterSpan := adjudicator.New(base)
	if v := afterSpan.Adjudicate(ctx, bareCall("run_shell")); v.Kind == abi.VerdictDeny {
		t.Fatalf("run_shell after span: want re-admitted, got Deny — span narrowing leaked")
	}
}

// TestSkillSpan_EmptyAllowedToolsFailsClosed pins the #2442 fail-closed invariant a
// security envelope lives or dies on: a skill frontmatter that declares NO allowed
// tools must compile to the EMPTY envelope — the skill may call nothing — never a
// wide-open one that silently inherits the base floor, and a blank name must refuse
// to compile a span at all. A regression that let an empty allow-set fall through to
// admit-everything would turn the whole span boundary into a no-op.
func TestSkillSpan_EmptyAllowedToolsFailsClosed(t *testing.T) {
	ctx := context.Background()

	// A name is required — a frontmatter with a blank name cannot compile a span.
	if _, err := policy.CompileSkillSpan(policy.SkillFrontmatter{Name: "   "}); err == nil {
		t.Fatalf("CompileSkillSpan with blank name: want error, got nil")
	}

	// The base session admits both tools.
	base := adjudicator.Policy{
		Posture: adjudicator.PostureFailClosed,
		Allow:   map[string]bool{"read_file": true, "run_shell": true},
	}

	// No AllowedTools declared: the span must compile to the fail-closed EMPTY
	// envelope, not silently inherit the base floor.
	span, err := policy.CompileSkillSpan(policy.SkillFrontmatter{
		Name:        "note",
		Description: "declares no tools",
	})
	if err != nil {
		t.Fatalf("CompileSkillSpan: %v", err)
	}

	// Mid-span EVERY tool the base would have admitted is refused with DEFAULT_DENY —
	// an empty allow-set means the skill may call nothing.
	midSpan := adjudicator.New(span.SpanFloor(base))
	for _, tool := range []string{"read_file", "run_shell"} {
		v := midSpan.Adjudicate(ctx, bareCall(tool))
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonDefaultDeny {
			t.Fatalf("%s mid-span with empty allow-set: want DEFAULT_DENY, got kind=%v reason=%s",
				tool, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}

// TestSkillSpan_CannotWidenBase proves a skill can only NARROW: a frontmatter tool
// the base session never admits is dropped from the span floor, so a skill can never
// grant itself authority the operator floor withholds.
func TestSkillSpan_CannotWidenBase(t *testing.T) {
	ctx := context.Background()
	base := adjudicator.Policy{
		Posture: adjudicator.PostureFailClosed,
		Allow:   map[string]bool{"read_file": true},
	}
	span, err := policy.CompileSkillSpan(policy.SkillFrontmatter{
		Name:         "escalate",
		Description:  "tries to reach a tool the base denies",
		AllowedTools: []string{"read_file", "delete_everything"},
	})
	if err != nil {
		t.Fatalf("CompileSkillSpan: %v", err)
	}
	midSpan := adjudicator.New(span.SpanFloor(base))
	if v := midSpan.Adjudicate(ctx, bareCall("delete_everything")); v.Kind != abi.VerdictDeny {
		t.Fatalf("delete_everything: want VerdictDeny (base never admits it), got %v", v.Kind)
	}
}
