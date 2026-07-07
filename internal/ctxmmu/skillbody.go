package ctxmmu

// skillbody.go is the screening half of the skill-page policy (#2442): a skill is a
// ctxmmu page whose BODY faults in through ScreenBytes. An injection-shaped or
// secret-bearing skill body seals to a descriptor stub — exactly like a memory note
// — instead of being admitted raw because a config file asked for it. And the
// shell-generated DYNAMIC CONTEXT a skill splices in is admitted like any other
// tool RESULT (taint-stamped, quarantinable through the same Admit gate), never
// trusted because the frontmatter named it.
//
// Both reuse the shipped write-time gate rather than a special-cased trusted path:
// ScreenSkillBody delegates to ScreenBytes (the reusable seal predicate behind the
// memory-note quarantine), and AdmitDynamicContext runs the preprocessing output
// through the ordinary MMU.Admit path so it gets the identical taint-stamp and
// injection/secret screen every tool result gets.

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ScreenSkillBody reports whether a skill BODY must seal to a descriptor stub
// instead of being paged into context raw. It is the named skill-page seam over the
// reusable ScreenBytes predicate (mmu.go) — an injection-shaped or secret-bearing
// body seals with the same reason a memory note would, so a poisoned skill can never
// splice raw bytes into the prompt just because its frontmatter declared it. A
// benign body returns (ReasonNone, false) and pages in normally.
func ScreenSkillBody(body []byte) (abi.ReasonCode, bool) {
	return ScreenBytes(body)
}

// AdmitSkillBody faults a skill body in through the write-time gate: a benign body
// is admitted (VerdictAllow / VerdictTransform for an oversize one), while an
// injection-shaped or secret-bearing body is QUARANTINED — its bytes replaced
// in-place with a descriptor stub and held out of context, page-in-able only after
// a witness Clear, exactly as a memory note seals. It is a thin, purpose-named
// wrapper over MMU.Admit so a skill body travels the SAME path as every other
// result, never a trusted bypass. The call carries the skill name for forensics.
func (m *MMU) AdmitSkillBody(ctx context.Context, skill string, body []byte) (*abi.Result, abi.Verdict) {
	c := &abi.ToolCall{
		Tool: "skill:" + skill,
		Meta: map[string]string{"skill_body": "true"},
	}
	r := &abi.Result{
		Call:    c,
		Status:  abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: body, Taint: abi.TaintTainted},
	}
	return r, m.Admit(ctx, c, r)
}

// AdmitDynamicContext admits a skill's shell-generated DYNAMIC CONTEXT preprocessing
// output as an ordinary, TAINTED tool result. The inspiring harness splices this
// straight into the prompt as trusted text; fak routes it through the same MMU.Admit
// gate every tool result travels, so it is taint-stamped and screened — an
// injection-shaped preprocessing output is quarantined, and a benign one is admitted
// still carrying TaintTainted (never TaintTrusted), quarantinable downstream. It is
// NOT a special trusted path: the descriptor is a tool call so the result inherits
// the default (fail-closed) TaintTainted label.
func (m *MMU) AdmitDynamicContext(ctx context.Context, skill string, out []byte) (*abi.Result, abi.Verdict) {
	c := &abi.ToolCall{
		Tool: "skill-dynamic-context:" + skill,
		Meta: map[string]string{"skill_dynamic_context": "true"},
	}
	r := &abi.Result{
		Call:    c,
		Status:  abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: out, Taint: abi.TaintTainted},
	}
	return r, m.Admit(ctx, c, r)
}
