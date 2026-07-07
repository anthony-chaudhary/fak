package policy

// skill.go compiles a skill page's frontmatter into a SPAN-SCOPED capability
// envelope (#2442). A skill is the right shape — name+description resident, body
// faulted in on invoke, frontmatter declaring the tools/model/effort the skill may
// use — but the inspiring harness enforces that frontmatter BY CONVENTION (it hopes
// the model reads it). fak's version compiles the frontmatter through the SAME
// manifest compile path the operator floor uses (Manifest.ToPolicy) into an
// adjudicator envelope, so a tool outside the skill's declared allow-set is refused
// by the KERNEL for the skill's activation window — not by hoping.
//
// The narrowing is span-scoped: SpanFloor intersects the frontmatter's declared
// tools with the session's base floor (a skill can only ever NARROW the session
// envelope, never widen it — the frontmatter is a boundary, not a grant), and the
// caller restores the base floor when the span ends, so the narrowing never leaks
// into the rest of the session's tool envelope.
//
// The body-screening half (an injection-shaped or secret-bearing skill body seals
// to a descriptor stub, same as a memory note) lives in internal/ctxmmu, reusing
// the shipped ScreenBytes predicate — this file is the kernel-policy half.

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// SkillFrontmatter is a skill page's declarative header. Name+Description are the
// RESIDENT card — the only bytes that stay in context when the skill is not active
// (the body faults in on invoke, and seals if unsafe). AllowedTools/Model/Effort
// declare the EXECUTION BOUNDARY applied for the skill's activation span.
type SkillFrontmatter struct {
	Name         string   `json:"name"`
	Description  string   `json:"description"`
	AllowedTools []string `json:"allowed_tools,omitempty"`
	Model        string   `json:"model,omitempty"`
	Effort       string   `json:"effort,omitempty"`
}

// ResidentBytes is a skill's resident cost: the name+description bytes only. The
// body is NOT resident (it faults in on invoke and seals if unsafe), so a large
// skill costs the context exactly its card, never its body — the progressive-
// disclosure invariant fak_capabilities reports.
func (fm SkillFrontmatter) ResidentBytes() int {
	return len(fm.Name) + len(fm.Description)
}

// SkillSpan is a compiled skill activation envelope: the resident card fields plus
// the adjudicator policy the frontmatter's allowed-tool set compiles to. Build it
// with CompileSkillSpan; apply it for the activation window with SpanFloor.
type SkillSpan struct {
	Name        string
	Description string
	Model       string
	Effort      string
	// Policy is the frontmatter's allowed-tool set compiled through the manifest
	// path into a fail-closed adjudicator envelope: exactly the declared tools are
	// affirmatively allowed, everything else resolves to DEFAULT_DENY.
	Policy adjudicator.Policy
}

// ResidentBytes reports the span's resident card cost (name+description bytes).
func (s SkillSpan) ResidentBytes() int { return len(s.Name) + len(s.Description) }

// CompileSkillSpan compiles a skill's frontmatter into a span-scoped envelope by
// routing it through the SAME manifest compile path (Manifest.ToPolicy) the
// operator floor uses — the frontmatter's allowed tools become a fail-closed
// allow-list, so a tool outside the set is refused at the floor. An empty
// AllowedTools compiles to the fail-closed empty envelope (the skill may call
// nothing), never a wide-open one. A name is required.
func CompileSkillSpan(fm SkillFrontmatter) (SkillSpan, error) {
	if strings.TrimSpace(fm.Name) == "" {
		return SkillSpan{}, fmt.Errorf("skill: name is required")
	}
	allow := make([]string, 0, len(fm.AllowedTools))
	for _, t := range fm.AllowedTools {
		if t = strings.TrimSpace(t); t != "" {
			allow = append(allow, t)
		}
	}
	sort.Strings(allow)
	m := Manifest{
		Version: Version,
		// Posture omitted => fail_closed: anything not in Allow is DEFAULT_DENY.
		Allow: allow,
	}
	p, err := m.ToPolicy()
	if err != nil {
		return SkillSpan{}, fmt.Errorf("skill %q: %w", fm.Name, err)
	}
	return SkillSpan{
		Name:        fm.Name,
		Description: fm.Description,
		Model:       fm.Model,
		Effort:      fm.Effort,
		Policy:      p,
	}, nil
}

// SpanFloor returns the effective adjudicator floor for the skill's activation
// window: the frontmatter's declared tools INTERSECTED with what the session base
// floor would ever admit. A skill can therefore only NARROW the session envelope —
// a frontmatter tool the base session never admits is dropped, so a skill can never
// widen its own authority past the operator floor (the frontmatter is a boundary,
// not a grant). The returned floor is fail-closed and enumerates exactly the
// surviving tools; everything else (including any base AllowPrefix family) resolves
// to DEFAULT_DENY for the span. When the span ends the caller restores base, so the
// narrowing never leaks into the rest of the session's tool envelope.
func (s SkillSpan) SpanFloor(base adjudicator.Policy) adjudicator.Policy {
	allow := map[string]bool{}
	for t := range s.Policy.Allow {
		if !base.NeverAdmits(t) {
			allow[t] = true
		}
	}
	return adjudicator.Policy{
		Posture: adjudicator.PostureFailClosed,
		Allow:   allow,
	}
}
