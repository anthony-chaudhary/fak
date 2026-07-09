package gateway

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

const gitPushRemedy = "push with the safe compiled verb: fak sync push (a trusted-binary non-force push the kernel admits), or preview first with git push --dry-run"

// The sanctioned-alternative sentence flows through ONE field (Detail["remedy"])
// and ONE renderer (remedyNote) for BOTH refusing rungs (#2749). The arg-predicate
// rung stamps its rule's fix as Meta["fix"]; the reversibility rung stamps its
// preview dry-run hint as Meta["dry_run_hint"]; renderVerdict folds EITHER onto the
// single remedy seam. Before the unification the arg rung rode Detail["fix"] while
// the reversibility hint was buried inside the confirm recipe — two Meta keys, two
// renderers, so a maintainer wiring a redirect for one rung never discovered the
// other. This is the witness that both now surface through the same field+renderer.
func TestRemedySeamUnifiesBothRungs(t *testing.T) {
	// Arg-predicate deny: its rule's fix rides Meta["fix"].
	argRule := renderVerdict(abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonPolicyBlock,
		By:     "monitor",
		Meta:   map[string]string{"fix": "use fak issue create"},
	}, nil)
	if argRule.Detail["remedy"] != "use fak issue create" {
		t.Fatalf("arg-rule sanctioned alternative not on the unified remedy field: %+v", argRule.Detail)
	}

	// Reversibility escalation: its preview dry-run hint rides Meta["dry_run_hint"].
	claim := `{"class":"outward-facing","preview":"outward-facing command: git push origin main","confirm_token":"fak-0011223344556677","dry_run_hint":"` + gitPushRemedy + `"}`
	rev := renderVerdict(abi.Verdict{
		Kind:    abi.VerdictRequireWitness,
		By:      "monitor/reversibility",
		Payload: abi.WitnessPayload{Claim: claim},
		Meta:    map[string]string{"dry_run_hint": gitPushRemedy},
	}, nil)
	if rev.Detail["remedy"] != gitPushRemedy {
		t.Fatalf("reversibility sanctioned alternative not on the unified remedy field: %+v", rev.Detail)
	}

	// The SAME renderer surfaces both — there is no rung-specific note for the
	// "here is the sanctioned alternative" sentence anymore.
	argAdj := ToolAdjudication{Tool: "Bash", Admitted: false, Verdict: argRule}
	revAdj := ToolAdjudication{Tool: "PowerShell", Admitted: false, Verdict: rev}
	if got := remedyNote(argAdj); got != "sanctioned alternative: use fak issue create" {
		t.Fatalf("remedyNote(arg-rule) = %q", got)
	}
	if got := remedyNote(revAdj); got != "sanctioned alternative: "+gitPushRemedy {
		t.Fatalf("remedyNote(reversibility) = %q", got)
	}

	// The reversibility recipe keeps ONLY the distinct preview+token recipe — the
	// sanctioned-alternative sentence no longer lives inside it (it rides the seam).
	recipe := reversibilityGateNote(revAdj)
	if !strings.Contains(recipe, "fak-0011223344556677") {
		t.Fatalf("recipe dropped its confirm token: %q", recipe)
	}
	if strings.Contains(recipe, gitPushRemedy) {
		t.Fatalf("recipe still embeds the alternative (it must ride the remedy seam): %q", recipe)
	}
}

// reversibilityRefusal builds a ToolAdjudication shaped exactly as the wire
// produces one for the reversibility rung: renderVerdict maps the adjudicator's
// RequireWitness to Kind=REQUIRE_WITNESS with an empty Reason, Disposition
// ESCALATE, the full preview envelope JSON in Detail["claim"], and the preview
// dry-run hint lifted onto the unified Detail["remedy"] seam (#2749).
func reversibilityRefusal() ToolAdjudication {
	claim := `{"class":"outward-facing","preview":"outward-facing command: git push origin main","confirm_token":"fak-0011223344556677","dry_run_hint":"` + gitPushRemedy + `"}`
	return ToolAdjudication{
		Tool:     "PowerShell",
		Admitted: false,
		Verdict: WireVerdict{
			Kind:        "REQUIRE_WITNESS",
			By:          "monitor/reversibility",
			Disposition: "ESCALATE",
			Detail:      map[string]string{"claim": claim, "remedy": gitPushRemedy},
		},
	}
}

// The in-band note must carry the whole confirm recipe — preview, token, and
// dry-run hint — because a content-channel-only client (Claude Code on the
// Anthropic wire) has no other way to learn how to complete the gate's
// two-phase confirm. Before this note, the refusal rendered as the dead-end
// "PowerShell (/ESCALATE)".
func TestAdjudicationNoteCarriesReversibilityConfirmRecipe(t *testing.T) {
	note := adjudicationNote([]ToolAdjudication{reversibilityRefusal()})
	for _, want := range []string{
		"PowerShell (REQUIRE_WITNESS/ESCALATE)",
		"outward-facing command: git push origin main",
		"re-propose it byte-identical",
		`"_fak_confirm":"fak-0011223344556677"`,
		"fak sync push",
		// The trailer must sanction the re-propose, not forbid it: the generic
		// "do not re-propose" trailer alongside the confirm recipe was a live
		// contradiction that wedged a fleet session (the agent obeyed the
		// prohibition, never echoed the token, and the push never happened).
		"A preview-confirm refusal is a pause, not a denial",
		// The trailer joins the note's terminal period without doubling it
		// (the old unconditional ". Do not..." rendered "first.. Do not").
		"git push --dry-run. A preview-confirm refusal",
		"This is per-tool feedback, not a session stop",
		"A session stop only comes from a declared stop policy",
	} {
		if !strings.Contains(note, want) {
			t.Fatalf("adjudicationNote missing %q:\n%s", want, note)
		}
	}
	if strings.Contains(note, "(/ESCALATE)") {
		t.Fatalf("empty-reason refusal still renders the malformed (/ESCALATE) form:\n%s", note)
	}
	if strings.Contains(note, "Do not re-propose a refused call unchanged") {
		t.Fatalf("confirm recipe still contradicted by the blanket do-not-re-propose trailer:\n%s", note)
	}
}

// The emphasis fix (#3306): when the gated call has a compiled sidestep (a `fak`
// verb that avoids the gate entirely), the in-band trailer must LEAD the agent to
// that verb and DEMOTE the _fak_confirm recipe to the raw-call fallback. Leading
// with the token dance is what wedged the f0e7ac0f fleet session for over an hour:
// the agent chased a (correctly) command-bound confirm token it misread as rotating
// while `fak sync push` — which worked first try — sat one demoted sentence away.
func TestReversibilityRefusalLeadsWithCompiledSidestep(t *testing.T) {
	note := adjudicationNote([]ToolAdjudication{reversibilityRefusal()})

	// The compiled sidestep survives and the trailer frames it as the primary path.
	if !strings.Contains(note, "fak sync push") {
		t.Fatalf("note dropped the compiled sidestep:\n%s", note)
	}
	if !strings.Contains(note, "sidesteps the gate entirely") {
		t.Fatalf("trailer does not lead with the sidestep as the resolution:\n%s", note)
	}

	// The confirm-token recipe is demoted to the raw-call fallback — present, but
	// gated behind "only if you specifically need the raw gated call" and AFTER the
	// sidestep lead, not before it.
	fallback := "Only if you specifically need the raw gated call"
	sidestepAt := strings.Index(note, "sidesteps the gate entirely")
	fallbackAt := strings.Index(note, fallback)
	if fallbackAt < 0 {
		t.Fatalf("confirm recipe not demoted to a fallback clause:\n%s", note)
	}
	if fallbackAt < sidestepAt {
		t.Fatalf("confirm recipe (idx %d) precedes the sidestep lead (idx %d) — emphasis not fixed:\n%s", fallbackAt, sidestepAt, note)
	}
	// The recipe token is still reachable for the raw-call path.
	if !strings.Contains(note, `"_fak_confirm":"fak-0011223344556677"`) {
		t.Fatalf("demoted recipe dropped its confirm token:\n%s", note)
	}
}

// A preview-confirm refusal with NO compiled sidestep (a preview-only affordance,
// e.g. `npm publish --dry-run`) keeps the recipe-first trailer: there is no verb to
// lead with, so the _fak_confirm re-propose remains the sanctioned next step.
func TestReversibilityRefusalWithoutSidestepKeepsRecipeLead(t *testing.T) {
	adj := reversibilityRefusal()
	// A dry-run-only remedy: no `fak ` compiled verb.
	adj.Verdict.Detail["remedy"] = "preview first with npm publish --dry-run"
	note := adjudicationNote([]ToolAdjudication{adj})
	if !strings.Contains(note, "the sanctioned next step is to re-propose that same call with only the _fak_confirm key added") {
		t.Fatalf("preview-only refusal lost the recipe-first trailer:\n%s", note)
	}
	if strings.Contains(note, "sidesteps the gate entirely") {
		t.Fatalf("preview-only refusal wrongly claimed a compiled sidestep:\n%s", note)
	}
}

// A turn mixing a preview-confirm refusal with a plain deny gets the
// confirm-aware trailer: the re-propose sanction for the tokened call, the
// prohibition scoped to every OTHER refused call.
func TestAdjudicationNoteTrailerScopesProhibitionWhenMixed(t *testing.T) {
	plainDeny := ToolAdjudication{
		Tool:     "Write",
		Admitted: false,
		Verdict:  WireVerdict{Kind: "DENY", Reason: "DEFAULT_DENY", Disposition: "TERMINAL"},
	}
	note := adjudicationNote([]ToolAdjudication{reversibilityRefusal(), plainDeny})
	for _, want := range []string{
		`"_fak_confirm":"fak-0011223344556677"`,
		"Write (DEFAULT_DENY/TERMINAL)",
		"A preview-confirm refusal is a pause, not a denial",
		"This is per-tool feedback, not a session stop",
		"Do not re-propose any other refused call unchanged",
	} {
		if !strings.Contains(note, want) {
			t.Fatalf("mixed-refusal note missing %q:\n%s", want, note)
		}
	}
}

// The deny-all summary path (every proposed call refused) must carry the same
// recipe — a deny-all turn is exactly where the agent is otherwise stuck.
func TestDenySummaryCarriesReversibilityConfirmRecipe(t *testing.T) {
	sum := denySummary([]ToolAdjudication{reversibilityRefusal()})
	for _, want := range []string{
		"REQUIRE_WITNESS",
		`"_fak_confirm":"fak-0011223344556677"`,
	} {
		if !strings.Contains(sum, want) {
			t.Fatalf("denySummary missing %q:\n%s", want, sum)
		}
	}
}

// A REQUIRE_WITNESS from any other gate (a plan-CFI approval, a registered
// restrictive kind) must NOT grow a confirm recipe: the recipe is only valid
// where the adjudicator's reversibility rung will actually verify the echoed
// token.
func TestReversibilityNoteScopedToReversibilityGate(t *testing.T) {
	adj := reversibilityRefusal()
	adj.Verdict.By = "monitor/plan-cfi"
	if got := reversibilityGateNote(adj); got != "" {
		t.Fatalf("non-reversibility REQUIRE_WITNESS grew a confirm recipe: %q", got)
	}

	adj = reversibilityRefusal()
	adj.Verdict.Detail = nil
	if got := reversibilityGateNote(adj); got != "" {
		t.Fatalf("missing claim payload still produced a recipe: %q", got)
	}

	adj = reversibilityRefusal()
	adj.Verdict.Detail = map[string]string{"claim": `{"class":"outward-facing"}`}
	if got := reversibilityGateNote(adj); got != "" {
		t.Fatalf("claim without a confirm token still produced a recipe: %q", got)
	}
}

// Refusals that carry a closed-vocabulary Reason keep their exact reason rendering
// while the generic trailer now states the control-plane boundary explicitly:
// a denied tool call is per-tool feedback, not a session stop.
func TestAdjudicationNotePlainDenySaysFeedbackNotSessionStop(t *testing.T) {
	note := adjudicationNote([]ToolAdjudication{{
		Tool:     "Write",
		Admitted: false,
		Verdict:  WireVerdict{Kind: "DENY", Reason: "DEFAULT_DENY", Disposition: "TERMINAL"},
	}})
	if !strings.Contains(note, "Write (DEFAULT_DENY/TERMINAL)") {
		t.Fatalf("plain deny rendering changed:\n%s", note)
	}
	for _, want := range []string{
		"This is per-tool feedback, not a session stop",
		"Do not re-propose a refused call unchanged",
		"fix the arguments/tool choice or choose an allowed alternative",
		"A session stop only comes from a declared stop policy",
	} {
		if !strings.Contains(note, want) {
			t.Fatalf("plain-deny note missing %q:\n%s", want, note)
		}
	}
	if strings.Contains(note, "session stopped") {
		t.Fatalf("plain deny implied a stopped session:\n%s", note)
	}
}
