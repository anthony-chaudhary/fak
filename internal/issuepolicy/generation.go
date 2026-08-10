package issuepolicy

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

// GenerationFit is an advisory grooming score. It checks whether generation
// labels match the issue body, proof, and time-horizon cues. It is intentionally
// not part of dispatchability: a label mismatch needs operator review, not a
// silent refusal of otherwise scoped work.
type GenerationFit struct {
	Stream        string   `json:"stream,omitempty"`
	LabelStream   string   `json:"label_stream,omitempty"`
	BodyStream    string   `json:"body_stream,omitempty"`
	Label         int      `json:"label"`
	Body          int      `json:"body"`
	Horizon       int      `json:"horizon"`
	Evidence      int      `json:"evidence"`
	Orthogonality int      `json:"orthogonality"`
	Total         int      `json:"total"`
	Flags         []string `json:"flags,omitempty"`
	NextAction    string   `json:"next_action,omitempty"`
}

func generationFit(c Candidate) GenerationFit {
	labelStream, labelFlags := generationStreamFromLabels(c.Labels)
	bodyStream := generationStreamFromCandidate(c)
	stream := strmatch.FirstTrimmed(labelStream, bodyStream)
	flags := append([]string(nil), labelFlags...)
	if labelStream != "" && bodyStream != "" && labelStream != bodyStream {
		flags = append(flags, "generation_body_mismatch")
	}
	if labelStream == "" && bodyStream != "" {
		flags = append(flags, "generation_label_missing")
	}

	text := generationCandidateText(c)
	lower := strings.ToLower(text)
	// Conflation flags fire only when the anti-pattern is ADVOCATED. Correct
	// grooming guidance names the same phrase to forbid it ("do not create a
	// branch per generation"), so a nearby prohibition cue must suppress the
	// flag — otherwise the cleanest issue trips it.
	if advocates(lower, "branch per generation", "feature branch per generation", "generation branch") {
		flags = append(flags, "generation_branch_conflation")
	}
	if advocates(lower, "future is lower priority", "future as lower priority", "future lower priority") {
		flags = append(flags, "generation_priority_conflation")
	}
	if advocates(lower, "gen/next flag", "gen/future flag", "generation label enables", "generation label decides exposure") {
		flags = append(flags, "generation_runtime_gate_conflation")
	}

	g := GenerationFit{
		Stream:      stream,
		LabelStream: labelStream,
		BodyStream:  bodyStream,
		Flags:       flags,
	}
	if labelStream != "" && !containsString(g.Flags, "generation_label_multiple") && !containsString(g.Flags, "generation_parent_label_missing") {
		g.Label = 20
	}
	if bodyStream != "" && (labelStream == "" || bodyStream == labelStream) {
		g.Body = 20
	}
	if stream != "" && generationHorizonMatches(stream, lower) {
		g.Horizon = 20
	} else if stream != "" {
		g.Flags = append(g.Flags, "generation_horizon_cue_missing")
	}
	ev := generationEvidence(c, lower)
	if ev.complete() {
		g.Evidence = 20
	} else if stream != "" {
		// Name WHICH evidence kind is missing, not a single opaque flag. The
		// acceptance criteria require promotion, demotion/retirement, and an
		// invalidating assumption as three distinct nameables; an operator (or a
		// future agent) reading the readout must see exactly which one is absent to
		// continue without rereading the generation epic's evidence rubric.
		g.Flags = append(g.Flags, ev.missingFlags()...)
	}
	if generationOrthogonalityNamed(lower) {
		g.Orthogonality = 20
	} else if stream != "" {
		g.Flags = append(g.Flags, "generation_orthogonality_missing")
	}
	g.Total = g.Label + g.Body + g.Horizon + g.Evidence + g.Orthogonality
	if len(g.Flags) > 0 {
		g.Flags = compact(g.Flags)
		g.NextAction = "review generation label, body horizon, promotion evidence, demotion evidence, and priority/trunk/runtime-gate separation"
	}
	return g
}

func issueDraftGeneration(d IssueDraft, generationSection string) string {
	if s := generationStreamFromText(generationSection); s != "" {
		return s
	}
	if s := generationStreamFromText(d.Title); s != "" {
		return s
	}
	return ""
}

func generationStreamFromCandidate(c Candidate) string {
	if s := generationStreamFromText(c.Generation); s != "" {
		return s
	}
	return generationStreamFromText(c.Title)
}

func generationStreamFromLabels(labels []string) (string, []string) {
	var streams []string
	hasGenerationLabel := false
	for _, label := range labels {
		label = strings.ToLower(strings.TrimSpace(label))
		if label == "generation" {
			hasGenerationLabel = true
		}
		if isGenerationStream(label) {
			streams = append(streams, label)
		}
	}
	streams = compact(streams)
	var flags []string
	if len(streams) > 1 {
		flags = append(flags, "generation_label_multiple")
	}
	if len(streams) == 1 && !hasGenerationLabel {
		flags = append(flags, "generation_parent_label_missing")
	}
	if len(streams) == 0 && hasGenerationLabel {
		flags = append(flags, "generation_stream_label_missing")
	}
	if len(streams) != 1 {
		return "", flags
	}
	return streams[0], flags
}

func generationStreamFromText(text string) string {
	text = strings.ToLower(text)
	checks := []struct {
		stream  string
		needles []string
	}{
		{"gen/second-next", []string{"gen/second-next", "gen=second-next", "generation(second-next)", "generation: second-next", "generation: gen/second-next"}},
		{"gen/future", []string{"gen/future", "gen=future", "generation(future)", "generation: future", "generation: gen/future"}},
		{"gen/next", []string{"gen/next", "gen=next", "generation(next)", "generation: next", "generation: gen/next"}},
		{"gen/now", []string{"gen/now", "gen=now", "generation(now)", "generation: now", "generation: gen/now"}},
	}
	for _, check := range checks {
		if hasAny(text, check.needles...) {
			return check.stream
		}
	}
	return ""
}

func isGenerationStream(label string) bool {
	switch label {
	case "gen/now", "gen/next", "gen/second-next", "gen/future":
		return true
	default:
		return false
	}
}

func generationCandidateText(c Candidate) string {
	parts := []string{
		c.Generation, c.Title, c.ParentRef, c.CurrentState, c.WhyNow, c.WorkingSpine,
		c.PriorityContext, c.WorkUnit, c.Trigger, c.BatchPolicy, c.InScope, c.OutOfScope,
		c.DoneCondition, c.Witness, c.AcceptanceGate, c.ClosureBinding,
	}
	parts = append(parts, c.Assumptions...)
	parts = append(parts, c.ConfusionRisks...)
	parts = append(parts, c.Coordination...)
	parts = append(parts, c.BoundaryNotes...)
	parts = append(parts, c.Labels...)
	return strings.Join(parts, "\n")
}

func generationHorizonMatches(stream, text string) bool {
	switch stream {
	case "gen/now":
		return hasAny(text, "current product", "current path", "immediate", "now", "today", "default path", "direct witness")
	case "gen/next":
		return hasAny(text, "next gen", "next-generation", "near-term", "foundation", "gate", "handoff", "dogfood", "operator visibility", "agent-runnable", "runnable soon")
	case "gen/second-next":
		return hasAny(text, "second-next", "architecture", "compatibility", "simulation", "dependency", "option", "adapter")
	case "gen/future":
		return hasAny(text, "future", "research", "long-horizon", "market", "standards", "narrative", "option value")
	default:
		return false
	}
}

// generationEvidenceParts records, per evidence kind, whether the candidate names
// it. The acceptance criteria treat promotion, demotion/retirement, and an
// invalidating assumption as three separately-required nameables (plus a witness),
// so the fit score reports each independently instead of collapsing them into one
// boolean — that is what lets the readout name the specific gap.
type generationEvidenceParts struct {
	promotion    bool
	demotion     bool
	invalidating bool
	witness      bool
}

func generationEvidence(c Candidate, text string) generationEvidenceParts {
	return generationEvidenceParts{
		promotion:    hasAny(text, "promotion", "promote", "readiness", "dogfood", "default-on", "move toward now"),
		demotion:     hasAny(text, "demotion", "demote", "retirement", "retire", "park", "parking"),
		invalidating: hasAny(text, "invalidating assumption", "assumption could fail", "if this assumption fails", "recheck"),
		witness:      strings.TrimSpace(c.Witness) != "" || hasAny(text, "witness", "captured command", "focused test", "readout"),
	}
}

// complete reports whether all four evidence kinds are named — the unchanged rule
// for awarding the evidence axis its full points.
func (e generationEvidenceParts) complete() bool {
	return e.promotion && e.demotion && e.invalidating && e.witness
}

// missingFlags names each absent evidence kind as its own flag. The promotion flag
// keeps its original spelling (generation_promotion_evidence_missing) so an existing
// readout string never silently disappears; the demotion/invalidating/witness flags
// are the added granularity criterion 3 and 4 need.
func (e generationEvidenceParts) missingFlags() []string {
	var flags []string
	if !e.promotion {
		flags = append(flags, "generation_promotion_evidence_missing")
	}
	if !e.demotion {
		flags = append(flags, "generation_demotion_evidence_missing")
	}
	if !e.invalidating {
		flags = append(flags, "generation_invalidating_assumption_missing")
	}
	if !e.witness {
		flags = append(flags, "generation_evidence_witness_missing")
	}
	return flags
}

func generationOrthogonalityNamed(text string) bool {
	hasPriority := strings.Contains(text, "priority")
	hasTrunk := hasAny(text, "shared trunk", "trunk", "main")
	hasRuntimeGate := hasAny(text, "runtime feature gate", "feature gate", "runtime gate", "exposure gate", "default-off", "default on")
	return hasPriority && hasTrunk && hasRuntimeGate
}
