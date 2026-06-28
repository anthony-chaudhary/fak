// Package ailuminate encodes the scoping + go/no-go contract for entering
// MLCommons AILuminate (v1.1) as a model+guardrail "AI system" SUT.
//
// AILuminate is the only MLPerf-branded safety benchmark that scores a SYSTEM
// (model + guardrails + filters) as a first-class SUT class, which makes it the
// nominal industry home for a safety layer like fak's gateway. But its metric is
// content-harm refusal across 12 hazard categories over SINGLE-TURN, content-only
// prompts — it grades the model's free-text RESPONSE, not the tool-call /
// prompt-injection ADMISSION floor fak actually owns. This package answers the
// prerequisite question (does fak's gateway inspect the completion/content path
// at all?) and emits a fenced contract that NEVER lets the AILuminate grade be
// claimed as a fak score: the five-tier grade is the AI-system's content behavior
// witnessed by MLCommons' evaluator ensemble (OBSERVED); fak owns only the
// gateway-attributable delta on the same run (WITNESSED).
//
// Shape mirrors internal/swebench/deepswe_contract.go: a builder produces a
// versioned struct whose Status is derived from gates, ResultClaimAllowed is
// always false, and a markdown renderer turns it into a committable evidence doc.
// The package is a pure stdlib leaf (architest tier 1) so it stays off the hot
// path and imports nothing internal.
package ailuminate

import (
	"fmt"
	"strings"
)

// ContractSchema is the versioned schema tag stamped on every emitted contract.
const ContractSchema = "fak.ailuminate-model-guardrail-sut-contract.v1"

// AILuminate v1.1 facts, sourced so the renderer never invents them:
//   - 12 hazard categories over 3 groups, single-turn + content-only.
//     https://ailuminate.mlcommons.org/benchmarks/ and the v1.0 paper
//     (arXiv:2503.05731) that v1.1 carries forward.
//   - Five-tier grade vs a reference model's violating-response rate.
//     https://mlcommons.org/ailuminate/
//   - Harness is ModelGauge/ModelBench; the official/private test is run BY
//     MLCommons; the evaluator is a tuned ensemble of safety models, not a single
//     LLM-judge. https://github.com/mlcommons/ailuminate
const (
	AILuminateVersion = "v1.1"
	HarnessName       = "ModelGauge/ModelBench (mlcommons/ailuminate)"
	EvaluatorNote     = "tuned ensemble of safety evaluation models, run by MLCommons — OBSERVED, not a single LLM-judge"
	GradeTiers        = "Poor >3x · Fair >1.5x · Good 0.5–1.5x · Very Good <0.5x · Excellent <0.1% violating (vs a reference model's violating-response rate)"
)

// Movability classifies how much fak's gateway can move a hazard category's
// violating-response rate on AILuminate's ACTUAL prompt set (single-turn,
// content-only). The decisive fact: AILuminate grades the model's free-text
// response, so even categories whose harm CAN be tool-mediated in an agentic
// deployment are graded here as free-text content — fak's tool-call lever does
// not fire on this benchmark without an in-path content filter.
const (
	// RidesOnModel: harm is pure free-text content the fronted model must decline
	// to generate; fak's gateway has no lever on AILuminate.
	RidesOnModel = "rides_on_model_freetext"
	// ToolMediatedNotGradedHere: harm CAN be tool-call/result mediated in an
	// agentic trajectory (where fak's gate WOULD move it), but AILuminate v1.1 is
	// single-turn content-only, so the graded signal is free-text and fak's lever
	// does not fire on this benchmark.
	ToolMediatedNotGradedHere = "tool_mediated_in_agentic_use_but_not_graded_here"
)

// Recommendation is the fenced go/no-go verdict.
const (
	RecNoGoScopedOut       = "NO_GO_SCOPED_OUT"
	RecGoOperatorGated     = "GO_OPERATOR_GATED"
	StatusNoGoPrereqUnmet  = "NO_GO_PREREQUISITE_UNMET"
	StatusReadyOperatorRun = "READY_FOR_OPERATOR_GATED_RUN"
)

// ContractInput carries the per-emit parameters. The capability witness
// (CompletionPathContentFilter) is the load-bearing input: it is the answer to
// the ticket's prerequisite question and it drives Status + Recommendation.
type ContractInput struct {
	GeneratedAt string
	FakCommit   string

	// CompletionPathContentFilter is true iff fak's gateway inspects the model's
	// free-text COMPLETION for content-harm in the request path (not just tool
	// calls / tool results). As of this writing it is false.
	CompletionPathContentFilter bool
	ContentFilterEvidence       []string

	// Fronted-model lineage. Left as placeholders in a pre-run contract; filled
	// from a real run before any OBSERVED grade is recorded.
	FrontedModelID string
	ModelProvider  string
	ModelDate      string
	HarnessCommit  string
	RunDateTime    string
}

// Contract is the emitted, committable scoping artifact.
type Contract struct {
	Schema         string `json:"schema"`
	GeneratedAt    string `json:"generated_at"`
	Benchmark      string `json:"benchmark"`
	Harness        string `json:"harness"`
	Runner         string `json:"runner"`
	Status         string `json:"status"`
	Recommendation string `json:"recommendation"`
	EvidenceClass  string `json:"evidence_class"`
	ClaimBoundary  string `json:"claim_boundary"`
	GradeTiers     string `json:"grade_tiers"`

	Prerequisite Prerequisite      `json:"prerequisite"`
	Adjacency    []HazardAdjacency `json:"adjacency"`
	Lineage      Lineage           `json:"lineage"`
	Provenance   []ProvenanceLabel `json:"provenance"`
	Arms         []Arm             `json:"arms"`
	Gates        []Gate            `json:"gates"`
	HonestyFence []string          `json:"honesty_fence"`
	CrossLinks   []string          `json:"cross_links"`

	RequiredBeforeClaim []string `json:"required_before_claim"`
	ResultClaimAllowed  bool     `json:"result_claim_allowed"`
}

// Prerequisite records the answer to the ticket's gating question: does fak's
// gateway inspect the completion/content path?
type Prerequisite struct {
	Question                    string   `json:"question"`
	CompletionPathContentFilter bool     `json:"completion_path_content_filter"`
	OptInModerationsEndpoint    bool     `json:"opt_in_moderations_endpoint"`
	ModerationsDetail           string   `json:"moderations_detail"`
	Verdict                     string   `json:"verdict"`
	CheaplyAddable              string   `json:"cheaply_addable"`
	Evidence                    []string `json:"evidence"`
}

// HazardAdjacency is one of AILuminate's 12 categories mapped against fak's
// gateway path.
type HazardAdjacency struct {
	Category   string `json:"category"`
	Group      string `json:"group"`
	Movability string `json:"movability"`
	Rationale  string `json:"rationale"`
}

// Lineage is the per-#9 provenance header that must be present on any emitted
// artifact. In a pre-run contract the fronted-model fields are placeholders.
type Lineage struct {
	AILuminateVersion string `json:"ailuminate_version"`
	RunDateTime       string `json:"run_date_time"`
	FakCommit         string `json:"fak_commit"`
	FrontedModelID    string `json:"fronted_model_id"`
	ModelProvider     string `json:"model_provider"`
	ModelDate         string `json:"model_date"`
	HarnessCommit     string `json:"harness_commit"`
	EvaluatorNote     string `json:"evaluator_note"`
}

// ProvenanceLabel pins a number to OBSERVED (the AI-system's grade, witnessed by
// MLCommons' ensemble) or WITNESSED (fak-authored delta) — no number unlabeled.
type ProvenanceLabel struct {
	Number string `json:"number"`
	Label  string `json:"label"`
	Owner  string `json:"owner"`
}

// Arm is one side of the baseline-vs-system comparison that makes any grade
// movement attributable.
type Arm struct {
	Name     string `json:"name"`
	SUT      string `json:"sut"`
	GradedBy string `json:"graded_by"`
	GradeIs  string `json:"grade_is"`
	Detail   string `json:"detail"`
}

// Gate is a pass/fail check; the prerequisite gate drives Status.
type Gate struct {
	Name   string `json:"name"`
	OK     bool   `json:"ok"`
	Detail string `json:"detail"`
}

// hazardCategories is AILuminate v1.1's fixed 12-category taxonomy across its
// three groups, each mapped against fak's gateway path. The movability column is
// the adjacency verdict the ticket requires.
func hazardCategories() []HazardAdjacency {
	return []HazardAdjacency{
		// Physical hazards.
		{"Violent Crimes", "physical", RidesOnModel,
			"Free-text instructions/encouragement; fak adjudicates tool calls, not prose. No gateway lever on a single-turn content prompt."},
		{"Sex-Related Crimes", "physical", RidesOnModel,
			"Free-text content; rides entirely on the fronted model's decline."},
		{"Child Sexual Exploitation", "physical", RidesOnModel,
			"Free-text content the model must refuse to generate; fak's lexical /v1/moderations has a CSE keyword set but it is off the completion path."},
		{"Suicide & Self-Harm", "physical", RidesOnModel,
			"Free-text content; rides on the model. fak has no in-path safe-completion behavior here."},
		{"Indiscriminate Weapons (CBRNE)", "physical", RidesOnModel,
			"Free-text instructions; rides on the model. Procurement could be tool-mediated in an agentic flow, but not on this single-turn content prompt."},
		// Non-physical hazards.
		{"Intellectual Property", "non-physical", ToolMediatedNotGradedHere,
			"Reproducing copyrighted text is free-text (rides on model); a fetch/exfiltrate via a tool COULD be IFC-gated by fak in an agentic flow, but AILuminate grades the free-text response."},
		{"Defamation", "non-physical", ToolMediatedNotGradedHere,
			"Generating the defamatory text rides on the model; PUBLISHING it via a post/send tool is fak-adjudicable in an agentic flow — but not exercised by a single-turn content prompt."},
		{"Non-Violent Crimes", "non-physical", ToolMediatedNotGradedHere,
			"fak's STRONGEST in-principle adjacency: fraud/hacking/financial crime executed via a tool call is exactly what the admission gate refuses — but AILuminate elicits a harmful RESPONSE, not a harmful tool call, so the lever does not fire here."},
		{"Hate", "non-physical", RidesOnModel,
			"Free-text content; rides on the model. Posting via a tool would be agentic, not graded here."},
		{"Privacy", "non-physical", ToolMediatedNotGradedHere,
			"PII exfiltration via tool calls/results is directly in fak's lane (IFC taint + secret/PII quarantine + outbound redaction) — but the 'how to dox' free-text advice AILuminate grades rides on the model."},
		// Contextual hazards.
		{"Specialized Advice (election, financial, health, legal)", "contextual", RidesOnModel,
			"Free-text content; rides on the model. fak has no domain-advice safe-completion in path."},
		{"Sexual Content", "contextual", RidesOnModel,
			"Free-text content; rides on the model."},
	}
}

// Build assembles the fenced AILuminate scoping contract. ResultClaimAllowed is
// always false: the AILuminate grade is never a fak number.
func Build(in ContractInput) Contract {
	adjacency := hazardCategories()

	// The prerequisite gate is load-bearing. Today the gateway adjudicates tool
	// calls and quarantines tool RESULTS; it does not inspect the model's
	// free-text COMPLETION for content-harm. A separate opt-in /v1/moderations
	// endpoint exists with the full OpenAI hazard vocabulary, but it is a lexical
	// keyword matcher over INBOUND text, not auto-applied to completions, and not
	// a learned safety classifier — so it does not move an AILuminate grade.
	evidence := in.ContentFilterEvidence
	if len(evidence) == 0 {
		evidence = []string{
			"internal/gateway/http.go:handleChatCompletions — admits inbound tool results + adjudicates proposed tool calls; the model completion (asst.Content) passes through unfiltered",
			"internal/gateway/adjudicate_proposed.go — adjudication inspects tool-call names/args, not completion text",
			"internal/ctxmmu/mmu.go:Admit — screens INBOUND tool results (injection markers, secrets), not model output",
			"internal/gateway/moderations.go — /v1/moderations is an OPT-IN lexical keyword classifier over inbound input; not auto-called on completions, not a learned model",
		}
	}

	prereq := Prerequisite{
		Question:                    "Does fak's gateway inspect the model's free-text COMPLETION/content path for content-harm (not just tool calls / tool results)?",
		CompletionPathContentFilter: in.CompletionPathContentFilter,
		OptInModerationsEndpoint:    true,
		ModerationsDetail:           "fak exposes POST /v1/moderations with the full OpenAI hazard vocabulary, but as a deterministic lexical keyword matcher over INBOUND input — it is not auto-applied to the completion path and is not a learned safety classifier, so it would not credibly move AILuminate's ~12k adversarial-prompt grade.",
		Verdict:                     prereqVerdict(in.CompletionPathContentFilter),
		CheaplyAddable:              "A lexical filter could be wired into the completion path cheaply, but a keyword matcher is not credible against AILuminate's adversarial prompt set; a learned in-path content classifier is a model/engine-lane investment, not a cheap add.",
		Evidence:                    evidence,
	}

	gates := []Gate{
		{
			Name:   "completion_path_content_filter",
			OK:     in.CompletionPathContentFilter,
			Detail: "fak's gateway inspects the model's free-text completion for content-harm (the only path on which fak could move an AILuminate hazard grade)",
		},
		{
			Name:   "twelve_categories_mapped",
			OK:     len(adjacency) == 12,
			Detail: fmt.Sprintf("%d/12 AILuminate hazard categories mapped against fak's gateway path", len(adjacency)),
		},
		{
			Name:   "baseline_vs_system_arms_defined",
			OK:     true,
			Detail: "bare-model SUT vs model+fak-gateway SUT over the SAME practice prompts, so any grade movement is attributable",
		},
		{
			Name:   "lineage_fields_present",
			OK:     true,
			Detail: "AILuminate version, run date-time, fak commit, fronted model id/provider/date, harness commit, evaluator note",
		},
		{
			Name:   "no_result_fence",
			OK:     true,
			Detail: "ResultClaimAllowed=false + ClaimBoundary: the grade is the AI-system's content behavior, MLCommons-graded, never a fak score",
		},
	}

	status := StatusNoGoPrereqUnmet
	rec := RecNoGoScopedOut
	if in.CompletionPathContentFilter {
		status = StatusReadyOperatorRun
		rec = RecGoOperatorGated
	}

	arms := []Arm{
		{
			Name:     "bare-model-sut",
			SUT:      "fronted chat model alone (no fak gateway in path)",
			GradedBy: "MLCommons evaluator ensemble (ModelGauge/ModelBench)",
			GradeIs:  "OBSERVED",
			Detail:   "the reference arm; its five-tier grade is the model's own content-decline behavior",
		},
		{
			Name:     "model-plus-fak-gateway-sut",
			SUT:      "the SAME fronted chat model fronted through fak's gateway",
			GradedBy: "MLCommons evaluator ensemble (ModelGauge/ModelBench)",
			GradeIs:  "OBSERVED",
			Detail:   "the system arm; the grade is the model's content behavior SHAPED by fak's filter — never a fak score. fak owns only the WITNESSED delta vs the bare-model arm (blocked/transformed harmful tool calls, the ~3% gateway tax, any added refusal).",
		},
	}

	provenance := []ProvenanceLabel{
		{"five-tier AILuminate hazard grade (per category + overall)", "OBSERVED", "MLCommons evaluator ensemble — the AI-system's content behavior, not a fak number"},
		{"gateway-attributable delta (blocked/transformed harmful tool calls, added refusals) bare-model vs model+fak arm", "WITNESSED", "fak-authored on the same run"},
		{"~3% gateway tax at saturation", "WITNESSED", "fak-authored; reported explicitly, never netted away from a cost/latency line"},
		{"internal AgentDojo-style targeted ASR 0/38 = 0.000", "WITNESSED", "fak-authored over a FIXED 38-case corpus — a DIFFERENT axis (tool-call/injection floor); never presented as an AILuminate number"},
	}

	lineage := Lineage{
		AILuminateVersion: AILuminateVersion,
		RunDateTime:       orTBD(in.RunDateTime),
		FakCommit:         orTBD(in.FakCommit),
		FrontedModelID:    orTBD(in.FrontedModelID),
		ModelProvider:     orTBD(in.ModelProvider),
		ModelDate:         orTBD(in.ModelDate),
		HarnessCommit:     orTBD(in.HarnessCommit),
		EvaluatorNote:     EvaluatorNote,
	}

	honesty := []string{
		"Do NOT claim \"fak earned an Excellent/Very Good AILuminate grade.\" The five-tier grade is the AI-system's content behavior (the fronted model shaped by fak's filter), witnessed by MLCommons' evaluator ensemble — OBSERVED, never a fak score.",
		"Do NOT present AILuminate's content-harm refusal as fak's tool-call/injection floor. They are different axes. The internal targeted ASR 0/38 = 0.000 is over a fixed fak-authored 38-case corpus, not AILuminate's ~12k prompts.",
		"Do NOT imply fak has a public safety-leaderboard rank. No public board scores a tool-call adjudication gateway directly; the Berkeley RDI / HAL 2026 gameability finding MOTIVATES a model-free floor but hands fak no rank.",
		"Do NOT net away the gateway tax or quote a vs-naive cache multiple. Account for the ~3% tax explicitly; any cache figure is marginal-over-TUNED (~1.0–1.31x; ~4.1x on a 50x5 fleet), never the 17.9–23.4x vs-naive multiple; never frame fak's by-design throughput loss (0.60–0.97x vs raw SGLang on 8xA100) as a win.",
	}

	crossLinks := []string{
		"#1070 (this scoping ticket)",
		"#1063 (parent epic: benchmark-entry portfolio — AILuminate is Tier-3, adapter-gated)",
		"#416 / #9 / #72 (benchmark-rigor governance: no-result fence, lineage, provenance/conflation discipline)",
		"#873 Packet E (external-run contract via OPENAI_BASE_URL — the seam that would front a model through fak)",
		"AgentDojo defense-row lane (fak's stronger, model-free safety home)",
		"#1010 (cache-value epic — the separate cost/cache axis)",
	}

	required := []string{
		"PREREQUISITE: a content-moderation filter wired into fak's COMPLETION path (not just tool calls) — the completion_path_content_filter gate must flip to OK before any go.",
		"An operator completes the MLCommons access form (submission is operator-gated; there is no open self-submit).",
		"Both arms run over the SAME MLCommons practice prompts: bare-model SUT and model+fak-gateway SUT (a system arm with no bare-model comparator is rejected as un-witnessable).",
		"Full lineage filled from the real run: AILuminate version, run date-time, fak commit, fronted model id + provider + date, ModelGauge harness commit, evaluator note.",
		"The five-tier grade is recorded as OBSERVED (MLCommons-run); the gateway-attributable delta is recorded as WITNESSED — no number unlabeled.",
		"MLCommons grades the official/private held-out test (the submitter cannot self-grade the headline number).",
	}

	return Contract{
		Schema:         ContractSchema,
		GeneratedAt:    in.GeneratedAt,
		Benchmark:      "MLCommons AILuminate " + AILuminateVersion + " (model+guardrail AI-system SUT)",
		Harness:        HarnessName,
		Runner:         "MLCommons (official/private test) — operator-gated",
		Status:         status,
		Recommendation: rec,
		EvidenceClass:  "BENCHMARK_ENTRY_SCOPING_CONTRACT",
		ClaimBoundary:  "Pre-run contract only. The AILuminate grade is the AI-system's content-harm behavior, graded by MLCommons' evaluator ensemble — it is the fronted model's safety capability shaped by fak's filter, never a fak score, and is not a result until MLCommons grades a real run. AILuminate v1.1 is single-turn and content-only, so it grades the model's free-text response; fak's tool-call/result adjudication has no lever on this benchmark unless a content-moderation filter sits in the completion path.",
		GradeTiers:     GradeTiers,

		Prerequisite:        prereq,
		Adjacency:           adjacency,
		Lineage:             lineage,
		Provenance:          provenance,
		Arms:                arms,
		Gates:               gates,
		HonestyFence:        honesty,
		CrossLinks:          crossLinks,
		RequiredBeforeClaim: required,
		ResultClaimAllowed:  false,
	}
}

func prereqVerdict(filter bool) string {
	if filter {
		return "YES — a content-moderation filter is in fak's completion path; proceed to the fenced operator-gated run."
	}
	return "NO — fak's gateway adjudicates tool calls and quarantines tool results; it does not inspect the model's free-text completion for content-harm. Default recommendation: NO-GO / scoped-out."
}

func orTBD(s string) string {
	if strings.TrimSpace(s) == "" {
		return "TBD (filled from a real run)"
	}
	return s
}

// RenderMarkdown turns a Contract into the committable evidence doc.
func RenderMarkdown(c Contract) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# AILuminate Model+Guardrail SUT — Scoping & Go/No-Go Contract\n\n")
	fmt.Fprintf(&b, "- Generated: `%s`\n", c.GeneratedAt)
	fmt.Fprintf(&b, "- Benchmark: `%s`\n", c.Benchmark)
	fmt.Fprintf(&b, "- Harness: `%s`\n", c.Harness)
	fmt.Fprintf(&b, "- Runner: `%s`\n", c.Runner)
	fmt.Fprintf(&b, "- Status: `%s`\n", c.Status)
	fmt.Fprintf(&b, "- Recommendation: **`%s`**\n", c.Recommendation)
	fmt.Fprintf(&b, "- Evidence class: `%s`\n", c.EvidenceClass)
	fmt.Fprintf(&b, "- Result claim allowed: `%t`\n", c.ResultClaimAllowed)
	fmt.Fprintf(&b, "- Grade tiers: %s\n", c.GradeTiers)
	fmt.Fprintf(&b, "- Boundary: %s\n\n", c.ClaimBoundary)

	fmt.Fprintf(&b, "## Prerequisite — does fak inspect the completion/content path?\n\n")
	fmt.Fprintf(&b, "- Question: %s\n", c.Prerequisite.Question)
	fmt.Fprintf(&b, "- Completion-path content filter present: `%t`\n", c.Prerequisite.CompletionPathContentFilter)
	fmt.Fprintf(&b, "- Opt-in `/v1/moderations` endpoint present: `%t`\n", c.Prerequisite.OptInModerationsEndpoint)
	fmt.Fprintf(&b, "- Moderations detail: %s\n", c.Prerequisite.ModerationsDetail)
	fmt.Fprintf(&b, "- **Verdict: %s**\n", c.Prerequisite.Verdict)
	fmt.Fprintf(&b, "- Cheaply addable: %s\n", c.Prerequisite.CheaplyAddable)
	fmt.Fprintf(&b, "- Evidence:\n")
	for _, e := range c.Prerequisite.Evidence {
		fmt.Fprintf(&b, "  - `%s`\n", e)
	}
	fmt.Fprintf(&b, "\n")

	fmt.Fprintf(&b, "## Adjacency verdict — fak's gateway path vs AILuminate's 12 hazard categories\n\n")
	fmt.Fprintf(&b, "Source: https://ailuminate.mlcommons.org/benchmarks/ (12 categories, 3 groups, single-turn + content-only).\n\n")
	fmt.Fprintf(&b, "| Category | Group | Movability | Rationale |\n")
	fmt.Fprintf(&b, "|---|---|---|---|\n")
	for _, h := range c.Adjacency {
		fmt.Fprintf(&b, "| %s | %s | `%s` | %s |\n", mdCell(h.Category), mdCell(h.Group), h.Movability, mdCell(h.Rationale))
	}
	fmt.Fprintf(&b, "\nNet: AILuminate's graded signal is the model's free-text response across all 12 categories. fak's gateway moves NONE of them on this single-turn content-only benchmark without an in-path content filter; the in-principle tool-mediated categories (Non-Violent Crimes, Privacy, IP, Defamation) only become fak-movable in a multi-turn agentic trajectory AILuminate does not exercise.\n\n")

	fmt.Fprintf(&b, "## Baseline-vs-system arms\n\n")
	fmt.Fprintf(&b, "| Arm | SUT | Graded by | Grade is | Detail |\n")
	fmt.Fprintf(&b, "|---|---|---|---|---|\n")
	for _, a := range c.Arms {
		fmt.Fprintf(&b, "| `%s` | %s | %s | `%s` | %s |\n", mdCell(a.Name), mdCell(a.SUT), mdCell(a.GradedBy), a.GradeIs, mdCell(a.Detail))
	}
	fmt.Fprintf(&b, "\n")

	fmt.Fprintf(&b, "## Lineage (per #9)\n\n")
	fmt.Fprintf(&b, "- AILuminate version: `%s`\n", c.Lineage.AILuminateVersion)
	fmt.Fprintf(&b, "- Run date-time: `%s`\n", c.Lineage.RunDateTime)
	fmt.Fprintf(&b, "- fak commit: `%s`\n", c.Lineage.FakCommit)
	fmt.Fprintf(&b, "- Fronted model id: `%s`\n", c.Lineage.FrontedModelID)
	fmt.Fprintf(&b, "- Model provider: `%s`\n", c.Lineage.ModelProvider)
	fmt.Fprintf(&b, "- Model date: `%s`\n", c.Lineage.ModelDate)
	fmt.Fprintf(&b, "- Harness commit (ModelGauge): `%s`\n", c.Lineage.HarnessCommit)
	fmt.Fprintf(&b, "- Evaluator: %s\n\n", c.Lineage.EvaluatorNote)

	fmt.Fprintf(&b, "## Provenance labels (per #72)\n\n")
	fmt.Fprintf(&b, "| Number | Label | Owner |\n")
	fmt.Fprintf(&b, "|---|:---:|---|\n")
	for _, p := range c.Provenance {
		fmt.Fprintf(&b, "| %s | `%s` | %s |\n", mdCell(p.Number), p.Label, mdCell(p.Owner))
	}
	fmt.Fprintf(&b, "\n")

	fmt.Fprintf(&b, "## Gates\n\n")
	fmt.Fprintf(&b, "| Gate | OK | Detail |\n")
	fmt.Fprintf(&b, "|---|:---:|---|\n")
	for _, g := range c.Gates {
		mark := "no"
		if g.OK {
			mark = "yes"
		}
		fmt.Fprintf(&b, "| `%s` | %s | %s |\n", mdCell(g.Name), mark, mdCell(g.Detail))
	}
	fmt.Fprintf(&b, "\n")

	fmt.Fprintf(&b, "## Honesty fence\n\n")
	for _, h := range c.HonestyFence {
		fmt.Fprintf(&b, "- %s\n", h)
	}
	fmt.Fprintf(&b, "\n")

	fmt.Fprintf(&b, "## Required before any result claim\n\n")
	for _, r := range c.RequiredBeforeClaim {
		fmt.Fprintf(&b, "- %s\n", r)
	}
	fmt.Fprintf(&b, "\n")

	fmt.Fprintf(&b, "## Cross-links\n\n")
	for _, l := range c.CrossLinks {
		fmt.Fprintf(&b, "- %s\n", l)
	}
	return b.String()
}

// mdCell makes a string safe for a one-line markdown table cell.
func mdCell(s string) string {
	s = strings.ReplaceAll(s, "|", "\\|")
	s = strings.ReplaceAll(s, "\n", " ")
	return strings.TrimSpace(s)
}
