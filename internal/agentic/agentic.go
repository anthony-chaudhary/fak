// Package agentic compiles broad objective text into a deterministic, bounded
// work plan. It performs no network access, filesystem access, issue creation,
// or worker launch.
package agentic

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

const (
	Schema            = "fak-agentic-work/1"
	LearningSchema    = "fak-agentic-learning/1"
	DomainDevelopment = "development"
	DomainNativeModel = "native_model"
	DomainMixed       = "mixed"
	ultracodeModel    = "gpt-5.6-sol"
	maxObjectiveBytes = 16 * 1024
)

type Plan struct {
	Schema      string          `json:"schema"`
	ObjectiveID string          `json:"objective_id"`
	Objective   string          `json:"objective"`
	Mode        Mode            `json:"mode"`
	Inference   Inference       `json:"inference"`
	Cohorts     []Cohort        `json:"cohorts"`
	Bounds      Bounds          `json:"bounds"`
	Stages      []Stage         `json:"stages"`
	WorkUnits   []WorkUnit      `json:"work_units"`
	Handoff     Handoff         `json:"handoff"`
	Native      *NativeContract `json:"native_contract,omitempty"`
	Learning    LearningRecord  `json:"learning_record"`
}

type Mode struct {
	ReadOnly bool `json:"read_only"`
	Offline  bool `json:"offline"`
}
type Inference struct {
	Domain      string   `json:"domain"`
	Domains     []string `json:"domains"`
	Scope       string   `json:"scope"`
	Depth       string   `json:"depth"`
	Uncertainty string   `json:"uncertainty"`
	Reasons     []string `json:"reasons"`
}
type Cohort struct {
	Domain           string   `json:"domain"`
	Intent           string   `json:"intent"`
	WorkUnitIDs      []string `json:"work_unit_ids"`
	RequiredEvidence []string `json:"required_evidence"`
}
type Bounds struct {
	MaxDirections  int `json:"max_directions"`
	MaxWorkUnits   int `json:"max_work_units"`
	MaxExperiments int `json:"max_experiments"`
	MaxConcurrency int `json:"max_concurrency"`
	MaxCycles      int `json:"max_cycles"`
}
type Stage struct {
	Name          string `json:"name"`
	MaxItems      int    `json:"max_items"`
	MaxConcurrent int    `json:"max_concurrent"`
	ExitCriterion string `json:"exit_criterion"`
}
type UnitBounds struct {
	MaxFiles        int `json:"max_files"`
	MaxDeliverables int `json:"max_deliverables"`
	MaxExperiments  int `json:"max_experiments"`
}
type Witness struct {
	Class             string `json:"class"`
	Evidence          string `json:"evidence"`
	CommandHint       string `json:"command_hint"`
	SelectionEvidence string `json:"selection_evidence,omitempty"`
	RejectionEvidence string `json:"rejection_evidence,omitempty"`
}
type CandidateDirection struct {
	Candidate string `json:"candidate"`
	Question  string `json:"question"`
	Decision  string `json:"decision"`
}
type WorkUnit struct {
	ID         string              `json:"id"`
	Stage      string              `json:"stage"`
	Cohort     string              `json:"cohort"`
	Title      string              `json:"title"`
	Body       string              `json:"body"`
	ScopeHints []string            `json:"scope_hints"`
	DependsOn  []string            `json:"depends_on"`
	Bounds     UnitBounds          `json:"bounds"`
	Direction  *CandidateDirection `json:"candidate_direction,omitempty"`
	Witness    Witness             `json:"witness"`
}
type Handoff struct {
	Tool    string   `json:"tool"`
	Profile string   `json:"profile"`
	Model   string   `json:"model"`
	Args    []string `json:"args"`
	Command string   `json:"command"`
}
type NativeReceipt struct {
	Schema         string   `json:"schema"`
	Required       bool     `json:"required"`
	RequiredFields []string `json:"required_fields"`
}
type NativeFallback struct {
	Engine             string   `json:"engine"`
	SilentFallback     bool     `json:"silent_fallback"`
	ExplicitExceptions []string `json:"explicit_exceptions"`
}
type NativeAccounting struct {
	Method   string   `json:"method"`
	Includes []string `json:"includes"`
	Rule     string   `json:"rule"`
}
type SanctionedHardware struct {
	Required  bool   `json:"required"`
	Reference string `json:"reference"`
	Rule      string `json:"rule"`
}
type NativeContract struct {
	Engine                   string             `json:"engine"`
	DefaultModel             string             `json:"default_model"`
	Receipt                  NativeReceipt      `json:"receipt"`
	Fallback                 NativeFallback     `json:"fallback"`
	MatchedQuality           bool               `json:"matched_quality"`
	MatchedWorkload          bool               `json:"matched_workload"`
	MatchedOperatingEnvelope bool               `json:"matched_operating_envelope"`
	Accounting               NativeAccounting   `json:"accounting"`
	Hardware                 SanctionedHardware `json:"hardware"`
}
type LearningRecord struct {
	Schema     string   `json:"schema"`
	Hypothesis string   `json:"hypothesis"`
	Outcome    string   `json:"outcome"`
	Evidence   []string `json:"evidence"`
	Adjustment string   `json:"adjustment"`
}

var pathPattern = regexp.MustCompile(`[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+`)

func Compile(raw string) (Plan, error) {
	if !utf8.ValidString(raw) {
		return Plan{}, errors.New("objective must be valid UTF-8")
	}
	for _, r := range raw {
		if (r < ' ' && !strings.ContainsRune("\t\n\v\f\r", r)) || r == 0x7f {
			return Plan{}, errors.New("objective contains an unsupported control character")
		}
	}
	objective := strings.Join(strings.Fields(raw), " ")
	if objective == "" {
		return Plan{}, errors.New("objective is required")
	}
	if len(objective) > maxObjectiveBytes {
		return Plan{}, fmt.Errorf("objective exceeds %d bytes", maxObjectiveBytes)
	}
	inference := infer(objective)
	bounds := inferBounds(inference)
	units := buildWorkUnits(objective, inference, bounds)
	if len(units) > bounds.MaxWorkUnits {
		units = units[:bounds.MaxWorkUnits]
	}
	plan := Plan{
		Schema: Schema, ObjectiveID: objectiveID(objective), Objective: objective,
		Mode: Mode{ReadOnly: true, Offline: true}, Inference: inference, Cohorts: buildCohorts(inference, units), Bounds: bounds,
		Stages: []Stage{
			{Name: "expand", MaxItems: bounds.MaxDirections, MaxConcurrent: min(2, bounds.MaxConcurrency), ExitCriterion: "candidate directions are distinct, bounded, and mapped to observable witnesses"},
			{Name: "experiment", MaxItems: bounds.MaxExperiments, MaxConcurrent: min(bounds.MaxExperiments, bounds.MaxConcurrency), ExitCriterion: "the smallest viable direction has a captured pass/fail artifact"},
			{Name: "contract", MaxItems: 1, MaxConcurrent: 1, ExitCriterion: "accepted effects, independent evidence, leftovers, and next-cycle adjustment are reconciled"},
		},
		WorkUnits: units,
		Handoff:   Handoff{Tool: "fak ultracode", Profile: "ultracode", Model: ultracodeModel, Args: []string{"--task-text", objective, "--json"}, Command: "fak ultracode --task-text " + shellQuote(objective) + " --json"},
		Learning: LearningRecord{
			Schema:     LearningSchema,
			Hypothesis: "a bounded expand and experiment cycle advances " + objective,
			Outcome:    "pending",
			Evidence:   []string{},
			Adjustment: "pending contract-stage reconciliation from independent witnesses",
		},
	}
	if hasDomain(inference.Domains, DomainNativeModel) {
		plan.Native = nativeContract()
	}
	return plan, nil
}

func Marshal(plan Plan) ([]byte, error) {
	b, err := json.MarshalIndent(plan, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}

func Render(plan Plan) string {
	var b strings.Builder
	fmt.Fprintf(&b, "FAK agentic plan %s\nobjective: %s\n", plan.ObjectiveID, plan.Objective)
	fmt.Fprintf(&b, "inference: domain=%s scope=%s depth=%s uncertainty=%s\n", plan.Inference.Domain, plan.Inference.Scope, plan.Inference.Depth, plan.Inference.Uncertainty)
	for _, r := range plan.Inference.Reasons {
		fmt.Fprintf(&b, "  reason: %s\n", r)
	}
	fmt.Fprintf(&b, "bounds: directions<=%d units<=%d experiments<=%d concurrency<=%d cycles<=%d\n", plan.Bounds.MaxDirections, plan.Bounds.MaxWorkUnits, plan.Bounds.MaxExperiments, plan.Bounds.MaxConcurrency, plan.Bounds.MaxCycles)
	for _, s := range plan.Stages {
		fmt.Fprintf(&b, "stage %s: items<=%d concurrency<=%d; exit=%s\n", s.Name, s.MaxItems, s.MaxConcurrent, s.ExitCriterion)
	}
	b.WriteString("issue-ready work:\n")
	for _, u := range plan.WorkUnits {
		fmt.Fprintf(&b, "  %s [%s/%s] %s; files<=%d deliverables<=%d witness=%s\n", u.ID, u.Stage, u.Cohort, u.Title, u.Bounds.MaxFiles, u.Bounds.MaxDeliverables, u.Witness.Class)
	}
	fmt.Fprintf(&b, "handoff: tool=%s profile=%s model=%s\n", plan.Handoff.Tool, plan.Handoff.Profile, plan.Handoff.Model)
	if plan.Native != nil {
		fmt.Fprintf(&b, "native contract: engine=%s model=%s receipt=%s matched_quality=%t matched_workload=%t matched_envelope=%t\n", plan.Native.Engine, plan.Native.DefaultModel, plan.Native.Receipt.Schema, plan.Native.MatchedQuality, plan.Native.MatchedWorkload, plan.Native.MatchedOperatingEnvelope)
		fmt.Fprintf(&b, "native fallback: llama.cpp silent=%t\n", plan.Native.Fallback.SilentFallback)
		fmt.Fprintf(&b, "native accounting: %s\nnative hardware: %s\n", plan.Native.Accounting.Method, plan.Native.Hardware.Reference)
	}
	fmt.Fprintf(&b, "learning: schema=%s fields=hypothesis,outcome,evidence,adjustment\nmode: read_only=%t offline=%t\n", plan.Learning.Schema, plan.Mode.ReadOnly, plan.Mode.Offline)
	return b.String()
}

func hasNativeExecutionIntent(lower string) bool {
	for _, clause := range intentClauses(lower) {
		clause = strings.TrimSpace(clause)
		if clause == "" || !hasNativeTechnology(clause) {
			continue
		}
		if developmentOnlyClause(clause) && !hasExplicitExecutionInDevelopmentSurface(clause) {
			continue
		}
		explicitAction := containsAny(" "+clause+" ",
			" run ", " runs ", " running", " execute", " execution", " serve ", " serving ",
			" benchmark", " measure", " profile", " optimiz", " tune ", " tuning ",
			" accelerat", " throughput", " latency", " tokens/sec", " tok/s",
		)
		hardNativeBuild := containsAny(clause, "kernel", "gemm", "inference runtime", "model engine", "decode path", "prefill path", "kv cache") &&
			containsAny(clause, "implement", "build", "add", "replace", "rewrite")
		if explicitAction || hardNativeBuild {
			return true
		}
	}
	return false
}

func hasNativeTechnology(s string) bool {
	return containsAny(s,
		"fak-native", "native model", "model engine", "model running", "model-running",
		"inference runtime", "inference engine", "qwen", "llama.cpp", "cuda", "gpu",
		"gguf", "gemm", "prefill", "decode", "kv cache", "quantiz", "tokens/sec", "tok/s",
	)
}

func hasExplicitExecutionInDevelopmentSurface(clause string) bool {
	padded := " " + clause + " "
	return hasPrefixAny(clause,
		"run ", "execute ", "serve ", "benchmark ", "measure ", "profile ",
		"optimize ", "tune ", "accelerate ",
	) || containsAny(padded,
		" to run ", " to execute ", " to serve ", " to benchmark ", " to measure ",
		" to profile ", " to optimize ", " to tune ", " to accelerate ",
		" that runs ", " that executes ", " that serves ", " that benchmarks ",
		" that measures ", " that profiles ", " that optimizes ", " that tunes ",
		" will run ", " will execute ", " will serve ", " will benchmark ",
	)
}

func hasDevelopmentIntent(lower string, native bool) bool {
	if !native {
		return true
	}
	padded := " " + lower + " "
	return containsAny(padded,
		" development ",
		" cli ", " command ", " api ", " api-client", " client ", " ui ", " dashboard ",
		" doc", " documentation", " config", " schema", " json", " workflow", " github issue",
		" go leaf", " package ", " integration", " public input", " parser",
	)
}

func intentClauses(lower string) []string {
	replacer := strings.NewReplacer(
		"\n", ";",
		". ", ";",
		";", ";",
		", then ", ";",
		" then ", ";",
		" and ", ";",
	)
	return strings.Split(replacer.Replace(lower), ";")
}

func developmentOnlyClause(clause string) bool {
	clause = strings.TrimSpace(clause)
	padded := " " + clause + " "
	if containsAny(padded, " api ", " api-client ", " client ", " ui ", " dashboard ", " config ", " configuration ", " documentation ", " docs ") {
		return true
	}
	return hasPrefixAny(clause,
		"document ", "documenting ", "write documentation", "update documentation", "add documentation",
		"write docs", "update docs", "add docs",
		"configure ", "configuration ", "add config", "update config",
		"improve documentation", "improve docs", "improve config",
	)
}

func hasPrefixAny(s string, prefixes ...string) bool {
	for _, prefix := range prefixes {
		if strings.HasPrefix(s, prefix) {
			return true
		}
	}
	return false
}

func infer(objective string) Inference {
	lower := strings.ToLower(objective)
	reasons := []string{}
	native := hasNativeExecutionIntent(lower)
	development := hasDevelopmentIntent(lower, native)
	domains := make([]string, 0, 2)
	if development {
		domains = append(domains, DomainDevelopment)
	}
	if native {
		domains = append(domains, DomainNativeModel)
	}
	domain := DomainDevelopment
	switch len(domains) {
	case 2:
		domain = DomainMixed
		reasons = append(reasons, "the objective contains independent development and native-model execution cohorts")
	case 1:
		domain = domains[0]
		if native {
			reasons = append(reasons, "native-model technology is paired with explicit execution or performance intent")
		} else {
			reasons = append(reasons, "the objective requests development work without a native execution campaign")
		}
	default:
		domains = []string{DomainDevelopment}
		reasons = append(reasons, "the objective defaults to development because no native execution campaign was detected")
	}
	paths := extractPaths(objective)
	signals := countSignals(" "+lower+" ", []string{" cli ", " command ", " api ", " package ", " test", " doc", " workflow", " schema", " json", " ui ", " integration"})
	scope := "focused"
	if containsAny(lower, "end-to-end", "system-wide", "architecture", "rollout", "migration") || signals >= 5 {
		scope = "system"
		reasons = append(reasons, "the objective crosses a system boundary or names at least five implementation surfaces")
	} else if signals >= 2 || len(paths) >= 2 {
		scope = "multi_surface"
		reasons = append(reasons, "the objective names multiple implementation or verification surfaces")
	} else {
		reasons = append(reasons, "the objective fits one primary implementation surface")
	}
	depth := "standard"
	if containsAny(lower, "architecture", "migration", "end-to-end", "performance", "benchmark", "optimiz", "concurren", "scheduler", "security") || (native && containsAny(lower, "engine", "kernel", "cuda", "gpu", "throughput")) {
		depth = "deep"
		reasons = append(reasons, "the objective requires design, performance, integration, or operating-envelope evidence")
	} else if containsAny(lower, "rename", "typo", "wording", "comment only", "documentation only") {
		depth = "shallow"
		reasons = append(reasons, "the objective is a narrow textual or naming change")
	} else {
		reasons = append(reasons, "the objective requires implementation plus targeted verification")
	}
	uncertainty := "low"
	if containsAny(lower, "unknown", "investigate", "explore", "research", "hypothesis", "compare", "evaluate", "benchmark", "optimiz", "bottleneck") {
		uncertainty = "high"
		reasons = append(reasons, "the objective contains exploratory, comparative, or optimization language")
	} else if native || depth == "deep" || scope == "system" || len(strings.Fields(objective)) > 32 {
		uncertainty = "medium"
		reasons = append(reasons, "the objective has material integration or execution-envelope uncertainty")
	} else {
		reasons = append(reasons, "the objective states a concrete bounded outcome")
	}
	return Inference{Domain: domain, Domains: domains, Scope: scope, Depth: depth, Uncertainty: uncertainty, Reasons: reasons}
}

func inferBounds(i Inference) Bounds {
	b := Bounds{MaxDirections: 3, MaxWorkUnits: 3, MaxExperiments: 1, MaxConcurrency: 2, MaxCycles: 1}
	if i.Scope == "multi_surface" {
		b = Bounds{4, 5, 2, 3, 1}
	} else if i.Scope == "system" {
		b = Bounds{5, 7, 3, 4, 1}
	}
	if i.Depth == "deep" && b.MaxWorkUnits < 8 {
		b.MaxWorkUnits++
	}
	if i.Uncertainty == "high" {
		b.MaxDirections = min(6, b.MaxDirections+1)
		b.MaxExperiments = min(3, b.MaxExperiments+1)
	}
	if hasDomain(i.Domains, DomainNativeModel) {
		b.MaxExperiments = max(2, b.MaxExperiments)
	}
	if i.Domain == DomainMixed {
		b.MaxDirections = max(4, b.MaxDirections)
		b.MaxExperiments = 3
	}
	// The work-unit cap covers the largest allowed direction set, every
	// cohort experiment, and the shared reconciliation unit. This prevents a
	// late slice from silently deleting contract evidence.
	b.MaxWorkUnits = max(b.MaxWorkUnits, b.MaxDirections+b.MaxExperiments+1)
	return b
}

type directionSpec struct {
	cohort    string
	candidate string
	question  string
}

func buildWorkUnits(objective string, i Inference, bounds Bounds) []WorkUnit {
	directions := candidateDirections(i)
	if len(directions) > bounds.MaxDirections {
		directions = directions[:bounds.MaxDirections]
	}
	units := make([]WorkUnit, 0, len(directions)+bounds.MaxExperiments+1)
	directionIDs := make([]string, 0, len(directions))
	directionIDsByCohort := map[string][]string{}
	for _, spec := range directions {
		id := fmt.Sprintf("work-%02d", len(units)+1)
		directionIDs = append(directionIDs, id)
		directionIDsByCohort[spec.cohort] = append(directionIDsByCohort[spec.cohort], id)
		hints := scopeHints(objective, spec.cohort)
		w := Witness{
			Class:             "candidate_direction",
			Evidence:          "a bounded artifact makes this direction independently selectable against the other candidates",
			CommandHint:       "capture the smallest read-only comparison or repro for this direction",
			SelectionEvidence: "record the observed advantage and witness that justify selecting this direction",
			RejectionEvidence: "record the failed assumption, inferior result, or scope reason that justifies rejecting this direction",
		}
		u := unit(id, "expand", spec.cohort, "explore(agentic): "+spec.candidate+" for "+cohortSummary(objective, i, spec.cohort), cohortObjective(objective, i, spec.cohort), hints, nil, UnitBounds{3, 2, 0}, w)
		u.Direction = &CandidateDirection{Candidate: spec.candidate, Question: spec.question, Decision: "pending"}
		units = append(units, u)
	}

	experimentIDs := make([]string, 0, bounds.MaxExperiments)
	if hasDomain(i.Domains, DomainDevelopment) {
		id := fmt.Sprintf("work-%02d", len(units)+1)
		experimentIDs = append(experimentIDs, id)
		units = append(units, unit(id, "experiment", DomainDevelopment, "feat(agentic): implement the selected development spine for "+cohortSummary(objective, i, DomainDevelopment), cohortObjective(objective, i, DomainDevelopment), scopeHints(objective, DomainDevelopment), directionIDsByCohort[DomainDevelopment], UnitBounds{4, 2, 1}, Witness{
			Class:             "logic_behavior",
			Evidence:          "the selected development direction passes its captured repro through existing project primitives with deterministic output",
			CommandHint:       "fak validate --mine <development paths>",
			SelectionEvidence: "carry forward the selected development direction ID and its passing witness",
			RejectionEvidence: "carry forward every rejected development direction ID and its failed witness or bounded rationale",
		}))
		if i.Scope == "system" && i.Domain != DomainMixed {
			id = fmt.Sprintf("work-%02d", len(units)+1)
			experimentIDs = append(experimentIDs, id)
			units = append(units, unit(id, "experiment", DomainDevelopment, "test(agentic): verify development integration boundaries for "+cohortSummary(objective, i, DomainDevelopment), cohortObjective(objective, i, DomainDevelopment), scopeHints(objective, DomainDevelopment), directionIDsByCohort[DomainDevelopment], UnitBounds{4, 2, 1}, Witness{
				Class:             "integration",
				Evidence:          "the public entry point and immediate consumers pass a byte-deterministic integration test",
				CommandHint:       "fak validate --mine <integration paths>",
				SelectionEvidence: "retain the development direction selected for integration verification and its passing witness",
				RejectionEvidence: "retain rejected development directions and why integration evidence did not select them",
			}))
		}
	}
	if hasDomain(i.Domains, DomainNativeModel) {
		id := fmt.Sprintf("work-%02d", len(units)+1)
		experimentIDs = append(experimentIDs, id)
		units = append(units, unit(id, "experiment", DomainNativeModel, "perf(agentic): implement the selected fak-native path for "+cohortSummary(objective, i, DomainNativeModel), cohortObjective(objective, i, DomainNativeModel), scopeHints(objective, DomainNativeModel), directionIDsByCohort[DomainNativeModel], UnitBounds{4, 2, 1}, Witness{
			Class:             "native_execution",
			Evidence:          "receipt v2 names engine fak-native and model Qwen3.8 with no implicit backend substitution under matched quality, workload, and envelope controls",
			CommandHint:       "run focused native package tests before sanctioned-hardware dispatch",
			SelectionEvidence: "carry forward the selected native direction ID and the receipt that proves engine identity and quality",
			RejectionEvidence: "carry forward every rejected native direction ID and the receipt or envelope reason that rejected it",
		}))
		id = fmt.Sprintf("work-%02d", len(units)+1)
		experimentIDs = append(experimentIDs, id)
		units = append(units, unit(id, "experiment", DomainNativeModel, "bench(agentic): witness the selected native direction on sanctioned hardware for "+cohortSummary(objective, i, DomainNativeModel), cohortObjective(objective, i, DomainNativeModel), scopeHints(objective, DomainNativeModel), directionIDsByCohort[DomainNativeModel], UnitBounds{3, 2, 1}, Witness{
			Class:             "native_performance",
			Evidence:          "a receipt v2 reports fak-native Qwen3.8 execution with matched quality/workload/envelope and net-true setup, recovery, and verification cost",
			CommandHint:       "dispatch through docs/fleet-compute-nodes.md and retain the scrubbed receipt",
			SelectionEvidence: "retain the selected native direction and the matched sanctioned-hardware result that supports it",
			RejectionEvidence: "retain rejected native directions and the matched result, quality failure, or net-true cost that rejected them",
		}))
	}

	contractEvidence := "independent readback preserves selected and rejected direction evidence, accepted development effects, and every real leftover"
	if hasDomain(i.Domains, DomainNativeModel) && hasDomain(i.Domains, DomainDevelopment) {
		contractEvidence = "independent readback contracts both evidence sets: deterministic development behavior plus fak-native receipt v2, matched-envelope, sanctioned-hardware, and net-true native evidence"
	} else if hasDomain(i.Domains, DomainNativeModel) {
		contractEvidence = "independent readback contracts selected/rejected directions, fak-native receipt v2, matched outcome, net-true totals, and next-cycle adjustment"
	}
	contractHints := scopeHints(objective, DomainDevelopment)
	if hasDomain(i.Domains, DomainNativeModel) {
		contractHints = append(contractHints, scopeHints(objective, DomainNativeModel)...)
	}
	contractID := fmt.Sprintf("work-%02d", len(units)+1)
	contractDeps := append(append([]string{}, directionIDs...), experimentIDs...)
	units = append(units, unit(contractID, "contract", "all", "chore(agentic): reconcile direction and cohort evidence for "+objectiveSummary(objective), objective, unique(contractHints, 4), contractDeps, UnitBounds{2, 2, 0}, Witness{
		Class:             "contract",
		Evidence:          contractEvidence,
		CommandHint:       "fak validate --mine <all changed paths>",
		SelectionEvidence: "retain the witness and cohort outcome for every selected candidate direction",
		RejectionEvidence: "retain the witness or bounded rationale for every rejected candidate direction",
	}))
	return units
}

func cohortSummary(objective string, i Inference, cohort string) string {
	if i.Domain != DomainMixed {
		return objectiveSummary(objective)
	}
	return fmt.Sprintf("%s cohort of %s", strings.ReplaceAll(cohort, "_", "-"), objectiveID(objective))
}

func cohortObjective(objective string, i Inference, cohort string) string {
	if i.Domain != DomainMixed || cohort == DomainNativeModel {
		return objective
	}
	return fmt.Sprintf("Implement and verify the development cohort of %s through existing project primitives; model execution is assigned to separate native-model work units.", objectiveID(objective))
}

func candidateDirections(i Inference) []directionSpec {
	count := 1
	if i.Scope == "multi_surface" {
		count = 3
	} else if i.Scope == "system" {
		count = 4
	}
	if i.Depth == "deep" || i.Uncertainty == "medium" {
		count = max(count, 3)
	}
	if i.Uncertainty == "high" {
		count = max(count, 4)
	}
	if i.Domain == DomainMixed {
		count = max(count, 4)
	}

	var candidates []directionSpec
	switch i.Domain {
	case DomainMixed:
		candidates = []directionSpec{
			{DomainDevelopment, "freeze the public input and deterministic schema contract", "Which smallest input and output contract preserves compatibility while removing ambiguity?"},
			{DomainNativeModel, "freeze the matched fak-native execution envelope", "Which quality, workload, hardware, and accounting controls make the native result falsifiable?"},
			{DomainDevelopment, "compose existing orchestration and issue primitives", "Which existing seams implement the objective without creating a second scheduler or ledger?"},
			{DomainNativeModel, "prove the fak-native Qwen3.8 execution path before performance claims", "Which focused receipt proves engine identity and rejects silent backend substitution?"},
			{DomainDevelopment, "capture adversarial CLI and deterministic integration dogfood", "Which edge cases distinguish a safe public contract from permissive parsing or unstable output?"},
			{DomainNativeModel, "compare the selected native direction on sanctioned hardware", "Which bounded hardware experiment can accept or reject the direction under net-true accounting?"},
		}
	case DomainNativeModel:
		candidates = []directionSpec{
			{DomainNativeModel, "freeze the matched fak-native execution envelope", "Which quality, workload, hardware, and accounting controls make the result falsifiable?"},
			{DomainNativeModel, "isolate the smallest fak-native runtime or kernel lever", "Which single native lever can change while every other receipt control remains fixed?"},
			{DomainNativeModel, "prove Qwen3.8 engine identity and quality parity", "Which receipt evidence rejects silent fallback and quality drift before measuring speed?"},
			{DomainNativeModel, "measure the lever on sanctioned hardware with net-true accounting", "Does the candidate survive setup, recovery, verification, and operating-envelope costs?"},
			{DomainNativeModel, "stress the matched workload and memory envelope", "Where does the candidate stop improving or violate the declared envelope?"},
			{DomainNativeModel, "reconcile native receipts into the next bounded cycle", "Which evidence selects, rejects, or narrows each candidate without an unbounded loop?"},
		}
	default:
		candidates = []directionSpec{
			{DomainDevelopment, "freeze the public behavior and witness contract", "Which smallest observable contract proves the requested outcome?"},
			{DomainDevelopment, "implement the narrowest end-to-end spine through existing primitives", "Which composition reaches the public outcome without adding a parallel system?"},
			{DomainDevelopment, "capture adversarial compatibility and deterministic output cases", "Which edge cases would expose ambiguous inputs, unstable output, or consumer breakage?"},
			{DomainDevelopment, "verify immediate integration boundaries", "Which direct consumers must agree before the direction can be selected?"},
			{DomainDevelopment, "bound issue-ready follow-on work", "Which leftovers are independently dispatchable rather than hidden in the spine?"},
			{DomainDevelopment, "reconcile selected and rejected direction evidence", "Which evidence should change the next cycle rather than merely narrate it?"},
		}
	}
	if count > len(candidates) {
		count = len(candidates)
	}
	return candidates[:count]
}

func buildCohorts(i Inference, units []WorkUnit) []Cohort {
	out := make([]Cohort, 0, len(i.Domains))
	for _, domain := range i.Domains {
		ids := make([]string, 0, len(units))
		for _, u := range units {
			if u.Cohort == domain || u.Cohort == "all" {
				ids = append(ids, u.ID)
			}
		}
		c := Cohort{Domain: domain, WorkUnitIDs: ids}
		if domain == DomainNativeModel {
			c.Intent = "execute or improve the model through the fak-native path"
			c.RequiredEvidence = []string{
				"selected and rejected candidate-direction evidence",
				"fak-native Qwen3.8 receipt v2 with no silent llama.cpp fallback",
				"matched quality, workload, operating envelope, sanctioned hardware, and net-true accounting",
			}
		} else {
			c.Intent = "implement and verify the non-model-execution product or developer surfaces"
			c.RequiredEvidence = []string{
				"selected and rejected candidate-direction evidence",
				"captured deterministic behavior or integration witness",
			}
		}
		out = append(out, c)
	}
	return out
}

func unit(id, stage, cohort, title, objective string, hints, deps []string, bounds UnitBounds, w Witness) WorkUnit {
	if deps == nil {
		deps = []string{}
	}
	decisionEvidence := ""
	if w.SelectionEvidence != "" || w.RejectionEvidence != "" {
		decisionEvidence = fmt.Sprintf("\n- Selection evidence: %s\n- Rejection evidence: %s", w.SelectionEvidence, w.RejectionEvidence)
	}
	return WorkUnit{ID: id, Stage: stage, Cohort: cohort, Title: title, ScopeHints: append([]string(nil), hints...), DependsOn: append([]string(nil), deps...), Bounds: bounds, Witness: w, Body: fmt.Sprintf("## Objective\n\n%s\n\n## Bound\n\n- Stage: `%s`\n- Cohort: `%s`\n- Files: at most %d\n- Deliverables: at most %d\n- Experiments: at most %d\n\n## Scope hints\n\n- %s\n\n## Witness\n\n- Class: `%s`\n- Evidence: %s\n- Command hint: `%s`%s\n", objective, stage, cohort, bounds.MaxFiles, bounds.MaxDeliverables, bounds.MaxExperiments, strings.Join(hints, "\n- "), w.Class, w.Evidence, w.CommandHint, decisionEvidence)}
}

func nativeContract() *NativeContract {
	return &NativeContract{
		Engine:       "fak-native",
		DefaultModel: "Qwen3.8",
		Receipt: NativeReceipt{
			Schema:   nativeperf.ReceiptSchema,
			Required: true,
			// These are the required top-level JSON fields of
			// nativeperf.ExperimentReceipt. Optional legacy ambient evidence and
			// v2 system-baseline attachments remain available but are not invented
			// here as conceptual receipt aliases.
			RequiredFields: []string{
				"schema", "role", "envelope_id", "changed_lever_id", "revision",
				"artifact_sha256", "machine", "controls", "unchanged_controls",
				"changed_axes", "repetitions", "memory", "execution", "quality",
				"module_versions", "commands", "profiler_artifacts",
			},
		},
		Fallback: NativeFallback{
			Engine:             "llama.cpp",
			SilentFallback:     false,
			ExplicitExceptions: []string{"benchmark", "parity or reference diagnosis", "migration or interoperability"},
		},
		MatchedQuality:           true,
		MatchedWorkload:          true,
		MatchedOperatingEnvelope: true,
		Accounting: NativeAccounting{
			Method:   "net_true",
			Includes: []string{"setup", "recovery", "verification"},
			Rule:     "all elapsed time, retries, recovery work, and verification cost count in the reported result",
		},
		Hardware: SanctionedHardware{
			Required:  true,
			Reference: "docs/fleet-compute-nodes.md",
			Rule:      "dispatch to a sanctioned node when local hardware is insufficient; missing local hardware is not a terminal blocker",
		},
	}
}

func scopeHints(objective, domain string) []string {
	h := extractPaths(objective)
	if domain == DomainNativeModel {
		h = append(h, "fak-native engine path", "native benchmark and receipt surface")
	} else {
		h = append(h, "target implementation surface", "targeted tests")
	}
	return unique(h, 4)
}
func extractPaths(s string) []string {
	found := pathPattern.FindAllString(s, -1)
	out := []string{}
	for _, p := range found {
		p = strings.TrimRight(p, ".,:;")
		if strings.HasSuffix(p, "/") {
			continue
		}
		leaf := p[strings.LastIndex(p, "/")+1:]
		if strings.HasPrefix(p, "cmd/") || strings.HasPrefix(p, "internal/") || strings.HasPrefix(p, "docs/") || strings.HasPrefix(p, "examples/") || strings.Contains(leaf, ".") {
			out = append(out, p)
		}
	}
	sort.Strings(out)
	return unique(out, len(out))
}
func unique(in []string, limit int) []string {
	if limit <= 0 {
		return []string{}
	}
	seen := map[string]bool{}
	out := []string{}
	for _, v := range in {
		v = strings.TrimSpace(v)
		if v == "" || seen[v] {
			continue
		}
		seen[v] = true
		out = append(out, v)
		if len(out) == limit {
			break
		}
	}
	return out
}
func objectiveID(s string) string {
	d := sha256.Sum256([]byte(s))
	return "objective-" + hex.EncodeToString(d[:6])
}
func objectiveSummary(s string) string {
	w := strings.Fields(s)
	if len(w) > 10 {
		w = w[:10]
	}
	x := strings.Trim(strings.Join(w, " "), " .,:;")
	if utf8.RuneCountInString(x) > 80 {
		x = strings.TrimSpace(string([]rune(x)[:80]))
	}
	return x
}
func containsAny(s string, xs ...string) bool {
	for _, x := range xs {
		if strings.Contains(s, x) {
			return true
		}
	}
	return false
}
func hasDomain(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
func countSignals(s string, xs []string) int {
	n := 0
	for _, x := range xs {
		if strings.Contains(s, x) {
			n++
		}
	}
	return n
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func shellQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'"'"'`) + "'"
}
