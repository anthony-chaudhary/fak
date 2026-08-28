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
	ultracodeModel    = "gpt-5.6-sol"
	maxObjectiveBytes = 16 * 1024
)

type Plan struct {
	Schema      string          `json:"schema"`
	ObjectiveID string          `json:"objective_id"`
	Objective   string          `json:"objective"`
	Mode        Mode            `json:"mode"`
	Inference   Inference       `json:"inference"`
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
	Scope       string   `json:"scope"`
	Depth       string   `json:"depth"`
	Uncertainty string   `json:"uncertainty"`
	Reasons     []string `json:"reasons"`
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
	Class       string `json:"class"`
	Evidence    string `json:"evidence"`
	CommandHint string `json:"command_hint"`
}
type WorkUnit struct {
	ID         string     `json:"id"`
	Stage      string     `json:"stage"`
	Title      string     `json:"title"`
	Body       string     `json:"body"`
	ScopeHints []string   `json:"scope_hints"`
	DependsOn  []string   `json:"depends_on"`
	Bounds     UnitBounds `json:"bounds"`
	Witness    Witness    `json:"witness"`
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
	hints := scopeHints(objective, inference.Domain)
	units := buildWorkUnits(objective, inference, hints)
	if len(units) > bounds.MaxWorkUnits {
		units = units[:bounds.MaxWorkUnits]
	}
	plan := Plan{
		Schema: Schema, ObjectiveID: objectiveID(objective), Objective: objective,
		Mode: Mode{ReadOnly: true, Offline: true}, Inference: inference, Bounds: bounds,
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
	if inference.Domain == DomainNativeModel {
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
		fmt.Fprintf(&b, "  %s [%s] %s; files<=%d deliverables<=%d witness=%s\n", u.ID, u.Stage, u.Title, u.Bounds.MaxFiles, u.Bounds.MaxDeliverables, u.Witness.Class)
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

func infer(objective string) Inference {
	lower := strings.ToLower(objective)
	reasons := []string{}
	domain := DomainDevelopment
	if containsAny(lower, "native model", "model engine", "model running", "inference runtime", "inference engine", "qwen", "llama.cpp", "cuda", "gpu", "gguf", "gemm", "prefill", "decode throughput", "kv cache", "quantiz", "tokens/sec", "tok/s") {
		domain = DomainNativeModel
		reasons = append(reasons, "native-model terms request model execution, engine, kernel, or accelerator work")
	} else {
		reasons = append(reasons, "no native-model execution terms were detected")
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
	if containsAny(lower, "architecture", "migration", "end-to-end", "performance", "benchmark", "optimiz", "concurren", "scheduler", "security") || (domain == DomainNativeModel && containsAny(lower, "engine", "kernel", "cuda", "gpu", "throughput")) {
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
	} else if domain == DomainNativeModel || depth == "deep" || scope == "system" || len(strings.Fields(objective)) > 32 {
		uncertainty = "medium"
		reasons = append(reasons, "the objective has material integration or execution-envelope uncertainty")
	} else {
		reasons = append(reasons, "the objective states a concrete bounded outcome")
	}
	return Inference{Domain: domain, Scope: scope, Depth: depth, Uncertainty: uncertainty, Reasons: reasons}
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
	// A native plan always has one contract-freeze unit, two independently
	// witnessed experiment units, and one reconciliation unit. Keep all four
	// visible instead of silently truncating the contract stage for a focused
	// objective.
	if i.Domain == DomainNativeModel {
		b.MaxWorkUnits = max(4, b.MaxWorkUnits)
		b.MaxExperiments = max(2, b.MaxExperiments)
	}
	return b
}

func buildWorkUnits(objective string, i Inference, hints []string) []WorkUnit {
	summary := objectiveSummary(objective)
	if i.Domain == DomainNativeModel {
		return []WorkUnit{
			unit("work-01", "expand", "bench(agentic): freeze the matched native envelope for "+summary, objective, hints, nil, UnitBounds{3, 2, 0}, Witness{"native_contract", "a frozen quality target, model artifact, workload, hardware envelope, and net-true accounting contract", "review the matched-envelope fixture before implementation"}),
			unit("work-02", "experiment", "perf(agentic): implement the fak-native path for "+summary, objective, hints, []string{"work-01"}, UnitBounds{4, 2, 1}, Witness{"native_execution", "the execution receipt names engine fak-native and model Qwen3.8 with no implicit backend substitution", "run focused native package tests before hardware dispatch"}),
			unit("work-03", "experiment", "bench(agentic): witness "+summary+" on sanctioned hardware", objective, hints, []string{"work-02"}, UnitBounds{3, 2, 1}, Witness{"native_performance", "a sanctioned-hardware receipt reports matched quality and operating envelope with setup, recovery, and verification included", "dispatch through docs/fleet-compute-nodes.md and retain the scrubbed receipt"}),
			unit("work-04", "contract", "chore(agentic): contract native receipts and learning for "+summary, objective, hints, []string{"work-03"}, UnitBounds{2, 2, 0}, Witness{"contract", "independent readback confirms the fak-native receipt, matched outcome, net-true totals, and next-cycle adjustment", "fak hwgate-lint <final report>"}),
		}
	}
	units := []WorkUnit{
		unit("work-01", "expand", "test(agentic): capture the objective contract for "+summary, objective, hints, nil, UnitBounds{3, 2, 0}, Witness{"logic_behavior", "a focused test or captured artifact fails before implementation and states the accepted outcome", "run the narrow affected-package test"}),
		unit("work-02", "experiment", "feat(agentic): implement the smallest working spine for "+summary, objective, hints, []string{"work-01"}, UnitBounds{4, 2, 1}, Witness{"logic_behavior", "the captured repro passes through the real implementation path without replacing existing schedulers", "fak validate --mine <unit paths>"}),
	}
	if i.Scope == "system" {
		units = append(units, unit("work-03", "experiment", "test(agentic): verify integration boundaries for "+summary, objective, hints, []string{"work-02"}, UnitBounds{4, 2, 1}, Witness{"integration", "the public entry point and immediate consumers pass a deterministic integration test", "fak validate --mine <integration paths>"}))
	}
	last := units[len(units)-1].ID
	units = append(units, unit(fmt.Sprintf("work-%02d", len(units)+1), "contract", "chore(agentic): reconcile witnesses and next-cycle work for "+summary, objective, hints, []string{last}, UnitBounds{2, 2, 0}, Witness{"contract", "accepted effects are independently read back and every real leftover is recorded or explicitly absent", "fak validate --mine <all changed paths>"}))
	return units
}

func unit(id, stage, title, objective string, hints, deps []string, bounds UnitBounds, w Witness) WorkUnit {
	if deps == nil {
		deps = []string{}
	}
	return WorkUnit{ID: id, Stage: stage, Title: title, ScopeHints: append([]string(nil), hints...), DependsOn: append([]string(nil), deps...), Bounds: bounds, Witness: w, Body: fmt.Sprintf("## Objective\n\n%s\n\n## Bound\n\n- Stage: `%s`\n- Files: at most %d\n- Deliverables: at most %d\n- Experiments: at most %d\n\n## Scope hints\n\n- %s\n\n## Witness\n\n- Class: `%s`\n- Evidence: %s\n- Command hint: `%s`\n", objective, stage, bounds.MaxFiles, bounds.MaxDeliverables, bounds.MaxExperiments, strings.Join(hints, "\n- "), w.Class, w.Evidence, w.CommandHint)}
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
