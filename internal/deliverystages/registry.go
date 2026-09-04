// Package deliverystages is the canonical inventory of agent-development
// delivery stages and bottleneck boundaries.
package deliverystages

import (
	"fmt"
	"sort"
	"strings"
)

// Schema is the schema identifier for delivery stages manifests.
const Schema = "fak.delivery-stages/v1"

// StageID uniquely identifies a delivery stage in the execution lifecycle.
type StageID string

// BottleneckID uniquely identifies a failure or obstruction classification.
type BottleneckID string

// Stage models a single discrete step in the agent development lifecycle,
// including its ownership, prerequisite gates, and invalidation rules.
type Stage struct {
	ID              StageID   `json:"id"`
	Name            string    `json:"name"`
	Inputs          []string  `json:"inputs"`
	Outputs         []string  `json:"outputs"`
	Owner           string    `json:"owner"`
	Prerequisites   []StageID `json:"prerequisites,omitempty"`
	Gate            string    `json:"gate"`
	Evidence        []string  `json:"evidence"`
	RetryCommand    string    `json:"retry_command"`
	SplitDimensions []string  `json:"split_dimensions"`
	Invalidates     []StageID `json:"invalidates,omitempty"`
}

// Bottleneck defines a structured recovery boundary when execution is blocked.
type Bottleneck struct {
	ID                BottleneckID `json:"id"`
	Name              string       `json:"name"`
	RequiredFields    []string     `json:"required_fields"`
	DefaultNextAction string       `json:"default_next_action"`
}

// Crosswalk maps local command or failure signals to standardized stages and bottlenecks.
type Crosswalk struct {
	Local      string       `json:"local"`
	Stage      StageID      `json:"stage"`
	Bottleneck BottleneckID `json:"bottleneck"`
}

// Registry contains the declared stages, bottlenecks, and crosswalk mappings.
type Registry struct {
	Schema      string       `json:"schema"`
	Stages      []Stage      `json:"stages"`
	Bottlenecks []Bottleneck `json:"bottlenecks"`
	Crosswalks  []Crosswalk  `json:"crosswalks"`
}

var stageSpecs = []struct {
	id                StageID
	name, owner, gate string
	prerequisites     []StageID
}{
	{"intent", "Intent and intake", "operator", "intent-declared", nil},
	{"issue-contract", "Issue contract", "planner", "issue-contract", []StageID{"intent"}},
	{"value-centrality", "Value and centrality", "planner", "problem-frame", []StageID{"issue-contract"}},
	{"scope", "Scope and decomposition", "planner", "ticket-scope", []StageID{"value-centrality"}},
	{"dependency-readiness", "Dependency readiness", "planner", "dependency-ready", []StageID{"scope"}},
	{"capacity-admission", "Account and capacity admission", "dispatcher", "fleet-preflight", []StageID{"dependency-readiness"}},
	{"lane-admission", "Lane and tree admission", "dispatcher", "dos-arbitrate", []StageID{"capacity-admission"}},
	{"context-acquisition", "Context acquisition", "worker", "context-ready", []StageID{"lane-admission"}},
	{"authoring", "Authoring", "worker", "work-produced", []StageID{"context-acquisition"}},
	{"recording", "Checkpoint and commit recording", "worker", "commit-recorded", []StageID{"authoring"}},
	{"compile-admission", "Compile-stream admission", "builder", "compile-set", []StageID{"recording"}},
	{"build", "Build", "builder", "go-build", []StageID{"compile-admission"}},
	{"static-analysis", "Vet and static analysis", "verifier", "go-vet", []StageID{"build"}},
	{"affected-tests", "Affected tests", "verifier", "affected-tests", []StageID{"static-analysis"}},
	{"full-tests", "Full tests", "verifier", "full-tests", []StageID{"affected-tests"}},
	{"evidence", "Evidence and witness", "witness", "evidence-complete", []StageID{"full-tests"}},
	{"integration", "Integration and push", "integrator", "pre-push", []StageID{"evidence"}},
	{"release-admission", "Release admission", "release", "shipgate", []StageID{"integration"}},
	{"release-publication", "Release publication", "release", "release-published", []StageID{"release-admission"}},
	{"runtime-observation", "Deployment and runtime observation", "operator", "runtime-observed", []StageID{"release-publication"}},
	{"closure", "Closure and reconciliation", "planner", "closure-witness", []StageID{"runtime-observation"}},
}

var bottleneckSpecs = []struct {
	id           BottleneckID
	name, action string
}{
	{"missing-input", "Missing input", "supply the named input artifact"},
	{"dependency", "Dependency not ready", "finish or reclassify the named prerequisite"},
	{"collision-lease", "Tree collision or lease", "wait, repartition, or acquire a disjoint lane"},
	{"capacity-seat", "Capacity or seat", "wait for admitted capacity or choose an available pool"},
	{"credential-provider", "Credential or provider", "use a healthy credential/provider route"},
	{"compile", "Compile failure", "diagnose the smallest failing package or file unit"},
	{"static-analysis", "Static-analysis failure", "rerun the named analyzer on the failing unit"},
	{"test", "Test failure", "rerun and split the failing test scope"},
	{"evidence", "Missing or invalid evidence", "capture the required witness"},
	{"integration-drift", "Integration drift", "refresh against the target and rerun invalidated stages"},
	{"release-policy", "Release policy", "satisfy the named release admission rule"},
	{"runtime-environment", "Runtime or environment", "route to the declared environment and capture evidence"},
	{"external-service", "External service", "retry or reroute at the external boundary"},
	{"unknown-irreducible", "Unknown or irreducible", "run recursive diagnosis or declare the missing discriminator"},
}

// Default returns the canonical delivery-stages registry with all standard
// stages, bottlenecks, and crosswalks populated.
func Default() Registry {
	r := Registry{Schema: Schema}
	for i, spec := range stageSpecs {
		stage := Stage{ID: spec.id, Name: spec.name, Owner: spec.owner, Prerequisites: append([]StageID(nil), spec.prerequisites...), Gate: spec.gate,
			Inputs: []string{"work-unit", "prerequisite-receipts"}, Outputs: []string{"stage-receipt"}, Evidence: []string{"gate-result"},
			RetryCommand: "fak work-delivery transition --stage " + string(spec.id), SplitDimensions: []string{"unit", "tree", "dependency", "test-group"}}
		for _, later := range stageSpecs[i+1:] {
			stage.Invalidates = append(stage.Invalidates, later.id)
		}
		r.Stages = append(r.Stages, stage)
	}
	for _, spec := range bottleneckSpecs {
		r.Bottlenecks = append(r.Bottlenecks, Bottleneck{ID: spec.id, Name: spec.name,
			RequiredFields: []string{"unit", "stage", "evidence", "owner", "retry_boundary", "next_action"}, DefaultNextAction: spec.action})
	}
	r.Crosswalks = []Crosswalk{
		{"git commit", "recording", "missing-input"}, {"TIER_DECLARED", "compile-admission", "static-analysis"},
		{"go build", "build", "compile"}, {"go vet", "static-analysis", "static-analysis"}, {"CI red", "full-tests", "unknown-irreducible"},
		{"dos arbitrate", "lane-admission", "collision-lease"}, {"REFUSE_AT_CAP", "capacity-admission", "capacity-seat"},
		{"pre-push", "integration", "integration-drift"}, {"shipgate", "release-admission", "release-policy"},
	}
	return r
}

// Validate checks the registry for schema compliance, duplicate IDs,
// dangling prerequisites or invalidations, cycles, and missing recovery metadata.
func (r Registry) Validate() error {
	if r.Schema != Schema {
		return fmt.Errorf("schema: want %q, got %q", Schema, r.Schema)
	}
	stageByID := map[StageID]Stage{}
	for _, stage := range r.Stages {
		if stage.ID == "" || stage.Name == "" || stage.Owner == "" || stage.Gate == "" {
			return fmt.Errorf("stage identity, owner, and gate are required: %#v", stage)
		}
		if len(stage.Inputs) == 0 || len(stage.Outputs) == 0 || len(stage.Evidence) == 0 || stage.RetryCommand == "" || len(stage.SplitDimensions) == 0 {
			return fmt.Errorf("stage %q lacks a composable retry/split contract", stage.ID)
		}
		if _, exists := stageByID[stage.ID]; exists {
			return fmt.Errorf("duplicate stage %q", stage.ID)
		}
		stageByID[stage.ID] = stage
	}
	for _, stage := range r.Stages {
		for _, dependency := range append(append([]StageID(nil), stage.Prerequisites...), stage.Invalidates...) {
			if _, ok := stageByID[dependency]; !ok {
				return fmt.Errorf("stage %q references unknown stage %q", stage.ID, dependency)
			}
		}
	}
	if cycle := stageCycle(r.Stages); len(cycle) > 0 {
		return fmt.Errorf("stage cycle: %s", strings.Join(cycle, " -> "))
	}
	bottlenecks := map[BottleneckID]bool{}
	for _, item := range r.Bottlenecks {
		if item.ID == "" || bottlenecks[item.ID] {
			return fmt.Errorf("duplicate or empty bottleneck %q", item.ID)
		}
		bottlenecks[item.ID] = true
		if item.DefaultNextAction == "" || len(item.RequiredFields) == 0 {
			return fmt.Errorf("bottleneck %q lacks recovery contract", item.ID)
		}
	}
	for _, crosswalk := range r.Crosswalks {
		if crosswalk.Local == "" || stageByID[crosswalk.Stage].ID == "" || !bottlenecks[crosswalk.Bottleneck] {
			return fmt.Errorf("invalid crosswalk %#v", crosswalk)
		}
	}
	return nil
}

// Stage looks up a stage by its StageID, returning false if not found.
func (r Registry) Stage(id StageID) (Stage, bool) {
	for _, stage := range r.Stages {
		if stage.ID == id {
			return stage, true
		}
	}
	return Stage{}, false
}

// InvalidatedAfter returns all downstream stages invalidated when the given stage changes.
func (r Registry) InvalidatedAfter(id StageID) ([]StageID, error) {
	stage, ok := r.Stage(id)
	if !ok {
		return nil, fmt.Errorf("unknown stage %q", id)
	}
	return append([]StageID(nil), stage.Invalidates...), nil
}

// ResolveLocal maps a tool command or failure string to its crosswalk definition using case-insensitive matching.
func (r Registry) ResolveLocal(local string) (Crosswalk, bool) {
	for _, item := range r.Crosswalks {
		if strings.EqualFold(item.Local, local) {
			return item, true
		}
	}
	return Crosswalk{}, false
}

func stageCycle(stages []Stage) []string {
	edges := map[StageID][]StageID{}
	ids := make([]StageID, 0, len(stages))
	for _, stage := range stages {
		ids = append(ids, stage.ID)
		edges[stage.ID] = append([]StageID(nil), stage.Prerequisites...)
		sort.Slice(edges[stage.ID], func(i, j int) bool { return edges[stage.ID][i] < edges[stage.ID][j] })
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	state := map[StageID]int{}
	stack := []StageID{}
	var visit func(StageID) []string
	visit = func(id StageID) []string {
		state[id] = 1
		stack = append(stack, id)
		for _, next := range edges[id] {
			if state[next] == 0 {
				if c := visit(next); c != nil {
					return c
				}
			}
			if state[next] == 1 {
				for i, v := range stack {
					if v == next {
						out := []string{}
						for _, x := range append(append([]StageID(nil), stack[i:]...), next) {
							out = append(out, string(x))
						}
						return out
					}
				}
			}
		}
		stack = stack[:len(stack)-1]
		state[id] = 2
		return nil
	}
	for _, id := range ids {
		if state[id] == 0 {
			if c := visit(id); c != nil {
				return c
			}
		}
	}
	return nil
}
