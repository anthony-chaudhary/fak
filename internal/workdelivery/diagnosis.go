package workdelivery

import (
	"fmt"
	"sort"
	"strings"
)

const DiagnosisSchema = "fak.work-delivery-diagnosis/v1"

type FailureClass string

const (
	FailureCompile  FailureClass = "compile"
	FailureTest     FailureClass = "test"
	FailureSafety   FailureClass = "safety"
	FailureExternal FailureClass = "external"
	FailureUnknown  FailureClass = "unknown"
)

type DiagnosticUnit struct {
	ID                     string   `json:"id"`
	Tree                   string   `json:"tree,omitempty"`
	TestGroup              string   `json:"test_group,omitempty"`
	Dependencies           []string `json:"dependencies,omitempty"`
	IndependentlyCheckable bool     `json:"independently_checkable"`
}

type FailureScope struct {
	ID    string           `json:"id"`
	Units []DiagnosticUnit `json:"units"`
}

type FailureObservation struct {
	Scope        FailureScope `json:"scope"`
	FailedUnitID string       `json:"failed_unit_id,omitempty"`
	Class        FailureClass `json:"class"`
	Gate         string       `json:"gate"`
	Evidence     []Evidence   `json:"evidence,omitempty"`
	CheckCommand string       `json:"check_command,omitempty"`
}

type DiagnosisKind string

const (
	DiagnosisLeaf        DiagnosisKind = "leaf"
	DiagnosisSplit       DiagnosisKind = "split"
	DiagnosisIrreducible DiagnosisKind = "irreducible"
)

type Diagnosis struct {
	Schema     string          `json:"schema"`
	Kind       DiagnosisKind   `json:"kind"`
	ScopeID    string          `json:"scope_id"`
	Unit       *DiagnosticUnit `json:"unit,omitempty"`
	Class      FailureClass    `json:"class"`
	Gate       string          `json:"gate"`
	Evidence   []Evidence      `json:"evidence,omitempty"`
	Children   []FailureScope  `json:"children,omitempty"`
	Blocker    *Blocker        `json:"blocker,omitempty"`
	NextAction string          `json:"next_action"`
}

// Diagnose localizes a gate failure without guessing. A known failed unit is
// returned directly. Otherwise it deterministically partitions the scope so
// the same operation can be repeated on the failing child.
func Diagnose(observation FailureObservation) (Diagnosis, error) {
	if err := validateObservation(observation); err != nil {
		return Diagnosis{}, err
	}
	base := Diagnosis{Schema: DiagnosisSchema, ScopeID: observation.Scope.ID, Class: observation.Class, Gate: observation.Gate, Evidence: observation.Evidence}
	units := append([]DiagnosticUnit(nil), observation.Scope.Units...)
	sort.Slice(units, func(i, j int) bool { return units[i].ID < units[j].ID })

	if observation.FailedUnitID != "" {
		for i := range units {
			if units[i].ID == observation.FailedUnitID {
				base.Kind, base.Unit = DiagnosisLeaf, &units[i]
				base.NextAction = checkAction(observation, units[i].ID)
				return base, nil
			}
		}
		return Diagnosis{}, fmt.Errorf("failed unit %q is outside scope %q", observation.FailedUnitID, observation.Scope.ID)
	}
	if len(units) == 1 {
		base.Kind, base.Unit = DiagnosisLeaf, &units[0]
		base.NextAction = checkAction(observation, units[0].ID)
		return base, nil
	}
	if cycle := dependencyCycle(units); len(cycle) > 0 {
		base.Kind = DiagnosisIrreducible
		base.Blocker = &Blocker{Code: "CYCLIC_DEPENDENCY", Gate: observation.Gate, Detail: strings.Join(cycle, " -> "), MissingDiscriminator: "acyclic dependency boundary", NextAction: "break or explicitly isolate the dependency cycle"}
		base.NextAction = base.Blocker.NextAction
		return base, nil
	}

	children := partitionBy(units, func(unit DiagnosticUnit) string { return unit.Tree })
	if len(children) < 2 {
		children = partitionBy(units, func(unit DiagnosticUnit) string { return unit.TestGroup })
	}
	if len(children) < 2 && allIndependentlyCheckable(units) {
		mid := (len(units) + 1) / 2
		children = [][]DiagnosticUnit{units[:mid], units[mid:]}
	}
	if len(children) < 2 {
		base.Kind = DiagnosisIrreducible
		base.Blocker = &Blocker{Code: "NO_DISCRIMINATOR", Gate: observation.Gate, Detail: "units share every declared boundary and are not independently checkable", MissingDiscriminator: "tree, test group, or independent check", NextAction: "declare a narrower boundary or an independently runnable check"}
		base.NextAction = base.Blocker.NextAction
		return base, nil
	}

	base.Kind = DiagnosisSplit
	for i, group := range children {
		base.Children = append(base.Children, FailureScope{ID: fmt.Sprintf("%s/%d", observation.Scope.ID, i+1), Units: append([]DiagnosticUnit(nil), group...)})
	}
	base.NextAction = "run the gate for each child scope, then diagnose the failing child"
	return base, nil
}

func validateObservation(observation FailureObservation) error {
	if strings.TrimSpace(observation.Scope.ID) == "" {
		return fmt.Errorf("scope id is required")
	}
	if len(observation.Scope.Units) == 0 {
		return fmt.Errorf("scope %q has no units", observation.Scope.ID)
	}
	if strings.TrimSpace(observation.Gate) == "" {
		return fmt.Errorf("gate is required")
	}
	if !oneOf(string(observation.Class), string(FailureCompile), string(FailureTest), string(FailureSafety), string(FailureExternal), string(FailureUnknown)) {
		return fmt.Errorf("unknown failure class %q", observation.Class)
	}
	seen := map[string]bool{}
	for _, unit := range observation.Scope.Units {
		if strings.TrimSpace(unit.ID) == "" {
			return fmt.Errorf("diagnostic unit id is required")
		}
		if seen[unit.ID] {
			return fmt.Errorf("duplicate diagnostic unit %q", unit.ID)
		}
		seen[unit.ID] = true
	}
	return nil
}

func partitionBy(units []DiagnosticUnit, key func(DiagnosticUnit) string) [][]DiagnosticUnit {
	groups := map[string][]DiagnosticUnit{}
	for _, unit := range units {
		value := strings.TrimSpace(key(unit))
		if value == "" {
			return nil
		}
		groups[value] = append(groups[value], unit)
	}
	if len(groups) < 2 {
		return nil
	}
	keys := make([]string, 0, len(groups))
	for value := range groups {
		keys = append(keys, value)
	}
	sort.Strings(keys)
	out := make([][]DiagnosticUnit, 0, len(keys))
	for _, value := range keys {
		out = append(out, groups[value])
	}
	return out
}

func allIndependentlyCheckable(units []DiagnosticUnit) bool {
	for _, unit := range units {
		if !unit.IndependentlyCheckable {
			return false
		}
	}
	return true
}

func dependencyCycle(units []DiagnosticUnit) []string {
	present := map[string]bool{}
	for _, unit := range units {
		present[unit.ID] = true
	}
	edges := map[string][]string{}
	for _, unit := range units {
		for _, dependency := range unit.Dependencies {
			if present[dependency] {
				edges[unit.ID] = append(edges[unit.ID], dependency)
			}
		}
		sort.Strings(edges[unit.ID])
	}
	state, stack := map[string]int{}, []string{}
	var visit func(string) []string
	visit = func(id string) []string {
		state[id], stack = 1, append(stack, id)
		for _, dependency := range edges[id] {
			if state[dependency] == 0 {
				if cycle := visit(dependency); cycle != nil {
					return cycle
				}
			}
			if state[dependency] == 1 {
				for i, member := range stack {
					if member == dependency {
						return append(append([]string(nil), stack[i:]...), dependency)
					}
				}
			}
		}
		stack, state[id] = stack[:len(stack)-1], 2
		return nil
	}
	ids := make([]string, 0, len(units))
	for _, unit := range units {
		ids = append(ids, unit.ID)
	}
	sort.Strings(ids)
	for _, id := range ids {
		if state[id] == 0 {
			if cycle := visit(id); cycle != nil {
				return cycle
			}
		}
	}
	return nil
}

func checkAction(observation FailureObservation, unitID string) string {
	if observation.CheckCommand != "" {
		return strings.ReplaceAll(observation.CheckCommand, "{unit}", unitID)
	}
	return fmt.Sprintf("run gate %s for unit %s", observation.Gate, unitID)
}
