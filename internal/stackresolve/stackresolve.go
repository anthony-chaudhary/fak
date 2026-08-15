// Package stackresolve resolves a versioned component request into a deterministic
// stack receipt. Domain packages retain ownership of component facts and expose
// them through Provider; this package owns only graph semantics and explanation.
package stackresolve

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
)

const (
	ManifestSchema = "fak-stack-manifest/1"
	ReceiptSchema  = "fak-stack-receipt/1"
)

// RelationKind is the deliberately small composition algebra supported by the spine.
type RelationKind string

const (
	Requires   RelationKind = "requires"
	Recommends RelationKind = "recommends"
	Optional   RelationKind = "optional"
	Conflicts  RelationKind = "conflicts"
)

// Evidence identifies the authority behind a component or relation claim.
type Evidence struct {
	Authority string `json:"authority"`
	Source    string `json:"source"`
	Tier      string `json:"tier,omitempty"`
	Freshness string `json:"freshness,omitempty"`
}

// Relation points to a component ID or provided capability. Substitutes are
// considered in their declared order only when Target has no available provider.
type Relation struct {
	Kind        RelationKind `json:"kind"`
	Target      string       `json:"target"`
	Substitutes []string     `json:"substitutes,omitempty"`
	Evidence    Evidence     `json:"evidence"`
}

// Component is a domain-owned, versioned unit that can provide capabilities and constraints.
type Component struct {
	ID        string     `json:"id"`
	Kind      string     `json:"kind"`
	Version   string     `json:"version"`
	Provides  []string   `json:"provides,omitempty"`
	Relations []Relation `json:"relations,omitempty"`
	Evidence  Evidence   `json:"evidence"`
}

// Manifest names the requested roots. Catalog is accepted through ManifestProvider
// by the demo; production domain owners can implement Provider directly.
type Manifest struct {
	Schema     string      `json:"schema"`
	Workload   string      `json:"workload"`
	Roots      []string    `json:"roots"`
	Components []Component `json:"components"`
}

// Provider is the ownership boundary: domain validators return normalized facts
// while the resolver remains independent of harness, model, policy, and hardware packages.
type Provider interface {
	Name() string
	Components(context.Context) ([]Component, error)
}

// ManifestProvider adapts a static manifest catalog to Provider.
type ManifestProvider struct{ Manifest Manifest }

func (p ManifestProvider) Name() string { return "manifest" }
func (p ManifestProvider) Components(context.Context) ([]Component, error) {
	return append([]Component(nil), p.Manifest.Components...), nil
}

// Decision records one decisive relation and its provenance.
type Decision struct {
	From       string       `json:"from"`
	Relation   RelationKind `json:"relation"`
	Wanted     string       `json:"wanted"`
	Chosen     string       `json:"chosen,omitempty"`
	Substitute bool         `json:"substitute,omitempty"`
	Evidence   Evidence     `json:"evidence"`
}

// Conflict is an actionable unsatisfied core. Chain is root-to-blocker and is
// inclusion-minimal for the single decisive requirement selected by this resolver.
type Conflict struct {
	Code        string   `json:"code"`
	Wanted      string   `json:"wanted"`
	Chain       []string `json:"chain"`
	Remediation []string `json:"remediation"`
	Evidence    Evidence `json:"evidence"`
}

// Warning reports a recommendation or optional edge without turning it into a hard gate.
type Warning struct {
	Code     string   `json:"code"`
	From     string   `json:"from"`
	Wanted   string   `json:"wanted"`
	Message  string   `json:"message"`
	Evidence Evidence `json:"evidence"`
}

// Receipt binds the request to normalized selections and decisive evidence.
type Receipt struct {
	Schema    string      `json:"schema"`
	Status    string      `json:"status"`
	Workload  string      `json:"workload"`
	Roots     []string    `json:"roots"`
	Selected  []Component `json:"selected,omitempty"`
	Decisions []Decision  `json:"decisions,omitempty"`
	Warnings  []Warning   `json:"warnings,omitempty"`
	Conflict  *Conflict   `json:"conflict,omitempty"`
}

// Parse validates and decodes a manifest.
func Parse(raw []byte) (Manifest, error) {
	var m Manifest
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&m); err != nil {
		return Manifest{}, fmt.Errorf("decode manifest: %w", err)
	}
	if m.Schema != ManifestSchema {
		return Manifest{}, fmt.Errorf("schema %q, want %q", m.Schema, ManifestSchema)
	}
	if strings.TrimSpace(m.Workload) == "" {
		return Manifest{}, errors.New("workload is required")
	}
	if len(m.Roots) == 0 {
		return Manifest{}, errors.New("at least one root is required")
	}
	return m, nil
}

// Resolve composes facts from independent providers and emits an allow or refuse receipt.
func Resolve(ctx context.Context, workload string, roots []string, providers ...Provider) (Receipt, error) {
	receipt := Receipt{Schema: ReceiptSchema, Status: "allow", Workload: workload, Roots: sortedUnique(roots)}
	if strings.TrimSpace(workload) == "" || len(receipt.Roots) == 0 {
		return Receipt{}, errors.New("workload and at least one root are required")
	}

	byID := map[string]Component{}
	byCapability := map[string][]string{}
	for _, provider := range providers {
		components, err := provider.Components(ctx)
		if err != nil {
			return Receipt{}, fmt.Errorf("provider %s: %w", provider.Name(), err)
		}
		for _, component := range components {
			if err := validateComponent(component, provider.Name()); err != nil {
				return Receipt{}, err
			}
			if _, exists := byID[component.ID]; exists {
				return Receipt{}, fmt.Errorf("duplicate component id %q", component.ID)
			}
			sort.Strings(component.Provides)
			sort.SliceStable(component.Relations, func(i, j int) bool {
				if component.Relations[i].Kind != component.Relations[j].Kind {
					return component.Relations[i].Kind < component.Relations[j].Kind
				}
				return component.Relations[i].Target < component.Relations[j].Target
			})
			byID[component.ID] = component
			byCapability[component.ID] = append(byCapability[component.ID], component.ID)
			for _, capability := range component.Provides {
				byCapability[capability] = append(byCapability[capability], component.ID)
			}
		}
	}
	for capability := range byCapability {
		byCapability[capability] = sortedUnique(byCapability[capability])
	}

	initial := resolveState{
		receipt:  receipt,
		selected: map[string]bool{},
		paths:    map[string][]string{},
	}
	for _, root := range receipt.Roots {
		initial.paths[root] = []string{root}
	}
	result, _ := resolveQueue(initial, append([]string(nil), receipt.Roots...), byID, byCapability)
	return finish(result, byID), nil
}

type resolveState struct {
	receipt  Receipt
	selected map[string]bool
	paths    map[string][]string
}

// resolveQueue searches only dependency choices reachable from the requested roots.
// Candidate order is stable, so the first satisfiable receipt is reproducible without
// eagerly enumerating the unrelated cross-product of the whole catalog.
func resolveQueue(state resolveState, queue []string, byID map[string]Component, byCapability map[string][]string) (resolveState, bool) {
	if len(queue) == 0 {
		return checkConflicts(state, byID, byCapability)
	}
	id := queue[0]
	queue = append([]string(nil), queue[1:]...)
	if state.selected[id] {
		return resolveQueue(state, queue, byID, byCapability)
	}
	component, ok := byID[id]
	if !ok {
		state.receipt = refused(state.receipt, "MISSING_ROOT", id, state.paths[id], Evidence{}, availableRemedies(id, byCapability))
		return state, false
	}
	state = cloneState(state)
	state.selected[id] = true
	return resolveRelations(state, component, 0, queue, byID, byCapability)
}

func resolveRelations(state resolveState, component Component, index int, queue []string, byID map[string]Component, byCapability map[string][]string) (resolveState, bool) {
	if index == len(component.Relations) {
		sort.Strings(queue)
		return resolveQueue(state, queue, byID, byCapability)
	}
	relation := component.Relations[index]
	next := func(candidate string, substitute bool) (resolveState, bool) {
		branch := cloneState(state)
		branch.receipt.Decisions = append(branch.receipt.Decisions, Decision{From: component.ID, Relation: relation.Kind, Wanted: relation.Target, Chosen: candidate, Substitute: substitute, Evidence: relation.Evidence})
		if _, seen := branch.paths[candidate]; !seen {
			branch.paths[candidate] = append(append([]string(nil), branch.paths[component.ID]...), candidate)
		}
		return resolveRelations(branch, component, index+1, append(append([]string(nil), queue...), candidate), byID, byCapability)
	}

	switch relation.Kind {
	case Requires:
		candidates := relationCandidates(relation, byCapability)
		if len(candidates) == 0 {
			state.receipt = refused(state.receipt, "UNSATISFIED_REQUIREMENT", relation.Target, append(append([]string(nil), state.paths[component.ID]...), relation.Target), relation.Evidence, availableRemedies(relation.Target, byCapability))
			return state, false
		}
		var firstFailure resolveState
		for i, candidate := range candidates {
			result, ok := next(candidate.id, candidate.substitute)
			if ok {
				return result, true
			}
			if i == 0 {
				firstFailure = result
			}
		}
		return firstFailure, false
	case Optional:
		candidates := relationCandidates(relation, byCapability)
		if len(candidates) == 0 {
			state = cloneState(state)
			state.receipt.Warnings = append(state.receipt.Warnings, Warning{Code: "OPTIONAL_UNAVAILABLE", From: component.ID, Wanted: relation.Target, Message: "optional component is unavailable", Evidence: relation.Evidence})
			return resolveRelations(state, component, index+1, queue, byID, byCapability)
		}
		// Availability is recorded, but optional components enter the closure only when requested as roots.
		state = cloneState(state)
		state.receipt.Decisions = append(state.receipt.Decisions, Decision{From: component.ID, Relation: Optional, Wanted: relation.Target, Chosen: candidates[0].id, Substitute: candidates[0].substitute, Evidence: relation.Evidence})
		return resolveRelations(state, component, index+1, queue, byID, byCapability)
	case Recommends:
		candidates := relationCandidates(relation, byCapability)
		state = cloneState(state)
		if len(candidates) == 0 {
			state.receipt.Warnings = append(state.receipt.Warnings, Warning{Code: "RECOMMENDATION_UNMET", From: component.ID, Wanted: relation.Target, Message: "recommendation is not a launch requirement", Evidence: relation.Evidence})
		} else {
			// Recommendations inform operators without expanding the mandatory closure.
			state.receipt.Decisions = append(state.receipt.Decisions, Decision{From: component.ID, Relation: Recommends, Wanted: relation.Target, Chosen: candidates[0].id, Substitute: candidates[0].substitute, Evidence: relation.Evidence})
		}
		return resolveRelations(state, component, index+1, queue, byID, byCapability)
	case Conflicts:
		return resolveRelations(state, component, index+1, queue, byID, byCapability)
	default:
		state.receipt = Receipt{}
		state.receipt.Conflict = &Conflict{Code: "INVALID_RELATION", Wanted: string(relation.Kind)}
		return state, false
	}
}

type candidate struct {
	id         string
	substitute bool
}

func relationCandidates(relation Relation, byCapability map[string][]string) []candidate {
	var out []candidate
	seen := map[string]bool{}
	add := func(capability string, substitute bool) {
		for _, id := range byCapability[capability] {
			if !seen[id] {
				seen[id] = true
				out = append(out, candidate{id: id, substitute: substitute})
			}
		}
	}
	add(relation.Target, false)
	for _, substitute := range relation.Substitutes {
		add(substitute, true)
	}
	return out
}

func checkConflicts(state resolveState, byID map[string]Component, byCapability map[string][]string) (resolveState, bool) {
	ids := sortedSelectedIDs(state.selected)
	for _, id := range ids {
		for _, relation := range byID[id].Relations {
			if relation.Kind != Conflicts {
				continue
			}
			other := firstSelected(byCapability[relation.Target], state.selected)
			if other == "" {
				continue
			}
			state.receipt = refused(state.receipt, "COMPONENT_CONFLICT", relation.Target, append(append([]string(nil), state.paths[id]...), "conflicts:"+other), relation.Evidence, []string{"remove " + id, "remove " + other})
			return state, false
		}
	}
	return state, true
}

func finish(state resolveState, byID map[string]Component) Receipt {
	if state.receipt.Status == "allow" {
		for _, id := range sortedSelectedIDs(state.selected) {
			state.receipt.Selected = append(state.receipt.Selected, byID[id])
		}
	}
	sortDecisions(state.receipt.Decisions)
	sortWarnings(state.receipt.Warnings)
	return state.receipt
}

func cloneState(in resolveState) resolveState {
	out := in
	out.receipt.Roots = append([]string(nil), in.receipt.Roots...)
	out.receipt.Selected = append([]Component(nil), in.receipt.Selected...)
	out.receipt.Decisions = append([]Decision(nil), in.receipt.Decisions...)
	out.receipt.Warnings = append([]Warning(nil), in.receipt.Warnings...)
	out.selected = make(map[string]bool, len(in.selected))
	for id, value := range in.selected {
		out.selected[id] = value
	}
	out.paths = make(map[string][]string, len(in.paths))
	for id, path := range in.paths {
		out.paths[id] = append([]string(nil), path...)
	}
	return out
}

func sortedSelectedIDs(selected map[string]bool) []string {
	ids := make([]string, 0, len(selected))
	for id := range selected {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

func validateComponent(c Component, provider string) error {
	if c.ID == "" || c.Kind == "" || c.Version == "" {
		return fmt.Errorf("provider %s: component id, kind, and version are required", provider)
	}
	if c.Evidence.Authority == "" || c.Evidence.Source == "" {
		return fmt.Errorf("component %s: evidence authority and source are required", c.ID)
	}
	for _, relation := range c.Relations {
		if relation.Target == "" || relation.Evidence.Authority == "" || relation.Evidence.Source == "" {
			return fmt.Errorf("component %s: relation target and evidence are required", c.ID)
		}
	}
	return nil
}

func firstSelected(ids []string, selected map[string]bool) string {
	for _, id := range ids {
		if selected[id] {
			return id
		}
	}
	return ""
}

func refused(base Receipt, code, wanted string, chain []string, evidence Evidence, remediation []string) Receipt {
	base.Status = "refuse"
	base.Conflict = &Conflict{Code: code, Wanted: wanted, Chain: chain, Remediation: remediation, Evidence: evidence}
	sortDecisions(base.Decisions)
	sortWarnings(base.Warnings)
	return base
}

func availableRemedies(wanted string, providers map[string][]string) []string {
	if ids := providers[wanted]; len(ids) > 0 {
		return []string{"select " + ids[0]}
	}
	return []string{"add a provider for " + wanted, "collect or refresh evidence for " + wanted}
}

func sortedUnique(in []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(in))
	for _, item := range in {
		if item != "" && !seen[item] {
			seen[item] = true
			out = append(out, item)
		}
	}
	sort.Strings(out)
	return out
}

func sortDecisions(in []Decision) {
	sort.SliceStable(in, func(i, j int) bool {
		if in[i].From != in[j].From {
			return in[i].From < in[j].From
		}
		if in[i].Relation != in[j].Relation {
			return in[i].Relation < in[j].Relation
		}
		return in[i].Wanted < in[j].Wanted
	})
}

func sortWarnings(in []Warning) {
	sort.SliceStable(in, func(i, j int) bool {
		if in[i].From != in[j].From {
			return in[i].From < in[j].From
		}
		return in[i].Wanted < in[j].Wanted
	})
}
