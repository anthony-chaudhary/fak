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

	selected := map[string]bool{}
	paths := map[string][]string{}
	queue := append([]string(nil), receipt.Roots...)
	for _, root := range receipt.Roots {
		paths[root] = []string{root}
	}
	for len(queue) > 0 {
		id := queue[0]
		queue = queue[1:]
		if selected[id] {
			continue
		}
		component, ok := byID[id]
		if !ok {
			return refused(receipt, "MISSING_ROOT", id, paths[id], Evidence{}, availableRemedies(id, byCapability)), nil
		}
		selected[id] = true
		for _, relation := range component.Relations {
			switch relation.Kind {
			case Requires:
				chosen, substitute := choose(relation, byCapability)
				if chosen == "" {
					chain := append(append([]string(nil), paths[id]...), relation.Target)
					return refused(receipt, "UNSATISFIED_REQUIREMENT", relation.Target, chain, relation.Evidence, availableRemedies(relation.Target, byCapability)), nil
				}
				receipt.Decisions = append(receipt.Decisions, Decision{From: id, Relation: Requires, Wanted: relation.Target, Chosen: chosen, Substitute: substitute, Evidence: relation.Evidence})
				if _, seen := paths[chosen]; !seen {
					paths[chosen] = append(append([]string(nil), paths[id]...), chosen)
				}
				queue = append(queue, chosen)
			case Optional:
				chosen, substitute := choose(relation, byCapability)
				if chosen == "" {
					receipt.Warnings = append(receipt.Warnings, Warning{Code: "OPTIONAL_UNAVAILABLE", From: id, Wanted: relation.Target, Message: "optional component is unavailable", Evidence: relation.Evidence})
					continue
				}
				// Availability is recorded, but optional components enter the closure only when requested as roots.
				receipt.Decisions = append(receipt.Decisions, Decision{From: id, Relation: Optional, Wanted: relation.Target, Chosen: chosen, Substitute: substitute, Evidence: relation.Evidence})
			case Recommends:
				chosen, substitute := choose(relation, byCapability)
				if chosen == "" {
					receipt.Warnings = append(receipt.Warnings, Warning{Code: "RECOMMENDATION_UNMET", From: id, Wanted: relation.Target, Message: "recommendation is not a launch requirement", Evidence: relation.Evidence})
					continue
				}
				// Recommendations inform operators without expanding the mandatory closure.
				receipt.Decisions = append(receipt.Decisions, Decision{From: id, Relation: Recommends, Wanted: relation.Target, Chosen: chosen, Substitute: substitute, Evidence: relation.Evidence})
			case Conflicts:
				// Evaluated after closure so relation order cannot change the result.
			default:
				return Receipt{}, fmt.Errorf("component %s: unsupported relation %q", id, relation.Kind)
			}
		}
		sort.Strings(queue)
	}

	selectedIDs := make([]string, 0, len(selected))
	for id := range selected {
		selectedIDs = append(selectedIDs, id)
	}
	sort.Strings(selectedIDs)
	for _, id := range selectedIDs {
		component := byID[id]
		for _, relation := range component.Relations {
			if relation.Kind != Conflicts {
				continue
			}
			other := firstSelected(byCapability[relation.Target], selected)
			if other == "" {
				continue
			}
			chain := append(append([]string(nil), paths[id]...), "conflicts:"+other)
			return refused(receipt, "COMPONENT_CONFLICT", relation.Target, chain, relation.Evidence, []string{"remove " + id, "remove " + other}), nil
		}
	}

	for _, id := range selectedIDs {
		receipt.Selected = append(receipt.Selected, byID[id])
	}
	sortDecisions(receipt.Decisions)
	sortWarnings(receipt.Warnings)
	return receipt, nil
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

func choose(r Relation, providers map[string][]string) (string, bool) {
	if ids := providers[r.Target]; len(ids) > 0 {
		return ids[0], false
	}
	for _, substitute := range r.Substitutes {
		if ids := providers[substitute]; len(ids) > 0 {
			return ids[0], true
		}
	}
	return "", false
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
