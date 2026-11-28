// Adapted from ai-dynamo/aiperf
// modified from the original): declarative RECORD/AGGREGATE/DERIVED metric
// registry with startup dependency-shape validation, cycle detection, and
// deterministic topological compute ordering.
package perfledger

import (
	"fmt"
	"sort"
	"strings"
)

// MetricKind classifies when a metric is evaluated in the pipeline.
type MetricKind string

const (
	MetricRecord    MetricKind = "RECORD"
	MetricAggregate MetricKind = "AGGREGATE"
	MetricDerived   MetricKind = "DERIVED"
)

// MetricSpec declaratively describes a metric and its compute dependencies.
// OptionalMetrics add topological ordering edges when registered, but may
// legitimately be absent.
type MetricSpec struct {
	Tag                string     `json:"tag"`
	Kind               MetricKind `json:"kind"`
	Header             string     `json:"header,omitempty"`
	ShortHeader        string     `json:"short_header,omitempty"`
	Unit               string     `json:"unit,omitempty"`
	DisplayUnit        string     `json:"display_unit,omitempty"`
	DisplayOrder       int        `json:"display_order,omitempty"`
	Flags              []string   `json:"flags,omitempty"`
	RequiredMetrics    []string   `json:"required_metrics,omitempty"`
	OptionalMetrics    []string   `json:"optional_metrics,omitempty"`
	TimesliceDerivable bool       `json:"timeslice_derivable,omitempty"`
}

// MetricRegistry stores unique tag-keyed metric declarations and guarantees a
// valid, acyclic topological compute order.
type MetricRegistry struct {
	specs map[string]MetricSpec
	order []MetricSpec
	dirty bool
}

// NewMetricRegistry registers specs and validates dependencies and compute
// order eagerly at startup.
func NewMetricRegistry(specs ...MetricSpec) (*MetricRegistry, error) {
	r := &MetricRegistry{
		specs: make(map[string]MetricSpec, len(specs)),
	}
	for _, spec := range specs {
		if err := r.Register(spec); err != nil {
			return nil, err
		}
	}
	if _, err := r.Validate(); err != nil {
		return nil, err
	}
	return r, nil
}

// Register adds one metric declaration, rejecting empty tags, unknown kinds,
// and duplicate registrations.
func (r *MetricRegistry) Register(spec MetricSpec) error {
	if r == nil {
		return fmt.Errorf("perfledger: MetricRegistry.Register: nil registry")
	}
	if r.specs == nil {
		r.specs = map[string]MetricSpec{}
	}
	if strings.TrimSpace(spec.Tag) == "" {
		return fmt.Errorf("perfledger: MetricRegistry: metric tag must be non-empty")
	}
	if !validMetricKind(spec.Kind) {
		return fmt.Errorf("perfledger: MetricRegistry: metric %q has unknown kind %q", spec.Tag, spec.Kind)
	}
	if _, exists := r.specs[spec.Tag]; exists {
		return fmt.Errorf("perfledger: MetricRegistry: duplicate metric tag %q", spec.Tag)
	}
	spec.Flags = append([]string(nil), spec.Flags...)
	spec.RequiredMetrics = append([]string(nil), spec.RequiredMetrics...)
	spec.OptionalMetrics = append([]string(nil), spec.OptionalMetrics...)
	r.specs[spec.Tag] = spec
	r.dirty = true
	return nil
}

// Validate enforces required-dependency existence, dependency-shape rules
// (RECORD->RECORD only, AGGREGATE->RECORD|AGGREGATE, DERIVED->any), and
// acyclicity, returning the deterministic topological compute order.
func (r *MetricRegistry) Validate() ([]MetricSpec, error) {
	if r == nil {
		return nil, fmt.Errorf("perfledger: MetricRegistry.Validate: nil registry")
	}
	tags := make([]string, 0, len(r.specs))
	for tag := range r.specs {
		tags = append(tags, tag)
	}
	sort.Strings(tags)

	// dep -> list of dependents; indegree[tag] = number of distinct dependencies.
	dependents := make(map[string][]string, len(tags))
	indegree := make(map[string]int, len(tags))
	for _, tag := range tags {
		indegree[tag] = 0
	}

	for _, tag := range tags {
		spec := r.specs[tag]
		seenDeps := make(map[string]struct{}, len(spec.RequiredMetrics)+len(spec.OptionalMetrics))

		checkDep := func(depTag string, required bool) error {
			if depTag == tag {
				return fmt.Errorf("perfledger: MetricRegistry: metric %q cannot depend on itself", tag)
			}
			dep, ok := r.specs[depTag]
			if !ok {
				if required {
					return fmt.Errorf("perfledger: MetricRegistry: metric %q requires unregistered dependency %q", tag, depTag)
				}
				return nil // optional dependency absent: no edge
			}
			if !dependencyShapeAllowed(spec.Kind, dep.Kind) {
				return fmt.Errorf("perfledger: MetricRegistry: %s metric %q cannot depend on %s metric %q", spec.Kind, tag, dep.Kind, depTag)
			}
			if _, dup := seenDeps[depTag]; !dup {
				seenDeps[depTag] = struct{}{}
				dependents[depTag] = append(dependents[depTag], tag)
				indegree[tag]++
			}
			return nil
		}

		for _, depTag := range spec.RequiredMetrics {
			if err := checkDep(depTag, true); err != nil {
				return nil, err
			}
		}
		for _, depTag := range spec.OptionalMetrics {
			if err := checkDep(depTag, false); err != nil {
				return nil, err
			}
		}
	}

	var ready []string
	for _, tag := range tags {
		if indegree[tag] == 0 {
			ready = append(ready, tag)
		}
	}
	sort.Strings(ready)

	order := make([]MetricSpec, 0, len(tags))
	for len(ready) > 0 {
		cur := ready[0]
		ready = ready[1:]
		order = append(order, r.specs[cur])
		for _, next := range dependents[cur] {
			indegree[next]--
			if indegree[next] == 0 {
				ready = append(ready, next)
			}
		}
		sort.Strings(ready)
	}

	if len(order) != len(tags) {
		var stuck []string
		for _, tag := range tags {
			if indegree[tag] > 0 {
				stuck = append(stuck, tag)
			}
		}
		return nil, fmt.Errorf("perfledger: MetricRegistry: dependency cycle detected involving [%s]", strings.Join(stuck, ", "))
	}

	r.order = order
	r.dirty = false
	return r.ComputeOrder()
}

// ComputeOrder returns the validated topological compute order (dependencies
// before dependents).
func (r *MetricRegistry) ComputeOrder() ([]MetricSpec, error) {
	if r == nil {
		return nil, fmt.Errorf("perfledger: MetricRegistry.ComputeOrder: nil registry")
	}
	if r.dirty {
		return r.Validate()
	}
	out := make([]MetricSpec, len(r.order))
	copy(out, r.order)
	return out, nil
}

// ComputeOrderTags returns just the tag names in topological compute order.
func (r *MetricRegistry) ComputeOrderTags() ([]string, error) {
	order, err := r.ComputeOrder()
	if err != nil {
		return nil, err
	}
	out := make([]string, len(order))
	for i, spec := range order {
		out[i] = spec.Tag
	}
	return out, nil
}

// Lookup returns a copy of the registered spec for tag.
func (r *MetricRegistry) Lookup(tag string) (MetricSpec, bool) {
	if r == nil {
		return MetricSpec{}, false
	}
	spec, ok := r.specs[tag]
	return spec, ok
}

func validMetricKind(k MetricKind) bool {
	return k == MetricRecord || k == MetricAggregate || k == MetricDerived
}

func dependencyShapeAllowed(parentKind, depKind MetricKind) bool {
	switch parentKind {
	case MetricRecord:
		return depKind == MetricRecord
	case MetricAggregate:
		return depKind == MetricRecord || depKind == MetricAggregate
	case MetricDerived:
		return validMetricKind(depKind)
	default:
		return false
	}
}
