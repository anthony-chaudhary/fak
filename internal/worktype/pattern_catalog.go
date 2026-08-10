package worktype

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
)

// PatternCatalogSchema is the stable machine contract for coding-workload patterns.
const PatternCatalogSchema = "fak.workpattern-catalog/1"

// Provenance distinguishes established terminology from local synthesis.
type Provenance string

const (
	ProvenanceBorrowed  Provenance = "borrowed"
	ProvenanceAdapted   Provenance = "adapted"
	ProvenanceSynthesis Provenance = "new-synthesis"
)

// Pattern describes a goal-shaped coding workload. Subpatterns are reusable moves
// that may compose into more than one Pattern.
type Pattern struct {
	ID          string     `json:"id"`
	Name        string     `json:"name"`
	Intent      string     `json:"intent"`
	Axes        []string   `json:"axes"`
	Aliases     []string   `json:"aliases,omitempty"`
	IncludeWhen string     `json:"include_when"`
	ExcludeWhen string     `json:"exclude_when"`
	Provenance  Provenance `json:"provenance"`
	Evidence    []string   `json:"evidence"`
	Subpatterns []string   `json:"subpatterns"`
}

type Subpattern struct {
	ID          string     `json:"id"`
	Name        string     `json:"name"`
	Intent      string     `json:"intent"`
	Axes        []string   `json:"axes"`
	Aliases     []string   `json:"aliases,omitempty"`
	IncludeWhen string     `json:"include_when"`
	ExcludeWhen string     `json:"exclude_when"`
	Provenance  Provenance `json:"provenance"`
	Evidence    []string   `json:"evidence"`
	Composes    []string   `json:"composes,omitempty"`
}

type PatternCatalog struct {
	Schema      string       `json:"schema"`
	Version     string       `json:"version"`
	Patterns    []Pattern    `json:"patterns"`
	Subpatterns []Subpattern `json:"subpatterns"`
}

var validAxes = map[string]bool{"workload-shape": true, "orchestration-topology": true, "verification-strategy": true, "failure-mode": true}
var validProvenance = map[Provenance]bool{ProvenanceBorrowed: true, ProvenanceAdapted: true, ProvenanceSynthesis: true}

func (c PatternCatalog) Validate() error {
	if c.Schema != PatternCatalogSchema {
		return fmt.Errorf("catalog schema %q, want %q", c.Schema, PatternCatalogSchema)
	}
	if strings.TrimSpace(c.Version) == "" {
		return errors.New("catalog version is required")
	}
	ids := map[string]string{}
	aliases := map[string]string{}
	add := func(id, name, intent, inc, exc string, p Provenance, axes, evidence, als []string, kind string) error {
		if id == "" || name == "" || intent == "" || inc == "" || exc == "" {
			return fmt.Errorf("%s %q: id, name, intent, inclusion, and exclusion are required", kind, id)
		}
		if old := ids[id]; old != "" {
			return fmt.Errorf("duplicate id %q (%s and %s)", id, old, kind)
		}
		ids[id] = kind
		if !validProvenance[p] {
			return fmt.Errorf("%s %q: unknown provenance %q", kind, id, p)
		}
		if len(axes) == 0 || len(evidence) == 0 {
			return fmt.Errorf("%s %q: axes and evidence are required", kind, id)
		}
		for _, a := range axes {
			if !validAxes[a] {
				return fmt.Errorf("%s %q: unknown axis %q", kind, id, a)
			}
		}
		names := append([]string{name}, als...)
		for _, a := range names {
			k := strings.ToLower(strings.TrimSpace(a))
			if k == "" {
				return fmt.Errorf("%s %q: empty alias", kind, id)
			}
			if old := aliases[k]; old != "" && old != id {
				return fmt.Errorf("alias %q is ambiguous between %s and %s", a, old, id)
			}
			aliases[k] = id
		}
		return nil
	}
	for _, p := range c.Patterns {
		if err := add(p.ID, p.Name, p.Intent, p.IncludeWhen, p.ExcludeWhen, p.Provenance, p.Axes, p.Evidence, p.Aliases, "pattern"); err != nil {
			return err
		}
	}
	for _, s := range c.Subpatterns {
		if err := add(s.ID, s.Name, s.Intent, s.IncludeWhen, s.ExcludeWhen, s.Provenance, s.Axes, s.Evidence, s.Aliases, "subpattern"); err != nil {
			return err
		}
	}
	sub := map[string]bool{}
	pat := map[string]bool{}
	for _, p := range c.Patterns {
		pat[p.ID] = true
	}
	for _, s := range c.Subpatterns {
		sub[s.ID] = true
	}
	for _, p := range c.Patterns {
		for _, id := range p.Subpatterns {
			if !sub[id] {
				return fmt.Errorf("pattern %q: dangling subpattern %q", p.ID, id)
			}
		}
	}
	graph := map[string][]string{}
	for _, s := range c.Subpatterns {
		for _, id := range s.Composes {
			if !sub[id] {
				return fmt.Errorf("subpattern %q: dangling composition %q", s.ID, id)
			}
			graph[s.ID] = append(graph[s.ID], id)
		}
	}
	state := map[string]byte{}
	var visit func(string) error
	visit = func(id string) error {
		if state[id] == 1 {
			return fmt.Errorf("composition cycle at %q", id)
		}
		if state[id] == 2 {
			return nil
		}
		state[id] = 1
		for _, to := range graph[id] {
			if err := visit(to); err != nil {
				return err
			}
		}
		state[id] = 2
		return nil
	}
	for id := range sub {
		if err := visit(id); err != nil {
			return err
		}
	}
	_ = pat
	return nil
}

func (c PatternCatalog) DeterministicJSON() ([]byte, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	out := c
	out.Patterns = append([]Pattern(nil), c.Patterns...)
	out.Subpatterns = append([]Subpattern(nil), c.Subpatterns...)
	sort.Slice(out.Patterns, func(i, j int) bool { return out.Patterns[i].ID < out.Patterns[j].ID })
	sort.Slice(out.Subpatterns, func(i, j int) bool { return out.Subpatterns[i].ID < out.Subpatterns[j].ID })
	for i := range out.Patterns {
		sort.Strings(out.Patterns[i].Axes)
		sort.Strings(out.Patterns[i].Aliases)
		sort.Strings(out.Patterns[i].Evidence)
		sort.Strings(out.Patterns[i].Subpatterns)
	}
	for i := range out.Subpatterns {
		sort.Strings(out.Subpatterns[i].Axes)
		sort.Strings(out.Subpatterns[i].Aliases)
		sort.Strings(out.Subpatterns[i].Evidence)
		sort.Strings(out.Subpatterns[i].Composes)
	}
	return json.MarshalIndent(out, "", "  ")
}

func ParsePatternCatalog(b []byte) (PatternCatalog, error) {
	var c PatternCatalog
	dec := json.NewDecoder(strings.NewReader(string(b)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		return c, err
	}
	return c, c.Validate()
}

func SeedPatternCatalog() PatternCatalog {
	ev := []string{"docs/research/coding-workload-vocabulary.md#candidate-patterns"}
	ax := []string{"workload-shape"}
	mkP := func(id, name, intent string, subs ...string) Pattern {
		return Pattern{ID: id, Name: name, Intent: intent, Axes: ax, IncludeWhen: "The stated goal and done artifact match this workload shape.", ExcludeWhen: "The goal is only a mechanism, orchestration topology, or failure label.", Provenance: ProvenanceAdapted, Evidence: ev, Subpatterns: subs}
	}
	mkS := func(id, name, intent string, p Provenance) Subpattern {
		return Subpattern{ID: id, Name: name, Intent: intent, Axes: []string{"orchestration-topology", "verification-strategy"}, IncludeWhen: "The ordered evidence contains every named move.", ExcludeWhen: "Only names or comments mention the move, or an ordered step is absent.", Provenance: p, Evidence: []string{"docs/research/coding-workload-vocabulary.md#candidate-subpatterns"}}
	}
	subs := []Subpattern{
		mkS("sp.inspect-edit-verify", "Inspect–Edit–Verify", "Inspect evidence, change the target, then run an independent check.", ProvenanceSynthesis),
		mkS("sp.lease-isolate-land", "Lease–Isolate–Land", "Acquire ownership, isolate execution, and serialize landing.", ProvenanceSynthesis),
		mkS("sp.reproduce-fix-witness", "Reproduce–Fix–Witness", "Capture the defect, repair it, and retain a checkable artifact.", ProvenanceAdapted),
		mkS("sp.plan-fanout-reconcile", "Plan–Fanout–Reconcile", "Partition work, execute independent leaves, then witness and fold.", ProvenanceAdapted),
		mkS("sp.read-edit-verify", "Read–Edit–Verify", "Read context before editing and verifying.", ProvenanceAdapted),
		mkS("sp.search-fanout", "Search Fanout", "Run multiple independent searches before synthesis.", ProvenanceSynthesis),
		mkS("sp.retry-after-refusal", "Retry After Refusal", "Adapt a denied tool call and retry through an allowed seam.", ProvenanceSynthesis),
		mkS("sp.redundant-replay", "Redundant Replay", "Repeat an equivalent call without intervening state change.", ProvenanceSynthesis),
		mkS("sp.blind-edit-run", "Blind Edit–Run", "Edit and execute without prior inspection.", ProvenanceSynthesis),
		mkS("sp.cache-warm-replay", "Cache-Warm Replay", "Repeat a call after a cache warm or reusable result.", ProvenanceSynthesis),
		mkS("sp.reproduce-first", "Reproduce First", "Build a deterministic failing check before editing.", ProvenanceBorrowed),
		mkS("sp.independent-adjudication", "Independent Adjudication", "Route a claim through a witness not authored by its worker.", ProvenanceAdapted),
	}
	return PatternCatalog{Schema: PatternCatalogSchema, Version: "1.0.0", Patterns: []Pattern{
		mkP("wp.issue-to-patch", "Issue-to-Patch Repair", "Turn a reported defect into a behavior-changing patch.", "sp.reproduce-first", "sp.reproduce-fix-witness", "sp.inspect-edit-verify"),
		mkP("wp.spec-to-feature", "Spec-to-Feature Construction", "Turn an explicit contract into a working vertical slice.", "sp.plan-fanout-reconcile", "sp.read-edit-verify"),
		mkP("wp.behavior-preserving-restructure", "Behavior-Preserving Restructure", "Change structure while preserving externally observed behavior.", "sp.inspect-edit-verify", "sp.independent-adjudication"),
		mkP("wp.mechanical-sweep", "Mechanical Sweep", "Apply one rule over a bounded set of sites.", "sp.plan-fanout-reconcile", "sp.independent-adjudication"),
		mkP("wp.interface-migration", "Interface Migration", "Move producers and consumers across a contract boundary.", "sp.inspect-edit-verify", "sp.plan-fanout-reconcile"),
		mkP("wp.env-remediation", "Environment Remediation", "Restore a broken development or execution environment.", "sp.retry-after-refusal", "sp.read-edit-verify"),
		mkP("wp.oracle-construction", "Oracle Construction", "Create a check that can independently decide a claim.", "sp.reproduce-first", "sp.independent-adjudication"),
		mkP("wp.comprehension-report", "Comprehension Report", "Inspect evidence and produce a bounded explanatory artifact.", "sp.search-fanout", "sp.independent-adjudication"),
	}, Subpatterns: subs}
}
