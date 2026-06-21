package agent

// lint.go — the agent's enriched tool-facts collector. The kernel-only collector
// (toollint.FromKernel) sees the fast-path tables and pre-flight schemas but NOT
// the hints, because hints are per-call Meta, not per-registration. This package is
// the layer that maps a tool name to its hints (metaFor) and advertises the
// model-facing catalog, so it is where the hint-shaped and advertised-vs-enforced
// lint rules get the inputs they need. LintFacts folds the kernel view together
// with the catalog, the classifier, and the grammar registry into the full picture.

import (
	"encoding/json"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/grammar"
	"github.com/anthony-chaudhary/fak/internal/toollint"
)

// LintFacts assembles the full static tool surface this agent configures: the
// vDSO/pre-flight kernel view enriched with each catalog tool's declared hints
// (metaFor), the params it advertises to the model, and whether a grammar enforces
// its args in-kernel. Call it AFTER Configure() so the grammar and schemas are
// registered. The result feeds toollint.Lint.
func LintFacts() []toollint.ToolFacts {
	byName := map[string]*toollint.ToolFacts{}
	get := func(name string) *toollint.ToolFacts {
		if f := byName[name]; f != nil {
			return f
		}
		f := &toollint.ToolFacts{Name: name}
		byName[name] = f
		return f
	}

	// Base: the kernel-derived facts (pure/static Kind, pre-flight schemas).
	for _, f := range toollint.FromKernel() {
		base := f
		*get(base.Name) = base
	}

	// Enrich each model-facing catalog tool with hints + advertised schema + grammar.
	for _, td := range ToolCatalog() {
		name := td.Function.Name
		f := get(name)
		f.Hints = toollint.HintsFromMeta(metaFor(name))
		if f.Kind == toollint.KindUnknown {
			f.Kind = toollint.KindEngine // routed to the localtools engine
		}
		f.Advertised = true
		f.AdvertisedRequired = requiredParams(td.Function.Parameters)
		f.HasGrammar = grammar.Default.Has(name)
	}

	// Mark policy-denied tools (TL008). We only flag a denied tool that is ALREADY in
	// the surface — i.e. registered on the fast path or in the catalog — because the
	// hazard is precisely "denied AND served by the vDSO before the Deny folds". A
	// denied tool that is on no fast path (the common, safe case, e.g. delete_account)
	// stays out of the facts rather than padding the surface with an inert entry.
	for _, name := range adjudicator.Default.DeniedTools() {
		if f := byName[name]; f != nil {
			f.PolicyDenied = true
		}
	}

	names := make([]string, 0, len(byName))
	for n := range byName {
		names = append(names, n)
	}
	sort.Strings(names)
	out := make([]toollint.ToolFacts, 0, len(byName))
	for _, n := range names {
		out = append(out, *byName[n])
	}
	return out
}

// requiredParams extracts the "required" string array from a JSON-Schema parameter
// block (the model-facing contract). Returns nil if the schema is absent or has no
// required fields.
func requiredParams(schema json.RawMessage) []string {
	if len(schema) == 0 {
		return nil
	}
	var s struct {
		Required []string `json:"required"`
	}
	if err := json.Unmarshal(schema, &s); err != nil {
		return nil
	}
	return s.Required
}
