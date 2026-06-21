package toollint

// collect.go — the kernel-only facts collector. It reads the two registries that
// describe a tool's STATIC surface without any per-call context: the vDSO fast-path
// tables (which tools are pure / static) and the pre-flight schemas (what the kernel
// enforces at rung-1). It has NO hint classifier, so the facts it returns carry no
// declared hints and the hint-shaped rules (TL001/002/007) stay silent — the
// name/schema rules (TL003/005/006) fire from this view alone. A richer collector
// that also knows the hint classifier (the agent/gateway layer) enriches these
// facts to light up the rest.

import (
	"sort"

	"github.com/anthony-chaudhary/fak/internal/preflight"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// FromKernel builds tool facts from the process-global registries (vdso.Default and
// preflight.Default) — the surface the booted kernel will actually serve. Call it
// after the drivers have registered and any host configuration has run.
func FromKernel() []ToolFacts {
	return FromRegistries(vdso.Default, preflight.Default)
}

// FromRegistries builds tool facts from the given registries (the seam tests use to
// pass fresh instances). The union of every tool the vDSO registered (pure or
// static) and every tool the pre-flight ladder has a schema for becomes one
// ToolFacts; a tool present in both keeps its fast-path Kind and gains the schema.
func FromRegistries(v *vdso.VDSO, l *preflight.Ladder) []ToolFacts {
	byName := map[string]*ToolFacts{}
	get := func(name string) *ToolFacts {
		if f := byName[name]; f != nil {
			return f
		}
		f := &ToolFacts{Name: name}
		byName[name] = f
		return f
	}

	if v != nil {
		for _, t := range v.PureTools() {
			get(t).Kind = KindPure
		}
		for _, t := range v.StaticTools() {
			// A name cannot be both pure and static in one vDSO; if it somehow is,
			// static wins here because tier-3 is served unconditionally (the more
			// dangerous path TL003 must see).
			get(t).Kind = KindStatic
		}
	}
	if l != nil {
		for tool, s := range l.Schemas() {
			f := get(tool)
			// HasPreflightSchema must mean "rung-1 actually validates this tool", not
			// merely "a schema object exists". preflight.Adjudicate runs rung-1 only
			// when len(Required) > 0; an empty-Required schema enforces nothing and
			// falls through to Defer. Gating the flag the same way keeps it from
			// suppressing TL004 for a contract the kernel does not actually enforce.
			if len(s.Required) > 0 {
				f.HasPreflightSchema = true
			}
			f.SchemaTypes = schemaTypes(s)
		}
	}

	names := make([]string, 0, len(byName))
	for n := range byName {
		names = append(names, n)
	}
	sort.Strings(names)
	out := make([]ToolFacts, 0, len(byName))
	for _, n := range names {
		out = append(out, *byName[n])
	}
	return out
}

// schemaTypes flattens a pre-flight schema's required fields to a field->type-string
// map, the form TL006 checks. Returns nil for an empty schema so the rule has
// nothing to iterate.
func schemaTypes(s preflight.Schema) map[string]string {
	if len(s.Required) == 0 {
		return nil
	}
	m := make(map[string]string, len(s.Required))
	for k, ft := range s.Required {
		m[k] = string(ft)
	}
	return m
}
