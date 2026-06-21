package toollint

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/preflight"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// factsByName indexes a collected slice for assertions.
func factsByName(fs []ToolFacts) map[string]ToolFacts {
	m := map[string]ToolFacts{}
	for _, f := range fs {
		m[f.Name] = f
	}
	return m
}

// FromRegistries classifies pure/static tools and lifts pre-flight schemas, and a
// static answer for a write-shaped tool lints to a TL003 error from that view alone
// (no hint classifier needed).
func TestFromRegistriesKindsAndSchemas(t *testing.T) {
	v := vdso.New(16)
	v.RegisterPure("calculate", func([]byte) ([]byte, bool) { return nil, false })
	v.RegisterStatic("list_all_airports", []byte(`{}`))
	v.RegisterStatic("send_alert", []byte(`{"ok":true}`)) // a write-shaped static answer: the hazard

	l := preflight.New()
	l.SetSchema("get_user", preflight.Schema{Required: map[string]preflight.FieldType{"user_id": preflight.TypeString}})

	facts := FromRegistries(v, l)
	by := factsByName(facts)

	if by["calculate"].Kind != KindPure {
		t.Fatalf("calculate Kind = %s, want pure", by["calculate"].Kind)
	}
	if by["list_all_airports"].Kind != KindStatic {
		t.Fatalf("list_all_airports Kind = %s, want static", by["list_all_airports"].Kind)
	}
	gu, ok := by["get_user"]
	if !ok || !gu.HasPreflightSchema || gu.SchemaTypes["user_id"] != "string" {
		t.Fatalf("get_user schema not lifted: %+v", gu)
	}

	// The write-shaped static answer is a TL003 error.
	r := Lint(facts)
	var got *Finding
	for i := range r.Findings {
		if r.Findings[i].Tool == "send_alert" && r.Findings[i].Code == StaticWriteShape {
			got = &r.Findings[i]
		}
	}
	if got == nil {
		t.Fatalf("expected TL003 on send_alert; findings=%+v", r.Findings)
	}
	if got.Severity != SevError || r.Errors() == 0 {
		t.Fatalf("TL003 must be an error severity; got %s (errors=%d)", got.Severity, r.Errors())
	}
}

// An empty-Required schema must NOT set HasPreflightSchema: preflight.Adjudicate
// runs rung-1 only when len(Required) > 0, so an empty schema enforces nothing and
// must not suppress TL004 (the flag would otherwise overstate enforcement).
func TestFromRegistriesEmptySchemaNotEnforced(t *testing.T) {
	l := preflight.New()
	l.SetSchema("empty", preflight.Schema{Required: map[string]preflight.FieldType{}})
	facts := FromRegistries(vdso.New(4), l)
	by := factsByName(facts)
	f, ok := by["empty"]
	if !ok {
		t.Fatalf("tool with an empty schema should still appear in the facts")
	}
	if f.HasPreflightSchema {
		t.Fatalf("empty-Required schema must not set HasPreflightSchema (rung-1 enforces nothing)")
	}
}

// FromKernel collects from the process-global registries (vdso.Default,
// preflight.Default) — the surface `fak lint --kernel-only` actually lints. It pins
// the partial-view contract: the init-seeded fast-path tools appear with their Kind,
// carry NO declared hints (no classifier in this view), and Lint produces no
// hint-shaped findings (TL001/002/007 stay silent except a name-only TL002, which the
// seeded read-shaped names do not trigger).
func TestFromKernelSeededSurfaceAndSilentHints(t *testing.T) {
	by := factsByName(FromKernel())

	calc, ok := by["calculate"]
	if !ok || calc.Kind != KindPure {
		t.Fatalf("calculate from kernel view: kind=%v ok=%v, want pure/present", calc.Kind, ok)
	}
	air, ok := by["list_all_airports"]
	if !ok || air.Kind != KindStatic {
		t.Fatalf("list_all_airports from kernel view: kind=%v ok=%v, want static/present", air.Kind, ok)
	}
	for _, f := range by {
		h := f.Hints
		if h.DeclaredReadOnly || h.DeclaredIdempotent || h.DeclaredDestructive {
			t.Fatalf("%s carries declared hints in the kernel-only view: %+v", f.Name, h)
		}
	}
	for _, fnd := range Lint(FromKernel()).Findings {
		switch fnd.Code {
		case DeadCacheHint, ContradictoryHints:
			t.Fatalf("hint-shaped finding %s leaked into the kernel-only view on %s", fnd.Code, fnd.Tool)
		}
	}
}

// A bad pre-flight type collected from a real ladder trips TL006.
func TestFromRegistriesSchemaTypeTypo(t *testing.T) {
	l := preflight.New()
	// "str" is not a preflight.FieldType the ladder validates; typeOK treats it as Any.
	l.SetSchema("t", preflight.Schema{Required: map[string]preflight.FieldType{"name": preflight.FieldType("str")}})
	facts := FromRegistries(vdso.New(4), l)
	r := Lint(facts)
	hit := false
	for _, f := range r.Findings {
		if f.Code == SchemaTypeUnsupported && f.Tool == "t" {
			hit = true
		}
	}
	if !hit {
		t.Fatalf("expected TL006 for typo'd schema type; findings=%+v", r.Findings)
	}
}
