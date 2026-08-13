package portability

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
)

type ConformanceResult struct {
	Kind        string   `json:"kind"`
	Passed      []string `json:"passed"`
	Support     Support  `json:"support"`
	Degradation string   `json:"degradation,omitempty"`
}

func RunConformance(ctx context.Context, a Adapter) (ConformanceResult, error) {
	i := a.Info()
	out := ConformanceResult{Kind: i.Kind, Support: i.Support, Degradation: i.Degradation}
	pass := func(n string) { out.Passed = append(out.Passed, n) }
	if i.Support == SupportInactive {
		return out, errx("INACTIVE_ADAPTER", i.Kind, "conformance", "support", "registered adapters must declare executable support")
	}
	base := Record{Kind: i.Kind, Name: "alpha", Version: i.Version, Active: true, Data: json.RawMessage(`{"dependencies":["z","a"],"value":1}`), Unknown: json.RawMessage(`{"future":true}`)}
	malformed := base
	malformed.Data = json.RawMessage(`{`)
	if a.Validate(ctx, malformed) == nil {
		return out, errors.New("malformed input accepted")
	}
	if e := a.Validate(ctx, base); e != nil {
		return out, fmt.Errorf("validate: %w", e)
	}
	pass("malformed/adversarial-input")
	x, e := a.Export(ctx, base)
	if e != nil {
		return out, e
	}
	y, e := a.Export(ctx, base)
	if e != nil || string(x) != string(y) {
		return out, errors.New("non-deterministic export")
	}
	id1, _ := a.Identity(ctx, base)
	id2, _ := a.Identity(ctx, base)
	if id1 == "" || id1 != id2 {
		return out, errors.New("non-deterministic identity")
	}
	pass("deterministic-identity-export")
	var round Record
	if e = json.Unmarshal(x, &round); e != nil || !equalRecord(base, round) {
		return out, errors.New("round-trip mismatch")
	}
	pass("round-trip-unknown-fields")
	bad := base
	bad.Data = json.RawMessage(`{"password":"do-not-export"}`)
	if a.Validate(ctx, bad) == nil {
		return out, errors.New("secret accepted")
	}
	pass("secret-non-leakage")
	deps, e := a.Dependencies(ctx, base)
	if e != nil || len(deps) != 2 || deps[0] != "a" {
		return out, errors.New("dependencies not deterministic")
	}
	pass("dependencies")
	migrated, e := a.Migrate(ctx, base, "2")
	if e != nil || migrated.Version != "2" || string(migrated.Unknown) != string(base.Unknown) {
		return out, errors.New("migration lost data")
	}
	pass("schema-evolution")
	st := NewMemoryState()
	plan, e := a.Preview(ctx, st, "target", []Record{base})
	if e != nil {
		return out, e
	}
	rec, e := a.Apply(ctx, st, "target", plan)
	if e != nil {
		return out, e
	}
	again, e := a.Preview(ctx, st, "target", []Record{base})
	if e != nil || len(again.Changes) != 0 {
		return out, errors.New("apply not idempotent")
	}
	if e = a.Rollback(ctx, st, "target", rec); e != nil {
		return out, e
	}
	pass("apply-rollback-idempotence")
	conflict := base
	conflict.Kind = "foreign"
	_ = st.Write(ctx, "target", conflict)
	if _, e = a.Preview(ctx, st, "target", []Record{base}); e == nil {
		return out, errors.New("precedence conflict accepted")
	}
	pass("precedence-conflict")
	if i.Support == SupportPartial && i.Degradation == "" {
		return out, errors.New("partial adapter lacks degradation")
	}
	pass("partial-translation-status")
	interrupted := NewMemoryState()
	interrupted.FailAfter = 1
	two := base
	two.Name = "beta"
	p, _ := a.Preview(ctx, interrupted, "x", []Record{base, two})
	if _, e = a.Apply(ctx, interrupted, "x", p); e == nil {
		return out, errors.New("interruption not surfaced")
	}
	left, _ := interrupted.Discover(ctx, "x")
	if len(left) != 0 {
		return out, errors.New("interruption rollback incomplete")
	}
	pass("interruption-rollback")
	return out, nil
}
func RunReferenceConformance(ctx context.Context) ([]ConformanceResult, error) {
	r := ReferenceRegistry()
	var out []ConformanceResult
	for _, i := range r.Matrix() {
		x, e := RunConformance(ctx, r.Adapter(i.Kind))
		if e != nil {
			return nil, fmt.Errorf("%s: %w", i.Kind, e)
		}
		out = append(out, x)
	}
	return out, nil
}
