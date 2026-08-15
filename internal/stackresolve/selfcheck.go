package stackresolve

import (
	"context"
	_ "embed"
	"fmt"
)

//go:embed testdata/coding-stack.json
var codingStackFixture []byte

//go:embed testdata/awq-sm75-unsat.json
var awqSM75UnsatFixture []byte

// Selfcheck drives the real parser and resolver through an allowed stack and a
// transitive hardware refusal. It is the reusable spine witness for adapters.
func Selfcheck(ctx context.Context) (Receipt, Receipt, error) {
	allowManifest, err := Parse(codingStackFixture)
	if err != nil {
		return Receipt{}, Receipt{}, fmt.Errorf("allow fixture: %w", err)
	}
	allow, err := Resolve(ctx, allowManifest.Workload, allowManifest.Roots, ManifestProvider{Manifest: allowManifest})
	if err != nil {
		return Receipt{}, Receipt{}, fmt.Errorf("allow resolve: %w", err)
	}
	if allow.Status != "allow" || len(allow.Selected) < 6 {
		return Receipt{}, Receipt{}, fmt.Errorf("allow fixture produced status=%s selected=%d", allow.Status, len(allow.Selected))
	}

	refuseManifest, err := Parse(awqSM75UnsatFixture)
	if err != nil {
		return Receipt{}, Receipt{}, fmt.Errorf("refuse fixture: %w", err)
	}
	refuse, err := Resolve(ctx, refuseManifest.Workload, refuseManifest.Roots, ManifestProvider{Manifest: refuseManifest})
	if err != nil {
		return Receipt{}, Receipt{}, fmt.Errorf("refuse resolve: %w", err)
	}
	if refuse.Status != "refuse" || refuse.Conflict == nil || refuse.Conflict.Code != "UNSATISFIED_REQUIREMENT" {
		return Receipt{}, Receipt{}, fmt.Errorf("refuse fixture did not produce expected conflict")
	}
	return allow, refuse, nil
}
