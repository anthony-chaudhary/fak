package model

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=1s lane=default
func TestV41DeviceOnlyHostOpInventoryIsClosedAndAdmitsWhenEmpty(t *testing.T) {
	seen := map[string]bool{}
	for _, op := range v41DeviceOnlyHostOps {
		if op.name == "" || op.site == "" || seen[op.name] {
			t.Fatalf("inventory entry %+v is blank or duplicated", op)
		}
		seen[op.name] = true
	}
	err := v41DeviceOnlyRefusal()
	var typed *v41DeviceOnlyHostOpError
	if !errors.Is(err, errV41DeviceOnlyHostOp) || !errors.As(err, &typed) || len(typed.ops) != len(v41DeviceOnlyHostOps) {
		t.Fatalf("refusal = %v, want typed inventory of %d ops", err, len(v41DeviceOnlyHostOps))
	}
	typed.ops[0].name = "mutated"
	if v41DeviceOnlyHostOps[0].name == "mutated" {
		t.Fatal("refusal aliases the inventory")
	}

	saved := v41DeviceOnlyHostOps
	t.Cleanup(func() { v41DeviceOnlyHostOps = saved })
	v41DeviceOnlyHostOps = nil
	if err := v41DeviceOnlyRefusal(); err != nil {
		t.Fatalf("empty inventory refusal = %v, want admission", err)
	}
}
