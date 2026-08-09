package syspromptmmu

import (
	"reflect"
	"testing"
)

// TestStableBlockInputsFieldSet is the architest-style construction gate for
// presence volatility. Adding any operand to StableBlockInputs is a deliberate,
// review-visible test change; in particular there is no current message, request,
// turn text, clock, timestamp, UUID, or nonce a resident gate could read.
func TestStableBlockInputsFieldSet(t *testing.T) {
	typ := reflect.TypeOf(StableBlockInputs{})
	want := []struct {
		name string
		kind reflect.Kind
	}{
		{name: "policyVersion", kind: reflect.String},
		{name: "toolsetDigest", kind: reflect.String},
		{name: "deploymentProfile", kind: reflect.String},
	}
	if typ.NumField() != len(want) {
		t.Fatalf("StableBlockInputs has %d fields, want the reviewed stable set of %d", typ.NumField(), len(want))
	}
	for i, field := range want {
		got := typ.Field(i)
		if got.Name != field.name || got.Type.Kind() != field.kind {
			t.Errorf("field %d = %s %s, want %s %s", i, got.Name, got.Type.Kind(), field.name, field.kind)
		}
		if got.PkgPath == "" {
			t.Errorf("field %s is exported; gate inputs must remain constructor-only", got.Name)
		}
	}
}

// TestResidentRegistryDeclaresEveryAuthoredBlock pins the one registry a reviewer
// audits. A block with no id/gate or a duplicate id cannot silently become a new
// resident presence decision.
func TestResidentRegistryDeclaresEveryAuthoredBlock(t *testing.T) {
	wantIDs := []ResidentBlockID{
		"spine.identity",
		"spine.gate",
		"spine.journal",
		"spine.capability",
		"policy.deny-floor",
		"policy.safety-resident",
	}
	if len(baseContext) != len(wantIDs) {
		t.Fatalf("resident registry has %d blocks, want reviewed inventory of %d", len(baseContext), len(wantIDs))
	}
	seen := make(map[ResidentBlockID]bool, len(baseContext))
	for i, spec := range baseContext {
		if spec.id != wantIDs[i] {
			t.Errorf("resident block %d id = %q, want %q", i, spec.id, wantIDs[i])
		}
		if spec.id == "" {
			t.Errorf("resident block %d has no stable id", i)
		}
		if seen[spec.id] {
			t.Errorf("resident block id %q is duplicated", spec.id)
		}
		seen[spec.id] = true
		if spec.gate == nil {
			t.Errorf("resident block %q has no typed gate", spec.id)
		}
		if !NonEvictable(spec.tier) {
			t.Errorf("resident block %q uses evictable tier %s", spec.id, spec.tier)
		}
		if spec.content == "" {
			t.Errorf("resident block %q has empty content", spec.id)
		}
	}
}

// TestResidentPresenceIsAppendOnly is the failure-class-matched regression: the
// conditional gate itself really does flicker false -> true -> false, but the
// conversation-scoped builder turns that volatile predicate result into monotone
// presence. The previously emitted resident plan remains an exact prefix.
func TestResidentPresenceIsAppendOnly(t *testing.T) {
	specs := []residentBlockSpec{
		{id: "test.always", tier: TierSpine, content: "always", gate: alwaysResident},
		{
			id:      "test.conditional",
			tier:    TierPolicy,
			content: "conditional",
			gate: func(in StableBlockInputs) bool {
				return in.deploymentProfile == "armed"
			},
		},
	}
	builder := newResidentBlocks(specs)
	off := NewStableBlockInputs("v1", "caps", "off")
	on := NewStableBlockInputs("v1", "caps", "armed")

	first := builder.Next(off)
	if len(first) != 1 || string(first[0].Content) != "always" {
		t.Fatalf("initial resident plan = %+v, want only the always block", first)
	}
	second := builder.Next(on)
	if len(second) != 2 || string(second[1].Content) != "conditional" {
		t.Fatalf("armed resident plan = %+v, want conditional block appended", second)
	}
	third := builder.Next(off) // raw predicate is false again; presence must stay latched.
	if !reflect.DeepEqual(third, second) {
		t.Fatalf("resident presence flickered off:\n after on  = %+v\n after off = %+v", second, third)
	}
	if !reflect.DeepEqual(second[:len(first)], first) {
		t.Fatalf("newly admitted block did not append after the prior resident prefix")
	}
}

// TestCanonicalResidentBuilderPreservesLegacyPlan proves the structural gate did not
// change the existing all-resident plan. The package golden independently pins the
// exact bytes, witnesses, and PlanDigest.
func TestCanonicalResidentBuilderPreservesLegacyPlan(t *testing.T) {
	got := NewResidentBlocks().Next(NewStableBlockInputs("v1", "caps", "default"))
	if want := BaseContext(); !reflect.DeepEqual(got, want) {
		t.Fatalf("canonical resident builder changed the legacy plan:\n got  %+v\n want %+v", got, want)
	}
}
