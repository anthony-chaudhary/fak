package policy

import (
	"reflect"
	"testing"
)

// TestEgressDenyHostsParse proves the manifest egress.deny_hosts block maps through to
// the runtime adjudicator.Policy.EgressExtraDenyHosts, so an operator can extend the
// hardwired cloud-metadata egress floor with their own destinations from the manifest.
func TestEgressDenyHostsParse(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{
		"version": "fak-policy/v1",
		"allow": ["WebFetch"],
		"egress": { "deny_hosts": ["secrets.corp.internal", "10.0.0.53"] }
	}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	got := rt.Adjudicator.EgressExtraDenyHosts
	want := []string{"secrets.corp.internal", "10.0.0.53"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("EgressExtraDenyHosts = %v, want %v", got, want)
	}
}

// TestEgressAbsentLeavesFloorBare confirms a manifest with no egress block leaves
// EgressExtraDenyHosts empty (the hardwired metadata set is the whole floor) and that
// an unknown key inside egress is rejected (DisallowUnknownFields), so a typo fails
// loud rather than silently dropping a deny.
func TestEgressAbsentLeavesFloorBare(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{"version":"fak-policy/v1","allow":["WebFetch"]}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	if len(rt.Adjudicator.EgressExtraDenyHosts) != 0 {
		t.Fatalf("expected no extra egress deny hosts, got %v", rt.Adjudicator.EgressExtraDenyHosts)
	}
	if _, err := ParseRuntime([]byte(`{"version":"fak-policy/v1","egress":{"denyhosts":["x"]}}`)); err == nil {
		t.Fatal("a misspelled egress key must be a hard error (DisallowUnknownFields)")
	}
}
