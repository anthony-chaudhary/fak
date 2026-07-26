package policy

import (
	"reflect"
	"strings"
	"testing"
)

// TestEgressListFieldsParse proves the adblock-style allow/block manifest fields
// (allow_hosts, block_hosts, block_lists) map through to the runtime adjudicator.Policy
// and survive a round-trip back to the manifest via FromPolicy.
func TestEgressListFieldsParse(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{
		"version": "fak-policy/v1",
		"allow": ["WebFetch"],
		"egress": {
			"allow_hosts": ["docs.internal", "wiki.corp"],
			"block_hosts": ["ads.example", "tracker.example"],
			"block_lists": ["sample-malware"]
		}
	}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	if got, want := rt.Adjudicator.EgressAllowHosts, []string{"docs.internal", "wiki.corp"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("EgressAllowHosts = %v, want %v", got, want)
	}
	if got, want := rt.Adjudicator.EgressBlockHosts, []string{"ads.example", "tracker.example"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("EgressBlockHosts = %v, want %v", got, want)
	}
	if got, want := rt.Adjudicator.EgressBlockLists, []string{"sample-malware"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("EgressBlockLists = %v, want %v", got, want)
	}
	// Round-trip: the compiled runtime dumps back to the same manifest egress block.
	dumped := FromPolicy(rt.Adjudicator)
	if dumped.Egress == nil {
		t.Fatal("FromPolicy dropped the egress block")
	}
	if !reflect.DeepEqual(dumped.Egress.AllowHosts, []string{"docs.internal", "wiki.corp"}) ||
		!reflect.DeepEqual(dumped.Egress.BlockHosts, []string{"ads.example", "tracker.example"}) ||
		!reflect.DeepEqual(dumped.Egress.BlockLists, []string{"sample-malware"}) {
		t.Fatalf("round-trip egress = %+v, want the parsed lists back", dumped.Egress)
	}
}

// TestEgressUnknownBlockListFailsLoud proves a block_lists name that does not resolve to a
// bundled list is a HARD parse error naming the available lists — a dropped block list can
// never silently become an all-permissive no-op.
func TestEgressUnknownBlockListFailsLoud(t *testing.T) {
	_, err := ParseRuntime([]byte(`{
		"version": "fak-policy/v1",
		"allow": ["WebFetch"],
		"egress": { "block_lists": ["no-such-list"] }
	}`))
	if err == nil {
		t.Fatal("unknown egress block_list must be a hard error")
	}
	if !strings.Contains(err.Error(), "no-such-list") || !strings.Contains(err.Error(), "available") {
		t.Fatalf("error %q should name the bad list and the available ones", err.Error())
	}
}

// TestEgressRestrictParsesAndRoundTrips proves the restrict knob maps to
// Policy.EgressRestrict and survives a round-trip through FromPolicy.
func TestEgressRestrictParsesAndRoundTrips(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{
		"version": "fak-policy/v1",
		"egress": { "restrict": true, "allow_hosts": ["docs.internal"] }
	}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	if !rt.Adjudicator.EgressRestrict {
		t.Fatal("EgressRestrict = false, want true")
	}
	dumped := FromPolicy(rt.Adjudicator)
	if dumped.Egress == nil || !dumped.Egress.Restrict {
		t.Fatalf("round-trip lost restrict: %+v", dumped.Egress)
	}
}

// TestEgressDefaultPostureIsNotRestrict proves the default (omitted restrict) is
// default-allowed — EgressRestrict stays false and Summary says so.
func TestEgressDefaultPostureIsNotRestrict(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{"version":"fak-policy/v1","allow":["WebFetch"],"egress":{"block_hosts":["ads.example"]}}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	if rt.Adjudicator.EgressRestrict {
		t.Fatal("EgressRestrict = true, want false (default-allowed)")
	}
	if out := Summary(rt.Adjudicator); !strings.Contains(out, "default-allowed") {
		t.Fatalf("Summary should report default-allowed posture:\n%s", out)
	}
}

// TestEgressListSummaryRenders proves Summary surfaces the new lists so an operator can
// see the whole egress posture in one dump.
func TestEgressListSummaryRenders(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{
		"version": "fak-policy/v1",
		"allow": ["WebFetch"],
		"egress": { "allow_hosts": ["docs.internal"], "block_hosts": ["ads.example"], "block_lists": ["sample-malware"] }
	}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	out := Summary(rt.Adjudicator)
	for _, want := range []string{"egress allow hosts", "docs.internal", "egress block hosts", "ads.example", "egress block lists", "sample-malware"} {
		if !strings.Contains(out, want) {
			t.Errorf("Explain output missing %q\n%s", want, out)
		}
	}
}
