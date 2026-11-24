package routing

import (
	"testing"
)

// This file witnesses the policy registry that closes the gap RouteChainWithPosture
// documented ("no manifest registry exists yet"): the control plane publishes a
// named, versioned PolicyRef and the fabric resolves the BODY, read live at
// routing time. Per PRESENCE-VS-INVOKABILITY.md these tests assert the registry
// is CONSUMED, not merely present: a resolved body must change a routing
// outcome, and a miss must fall back rather than guess.

// appliancePolicyBody is a valid named body that prefers the appliance surface
// without naming a role, so it outranks the default policy's hosted-worker rule.
func appliancePolicyBody() Policy {
	return Policy{
		Version:        "routing-policy/1-registry-test",
		DefaultSurface: SurfaceAppliance,
		Rules: []PolicyRule{
			{Name: "appliance-first", Surface: SurfaceAppliance},
		},
	}
}

// TestPolicyRegistryResolvesPublishedBody proves a registered name+version
// resolves to the exact body, so the fabric routes with what was published.
func TestPolicyRegistryResolvesPublishedBody(t *testing.T) {
	r := NewPolicyRegistry()
	body := appliancePolicyBody()
	if err := r.Register("appliance-first", "", body); err != nil {
		t.Fatalf("Register: %v", err)
	}

	// An empty version defaults to the body's own Version.
	got, ok := r.Resolve("appliance-first", body.Version)
	if !ok {
		t.Fatal("Resolve(name, version) missed a just-registered body")
	}
	if got.DefaultSurface != SurfaceAppliance || got.Version != body.Version {
		t.Errorf("Resolve = %+v, want %+v", got, body)
	}
}

// TestPolicyRegistryRejectsMalformedLoud pins the fail-loud contract: a
// malformed manifest is a caller bug and must be refused before it can route.
func TestPolicyRegistryRejectsMalformedLoud(t *testing.T) {
	r := NewPolicyRegistry()
	if err := r.Register("bad", "", Policy{DefaultSurface: Surface("nonsense")}); err == nil {
		t.Error("Register with an unknown default surface = nil, want a validation error")
	}
	if _, ok := r.Resolve("bad", ""); ok {
		t.Error("a refused registration must not be resolvable")
	}
	if err := r.Register("", "", DefaultPolicy()); err == nil {
		t.Error("Register with an empty name = nil, want an error")
	}
}

// TestPolicyRegistryVersionedBodiesCoexist pins the audited-history contract: a
// different version under the same name coexists; an exact version must match,
// and an empty version is a miss when several coexist (never "latest wins").
func TestPolicyRegistryVersionedBodiesCoexist(t *testing.T) {
	r := NewPolicyRegistry()
	v1 := appliancePolicyBody()
	v2 := appliancePolicyBody()
	v2.Version = "routing-policy/1-registry-test-v2"
	if err := r.Register("appliance-first", v1.Version, v1); err != nil {
		t.Fatalf("Register v1: %v", err)
	}
	if err := r.Register("appliance-first", v2.Version, v2); err != nil {
		t.Fatalf("Register v2: %v", err)
	}

	if got, ok := r.Resolve("appliance-first", v1.Version); !ok || got.Version != v1.Version {
		t.Errorf("Resolve(v1) = %+v, %v; want v1 body", got, ok)
	}
	if _, ok := r.Resolve("appliance-first", ""); ok {
		t.Error("Resolve(name, \"\") with two versions coexisting = hit, want a deterministic miss")
	}
	if _, ok := r.Resolve("appliance-first", "routing-policy/1-nope"); ok {
		t.Error("Resolve with an unknown version = hit, want a miss")
	}
	if _, ok := r.Resolve("never-registered", ""); ok {
		t.Error("Resolve of an unknown name = hit, want a miss")
	}
}

// TestDefaultPolicyRegistryCarriesTheBuiltIn is the process-wide singleton
// contract: the default registry ships the built-in "default" body, so a
// consumer that publishes nothing still resolves a named default rather than
// having to special-case it.
func TestDefaultPolicyRegistryCarriesTheBuiltIn(t *testing.T) {
	got, ok := DefaultPolicyRegistry().Resolve(DefaultPolicyName, PolicySchema)
	if !ok {
		t.Fatalf("DefaultPolicyRegistry().Resolve(%q, %q) missed the built-in body", DefaultPolicyName, PolicySchema)
	}
	if got.Version != DefaultPolicy().Version {
		t.Errorf("built-in body version = %q, want %q", got.Version, DefaultPolicy().Version)
	}
}

// TestPolicyRegistrySnapshotIsolated is the TICKET-05 wave-insulation witness:
// a Snapshot carries the bodies present at capture, and later Register calls
// on either side do not cross. A spawn wave routing against its snapshot is
// therefore immune to a concurrent publish (or a test registering into the
// shared global) after admission.
func TestPolicyRegistrySnapshotIsolated(t *testing.T) {
	r := NewPolicyRegistry()
	first := appliancePolicyBody()
	if err := r.Register("wave-policy", first.Version, first); err != nil {
		t.Fatalf("Register: %v", err)
	}
	snap := r.Snapshot()

	// Overwrite the same name+version on the SOURCE: the snapshot keeps the
	// original body.
	second := appliancePolicyBody()
	second.DefaultSurface = SurfaceHosted
	if err := r.Register("wave-policy", first.Version, second); err != nil {
		t.Fatalf("re-Register: %v", err)
	}
	got, ok := snap.Resolve("wave-policy", first.Version)
	if !ok {
		t.Fatal("snapshot missed a body present at capture")
	}
	if got.DefaultSurface != SurfaceAppliance {
		t.Errorf("snapshot body surface = %q, want %q (post-capture overwrite leaked in)", got.DefaultSurface, SurfaceAppliance)
	}

	// A name registered only on the snapshot never appears on the source.
	only := appliancePolicyBody()
	only.Version = "routing-policy/1-snapshot-only"
	if err := snap.Register("snapshot-only", "", only); err != nil {
		t.Fatalf("snapshot Register: %v", err)
	}
	if _, ok := r.Resolve("snapshot-only", ""); ok {
		t.Error("source resolved a name registered only on the snapshot — the copy is not isolated")
	}

	// A nil receiver snapshots to an empty registry, never a panic.
	var nilReg *PolicyRegistry
	empty := nilReg.Snapshot()
	if _, ok := empty.Resolve("anything", ""); ok {
		t.Error("nil snapshot resolved a body, want an empty registry")
	}
}

// TestPolicyRegistryBodySteersRoutingOutcome is the consumed-not-present
// witness: two policies differing only in preferred surface, resolved BY NAME
// from the registry, must select different winning engines from the SAME engine
// set. If resolving the body did not feed routing, the registry would be
// decorative.
func TestPolicyRegistryBodySteersRoutingOutcome(t *testing.T) {
	engines := []Engine{
		{ID: "hosted-1", Surface: SurfaceHosted, Healthy: true, CapacityHeadroom: 4},
		{ID: "appliance-1", Surface: SurfaceAppliance, Healthy: true, CapacityHeadroom: 4},
	}
	req := TaskRequest{Candidate: &TaskCandidate{ID: "T1"}, Role: "worker"}

	r := NewPolicyRegistry()
	applianceFirst := appliancePolicyBody()
	if err := r.Register("appliance-first", "", applianceFirst); err != nil {
		t.Fatalf("Register: %v", err)
	}

	body, ok := r.Resolve("appliance-first", "")
	if !ok {
		t.Fatal("Resolve missed the registered body")
	}
	got, _, err := RouteRequestWithPolicy(req, engines, body)
	if err != nil {
		t.Fatalf("RouteRequestWithPolicy: %v", err)
	}
	if got.EngineID != "appliance-1" {
		t.Fatalf("resolved appliance-first body selected %q, want appliance-1", got.EngineID)
	}

	// The control: the built-in default body prefers the hosted worker, so the
	// SAME request over the SAME engines selects a different engine.
	def, ok := DefaultPolicyRegistry().Resolve(DefaultPolicyName, PolicySchema)
	if !ok {
		t.Fatal("Resolve missed the built-in default body")
	}
	base, _, err := RouteRequestWithPolicy(req, engines, def)
	if err != nil {
		t.Fatalf("RouteRequestWithPolicy(default): %v", err)
	}
	if base.EngineID == got.EngineID {
		t.Fatalf("both bodies selected %q — the resolved body does not steer the routing outcome", got.EngineID)
	}
}
