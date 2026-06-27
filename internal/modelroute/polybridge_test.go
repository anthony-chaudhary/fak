package modelroute

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// #604 — co-residency bridge from ensemble roles to polymodel spec-decode. These
// tests cover the issue's bar: (a) a co-resident drafter+verifier maps onto a
// bridged pairing that drives polymodel's drafter selection, (b) different-family
// members are NOT bridgeable, and (c) the mapping is deterministic and preserves
// member order.

// specPlan is a 1-drafter/1-verifier ensemble Plan in the given member order.
func specPlan(drafter, verifier string) Plan {
	return Plan{Members: []Member{
		{Model: drafter, Role: "drafter"},
		{Model: verifier, Role: "verifier"},
	}, Reduce: ReduceFirst}
}

// coResidentPool admits a drafter and verifier in the SAME family with the SAME
// prefix digest, so polymodel.CanShare(verifier, drafter) holds — the co-residency
// the bridge requires. The drafter is the cheaper (smaller) of the two.
func coResidentPool(t *testing.T) *polymodel.Pool {
	t.Helper()
	p := polymodel.NewPool(1 << 20)
	for _, m := range []polymodel.Model{
		{ID: "draft-7b", Family: "glm", PrefixDigest: "sha256:band", WeightBytes: 7000},
		{ID: "verify-70b", Family: "glm", PrefixDigest: "sha256:band", WeightBytes: 70000},
	} {
		if _, err := p.Admit(m); err != nil {
			t.Fatalf("admit %s: %v", m.ID, err)
		}
	}
	return p
}

// TestBridgeCoResidentDrafterVerifier proves a roles-tagged Plan over co-resident
// same-family members maps onto a polymodel drafter/verifier pairing the host
// hands straight to polymodel.Schedule / AcceptGreedy.
func TestBridgeCoResidentDrafterVerifier(t *testing.T) {
	pool := coResidentPool(t)
	b, err := specPlan("draft-7b", "verify-70b").BridgeToPolymodel(pool)
	if err != nil {
		t.Fatalf("co-resident spec ensemble should bridge: %v", err)
	}
	if b.Drafter != "draft-7b" || b.Verifier != "verify-70b" {
		t.Fatalf("bridge = %+v, want drafter=draft-7b verifier=verify-70b", b)
	}
}

// TestBridgeDifferentFamilyNotBridgeable proves two members in DIFFERENT families
// (incompatible tokenizers) are refused — never silently run as spec-decode
// against a target whose vocabulary the draft tokens don't fit.
func TestBridgeDifferentFamilyNotBridgeable(t *testing.T) {
	p := polymodel.NewPool(1 << 20)
	// Both resident, but different families → not a valid speculation target.
	mustAdmitBridge(t, p, polymodel.Model{ID: "draft-a", Family: "glm", PrefixDigest: "sha256:x", WeightBytes: 7000})
	mustAdmitBridge(t, p, polymodel.Model{ID: "verify-b", Family: "qwen", PrefixDigest: "sha256:y", WeightBytes: 70000})

	_, err := specPlan("draft-a", "verify-b").BridgeToPolymodel(p)
	if err == nil {
		t.Fatal("different-family members must not be bridgeable")
	}
	if !errors.Is(err, ErrNotBridgeable) {
		t.Fatalf("want ErrNotBridgeable, got %v", err)
	}
}

// TestBridgeNotResidentNotBridgeable proves a Plan whose members are not (both)
// resident is not bridgeable — a nil pool and a half-resident pool both refuse.
func TestBridgeNotResidentNotBridgeable(t *testing.T) {
	// nil pool: nothing resident.
	if _, err := specPlan("d", "v").BridgeToPolymodel(nil); !errors.Is(err, ErrNotBridgeable) {
		t.Fatalf("nil pool should be not-bridgeable, got %v", err)
	}
	// Only the drafter resident; the verifier is absent.
	p := polymodel.NewPool(1 << 20)
	mustAdmitBridge(t, p, polymodel.Model{ID: "draft-7b", Family: "glm", PrefixDigest: "sha256:band", WeightBytes: 7000})
	if _, err := specPlan("draft-7b", "verify-70b").BridgeToPolymodel(p); !errors.Is(err, ErrNotBridgeable) {
		t.Fatalf("absent verifier should be not-bridgeable, got %v", err)
	}
}

// TestBridgeSameFamilyNoSharedDigestNotBridgeable proves same-Family is necessary
// but not sufficient: distinct prefix digests fail CanShare (the reused prefill KV
// would NOT be bit-identical), so the pair is still refused.
func TestBridgeSameFamilyNoSharedDigestNotBridgeable(t *testing.T) {
	p := polymodel.NewPool(1 << 20)
	mustAdmitBridge(t, p, polymodel.Model{ID: "draft-7b", Family: "glm", PrefixDigest: "sha256:a", WeightBytes: 7000})
	mustAdmitBridge(t, p, polymodel.Model{ID: "verify-70b", Family: "glm", PrefixDigest: "sha256:b", WeightBytes: 70000})
	if _, err := specPlan("draft-7b", "verify-70b").BridgeToPolymodel(p); !errors.Is(err, ErrNotBridgeable) {
		t.Fatalf("same family but mismatched digest should be not-bridgeable, got %v", err)
	}
}

// TestBridgeNonSpecEnsembleNotBridgeable proves a vote/best-of ensemble (no
// drafter/verifier roles) is refused before residency is even consulted — a
// non-spec ensemble must not be run as spec-decode.
func TestBridgeNonSpecEnsembleNotBridgeable(t *testing.T) {
	pool := coResidentPool(t)
	vote := Plan{Members: []Member{{Model: "draft-7b"}, {Model: "verify-70b"}}, Reduce: ReduceVote}
	if _, err := vote.BridgeToPolymodel(pool); err == nil {
		t.Fatal("a role-less vote ensemble must not bridge to spec-decode")
	}
}

// TestBridgeDeterministicAndOrderPreserving proves the mapping is deterministic
// (same Plan + Pool always yields the same pairing) and preserves member order:
// the drafter role and verifier role bind to the members in declared order
// regardless of which member appears first.
func TestBridgeDeterministicAndOrderPreserving(t *testing.T) {
	pool := coResidentPool(t)
	plan := specPlan("draft-7b", "verify-70b")

	first, err := plan.BridgeToPolymodel(pool)
	if err != nil {
		t.Fatalf("bridge: %v", err)
	}
	for i := 0; i < 50; i++ {
		got, err := plan.BridgeToPolymodel(pool)
		if err != nil {
			t.Fatalf("bridge at %d: %v", i, err)
		}
		if got != first {
			t.Fatalf("non-deterministic bridge at %d: %+v != %+v", i, got, first)
		}
	}
	// Order preservation: verifier-first declaration still binds roles by Role tag,
	// not by position — drafter stays drafter, verifier stays verifier.
	reversed := Plan{Members: []Member{
		{Model: "verify-70b", Role: "verifier"},
		{Model: "draft-7b", Role: "drafter"},
	}}
	rb, err := reversed.BridgeToPolymodel(pool)
	if err != nil {
		t.Fatalf("reversed-order bridge: %v", err)
	}
	if rb != first {
		t.Fatalf("role binding should be order-independent: %+v != %+v", rb, first)
	}
}

// TestBridgeEnabledGate proves the live form refuses when FAK_POLYMODEL is OFF
// (the default) and bridges when ON, so the spec-decode lane never activates
// implicitly.
func TestBridgeEnabledGate(t *testing.T) {
	pool := coResidentPool(t)
	plan := specPlan("draft-7b", "verify-70b")

	// Default (flag unset / off): refused even though the members ARE co-resident.
	t.Setenv(polymodel.FlagEnv, "")
	if _, err := plan.BridgeToPolymodelEnabled(pool); !errors.Is(err, ErrNotBridgeable) {
		t.Fatalf("disabled lane should refuse, got %v", err)
	}
	// Enabled: identical to the pure bridge.
	t.Setenv(polymodel.FlagEnv, "on")
	if !polymodel.Enabled() {
		t.Fatal("FAK_POLYMODEL=on should enable the lane")
	}
	b, err := plan.BridgeToPolymodelEnabled(pool)
	if err != nil {
		t.Fatalf("enabled co-resident bridge should succeed: %v", err)
	}
	if b.Drafter != "draft-7b" || b.Verifier != "verify-70b" {
		t.Fatalf("enabled bridge = %+v", b)
	}
}

// TestPickPolyDrafterAgreesWithRole proves the residency-derived drafter pick
// (polymodel.PickDrafter, which prefers the cheapest co-resident family member)
// agrees with the Plan's declared drafter when the declared drafter IS the cheapest
// co-resident — and surfaces a divergence when a cheaper warm member exists.
func TestPickPolyDrafterAgreesWithRole(t *testing.T) {
	pool := coResidentPool(t) // draft-7b (7000) is the only other glm member of verify-70b
	b, preferred, agree, err := specPlan("draft-7b", "verify-70b").PickPolyDrafter(pool)
	if err != nil {
		t.Fatalf("pick: %v", err)
	}
	if !agree || preferred != b.Drafter {
		t.Fatalf("declared drafter should be the residency pick: preferred=%q drafter=%q agree=%v", preferred, b.Drafter, agree)
	}
	// Admit a CHEAPER co-resident family member: PickDrafter now prefers it, so the
	// declared drafter (draft-7b) no longer agrees — the divergence is surfaced.
	mustAdmitBridge(t, pool, polymodel.Model{ID: "draft-1b", Family: "glm", PrefixDigest: "sha256:band", WeightBytes: 1000})
	_, preferred2, agree2, err := specPlan("draft-7b", "verify-70b").PickPolyDrafter(pool)
	if err != nil {
		t.Fatalf("pick after cheaper admit: %v", err)
	}
	if agree2 || preferred2 != "draft-1b" {
		t.Fatalf("a cheaper co-resident should be the residency pick: preferred=%q agree=%v", preferred2, agree2)
	}
}

// mustAdmitBridge admits a model into a pool, failing the test on error.
func mustAdmitBridge(t *testing.T, p *polymodel.Pool, m polymodel.Model) {
	t.Helper()
	if _, err := p.Admit(m); err != nil {
		t.Fatalf("admit %s: %v", m.ID, err)
	}
}
