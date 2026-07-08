package conformance

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestConformanceRunPasses asserts the standalone suite certifies THIS build: the live ABI
// numbering matches the frozen contract and every verdict-matrix case adjudicates as the
// shipped floor requires. This is the baseline a fork/auditor expects to pass green.
func TestConformanceRunPasses(t *testing.T) {
	r := Run()
	if !r.Pass {
		t.Fatalf("conformance suite failed on this build:\n%s", Render(r))
	}
	if len(r.Checks) != 2 {
		t.Fatalf("expected 2 checks (abi + adjudication), got %d", len(r.Checks))
	}
	if r.ABIVersion == "" {
		t.Fatal("report carries no ABI version")
	}
	for _, c := range r.Checks {
		if !c.Pass {
			t.Errorf("check %q failed: %v", c.Name, c.Failures)
		}
		if c.Cases == 0 {
			t.Errorf("check %q asserted zero cases (vacuous)", c.Name)
		}
	}
}

// TestEmbeddedGoldenMatchesSource is the #453 SLA-to-ABI-cadence pin: the golden embedded in
// the standalone suite must be byte-identical to internal/abi's source-of-truth golden. If
// the kernel re-freezes the ABI and this copy is not updated, this test fails — the suite
// cannot silently attest a stale floor.
func TestEmbeddedGoldenMatchesSource(t *testing.T) {
	src, err := os.ReadFile("../abi/testdata/abi_v0.1.golden")
	if err != nil {
		t.Fatalf("read source golden: %v", err)
	}
	if !bytes.Equal(src, abiGolden) {
		t.Fatalf("embedded abi golden drifted from internal/abi/testdata/abi_v0.1.golden — "+
			"re-copy it so the conformance suite tracks the frozen ABI.\n--- source ---\n%s\n--- embedded ---\n%s",
			src, abiGolden)
	}
}

// TestEmbeddedPolicyMatchesSource pins the embedded dogfood policy to the shipped manifest,
// for the same reason: the verdict matrix must be re-run against the ACTUAL floor the
// launcher hands the kernel, not a copy that quietly diverged.
func TestEmbeddedPolicyMatchesSource(t *testing.T) {
	src, err := os.ReadFile("../../examples/dogfood-claude-policy.json")
	if err != nil {
		t.Fatalf("read source policy: %v", err)
	}
	if !bytes.Equal(src, dogfoodPolicy) {
		t.Fatalf("embedded dogfood policy drifted from examples/dogfood-claude-policy.json — "+
			"re-copy it so the conformance suite locks the real floor.\n--- source ---\n%s\n--- embedded ---\n%s",
			src, dogfoodPolicy)
	}
}

// TestABIContractHasTeeth proves the wire-contract check is a real diff, not an always-pass:
// a single mutated value in the frozen map must no longer equal the live-computed JSON.
func TestABIContractHasTeeth(t *testing.T) {
	// Baseline: the check passes on the true contract.
	if !checkABIContract().Pass {
		t.Fatal("abi-wire-contract check should pass on the true contract")
	}
	// Mutate one closed value and confirm the marshaled forms diverge — i.e. a renumber
	// would be caught. (We compare marshaled maps directly rather than mutating package state.)
	tampered := liveABIMatrix()
	tampered["verdict_kinds"]["Deny"] = 99 // a renumber of a closed value
	tamperedJSON, _ := json.MarshalIndent(tampered, "", "  ")
	if string(tamperedJSON) == string(abiGolden) {
		t.Fatal("a renumbered Deny still matched the golden — the freeze has no teeth")
	}
}

// TestVerdictMatrixHasTeeth proves the adjudication check discriminates: the matrix must
// contain BOTH allow and deny expectations, so a floor that vacuously allowed (or denied)
// everything would fail Run. Belt-and-suspenders on top of Run's full pass.
func TestVerdictMatrixHasTeeth(t *testing.T) {
	var allow, deny int
	for _, c := range verdictMatrix {
		if !strings.HasPrefix(c.args, "{") {
			t.Fatalf("case %q has non-JSON args %q", c.name, c.args)
		}
		if c.kind == abi.VerdictAllow {
			allow++
		} else {
			deny++
		}
	}
	if allow == 0 || deny == 0 {
		t.Fatalf("verdict matrix is not discriminating: allow=%d deny=%d (need both)", allow, deny)
	}
}
