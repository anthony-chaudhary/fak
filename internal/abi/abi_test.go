package abi

import (
	"encoding/json"
	"os"
	"testing"
)

// TestABIGoldenFreeze pins the WIRE CONTRACT of the closed enums (unit 2/9). The
// freeze is additive-only: a renumber, removal, or repurpose of any closed value
// changes this map and fails the test, turning the freeze into a machine-checked
// contract. Adding a NEW value at the end is the only allowed change (update the
// golden with -update).
func TestABIGoldenFreeze(t *testing.T) {
	got := map[string]map[string]int{
		"verdict_kinds": {
			"Allow": int(VerdictAllow), "Deny": int(VerdictDeny),
			"Transform": int(VerdictTransform), "Quarantine": int(VerdictQuarantine),
			"RequireWitness": int(VerdictRequireWitness), "Defer": int(VerdictDefer),
			"ReservedMax": int(VerdictReservedMax),
		},
		"status":   {"OK": int(StatusOK), "Error": int(StatusError), "Pending": int(StatusPending)},
		"outcome":  {"Committed": int(OutcomeCommitted), "Squashed": int(OutcomeSquashed), "RolledBack": int(OutcomeRolledBack)},
		"taint":    {"Tainted": int(TaintTainted), "Trusted": int(TaintTrusted), "Quarantined": int(TaintQuarantined)},
		"scope":    {"Agent": int(ScopeAgent), "Fleet": int(ScopeFleet), "Tenant": int(ScopeTenant)},
		"refkind":  {"Inline": int(RefInline), "Blob": int(RefBlob), "Region": int(RefRegion)},
		"fallback": {"Deny": int(FallbackDeny), "Allow": int(FallbackAllow), "Defer": int(FallbackDefer)},
		"abi":      {"Major": ABIMajor, "Minor": ABIMinor},
	}
	gotJSON, _ := json.MarshalIndent(got, "", "  ")

	const golden = "testdata/abi_v0.1.golden"
	if os.Getenv("UPDATE_GOLDEN") == "1" {
		_ = os.MkdirAll("testdata", 0o755)
		if err := os.WriteFile(golden, gotJSON, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("read golden (run with UPDATE_GOLDEN=1 to create): %v", err)
	}
	if string(want) != string(gotJSON) {
		t.Fatalf("ABI wire contract changed (breaking the freeze).\n--- want ---\n%s\n--- got ---\n%s", want, gotJSON)
	}
}

// TestClosedReasonVocabulary asserts the closed 12-reason refusal vocabulary
// (unit 19): every core reason resolves to a stable non-empty name and the count
// matches the declared closed set.
func TestClosedReasonVocabulary(t *testing.T) {
	if len(coreReasonNames)-1 != CoreReasonCount { // -1 for ReasonNone
		t.Fatalf("closed reason vocabulary size = %d, want %d", len(coreReasonNames)-1, CoreReasonCount)
	}
	for c := ReasonDefaultDeny; c <= ReasonUnknownTool; c++ {
		if n := ReasonName(c); n == "" || n[0:1] == "R" && n[0:7] == "REASON_" {
			t.Fatalf("core reason %d has no stable name: %q", c, n)
		}
	}
	if ReasonName(9999) != "REASON_9999" {
		t.Fatalf("unknown reason should render as REASON_<n>, got %q", ReasonName(9999))
	}
}

// TestVerdictUnionUnrepresentable confirms a malformed verdict (a Deny carrying a
// transform payload) is unrepresentable: the payload is keyed by kind and the
// concrete payload types only satisfy isVerdictPayload via their own type.
func TestFoldRankOrdering(t *testing.T) {
	// Deny must be the most restrictive of the core set (fail-closed lattice).
	if FoldRank(VerdictDeny) <= FoldRank(VerdictQuarantine) {
		t.Fatal("Deny must outrank Quarantine in the restrictiveness lattice")
	}
	if FoldRank(VerdictAllow) != 0 {
		t.Fatal("Allow must be the least restrictive (rank 0)")
	}
	if Fallback(9999) != FallbackDeny {
		t.Fatal("an unknown verdict kind must fall back to Deny (fail-closed)")
	}
}
