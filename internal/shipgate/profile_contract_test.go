package shipgate

// profile_contract_test.go pins the structural contract of the EvidenceProfile map
// for issue #681. The behavioral equivalence (ClassFull == legacy all-three AND) is
// proven in evidence_profile_test.go; this is the complementary REGRESSION GUARD on
// the map's literal shape: every declared class, the exact required-signal subset of
// each, and the non-forgeability floor that NO class may drop the truth-clean signal.
// A future edit that silently weakens a profile (e.g. drops needTruth) reddens here.

import "testing"

// TestEvidenceProfileMapContract asserts the EvidenceProfile map declares exactly the
// three graduated classes with exactly the signal subsets issue #681's family pins:
// ClassFull is the legacy all-three AND, ClassDocsOnly needs truth alone, and
// ClassProofCarrying needs gain plus truth. The map is checked against an independent
// expectation table so a drift in either direction (a missing class, an extra class, a
// flipped need-bit) fails.
func TestEvidenceProfileMapContract(t *testing.T) {
	want := map[EvidenceClass]Profile{
		ClassFull:          {needGain: true, needSuite: true, needTruth: true},
		ClassDocsOnly:      {needTruth: true},
		ClassProofCarrying: {needGain: true, needTruth: true},
	}
	if len(EvidenceProfile) != len(want) {
		t.Fatalf("EvidenceProfile has %d classes, want %d: %+v", len(EvidenceProfile), len(want), EvidenceProfile)
	}
	for c, exp := range want {
		got, ok := EvidenceProfile[c]
		if !ok {
			t.Fatalf("EvidenceProfile is missing class %s", c)
		}
		if got != exp {
			t.Fatalf("EvidenceProfile[%s] = %+v, want %+v", c, got, exp)
		}
	}
}

// TestClassFullProfileIsLegacyAllThree pins criterion 1 at the profile level: the
// ClassFull zero value requires every one of the three signals, so its keep rule is the
// legacy AND of gain, suite, and truth — no signal is droppable for the default class.
func TestClassFullProfileIsLegacyAllThree(t *testing.T) {
	p := ProfileFor(ClassFull)
	if !(p.needGain && p.needSuite && p.needTruth) {
		t.Fatalf("ClassFull profile is not the all-three AND: %+v", p)
	}
}

// TestEveryProfileKeepsTheTruthFloor proves the non-forgeability floor that makes the
// graduated classes safe: every declared profile — and the ClassFull fallback an
// unknown class resolves to — still REQUIRES the truth-clean signal. A class may drop
// the metric or the suite, but none may ever keep without a clean truth syscall, so no
// narrowing can manufacture a keep from an author-controlled input alone.
func TestEveryProfileKeepsTheTruthFloor(t *testing.T) {
	for c, p := range EvidenceProfile {
		if !p.needTruth {
			t.Fatalf("class %s drops the truth-clean floor: %+v", c, p)
		}
	}
	if !ProfileFor(EvidenceClass(0xFF)).needTruth {
		t.Fatalf("unknown-class fallback dropped the truth-clean floor")
	}
}
