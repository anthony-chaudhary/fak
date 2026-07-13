package quality

import "testing"

func completeReleaseProvenance(rev string) EvidenceProvenance {
	return EvidenceProvenance{Model: "model-r1", Tokenizer: "tok-r1", Engine: "cpu/deterministic", Seed: 7, Oracle: "exact-r1", Revision: rev, Baseline: "base-r1/tol-exact"}
}

func TestQualifyReleaseFaithfulAndPlantedDefect(t *testing.T) {
	rev := "artifact-r1"
	oracles, err := Lookup(DemoCase().Oracles)
	if err != nil {
		t.Fatal(err)
	}
	good, err := RunCase(DemoCase(), ReferenceRunner{}, DemoEngine(""), oracles)
	if err != nil {
		t.Fatal(err)
	}
	bad, err := RunCase(DemoCase(), ReferenceRunner{}, DemoEngine("decode"), oracles)
	if err != nil {
		t.Fatal(err)
	}
	required := []RequiredGate{{CaseID: DemoCase().ID, Kind: KindDeterministic, Tier: TierRelease, CostSeconds: 1}}
	pass := QualifyRelease(rev, required, []Evidence{EvidenceFromResult(completeReleaseProvenance(rev), good)})
	if !pass.Released || len(pass.Blocks) != 0 {
		t.Fatalf("faithful evidence must release: %+v", pass)
	}
	fail := QualifyRelease(rev, required, []Evidence{EvidenceFromResult(completeReleaseProvenance(rev), bad)})
	if fail.Released || len(fail.Blocks) != 1 || fail.Blocks[0].State != StateFail {
		t.Fatalf("planted defect must block: %+v", fail)
	}
	if fail.Blocks[0].FirstDivergence == nil || fail.Blocks[0].Replay == nil {
		t.Fatalf("block must preserve divergence and replay: %+v", fail.Blocks[0])
	}
}

func TestQualifyReleaseFailsClosedOnMissingStaleAndInconclusive(t *testing.T) {
	rev := "artifact-r1"
	required := []RequiredGate{{CaseID: "det", Kind: KindDeterministic, Tier: TierRelease}, {CaseID: "stats", Kind: KindStatistical, Tier: TierRelease}, {CaseID: "hw", Kind: KindHardware, Tier: TierRelease}, {CaseID: "report", Kind: KindReport, Tier: TierRelease}}
	for _, tc := range []struct {
		name     string
		evidence []Evidence
	}{
		{"missing", nil},
		{"stale", []Evidence{{CaseID: "det", State: StatePass, Provenance: completeReleaseProvenance("old")}}},
		{"inconclusive", []Evidence{{CaseID: "det", State: StateInconclusive, Provenance: completeReleaseProvenance(rev)}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := QualifyRelease(rev, required, tc.evidence)
			if d.Released || len(d.Blocks) == 0 {
				t.Fatalf("%s evidence released: %+v", tc.name, d)
			}
		})
	}
}
