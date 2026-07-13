package rungobs

import (
	"strings"
	"testing"
)

// TestCostLedgerFaithfulFoldsReproduceReference is the passing half of the
// witness: every faithful fold (each a different replay order of the same month)
// reproduces the reference ledger byte-for-byte, passes the pre-publish gate,
// carries complete provenance, and is judged pass.
func TestCostLedgerFaithfulFoldsReproduceReference(t *testing.T) {
	ref := foldLedger(costLedgerFixtureMonth())
	baseline := costLedgerFingerprint(ref)

	for _, f := range costLedgerFaithfulFolds() {
		got := f.fold(costLedgerFixtureMonth())
		if err := costPublishGate(got, baseline); err != nil {
			t.Fatalf("faithful fold %q was refused before publish: %v", f.name, err)
		}
		prov := costProvenanceOf("cost-ledger/"+f.name, got, f.backend, baseline)
		if !prov.complete() {
			t.Fatalf("faithful fold %q has incomplete provenance: %+v", f.name, prov)
		}
		v := costJudge(ref, got, prov)
		if !v.Pass {
			t.Fatalf("faithful fold %q diverged from reference: %s", f.name, v.Detail)
		}
	}
}

// TestCostLedgerPlantedDefectsFailBeforePublishAndLocalize is the failing half of
// the witness: each planted representative defect fails the pre-publish gate,
// actually perturbs the ledger, is judged fail, and emits a scrubbed replay
// artifact localizing the first divergent row.
func TestCostLedgerPlantedDefectsFailBeforePublishAndLocalize(t *testing.T) {
	ref := foldLedger(costLedgerFixtureMonth())
	baseline := costLedgerFingerprint(ref)

	defects := []struct {
		name string
		fold func([]suiteRun) costLedger
	}{
		{"double-count", costDoubleCountFold},
		{"dropped-judge-spend", costDroppedComponentFold},
		{"mis-tiered-suite", costMisTierFold},
	}

	for _, tc := range defects {
		got := tc.fold(costLedgerFixtureMonth())

		// Acceptance: a defective fold fails BEFORE the ledger is published.
		if err := costPublishGate(got, baseline); err == nil {
			t.Fatalf("defect %q passed the pre-publish gate — it must fail before publish", tc.name)
		}

		// The defect must actually change the ledger (else it is not a witness).
		wantIdx := costFirstDiff(ref, got)
		if wantIdx < 0 {
			t.Fatalf("defect %q did not change the ledger — not a representative defect", tc.name)
		}

		prov := costProvenanceOf("cost-ledger/"+tc.name, got, "defect", baseline)
		v := costJudge(ref, got, prov)
		if v.Pass {
			t.Fatalf("defect %q was judged pass — a planted defect must fail", tc.name)
		}
		if v.Artifact == nil || v.Artifact.Divergence == nil {
			t.Fatalf("defect %q produced no replay artifact", tc.name)
		}
		if got := v.Artifact.Divergence.Index; got != wantIdx {
			t.Fatalf("defect %q first divergence reported at row %d, want %d (detail: %s)", tc.name, got, wantIdx, v.Detail)
		}
		// The artifact renders scrubbed provenance (suite names + tiers + counts)
		// and never the raw per-suite defect-ID membership.
		s := v.Artifact.String()
		if !strings.Contains(s, "suites=") || !strings.Contains(s, "unit:PR") {
			t.Fatalf("defect %q replay artifact is not renderable/scrubbed: %s", tc.name, s)
		}
		if strings.Contains(s, "D-101") || strings.Contains(s, "D-201") || strings.Contains(s, "D-301") {
			t.Fatalf("defect %q replay artifact leaked raw defect-ID membership: %s", tc.name, s)
		}
	}
}

// TestCostLedgerNoDoubleCountInvariant asserts the acceptance invariant directly:
// the reference ledger's summed per-suite unique defects equal its distinct-defect
// count, and the double-count fold violates it.
func TestCostLedgerNoDoubleCountInvariant(t *testing.T) {
	ref := foldLedger(costLedgerFixtureMonth())
	if !noDoubleCount(ref) {
		t.Fatalf("reference ledger double-counts: summed unique %d != distinct %d", sumUnique(ref), ref.distinctDefects)
	}
	if want := 4; ref.distinctDefects != want {
		t.Fatalf("reference distinct defects = %d, want %d", ref.distinctDefects, want)
	}
	bad := costDoubleCountFold(costLedgerFixtureMonth())
	if noDoubleCount(bad) {
		t.Fatalf("double-count fold satisfied the invariant — it must not (summed unique %d, distinct %d)", sumUnique(bad), bad.distinctDefects)
	}
}

// TestCostLedgerInconclusiveIsNeverPass asserts that an empty candidate ledger
// (no evidence) is judged fail and still emits a replay artifact.
func TestCostLedgerInconclusiveIsNeverPass(t *testing.T) {
	ref := foldLedger(costLedgerFixtureMonth())
	baseline := costLedgerFingerprint(ref)
	prov := costProvenanceOf("cost-ledger/empty", costLedger{}, "empty-fold", baseline)
	v := costJudge(ref, costLedger{}, prov)
	if v.Pass {
		t.Fatalf("an empty ledger was judged pass — inconclusive evidence must never pass")
	}
	if v.Artifact == nil {
		t.Fatalf("inconclusive case produced no replay artifact")
	}
}

// TestCostLedgerProvenanceComplete asserts every faithful fold carries complete,
// PR-tier provenance.
func TestCostLedgerProvenanceComplete(t *testing.T) {
	ref := foldLedger(costLedgerFixtureMonth())
	baseline := costLedgerFingerprint(ref)
	for _, f := range costLedgerFaithfulFolds() {
		prov := costProvenanceOf("cost-ledger/"+f.name, f.fold(costLedgerFixtureMonth()), f.backend, baseline)
		if !prov.complete() {
			t.Fatalf("fold %q incomplete provenance: %+v", f.name, prov)
		}
		if prov.Tier != "PR" {
			t.Fatalf("fold %q tier=%q, want PR", f.name, prov.Tier)
		}
	}
}
