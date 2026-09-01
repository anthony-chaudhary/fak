package issuepolicy

import (
	"testing"
	"time"
)

func evidence(status, fp, module, class, commit, witness string) ErrorEvidence {
	return ErrorEvidence{Status: status, Fingerprint: fp, Module: module, FailureClass: class, Commit: commit, ModuleVersion: module + "@r1+g" + commit, Witness: witness}
}

func TestBuildErrorInventoryDispositionMatrix(t *testing.T) {
	fail := evidence("fail", "fp", "internal/x", "panic", "aaa", "observed")
	pass := evidence("pass", "", "internal/x", "", "ccc", "current")
	cases := []struct {
		name  string
		obs   []ErrorObservation
		issue int
		want  ErrorDisposition
		exit  int
	}{
		{"duplicate proven", []ErrorObservation{{Issue: 1, Observed: fail}, {Issue: 2, Observed: fail}}, 2, DispositionDuplicateProven, 3},
		{"possible duplicate", []ErrorObservation{{Issue: 3, Observed: fail, PossibleDuplicate: 9}}, 3, DispositionPossibleDuplicate, 4},
		{"fix on trunk", []ErrorObservation{{Issue: 4, Observed: fail, Fix: evidence("pass", "", "internal/x", "", "bbb", "fix"), Current: pass}}, 4, DispositionFixPresentTrunk, 3},
		{"fix released ancestry", []ErrorObservation{{Issue: 5, Observed: fail, Fix: evidence("pass", "", "internal/x", "", "bbb", "fix"), Current: pass, Releases: []ReleaseEvidence{{Tag: "v1.2.0", FixAncestor: true, Witness: "git"}}}}, 5, DispositionFixReleased, 3},
		{"fix released receipt", []ErrorObservation{{Issue: 6, Observed: fail, Fix: evidence("pass", "", "internal/x", "", "bbb", "fix"), Current: pass, Releases: []ReleaseEvidence{{Tag: "v1.2.0", PassingReceipt: true, Witness: "release-ci"}}}}, 6, DispositionFixReleased, 3},
		{"actionable", []ErrorObservation{{Issue: 7, Observed: fail, Current: evidence("fail", "fp", "internal/x", "panic", "ccc", "current")}}, 7, DispositionActionable, 0},
		{"repro required", []ErrorObservation{{Issue: 8, Observed: ErrorEvidence{}}}, 8, DispositionReproRequired, 4},
		{"stale", []ErrorObservation{{Issue: 9, Observed: ErrorEvidence{Stale: true}}}, 9, DispositionStaleEvidence, 4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rep, err := BuildErrorInventory(ErrorInventoryInput{GeneratedAt: time.Unix(1, 0), SnapshotDigest: "sha256:x", Observations: tc.obs})
			if err != nil {
				t.Fatal(err)
			}
			var got ErrorInventoryIssue
			for _, row := range rep.Issues {
				if row.Issue == tc.issue {
					got = row
				}
			}
			if got.Disposition != tc.want {
				t.Fatalf("disposition=%s want %s", got.Disposition, tc.want)
			}
			if ActionabilityExit(got.Disposition) != tc.exit {
				t.Fatalf("exit=%d want %d", ActionabilityExit(got.Disposition), tc.exit)
			}
		})
	}
}

func TestErrorInventoryPreservesThreeProvenancePoints(t *testing.T) {
	obs := ErrorObservation{Issue: 1,
		Observed: evidence("fail", "fp", "internal/x", "panic", "aaa", "observed"),
		Fix:      evidence("pass", "", "internal/x", "", "bbb", "fix"),
		Current:  evidence("pass", "", "internal/x", "", "ccc", "current")}
	rep, err := BuildErrorInventory(ErrorInventoryInput{GeneratedAt: time.Unix(1, 0), SnapshotDigest: "sha256:x", Observations: []ErrorObservation{obs}})
	if err != nil {
		t.Fatal(err)
	}
	row := rep.Issues[0]
	if row.ObservedFailure.Commit != "aaa" || row.ProvenFix.Commit != "bbb" || row.CurrentTestedState.Commit != "ccc" {
		t.Fatalf("provenance collapsed: %+v", row)
	}
}

func TestCurrentRegressionOutranksPossibleDuplicate(t *testing.T) {
	fail := evidence("fail", "fp", "internal/x", "panic", "aaa", "observed")
	rep, err := BuildErrorInventory(ErrorInventoryInput{GeneratedAt: time.Unix(1, 0), SnapshotDigest: "sha256:x", Observations: []ErrorObservation{{Issue: 1, Observed: fail, Current: fail, PossibleDuplicate: 2}}})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Issues[0].Disposition != DispositionActionable {
		t.Fatalf("got %s", rep.Issues[0].Disposition)
	}
}
