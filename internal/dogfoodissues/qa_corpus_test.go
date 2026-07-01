package dogfoodissues

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

func TestQADogfoodCorpusMapsLateFailuresToOriginControlsAndIssueBodies(t *testing.T) {
	rows := QADogfoodCorpus()
	if len(rows) < 4 {
		t.Fatalf("corpus rows = %d, want at least 4", len(rows))
	}

	seen := map[string]bool{}
	for _, row := range rows {
		if row.Key == "" || row.LateFailure == "" || row.OriginControl == "" || row.WitnessCommand == "" {
			t.Fatalf("incomplete corpus row: %+v", row)
		}
		if seen[row.Key] {
			t.Fatalf("duplicate corpus key %q", row.Key)
		}
		seen[row.Key] = true

		review := ReviewActionItem(row.Item, BuildOptions{Live: true, DedupeChecked: true, DedupeCap: 300})
		if !review.OK || review.Dispatchability != "dispatchable" {
			t.Fatalf("%s review = %+v, want dispatchable", row.Key, review)
		}
		plan, skipped := BuildPlanWithOptions([]ActionItem{row.Item}, nil, BuildOptions{Live: true, DedupeChecked: true, DedupeCap: 300})
		if len(skipped) != 0 || len(plan) != 1 {
			t.Fatalf("%s plan=%+v skipped=%+v, want one dispatchable issue", row.Key, plan, skipped)
		}
		body := plan[0].Body
		for _, want := range []string{
			row.LateFailure,
			row.OriginControl,
			row.WitnessCommand,
			"## Done condition",
			"## Witness",
			"Path hints:",
		} {
			if !strings.Contains(body, want) {
				t.Fatalf("%s issue body missing %q:\n%s", row.Key, want, body)
			}
		}

		assertQADogfoodOriginControl(t, row)
	}
}

func assertQADogfoodOriginControl(t *testing.T, row QADogfoodCorpusRow) {
	t.Helper()
	switch row.Key {
	case "qa-dogfood-corpus/task-output-shape":
		m := taskmgr.NewManager()
		task, err := m.StartTask(taskmgr.TaskSpec{TaskID: "shape_origin"})
		if err != nil {
			t.Fatalf("start shape task: %v", err)
		}
		rec, err := task.BeatWithEvidence([]byte(strings.Repeat("LOOP ", 80)))
		if err != nil {
			t.Fatalf("shape beat witness: %v", err)
		}
		if rec.VerifiedState != taskmgr.VerifiedRefused || m.Snapshot().Tasks[0].VerifiedProgressing() {
			t.Fatalf("shape origin control did not refuse progress: rec=%+v snap=%+v", rec, m.Snapshot().Tasks[0])
		}
	case "qa-dogfood-corpus/task-origin-path-evidence":
		m := taskmgr.NewManager(taskmgr.WithDefaultOriginWitnesses())
		ref := taskmgr.EvidenceRef{Kind: taskmgr.PathRefKind, Ref: filepath.Join(t.TempDir(), "missing-artifact.txt")}
		if _, err := m.StartTask(taskmgr.TaskSpec{TaskID: "path_origin", EvidenceRefs: []taskmgr.EvidenceRef{ref}}); err != nil {
			t.Fatalf("start path task: %v", err)
		}
		got := m.Snapshot().Tasks[0].Witness
		if got == nil || got.VerifiedState != taskmgr.VerifiedRefused {
			t.Fatalf("path origin control witness = %+v, want verified_refused", got)
		}
	case "qa-dogfood-corpus/handoff-evidence-refs":
		h := taskmgr.DraftHandoffFromTask(taskmgr.TaskSnapshot{
			TaskID: "handoff_origin",
			State:  taskmgr.StateDone,
			Witness: &taskmgr.WitnessRecord{
				VerifiedState: taskmgr.VerifiedDone,
				Source:        "commit-audit",
			},
		}, taskmgr.HandoffDraftOptions{
			CurrentState: "Handoff origin has producer-side evidence.",
			Evidence: taskmgr.HandoffEvidenceInputs{
				ChangedPaths: []string{"internal/taskmgr/handoff.go"},
				TestCommands: []string{"go test ./internal/taskmgr ./cmd/fak -run Handoff.*Evidence"},
			},
		})
		if !handoffHasEvidence(h.CompletionEvidence, taskmgr.PathRefKind, "internal/taskmgr/handoff.go") ||
			!handoffHasEvidence(h.CompletionEvidence, taskmgr.TestRefKind, "go test ./internal/taskmgr ./cmd/fak -run Handoff.*Evidence") {
			t.Fatalf("handoff origin evidence refs = %+v", h.CompletionEvidence)
		}
	case "qa-dogfood-corpus/dogfood-issue-scope":
		vague := row.Item
		vague.WorkingSpine = ""
		vague.InScope = ""
		vague.DoneCondition = ""
		vague.Witness = ""
		vague.Lane = ""
		vague.Paths = nil
		plan, skipped := BuildPlanWithOptions([]ActionItem{vague}, nil, BuildOptions{})
		if len(plan) != 0 || len(skipped) != 1 {
			t.Fatalf("vague issue scope plan=%+v skipped=%+v, want one skipped row", plan, skipped)
		}
		if !strings.Contains(skipped[0].Reason, "ISSUE_SCOPE_INCOMPLETE") ||
			!strings.Contains(skipped[0].Reason, "ISSUE_UNROUTED") {
			t.Fatalf("vague issue skip reason = %q", skipped[0].Reason)
		}
	default:
		t.Fatalf("row %q has no origin-control assertion", row.Key)
	}
}

func handoffHasEvidence(refs []taskmgr.EvidenceRef, kind, ref string) bool {
	for _, got := range refs {
		if got.Kind == kind && got.Ref == ref {
			return true
		}
	}
	return false
}
