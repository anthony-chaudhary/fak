package sessionreset

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/taskdecision"
)

func TestTaskDecisionLogReloadsAcrossReset(t *testing.T) {
	path := filepath.Join(t.TempDir(), "decisions.jsonl")
	for _, entry := range []taskdecision.Entry{
		{TaskID: "issue-2122", Decision: "Use sessionreset contributor", Rationale: "reset seeds already auto-reload carryover parts", EvidenceRef: "code:sessionreset", OpenThreads: []string{"wire CLI"}},
		{TaskID: "issue-2122", Decision: "Keep log bounded", Rationale: "context should carry the newest task reasoning only", EvidenceRef: "test:bounded", OpenThreads: []string{"watch file growth"}},
		{TaskID: "other", Decision: "Do not load me", Rationale: "wrong task", EvidenceRef: "fixture"},
	} {
		if err := taskdecision.Append(path, entry); err != nil {
			t.Fatal(err)
		}
	}
	loaded, err := taskdecision.Load(path, "issue-2122", taskdecision.DefaultReloadLimit)
	if err != nil {
		t.Fatal(err)
	}
	seed := BuildSeed(Input{
		Trace:       "trace-after-reset",
		Messages:    sampleTranscript(),
		DecisionLog: loaded,
	})
	for _, want := range []string{"Task decision log", "Use sessionreset contributor", "Keep log bounded", "reset seeds already auto-reload", "wire CLI", "watch file growth"} {
		if !strings.Contains(seed.Recap, want) {
			t.Fatalf("seed recap missing %q:\n%s", want, seed.Recap)
		}
	}
	if strings.Contains(seed.Recap, "Do not load me") {
		t.Fatalf("seed loaded a different task's decision:\n%s", seed.Recap)
	}
	var found bool
	for _, p := range seed.Parts {
		if p.Name == "task_decision_log" {
			found = true
			if p.Meta["entries"] != "2" || p.Meta["task_id"] != "issue-2122" {
				t.Fatalf("decision log meta = %+v", p.Meta)
			}
		}
	}
	if !found {
		t.Fatalf("seed parts missing task_decision_log: %+v", seed.Parts)
	}
}
