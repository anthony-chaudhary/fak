package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelaccept"
)

func TestRefoldAcceptanceReportUsesCurrentParserAndPreservesIdentity(t *testing.T) {
	in := topThreeInputForRefold()
	rawDir := t.TempDir()
	for _, run := range in.Runs {
		task := in.Corpus.Tasks[0]
		raw := streamFixture(run.Model, task.Expected, []string{"mcp__acceptance__flaky_lookup", "mcp__acceptance__flaky_lookup"}, []bool{true, false})
		name := run.Model + "--" + task.ID + "--01.jsonl"
		if err := os.WriteFile(filepath.Join(rawDir, name), raw, 0600); err != nil {
			t.Fatal(err)
		}
	}
	got, err := refoldAcceptanceReport(in, rawDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Runs) != len(in.Runs) {
		t.Fatalf("runs=%d want %d", len(got.Runs), len(in.Runs))
	}
	for _, run := range got.Runs {
		if !run.Recovered || run.RetryCount != 1 {
			t.Fatalf("run=%+v want recovered retry", run)
		}
		if run.ObservedAt != "2026-07-15T01:00:00-07:00" {
			t.Fatalf("observed_at changed: %q", run.ObservedAt)
		}
	}
}

func TestRefoldAcceptanceReportFailsClosed(t *testing.T) {
	tests := []struct {
		name string
		edit func(*modelaccept.Input, string)
		want string
	}{
		{"missing prior", func(in *modelaccept.Input, _ string) { in.Runs = in.Runs[:len(in.Runs)-1] }, "prior report has"},
		{"duplicate prior", func(in *modelaccept.Input, _ string) { in.Runs = append(in.Runs, in.Runs[0]) }, "duplicate prior"},
		{"missing raw", func(_ *modelaccept.Input, dir string) { os.Remove(filepath.Join(dir, "exact-a--retry--01.jsonl")) }, "missing raw stream"},
		{"unexpected raw", func(_ *modelaccept.Input, dir string) {
			os.WriteFile(filepath.Join(dir, "surprise.jsonl"), []byte("{}\n"), 0600)
		}, "unexpected raw stream"},
		{"model mismatch", func(_ *modelaccept.Input, dir string) {
			os.WriteFile(filepath.Join(dir, "exact-a--retry--01.jsonl"), streamFixture("wrong-model", "RECOVERED", nil, nil), 0600)
		}, "requested"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			in := topThreeInputForRefold()
			dir := writeRefoldFixtures(t, in)
			tt.edit(&in, dir)
			_, err := refoldAcceptanceReport(in, dir)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err=%v want %q", err, tt.want)
			}
		})
	}
}

func topThreeInputForRefold() modelaccept.Input {
	task := modelaccept.Task{ID: "retry", Tier: 2, Repetitions: 1, Expected: "RECOVERED", ToolRequired: true, MinToolCalls: 2, RetryRequired: true, RecoveryRequired: true}
	models := []modelaccept.ModelRequest{{Model: "exact-a", Family: "exact-a", Generation: "current", Lifecycle: modelaccept.LifecycleLatest, RequestedTier: 0}, {Model: "exact-b", Family: "exact-b", Generation: "current", Lifecycle: modelaccept.LifecycleLatest, RequestedTier: 1}}
	runs := []modelaccept.Run{}
	for _, m := range models {
		runs = append(runs, modelaccept.Run{Model: m.Model, ActualModel: m.Model, Task: task.ID, Repetition: 1, ObservedAt: "2026-07-15T01:00:00-07:00"})
	}
	return modelaccept.Input{Schema: modelaccept.Schema, Corpus: modelaccept.Corpus{ID: "refold", DeclaredAt: "2026-07-15T00:00:00-07:00", Tasks: []modelaccept.Task{task}}, Models: models, Runs: runs}
}
func writeRefoldFixtures(t *testing.T, in modelaccept.Input) string {
	t.Helper()
	dir := t.TempDir()
	task := in.Corpus.Tasks[0]
	for _, run := range in.Runs {
		name := run.Model + "--" + task.ID + "--01.jsonl"
		if err := os.WriteFile(filepath.Join(dir, name), streamFixture(run.Model, task.Expected, []string{"mcp__acceptance__flaky_lookup", "mcp__acceptance__flaky_lookup"}, []bool{true, false}), 0600); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}
