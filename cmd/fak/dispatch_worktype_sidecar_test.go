package main

import (
	"encoding/json"
	"github.com/anthony-chaudhary/fak/internal/worktype"
	"os"
	"path/filepath"
	"testing"
)

func TestWriteDispatchWorktypeSidecarFeedsSpendJoin(t *testing.T) {
	d := t.TempDir()
	log := filepath.Join(d, "resolve-cmd-8084.log")
	path := writeDispatchWorktypeSidecar(log, "Issue title: feat(worktype): emit sidecars")
	if path == "" {
		t.Fatal("no sidecar")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var got worktype.Classification
	if json.Unmarshal(b, &got) != nil {
		t.Fatal(string(b))
	}
	if got.Schema != worktype.ClassificationSchema || got.SessionID != "resolve-cmd-8084" || got.PatternID != "wp.spec-to-feature" {
		t.Fatalf("%+v", got)
	}
	classes, err := readSpendClassesForTest(path)
	if err != nil || classes[got.SessionID].PatternID != got.PatternID {
		t.Fatalf("consume err=%v classes=%+v", err, classes)
	}
}
func TestWriteDispatchWorktypeSidecarUnknownAndNoSpawn(t *testing.T) {
	d := t.TempDir()
	if got := writeDispatchWorktypeSidecar("", "ambiguous"); got != "" {
		t.Fatal(got)
	}
	p := writeDispatchWorktypeSidecar(filepath.Join(d, "worker.log"), "Investigate")
	b, _ := os.ReadFile(p)
	var got worktype.Classification
	_ = json.Unmarshal(b, &got)
	if got.PatternID != "unknown" {
		t.Fatalf("%+v", got)
	}
}
func readSpendClassesForTest(path string) (map[string]worktypeSpendClass, error) {
	m := map[string]worktypeSpendClass{}
	e := scanSpendJSONL(path, func(b []byte) error {
		var x worktypeSpendClass
		if e := json.Unmarshal(b, &x); e != nil {
			return e
		}
		m[x.TraceID] = x
		return nil
	})
	return m, e
}
