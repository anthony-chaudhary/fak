package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modver"
)

func TestVersionScoreAdapterFeedsModuleJoin(t *testing.T) {
	fixture := `{"files":[
		{"path":"internal/gateway/a.go","score":80},
		{"path":"internal/gateway/b.go","score":60},
		{"path":"cmd/fak/main.go","score":90}
	]}`
	path := filepath.Join(t.TempDir(), "scorecard.json")
	if err := os.WriteFile(path, []byte(fixture), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errOut strings.Builder
	if rc := runVersionScoreAdapter(&out, &errOut, []string{"--input", path}); rc != 0 {
		t.Fatalf("score-adapter rc=%d stderr=%s", rc, errOut.String())
	}
	scores, err := modver.LoadScores([]byte(out.String()))
	if err != nil {
		t.Fatalf("LoadScores(adapter output): %v", err)
	}
	rep := modver.Report{Modules: []modver.Module{
		{Name: "internal/gateway"}, {Name: "cmd/fak"}, {Name: "internal/other"},
	}}
	if matched := rep.JoinScores(scores); matched != 2 {
		t.Fatalf("joined %d scores, want 2", matched)
	}
	if rep.Modules[0].Score == nil || *rep.Modules[0].Score != 70 {
		t.Fatalf("gateway score = %v, want mean 70", rep.Modules[0].Score)
	}
	if rep.Modules[1].Score == nil || *rep.Modules[1].Score != 90 {
		t.Fatalf("cmd/fak score = %v, want 90", rep.Modules[1].Score)
	}
	if rep.Modules[2].Score != nil {
		t.Fatalf("unmatched module acquired score %v", *rep.Modules[2].Score)
	}
}
