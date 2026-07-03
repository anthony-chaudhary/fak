package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modver"
)

func modverFixtureReport() modver.Report {
	score := 8.5
	return modver.Report{
		Head:       "deadbee1",
		AppVersion: "0.37.0",
		Modules: []modver.Module{
			{Name: "cmd/fak", Kind: "cmd", Rev: 12, LastCommit: "aaa11111", LastDate: "2026-07-02T10:00:00Z"},
			{Name: "internal/gateway", Kind: "internal", Rev: 5, LastCommit: "bbb22222", LastDate: "2026-07-01T09:00:00Z", Score: &score},
		},
	}
}

func TestRenderModuleReport(t *testing.T) {
	var sb strings.Builder
	renderModuleReport(&sb, modverFixtureReport())
	out := sb.String()
	for _, want := range []string{
		"head deadbee1", "app 0.37.0", "2 modules",
		"r12+gaaa11111", "2026-07-02  cmd/fak",
		"r5+gbbb22222", "internal/gateway  score 8.5",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("render missing %q in:\n%s", want, out)
		}
	}
}

func TestStampModverLedgerRoundtrip(t *testing.T) {
	root := t.TempDir()
	rep := modverFixtureReport()
	var out, errOut strings.Builder

	if code := stampModverLedger(&out, &errOut, root, "sub/ledger.jsonl", rep); code != 0 {
		t.Fatalf("first stamp exit %d: %s", code, errOut.String())
	}
	path := filepath.Join(root, "sub", "ledger.jsonl")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if lines := strings.Count(string(b), "\n"); lines != 2 {
		t.Fatalf("first stamp wrote %d rows, want 2:\n%s", lines, b)
	}
	if !strings.Contains(string(b), `"schema":"fak-module-versions/1"`) {
		t.Errorf("ledger rows missing schema tag:\n%s", b)
	}

	out.Reset()
	if code := stampModverLedger(&out, &errOut, root, "sub/ledger.jsonl", rep); code != 0 {
		t.Fatalf("second stamp exit %d: %s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "0 of 2 modules moved") {
		t.Errorf("second stamp should be a converged no-op, got: %s", out.String())
	}
}
