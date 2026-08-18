package main

import (
	"bytes"
	"encoding/json"
	"github.com/anthony-chaudhary/fak/internal/disambiguation"
	"os"
	"path/filepath"
	"testing"
)

func TestDisambiguationDiffCLI(t *testing.T) {
	dir := t.TempDir()
	beforePath, afterPath := filepath.Join(dir, "before.json"), filepath.Join(dir, "after.json")
	before := `{"schema":"fak-disambiguation-index/1","entries":[{"schema":"fak-disambiguation-entry/1","identity":{"canonical_term":"old","aliases":[]},"definition":"old","contrasts":[],"scope":{"kind":"x","value":"x"},"owner":{"leaf":"x","lane":"x"},"sources":[{"kind":"go-source","locator":"x.go","revision":"x","checked_at":"2026-08-17T00:00:00Z","probe":"x"}],"freshness":{"verdict":"fresh","reason_code":"SOURCE_CURRENT","checked_at":"2026-08-17T00:00:00Z","probe":"x"},"lifecycle":{"class":"current","rollout":"on"}}]}`
	after := `{"schema":"fak-disambiguation-index/1","entries":[]}`
	if err := os.WriteFile(beforePath, []byte(before), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(afterPath, []byte(after), 0600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := runDisambiguation(&stdout, &stderr, []string{"diff", "--before", beforePath, "--after", afterPath, "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report disambiguation.DiffReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if len(report.Changes) != 1 || report.Changes[0].Kind != disambiguation.ChangeRemoval || report.Changes[0].QueryImpact != disambiguation.ImpactBreaking {
		t.Fatalf("report=%#v", report)
	}
}
