package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/discoveryrouter"
)

func TestRunDiscoveryJSONReportsSkippedSource(t *testing.T) {
	var out, errOut bytes.Buffer
	root := filepath.Clean(filepath.Join("..", ".."))
	if rc := runDiscovery(&out, &errOut, []string{"--json", "--skip-sessions", "--root", root, "native inference goal"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errOut.String())
	}
	var got discoveryrouter.Report
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != discoveryrouter.Schema || got.CoverageComplete || len(got.Results) == 0 {
		t.Fatalf("report=%+v", got)
	}
	if len(got.Coverage) != 2 || got.Coverage[1].Status != discoveryrouter.Skipped {
		t.Fatalf("coverage=%+v", got.Coverage)
	}
}
