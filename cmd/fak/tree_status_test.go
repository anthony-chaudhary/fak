package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/treestatus"
)

func TestTreeUsage(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runTree(&out, &errOut, []string{"--help"})
	if code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	if !strings.Contains(out.String(), "fak tree status") {
		t.Fatalf("usage output missing status command: %s", out.String())
	}
}

func TestTreeStatusJSON(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runTreeStatus(&out, &errOut, []string{"--json"})
	if code != 0 {
		t.Fatalf("exit code = %d, want 0, stderr = %s", code, errOut.String())
	}

	var rep treestatus.Report
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal json: %v, raw: %s", err, out.String())
	}
	if rep.Branch == "" {
		t.Error("expected branch in tree status")
	}
}

func TestTreeStatusLaneFilter(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runTreeStatus(&out, &errOut, []string{"--lane", "gateway", "--json"})
	if code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}

	var rep treestatus.Report
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal json: %v", err)
	}

	for _, p := range rep.OwnedPaths {
		if p.Lane != "gateway" {
			t.Errorf("expected owned path %s to belong to gateway, got %s", p.Path, p.Lane)
		}
	}
}
