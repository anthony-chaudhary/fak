package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/orgdebt"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

func TestOrgDebtRouteRegistered(t *testing.T) {
	if scoreRoutes["org-debt"] == nil {
		t.Fatal("scoreRoutes[org-debt] is nil")
	}
}

func TestRunOrgDebtScoreText(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runOrgDebtScore(&stdout, &stderr, []string{"--workspace", repoRoot(), "--limit", "10"})
	if code != 0 {
		t.Fatalf("runOrgDebtScore exit code %d, stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "Organization Debt Scorecard:") {
		t.Fatalf("expected header in output, got: %s", out)
	}
	if !strings.Contains(out, "KPI breakdown:") {
		t.Fatalf("expected KPI breakdown in output, got: %s", out)
	}
}

func TestRunOrgDebtScoreJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runOrgDebtScore(&stdout, &stderr, []string{"--json", "--workspace", repoRoot(), "--limit", "10"})
	if code != 0 {
		t.Fatalf("runOrgDebtScore exit code %d, stderr: %s", code, stderr.String())
	}
	var rep scorecard.Payload
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("failed to unmarshal json output: %v, raw: %s", err, stdout.String())
	}
	if rep.Schema != orgdebt.Schema {
		t.Fatalf("schema got %q, want %q", rep.Schema, orgdebt.Schema)
	}
	if len(rep.KPIs) == 0 {
		t.Fatal("expected KPIs to be present in json report")
	}
}

func TestRunOrgDebtScoreUnexpectedArg(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runOrgDebtScore(&stdout, &stderr, []string{"unexpected_extra"})
	if code != 2 {
		t.Fatalf("expected exit code 2 for unexpected arg, got %d", code)
	}
}
