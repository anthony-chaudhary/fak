package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/overtonscore"
)

func TestOvertonScoreRouteRegistered(t *testing.T) {
	if scoreRoutes["overton"] == nil {
		t.Fatal("scoreRoutes[overton] is nil")
	}
}

func TestRunOvertonScoreText(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runOvertonScore(&stdout, &stderr, []string{"--workspace", repoRoot()})
	if code != 0 {
		t.Fatalf("runOvertonScore exit code %d, stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "Overton Baseline Evaluation:") {
		t.Fatalf("expected header in output, got: %s", out)
	}
	if !strings.Contains(out, "Dispositions:") {
		t.Fatalf("expected Dispositions in output, got: %s", out)
	}
}

func TestRunOvertonScoreJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runOvertonScore(&stdout, &stderr, []string{"--json", "--workspace", repoRoot()})
	if code != 0 {
		t.Fatalf("runOvertonScore exit code %d, stderr: %s", code, stderr.String())
	}
	var rep overtonscore.Report
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("failed to unmarshal json output: %v, raw: %s", err, stdout.String())
	}
	if rep.Schema != overtonscore.Schema {
		t.Fatalf("schema got %q, want %q", rep.Schema, overtonscore.Schema)
	}
	if len(rep.Evaluations) == 0 {
		t.Fatal("expected evaluations to be present in json report")
	}
}

func TestRunOvertonScoreUnexpectedArg(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runOvertonScore(&stdout, &stderr, []string{"unexpected_extra"})
	if code != 2 {
		t.Fatalf("expected exit code 2 for unexpected arg, got %d", code)
	}
}
