package main

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/studylink"
)

func TestStudyLinkBuildWiresInputsAndOutputs(t *testing.T) {
	wantLedger := studylink.Ledger{Schema: "fak.study-join-ledger/1"}
	wantSummary := studylink.Summary{Counts: map[studylink.Disposition]int{studylink.Uncovered: 1}}
	var gotOptions studylink.BuildOptions
	var gotLedgerPath, gotSummaryPath string
	ops := studyLinkOperations{
		build: func(options studylink.BuildOptions) (studylink.Ledger, studylink.Summary, error) {
			gotOptions = options
			return wantLedger, wantSummary, nil
		},
		writeLedger: func(path string, ledger studylink.Ledger) error {
			gotLedgerPath = path
			if ledger.Schema != wantLedger.Schema {
				t.Fatalf("ledger = %+v, want %+v", ledger, wantLedger)
			}
			return nil
		},
		writeSummary: func(path string, summary studylink.Summary) error {
			gotSummaryPath = path
			if summary.Counts[studylink.Uncovered] != 1 {
				t.Fatalf("summary = %+v, want %+v", summary, wantSummary)
			}
			return nil
		},
	}
	var stdout, stderr bytes.Buffer
	code := runStudyLinkWithOperations(&stdout, &stderr, []string{
		"build",
		"--index", "index.json",
		"--forge", "forge.json",
		"--adjacency", "adjacency.json",
		"--repo", "repo",
		"--out", "ledger.json",
		"--summary", "SUMMARY.md",
	}, ops)
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if gotOptions.IndexPath != "index.json" || gotOptions.ForgePath != "forge.json" || gotOptions.AdjacencyPath != "adjacency.json" || gotOptions.RepoRoot != "repo" {
		t.Fatalf("options = %+v", gotOptions)
	}
	if gotLedgerPath != "ledger.json" || gotSummaryPath != "SUMMARY.md" {
		t.Fatalf("outputs = (%q, %q)", gotLedgerPath, gotSummaryPath)
	}
	if !strings.Contains(stdout.String(), "ledger.json") || !strings.Contains(stdout.String(), "SUMMARY.md") {
		t.Fatalf("stdout = %q", stdout.String())
	}
}

func TestStudyLinkValidateWiresAllSources(t *testing.T) {
	var gotOptions studylink.ValidateOptions
	ops := studyLinkOperations{
		validate: func(options studylink.ValidateOptions) error {
			gotOptions = options
			return nil
		},
	}
	var stdout, stderr bytes.Buffer
	code := runStudyLinkWithOperations(&stdout, &stderr, []string{
		"validate",
		"--ledger", "ledger.json",
		"--index", "index.json",
		"--forge", "forge.json",
		"--adjacency", "adjacency.json",
		"--repo", "repo",
	}, ops)
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if gotOptions.LedgerPath != "ledger.json" || gotOptions.IndexPath != "index.json" || gotOptions.ForgePath != "forge.json" || gotOptions.AdjacencyPath != "adjacency.json" || gotOptions.RepoRoot != "repo" {
		t.Fatalf("options = %+v", gotOptions)
	}
	if !strings.Contains(stdout.String(), "valid study-link ledger ledger.json") {
		t.Fatalf("stdout = %q", stdout.String())
	}
}

func TestStudyLinkUsageAndErrors(t *testing.T) {
	t.Run("missing required flag", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := runStudyLinkWithOperations(&stdout, &stderr, []string{"build", "--index", "index.json"}, studyLinkOperations{})
		if code != 2 || !strings.Contains(stderr.String(), "--summary PATH") {
			t.Fatalf("code=%d stderr=%q", code, stderr.String())
		}
	})

	t.Run("validation failure", func(t *testing.T) {
		ops := studyLinkOperations{
			validate: func(studylink.ValidateOptions) error { return errors.New("closed issue") },
		}
		var stdout, stderr bytes.Buffer
		code := runStudyLinkWithOperations(&stdout, &stderr, []string{
			"validate", "--ledger", "ledger.json", "--index", "index.json", "--forge", "forge.json",
			"--adjacency", "adjacency.json", "--repo", "repo",
		}, ops)
		if code != 1 || !strings.Contains(stderr.String(), "study-link: validate: closed issue") {
			t.Fatalf("code=%d stderr=%q", code, stderr.String())
		}
	})
}
