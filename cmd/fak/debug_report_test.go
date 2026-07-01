package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cdb"
	"github.com/anthony-chaudhary/fak/internal/contextq"
)

func TestDebugReportShowsKnownUnknownAssumedContext(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	rec, _, err := cdb.IngestSession(ctx, "../../testdata/cdb/session.jsonl", "cmd-debug-context")
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}
	if err := rec.Persist(dir); err != nil {
		t.Fatalf("persist: %v", err)
	}

	tmp := t.TempDir()
	outPath := filepath.Join(tmp, "report.json")
	assumptionsPath := filepath.Join(tmp, "assumptions.json")
	if err := os.WriteFile(assumptionsPath, []byte(`[
  {
    "key": "customer-tier",
    "statement": "customer tier is gold until the account row changes",
    "source": "operator",
    "confidence": 0.8,
    "action": "verify-before-refund",
    "reason": "debug-session default"
  }
]`), 0o644); err != nil {
		t.Fatalf("write assumptions: %v", err)
	}

	stdout := captureStdout(t, func() {
		cmdDebug([]string{
			"--dir", dir,
			"--cmd", "report",
			"--query", "read context",
			"--out", outPath,
			"--assumptions", assumptionsPath,
		})
	})
	for _, want := range []string{
		"# context evidence split",
		"## known",
		"## unknown",
		"## assumed",
		"customer-tier",
		"sealed_by_trust_gate",
	} {
		if !strings.Contains(stdout, want) {
			t.Fatalf("debug report stdout missing %q:\n%s", want, stdout)
		}
	}

	var report struct {
		ContextEvidence contextq.ContextRender `json:"context_evidence"`
	}
	b, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("read report: %v", err)
	}
	if err := json.Unmarshal(b, &report); err != nil {
		t.Fatalf("decode report: %v\n%s", err, string(b))
	}
	if len(report.ContextEvidence.Known) == 0 {
		t.Fatalf("report context evidence has no known rows: %+v", report.ContextEvidence)
	}
	if len(report.ContextEvidence.Unknown) == 0 {
		t.Fatalf("report context evidence has no unknown rows: %+v", report.ContextEvidence)
	}
	if len(report.ContextEvidence.Assumed) != 1 || report.ContextEvidence.Assumed[0].Key != "customer-tier" {
		t.Fatalf("report context evidence assumptions = %+v", report.ContextEvidence.Assumed)
	}
	if report.ContextEvidence.Known[0].SourceDigest == "" {
		t.Fatalf("known row lacks source digest: %+v", report.ContextEvidence.Known[0])
	}
	if report.ContextEvidence.Unknown[0].SourceDigest == "" {
		t.Fatalf("unknown row lacks source digest: %+v", report.ContextEvidence.Unknown[0])
	}
}
