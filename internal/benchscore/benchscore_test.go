package benchscore

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestScanAcceptsVerifiedPrefillScore(t *testing.T) {
	root := t.TempDir()
	writeScore(t, root, "accepted", `{
	  "schema": "fak.arm64-qkernel-score.v1",
	  "captured_at": "2026-06-26T01:58:51Z",
	  "machine": "node-macos-a",
	  "model": {"name": "qwen2.5-1.5b-instruct", "source_kind": "HuggingFace safetensors"},
	  "results": {
	    "prefill": [{"tokens": 256, "median_ms": 100, "tok_per_sec": 1200}],
	    "decode": {"tok_per_sec": 44}
	  },
	  "baseline": {
	    "fak_cpu_q8_prefill_p256_tok_per_sec": 240,
	    "llamacpp_cpu_prefill_p256_tok_per_sec": 400,
	    "fak_cpu_q8_decode_tok_per_sec": 40
	  },
	  "improvement": {
	    "prefill_over_fak_cpu_q8": 5,
	    "prefill_over_llamacpp_cpu": 3,
	    "decode_over_fak_cpu_q8": 1.1
	  },
	  "verification": {"status": "pass"},
	  "interpretation": {"status": "accepted_for_prefill_p256"}
	}`)

	report, err := Scan(root)
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	if len(report.Issues) != 0 {
		t.Fatalf("issues = %+v", report.Issues)
	}
	if len(report.Rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(report.Rows))
	}
	row := report.Rows[0]
	if row.Model != "qwen2.5-1.5b-instruct" || row.Workload != "prefill" || row.Speedup != 5 {
		t.Fatalf("unexpected row: %+v", row)
	}
	if len(report.Models) != 1 || report.Models[0].Accepted != 1 {
		t.Fatalf("models = %+v", report.Models)
	}
}

func TestScanFlagsBadRatio(t *testing.T) {
	root := t.TempDir()
	writeScore(t, root, "bad", `{
	  "schema": "fak.arm64-qkernel-score.v1",
	  "machine": "node-macos-a",
	  "model": {"name": "qwen2.5-1.5b-instruct"},
	  "results": {"prefill": [{"tokens": 256, "tok_per_sec": 1200}]},
	  "baseline": {"fak_cpu_q8_prefill_p256_tok_per_sec": 240},
	  "improvement": {"prefill_over_fak_cpu_q8": 4},
	  "verification": {"status": "pass"},
	  "interpretation": {"status": "accepted_for_prefill_p256"}
	}`)

	report, err := Scan(root)
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	if len(report.Issues) == 0 {
		t.Fatal("expected ratio issue")
	}
	if got := report.Issues[0].Field; got != "improvement.prefill_over_fak_cpu_q8" {
		t.Fatalf("field = %q", got)
	}
}

func TestScanFlagsAcceptedWithoutPassingVerification(t *testing.T) {
	root := t.TempDir()
	writeScore(t, root, "unverified", `{
	  "schema": "fak.arm64-qkernel-score.v1",
	  "machine": "node-macos-a",
	  "model": {"name": "qwen2.5-1.5b-instruct"},
	  "verification": {"status": "failed_overall"},
	  "interpretation": {"status": "accepted_for_prefill_p256"}
	}`)

	report, err := Scan(root)
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	if len(report.Issues) == 0 {
		t.Fatal("expected verification issue")
	}
	if !strings.Contains(report.Issues[0].Message, "requires verification.status=pass") {
		t.Fatalf("issue = %+v", report.Issues[0])
	}
}

func TestScanBuildsDecodeProbeRows(t *testing.T) {
	root := t.TempDir()
	writeScore(t, root, "decode", `{
	  "schema": "fak.arm64-qkernel-score.v1",
	  "captured_at": "2026-06-26T02:30:00Z",
	  "machine": "node-macos-a",
	  "model": {"name": "qwen2.5-1.5b-instruct"},
	  "baseline": {"decode_tok_per_sec": 40},
	  "probes": {
	    "batch_q8": {
	      "baseline_b1_tok_per_sec": 50,
	      "peak": {
	        "batch": 2,
	        "agg_tok_per_sec": 60,
	        "speedup_vs_in_run_b1": 1.2,
	        "speedup_vs_canonical_q8": 1.5
	      },
	      "points": [
	        {"batch": 1, "agg_tok_per_sec": 50},
	        {"batch": 2, "agg_tok_per_sec": 60}
	      ]
	    },
	    "q8_gguf_lean": {
	      "decode_tok_per_sec": 44,
	      "speedup_vs_canonical_q8": 1.1
	    }
	  },
	  "interpretation": {"status": "decode_followups_negative"}
	}`)

	report, err := Scan(root)
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	if len(report.Issues) != 0 {
		t.Fatalf("issues = %+v", report.Issues)
	}
	if len(report.Rows) != 2 {
		t.Fatalf("rows = %d, want 2: %+v", len(report.Rows), report.Rows)
	}
	if report.Models[0].Negative != 2 {
		t.Fatalf("models = %+v", report.Models)
	}
}

func TestScanKeepsExploratoryDecodeWithoutBaseline(t *testing.T) {
	root := t.TempDir()
	writeScore(t, root, "q4k", `{
	  "schema": "fak.arm64-qkernel-score.v1",
	  "captured_at": "2026-06-26T01:47:01Z",
	  "machine": "node-macos-a",
	  "model": {
	    "name": "qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf",
	    "source_kind": "GGUF split shard root",
	    "q4k": true
	  },
	  "results": {
	    "decode": {"tok_per_sec": 4.439272080837227}
	  },
	  "interpretation": {"status": "exploratory"}
	}`)

	report, err := Scan(root)
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	if len(report.Warnings) != 0 || len(report.Issues) != 0 {
		t.Fatalf("warnings=%+v issues=%+v", report.Warnings, report.Issues)
	}
	if len(report.Rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(report.Rows))
	}
	row := report.Rows[0]
	if row.Metric != "q4k_exploratory_tok_per_sec" || row.Speedup != 0 {
		t.Fatalf("unexpected row: %+v", row)
	}
	if report.Models[0].Exploratory != 1 {
		t.Fatalf("models = %+v", report.Models)
	}
}

func writeScore(t *testing.T, root, name, body string) {
	t.Helper()
	dir := filepath.Join(root, "runs", "by-machine", "node-macos-a", name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "score.json"), []byte(body), 0o644); err != nil {
		t.Fatalf("write score: %v", err)
	}
}
