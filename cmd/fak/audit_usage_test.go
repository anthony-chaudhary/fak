package main

// audit_usage_test.go exercises the `fak audit usage` CLI shell (runAuditUsage):
// its flag parsing, its presence/absence classification per sink, and — using
// REAL on-disk fixtures built through each sink's own writer — that a tampered
// chain surfaces as a CHAIN_BROKEN finding rather than being silently dropped.
// The pure fold logic itself is covered by internal/auditusage's own tests;
// this file is about the shell's I/O wiring.

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/auditusage"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/usagelog"
)

func TestRunAuditUsage_AllAbsent(t *testing.T) {
	root := t.TempDir()
	journalPath := filepath.Join(root, "missing-guard-audit.jsonl")
	usagePath := filepath.Join(root, "missing-usage.jsonl")
	t.Setenv("FAK_LOOP_LEDGER", filepath.Join(root, "missing-loops.jsonl"))

	var stdout, stderr bytes.Buffer
	code := runAuditUsage(&stdout, &stderr, []string{
		"--root", root,
		"--journal", journalPath,
		"--usage-log", usagePath,
		"--json",
	})
	if code != 0 {
		t.Fatalf("runAuditUsage exit=%d, stderr=%s", code, stderr.String())
	}

	var rep auditusage.Report
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("decode JSON: %v\n%s", err, stdout.String())
	}
	if len(rep.Findings) != 0 {
		t.Errorf("want no findings when every sink is absent, got %+v", rep.Findings)
	}
	for _, s := range rep.Sinks {
		if s.Present {
			t.Errorf("sink %s: want absent in a fresh temp root, got present (path=%s)", s.Kind, s.Path)
		}
		if s.Chain != auditusage.ChainAbsent {
			t.Errorf("sink %s: want chain=absent, got %s", s.Kind, s.Chain)
		}
	}
}

func TestRunAuditUsage_RealFixtures(t *testing.T) {
	root := t.TempDir()
	usagePath := filepath.Join(root, "usage.jsonl")
	journalPath := filepath.Join(root, "missing-guard-audit.jsonl")
	loopLedger := filepath.Join(root, "loops.jsonl")
	t.Setenv("FAK_LOOP_LEDGER", loopLedger)

	// Real usagelog fixture, built through the package's own writer so the hash
	// chain is genuine.
	logger, err := usagelog.Open(usagePath)
	if err != nil {
		t.Fatalf("usagelog.Open: %v", err)
	}
	if _, err := logger.Append(usagelog.Row{Verb: "guard", ExitCode: 0}); err != nil {
		t.Fatalf("usagelog.Append: %v", err)
	}
	if _, err := logger.Append(usagelog.Row{Verb: "audit", ExitCode: 1}); err != nil {
		t.Fatalf("usagelog.Append: %v", err)
	}

	// Real loop-ledger fixture, built through loopmgr.Append.
	if _, err := loopmgr.Append(loopLedger, loopmgr.Event{LoopID: "l1", Kind: loopmgr.EventFire}); err != nil {
		t.Fatalf("loopmgr.Append: %v", err)
	}
	if _, err := loopmgr.Append(loopLedger, loopmgr.Event{LoopID: "l1", Kind: loopmgr.EventAdmit}); err != nil {
		t.Fatalf("loopmgr.Append: %v", err)
	}

	// Cache-value / gateway-usage ledgers carry no hash chain by design -- a
	// plain JSONL row is a faithful fixture without needing each package's
	// full writer plumbing.
	cachePath := cacheValuePathForRoot(root)
	mustWriteJSONLRow(t, cachePath, map[string]any{"schema": "fak-cache-value-ledger/1", "date": "2026-06-30", "session_type": "guard", "unix_millis": 1000})
	gwPath := gatewayUsagePathForRoot(root)
	mustWriteJSONLRow(t, gwPath, map[string]any{"schema": "fak-gateway-usage-ledger/1", "unix_millis": 1000, "kind": "exit", "session_type": "guard"})

	var stdout, stderr bytes.Buffer
	code := runAuditUsage(&stdout, &stderr, []string{
		"--root", root,
		"--journal", journalPath,
		"--usage-log", usagePath,
		"--json",
	})
	if code != 0 {
		t.Fatalf("runAuditUsage exit=%d, stderr=%s", code, stderr.String())
	}

	var rep auditusage.Report
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("decode JSON: %v\n%s", err, stdout.String())
	}
	if len(rep.Findings) != 0 {
		t.Fatalf("want no findings over sound fixtures, got %+v", rep.Findings)
	}
	if rep.Usage.Total != 2 || rep.Usage.Errors != 1 {
		t.Errorf("usage rollup mismatch: %+v", rep.Usage)
	}
	if rep.Loop.Loops != 1 || rep.Loop.Fires != 1 || rep.Loop.Admitted != 1 {
		t.Errorf("loop rollup mismatch: %+v", rep.Loop)
	}
	if rep.Cache.Sessions != 1 {
		t.Errorf("cache rollup mismatch: %+v", rep.Cache)
	}
	if rep.Gateway.Sessions != 1 {
		t.Errorf("gateway rollup mismatch: %+v", rep.Gateway)
	}
	for _, s := range rep.Sinks {
		switch s.Kind {
		case auditusage.SinkDecisionJournal, auditusage.SinkDispatchRuns:
			if s.Present {
				t.Errorf("sink %s: want absent (no fixture written), got present", s.Kind)
			}
		default:
			if !s.Present {
				t.Errorf("sink %s: want present, got absent", s.Kind)
			}
		}
	}
}

func TestRunAuditUsage_ChainBroken_SurfacesFinding(t *testing.T) {
	root := t.TempDir()
	usagePath := filepath.Join(root, "usage.jsonl")
	journalPath := filepath.Join(root, "missing-guard-audit.jsonl")
	t.Setenv("FAK_LOOP_LEDGER", filepath.Join(root, "missing-loops.jsonl"))

	logger, err := usagelog.Open(usagePath)
	if err != nil {
		t.Fatalf("usagelog.Open: %v", err)
	}
	if _, err := logger.Append(usagelog.Row{Verb: "guard", ExitCode: 0}); err != nil {
		t.Fatalf("usagelog.Append: %v", err)
	}
	if _, err := logger.Append(usagelog.Row{Verb: "audit", ExitCode: 0}); err != nil {
		t.Fatalf("usagelog.Append: %v", err)
	}

	// Tamper the chain: corrupt row 1's recorded hash so Verify's recomputed
	// hash no longer matches. ReadRows (a tolerant JSON-syntax-only read) still
	// recovers both rows -- the sink must NOT silently drop them.
	raw, err := os.ReadFile(usagePath)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	lines := strings.Split(strings.TrimRight(string(raw), "\n"), "\n")
	if len(lines) != 2 {
		t.Fatalf("want 2 fixture lines, got %d: %q", len(lines), lines)
	}
	var row map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &row); err != nil {
		t.Fatalf("unmarshal row 0: %v", err)
	}
	row["hash"] = "0000000000000000000000000000000000000000000000000000000000000000"
	tampered, err := json.Marshal(row)
	if err != nil {
		t.Fatalf("marshal tampered row: %v", err)
	}
	lines[0] = string(tampered)
	if err := os.WriteFile(usagePath, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatalf("write tampered fixture: %v", err)
	}

	var stdout, stderr bytes.Buffer
	code := runAuditUsage(&stdout, &stderr, []string{
		"--root", root,
		"--journal", journalPath,
		"--usage-log", usagePath,
		"--json",
	})
	if code != 0 {
		t.Fatalf("runAuditUsage exit=%d, stderr=%s", code, stderr.String())
	}

	var rep auditusage.Report
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("decode JSON: %v\n%s", err, stdout.String())
	}

	var found bool
	for _, f := range rep.Findings {
		if f.Kind == "CHAIN_BROKEN" && f.Sink == auditusage.SinkUsageLog {
			found = true
		}
	}
	if !found {
		t.Fatalf("want a CHAIN_BROKEN finding for sink usage_log, got findings=%+v", rep.Findings)
	}
	if rep.Usage.Total != 2 {
		t.Errorf("a broken chain must not drop the recovered rows from the rollup: usage.total=%d, want 2", rep.Usage.Total)
	}
}

func TestRunAuditUsage_TextOutput(t *testing.T) {
	root := t.TempDir()
	t.Setenv("FAK_LOOP_LEDGER", filepath.Join(root, "missing-loops.jsonl"))

	var stdout, stderr bytes.Buffer
	code := runAuditUsage(&stdout, &stderr, []string{
		"--root", root,
		"--journal", filepath.Join(root, "missing-guard-audit.jsonl"),
		"--usage-log", filepath.Join(root, "missing-usage.jsonl"),
	})
	if code != 0 {
		t.Fatalf("runAuditUsage exit=%d, stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "fak audit usage") {
		t.Errorf("want a human-readable header, got:\n%s", stdout.String())
	}
}

func TestRunAuditUsage_BadSinceIsUsageError(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runAuditUsage(&stdout, &stderr, []string{"--since", "not-a-duration"})
	if code != 2 {
		t.Fatalf("want exit 2 for an unparseable --since, got %d", code)
	}
}

func mustWriteJSONLRow(t *testing.T, path string, row map[string]any) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	b, err := json.Marshal(row)
	if err != nil {
		t.Fatalf("marshal row: %v", err)
	}
	if err := os.WriteFile(path, append(b, '\n'), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}
