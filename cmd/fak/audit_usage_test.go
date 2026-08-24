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
	"time"

	"github.com/anthony-chaudhary/fak/internal/auditusage"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/internal/logvault"
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
		"--vault", filepath.Join(root, "missing-vault"),
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
	// Close the append handle before the rollup reads the file: a leaked handle
	// keeps usage.jsonl open past the test, and on Windows that blocks t.TempDir's
	// RemoveAll cleanup ("being used by another process").
	if err := logger.Close(); err != nil {
		t.Fatalf("usagelog.Close: %v", err)
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
		"--vault", filepath.Join(root, "missing-vault"),
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
		case auditusage.SinkDecisionJournal, auditusage.SinkDispatchRuns, auditusage.SinkVault:
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
	// Close the append handle before the tamper step re-reads and rewrites the
	// file: a leaked handle keeps usage.jsonl open past the test, and on Windows
	// that blocks t.TempDir's RemoveAll cleanup ("being used by another process").
	if err := logger.Close(); err != nil {
		t.Fatalf("usagelog.Close: %v", err)
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
		"--vault", filepath.Join(root, "missing-vault"),
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
		"--vault", filepath.Join(root, "missing-vault"),
	})
	if code != 0 {
		t.Fatalf("runAuditUsage exit=%d, stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "fak audit usage") {
		t.Errorf("want a human-readable header, got:\n%s", stdout.String())
	}
}

// TestRunAuditUsage_VaultSection drives `fak audit usage` against a real captured
// vault fixture and asserts the vault section folds the WITNESSED footprint (rows,
// bytes, files, last-capture) and reports the vault sink present + chain verified —
// the "is my backup current?" answer #2455 puts in the cross-journal rollup.
func TestRunAuditUsage_VaultSection(t *testing.T) {
	root := t.TempDir()
	srcDir := t.TempDir()
	if err := writeFileLV(filepath.Join(srcDir, "loops.jsonl"), "row1\nrow2\n"); err != nil {
		t.Fatal(err)
	}
	vaultDir := t.TempDir()
	v := &logvault.Vault{Dir: vaultDir, Sources: []logvault.Source{{ID: "s", Root: srcDir}}}
	if _, err := v.Capture(); err != nil {
		t.Fatalf("capture: %v", err)
	}
	t.Setenv("FAK_LOOP_LEDGER", filepath.Join(root, "missing-loops.jsonl"))

	args := []string{
		"--root", root,
		"--journal", filepath.Join(root, "missing-guard-audit.jsonl"),
		"--usage-log", filepath.Join(root, "missing-usage.jsonl"),
		"--vault", vaultDir,
	}

	var stdout, stderr bytes.Buffer
	if code := runAuditUsage(&stdout, &stderr, append(append([]string{}, args...), "--json")); code != 0 {
		t.Fatalf("runAuditUsage exit=%d, stderr=%s", code, stderr.String())
	}
	var rep auditusage.Report
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("decode JSON: %v\n%s", err, stdout.String())
	}
	if !rep.Vault.Present || rep.Vault.Basis != "witnessed" {
		t.Fatalf("vault rollup: want present+witnessed after a real capture, got %+v", rep.Vault)
	}
	if rep.Vault.Sources != 1 || rep.Vault.Files != 1 || rep.Vault.Bytes != 10 {
		t.Errorf("vault footprint: want sources=1 files=1 bytes=10, got %+v", rep.Vault)
	}
	if rep.Vault.Rows < 1 || rep.Vault.LastCaptureNano <= 0 {
		t.Errorf("vault fold: want a witnessed row + last-capture timestamp, got %+v", rep.Vault)
	}
	if rep.Vault.VerifyMismatches != 0 || rep.Vault.ChainBroken {
		t.Errorf("a freshly captured vault must verify clean, got %+v", rep.Vault)
	}
	var vaultSinkOK bool
	for _, s := range rep.Sinks {
		if s.Kind == auditusage.SinkVault {
			vaultSinkOK = s.Present && s.Chain == auditusage.ChainVerified
		}
	}
	if !vaultSinkOK {
		t.Errorf("vault sink: want present+verified, got sinks=%+v", rep.Sinks)
	}
	if len(rep.Findings) != 0 {
		t.Errorf("want no findings over a clean vault, got %+v", rep.Findings)
	}

	// The human-readable render carries the vault line with its WITNESSED footprint.
	var textOut, textErr bytes.Buffer
	if code := runAuditUsage(&textOut, &textErr, args); code != 0 {
		t.Fatalf("runAuditUsage text exit=%d, stderr=%s", code, textErr.String())
	}
	got := textOut.String()
	if !strings.Contains(got, "vault (witnessed):") || !strings.Contains(got, "present=true") || !strings.Contains(got, "bytes=10B") {
		t.Errorf("text render missing the witnessed vault section:\n%s", got)
	}
}

// TestRunAuditUsage_VaultForcedMismatchSurfaces proves the "is the vault intact?"
// half: a forced mirror corruption surfaces as a CHAIN_BROKEN vault finding rather
// than hiding behind the fold — a silent backup corruption cannot stay silent.
func TestRunAuditUsage_VaultForcedMismatchSurfaces(t *testing.T) {
	root := t.TempDir()
	srcDir := t.TempDir()
	if err := writeFileLV(filepath.Join(srcDir, "loops.jsonl"), "row1\nrow2\n"); err != nil {
		t.Fatal(err)
	}
	vaultDir := t.TempDir()
	v := &logvault.Vault{Dir: vaultDir, Sources: []logvault.Source{{ID: "s", Root: srcDir}}}
	if _, err := v.Capture(); err != nil {
		t.Fatalf("capture: %v", err)
	}
	// Force a mirror mismatch: overwrite the captured mirror so its re-hash no
	// longer matches the manifest — the corruption the issue says must surface.
	mirror := filepath.Join(vaultDir, "by-source", "s", "loops.jsonl")
	if err := os.WriteFile(mirror, []byte("CORRUPTED"), 0o644); err != nil {
		t.Fatalf("tamper mirror: %v", err)
	}
	t.Setenv("FAK_LOOP_LEDGER", filepath.Join(root, "missing-loops.jsonl"))

	var stdout, stderr bytes.Buffer
	code := runAuditUsage(&stdout, &stderr, []string{
		"--root", root,
		"--journal", filepath.Join(root, "missing-guard-audit.jsonl"),
		"--usage-log", filepath.Join(root, "missing-usage.jsonl"),
		"--vault", vaultDir,
		"--json",
	})
	if code != 0 {
		t.Fatalf("runAuditUsage exit=%d, stderr=%s", code, stderr.String())
	}
	var rep auditusage.Report
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("decode JSON: %v\n%s", err, stdout.String())
	}
	if rep.Vault.VerifyMismatches < 1 && !rep.Vault.ChainBroken {
		t.Fatalf("forced mismatch must surface in the vault rollup, got %+v", rep.Vault)
	}
	var found bool
	for _, f := range rep.Findings {
		if f.Kind == "CHAIN_BROKEN" && f.Sink == auditusage.SinkVault {
			found = true
		}
	}
	if !found {
		t.Fatalf("want a CHAIN_BROKEN finding for the vault after a forced mismatch, got %+v", rep.Findings)
	}
}

func TestRunAuditUsage_BadSinceIsUsageError(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runAuditUsage(&stdout, &stderr, []string{"--since", "not-a-duration"})
	if code != 2 {
		t.Fatalf("want exit 2 for an unparseable --since, got %d", code)
	}
}

func TestMergeUsageInputs_MultiRootOrder(t *testing.T) {
	rootA := t.TempDir()
	rootMissing := t.TempDir() // no ledgers: must be skipped, not error
	rootC := t.TempDir()
	roots := []string{rootA, rootMissing, rootC}

	mustWriteJSONLRow(t, gatewayUsagePathForRoot(rootA), map[string]any{"schema": "fak-gateway-usage-ledger/1", "unix_millis": 1000, "kind": "exit", "session_type": "guard"})
	mustWriteJSONLRow(t, gatewayUsagePathForRoot(rootC), map[string]any{"schema": "fak-gateway-usage-ledger/1", "unix_millis": 3000, "kind": "exit", "session_type": "guard"})
	mustWriteJSONLRow(t, cacheValuePathForRoot(rootA), map[string]any{"schema": "fak-cache-value-ledger/1", "date": "2026-06-30", "session_type": "guard", "unix_millis": 1000})
	mustWriteJSONLRow(t, cacheValuePathForRoot(rootC), map[string]any{"schema": "fak-cache-value-ledger/1", "date": "2026-06-30", "session_type": "guard", "unix_millis": 3000})

	gw := mergeGatewayUsageInputs(roots)
	if !gw.Present || gw.Path != gatewayUsagePathForRoot(rootA) {
		t.Errorf("gateway: want present with first-root path %s, got present=%v path=%s", gatewayUsagePathForRoot(rootA), gw.Present, gw.Path)
	}
	if len(gw.Rows) != 2 || gw.Rows[0].UnixMillis != 1000 || gw.Rows[1].UnixMillis != 3000 {
		t.Errorf("gateway: want rows appended in root order [1000 3000], got %+v", gw.Rows)
	}

	cv := mergeCacheValueInputs(roots)
	if !cv.Present || cv.Path != cacheValuePathForRoot(rootA) {
		t.Errorf("cache: want present with first-root path %s, got present=%v path=%s", cacheValuePathForRoot(rootA), cv.Present, cv.Path)
	}
	if len(cv.Rows) != 2 || cv.Rows[0].UnixMillis != 1000 || cv.Rows[1].UnixMillis != 3000 {
		t.Errorf("cache: want rows appended in root order [1000 3000], got %+v", cv.Rows)
	}

	// A root set with no ledgers anywhere stays absent: no path, no rows.
	gwAbsent := mergeGatewayUsageInputs([]string{rootMissing})
	if gwAbsent.Present || gwAbsent.Path != "" || len(gwAbsent.Rows) != 0 {
		t.Errorf("gateway: want absent over a ledgerless root, got %+v", gwAbsent)
	}
	cvAbsent := mergeCacheValueInputs([]string{rootMissing})
	if cvAbsent.Present || cvAbsent.Path != "" || len(cvAbsent.Rows) != 0 {
		t.Errorf("cache: want absent over a ledgerless root, got %+v", cvAbsent)
	}

	gwEmpty := mergeGatewayUsageInputs(nil)
	if gwEmpty.Present || gwEmpty.Path != "" || len(gwEmpty.Rows) != 0 {
		t.Errorf("gateway: want absent over no roots, got %+v", gwEmpty)
	}
	cvEmpty := mergeCacheValueInputs(nil)
	if cvEmpty.Present || cvEmpty.Path != "" || len(cvEmpty.Rows) != 0 {
		t.Errorf("cache: want absent over no roots, got %+v", cvEmpty)
	}

	called := false
	present, path, rows := collectRootLedgerRows[int](nil,
		func(string) string {
			called = true
			return ""
		},
		func(string) []int {
			called = true
			return []int{1}
		})
	if called || present || path != "" || rows != nil {
		t.Errorf("helper: empty roots should not call callbacks or return data, called=%v present=%v path=%q rows=%v", called, present, path, rows)
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

func TestRunAuditUsageDashboardFold(t *testing.T) {
	root := t.TempDir()
	path := gatewayUsagePathForRoot(root)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	for _, event := range []string{"lightweight_open", "lightweight_open", "rich_ready", "rich_unavailable"} {
		row, err := gatewayusageledger.DashboardEventRow(event, now)
		if err != nil {
			t.Fatal(err)
		}
		if err := gatewayusageledger.Append(path, row); err != nil {
			t.Fatal(err)
		}
	}
	var stdout, stderr bytes.Buffer
	if code := runAuditUsage(&stdout, &stderr, []string{"--root", root, "--since", "168h"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	want := "dashboards (window): lightweight=2 rich-ready=1 rich-unavailable=1"
	if !strings.Contains(stdout.String(), want) {
		t.Fatalf("missing %q:\n%s", want, stdout.String())
	}
}
