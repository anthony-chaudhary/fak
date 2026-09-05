package logvault

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

var (
	benchSinkStats      []SourceStats
	benchSinkSyncStats  SyncStats
	benchSinkProblems   []VerifyProblem
	benchSinkRestoreRep RestoreReport
	benchSinkDrillRow   DrillRow
	benchSinkFootprint  []SourceFootprint
	benchSinkVerifyN    int
	benchSinkRow        ManifestRow
)

func writeFileBench(b *testing.B, path, content string) {
	b.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		b.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		b.Fatal(err)
	}
}

// BenchmarkStore measures production capture/store operations: steady-state unchanged check,
// incremental journal append, and full new-file capture.
func BenchmarkStore(b *testing.B) {
	b.Run("Unchanged", BenchmarkCapture_Unchanged)
	b.Run("Append", BenchmarkCapture_Append)
	b.Run("Full", BenchmarkCapture_Full)
}

// BenchmarkCapture_Unchanged measures the dominant production steady-state:
// walking sources, replaying manifest states, and verifying files are unchanged.
func BenchmarkCapture_Unchanged(b *testing.B) {
	srcDir := b.TempDir()
	writeFileBench(b, filepath.Join(srcDir, "session.jsonl"), "turn 1 ok\nturn 2 ok\n")
	writeFileBench(b, filepath.Join(srcDir, "guard-audit.jsonl"), "event 1\nevent 2\n")
	writeFileBench(b, filepath.Join(srcDir, "nested", "trace.log"), "trace payload line\n")

	vaultDir := b.TempDir()
	v := &Vault{Dir: vaultDir, Sources: []Source{{ID: "bench-src", Root: srcDir}}}
	if _, err := v.Capture(); err != nil {
		b.Fatalf("initial capture: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stats, err := v.Capture()
		if err != nil {
			b.Fatalf("capture: %v", err)
		}
		benchSinkStats = stats
	}
}

// BenchmarkCapture_Append measures incremental capture when growing logs
// (such as guard-audit.jsonl) have new records appended.
func BenchmarkCapture_Append(b *testing.B) {
	srcDir := b.TempDir()
	logPath := filepath.Join(srcDir, "guard-audit.jsonl")
	writeFileBench(b, logPath, "initial-header-event\n")

	vaultDir := b.TempDir()
	v := &Vault{Dir: vaultDir, Sources: []Source{{ID: "bench-src", Root: srcDir}}}
	if _, err := v.Capture(); err != nil {
		b.Fatalf("initial capture: %v", err)
	}

	payload := []byte("agent-turn-event-witness-record-payload\n")
	b.SetBytes(int64(len(payload)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f, err := os.OpenFile(logPath, os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			b.Fatalf("open: %v", err)
		}
		if _, err := f.Write(payload); err != nil {
			f.Close()
			b.Fatalf("write: %v", err)
		}
		f.Close()

		stats, err := v.Capture()
		if err != nil {
			b.Fatalf("capture: %v", err)
		}
		benchSinkStats = stats
	}
}

// BenchmarkCapture_Full measures new logfile capture into the vault (mirroring,
// SHA-256 computation, and hash-chained manifest append).
func BenchmarkCapture_Full(b *testing.B) {
	srcDir := b.TempDir()
	vaultDir := b.TempDir()
	v := &Vault{Dir: vaultDir, Sources: []Source{{ID: "bench-src", Root: srcDir}}}
	payload := []byte("new-session-transcript-data-row-bytes\n")
	b.SetBytes(int64(len(payload)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		filePath := filepath.Join(srcDir, fmt.Sprintf("session_%d.jsonl", i))
		if err := os.WriteFile(filePath, payload, 0o644); err != nil {
			b.Fatalf("write file: %v", err)
		}

		stats, err := v.Capture()
		if err != nil {
			b.Fatalf("capture: %v", err)
		}
		benchSinkStats = stats
	}
}

// BenchmarkRestore measures restoring source trees out of the vault and automated drills.
func BenchmarkRestore(b *testing.B) {
	b.Run("Full", BenchmarkRestore_Full)
	b.Run("Drill", BenchmarkRestore_Drill)
}

// BenchmarkRestore_Full measures full restore to a clean target directory,
// including path validation, content transfer, and on-the-fly SHA-256 verification.
func BenchmarkRestore_Full(b *testing.B) {
	srcDir := b.TempDir()
	writeFileBench(b, filepath.Join(srcDir, "session.jsonl"), "session data row a\nsession data row b\n")
	writeFileBench(b, filepath.Join(srcDir, "sub", "audit.log"), "audit log entry line\n")

	vaultDir := b.TempDir()
	v := &Vault{Dir: vaultDir, Sources: []Source{{ID: "bench-src", Root: srcDir}}}
	if _, err := v.Capture(); err != nil {
		b.Fatalf("initial capture: %v", err)
	}

	destBase := b.TempDir()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		to := filepath.Join(destBase, fmt.Sprintf("restore_%d", i))
		rep, err := v.Restore(RestoreOptions{Source: "bench-src", To: to})
		if err != nil {
			b.Fatalf("restore: %v", err)
		}
		benchSinkRestoreRep = rep
	}
}

// BenchmarkRestore_Drill measures the automated disaster recovery drill cadence:
// temp directory setup, restore, end-to-end verification, and journal append to drill-log.
func BenchmarkRestore_Drill(b *testing.B) {
	srcDir := b.TempDir()
	writeFileBench(b, filepath.Join(srcDir, "session.jsonl"), "drill test payload content\n")

	vaultDir := b.TempDir()
	v := &Vault{Dir: vaultDir, Sources: []Source{{ID: "bench-src", Root: srcDir}}}
	if _, err := v.Capture(); err != nil {
		b.Fatalf("initial capture: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		row, rep, err := v.Drill("bench-src", "")
		if err != nil {
			b.Fatalf("drill: %v", err)
		}
		benchSinkDrillRow = row
		benchSinkRestoreRep = rep
	}
}

// BenchmarkSync measures replication of vault state to off-box/replica destinations.
func BenchmarkSync(b *testing.B) {
	b.Run("Full", BenchmarkSyncTo_Full)
	b.Run("Incremental", BenchmarkSyncTo_Incremental)
}

// BenchmarkSyncTo_Full measures replicating vault contents to a fresh destination replica,
// verifying source chain, running PII/secret redaction scrub, and verifying arrival.
func BenchmarkSyncTo_Full(b *testing.B) {
	srcDir := b.TempDir()
	writeFileBench(b, filepath.Join(srcDir, "clean.jsonl"), "regular unredacted content line\n")
	writeFileBench(b, filepath.Join(srcDir, "token.jsonl"), "header used ghp_Ab1Ab1Ab1Ab1Ab1Ab1Ab1Ab1Ab1Ab1Ab1Ab1 token\n")

	vaultDir := b.TempDir()
	v := &Vault{Dir: vaultDir, Sources: []Source{{ID: "bench-src", Root: srcDir}}}
	if _, err := v.Capture(); err != nil {
		b.Fatalf("initial capture: %v", err)
	}

	replicaBase := b.TempDir()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		replicaDir := filepath.Join(replicaBase, fmt.Sprintf("replica_%d", i))
		stats, problems, err := v.SyncTo(replicaDir, 0)
		if err != nil || len(problems) != 0 {
			b.Fatalf("sync: err=%v problems=%v", err, problems)
		}
		benchSinkSyncStats = stats
		benchSinkProblems = problems
	}
}

// BenchmarkSyncTo_Incremental measures syncing to a destination that is already current,
// verifying source chain and skipping unchanged versions.
func BenchmarkSyncTo_Incremental(b *testing.B) {
	srcDir := b.TempDir()
	writeFileBench(b, filepath.Join(srcDir, "session.jsonl"), "incremental sync payload\n")

	vaultDir := b.TempDir()
	v := &Vault{Dir: vaultDir, Sources: []Source{{ID: "bench-src", Root: srcDir}}}
	if _, err := v.Capture(); err != nil {
		b.Fatalf("initial capture: %v", err)
	}

	replicaDir := filepath.Join(b.TempDir(), "replica")
	if _, problems, err := v.SyncTo(replicaDir, 0); err != nil || len(problems) != 0 {
		b.Fatalf("initial sync: err=%v problems=%v", err, problems)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stats, problems, err := v.SyncTo(replicaDir, 0)
		if err != nil || len(problems) != 0 {
			b.Fatalf("sync: err=%v problems=%v", err, problems)
		}
		benchSinkSyncStats = stats
		benchSinkProblems = problems
	}
}

// BenchmarkManifest measures low-level manifest appending and cryptographic verification.
func BenchmarkManifest(b *testing.B) {
	b.Run("Append", BenchmarkManifest_Append)
	b.Run("Verify", BenchmarkManifest_Verify)
}

// BenchmarkManifest_Append measures raw manifest hash-chain append speed.
func BenchmarkManifest_Append(b *testing.B) {
	dir := b.TempDir()
	m, err := OpenManifest(dir)
	if err != nil {
		b.Fatalf("open manifest: %v", err)
	}
	defer m.Close()

	row := ManifestRow{
		Op:        OpAppend,
		Source:    "bench-src",
		RelPath:   "audit.jsonl",
		Bytes:     128,
		SizeAfter: 1024,
		SHA256:    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		row.TSUnixNano = int64(i + 1)
		res, err := m.Append(row)
		if err != nil {
			b.Fatalf("append: %v", err)
		}
		benchSinkRow = res
	}
}

// BenchmarkManifest_Verify measures verification of the manifest SHA-256 rolling chain.
func BenchmarkManifest_Verify(b *testing.B) {
	dir := b.TempDir()
	m, err := OpenManifest(dir)
	if err != nil {
		b.Fatalf("open manifest: %v", err)
	}
	for j := 0; j < 100; j++ {
		if _, err := m.Append(ManifestRow{
			TSUnixNano: int64(j + 1),
			Op:         OpFull,
			Source:     "bench-src",
			RelPath:    fmt.Sprintf("file_%d.log", j),
			Bytes:      64,
			SizeAfter:  64,
			SHA256:     "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
		}); err != nil {
			b.Fatalf("setup append: %v", err)
		}
	}
	m.Close()

	manifestPath := filepath.Join(dir, ManifestName)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		n, err := VerifyManifest(manifestPath)
		if err != nil || n != 100 {
			b.Fatalf("verify: n=%d err=%v", n, err)
		}
		benchSinkVerifyN = n
	}
}

// BenchmarkObserve_Footprint measures pure in-memory aggregation of manifest rows into source footprints.
func BenchmarkObserve_Footprint(b *testing.B) {
	rows := make([]ManifestRow, 200)
	for j := 0; j < len(rows); j++ {
		rows[j] = ManifestRow{
			TSUnixNano: int64(j * 1000),
			Op:         OpFull,
			Source:     fmt.Sprintf("source-%d", j%5),
			RelPath:    fmt.Sprintf("logs/session_%d.jsonl", j),
			Bytes:      256,
			SizeAfter:  256,
			SHA256:     "d41d8cd98f00b204e9800998ecf8427e",
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkFootprint = Footprint(rows)
	}
}
