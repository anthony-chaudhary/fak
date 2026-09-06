package resultstier

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPayloadEntryCompleteAndVolumeKey(t *testing.T) {
	validSHA := strings.Repeat("a", 64)
	entry := PayloadEntry{
		Path:   "predictions.json",
		Bytes:  123,
		SHA256: validSHA,
	}
	if err := entry.Complete(); err != nil {
		t.Fatalf("unexpected error for valid entry: %v", err)
	}

	key, err := entry.VolumeKey()
	if err != nil {
		t.Fatalf("unexpected error for VolumeKey: %v", err)
	}
	wantKey := "results-payload/aa/" + validSHA
	if key != wantKey {
		t.Errorf("VolumeKey() = %q, want %q", key, wantKey)
	}

	// Empty path
	invalid := entry
	invalid.Path = ""
	if err := invalid.Complete(); err == nil {
		t.Error("expected error for empty path")
	}

	// Traversal or absolute path
	invalid = entry
	invalid.Path = "../secret.json"
	if err := invalid.Complete(); err == nil {
		t.Error("expected error for path traversal")
	}
	invalid.Path = "/etc/passwd"
	if err := invalid.Complete(); err == nil {
		t.Error("expected error for absolute path")
	}
	invalid.Path = "C:\\Windows\\system32"
	if err := invalid.Complete(); err == nil {
		t.Error("expected error for drive path")
	}

	// Zero / negative bytes
	invalid = entry
	invalid.Bytes = 0
	if err := invalid.Complete(); err == nil {
		t.Error("expected error for zero bytes")
	}
	invalid.Bytes = -5
	if err := invalid.Complete(); err == nil {
		t.Error("expected error for negative bytes")
	}

	// Invalid SHA length
	invalid = entry
	invalid.SHA256 = "abc"
	if err := invalid.Complete(); err == nil {
		t.Error("expected error for short sha256")
	}

	// Invalid SHA characters (uppercase or non-hex)
	invalid = entry
	invalid.SHA256 = strings.ToUpper(validSHA)
	if err := invalid.Complete(); err == nil {
		t.Error("expected error for uppercase sha256")
	}

	invalid = entry
	invalid.SHA256 = strings.Repeat("z", 64)
	if err := invalid.Complete(); err == nil {
		t.Error("expected error for non-hex sha256")
	}

	// VolumeKey on incomplete entry
	if _, err := invalid.VolumeKey(); err == nil {
		t.Error("expected error from VolumeKey on incomplete entry")
	}
}

func TestMintPayloadIndex(t *testing.T) {
	tmpDir := t.TempDir()

	// 1. Create claim files
	mustWriteFile(t, filepath.Join(tmpDir, "INDEX.md"), []byte("# Index\n"))
	mustWriteFile(t, filepath.Join(tmpDir, "perf.json"), []byte(`{"latency": 12}`))
	mustWriteFile(t, filepath.Join(tmpDir, "summary.json"), []byte(`{"total": 1}`))

	// 2. Create payload files (including nested)
	payload1 := filepath.Join(tmpDir, "predictions_epoch1.json")
	mustWriteFile(t, payload1, []byte("pred-data-1"))
	subDir := filepath.Join(tmpDir, "nested", "run")
	if err := os.MkdirAll(subDir, 0755); err != nil {
		t.Fatal(err)
	}
	payload2 := filepath.Join(subDir, "build.log")
	mustWriteFile(t, payload2, []byte("log output line"))
	payload3 := filepath.Join(subDir, "times-001.json")
	mustWriteFile(t, payload3, []byte("[1.0, 2.0]"))

	// 3. Create unknown file
	mustWriteFile(t, filepath.Join(tmpDir, "misc.bin"), []byte{0x00, 0x01})

	// 4. Create files to skip
	gitDir := filepath.Join(tmpDir, ".git")
	if err := os.MkdirAll(gitDir, 0755); err != nil {
		t.Fatal(err)
	}
	mustWriteFile(t, filepath.Join(gitDir, "HEAD"), []byte("ref: refs/heads/main"))
	mustWriteFile(t, filepath.Join(tmpDir, ".DS_Store"), []byte("junk"))
	mustWriteFile(t, filepath.Join(tmpDir, ".gitignore"), []byte("*.tmp"))
	mustWriteFile(t, filepath.Join(tmpDir, ".gitkeep"), []byte(""))

	storeURI := "s3://my-bucket/store"
	idx, census, err := MintPayloadIndex(tmpDir, storeURI)
	if err != nil {
		t.Fatalf("MintPayloadIndex failed: %v", err)
	}

	if idx.Schema != PayloadIndexSchema {
		t.Errorf("idx.Schema = %q, want %q", idx.Schema, PayloadIndexSchema)
	}
	if idx.StoreURI != storeURI {
		t.Errorf("idx.StoreURI = %q, want %q", idx.StoreURI, storeURI)
	}

	// Expect exactly 3 payload entries
	if len(idx.Entries) != 3 {
		t.Fatalf("len(idx.Entries) = %d, want 3", len(idx.Entries))
	}

	// Verify entries are sorted by path ascending
	for i := 1; i < len(idx.Entries); i++ {
		if idx.Entries[i-1].Path >= idx.Entries[i].Path {
			t.Errorf("Entries not sorted: %q >= %q", idx.Entries[i-1].Path, idx.Entries[i].Path)
		}
	}

	// Check that each payload entry is complete and correct
	for _, e := range idx.Entries {
		if err := e.Complete(); err != nil {
			t.Errorf("Entry %q not complete: %v", e.Path, err)
		}
	}

	// Verify census
	if census.ClaimFiles != 3 {
		t.Errorf("census.ClaimFiles = %d, want 3", census.ClaimFiles)
	}
	if census.PayloadFiles != 3 {
		t.Errorf("census.PayloadFiles = %d, want 3", census.PayloadFiles)
	}
	if census.UnknownFiles != 1 {
		t.Errorf("census.UnknownFiles = %d, want 1", census.UnknownFiles)
	}
	if census.UnknownExts[".bin"] != 1 {
		t.Errorf("census.UnknownExts[.bin] = %d, want 1", census.UnknownExts[".bin"])
	}

	// Verify TotalBytes
	if idx.TotalBytes() != census.PayloadBytes {
		t.Errorf("idx.TotalBytes() = %d != census.PayloadBytes = %d", idx.TotalBytes(), census.PayloadBytes)
	}
}

func TestCanMigrate(t *testing.T) {
	validSHA := strings.Repeat("b", 64)
	idx := PayloadIndex{
		Schema:   PayloadIndexSchema,
		StoreURI: "s3://results/store",
		Entries: []PayloadEntry{
			{
				Path:   "logs/build.log",
				Bytes:  42,
				SHA256: validSHA,
			},
			{
				Path:   "data/zero.log",
				Bytes:  0, // zero bytes
				SHA256: validSHA,
			},
			{
				Path:   "data/badsha.log",
				Bytes:  10,
				SHA256: "BADSHA", // malformed sha
			},
		},
	}

	// 1. Allowed migration
	if err := CanMigrate(idx, "logs/build.log"); err != nil {
		t.Errorf("CanMigrate(build.log) failed unexpectedly: %v", err)
	}

	// 2. Rejected: empty StoreURI
	noStoreIdx := idx
	noStoreIdx.StoreURI = ""
	if err := CanMigrate(noStoreIdx, "logs/build.log"); err == nil {
		t.Error("expected error when StoreURI is empty")
	}

	// 3. Rejected: claim tier file
	if err := CanMigrate(idx, "summary.json"); err == nil {
		t.Error("expected error for claim tier file")
	}

	// 4. Rejected: missing entry in index
	if err := CanMigrate(idx, "missing.log"); err == nil {
		t.Error("expected error for file not in index")
	}

	// 5. Rejected: zero bytes entry
	if err := CanMigrate(idx, "data/zero.log"); err == nil {
		t.Error("expected error for entry with zero bytes")
	}

	// 6. Rejected: malformed hash
	if err := CanMigrate(idx, "data/badsha.log"); err == nil {
		t.Error("expected error for entry with malformed hash")
	}
}

func TestVerifyPayloadIndex(t *testing.T) {
	tmpDir := t.TempDir()

	p1 := filepath.Join(tmpDir, "times-01.json")
	p2 := filepath.Join(tmpDir, "predictions.json")
	mustWriteFile(t, p1, []byte("time data 123"))
	mustWriteFile(t, p2, []byte("predictions 456"))

	idx, _, err := MintPayloadIndex(tmpDir, "s3://store")
	if err != nil {
		t.Fatalf("MintPayloadIndex failed: %v", err)
	}

	// 1. Passes when clean
	discrepancies, err := VerifyPayloadIndex(tmpDir, idx)
	if err != nil {
		t.Fatalf("VerifyPayloadIndex returned unexpected err: %v", err)
	}
	if len(discrepancies) != 0 {
		t.Fatalf("expected 0 discrepancies for clean index, got: %v", discrepancies)
	}

	// 2. Catches modified file (same size, different content)
	mustWriteFile(t, p1, []byte("time data 999"))
	discrepancies, err = VerifyPayloadIndex(tmpDir, idx)
	if err != nil {
		t.Fatalf("VerifyPayloadIndex failed: %v", err)
	}
	if len(discrepancies) == 0 || !strings.Contains(discrepancies[0], "sha256 mismatch") {
		t.Errorf("expected sha256 mismatch discrepancy, got: %v", discrepancies)
	}
	// Restore p1
	mustWriteFile(t, p1, []byte("time data 123"))

	// 3. Catches missing file
	if err := os.Remove(p2); err != nil {
		t.Fatal(err)
	}
	discrepancies, err = VerifyPayloadIndex(tmpDir, idx)
	if err != nil {
		t.Fatalf("VerifyPayloadIndex failed: %v", err)
	}
	if len(discrepancies) == 0 || !strings.Contains(discrepancies[0], "missing payload file") {
		t.Errorf("expected missing file discrepancy, got: %v", discrepancies)
	}
	// Restore p2
	mustWriteFile(t, p2, []byte("predictions 456"))

	// 4. Catches truncated file
	mustWriteFile(t, p2, []byte("short"))
	discrepancies, err = VerifyPayloadIndex(tmpDir, idx)
	if err != nil {
		t.Fatalf("VerifyPayloadIndex failed: %v", err)
	}
	foundSizeMismatch := false
	for _, d := range discrepancies {
		if strings.Contains(d, "size mismatch") {
			foundSizeMismatch = true
			break
		}
	}
	if !foundSizeMismatch {
		t.Errorf("expected size mismatch discrepancy for truncated file, got: %v", discrepancies)
	}
}

func TestDeterministicSortStability(t *testing.T) {
	idx := PayloadIndex{
		Schema:   PayloadIndexSchema,
		StoreURI: "s3://store",
		Entries: []PayloadEntry{
			{Path: "z/run.log", Bytes: 10, SHA256: strings.Repeat("1", 64)},
			{Path: "a/run.log", Bytes: 20, SHA256: strings.Repeat("2", 64)},
			{Path: "m/run.log", Bytes: 30, SHA256: strings.Repeat("3", 64)},
			{Path: "b/run.log", Bytes: 40, SHA256: strings.Repeat("4", 64)},
		},
	}

	idx.Sort()
	wantOrder := []string{"a/run.log", "b/run.log", "m/run.log", "z/run.log"}
	for i, want := range wantOrder {
		if idx.Entries[i].Path != want {
			t.Errorf("entry[%d].Path = %q, want %q", i, idx.Entries[i].Path, want)
		}
	}

	// Lookup test
	e, ok := idx.Lookup("m/run.log")
	if !ok || e.Bytes != 30 {
		t.Errorf("Lookup(m/run.log) = (%v, %v), want bytes 30", e, ok)
	}
	e, ok = idx.Lookup("./m/run.log")
	if !ok || e.Bytes != 30 {
		t.Errorf("Lookup(./m/run.log) = (%v, %v), want bytes 30", e, ok)
	}
	_, ok = idx.Lookup("nonexistent.log")
	if ok {
		t.Error("Lookup(nonexistent.log) returned ok=true, want false")
	}
}

func mustWriteFile(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatalf("failed to write %s: %v", path, err)
	}
}
