package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/resultstier"
)

func setupTestResultsFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()

	// Claim files (3)
	mustWriteTestFile(t, filepath.Join(dir, "INDEX.md"), []byte("# Results Index\n"))
	mustWriteTestFile(t, filepath.Join(dir, "perf.json"), []byte(`{"latency_ms": 42.5}`))
	mustWriteTestFile(t, filepath.Join(dir, "eval_summary.json"), []byte(`{"accuracy": 0.98}`))

	// Payload files (3)
	mustWriteTestFile(t, filepath.Join(dir, "predictions_run1.json"), []byte("model raw output 12345"))
	mustWriteTestFile(t, filepath.Join(dir, "build.log"), []byte("step 1: compile\nstep 2: test\n"))
	nestedDir := filepath.Join(dir, "artifacts")
	if err := os.MkdirAll(nestedDir, 0755); err != nil {
		t.Fatal(err)
	}
	mustWriteTestFile(t, filepath.Join(nestedDir, "times-001.json"), []byte("[10.1, 10.2, 10.3]"))

	// Unknown files (1)
	mustWriteTestFile(t, filepath.Join(dir, "artifact.bin"), []byte{0xde, 0xad, 0xbe, 0xef})

	return dir
}

func mustWriteTestFile(t *testing.T, path string, content []byte) {
	t.Helper()
	if err := os.WriteFile(path, content, 0644); err != nil {
		t.Fatalf("failed to write %s: %v", path, err)
	}
}

func TestResults(t *testing.T) {
	t.Run("DefaultCensusText", func(t *testing.T) {
		dir := setupTestResultsFixture(t)
		var out, errb bytes.Buffer
		rc := runResults(&out, &errb, []string{"tier", "--dir", dir})
		if rc != 0 {
			t.Fatalf("expected rc 0, got %d, stderr: %s", rc, errb.String())
		}
		s := out.String()
		if !strings.Contains(s, "Results Tier Census:") {
			t.Errorf("output missing header: %s", s)
		}
		if !strings.Contains(s, "Claim:   3 files") {
			t.Errorf("expected 3 claim files in output: %s", s)
		}
		if !strings.Contains(s, "Payload: 3 files") {
			t.Errorf("expected 3 payload files in output: %s", s)
		}
		if !strings.Contains(s, "Unknown: 1 files") {
			t.Errorf("expected 1 unknown file in output: %s", s)
		}
		if !strings.Contains(s, "Total: 7 files") {
			t.Errorf("expected 7 total files in output: %s", s)
		}
		if !strings.Contains(s, "Externalize shrink:") {
			t.Errorf("output missing shrink: %s", s)
		}
		// Without --unknown, individual unknown files should not be listed
		if strings.Contains(s, "Unknown files (1):") {
			t.Errorf("unexpected unknown files list without --unknown flag: %s", s)
		}
	})

	t.Run("DefaultCensusJSON", func(t *testing.T) {
		dir := setupTestResultsFixture(t)
		var out, errb bytes.Buffer
		rc := runResults(&out, &errb, []string{"tier", "--dir", dir, "--json"})
		if rc != 0 {
			t.Fatalf("expected rc 0, got %d, stderr: %s", rc, errb.String())
		}

		var rep map[string]any
		if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
			t.Fatalf("failed to unmarshal JSON: %v, raw: %s", err, out.String())
		}

		if rep["schema"] != "fak-results-tier/1" {
			t.Errorf("schema = %v, want fak-results-tier/1", rep["schema"])
		}
		if rep["claim_files"] != float64(3) {
			t.Errorf("claim_files = %v, want 3", rep["claim_files"])
		}
		if rep["payload_files"] != float64(3) {
			t.Errorf("payload_files = %v, want 3", rep["payload_files"])
		}
		if rep["unknown_files"] != float64(1) {
			t.Errorf("unknown_files = %v, want 1", rep["unknown_files"])
		}
		if rep["total_files"] != float64(7) {
			t.Errorf("total_files = %v, want 7", rep["total_files"])
		}
		if share, ok := rep["payload_share"].(float64); !ok || share <= 0 {
			t.Errorf("payload_share = %v, want > 0", rep["payload_share"])
		}
		if shrink, ok := rep["shrink"].(float64); !ok || shrink < 1.0 {
			t.Errorf("shrink = %v, want >= 1.0", rep["shrink"])
		}
	})

	t.Run("MintMode", func(t *testing.T) {
		dir := setupTestResultsFixture(t)
		var out, errb bytes.Buffer
		rc := runResults(&out, &errb, []string{"tier", "--dir", dir, "--mint"})
		if rc != 0 {
			t.Fatalf("expected rc 0, got %d, stderr: %s", rc, errb.String())
		}
		s := out.String()
		if !strings.Contains(s, "minted") || !strings.Contains(s, "3 entries") {
			t.Errorf("unexpected mint output: %s", s)
		}

		indexPath := filepath.Join(dir, "payload-index.json")
		data, err := os.ReadFile(indexPath)
		if err != nil {
			t.Fatalf("failed to read minted index: %v", err)
		}

		var idx resultstier.PayloadIndex
		if err := json.Unmarshal(data, &idx); err != nil {
			t.Fatalf("minted index invalid JSON: %v", err)
		}

		if idx.Schema != resultstier.PayloadIndexSchema {
			t.Errorf("schema = %q, want %q", idx.Schema, resultstier.PayloadIndexSchema)
		}
		if idx.StoreURI != "blob://fak-results-payload" {
			t.Errorf("storeURI = %q, want default", idx.StoreURI)
		}
		if len(idx.Entries) != 3 {
			t.Fatalf("len(idx.Entries) = %d, want 3", len(idx.Entries))
		}
		for _, e := range idx.Entries {
			if err := e.Complete(); err != nil {
				t.Errorf("entry incomplete: %v", err)
			}
		}
	})

	t.Run("MintModeJSON", func(t *testing.T) {
		dir := setupTestResultsFixture(t)
		var out, errb bytes.Buffer
		rc := runResults(&out, &errb, []string{"tier", "--dir", dir, "--mint", "--json"})
		if rc != 0 {
			t.Fatalf("expected rc 0, got %d, stderr: %s", rc, errb.String())
		}
		var rep map[string]any
		if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
			t.Fatalf("failed to decode JSON: %v, raw: %s", err, out.String())
		}
		if rep["schema"] != "fak-results-tier/1" || rep["action"] != "mint" {
			t.Errorf("unexpected mint JSON: %v", rep)
		}
		if rep["entries"] != float64(3) {
			t.Errorf("entries = %v, want 3", rep["entries"])
		}
	})

	t.Run("VerifyModeSuccess", func(t *testing.T) {
		dir := setupTestResultsFixture(t)
		var out, errb bytes.Buffer
		// First mint
		rc := runResults(&out, &errb, []string{"tier", "--dir", dir, "--mint"})
		if rc != 0 {
			t.Fatalf("mint failed: %d, stderr: %s", rc, errb.String())
		}

		// Then verify
		out.Reset()
		errb.Reset()
		rc = runResults(&out, &errb, []string{"tier", "--dir", dir, "--verify"})
		if rc != 0 {
			t.Fatalf("verify failed: %d, stderr: %s, stdout: %s", rc, errb.String(), out.String())
		}
		if !strings.Contains(out.String(), "OK") {
			t.Errorf("expected OK in verify output: %s", out.String())
		}
	})

	t.Run("VerifyModeMutatedFile", func(t *testing.T) {
		dir := setupTestResultsFixture(t)
		var out, errb bytes.Buffer
		// Mint first
		rc := runResults(&out, &errb, []string{"tier", "--dir", dir, "--mint"})
		if rc != 0 {
			t.Fatalf("mint failed: %d", rc)
		}

		// Mutate a payload file
		mustWriteTestFile(t, filepath.Join(dir, "predictions_run1.json"), []byte("corrupted predictions"))

		out.Reset()
		errb.Reset()
		rc = runResults(&out, &errb, []string{"tier", "--dir", dir, "--verify"})
		if rc != 1 {
			t.Fatalf("expected rc 1 on discrepancy, got %d", rc)
		}
		if !strings.Contains(out.String(), "sha256 mismatch") {
			t.Errorf("expected sha256 mismatch in output: %s", out.String())
		}
	})

	t.Run("VerifyModeDeletedFile", func(t *testing.T) {
		dir := setupTestResultsFixture(t)
		var out, errb bytes.Buffer
		// Mint first
		rc := runResults(&out, &errb, []string{"tier", "--dir", dir, "--mint"})
		if rc != 0 {
			t.Fatalf("mint failed: %d", rc)
		}

		// Delete a payload file
		if err := os.Remove(filepath.Join(dir, "build.log")); err != nil {
			t.Fatal(err)
		}

		out.Reset()
		errb.Reset()
		rc = runResults(&out, &errb, []string{"tier", "--dir", dir, "--verify"})
		if rc != 1 {
			t.Fatalf("expected rc 1 on discrepancy, got %d", rc)
		}
		if !strings.Contains(out.String(), "missing payload file") {
			t.Errorf("expected missing payload file in output: %s", out.String())
		}
	})

	t.Run("UnknownMode", func(t *testing.T) {
		dir := setupTestResultsFixture(t)
		var out, errb bytes.Buffer
		rc := runResults(&out, &errb, []string{"tier", "--dir", dir, "--unknown"})
		if rc != 0 {
			t.Fatalf("expected rc 0, got %d, stderr: %s", rc, errb.String())
		}
		s := out.String()
		if !strings.Contains(s, "Unknown files (1):") {
			t.Errorf("expected Unknown files header: %s", s)
		}
		if !strings.Contains(s, "artifact.bin") {
			t.Errorf("expected artifact.bin in unknown files: %s", s)
		}
	})

	t.Run("MutuallyExclusiveFlags", func(t *testing.T) {
		dir := setupTestResultsFixture(t)
		var out, errb bytes.Buffer
		rc := runResults(&out, &errb, []string{"tier", "--dir", dir, "--mint", "--verify"})
		if rc != 2 {
			t.Fatalf("expected rc 2 for mutually exclusive flags, got %d", rc)
		}
		if !strings.Contains(errb.String(), "mutually exclusive") {
			t.Errorf("expected mutually exclusive message in stderr: %s", errb.String())
		}
	})

	t.Run("CustomStoreURI", func(t *testing.T) {
		dir := setupTestResultsFixture(t)
		var out, errb bytes.Buffer
		customStore := "s3://custom-bucket/store"
		rc := runResults(&out, &errb, []string{"tier", "--dir", dir, "--mint", "--store", customStore})
		if rc != 0 {
			t.Fatalf("mint failed: %d, stderr: %s", rc, errb.String())
		}
		indexPath := filepath.Join(dir, "payload-index.json")
		data, err := os.ReadFile(indexPath)
		if err != nil {
			t.Fatal(err)
		}
		var idx resultstier.PayloadIndex
		if err := json.Unmarshal(data, &idx); err != nil {
			t.Fatal(err)
		}
		if idx.StoreURI != customStore {
			t.Errorf("StoreURI = %q, want %q", idx.StoreURI, customStore)
		}
	})

	t.Run("UsageAndSubcommands", func(t *testing.T) {
		var out, errb bytes.Buffer
		// Empty argv -> usage to stderr, rc 2
		if rc := runResults(&out, &errb, []string{}); rc != 2 {
			t.Errorf("expected rc 2 for empty argv, got %d", rc)
		}
		if !strings.Contains(errb.String(), "fak results") {
			t.Errorf("expected usage in stderr: %s", errb.String())
		}

		// Help flag -> usage to stdout, rc 0
		out.Reset()
		errb.Reset()
		if rc := runResults(&out, &errb, []string{"-h"}); rc != 0 {
			t.Errorf("expected rc 0 for -h, got %d", rc)
		}
		if !strings.Contains(out.String(), "Usage:") {
			t.Errorf("expected usage in stdout: %s", out.String())
		}

		// Unknown subcommand -> rc 2
		out.Reset()
		errb.Reset()
		if rc := runResults(&out, &errb, []string{"foo"}); rc != 2 {
			t.Errorf("expected rc 2 for unknown subcommand, got %d", rc)
		}
		if !strings.Contains(errb.String(), "unknown subcommand") {
			t.Errorf("expected unknown subcommand in stderr: %s", errb.String())
		}
	})

	t.Run("NonExistentDir", func(t *testing.T) {
		var out, errb bytes.Buffer
		nonExistent := filepath.Join(t.TempDir(), "does-not-exist")
		rc := runResults(&out, &errb, []string{"tier", "--dir", nonExistent})
		if rc != 1 {
			t.Fatalf("expected rc 1 for non-existent dir, got %d", rc)
		}
		if len(errb.String()) == 0 {
			t.Error("expected error message in stderr for non-existent dir")
		}
	})
}
