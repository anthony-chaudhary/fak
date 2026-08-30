package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessartifact"
	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

func TestHarnessReleaseWitnessRequiresInputs(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runHarnessRelease(&out, &errOut, []string{"witness"}); code != 1 {
		t.Fatalf("code=%d out=%s err=%s", code, out.String(), errOut.String())
	}
	if !strings.Contains(errOut.String(), "required") {
		t.Fatalf("err=%s", errOut.String())
	}
}

func TestHarnessReleaseModelUpdatePlanJSON(t *testing.T) {
	dir := t.TempDir()
	v1Path, v1Hash := writeHarnessReleaseBlob(t, dir, "v1.gguf", "v1")
	v2Path, v2Hash := writeHarnessReleaseBlob(t, dir, "v2.gguf", "v2")
	statePath := filepath.Join(dir, "state.json")
	state := harnessartifact.ModelUpgradeState{
		Schema: harnessartifact.ModelUpgradeStateSchema,
		Active: harnessartifact.ModelRevision{
			Declaration: harnessReleaseDeclaration(v1Path, v1Hash, "v1"),
			Receipt:     harnessartifact.ModelRuntimeReceipt{ID: "receipt-v1", BlobSHA256: v1Hash},
		},
	}
	writeHarnessReleaseJSON(t, statePath, state)
	requestPath := filepath.Join(dir, "request.json")
	writeHarnessReleaseJSON(t, requestPath, harnessartifact.ModelUpgradeRequest{
		StatePath:       statePath,
		Candidate:       harnessReleaseDeclaration(v2Path, v2Hash, "v2"),
		PinnedBlobBytes: 2,
	})
	var out, errOut bytes.Buffer
	if code := runHarnessRelease(&out, &errOut, []string{"model-update", "plan", "--request", requestPath}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var plan harnessartifact.ModelUpgradePlan
	if err := json.Unmarshal(out.Bytes(), &plan); err != nil {
		t.Fatal(err)
	}
	if plan.Schema != harnessartifact.ModelUpgradePlanSchema || plan.Previous.Receipt.ID != "receipt-v1" || plan.Candidate.GGUFSHA256 != v2Hash {
		t.Fatalf("plan=%+v", plan)
	}
}

func TestHarnessReleaseModelCleanupRequiresCapturedPreview(t *testing.T) {
	dir := t.TempDir()
	keep, _ := writeHarnessReleaseBlob(t, dir, "keep.gguf", "keep")
	remove, _ := writeHarnessReleaseBlob(t, dir, "remove.gguf", "remove")
	requestPath := filepath.Join(dir, "cleanup-request.json")
	writeHarnessReleaseJSON(t, requestPath, harnessartifact.ModelCleanupRequest{
		Operation: harnessartifact.CleanupPurge, CacheDir: dir, Referenced: []string{keep},
	})
	var previewOut, errOut bytes.Buffer
	if code := runHarnessRelease(&previewOut, &errOut, []string{"model-cleanup", "preview", "--request", requestPath}); code != 0 {
		t.Fatalf("preview code=%d stderr=%s", code, errOut.String())
	}
	var preview harnessartifact.ModelCleanupPreview
	if err := json.Unmarshal(previewOut.Bytes(), &preview); err != nil {
		t.Fatal(err)
	}
	if len(preview.Delete) != 1 || preview.Delete[0].Path != remove {
		t.Fatalf("preview=%+v", preview)
	}
	previewPath := filepath.Join(dir, "cleanup-preview.json")
	writeHarnessReleaseJSON(t, previewPath, preview)
	var applyOut bytes.Buffer
	errOut.Reset()
	if code := runHarnessRelease(&applyOut, &errOut, []string{"model-cleanup", "apply", "--preview", previewPath}); code != 0 {
		t.Fatalf("apply code=%d stderr=%s", code, errOut.String())
	}
	if _, err := os.Stat(remove); !os.IsNotExist(err) {
		t.Fatalf("remove path survives: %v", err)
	}
	if _, err := os.Stat(keep); err != nil {
		t.Fatalf("referenced path removed: %v", err)
	}
}

func harnessReleaseDeclaration(path, digest, id string) harnesskit.LocalModelDeclaration {
	return harnesskit.LocalModelDeclaration{
		Schema: harnessartifact.LocalModelDeclarationSchema, ModelID: id, GGUFPath: path,
		GGUFSHA256: digest, Runtime: "fake", ContextTokens: 32,
	}
}

func writeHarnessReleaseBlob(t *testing.T, dir, name, contents string) (string, string) {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256([]byte(contents))
	return path, hex.EncodeToString(sum[:])
}

func writeHarnessReleaseJSON(t *testing.T, path string, value any) {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
}
