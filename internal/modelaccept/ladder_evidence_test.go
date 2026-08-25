package modelaccept

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestBuildInventoryLadderEvidenceChecksumAdmission(t *testing.T) {
	validDir := t.TempDir()
	writeLadderFixture(t, validDir, map[string]string{
		"evaluator-output.json":  `{"verdict":"PASS","improvement_pct":88.8}`,
		"evidence-complete.json": `{"schema":"fak.qwen38-ladder-evidence/1"}`,
	})
	_, sourceFile, _, _ := runtime.Caller(0)
	committedDir := filepath.Clean(filepath.Join(filepath.Dir(sourceFile), "..", "..", "docs", "_witnesses", "issue-8623-qwen38-27b"))

	tests := []struct {
		name string
		dir  string
		pass bool
	}{
		{name: "byte-identical fixture admitted", dir: validDir, pass: true},
		{name: "committed packet mismatch holds", dir: committedDir, pass: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := BuildInventory(inventoryFixture(), InventoryOptions{
				Artifact: "acceptance.json", ArtifactRevision: "internal/modelaccept@r1+gabc", ExpectedCorpusID: "c1",
				AsOf: time.Date(2026, 7, 15, 0, 0, 0, 0, time.UTC), LadderEvidenceDir: tc.dir, LadderChecksumManifest: "checksums.json",
			})
			if tc.pass {
				if got.Verdict != Pass || got.LadderEvidence == nil || got.LadderEvidence.Verdict != Pass {
					t.Fatalf("valid ladder evidence inventory = %#v, want PASS admission", got)
				}
				if len(got.Rows) == 0 || got.Rows[0].WitnessedTier == nil {
					t.Fatalf("valid ladder evidence did not preserve accepted tier: %#v", got.Rows)
				}
				return
			}
			if got.Verdict != Hold || got.LadderEvidence == nil || got.LadderEvidence.Verdict != Hold {
				t.Fatalf("mismatched ladder evidence inventory = %#v, want HOLD admission", got)
			}
			if got.LadderEvidence.Code != "ladder_evidence_checksum_mismatch" || got.LadderEvidence.Path == "" || got.LadderEvidence.ExpectedSHA256 == "" || got.LadderEvidence.ActualSHA256 == "" {
				t.Fatalf("mismatch decision is not typed with path and digests: %#v", got.LadderEvidence)
			}
			if !strings.Contains(got.LadderEvidence.Reason, got.LadderEvidence.Path) || !strings.Contains(got.LadderEvidence.Reason, got.LadderEvidence.ExpectedSHA256) || !strings.Contains(got.LadderEvidence.Reason, got.LadderEvidence.ActualSHA256) {
				t.Fatalf("mismatch reason lacks path or digests: %q", got.LadderEvidence.Reason)
			}
			for _, row := range got.Rows {
				if row.CapabilityGate == Pass || row.WitnessedTier != nil {
					t.Fatalf("checksum HOLD promoted arithmetic PASS fields: %#v", row)
				}
			}
		})
	}
}

func TestVerifyLadderEvidenceRejectsUnsafeManifestEntries(t *testing.T) {
	tests := []struct{ name, manifest, code string }{
		{"omitted artifact", `[{"file":"missing.json","sha256":"` + strings.Repeat("0", 64) + `"}]`, "ladder_evidence_artifact_unreadable"},
		{"duplicate path", `[{"file":"a.json","sha256":"` + strings.Repeat("0", 64) + `"},{"file":"a.json","sha256":"` + strings.Repeat("0", 64) + `"}]`, "ladder_evidence_duplicate_path"},
		{"traversal", `[{"file":"../a.json","sha256":"` + strings.Repeat("0", 64) + `"}]`, "ladder_evidence_path_traversal"},
		{"unlisted artifact", `[{"file":"listed.json","sha256":"` + fmt.Sprintf("%x", sha256.Sum256([]byte("listed"))) + `"}]`, "ladder_evidence_unlisted_artifact"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			if tc.name == "unlisted artifact" {
				if err := os.WriteFile(filepath.Join(dir, "listed.json"), []byte("listed"), 0o600); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(dir, "extra.json"), []byte("extra"), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			if err := os.WriteFile(filepath.Join(dir, "checksums.json"), []byte(tc.manifest), 0o600); err != nil {
				t.Fatal(err)
			}
			got := verifyLadderEvidence(dir, "checksums.json")
			if got.Verdict != Hold || got.Code != tc.code {
				t.Fatalf("decision = %#v, want HOLD %s", got, tc.code)
			}
		})
	}
}

func writeLadderFixture(t *testing.T, dir string, files map[string]string) {
	t.Helper()
	type entry struct {
		File   string `json:"file"`
		SHA256 string `json:"sha256"`
	}
	entries := make([]entry, 0, len(files))
	for name, text := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(text), 0o600); err != nil {
			t.Fatal(err)
		}
		entries = append(entries, entry{File: name, SHA256: fmt.Sprintf("%x", sha256.Sum256([]byte(text)))})
	}
	data, err := json.Marshal(entries)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "checksums.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
}
