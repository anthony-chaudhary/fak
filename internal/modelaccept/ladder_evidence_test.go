package modelaccept

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestBuildInventoryWithLadderEvidenceAdmission(t *testing.T) {
	validDir := writeLadderEvidenceFixture(t, map[string][]byte{
		"evidence-complete.json": []byte(`{"verdict":"PASS","improvement_pct":88.8}`),
		"raw-run.json":           []byte(`{"trials":3}`),
	}, nil)

	t.Run("valid bytes admit existing inventory evaluation", func(t *testing.T) {
		inventory, admission := BuildInventoryWithLadderEvidence(inventoryFixture(), inventoryOptions(), LadderEvidenceOptions{
			Directory: validDir,
			Manifest:  filepath.Join(validDir, "checksums.json"),
		})
		if admission.Verdict != Pass || admission.Reason.Code != "" {
			t.Fatalf("admission=%+v", admission)
		}
		if inventory.Verdict != Pass || len(inventory.Rows) != 1 || inventory.Rows[0].CapabilityGate != Pass || inventory.Rows[0].WitnessedTier == nil {
			t.Fatalf("inventory=%+v", inventory)
		}
	})

	t.Run("checksum mismatch holds without arithmetic pass fields", func(t *testing.T) {
		dir := cloneLadderEvidenceFixture(t, validDir)
		if err := os.WriteFile(filepath.Join(dir, "raw-run.json"), []byte(`{"trials":4}`), 0o600); err != nil {
			t.Fatal(err)
		}
		inventory, admission := BuildInventoryWithLadderEvidence(inventoryFixture(), inventoryOptions(), LadderEvidenceOptions{
			Directory: dir,
			Manifest:  filepath.Join(dir, "checksums.json"),
		})
		assertLadderEvidenceHold(t, inventory, admission, LadderEvidenceChecksumMismatch, "raw-run.json")
		if admission.Reason.ExpectedSHA256 == "" || admission.Reason.ActualSHA256 == "" || admission.Reason.ExpectedSHA256 == admission.Reason.ActualSHA256 {
			t.Fatalf("mismatch did not carry distinct expected/actual SHA-256: %+v", admission.Reason)
		}
	})

	t.Run("missing declared artifact holds", func(t *testing.T) {
		dir := cloneLadderEvidenceFixture(t, validDir)
		if err := os.Remove(filepath.Join(dir, "raw-run.json")); err != nil {
			t.Fatal(err)
		}
		inventory, admission := BuildInventoryWithLadderEvidence(inventoryFixture(), inventoryOptions(), LadderEvidenceOptions{
			Directory: dir,
			Manifest:  filepath.Join(dir, "checksums.json"),
		})
		assertLadderEvidenceHold(t, inventory, admission, LadderEvidenceMissingArtifact, "raw-run.json")
	})

	t.Run("extra undeclared artifact holds", func(t *testing.T) {
		dir := cloneLadderEvidenceFixture(t, validDir)
		if err := os.WriteFile(filepath.Join(dir, "unlisted.json"), []byte(`{}`), 0o600); err != nil {
			t.Fatal(err)
		}
		inventory, admission := BuildInventoryWithLadderEvidence(inventoryFixture(), inventoryOptions(), LadderEvidenceOptions{
			Directory: dir,
			Manifest:  filepath.Join(dir, "checksums.json"),
		})
		assertLadderEvidenceHold(t, inventory, admission, LadderEvidenceExtraArtifact, "unlisted.json")
	})
}

func TestCommittedLadderChecksumMismatchFailsClosed(t *testing.T) {
	dir := filepath.Join("..", "..", "docs", "_witnesses", "issue-8623-qwen38-27b")
	inventory, admission := BuildInventoryWithLadderEvidence(inventoryFixture(), inventoryOptions(), LadderEvidenceOptions{
		Directory: dir,
		Manifest:  filepath.Join(dir, "checksums.json"),
	})
	assertLadderEvidenceHold(t, inventory, admission, LadderEvidenceChecksumMismatch, "environment.json")
	if admission.Reason.ExpectedSHA256 != "daff998a66ab76358b02f19f8dc59b39e484232920b8a833b531e06c2462eda5" {
		t.Fatalf("committed expected SHA-256 changed: %+v", admission.Reason)
	}
	if admission.Reason.ActualSHA256 != "10b860cc29e30ebe4edd2062b2ef1a8a7c6a59d11523d9ea64bc19dbad0dfe17" {
		t.Fatalf("committed actual SHA-256 changed: %+v", admission.Reason)
	}
}

func TestVerifyLadderEvidenceCanonicalizesManifestIdentity(t *testing.T) {
	t.Run("symlink spelling of evidence root", func(t *testing.T) {
		parent := t.TempDir()
		realDir := filepath.Join(parent, "real")
		if err := os.Mkdir(realDir, 0o700); err != nil {
			t.Fatal(err)
		}
		writeLadderEvidenceFixtureAt(t, realDir, map[string][]byte{"artifact.json": []byte("bound")}, nil)
		aliasDir := filepath.Join(parent, "alias")
		if err := os.Symlink(realDir, aliasDir); err != nil {
			t.Fatal(err)
		}

		got := VerifyLadderEvidence(LadderEvidenceOptions{
			Directory: aliasDir,
			Manifest:  filepath.Join(aliasDir, "checksums.json"),
		})
		if got.Verdict != Pass {
			t.Fatalf("symlink-equivalent manifest was treated as an extra artifact: %+v", got)
		}
	})

	t.Run("external manifest remains external", func(t *testing.T) {
		dir := writeLadderEvidenceFixture(t, map[string][]byte{"artifact.json": []byte("bound")}, nil)
		externalManifest := filepath.Join(t.TempDir(), "checksums.json")
		manifest, err := os.ReadFile(filepath.Join(dir, "checksums.json"))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(externalManifest, manifest, 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Remove(filepath.Join(dir, "checksums.json")); err != nil {
			t.Fatal(err)
		}

		got := VerifyLadderEvidence(LadderEvidenceOptions{Directory: dir, Manifest: externalManifest})
		if got.Verdict != Pass {
			t.Fatalf("external manifest was folded into the evidence artifact set: %+v", got)
		}
	})
}

func TestVerifyLadderEvidenceRejectsMalformedBoundaries(t *testing.T) {
	content := []byte("bound")
	digest := sha256.Sum256(content)
	validSHA := hex.EncodeToString(digest[:])
	for _, tc := range []struct {
		name     string
		manifest []ladderChecksumEntry
		prepare  func(t *testing.T, dir string)
		code     LadderEvidenceReasonCode
		path     string
	}{
		{
			name: "duplicate path",
			manifest: []ladderChecksumEntry{
				{File: "artifact.json", SHA256: validSHA},
				{File: "artifact.json", SHA256: validSHA},
			},
			code: LadderEvidenceDuplicatePath,
			path: "artifact.json",
		},
		{
			name:     "traversal",
			manifest: []ladderChecksumEntry{{File: "../artifact.json", SHA256: validSHA}},
			code:     LadderEvidencePathTraversal,
			path:     "../artifact.json",
		},
		{
			name:     "unreadable content",
			manifest: []ladderChecksumEntry{{File: "artifact.json", SHA256: validSHA}},
			prepare: func(t *testing.T, dir string) {
				t.Helper()
				if err := os.Remove(filepath.Join(dir, "artifact.json")); err != nil {
					t.Fatal(err)
				}
				if err := os.Mkdir(filepath.Join(dir, "artifact.json"), 0o700); err != nil {
					t.Fatal(err)
				}
			},
			code: LadderEvidenceUnreadableArtifact,
			path: "artifact.json",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := writeLadderEvidenceFixture(t, map[string][]byte{"artifact.json": content}, tc.manifest)
			if tc.prepare != nil {
				tc.prepare(t, dir)
			}
			got := VerifyLadderEvidence(LadderEvidenceOptions{Directory: dir, Manifest: filepath.Join(dir, "checksums.json")})
			if got.Verdict != Hold || got.Reason.Code != tc.code || got.Reason.Path != tc.path {
				t.Fatalf("admission=%+v", got)
			}
		})
	}
}

func inventoryOptions() InventoryOptions {
	return InventoryOptions{
		Artifact:         "evidence-complete.json",
		ArtifactRevision: "internal/modelaccept@r1+gfixture",
		ExpectedCorpusID: "c1",
		AsOf:             time.Date(2026, 7, 15, 0, 0, 0, 0, time.UTC),
	}
}

func assertLadderEvidenceHold(t *testing.T, inventory Inventory, admission LadderEvidenceAdmission, code LadderEvidenceReasonCode, path string) {
	t.Helper()
	if admission.Verdict != Hold || admission.Reason.Code != code || admission.Reason.Path != path {
		t.Fatalf("admission=%+v", admission)
	}
	if inventory.Verdict != Hold || len(inventory.Rows) != 1 {
		t.Fatalf("inventory=%+v", inventory)
	}
	row := inventory.Rows[0]
	if row.CapabilityGate != Hold || row.WitnessedTier != nil || row.Samples != 0 || row.ObservedFirst != "" || row.ObservedLast != "" {
		t.Fatalf("HOLD promoted arithmetic evidence: %+v", row)
	}
	wantReason := admission.Reason.String()
	if !strings.Contains(strings.Join(row.Reasons, "\n"), wantReason) {
		t.Fatalf("row reasons do not contain typed admission evidence %q: %+v", wantReason, row.Reasons)
	}
}

func writeLadderEvidenceFixture(t *testing.T, files map[string][]byte, manifest []ladderChecksumEntry) string {
	t.Helper()
	dir := t.TempDir()
	writeLadderEvidenceFixtureAt(t, dir, files, manifest)
	return dir
}

func writeLadderEvidenceFixtureAt(t *testing.T, dir string, files map[string][]byte, manifest []ladderChecksumEntry) {
	t.Helper()
	for name, content := range files {
		path := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, content, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if manifest == nil {
		for name, content := range files {
			digest := sha256.Sum256(content)
			manifest = append(manifest, ladderChecksumEntry{File: filepath.ToSlash(name), SHA256: hex.EncodeToString(digest[:])})
		}
	}
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "checksums.json"), append(data, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
}

func cloneLadderEvidenceFixture(t *testing.T, source string) string {
	t.Helper()
	dir := t.TempDir()
	entries, err := os.ReadDir(source)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		data, err := os.ReadFile(filepath.Join(source, entry.Name()))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, entry.Name()), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}
