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

	"github.com/anthony-chaudhary/fak/internal/qwen38ladder"
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

func TestCommittedLadderEvidencePassesAndTamperFailsClosed(t *testing.T) {
	dir := filepath.Join("..", "..", "docs", "_witnesses", "issue-8623-qwen38-27b")
	admission := VerifyLadderEvidence(LadderEvidenceOptions{
		Directory: dir,
		Manifest:  filepath.Join(dir, "checksums.json"),
	})
	if admission.Verdict != Pass || admission.Reason.Code != "" {
		t.Fatalf("committed evidence admission = %+v", admission)
	}
	assertCommittedLadderSemantics(t, dir)

	tampered := cloneLadderEvidenceFixture(t, dir)
	rawPath := filepath.Join(tampered, "raw-run.json")
	raw, err := os.ReadFile(rawPath)
	if err != nil {
		t.Fatal(err)
	}
	raw[0] ^= 1
	if err := os.WriteFile(rawPath, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	refused := VerifyLadderEvidence(LadderEvidenceOptions{
		Directory: tampered,
		Manifest:  filepath.Join(tampered, "checksums.json"),
	})
	if refused.Verdict != Hold || refused.Reason.Code != LadderEvidenceChecksumMismatch || refused.Reason.Path != "raw-run.json" {
		t.Fatalf("one-byte tamper admission = %+v", refused)
	}
	if refused.Reason.ExpectedSHA256 == "" || refused.Reason.ActualSHA256 == "" || refused.Reason.ExpectedSHA256 == refused.Reason.ActualSHA256 {
		t.Fatalf("tamper did not carry distinct expected/actual SHA-256: %+v", refused.Reason)
	}
}

func TestReadinessInventoryLadderTamperPublishesNoArithmeticPass(t *testing.T) {
	source := filepath.Join("..", "..", "docs", "_witnesses", "issue-8623-qwen38-27b")
	dir := cloneLadderEvidenceFixture(t, source)
	rawPath := filepath.Join(dir, "raw-run.json")
	raw, err := os.ReadFile(rawPath)
	if err != nil {
		t.Fatal(err)
	}
	raw[len(raw)-1] ^= 1
	if err := os.WriteFile(rawPath, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	got, admission := BuildQwen38LadderReadinessInventory(InventoryOptions{ArtifactRevision: "docs@8cd6a82af97f"}, LadderEvidenceOptions{
		Directory: dir,
		Manifest:  filepath.Join(dir, "checksums.json"),
	})
	if admission.Verdict != Hold || admission.Reason.Code != LadderEvidenceChecksumMismatch || admission.Reason.Path != "raw-run.json" || len(got.Rows) != 1 || got.Rows[0].LadderEvidence != nil {
		t.Fatalf("inventory=%+v admission=%+v", got, admission)
	}
	for _, cell := range got.Rows[0].ReadinessCells {
		if cell.Status == ReadinessCellPass {
			t.Fatalf("tampered evidence published PASS: %+v", cell)
		}
	}
}

func assertCommittedLadderSemantics(t *testing.T, dir string) {
	t.Helper()
	type result struct {
		StageID         string  `json:"stage_id"`
		Model           string  `json:"model"`
		Revision        string  `json:"revision"`
		CorpusSHA       string  `json:"corpus_sha256"`
		EnvironmentSHA  string  `json:"environment_sha256"`
		Trials          int     `json:"trials"`
		BaselinePassed  int     `json:"baseline_passed"`
		CandidatePassed int     `json:"candidate_passed"`
		BaselineMetric  float64 `json:"baseline_metric"`
		CandidateMetric float64 `json:"candidate_metric"`
	}
	var evidence struct {
		Schema              string   `json:"schema"`
		BaselineRuntimeSHA  string   `json:"baseline_runtime_sha"`
		CandidateRuntimeSHA string   `json:"candidate_runtime_sha"`
		Results             []result `json:"results"`
	}
	decodeJSONFile(t, filepath.Join(dir, "evidence-complete.json"), &evidence)
	if evidence.Schema != qwen38ladder.Schema || len(evidence.Results) != len(qwen38ladder.Stages) {
		t.Fatalf("evidence identity = schema %q, results %d", evidence.Schema, len(evidence.Results))
	}
	for i, stage := range qwen38ladder.Stages {
		result := evidence.Results[i]
		if result.StageID != stage.ID || result.Model != stage.Model || result.Revision != stage.Revision {
			t.Fatalf("evidence result[%d]=%+v, want stage %+v", i, result, stage)
		}
	}
	target := evidence.Results[len(evidence.Results)-1]
	if target.StageID != "target" || target.Model != "Qwen/Qwen3.8-27B" || target.Revision != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" || target.Trials != 3 || target.BaselinePassed != 3 || target.CandidatePassed != 3 || target.BaselineMetric != 3378.019733 || target.CandidateMetric != 376.181809 {
		t.Fatalf("target evidence = %+v", target)
	}

	var environment struct {
		Model              string `json:"model"`
		Revision           string `json:"revision"`
		DType              string `json:"dtype"`
		TensorParallelSize int    `json:"tensor_parallel_size"`
		EnvironmentSHA     string `json:"environment_sha256"`
		Arms               struct {
			Baseline struct {
				RuntimeSHA string `json:"runtime_sha"`
			} `json:"baseline"`
			Candidate struct {
				RuntimeSHA string `json:"runtime_sha"`
			} `json:"candidate"`
		} `json:"arms"`
	}
	decodeJSONFile(t, filepath.Join(dir, "environment.json"), &environment)
	if environment.Model != target.Model || environment.Revision != target.Revision || environment.DType != "bfloat16" || environment.TensorParallelSize != 2 || environment.EnvironmentSHA != target.EnvironmentSHA || environment.Arms.Baseline.RuntimeSHA != evidence.BaselineRuntimeSHA || environment.Arms.Candidate.RuntimeSHA != evidence.CandidateRuntimeSHA {
		t.Fatalf("environment does not bind target evidence: %+v", environment)
	}

	var raw struct {
		EnvironmentSHA string `json:"environment_sha256"`
		CorpusSHA      string `json:"corpus_sha256"`
		Summary        struct {
			Baseline struct {
				Passed, Trials int
				P95MS          float64 `json:"p95_ms"`
			} `json:"baseline"`
			Candidate struct {
				Passed, Trials int
				P95MS          float64 `json:"p95_ms"`
			} `json:"candidate"`
		} `json:"summary"`
	}
	decodeJSONFile(t, filepath.Join(dir, "raw-run.json"), &raw)
	corpus, err := os.ReadFile(filepath.Join(dir, "corpus.json"))
	if err != nil {
		t.Fatal(err)
	}
	corpusDigest := sha256.Sum256(corpus)
	corpusSHA := hex.EncodeToString(corpusDigest[:])
	if raw.EnvironmentSHA != target.EnvironmentSHA || raw.CorpusSHA != target.CorpusSHA || corpusSHA != target.CorpusSHA || raw.Summary.Baseline.Passed != target.BaselinePassed || raw.Summary.Baseline.Trials != target.Trials || raw.Summary.Baseline.P95MS != target.BaselineMetric || raw.Summary.Candidate.Passed != target.CandidatePassed || raw.Summary.Candidate.Trials != target.Trials || raw.Summary.Candidate.P95MS != target.CandidateMetric {
		t.Fatalf("raw run does not bind target evidence: raw=%+v corpus_sha256=%s", raw, corpusSHA)
	}

	var evaluator struct {
		Verdict        string  `json:"verdict"`
		ImprovementPct float64 `json:"improvement_pct"`
	}
	decodeJSONFile(t, filepath.Join(dir, "evaluator-output.json"), &evaluator)
	improvement := (target.BaselineMetric - target.CandidateMetric) / target.BaselineMetric * 100
	if evaluator.Verdict != "PASS" || evaluator.ImprovementPct != improvement {
		t.Fatalf("evaluator output = %+v, computed improvement = %.17g", evaluator, improvement)
	}
}

func decodeJSONFile(t *testing.T, name string, dst any) {
	t.Helper()
	data, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, dst); err != nil {
		t.Fatal(err)
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
