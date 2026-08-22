package agenticbench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestEvaluatorEvidenceRungsAndRendering(t *testing.T) {
	tests := []struct {
		name      string
		level     EvaluatorEvidenceLevel
		scorer    EvaluatorScorerProvenance
		wantScore float64
	}{
		{
			name:      "scalar",
			level:     EvaluatorEvidenceOfficialScalar,
			wantScore: 0.75,
		},
		{
			name:  "structured",
			level: EvaluatorEvidenceStructuredBreakdown,
			scorer: EvaluatorScorerProvenance{
				Kind:      EvaluatorScorerDeterministic,
				Version:   "component-scorer/v3",
				Code:      evidenceRef(t, "scorer.txt"),
				Reference: ptrEvidenceRef(evidenceRef(t, "reference.json")),
			},
			wantScore: 0.6,
		},
		{
			name:  "raw",
			level: EvaluatorEvidenceRawGraderPayload,
			scorer: EvaluatorScorerProvenance{
				Kind:   EvaluatorScorerLLMJudge,
				Model:  "judge-model-2026-08-01",
				Prompt: ptrEvidenceRef(evidenceRef(t, "prompt.txt")),
				Rubric: ptrEvidenceRef(evidenceRef(t, "rubric.md")),
			},
			wantScore: 0.8,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			root := evaluatorEvidenceFixtureRoot(t, tc.name)
			manifest := fixtureEvidenceManifest(t, root, tc.name, tc.level, tc.scorer)
			path := writeEvaluatorEvidenceManifest(t, root, manifest)

			report, err := LoadEvaluatorEvidence(root, path)
			if err != nil {
				t.Fatal(err)
			}
			if report.Gate != EvaluatorEvidenceGatePass {
				t.Fatalf("gate = %q, want PASS: %+v", report.Gate, report)
			}
			if report.DeclaredLevel != tc.level || report.ResolvedLevel != tc.level {
				t.Fatalf("levels = declared %q resolved %q, want %q", report.DeclaredLevel, report.ResolvedLevel, tc.level)
			}
			if report.Score == nil || *report.Score != tc.wantScore {
				t.Fatalf("score = %v, want %v", report.Score, tc.wantScore)
			}

			rendered := RenderEvaluatorEvidence(report)
			if !strings.Contains(rendered, string(tc.level)) {
				t.Fatalf("render missing resolved level %q:\n%s", tc.level, rendered)
			}
			for _, level := range evaluatorEvidenceLevels {
				want := "available"
				if levelRank(level) > levelRank(tc.level) {
					want = "unavailable"
				}
				if !strings.Contains(rendered, "`"+string(level)+"`: "+want) {
					t.Fatalf("render missing %s %s:\n%s", level, want, rendered)
				}
			}
			if tc.level == EvaluatorEvidenceOfficialScalar &&
				!strings.Contains(rendered, "official scalar remains valid") {
				t.Fatalf("scalar render did not preserve official authority:\n%s", rendered)
			}
		})
	}
}

func TestEvaluatorEvidenceOverclaimsFailClosed(t *testing.T) {
	tests := []struct {
		name       string
		fixture    string
		mutate     func(*EvaluatorEvidenceManifest)
		wantReason string
	}{
		{
			name:    "stock scalar declared raw",
			fixture: "scalar",
			mutate: func(m *EvaluatorEvidenceManifest) {
				m.DeclaredLevel = EvaluatorEvidenceRawGraderPayload
				m.Artifacts[0].Level = EvaluatorEvidenceRawGraderPayload
			},
			wantReason: EvaluatorEvidenceReasonLevelMismatch,
		},
		{
			name:    "raw payload is stock scalar projection",
			fixture: "raw",
			mutate: func(m *EvaluatorEvidenceManifest) {
				for i := range m.Artifacts {
					if m.Artifacts[i].Kind == EvaluatorArtifactRawGraderPayload {
						m.Artifacts[i].Path = "eval_result.json"
						m.Artifacts[i].SHA256 = hashFixtureFile(t, "raw", "eval_result.json")
					}
				}
			},
			wantReason: EvaluatorEvidenceReasonRawIsScalar,
		},
		{
			name:    "stale prompt hash",
			fixture: "raw",
			mutate: func(m *EvaluatorEvidenceManifest) {
				m.Scorer.Prompt.SHA256 = "sha256:" + strings.Repeat("0", 64)
			},
			wantReason: EvaluatorEvidenceReasonHashMismatch,
		},
		{
			name:    "missing judge model identity",
			fixture: "raw",
			mutate: func(m *EvaluatorEvidenceManifest) {
				m.Scorer.Model = ""
			},
			wantReason: EvaluatorEvidenceReasonModelMissing,
		},
		{
			name:    "unresolved structured breakdown",
			fixture: "structured",
			mutate: func(m *EvaluatorEvidenceManifest) {
				for i := range m.Artifacts {
					if m.Artifacts[i].Kind == EvaluatorArtifactStructuredBreakdown {
						m.Artifacts[i].Path = "missing-breakdown.json"
					}
				}
			},
			wantReason: EvaluatorEvidenceReasonArtifactMissing,
		},
		{
			name:    "deterministic scorer missing version",
			fixture: "structured",
			mutate: func(m *EvaluatorEvidenceManifest) {
				m.Scorer.Version = ""
			},
			wantReason: EvaluatorEvidenceReasonScorerProvenance,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			root := evaluatorEvidenceFixtureRoot(t, tc.fixture)
			manifest := fixtureEvidenceManifest(t, root, tc.fixture, fixtureEvidenceLevel(tc.fixture), fixtureScorer(t, tc.fixture))
			tc.mutate(&manifest)
			path := writeEvaluatorEvidenceManifest(t, root, manifest)

			_, err := LoadEvaluatorEvidence(root, path)
			var evidenceErr *EvaluatorEvidenceError
			if !errors.As(err, &evidenceErr) {
				t.Fatalf("err = %v, want EvaluatorEvidenceError", err)
			}
			if evidenceErr.Reason != tc.wantReason {
				t.Fatalf("reason = %q, want %q (err=%v)", evidenceErr.Reason, tc.wantReason, err)
			}
		})
	}
}

func fixtureEvidenceManifest(t *testing.T, root, fixture string, level EvaluatorEvidenceLevel, scorer EvaluatorScorerProvenance) EvaluatorEvidenceManifest {
	t.Helper()
	artifacts := []EvaluatorEvidenceArtifact{{
		Kind:      EvaluatorArtifactALEEvalResult,
		Level:     EvaluatorEvidenceOfficialScalar,
		Authority: "ale_official_evaluator",
		Path:      "eval_result.json",
		SHA256:    hashFile(t, filepath.Join(root, "eval_result.json")),
	}}
	if levelRank(level) >= levelRank(EvaluatorEvidenceStructuredBreakdown) {
		artifacts = append(artifacts, EvaluatorEvidenceArtifact{
			Kind:      EvaluatorArtifactStructuredBreakdown,
			Level:     EvaluatorEvidenceStructuredBreakdown,
			Authority: "ale_task_evaluator",
			Path:      "breakdown.json",
			SHA256:    hashFile(t, filepath.Join(root, "breakdown.json")),
		})
	}
	if level == EvaluatorEvidenceRawGraderPayload {
		artifacts = append(artifacts, EvaluatorEvidenceArtifact{
			Kind:      EvaluatorArtifactRawGraderPayload,
			Level:     EvaluatorEvidenceRawGraderPayload,
			Authority: "ale_task_evaluator",
			Path:      "raw_payload.json",
			SHA256:    hashFile(t, filepath.Join(root, "raw_payload.json")),
		})
	}
	scorer = resolveFixtureRefs(t, root, scorer)
	return EvaluatorEvidenceManifest{
		Schema:        EvaluatorEvidenceSchema,
		DeclaredLevel: level,
		Artifacts:     artifacts,
		Scorer:        scorer,
	}
}

func fixtureScorer(t *testing.T, fixture string) EvaluatorScorerProvenance {
	t.Helper()
	switch fixture {
	case "structured":
		return EvaluatorScorerProvenance{
			Kind:      EvaluatorScorerDeterministic,
			Version:   "component-scorer/v3",
			Code:      evidenceRef(t, "scorer.txt"),
			Reference: ptrEvidenceRef(evidenceRef(t, "reference.json")),
		}
	case "raw":
		return EvaluatorScorerProvenance{
			Kind:   EvaluatorScorerLLMJudge,
			Model:  "judge-model-2026-08-01",
			Prompt: ptrEvidenceRef(evidenceRef(t, "prompt.txt")),
			Rubric: ptrEvidenceRef(evidenceRef(t, "rubric.md")),
		}
	default:
		return EvaluatorScorerProvenance{}
	}
}

func fixtureEvidenceLevel(fixture string) EvaluatorEvidenceLevel {
	switch fixture {
	case "structured":
		return EvaluatorEvidenceStructuredBreakdown
	case "raw":
		return EvaluatorEvidenceRawGraderPayload
	default:
		return EvaluatorEvidenceOfficialScalar
	}
}

func evaluatorEvidenceFixtureRoot(t *testing.T, fixture string) string {
	t.Helper()
	src := filepath.Join("testdata", "evaluator-evidence", fixture)
	root := t.TempDir()
	err := filepath.WalkDir(src, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		if rel == "." {
			return nil
		}
		dst := filepath.Join(root, rel)
		if entry.IsDir() {
			return os.MkdirAll(dst, 0o755)
		}
		body, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(dst, body, 0o644)
	})
	if err != nil {
		t.Fatal(err)
	}
	return root
}

func writeEvaluatorEvidenceManifest(t *testing.T, root string, manifest EvaluatorEvidenceManifest) string {
	t.Helper()
	body, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "evaluator-evidence.json")
	if err := os.WriteFile(path, append(body, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func resolveFixtureRefs(t *testing.T, root string, scorer EvaluatorScorerProvenance) EvaluatorScorerProvenance {
	t.Helper()
	if scorer.Code.Path != "" {
		scorer.Code.SHA256 = hashFile(t, filepath.Join(root, scorer.Code.Path))
	}
	if scorer.Reference != nil {
		scorer.Reference.SHA256 = hashFile(t, filepath.Join(root, scorer.Reference.Path))
	}
	if scorer.Prompt != nil {
		scorer.Prompt.SHA256 = hashFile(t, filepath.Join(root, scorer.Prompt.Path))
	}
	if scorer.Rubric != nil {
		scorer.Rubric.SHA256 = hashFile(t, filepath.Join(root, scorer.Rubric.Path))
	}
	return scorer
}

func evidenceRef(t *testing.T, path string) EvaluatorHashedArtifact {
	t.Helper()
	return EvaluatorHashedArtifact{Path: path}
}

func ptrEvidenceRef(ref EvaluatorHashedArtifact) *EvaluatorHashedArtifact {
	return &ref
}

func hashFixtureFile(t *testing.T, fixture, name string) string {
	t.Helper()
	return hashFile(t, filepath.Join("testdata", "evaluator-evidence", fixture, name))
}

func hashFile(t *testing.T, path string) string {
	t.Helper()
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(body)
	return "sha256:" + hex.EncodeToString(sum[:])
}
