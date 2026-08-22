package agenticbench

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
)

const EvaluatorEvidenceSchema = "fak.agenticbench-evaluator-evidence.v1"

type EvaluatorEvidenceLevel string

const (
	EvaluatorEvidenceOfficialScalar      EvaluatorEvidenceLevel = "official_scalar"
	EvaluatorEvidenceStructuredBreakdown EvaluatorEvidenceLevel = "structured_breakdown"
	EvaluatorEvidenceRawGraderPayload    EvaluatorEvidenceLevel = "raw_grader_payload"
)

var evaluatorEvidenceLevels = []EvaluatorEvidenceLevel{
	EvaluatorEvidenceOfficialScalar,
	EvaluatorEvidenceStructuredBreakdown,
	EvaluatorEvidenceRawGraderPayload,
}

const (
	EvaluatorArtifactALEEvalResult       = "ale_eval_result"
	EvaluatorArtifactStructuredBreakdown = "structured_breakdown"
	EvaluatorArtifactRawGraderPayload    = "raw_grader_payload"

	EvaluatorScorerDeterministic = "deterministic"
	EvaluatorScorerLLMJudge      = "llm_judge"

	EvaluatorEvidenceGatePass = "PASS"

	EvaluatorEvidenceReasonSchema            = "EVALUATOR_EVIDENCE_SCHEMA"
	EvaluatorEvidenceReasonLevelMismatch     = "EVALUATOR_EVIDENCE_LEVEL_MISMATCH"
	EvaluatorEvidenceReasonArtifactMissing   = "EVALUATOR_EVIDENCE_ARTIFACT_MISSING"
	EvaluatorEvidenceReasonHashMismatch      = "EVALUATOR_EVIDENCE_HASH_MISMATCH"
	EvaluatorEvidenceReasonRawIsScalar       = "EVALUATOR_EVIDENCE_RAW_IS_SCALAR"
	EvaluatorEvidenceReasonModelMissing      = "EVALUATOR_EVIDENCE_MODEL_MISSING"
	EvaluatorEvidenceReasonScorerProvenance  = "EVALUATOR_EVIDENCE_SCORER_PROVENANCE"
	EvaluatorEvidenceReasonMalformedArtifact = "EVALUATOR_EVIDENCE_MALFORMED_ARTIFACT"
	EvaluatorEvidenceReasonOverclaim         = "EVALUATOR_EVIDENCE_OVERCLAIM"
)

type EvaluatorHashedArtifact struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

type EvaluatorScorerProvenance struct {
	Kind      string                   `json:"kind,omitempty"`
	Version   string                   `json:"version,omitempty"`
	Code      EvaluatorHashedArtifact  `json:"code,omitempty"`
	Reference *EvaluatorHashedArtifact `json:"reference,omitempty"`
	Model     string                   `json:"model,omitempty"`
	Prompt    *EvaluatorHashedArtifact `json:"prompt,omitempty"`
	Rubric    *EvaluatorHashedArtifact `json:"rubric,omitempty"`
}

type EvaluatorEvidenceArtifact struct {
	Kind      string                 `json:"kind"`
	Level     EvaluatorEvidenceLevel `json:"level"`
	Authority string                 `json:"authority"`
	Path      string                 `json:"path"`
	SHA256    string                 `json:"sha256"`
}

type EvaluatorEvidenceManifest struct {
	Schema        string                      `json:"schema"`
	DeclaredLevel EvaluatorEvidenceLevel      `json:"declared_level"`
	Artifacts     []EvaluatorEvidenceArtifact `json:"artifacts"`
	Scorer        EvaluatorScorerProvenance   `json:"scorer,omitempty"`
}

type EvaluatorEvidenceLevelStatus struct {
	Level             EvaluatorEvidenceLevel `json:"level"`
	Available         bool                   `json:"available"`
	Authority         string                 `json:"authority,omitempty"`
	ArtifactPath      string                 `json:"artifact_path,omitempty"`
	ArtifactSHA256    string                 `json:"artifact_sha256,omitempty"`
	UnavailableReason string                 `json:"unavailable_reason,omitempty"`
}

type EvaluatorEvidenceReport struct {
	Schema        string                         `json:"schema"`
	ManifestPath  string                         `json:"manifest_path"`
	Gate          string                         `json:"gate"`
	DeclaredLevel EvaluatorEvidenceLevel         `json:"declared_level"`
	ResolvedLevel EvaluatorEvidenceLevel         `json:"resolved_level"`
	EvalStatus    string                         `json:"eval_status"`
	Score         *float64                       `json:"score"`
	Scorer        EvaluatorScorerProvenance      `json:"scorer,omitempty"`
	Levels        []EvaluatorEvidenceLevelStatus `json:"levels"`
}

type EvaluatorEvidenceError struct {
	Reason string
	Detail string
}

func (e *EvaluatorEvidenceError) Error() string {
	return e.Reason + ": " + e.Detail
}

// LoadEvaluatorEvidence resolves every declared artifact from disk before it
// grants the manifest's evidence rung. Stock ALE eval_result.json is always
// typed as an official scalar projection, regardless of what the caller claims.
func LoadEvaluatorEvidence(root, manifestPath string) (EvaluatorEvidenceReport, error) {
	manifest, err := readEvaluatorEvidenceManifest(manifestPath)
	if err != nil {
		return EvaluatorEvidenceReport{}, err
	}
	if manifest.Schema != EvaluatorEvidenceSchema {
		return EvaluatorEvidenceReport{}, evidenceError(EvaluatorEvidenceReasonSchema, "schema must be "+EvaluatorEvidenceSchema)
	}
	if levelRank(manifest.DeclaredLevel) == 0 {
		return EvaluatorEvidenceReport{}, evidenceError(EvaluatorEvidenceReasonSchema, fmt.Sprintf("unknown declared level %q", manifest.DeclaredLevel))
	}

	byLevel := make(map[EvaluatorEvidenceLevel]EvaluatorEvidenceArtifact, len(manifest.Artifacts))
	var evalStatus string
	var score *float64
	for _, artifact := range manifest.Artifacts {
		expectedLevel := artifactLevel(artifact.Kind)
		if expectedLevel == "" {
			return EvaluatorEvidenceReport{}, evidenceError(EvaluatorEvidenceReasonSchema, fmt.Sprintf("unknown artifact kind %q", artifact.Kind))
		}
		if artifact.Level != expectedLevel {
			return EvaluatorEvidenceReport{}, evidenceError(
				EvaluatorEvidenceReasonLevelMismatch,
				fmt.Sprintf("artifact kind %q resolves only to %q, not %q", artifact.Kind, expectedLevel, artifact.Level),
			)
		}
		if strings.TrimSpace(artifact.Authority) == "" {
			return EvaluatorEvidenceReport{}, evidenceError(EvaluatorEvidenceReasonSchema, fmt.Sprintf("%s artifact authority is required", artifact.Level))
		}
		if _, exists := byLevel[artifact.Level]; exists {
			return EvaluatorEvidenceReport{}, evidenceError(EvaluatorEvidenceReasonSchema, fmt.Sprintf("duplicate %s artifact", artifact.Level))
		}

		body, err := readVerifiedArtifact(root, EvaluatorHashedArtifact{Path: artifact.Path, SHA256: artifact.SHA256})
		if err != nil {
			return EvaluatorEvidenceReport{}, err
		}
		doc, err := decodeJSONObject(body)
		if err != nil {
			return EvaluatorEvidenceReport{}, evidenceError(
				EvaluatorEvidenceReasonMalformedArtifact,
				fmt.Sprintf("%s artifact %q: %v", artifact.Level, artifact.Path, err),
			)
		}
		switch artifact.Kind {
		case EvaluatorArtifactALEEvalResult:
			evalStatus, score, err = parseALEEvalResult(doc)
			if err != nil {
				return EvaluatorEvidenceReport{}, err
			}
		case EvaluatorArtifactStructuredBreakdown:
			if !hasStructuredBreakdown(doc) {
				return EvaluatorEvidenceReport{}, evidenceError(
					EvaluatorEvidenceReasonMalformedArtifact,
					fmt.Sprintf("structured breakdown %q has no non-empty components, checks, reasons, or breakdown", artifact.Path),
				)
			}
		case EvaluatorArtifactRawGraderPayload:
			if isStockALEEvalResult(doc) {
				return EvaluatorEvidenceReport{}, evidenceError(
					EvaluatorEvidenceReasonRawIsScalar,
					fmt.Sprintf("%q is the stock ALE scalar projection and cannot satisfy raw_grader_payload", artifact.Path),
				)
			}
			if len(doc) == 0 {
				return EvaluatorEvidenceReport{}, evidenceError(EvaluatorEvidenceReasonMalformedArtifact, "raw grader payload is empty")
			}
		}
		byLevel[artifact.Level] = artifact
	}

	for _, level := range evaluatorEvidenceLevels {
		if levelRank(level) > levelRank(manifest.DeclaredLevel) {
			break
		}
		if _, ok := byLevel[level]; !ok {
			return EvaluatorEvidenceReport{}, evidenceError(
				EvaluatorEvidenceReasonOverclaim,
				fmt.Sprintf("declared %s but no resolved %s artifact exists", manifest.DeclaredLevel, level),
			)
		}
	}
	if err := validateEvaluatorScorer(root, manifest.DeclaredLevel, manifest.Scorer); err != nil {
		return EvaluatorEvidenceReport{}, err
	}

	report := EvaluatorEvidenceReport{
		Schema:        EvaluatorEvidenceSchema,
		ManifestPath:  filepath.ToSlash(manifestPath),
		Gate:          EvaluatorEvidenceGatePass,
		DeclaredLevel: manifest.DeclaredLevel,
		ResolvedLevel: manifest.DeclaredLevel,
		EvalStatus:    evalStatus,
		Score:         score,
		Scorer:        manifest.Scorer,
	}
	for _, level := range evaluatorEvidenceLevels {
		artifact, available := byLevel[level]
		status := EvaluatorEvidenceLevelStatus{Level: level, Available: available}
		if available {
			status.Authority = artifact.Authority
			status.ArtifactPath = artifact.Path
			status.ArtifactSHA256 = artifact.SHA256
		} else {
			status.UnavailableReason = unavailableEvidenceReason(level, manifest.DeclaredLevel)
		}
		report.Levels = append(report.Levels, status)
	}
	return report, nil
}

func RenderEvaluatorEvidence(report EvaluatorEvidenceReport) string {
	var b strings.Builder
	fmt.Fprintf(&b, "## Evaluator Evidence\n\n")
	fmt.Fprintf(&b, "- Gate: `%s`\n", report.Gate)
	fmt.Fprintf(&b, "- Declared level: `%s`\n", report.DeclaredLevel)
	fmt.Fprintf(&b, "- Resolved level: `%s`\n", report.ResolvedLevel)
	fmt.Fprintf(&b, "- Evaluator status: `%s`\n", report.EvalStatus)
	if report.Score == nil {
		fmt.Fprintf(&b, "- Official score: `null`\n")
	} else {
		fmt.Fprintf(&b, "- Official score: `%g`\n", *report.Score)
	}
	fmt.Fprintf(&b, "\n### Evidence Rungs\n\n")
	for _, status := range report.Levels {
		if status.Available {
			fmt.Fprintf(&b, "- `%s`: available — authority `%s`, artifact `%s`, `%s`\n",
				status.Level, status.Authority, status.ArtifactPath, status.ArtifactSHA256)
			continue
		}
		fmt.Fprintf(&b, "- `%s`: unavailable — %s\n", status.Level, status.UnavailableReason)
	}
	return b.String()
}

func readEvaluatorEvidenceManifest(path string) (EvaluatorEvidenceManifest, error) {
	body, err := os.ReadFile(path)
	if err != nil {
		return EvaluatorEvidenceManifest{}, evidenceError(EvaluatorEvidenceReasonArtifactMissing, fmt.Sprintf("manifest %q: %v", path, err))
	}
	var manifest EvaluatorEvidenceManifest
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&manifest); err != nil {
		return EvaluatorEvidenceManifest{}, evidenceError(EvaluatorEvidenceReasonSchema, fmt.Sprintf("manifest %q: %v", path, err))
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return EvaluatorEvidenceManifest{}, evidenceError(EvaluatorEvidenceReasonSchema, "manifest has a trailing JSON value")
		}
		return EvaluatorEvidenceManifest{}, evidenceError(EvaluatorEvidenceReasonSchema, fmt.Sprintf("manifest trailing data: %v", err))
	}
	return manifest, nil
}

func validateEvaluatorScorer(root string, level EvaluatorEvidenceLevel, scorer EvaluatorScorerProvenance) error {
	if level == EvaluatorEvidenceOfficialScalar && scorer.Kind == "" {
		return nil
	}
	switch scorer.Kind {
	case EvaluatorScorerDeterministic:
		if strings.TrimSpace(scorer.Version) == "" || scorer.Code.Path == "" || scorer.Code.SHA256 == "" {
			return evidenceError(EvaluatorEvidenceReasonScorerProvenance, "deterministic scorer requires version and hashed code")
		}
		if _, err := readVerifiedArtifact(root, scorer.Code); err != nil {
			return err
		}
		if scorer.Reference != nil {
			if _, err := readVerifiedArtifact(root, *scorer.Reference); err != nil {
				return err
			}
		}
	case EvaluatorScorerLLMJudge:
		if strings.TrimSpace(scorer.Model) == "" {
			return evidenceError(EvaluatorEvidenceReasonModelMissing, "LLM judge model identity is required")
		}
		if scorer.Prompt == nil || scorer.Rubric == nil {
			return evidenceError(EvaluatorEvidenceReasonScorerProvenance, "LLM judge requires hashed prompt and rubric artifacts")
		}
		if _, err := readVerifiedArtifact(root, *scorer.Prompt); err != nil {
			return err
		}
		if _, err := readVerifiedArtifact(root, *scorer.Rubric); err != nil {
			return err
		}
	default:
		return evidenceError(EvaluatorEvidenceReasonScorerProvenance, fmt.Sprintf("evidence level %s requires deterministic or llm_judge scorer provenance", level))
	}
	return nil
}

func readVerifiedArtifact(root string, ref EvaluatorHashedArtifact) ([]byte, error) {
	if ref.Path == "" {
		return nil, evidenceError(EvaluatorEvidenceReasonArtifactMissing, "artifact path is required")
	}
	if filepath.IsAbs(ref.Path) {
		return nil, evidenceError(EvaluatorEvidenceReasonArtifactMissing, fmt.Sprintf("artifact path %q must be relative", ref.Path))
	}
	clean := filepath.Clean(filepath.FromSlash(ref.Path))
	if clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return nil, evidenceError(EvaluatorEvidenceReasonArtifactMissing, fmt.Sprintf("artifact path %q escapes the evidence root", ref.Path))
	}
	path := filepath.Join(root, clean)
	body, err := os.ReadFile(path)
	if err != nil {
		return nil, evidenceError(EvaluatorEvidenceReasonArtifactMissing, fmt.Sprintf("artifact %q: %v", ref.Path, err))
	}
	actual := sha256.Sum256(body)
	actualDigest := "sha256:" + hex.EncodeToString(actual[:])
	if ref.SHA256 != actualDigest {
		return nil, evidenceError(
			EvaluatorEvidenceReasonHashMismatch,
			fmt.Sprintf("artifact %q hash mismatch: got %s, want %s", ref.Path, actualDigest, ref.SHA256),
		)
	}
	return body, nil
}

func parseALEEvalResult(doc map[string]any) (string, *float64, error) {
	if !isStockALEEvalResult(doc) {
		return "", nil, evidenceError(
			EvaluatorEvidenceReasonMalformedArtifact,
			"ALE eval_result.json must contain only eval_status, score, eval_duration_s, and error",
		)
	}
	status, _ := doc["eval_status"].(string)
	if strings.TrimSpace(status) == "" {
		return "", nil, evidenceError(EvaluatorEvidenceReasonMalformedArtifact, "ALE eval_result.json eval_status is required")
	}
	rawScore, present := doc["score"]
	if !present || rawScore == nil {
		return status, nil, nil
	}
	var score float64
	switch value := rawScore.(type) {
	case json.Number:
		parsed, err := value.Float64()
		if err != nil {
			return "", nil, evidenceError(EvaluatorEvidenceReasonMalformedArtifact, "ALE eval_result.json score must be numeric or null")
		}
		score = parsed
	case float64:
		score = value
	default:
		return "", nil, evidenceError(EvaluatorEvidenceReasonMalformedArtifact, "ALE eval_result.json score must be numeric or null")
	}
	if math.IsNaN(score) || math.IsInf(score, 0) {
		return "", nil, evidenceError(EvaluatorEvidenceReasonMalformedArtifact, "ALE eval_result.json score must be finite")
	}
	return status, &score, nil
}

func decodeJSONObject(body []byte) (map[string]any, error) {
	var doc map[string]any
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	if err := dec.Decode(&doc); err != nil {
		return nil, err
	}
	if doc == nil {
		return nil, fmt.Errorf("expected JSON object")
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return nil, fmt.Errorf("trailing JSON value")
		}
		return nil, err
	}
	return doc, nil
}

func artifactLevel(kind string) EvaluatorEvidenceLevel {
	switch kind {
	case EvaluatorArtifactALEEvalResult:
		return EvaluatorEvidenceOfficialScalar
	case EvaluatorArtifactStructuredBreakdown:
		return EvaluatorEvidenceStructuredBreakdown
	case EvaluatorArtifactRawGraderPayload:
		return EvaluatorEvidenceRawGraderPayload
	default:
		return ""
	}
}

func levelRank(level EvaluatorEvidenceLevel) int {
	for i, candidate := range evaluatorEvidenceLevels {
		if level == candidate {
			return i + 1
		}
	}
	return 0
}

func isStockALEEvalResult(doc map[string]any) bool {
	if _, ok := doc["eval_status"]; !ok {
		return false
	}
	if _, ok := doc["score"]; !ok {
		return false
	}
	allowed := map[string]bool{
		"eval_status":     true,
		"score":           true,
		"eval_duration_s": true,
		"error":           true,
	}
	for key := range doc {
		if !allowed[key] {
			return false
		}
	}
	return true
}

func hasStructuredBreakdown(doc map[string]any) bool {
	for _, key := range []string{"components", "checks", "reasons", "breakdown"} {
		switch value := doc[key].(type) {
		case []any:
			if len(value) > 0 {
				return true
			}
		case map[string]any:
			if len(value) > 0 {
				return true
			}
		case string:
			if strings.TrimSpace(value) != "" {
				return true
			}
		}
	}
	return false
}

func unavailableEvidenceReason(level, declared EvaluatorEvidenceLevel) string {
	switch level {
	case EvaluatorEvidenceStructuredBreakdown:
		return "not supplied; the official scalar remains valid without per-check reasons"
	case EvaluatorEvidenceRawGraderPayload:
		return "not supplied; stock ALE eval_result.json is a normalized scalar, not a raw grader payload, and the official scalar remains valid"
	default:
		return fmt.Sprintf("not supplied above declared level %s", declared)
	}
}

func evidenceError(reason, detail string) *EvaluatorEvidenceError {
	return &EvaluatorEvidenceError{Reason: reason, Detail: detail}
}
