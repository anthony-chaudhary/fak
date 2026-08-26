package agenticbench

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	FailureAnalysisCardSchema = "fak.agenticbench.failure-analysis-card.v1"
	FailureAnalysisAdvisory   = "ADVISORY"

	maxFailureAnalysisArtifactBytes = 8 << 20
)

type FailureAnalysisClaimKind string

const (
	FailureAnalysisObservation FailureAnalysisClaimKind = "observation"
	FailureAnalysisInference   FailureAnalysisClaimKind = "inference"
)

type FailureAnalysisEvidence struct {
	ArtifactPath string `json:"artifact_path"`
	JSONPointer  string `json:"json_pointer,omitempty"`
	EventOrdinal *int   `json:"event_ordinal,omitempty"`
	ArtifactHash string `json:"artifact_hash"`
}

type FailureAnalysisClaim struct {
	Text        string                    `json:"text"`
	Kind        FailureAnalysisClaimKind  `json:"kind"`
	Uncertainty string                    `json:"uncertainty,omitempty"`
	Evidence    []FailureAnalysisEvidence `json:"evidence"`
}

type FailureAnalysisGenerator struct {
	Recipe          string `json:"recipe"`
	Model           string `json:"model"`
	PromptVersion   string `json:"prompt_version"`
	SoftwareVersion string `json:"software_version"`
	GeneratedAt     string `json:"generated_at"`
}

type FailureAnalysisArtifact struct {
	ArtifactPath string `json:"artifact_path"`
	SHA256       string `json:"sha256"`
}

type FailureAnalysisVerdict struct {
	Outcome              string               `json:"outcome"`
	Summary              FailureAnalysisClaim `json:"summary"`
	MostImportantProblem FailureAnalysisClaim `json:"most_important_problem"`
}

type FailureAnalysisTask struct {
	Summary     FailureAnalysisClaim   `json:"summary"`
	Constraints []FailureAnalysisClaim `json:"constraints"`
}

type FailureAnalysisFailures struct {
	Observations []FailureAnalysisClaim `json:"observations"`
	Inferences   []FailureAnalysisClaim `json:"inferences"`
}

type FailureAnalysisScoring struct {
	FinalScore  FailureAnalysisClaim   `json:"final_score"`
	Breakdown   []FailureAnalysisClaim `json:"breakdown"`
	Confidence  float64                `json:"confidence"`
	Uncertainty string                 `json:"uncertainty"`
}

type FailureAnalysisClassification struct {
	Class       string               `json:"class"`
	Subcategory string               `json:"subcategory"`
	Confidence  float64              `json:"confidence"`
	Uncertainty string               `json:"uncertainty"`
	Rationale   FailureAnalysisClaim `json:"rationale"`
}

// FailureAnalysisCard is an advisory sidecar. It deliberately has no hook into
// official scoring, retry, routing, or policy decisions.
type FailureAnalysisCard struct {
	Schema         string                        `json:"schema"`
	Label          string                        `json:"label"`
	RunID          string                        `json:"run_id"`
	Generator      FailureAnalysisGenerator      `json:"generator"`
	Artifacts      []FailureAnalysisArtifact     `json:"artifacts"`
	Verdict        FailureAnalysisVerdict        `json:"verdict"`
	Task           FailureAnalysisTask           `json:"task"`
	Strengths      []FailureAnalysisClaim        `json:"strengths"`
	Failures       FailureAnalysisFailures       `json:"failures"`
	Scoring        FailureAnalysisScoring        `json:"scoring"`
	Classification FailureAnalysisClassification `json:"classification"`
}

var failureAnalysisCategories = map[string]map[string]struct{}{
	"understanding": {
		"domain_knowledge_gap":      {},
		"hallucination_fabrication": {},
	},
	"approach": {
		"wrong_strategy":       {},
		"incomplete_abandoned": {},
	},
	"execution": {
		"implementation_bug":  {},
		"output_format_error": {},
	},
	"infrastructure": {
		"gui_browser_failure": {},
		"timeout_resources":   {},
	},
}

// DecodeFailureAnalysisCard rejects schema drift instead of silently ignoring
// fields emitted by a newer or malformed generator.
func DecodeFailureAnalysisCard(data []byte) (FailureAnalysisCard, error) {
	var card FailureAnalysisCard
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&card); err != nil {
		return FailureAnalysisCard{}, fmt.Errorf("decode failure analysis card: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			err = fmt.Errorf("multiple JSON values")
		}
		return FailureAnalysisCard{}, fmt.Errorf("decode failure analysis card: %w", err)
	}
	return card, nil
}

// ValidateFailureAnalysisCard resolves every cited field or event inside the
// declared run root before the advisory card can be promoted.
func ValidateFailureAnalysisCard(runRoot string, card *FailureAnalysisCard) error {
	if card == nil {
		return fmt.Errorf("failure analysis card is nil")
	}
	if card.Schema != FailureAnalysisCardSchema {
		return fmt.Errorf("failure analysis schema %q, want %q", card.Schema, FailureAnalysisCardSchema)
	}
	if card.Label != FailureAnalysisAdvisory {
		return fmt.Errorf("failure analysis label %q, want %q", card.Label, FailureAnalysisAdvisory)
	}
	if strings.TrimSpace(card.RunID) == "" {
		return fmt.Errorf("failure analysis card requires run_id")
	}
	if err := validateFailureAnalysisGenerator(card.Generator); err != nil {
		return err
	}

	manifest, err := validateFailureAnalysisArtifacts(runRoot, card.Artifacts)
	if err != nil {
		return err
	}
	validateClaim := func(name string, claim FailureAnalysisClaim, expected FailureAnalysisClaimKind) error {
		return validateFailureAnalysisClaim(runRoot, manifest, name, claim, expected)
	}
	validateClaims := func(name string, claims []FailureAnalysisClaim, expected FailureAnalysisClaimKind) error {
		for i, claim := range claims {
			if err := validateClaim(fmt.Sprintf("%s %d", name, i+1), claim, expected); err != nil {
				return err
			}
		}
		return nil
	}

	switch card.Verdict.Outcome {
	case "success", "partial_success", "failure":
	default:
		return fmt.Errorf("unknown analysis outcome %q", card.Verdict.Outcome)
	}
	if err := validateClaim("verdict summary", card.Verdict.Summary, FailureAnalysisObservation); err != nil {
		return err
	}
	if err := validateClaim("most important problem", card.Verdict.MostImportantProblem, FailureAnalysisObservation); err != nil {
		return err
	}
	if err := validateClaim("task summary", card.Task.Summary, FailureAnalysisObservation); err != nil {
		return err
	}
	if len(card.Task.Constraints) == 0 {
		return fmt.Errorf("task section requires at least one evidence-cited constraint")
	}
	if err := validateClaims("task constraint", card.Task.Constraints, FailureAnalysisObservation); err != nil {
		return err
	}
	if len(card.Strengths) == 0 {
		return fmt.Errorf("strengths section requires at least one evidence-cited claim")
	}
	if err := validateClaims("strength", card.Strengths, FailureAnalysisObservation); err != nil {
		return err
	}
	if len(card.Failures.Observations) == 0 {
		return fmt.Errorf("failures section requires at least one observation")
	}
	if err := validateClaims("failure observation", card.Failures.Observations, FailureAnalysisObservation); err != nil {
		return err
	}
	if err := validateClaims("failure inference", card.Failures.Inferences, FailureAnalysisInference); err != nil {
		return err
	}
	if err := validateClaim("final score", card.Scoring.FinalScore, FailureAnalysisObservation); err != nil {
		return err
	}
	if len(card.Scoring.Breakdown) == 0 {
		return fmt.Errorf("scoring section requires at least one evidence-cited breakdown claim")
	}
	if err := validateClaims("score breakdown", card.Scoring.Breakdown, FailureAnalysisObservation); err != nil {
		return err
	}
	if err := validateConfidence("scoring", card.Scoring.Confidence, card.Scoring.Uncertainty); err != nil {
		return err
	}
	if err := validateFailureAnalysisClassification(card.Classification); err != nil {
		return err
	}
	if err := validateClaim("classification rationale", card.Classification.Rationale, FailureAnalysisInference); err != nil {
		return err
	}
	return nil
}

func validateFailureAnalysisGenerator(generator FailureAnalysisGenerator) error {
	fields := []struct {
		name  string
		value string
	}{
		{"recipe", generator.Recipe},
		{"model", generator.Model},
		{"prompt_version", generator.PromptVersion},
		{"software_version", generator.SoftwareVersion},
		{"generated_at", generator.GeneratedAt},
	}
	for _, field := range fields {
		if strings.TrimSpace(field.value) == "" {
			return fmt.Errorf("generator provenance requires %s", field.name)
		}
	}
	if _, err := time.Parse(time.RFC3339, generator.GeneratedAt); err != nil {
		return fmt.Errorf("generator generated_at must be RFC3339: %w", err)
	}
	return nil
}

func validateFailureAnalysisArtifacts(runRoot string, artifacts []FailureAnalysisArtifact) (map[string]string, error) {
	if len(artifacts) == 0 {
		return nil, fmt.Errorf("failure analysis card requires source artifacts")
	}
	manifest := make(map[string]string, len(artifacts))
	for _, artifact := range artifacts {
		if _, exists := manifest[artifact.ArtifactPath]; exists {
			return nil, fmt.Errorf("duplicate source artifact %q", artifact.ArtifactPath)
		}
		if err := validateSHA256(artifact.SHA256); err != nil {
			return nil, fmt.Errorf("source artifact %q hash: %w", artifact.ArtifactPath, err)
		}
		_, data, err := readFailureAnalysisArtifact(runRoot, artifact.ArtifactPath)
		if err != nil {
			return nil, err
		}
		actual := fmt.Sprintf("sha256:%x", sha256.Sum256(data))
		if actual != artifact.SHA256 {
			return nil, fmt.Errorf("source artifact %q hash mismatch: got %s, want %s", artifact.ArtifactPath, actual, artifact.SHA256)
		}
		manifest[artifact.ArtifactPath] = artifact.SHA256
	}
	return manifest, nil
}

func validateFailureAnalysisClaim(runRoot string, manifest map[string]string, name string, claim FailureAnalysisClaim, expected FailureAnalysisClaimKind) error {
	if strings.TrimSpace(claim.Text) == "" {
		return fmt.Errorf("%s requires text", name)
	}
	if claim.Kind != FailureAnalysisObservation && claim.Kind != FailureAnalysisInference {
		return fmt.Errorf("%s kind %q is not observation or inference", name, claim.Kind)
	}
	if expected != "" && claim.Kind != expected {
		return fmt.Errorf("%s kind %q, want %q", name, claim.Kind, expected)
	}
	if claim.Kind == FailureAnalysisInference && strings.TrimSpace(claim.Uncertainty) == "" {
		return fmt.Errorf("%s inference requires uncertainty", name)
	}
	if len(claim.Evidence) == 0 {
		return fmt.Errorf("%s requires evidence", name)
	}
	for i, evidence := range claim.Evidence {
		if _, _, err := resolveFailureAnalysisEvidence(runRoot, manifest, evidence); err != nil {
			return fmt.Errorf("%s evidence %d: %w", name, i+1, err)
		}
	}
	return nil
}

func validateFailureAnalysisClassification(classification FailureAnalysisClassification) error {
	subcategories, ok := failureAnalysisCategories[classification.Class]
	if !ok {
		return fmt.Errorf("unknown failure class %q", classification.Class)
	}
	if _, ok := subcategories[classification.Subcategory]; !ok {
		return fmt.Errorf("unknown %s failure subcategory %q", classification.Class, classification.Subcategory)
	}
	return validateConfidence("classification", classification.Confidence, classification.Uncertainty)
}

func validateConfidence(name string, confidence float64, uncertainty string) error {
	if math.IsNaN(confidence) || math.IsInf(confidence, 0) || confidence < 0 || confidence > 1 {
		return fmt.Errorf("%s confidence %g must be within [0,1]", name, confidence)
	}
	if strings.TrimSpace(uncertainty) == "" {
		return fmt.Errorf("%s requires uncertainty", name)
	}
	return nil
}

// ResolveFailureAnalysisEvidence returns the cited JSON value or JSONL event
// after path, size, and digest checks succeed.
func ResolveFailureAnalysisEvidence(runRoot string, evidence FailureAnalysisEvidence) (json.RawMessage, error) {
	value, _, err := resolveFailureAnalysisEvidence(runRoot, nil, evidence)
	return value, err
}

func resolveFailureAnalysisEvidence(runRoot string, manifest map[string]string, evidence FailureAnalysisEvidence) (json.RawMessage, string, error) {
	if err := validateSHA256(evidence.ArtifactHash); err != nil {
		return nil, "", fmt.Errorf("artifact hash: %w", err)
	}
	_, data, err := readFailureAnalysisArtifact(runRoot, evidence.ArtifactPath)
	if err != nil {
		return nil, "", err
	}
	if manifest != nil {
		declared, ok := manifest[evidence.ArtifactPath]
		if !ok {
			return nil, "", fmt.Errorf("artifact %q is not declared in the source manifest", evidence.ArtifactPath)
		}
		if evidence.ArtifactHash != declared {
			return nil, "", fmt.Errorf("artifact %q evidence hash %s does not match manifest %s", evidence.ArtifactPath, evidence.ArtifactHash, declared)
		}
	}
	actual := fmt.Sprintf("sha256:%x", sha256.Sum256(data))
	if actual != evidence.ArtifactHash {
		return nil, "", fmt.Errorf("artifact %q hash mismatch: got %s, want %s", evidence.ArtifactPath, actual, evidence.ArtifactHash)
	}

	hasPointer := evidence.JSONPointer != ""
	hasEvent := evidence.EventOrdinal != nil
	if hasPointer == hasEvent {
		return nil, "", fmt.Errorf("evidence must name exactly one JSON pointer or event ordinal")
	}
	if hasPointer {
		value, err := resolveJSONPointer(data, evidence.JSONPointer)
		if err != nil {
			return nil, "", fmt.Errorf("artifact %q pointer %q: %w", evidence.ArtifactPath, evidence.JSONPointer, err)
		}
		return value, evidence.ArtifactPath + "#" + evidence.JSONPointer, nil
	}
	value, err := resolveJSONLEvent(data, *evidence.EventOrdinal)
	if err != nil {
		return nil, "", fmt.Errorf("artifact %q event ordinal %d: %w", evidence.ArtifactPath, *evidence.EventOrdinal, err)
	}
	return value, fmt.Sprintf("%s@event:%d", evidence.ArtifactPath, *evidence.EventOrdinal), nil
}

func readFailureAnalysisArtifact(runRoot, artifactPath string) (string, []byte, error) {
	clean, err := cleanFailureAnalysisPath(artifactPath)
	if err != nil {
		return "", nil, err
	}
	root, err := filepath.Abs(runRoot)
	if err != nil {
		return "", nil, fmt.Errorf("resolve run root: %w", err)
	}
	root, err = filepath.EvalSymlinks(root)
	if err != nil {
		return "", nil, fmt.Errorf("resolve run root: %w", err)
	}
	candidate := filepath.Join(root, filepath.FromSlash(clean))
	if !pathInsideRoot(root, candidate) {
		return "", nil, fmt.Errorf("artifact path %q escapes run root", artifactPath)
	}
	resolved, err := filepath.EvalSymlinks(candidate)
	if err != nil {
		return "", nil, fmt.Errorf("resolve artifact %q: %w", artifactPath, err)
	}
	if !pathInsideRoot(root, resolved) {
		return "", nil, fmt.Errorf("artifact path %q escapes run root through symlink", artifactPath)
	}
	info, err := os.Stat(resolved)
	if err != nil {
		return "", nil, fmt.Errorf("stat artifact %q: %w", artifactPath, err)
	}
	if !info.Mode().IsRegular() {
		return "", nil, fmt.Errorf("artifact %q is not a regular file", artifactPath)
	}
	if info.Size() > maxFailureAnalysisArtifactBytes {
		return "", nil, fmt.Errorf("artifact %q exceeds bounded %d-byte analysis limit", artifactPath, maxFailureAnalysisArtifactBytes)
	}
	data, err := os.ReadFile(resolved)
	if err != nil {
		return "", nil, fmt.Errorf("read artifact %q: %w", artifactPath, err)
	}
	return clean, data, nil
}

func cleanFailureAnalysisPath(artifactPath string) (string, error) {
	if strings.TrimSpace(artifactPath) == "" || strings.Contains(artifactPath, "\\") {
		return "", fmt.Errorf("artifact path %q must be a relative slash path", artifactPath)
	}
	clean := path.Clean(artifactPath)
	if path.IsAbs(clean) || clean == "." || clean == ".." || strings.HasPrefix(clean, "../") {
		return "", fmt.Errorf("artifact path %q escapes run root", artifactPath)
	}
	if clean != artifactPath {
		return "", fmt.Errorf("artifact path %q is not clean", artifactPath)
	}
	if strings.EqualFold(path.Base(clean), "transcript.jsonl") {
		return "", fmt.Errorf("artifact %q is outside the bounded analysis surface", artifactPath)
	}
	return clean, nil
}

func pathInsideRoot(root, candidate string) bool {
	rel, err := filepath.Rel(root, candidate)
	return err == nil && rel != ".." && !filepath.IsAbs(rel) && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

func validateSHA256(value string) error {
	const prefix = "sha256:"
	if !strings.HasPrefix(value, prefix) || len(value) != len(prefix)+sha256.Size*2 {
		return fmt.Errorf("must be sha256:<64 lowercase hex>")
	}
	hexPart := strings.TrimPrefix(value, prefix)
	if hexPart != strings.ToLower(hexPart) {
		return fmt.Errorf("must be sha256:<64 lowercase hex>")
	}
	if _, err := hex.DecodeString(hexPart); err != nil {
		return fmt.Errorf("must be sha256:<64 lowercase hex>: %w", err)
	}
	return nil
}

func resolveJSONPointer(data []byte, pointer string) (json.RawMessage, error) {
	if !strings.HasPrefix(pointer, "/") {
		return nil, fmt.Errorf("must start with /")
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, fmt.Errorf("decode JSON: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return nil, fmt.Errorf("artifact contains trailing JSON")
	}
	for _, encoded := range strings.Split(strings.TrimPrefix(pointer, "/"), "/") {
		token, err := decodeJSONPointerToken(encoded)
		if err != nil {
			return nil, err
		}
		switch node := value.(type) {
		case map[string]any:
			var ok bool
			value, ok = node[token]
			if !ok {
				return nil, fmt.Errorf("object has no field %q", token)
			}
		case []any:
			index, err := strconv.Atoi(token)
			if err != nil || index < 0 || index >= len(node) || (len(token) > 1 && token[0] == '0') {
				return nil, fmt.Errorf("array index %q is invalid", token)
			}
			value = node[index]
		default:
			return nil, fmt.Errorf("cannot descend through %T", value)
		}
	}
	resolved, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("encode resolved value: %w", err)
	}
	return resolved, nil
}

func decodeJSONPointerToken(token string) (string, error) {
	var b strings.Builder
	for i := 0; i < len(token); i++ {
		if token[i] != '~' {
			b.WriteByte(token[i])
			continue
		}
		if i+1 >= len(token) {
			return "", fmt.Errorf("invalid JSON pointer escape")
		}
		i++
		switch token[i] {
		case '0':
			b.WriteByte('~')
		case '1':
			b.WriteByte('/')
		default:
			return "", fmt.Errorf("invalid JSON pointer escape ~%c", token[i])
		}
	}
	return b.String(), nil
}

func resolveJSONLEvent(data []byte, ordinal int) (json.RawMessage, error) {
	if ordinal < 1 {
		return nil, fmt.Errorf("must be one-based")
	}
	scanner := bufio.NewScanner(bytes.NewReader(data))
	scanner.Buffer(make([]byte, 64<<10), 1<<20)
	seen := 0
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		seen++
		if seen != ordinal {
			continue
		}
		if !json.Valid(line) {
			return nil, fmt.Errorf("event is not valid JSON")
		}
		return append(json.RawMessage(nil), line...), nil
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan events: %w", err)
	}
	return nil, fmt.Errorf("out of range; artifact has %d events", seen)
}

// RenderFailureAnalysisJSON validates the card before emitting stable JSON.
func RenderFailureAnalysisJSON(runRoot string, card *FailureAnalysisCard) ([]byte, error) {
	if err := ValidateFailureAnalysisCard(runRoot, card); err != nil {
		return nil, err
	}
	data, err := json.MarshalIndent(card, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("render failure analysis JSON: %w", err)
	}
	return append(data, '\n'), nil
}

// RenderFailureAnalysisMarkdown validates the card before emitting the five
// mandatory advisory sections.
func RenderFailureAnalysisMarkdown(runRoot string, card *FailureAnalysisCard) (string, error) {
	if err := ValidateFailureAnalysisCard(runRoot, card); err != nil {
		return "", err
	}
	var b strings.Builder
	fmt.Fprintln(&b, "# ADVISORY Failure Analysis Card")
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "- Schema: `%s`\n", card.Schema)
	fmt.Fprintf(&b, "- Run: `%s`\n", analysisMarkdownText(card.RunID))
	fmt.Fprintf(&b, "- Generator: `%s / %s / %s`\n", analysisMarkdownText(card.Generator.Model), analysisMarkdownText(card.Generator.PromptVersion), analysisMarkdownText(card.Generator.SoftwareVersion))
	fmt.Fprintf(&b, "- Recipe: `%s` at `%s`\n", analysisMarkdownText(card.Generator.Recipe), analysisMarkdownText(card.Generator.GeneratedAt))
	fmt.Fprintln(&b)

	fmt.Fprintln(&b, "## 1. Conclusion")
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "- Outcome: `%s`\n", card.Verdict.Outcome)
	fmt.Fprintf(&b, "- Classification: `%s / %s` (confidence %.2f)\n", card.Classification.Class, card.Classification.Subcategory, card.Classification.Confidence)
	fmt.Fprintf(&b, "- Classification uncertainty: %s\n", analysisMarkdownText(card.Classification.Uncertainty))
	writeFailureAnalysisClaim(&b, "Verdict", card.Verdict.Summary)
	writeFailureAnalysisClaim(&b, "Most important problem", card.Verdict.MostImportantProblem)
	writeFailureAnalysisClaim(&b, "Classification rationale", card.Classification.Rationale)

	fmt.Fprintln(&b, "\n## 2. Task and constraints")
	fmt.Fprintln(&b)
	writeFailureAnalysisClaim(&b, "Task", card.Task.Summary)
	writeFailureAnalysisClaims(&b, "Constraint", card.Task.Constraints)

	fmt.Fprintln(&b, "\n## 3. What the agent did right")
	fmt.Fprintln(&b)
	writeFailureAnalysisClaims(&b, "Strength", card.Strengths)

	fmt.Fprintln(&b, "\n## 4. What the agent did wrong")
	fmt.Fprintln(&b)
	writeFailureAnalysisClaims(&b, "Observed", card.Failures.Observations)
	writeFailureAnalysisClaims(&b, "Inferred", card.Failures.Inferences)

	fmt.Fprintln(&b, "\n## 5. Scoring")
	fmt.Fprintln(&b)
	writeFailureAnalysisClaim(&b, "Final score", card.Scoring.FinalScore)
	writeFailureAnalysisClaims(&b, "Breakdown", card.Scoring.Breakdown)
	fmt.Fprintf(&b, "- Confidence: `%.2f`\n", card.Scoring.Confidence)
	fmt.Fprintf(&b, "- Uncertainty: %s\n", analysisMarkdownText(card.Scoring.Uncertainty))
	return b.String(), nil
}

func writeFailureAnalysisClaims(b *strings.Builder, label string, claims []FailureAnalysisClaim) {
	for _, claim := range claims {
		writeFailureAnalysisClaim(b, label, claim)
	}
}

func writeFailureAnalysisClaim(b *strings.Builder, label string, claim FailureAnalysisClaim) {
	fmt.Fprintf(b, "- %s (%s): %s", label, claim.Kind, analysisMarkdownText(claim.Text))
	if claim.Uncertainty != "" {
		fmt.Fprintf(b, " [uncertainty: %s]", analysisMarkdownText(claim.Uncertainty))
	}
	fmt.Fprintf(b, " — evidence: %s\n", formatFailureAnalysisEvidence(claim.Evidence))
}

func formatFailureAnalysisEvidence(evidence []FailureAnalysisEvidence) string {
	refs := make([]string, 0, len(evidence))
	for _, ref := range evidence {
		locator := ref.ArtifactPath + "#" + ref.JSONPointer
		if ref.EventOrdinal != nil {
			locator = fmt.Sprintf("%s@event:%d", ref.ArtifactPath, *ref.EventOrdinal)
		}
		refs = append(refs, fmt.Sprintf("`%s` (`%s`)", analysisMarkdownText(locator), ref.ArtifactHash))
	}
	return strings.Join(refs, ", ")
}

func analysisMarkdownText(value string) string {
	return strings.NewReplacer("\r", " ", "\n", " ", "`", "\\`").Replace(strings.TrimSpace(value))
}
