package codebookmeta

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
)

const SchemaV1 = "fak.codebookmeta/v1"

type Outcome string

const (
	OutcomeSupported Outcome = "supported"
	OutcomeRefused   Outcome = "unsupported"
	OutcomeDelegate  Outcome = "delegate"
)

type ReasonCode string

const (
	ReasonAccepted                  ReasonCode = "CODEBOOK_ACCEPTED"
	ReasonInvalidJSON               ReasonCode = "INVALID_JSON"
	ReasonUnknownSchema             ReasonCode = "UNKNOWN_SCHEMA"
	ReasonMissingField              ReasonCode = "MISSING_FIELD"
	ReasonUnknownCodebookKind       ReasonCode = "UNKNOWN_CODEBOOK_KIND"
	ReasonMissingCodebookPayload    ReasonCode = "MISSING_CODEBOOK_PAYLOAD"
	ReasonInvalidCodebookPayload    ReasonCode = "INVALID_CODEBOOK_PAYLOAD"
	ReasonInvalidArtifactDigest     ReasonCode = "INVALID_ARTIFACT_DIGEST"
	ReasonInvalidRecipeDigest       ReasonCode = "INVALID_RECIPE_DIGEST"
	ReasonCodebookDigestMismatch    ReasonCode = "CODEBOOK_DIGEST_MISMATCH"
	ReasonPackingUnavailable        ReasonCode = "INDEX_PACKING_UNAVAILABLE"
	ReasonDecodeUnavailable         ReasonCode = "DECODE_REQUIREMENT_UNAVAILABLE"
	ReasonRuntimeDelegationRequired ReasonCode = "RUNTIME_DELEGATION_REQUIRED"
)

type CodebookKind string

const (
	KindIntegerGrid CodebookKind = "integer_grid"
	KindNF4         CodebookKind = "nf4"
	KindLearned     CodebookKind = "learned"
	KindParametric  CodebookKind = "parametric"
)

type EvidenceKind string

const (
	EvidenceObserved EvidenceKind = "observed"
	EvidenceModeled  EvidenceKind = "modeled"
)

type Digest struct {
	Algorithm string `json:"algorithm"`
	Value     string `json:"value"`
}

type Provenance struct {
	ArtifactID     string `json:"artifact_id"`
	ArtifactDigest Digest `json:"artifact_digest"`
	RecipeID       string `json:"recipe_id"`
	RecipeVersion  string `json:"recipe_version"`
	RecipeDigest   Digest `json:"recipe_digest"`
	RuntimeID      string `json:"runtime_id"`
	RuntimeVersion string `json:"runtime_version"`
	ModelID        string `json:"model_id"`
	ModelRevision  string `json:"model_revision"`
}

type Packing struct {
	ID             string `json:"id"`
	Version        string `json:"version"`
	BitsPerIndex   int    `json:"bits_per_index"`
	ByteOrder      string `json:"byte_order"`
	IndicesPerWord int    `json:"indices_per_word,omitempty"`
}

type DecodeRequirements struct {
	CodebookPlacement string   `json:"codebook_placement"`
	ScaleType         string   `json:"scale_type"`
	GroupSize         int      `json:"group_size"`
	RequiredFeatures  []string `json:"required_features,omitempty"`
	RoutedRuntime     string   `json:"routed_runtime,omitempty"`
}

type Codebook struct {
	Kind       CodebookKind       `json:"kind"`
	ID         string             `json:"id"`
	Version    string             `json:"version"`
	Entries    []float64          `json:"entries,omitempty"`
	Parameters map[string]float64 `json:"parameters,omitempty"`
	Payload    []byte             `json:"payload,omitempty"`
	Digest     Digest             `json:"digest"`
}

type Evaluation struct {
	Kind     EvidenceKind `json:"kind"`
	Claim    string       `json:"claim"`
	Metric   string       `json:"metric"`
	Value    *float64     `json:"value,omitempty"`
	Unit     string       `json:"unit,omitempty"`
	Witness  string       `json:"witness"`
	Hardware string       `json:"hardware,omitempty"`
	Dataset  string       `json:"dataset,omitempty"`
}

type Descriptor struct {
	Schema             string             `json:"schema"`
	Provenance         Provenance         `json:"provenance"`
	Codebook           Codebook           `json:"codebook"`
	Packing            Packing            `json:"packing"`
	DecodeRequirements DecodeRequirements `json:"decode_requirements"`
	Evaluations        []Evaluation       `json:"evaluations"`
}

type Capability struct {
	PackingIDs     []string
	DecodeFeatures []string
	RoutedRuntimes []string
}

// DefaultCapability returns the baseline recognized packing schemes, decode features,
// and routed runtime environments for codebook metadata adjudication.
func DefaultCapability() Capability {
	return Capability{
		PackingIDs:     []string{"nibble-lsb@1", "bitpack-lsb@1"},
		DecodeFeatures: []string{"per-block-scale", "explicit-codebook"},
		RoutedRuntimes: []string{"lab-runtime@2.4.1"},
	}
}

type Result struct {
	Outcome    Outcome     `json:"outcome"`
	Reason     ReasonCode  `json:"reason"`
	Detail     string      `json:"detail,omitempty"`
	Descriptor *Descriptor `json:"descriptor,omitempty"`
}

func Parse(raw []byte, capability Capability) (Result, error) {
	var d Descriptor
	if err := json.Unmarshal(raw, &d); err != nil {
		return Result{Outcome: OutcomeRefused, Reason: ReasonInvalidJSON, Detail: err.Error()}, err
	}
	return Adjudicate(d, capability), nil
}

func Adjudicate(d Descriptor, capability Capability) Result {
	result := func(outcome Outcome, reason ReasonCode, detail string) Result {
		copy := d
		return Result{Outcome: outcome, Reason: reason, Detail: detail, Descriptor: &copy}
	}
	if d.Schema != SchemaV1 {
		return result(OutcomeRefused, ReasonUnknownSchema, d.Schema)
	}
	if missing := missingField(d); missing != "" {
		return result(OutcomeRefused, ReasonMissingField, missing)
	}
	if !knownKind(d.Codebook.Kind) {
		return result(OutcomeRefused, ReasonUnknownCodebookKind, string(d.Codebook.Kind))
	}
	if payloadRequired(d.Codebook.Kind) && len(d.Codebook.Entries) == 0 && len(d.Codebook.Parameters) == 0 && len(d.Codebook.Payload) == 0 {
		return result(OutcomeRefused, ReasonMissingCodebookPayload, string(d.Codebook.Kind))
	}
	if err := validateShape(d.Codebook, d.Packing); err != nil {
		return result(OutcomeRefused, ReasonInvalidCodebookPayload, err.Error())
	}
	if !validSHA256(d.Provenance.ArtifactDigest) {
		return result(OutcomeRefused, ReasonInvalidArtifactDigest, "artifact_digest")
	}
	if !validSHA256(d.Provenance.RecipeDigest) {
		return result(OutcomeRefused, ReasonInvalidRecipeDigest, "recipe_digest")
	}
	if !validSHA256(d.Codebook.Digest) || !codebookDigestMatches(d.Codebook) {
		return result(OutcomeRefused, ReasonCodebookDigestMismatch, "codebook.digest")
	}
	if !contains(capability.PackingIDs, d.Packing.ID+"@"+d.Packing.Version) {
		return result(OutcomeRefused, ReasonPackingUnavailable, d.Packing.ID+"@"+d.Packing.Version)
	}
	for _, feature := range d.DecodeRequirements.RequiredFeatures {
		if !contains(capability.DecodeFeatures, feature) {
			if d.DecodeRequirements.RoutedRuntime != "" && contains(capability.RoutedRuntimes, d.DecodeRequirements.RoutedRuntime) {
				return result(OutcomeDelegate, ReasonRuntimeDelegationRequired, d.DecodeRequirements.RoutedRuntime+":"+feature)
			}
			return result(OutcomeRefused, ReasonDecodeUnavailable, feature)
		}
	}
	return result(OutcomeSupported, ReasonAccepted, "")
}

func MarshalCanonical(d Descriptor) ([]byte, error) { return json.MarshalIndent(d, "", "  ") }

func CodebookDigest(c Codebook) Digest {
	payload, _ := json.Marshal(struct {
		Kind       CodebookKind       `json:"kind"`
		ID         string             `json:"id"`
		Version    string             `json:"version"`
		Entries    []float64          `json:"entries,omitempty"`
		Parameters map[string]float64 `json:"parameters,omitempty"`
		Payload    []byte             `json:"payload,omitempty"`
	}{c.Kind, c.ID, c.Version, c.Entries, c.Parameters, c.Payload})
	sum := sha256.Sum256(payload)
	return Digest{Algorithm: "sha256", Value: hex.EncodeToString(sum[:])}
}

func missingField(d Descriptor) string {
	fields := []struct{ name, value string }{
		{"provenance.artifact_id", d.Provenance.ArtifactID}, {"provenance.artifact_digest", d.Provenance.ArtifactDigest.Value},
		{"provenance.recipe_id", d.Provenance.RecipeID}, {"provenance.recipe_version", d.Provenance.RecipeVersion},
		{"provenance.recipe_digest", d.Provenance.RecipeDigest.Value}, {"provenance.runtime_id", d.Provenance.RuntimeID},
		{"provenance.runtime_version", d.Provenance.RuntimeVersion}, {"provenance.model_id", d.Provenance.ModelID},
		{"provenance.model_revision", d.Provenance.ModelRevision}, {"codebook.id", d.Codebook.ID},
		{"codebook.version", d.Codebook.Version}, {"codebook.digest", d.Codebook.Digest.Value},
		{"packing.id", d.Packing.ID}, {"packing.version", d.Packing.Version}, {"packing.byte_order", d.Packing.ByteOrder},
		{"decode_requirements.codebook_placement", d.DecodeRequirements.CodebookPlacement}, {"decode_requirements.scale_type", d.DecodeRequirements.ScaleType},
	}
	for _, field := range fields {
		if strings.TrimSpace(field.value) == "" {
			return field.name
		}
	}
	if d.Packing.BitsPerIndex <= 0 {
		return "packing.bits_per_index"
	}
	if d.DecodeRequirements.GroupSize <= 0 {
		return "decode_requirements.group_size"
	}
	for i, evaluation := range d.Evaluations {
		if evaluation.Kind != EvidenceObserved && evaluation.Kind != EvidenceModeled {
			return fmt.Sprintf("evaluations[%d].kind", i)
		}
		if evaluation.Claim == "" || evaluation.Metric == "" || evaluation.Witness == "" {
			return fmt.Sprintf("evaluations[%d]", i)
		}
	}
	return ""
}

func knownKind(kind CodebookKind) bool {
	return kind == KindIntegerGrid || kind == KindNF4 || kind == KindLearned || kind == KindParametric
}
func payloadRequired(kind CodebookKind) bool { return kind == KindLearned || kind == KindParametric }
func validateShape(c Codebook, p Packing) error {
	if p.BitsPerIndex > 30 {
		return fmt.Errorf("bits_per_index=%d", p.BitsPerIndex)
	}
	maxEntries := 1 << p.BitsPerIndex
	if len(c.Entries) > maxEntries {
		return fmt.Errorf("entries=%d exceeds index capacity=%d", len(c.Entries), maxEntries)
	}
	if c.Kind == KindIntegerGrid && len(c.Entries) == 0 {
		return fmt.Errorf("integer grid entries are required")
	}
	if c.Kind == KindNF4 && len(c.Entries) != 16 {
		return fmt.Errorf("nf4 requires 16 entries, got %d", len(c.Entries))
	}
	return nil
}
func validSHA256(d Digest) bool {
	if d.Algorithm != "sha256" || len(d.Value) != 64 {
		return false
	}
	_, err := hex.DecodeString(d.Value)
	return err == nil
}
func codebookDigestMatches(c Codebook) bool { return c.Digest == CodebookDigest(c) }
func contains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
