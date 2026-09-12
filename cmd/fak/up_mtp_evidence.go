package main

import (
	"bytes"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"strings"
	"time"

	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

const turnkeyMTPEvidenceCatalogSchema = "fak.up.mtp-evidence-catalog/1"

//go:embed up_mtp_evidence_catalog.json
var turnkeyMTPEvidenceCatalogJSON []byte

type turnkeyMTPWorkloadEnvelope struct {
	Greedy        bool `json:"greedy"`
	ContextTokens int  `json:"context_tokens"`
	DraftDepth    int  `json:"draft_depth"`
}

// turnkeyMTPQualificationKey is the exact compatibility identity shared by a
// reviewed record and a runtime request. Measured headroom is kept separate.
type turnkeyMTPQualificationKey struct {
	ModelArtifactSHA256 string                         `json:"model_artifact_sha256"`
	ModelFamily         string                         `json:"model_family"`
	MTPFormat           fakmodel.Qwen38MTPTensorFormat `json:"mtp_format"`
	DeviceCompatibility string                         `json:"device_compatibility"`
	MetalCompatibility  string                         `json:"metal_compatibility"`
	EngineCompatibility string                         `json:"engine_compatibility"`
	NativeConfigSHA256  string                         `json:"native_config_sha256"`
	Workload            turnkeyMTPWorkloadEnvelope     `json:"workload"`
}

type turnkeyMTPQualificationContext struct {
	Qualification          turnkeyMTPQualificationKey `json:"qualification"`
	AvailableHeadroomBytes uint64                     `json:"available_headroom_bytes"`
}

type turnkeyMTPQualificationRefusal string

const (
	turnkeyMTPQualificationAvailable turnkeyMTPQualificationRefusal = ""
	turnkeyMTPNoEligibleContext      turnkeyMTPQualificationRefusal = "NO_ELIGIBLE_QUALIFICATION_CONTEXT"
	turnkeyMTPMalformedCatalog       turnkeyMTPQualificationRefusal = "MALFORMED_TRUSTED_CATALOG"
	turnkeyMTPAmbiguousQualification turnkeyMTPQualificationRefusal = "AMBIGUOUS_QUALIFICATION"
)

type turnkeyMTPQualificationSelection struct {
	Evidence              fakmodel.Qwen38MTPCanaryEvidence `json:"evidence"`
	ReceiptID             string                           `json:"receipt_id"`
	EvidenceSHA256        string                           `json:"evidence_sha256"`
	SourceURL             string                           `json:"source_url"`
	SourceRevision        string                           `json:"source_revision"`
	TestedSourceRevision  string                           `json:"tested_source_revision"`
	Qualification         turnkeyMTPQualificationKey       `json:"qualification"`
	RequiredHeadroomBytes uint64                           `json:"required_headroom_bytes"`
	RuntimeContext        turnkeyMTPQualificationContext   `json:"runtime_context"`
}

type turnkeyMTPQualificationResult struct {
	Selection *turnkeyMTPQualificationSelection `json:"selection,omitempty"`
	Refusal   turnkeyMTPQualificationRefusal    `json:"refusal,omitempty"`
	Detail    string                            `json:"detail,omitempty"`
}

type turnkeyMTPEvidenceCatalog struct {
	Schema      string                         `json:"schema"`
	Records     []turnkeyMTPEvidenceRecord     `json:"records"`
	Revocations []turnkeyMTPEvidenceRevocation `json:"revocations"`
}

type turnkeyMTPEvidenceRecord struct {
	Evidence             json.RawMessage            `json:"evidence"`
	EvidenceSHA256       string                     `json:"evidence_sha256"`
	SourceURL            string                     `json:"source_url"`
	SourceRevision       string                     `json:"source_revision"`
	TestedSourceRevision string                     `json:"tested_source_revision"`
	Qualification        turnkeyMTPQualificationKey `json:"qualification"`
}

type turnkeyMTPEvidenceRevocation struct {
	ReceiptID      string `json:"receipt_id"`
	EvidenceSHA256 string `json:"evidence_sha256"`
}

func embeddedTurnkeyMTPQualification(now time.Time, runtime turnkeyMTPQualificationContext) turnkeyMTPQualificationResult {
	return selectTurnkeyMTPQualificationCatalog(turnkeyMTPEvidenceCatalogJSON, now, runtime)
}

// selectTurnkeyMTPQualificationCatalog validates the complete reviewed catalog
// before selecting anything. It returns evidence bound to the exact runtime
// context; registration and activation are deliberately left to the caller.
func selectTurnkeyMTPQualificationCatalog(raw []byte, now time.Time, runtime turnkeyMTPQualificationContext) turnkeyMTPQualificationResult {
	if err := validateTurnkeyMTPQualificationContext(runtime); err != nil {
		return turnkeyMTPRefusal(turnkeyMTPNoEligibleContext, err)
	}
	records, revoked, err := decodeTurnkeyMTPEvidenceCatalog(raw)
	if err != nil {
		return turnkeyMTPRefusal(turnkeyMTPMalformedCatalog, err)
	}

	var selected []*turnkeyMTPQualificationSelection
	for i := range records {
		record := records[i]
		key := record.Evidence.Receipt.ReceiptID + "\x00" + record.EvidenceSHA256
		if revoked[key] {
			continue
		}
		if now.IsZero() || record.Evidence.ObservedAt.After(now) || !now.Before(record.Evidence.ValidUntil) {
			continue
		}
		if !turnkeyMTPQualificationMatches(record, runtime) {
			continue
		}
		selection := record
		selection.RuntimeContext = runtime
		selected = append(selected, &selection)
	}
	if len(selected) == 0 {
		return turnkeyMTPQualificationResult{Refusal: turnkeyMTPNoEligibleContext}
	}
	if len(selected) != 1 {
		return turnkeyMTPQualificationResult{Refusal: turnkeyMTPAmbiguousQualification, Detail: "multiple reviewed records match the exact runtime context"}
	}
	return turnkeyMTPQualificationResult{Selection: selected[0]}
}

func decodeTurnkeyMTPEvidenceCatalog(raw []byte) ([]turnkeyMTPQualificationSelection, map[string]bool, error) {
	var catalog turnkeyMTPEvidenceCatalog
	if err := decodeStrictJSON(raw, &catalog); err != nil {
		return nil, nil, fmt.Errorf("decode catalog: %w", err)
	}
	if catalog.Schema != turnkeyMTPEvidenceCatalogSchema {
		return nil, nil, fmt.Errorf("catalog schema %q, want %q", catalog.Schema, turnkeyMTPEvidenceCatalogSchema)
	}
	revoked := make(map[string]bool, len(catalog.Revocations))
	for i, tombstone := range catalog.Revocations {
		if tombstone.ReceiptID == "" || tombstone.ReceiptID != strings.TrimSpace(tombstone.ReceiptID) || !turnkeyMTPIsLowerSHA256(tombstone.EvidenceSHA256) {
			return nil, nil, fmt.Errorf("revocation %d has an invalid identity", i)
		}
		key := tombstone.ReceiptID + "\x00" + tombstone.EvidenceSHA256
		if revoked[key] {
			return nil, nil, fmt.Errorf("duplicate revocation for receipt %q", tombstone.ReceiptID)
		}
		revoked[key] = true
	}

	receiptIDs := make(map[string]bool, len(catalog.Records))
	bindings := make(map[string]bool, len(catalog.Records))
	records := make([]turnkeyMTPQualificationSelection, 0, len(catalog.Records))
	for i, record := range catalog.Records {
		var evidence fakmodel.Qwen38MTPCanaryEvidence
		if err := decodeStrictJSON(record.Evidence, &evidence); err != nil {
			return nil, nil, fmt.Errorf("record %d evidence: %w", i, err)
		}
		if err := evidence.Validate(); err != nil {
			return nil, nil, fmt.Errorf("record %d evidence: %w", i, err)
		}
		actual := sha256.Sum256(record.Evidence)
		// EvidenceSHA256 is the publication binding: it hashes the exact embedded
		// evidence JSON value bytes that this loader validates and returns.
		if !turnkeyMTPIsLowerSHA256(record.EvidenceSHA256) || record.EvidenceSHA256 != hex.EncodeToString(actual[:]) {
			return nil, nil, fmt.Errorf("record %d evidence digest mismatch", i)
		}
		if err := validateTurnkeyMTPQualificationKey(record.Qualification); err != nil {
			return nil, nil, fmt.Errorf("record %d context: %w", i, err)
		}
		if record.Qualification.Workload.DraftDepth != evidence.Receipt.Envelope.DraftDepth {
			return nil, nil, fmt.Errorf("record %d draft depth does not match evidence", i)
		}
		if normalizedSHA256(evidence.Receipt.Envelope.ArtifactHash) != record.Qualification.ModelArtifactSHA256 ||
			evidence.Receipt.Envelope.ModelFamily != record.Qualification.ModelFamily ||
			evidence.Receipt.Envelope.Format != record.Qualification.MTPFormat || evidence.Receipt.Envelope.Backend != fakmodel.Qwen38MTPBackendMetal {
			return nil, nil, fmt.Errorf("record %d evidence envelope does not match qualification context", i)
		}
		if err := validateImmutableTurnkeyMTPSource(record.SourceURL, record.SourceRevision); err != nil {
			return nil, nil, fmt.Errorf("record %d source: %w", i, err)
		}
		if !turnkeyMTPIsLowerHex(record.TestedSourceRevision, 40) {
			return nil, nil, fmt.Errorf("record %d tested source revision must be a full lowercase commit", i)
		}
		receiptID := strings.TrimSpace(evidence.Receipt.ReceiptID)
		if receiptID == "" || evidence.Receipt.ReceiptID != receiptID {
			return nil, nil, fmt.Errorf("record %d receipt id is not canonical", i)
		}
		if receiptIDs[receiptID] {
			return nil, nil, fmt.Errorf("duplicate receipt id %q", receiptID)
		}
		receiptIDs[receiptID] = true
		binding := record.EvidenceSHA256
		if bindings[binding] {
			return nil, nil, fmt.Errorf("duplicate evidence binding for receipt %q", receiptID)
		}
		bindings[binding] = true
		records = append(records, turnkeyMTPQualificationSelection{
			Evidence: evidence, ReceiptID: receiptID, EvidenceSHA256: record.EvidenceSHA256,
			SourceURL: record.SourceURL, SourceRevision: record.SourceRevision,
			TestedSourceRevision: record.TestedSourceRevision, Qualification: record.Qualification,
			RequiredHeadroomBytes: evidence.Receipt.Envelope.HeadroomBytes,
		})
	}
	return records, revoked, nil
}

func turnkeyMTPQualificationMatches(record turnkeyMTPQualificationSelection, runtime turnkeyMTPQualificationContext) bool {
	return record.Qualification == runtime.Qualification &&
		runtime.AvailableHeadroomBytes >= record.RequiredHeadroomBytes
}

func validateTurnkeyMTPQualificationContext(context turnkeyMTPQualificationContext) error {
	return validateTurnkeyMTPQualificationKey(context.Qualification)
}

func validateTurnkeyMTPQualificationKey(key turnkeyMTPQualificationKey) error {
	if !turnkeyMTPIsLowerSHA256(key.ModelArtifactSHA256) || !turnkeyMTPIsLowerSHA256(key.NativeConfigSHA256) {
		return fmt.Errorf("qualification context requires lowercase SHA-256 artifact and native-config identities")
	}
	if key.ModelFamily != "Qwen3.8" || key.MTPFormat == "" {
		return fmt.Errorf("qualification context requires model family and MTP format")
	}
	for name, value := range map[string]string{
		"device": key.DeviceCompatibility, "metal": key.MetalCompatibility, "engine": key.EngineCompatibility,
	} {
		if value != strings.TrimSpace(value) || !isVersionedCompatibility(value) {
			return fmt.Errorf("qualification context %s compatibility %q is not an exact versioned identity", name, value)
		}
	}
	if !key.Workload.Greedy || key.Workload.ContextTokens <= 0 || key.Workload.DraftDepth <= 0 || key.Workload.DraftDepth > fakmodel.Qwen35MTPMaxDraftDepth {
		return fmt.Errorf("qualification context requires greedy decode and supported positive context/depth")
	}
	return nil
}

func isVersionedCompatibility(value string) bool {
	value = strings.TrimSpace(value)
	return value != "" && !strings.ContainsAny(value, "*?") && strings.Contains(value, "/v")
}

func validateImmutableTurnkeyMTPSource(rawURL, revision string) error {
	if !turnkeyMTPIsLowerHex(revision, 40) {
		return fmt.Errorf("source revision must be a full lowercase commit")
	}
	u, err := url.Parse(rawURL)
	if err != nil || u.Scheme != "https" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || u.Host != u.Hostname() {
		return fmt.Errorf("source URL is not immutable HTTPS provenance")
	}
	parts := strings.Split(strings.Trim(u.EscapedPath(), "/"), "/")
	switch strings.ToLower(u.Hostname()) {
	case "github.com":
		if len(parts) < 5 || parts[2] != "blob" || parts[3] != revision {
			return fmt.Errorf("GitHub source URL must contain /owner/repo/blob/<full-commit>/<path>")
		}
	case "raw.githubusercontent.com":
		if len(parts) < 4 || parts[2] != revision {
			return fmt.Errorf("raw GitHub source URL must contain /owner/repo/<full-commit>/<path>")
		}
	default:
		return fmt.Errorf("source host is not in the immutable-public-source allowlist")
	}
	return nil
}

func decodeStrictJSON(raw []byte, dst any) error {
	if err := rejectTurnkeyMTPDuplicateJSONNames(raw); err != nil {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(dst); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			return fmt.Errorf("trailing JSON value")
		}
		return fmt.Errorf("trailing JSON: %w", err)
	}
	return nil
}

// rejectTurnkeyMTPDuplicateJSONNames walks every nested object before typed
// decoding so encoding/json's last-key-wins behavior cannot rewrite authority.
func rejectTurnkeyMTPDuplicateJSONNames(raw []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	var walk func(json.Token) error
	var next func() error
	next = func() error {
		token, err := decoder.Token()
		if err != nil {
			return err
		}
		return walk(token)
	}
	walk = func(token json.Token) error {
		delim, ok := token.(json.Delim)
		if !ok {
			return nil
		}
		switch delim {
		case '{':
			seen := make(map[string]struct{})
			for decoder.More() {
				nameToken, err := decoder.Token()
				if err != nil {
					return err
				}
				name, ok := nameToken.(string)
				if !ok {
					return fmt.Errorf("JSON object name is not a string")
				}
				canonicalName := strings.ToLower(name)
				if name != canonicalName || strings.Trim(name, "abcdefghijklmnopqrstuvwxyz0123456789_") != "" {
					return fmt.Errorf("JSON object name %q is not canonical lowercase", name)
				}
				if _, exists := seen[canonicalName]; exists {
					return fmt.Errorf("duplicate JSON object name %q", name)
				}
				seen[canonicalName] = struct{}{}
				if err := next(); err != nil {
					return err
				}
			}
			end, err := decoder.Token()
			if err != nil || end != json.Delim('}') {
				return fmt.Errorf("malformed JSON object")
			}
		case '[':
			for decoder.More() {
				if err := next(); err != nil {
					return err
				}
			}
			end, err := decoder.Token()
			if err != nil || end != json.Delim(']') {
				return fmt.Errorf("malformed JSON array")
			}
		default:
			return fmt.Errorf("unexpected JSON delimiter %q", delim)
		}
		return nil
	}
	return next()
}

func turnkeyMTPIsLowerSHA256(value string) bool { return turnkeyMTPIsLowerHex(value, sha256.Size*2) }

func turnkeyMTPIsLowerHex(value string, size int) bool {
	if len(value) != size || value != strings.ToLower(value) {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func normalizedSHA256(value string) string {
	value = strings.TrimSpace(strings.ToLower(value))
	return strings.TrimPrefix(value, "sha256:")
}

func turnkeyMTPRefusal(code turnkeyMTPQualificationRefusal, err error) turnkeyMTPQualificationResult {
	detail := err.Error()
	if len(detail) > 256 {
		detail = detail[:256]
	}
	return turnkeyMTPQualificationResult{Refusal: code, Detail: detail}
}
