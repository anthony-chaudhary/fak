package studymonitor

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const StudyForgeReceiptSchema = "fak-studyforge-receipt/1"
const studyForgeCorpusSchema = "fak-studyforge-corpus/1"

var requiredStudyForgeSources = []string{
	"issues",
	"pulls",
	"discussions",
	"releases",
	"labels",
	"milestones",
}

// StudyForgeReceiptEvidence is the stable subset of a studyforge receipt that
// the inventory gate needs. Deliberately omitting collector-internal fields
// lets the forge leaf add page and normalization evidence without coupling the
// monitor to its storage format.
type StudyForgeReceiptEvidence struct {
	Schema        string                            `json:"schema"`
	Repository    string                            `json:"repository"`
	Revision      string                            `json:"revision"`
	Cutoff        string                            `json:"cutoff"`
	Status        string                            `json:"status"`
	Sources       []StudyForgeSourceReceiptEvidence `json:"sources"`
	IndexChecksum string                            `json:"index_checksum"`
}

type StudyForgeSourceReceiptEvidence struct {
	Name            string            `json:"name"`
	Status          string            `json:"status"`
	Pages           []json.RawMessage `json:"pages"`
	FetchedCount    int               `json:"fetched_count"`
	NormalizedCount int               `json:"normalized_count"`
	UniqueCount     int               `json:"unique_count"`
	Checksum        string            `json:"checksum"`
	Failure         json.RawMessage   `json:"failure,omitempty"`
}

func validateStudyForgeReceiptFile(row *InventoryRow, repo Repository, repoRoot string) bool {
	if row.Mode != InventoryModeExhaustive || row.ForgeReceiptPath == "" {
		return false
	}
	path := row.ForgeReceiptPath
	if !filepath.IsAbs(path) {
		path = filepath.Join(repoRoot, filepath.FromSlash(path))
	}
	data, err := os.ReadFile(path)
	if err != nil {
		addInventoryRowReason(row, "studyforge receipt is not readable JSON: "+err.Error())
		return false
	}
	receipt, err := decodeStudyForgeReceiptEvidence(data)
	if err != nil {
		addInventoryRowReason(row, "studyforge receipt is not readable JSON: "+err.Error())
		return false
	}
	if err := ValidateStudyForgeReceiptEvidence(receipt, repo.Repository, repo.CheckedRevision); err != nil {
		addInventoryRowReason(row, "studyforge receipt is invalid: "+err.Error())
		return false
	}
	return true
}

// ValidateStudyForgeReceiptEvidence rejects partial or mismatched forge claims.
// The studyforge package remains responsible for its deeper page-continuity and
// record validation; this boundary verifies the facts the inventory consumes.
func ValidateStudyForgeReceiptEvidence(receipt StudyForgeReceiptEvidence, repository, revision string) error {
	if receipt.Schema != StudyForgeReceiptSchema {
		return fmt.Errorf("schema must be %q", StudyForgeReceiptSchema)
	}
	if !strings.EqualFold(strings.TrimSpace(receipt.Repository), strings.TrimSpace(repository)) {
		return fmt.Errorf("repository does not match registry row")
	}
	if strings.TrimSpace(receipt.Revision) != strings.TrimSpace(revision) {
		return fmt.Errorf("revision does not match checked_revision")
	}
	if _, err := time.Parse(time.RFC3339, receipt.Cutoff); err != nil {
		return fmt.Errorf("cutoff must be RFC3339")
	}
	if receipt.Status != "complete" {
		return fmt.Errorf("status must be complete, got %q", receipt.Status)
	}
	if strings.TrimSpace(receipt.IndexChecksum) == "" {
		return fmt.Errorf("index_checksum is required")
	}
	seen := make(map[string]bool, len(receipt.Sources))
	for i, source := range receipt.Sources {
		name := strings.TrimSpace(source.Name)
		if seen[name] {
			return fmt.Errorf("sources[%d]: duplicate source %q", i, name)
		}
		seen[name] = true
		if !requiredStudyForgeSource(name) {
			return fmt.Errorf("sources[%d]: unsupported source %q", i, name)
		}
		if source.Status != "complete" {
			return fmt.Errorf("source %s status must be complete", name)
		}
		if len(source.Pages) == 0 {
			return fmt.Errorf("source %s pages must be positive", name)
		}
		if source.FetchedCount < 0 || source.NormalizedCount < 0 || source.UniqueCount < 0 {
			return fmt.Errorf("source %s counts must be non-negative", name)
		}
		if source.FetchedCount < source.NormalizedCount {
			return fmt.Errorf("source %s fetched_count is smaller than normalized_count", name)
		}
		if source.NormalizedCount != source.UniqueCount {
			return fmt.Errorf("source %s normalized_count does not match unique_count", name)
		}
		if strings.TrimSpace(source.Checksum) == "" {
			return fmt.Errorf("source %s checksum is required", name)
		}
		if hasStudyForgeFailure(source.Failure) {
			return fmt.Errorf("source %s contains failure evidence", name)
		}
	}
	for _, required := range requiredStudyForgeSources {
		if !seen[required] {
			return fmt.Errorf("missing source %s", required)
		}
	}
	return nil
}

func decodeStudyForgeReceiptEvidence(data []byte) (StudyForgeReceiptEvidence, error) {
	var envelope struct {
		Schema  string          `json:"schema"`
		Receipt json.RawMessage `json:"receipt"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		return StudyForgeReceiptEvidence{}, err
	}
	payload := data
	if envelope.Schema == studyForgeCorpusSchema {
		if len(envelope.Receipt) == 0 || string(envelope.Receipt) == "null" {
			return StudyForgeReceiptEvidence{}, fmt.Errorf("studyforge corpus receipt is required")
		}
		payload = envelope.Receipt
	}
	var receipt StudyForgeReceiptEvidence
	if err := json.Unmarshal(payload, &receipt); err != nil {
		return StudyForgeReceiptEvidence{}, err
	}
	return receipt, nil
}

func requiredStudyForgeSource(name string) bool {
	for _, required := range requiredStudyForgeSources {
		if name == required {
			return true
		}
	}
	return false
}

func hasStudyForgeFailure(raw json.RawMessage) bool {
	trimmed := strings.TrimSpace(string(raw))
	return trimmed != "" && trimmed != "null" && trimmed != `""`
}
