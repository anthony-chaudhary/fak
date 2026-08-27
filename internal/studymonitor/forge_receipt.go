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
	Schema         string                            `json:"schema"`
	Repository     string                            `json:"repository"`
	Revision       string                            `json:"revision"`
	Cutoff         string                            `json:"cutoff"`
	Status         string                            `json:"status"`
	Sources        []StudyForgeSourceReceiptEvidence `json:"sources"`
	NonAtomicDelta *StudyForgeNonAtomicDeltaEvidence `json:"non_atomic_delta"`
	IndexChecksum  string                            `json:"index_checksum"`
}

type StudyForgeSourceReceiptEvidence struct {
	Name                string            `json:"name"`
	Status              string            `json:"status"`
	Pages               []json.RawMessage `json:"pages"`
	FetchedCount        int               `json:"fetched_count"`
	NormalizedCount     int               `json:"normalized_count"`
	UniqueCount         int               `json:"unique_count"`
	ClassifiedPullCount int               `json:"classified_pull_count,omitempty"`
	Checksum            string            `json:"checksum"`
	Failure             json.RawMessage   `json:"failure,omitempty"`
}

type StudyForgeNonAtomicDeltaEvidence struct {
	Type                 string                            `json:"type"`
	MixedSource          string                            `json:"mixed_source"`
	DedicatedSource      string                            `json:"dedicated_source"`
	IdentityBasis        string                            `json:"identity_basis"`
	MixedCrawl           StudyForgeCrawlWindow             `json:"mixed_crawl"`
	DedicatedCrawl       StudyForgeCrawlWindow             `json:"dedicated_crawl"`
	MixedCount           int                               `json:"mixed_count"`
	DedicatedCount       int                               `json:"dedicated_count"`
	OverlapCount         int                               `json:"overlap_count"`
	OnlyInMixedCount     int                               `json:"only_in_mixed_count"`
	OnlyInDedicatedCount int                               `json:"only_in_dedicated_count"`
	Overlap              []StudyForgeCrossEndpointIdentity `json:"overlap"`
	OnlyInMixed          []StudyForgeCrossEndpointIdentity `json:"only_in_mixed"`
	OnlyInDedicated      []StudyForgeCrossEndpointIdentity `json:"only_in_dedicated"`
	Policy               *StudyForgeNonAtomicDeltaPolicy   `json:"policy"`
	Accepted             bool                              `json:"accepted"`
}

type StudyForgeCrossEndpointIdentity struct {
	ID     int64  `json:"id"`
	Number int    `json:"number,omitempty"`
	NodeID string `json:"node_id,omitempty"`
}

type StudyForgeCrawlWindow struct {
	StartedAt string `json:"started_at"`
	EndedAt   string `json:"ended_at"`
}

type StudyForgeNonAtomicDeltaPolicy struct {
	Type               string `json:"type"`
	MaxOnlyInMixed     int    `json:"max_only_in_mixed"`
	MaxOnlyInDedicated int    `json:"max_only_in_dedicated"`
	MaxTotal           int    `json:"max_total"`
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
	var issues, pulls *StudyForgeSourceReceiptEvidence
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
		copy := source
		if name == "issues" {
			issues = &copy
		}
		if name == "pulls" {
			pulls = &copy
		}
	}
	for _, required := range requiredStudyForgeSources {
		if !seen[required] {
			return fmt.Errorf("missing source %s", required)
		}
	}
	if issues == nil || pulls == nil {
		return fmt.Errorf("issues and pulls sources are required for non_atomic_delta")
	}
	if err := validateStudyForgeNonAtomicDelta(receipt.NonAtomicDelta, *issues, *pulls); err != nil {
		return err
	}
	return nil
}

func validateStudyForgeNonAtomicDelta(delta *StudyForgeNonAtomicDeltaEvidence, issues, pulls StudyForgeSourceReceiptEvidence) error {
	if delta == nil {
		return fmt.Errorf("non_atomic_delta is required")
	}
	if delta.Type != "non_atomic_delta" || delta.MixedSource != "issues" || delta.DedicatedSource != "pulls" {
		return fmt.Errorf("non_atomic_delta has invalid type or sources")
	}
	if delta.IdentityBasis != "captured_endpoint_rows" && delta.IdentityBasis != "legacy_checkpoint_projection" {
		return fmt.Errorf("non_atomic_delta has invalid identity_basis")
	}
	for name, window := range map[string]StudyForgeCrawlWindow{"mixed": delta.MixedCrawl, "dedicated": delta.DedicatedCrawl} {
		started, startErr := time.Parse(time.RFC3339Nano, window.StartedAt)
		ended, endErr := time.Parse(time.RFC3339Nano, window.EndedAt)
		if startErr != nil || endErr != nil || ended.Before(started) {
			return fmt.Errorf("non_atomic_delta %s crawl window is invalid", name)
		}
	}
	if delta.Overlap == nil || delta.OnlyInMixed == nil || delta.OnlyInDedicated == nil {
		return fmt.Errorf("non_atomic_delta exact identity sets are required")
	}
	seenIdentities := make(map[int64]bool, len(delta.Overlap)+len(delta.OnlyInMixed)+len(delta.OnlyInDedicated))
	for name, identities := range map[string][]StudyForgeCrossEndpointIdentity{"overlap": delta.Overlap, "only_in_mixed": delta.OnlyInMixed, "only_in_dedicated": delta.OnlyInDedicated} {
		for i, identity := range identities {
			if identity.ID <= 0 || seenIdentities[identity.ID] {
				return fmt.Errorf("non_atomic_delta %s contains invalid or duplicate identity", name)
			}
			seenIdentities[identity.ID] = true
			if i > 0 && identities[i-1].ID >= identity.ID {
				return fmt.Errorf("non_atomic_delta %s is not in canonical order", name)
			}
		}
	}
	if delta.MixedCount != issues.ClassifiedPullCount || delta.DedicatedCount != pulls.NormalizedCount || delta.OverlapCount != len(delta.Overlap) || delta.OnlyInMixedCount != len(delta.OnlyInMixed) || delta.OnlyInDedicatedCount != len(delta.OnlyInDedicated) || delta.MixedCount != delta.OverlapCount+delta.OnlyInMixedCount || delta.DedicatedCount != delta.OverlapCount+delta.OnlyInDedicatedCount {
		return fmt.Errorf("non_atomic_delta counts contradict endpoint evidence")
	}
	if delta.Policy == nil || delta.Policy.Type != "bounded_identity_delta" || delta.Policy.MaxOnlyInMixed < 0 || delta.Policy.MaxOnlyInDedicated < 0 || delta.Policy.MaxTotal < 0 || delta.Policy.MaxOnlyInMixed > 1000 || delta.Policy.MaxOnlyInDedicated > 1000 || delta.Policy.MaxTotal > 1000 {
		return fmt.Errorf("non_atomic_delta acceptance policy is missing or unbounded")
	}
	accepted := delta.OnlyInMixedCount <= delta.Policy.MaxOnlyInMixed && delta.OnlyInDedicatedCount <= delta.Policy.MaxOnlyInDedicated && delta.OnlyInMixedCount+delta.OnlyInDedicatedCount <= delta.Policy.MaxTotal
	if !delta.Accepted || !accepted {
		return fmt.Errorf("non_atomic_delta policy did not accept the endpoint drift")
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
