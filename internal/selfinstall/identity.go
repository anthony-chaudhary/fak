package selfinstall

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const installIdentitySchema = "fak.selfinstall.identity/v1"

// StateUpdate is the verified identity produced by the build gate. The selected
// source may advance while ArtifactSourceCommit and ArtifactDigest continue to name reused
// bytes.
type StateUpdate struct {
	SignedMetadataGeneration uint64
	SelectedSourceCommit     string
	ArtifactSourceCommit     string
	BuildInputDigest         string
	ArtifactDigest           string
	ArtifactSize             int64
	AppVersion               string
}

// ArtifactRecord is a verified activation or rollback unit. ID is always the artifact digest;
// version strings are descriptive and are never accepted as slot selectors.
type ArtifactRecord struct {
	ID                   string `json:"id"`
	Path                 string `json:"path"`
	ArtifactSourceCommit string `json:"artifact_source_commit"`
	ArtifactDigest       string `json:"artifact_digest"`
	ArtifactSize         int64  `json:"artifact_size"`
	AppVersion           string `json:"app_version"`
}

// InstallIdentity is the persisted identity contract for one installed binary.
type InstallIdentity struct {
	Schema                   string           `json:"schema"`
	MetadataGeneration       uint64           `json:"metadata_generation"`
	SignedMetadataGeneration uint64           `json:"signed_metadata_generation,omitempty"`
	SelectedSourceCommit     string           `json:"selected_source_commit"`
	BuildInputDigest         string           `json:"build_input_digest"`
	ArtifactDigest           string           `json:"artifact_digest"`
	AppVersion               string           `json:"app_version"`
	CurrentDigest            string           `json:"current_artifact_digest"`
	Artifacts                []ArtifactRecord `json:"slots"`
}

// IdentityStatePath is owned by selfinstall and stays beside the binary whose identity it
// describes. This keeps independently installed copies from sharing ambiguous activation state.
func IdentityStatePath(target string) string {
	return filepath.Clean(target) + ".self-update-identity.json"
}

// ArtifactsEqual compares artifact bytes by size and SHA-256. It never trusts source
// revisions or version strings as evidence that activation would be a no-op.
func ArtifactsEqual(left, right string) (bool, error) {
	leftDigest, leftSize, err := fileSHA256(left)
	if err != nil {
		return false, err
	}
	rightDigest, rightSize, err := fileSHA256(right)
	if err != nil {
		return false, err
	}
	return leftSize == rightSize && leftDigest == rightDigest, nil
}

// ReadInstallIdentity returns an empty identity when no state has been persisted yet.
func ReadInstallIdentity(path string) (InstallIdentity, error) {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return InstallIdentity{}, nil
	}
	if err != nil {
		return InstallIdentity{}, err
	}
	var state InstallIdentity
	if err := json.Unmarshal(data, &state); err != nil {
		return InstallIdentity{}, fmt.Errorf("decode install identity: %w", err)
	}
	if err := validateInstallIdentity(state); err != nil {
		return InstallIdentity{}, err
	}
	return state, nil
}

// AdvanceInstallIdentity persists one verified selection. MetadataGeneration advances on
// every selection, including digest-equal refreshes. When activated is false, the active app
// version is preserved; when true, the prior verified active slot becomes the rollback slot.
func AdvanceInstallIdentity(path string, prior InstallIdentity, candidate StateUpdate, activePath, rollbackPath string, activated bool) (InstallIdentity, error) {
	if err := validateStateUpdate(candidate); err != nil {
		return InstallIdentity{}, err
	}
	if prior.Schema != "" {
		if err := validateInstallIdentity(prior); err != nil {
			return InstallIdentity{}, err
		}
	}
	if candidate.SignedMetadataGeneration != 0 {
		if candidate.SignedMetadataGeneration < prior.SignedMetadataGeneration {
			return InstallIdentity{}, fmt.Errorf("signed metadata generation rollback: got %d, installed %d", candidate.SignedMetadataGeneration, prior.SignedMetadataGeneration)
		}
		if candidate.SignedMetadataGeneration == prior.SignedMetadataGeneration && prior.SignedMetadataGeneration != 0 {
			if prior.SelectedSourceCommit == strings.ToLower(strings.TrimSpace(candidate.SelectedSourceCommit)) &&
				prior.ArtifactDigest == strings.ToLower(strings.TrimSpace(candidate.ArtifactDigest)) &&
				prior.AppVersion == strings.TrimSpace(candidate.AppVersion) {
				return prior, nil
			}
			return InstallIdentity{}, fmt.Errorf("signed metadata generation freeze mismatch at %d", candidate.SignedMetadataGeneration)
		}
	}
	signedGeneration := candidate.SignedMetadataGeneration
	if signedGeneration == 0 {
		signedGeneration = prior.SignedMetadataGeneration
	}

	appVersion := strings.TrimSpace(candidate.AppVersion)
	artifactSource := strings.ToLower(strings.TrimSpace(candidate.ArtifactSourceCommit))
	records := make([]ArtifactRecord, 0, 2)
	if !activated {
		records = append(records, prior.Artifacts...)
		if active, ok := VerifiedArtifact(prior, candidate.ArtifactDigest); ok {
			if strings.TrimSpace(active.AppVersion) != "" {
				appVersion = active.AppVersion
			}
			if strings.TrimSpace(active.ArtifactSourceCommit) != "" {
				artifactSource = active.ArtifactSourceCommit
			}
		}
	} else if active, ok := VerifiedArtifact(prior, prior.CurrentDigest); ok {
		switch {
		case recordMatchesPath(active, active.Path):
			// A persistent verified slot is already independently recoverable.
			records = append(records, active)
		case recordMatchesPath(active, rollbackPath):
			active.Path = filepath.Clean(rollbackPath)
			records = append(records, active)
		}
	}

	active := ArtifactRecord{
		ID:                   candidate.ArtifactDigest,
		Path:                 filepath.Clean(activePath),
		ArtifactSourceCommit: artifactSource,
		ArtifactDigest:       candidate.ArtifactDigest,
		ArtifactSize:         candidate.ArtifactSize,
		AppVersion:           appVersion,
	}
	records = append(records, active)
	state := InstallIdentity{
		Schema:                   installIdentitySchema,
		MetadataGeneration:       prior.MetadataGeneration + 1,
		SignedMetadataGeneration: signedGeneration,
		SelectedSourceCommit:     strings.ToLower(strings.TrimSpace(candidate.SelectedSourceCommit)),
		BuildInputDigest:         candidate.BuildInputDigest,
		ArtifactDigest:           candidate.ArtifactDigest,
		AppVersion:               appVersion,
		CurrentDigest:            active.ID,
		Artifacts:                dedupeRecords(records),
	}
	if err := validateInstallIdentity(state); err != nil {
		return InstallIdentity{}, err
	}
	data, err := json.Marshal(state)
	if err != nil {
		return InstallIdentity{}, err
	}
	data = append(data, '\n')
	if err := atomicWriteFile(path, 0o600, bytes.NewReader(data)); err != nil {
		return InstallIdentity{}, fmt.Errorf("persist install identity: %w", err)
	}
	return state, nil
}

// VerifiedArtifact resolves rollback by artifact digest. A release/app version is deliberately
// not a valid selector because multiple verified artifacts may report the same version.
func VerifiedArtifact(state InstallIdentity, digest string) (ArtifactRecord, bool) {
	digest = strings.ToLower(strings.TrimSpace(digest))
	if !validSHA256(digest) {
		return ArtifactRecord{}, false
	}
	for _, record := range state.Artifacts {
		if record.ID == digest && record.ArtifactDigest == digest && validRecord(record) {
			return record, true
		}
	}
	return ArtifactRecord{}, false
}

func validateStateUpdate(candidate StateUpdate) error {
	if !validCommit(strings.ToLower(strings.TrimSpace(candidate.SelectedSourceCommit))) {
		return fmt.Errorf("selected source commit is not a full Git object ID")
	}
	if !validCommit(strings.ToLower(strings.TrimSpace(candidate.ArtifactSourceCommit))) {
		return fmt.Errorf("artifact source commit is not a full Git object ID")
	}
	if !validSHA256(candidate.BuildInputDigest) {
		return fmt.Errorf("build-input digest is not SHA-256")
	}
	if !validSHA256(candidate.ArtifactDigest) || candidate.ArtifactSize < 1 {
		return fmt.Errorf("artifact identity is invalid")
	}
	if strings.TrimSpace(candidate.AppVersion) == "" {
		return fmt.Errorf("app version is required")
	}
	return nil
}

func validateInstallIdentity(state InstallIdentity) error {
	if state.Schema != installIdentitySchema {
		return fmt.Errorf("install identity schema %q is not supported", state.Schema)
	}
	if state.MetadataGeneration == 0 {
		return fmt.Errorf("install identity metadata generation must be positive")
	}
	if !validCommit(state.SelectedSourceCommit) || !validSHA256(state.BuildInputDigest) ||
		!validSHA256(state.ArtifactDigest) || state.CurrentDigest != state.ArtifactDigest ||
		strings.TrimSpace(state.AppVersion) == "" {
		return fmt.Errorf("install identity metadata is malformed")
	}
	if _, ok := VerifiedArtifact(state, state.CurrentDigest); !ok {
		return fmt.Errorf("install identity active slot is not verified")
	}
	return nil
}

func validRecord(record ArtifactRecord) bool {
	return validSHA256(record.ID) && record.ID == record.ArtifactDigest &&
		validCommit(record.ArtifactSourceCommit) && record.ArtifactSize > 0 &&
		strings.TrimSpace(record.Path) != "" && strings.TrimSpace(record.AppVersion) != ""
}

func recordMatchesPath(record ArtifactRecord, path string) bool {
	digest, size, err := fileSHA256(path)
	return err == nil && digest == record.ArtifactDigest && size == record.ArtifactSize
}

func dedupeRecords(records []ArtifactRecord) []ArtifactRecord {
	seen := make(map[string]bool, len(records))
	out := make([]ArtifactRecord, 0, len(records))
	for i := len(records) - 1; i >= 0; i-- {
		if seen[records[i].ID] {
			continue
		}
		seen[records[i].ID] = true
		out = append(out, records[i])
	}
	for left, right := 0, len(out)-1; left < right; left, right = left+1, right-1 {
		out[left], out[right] = out[right], out[left]
	}
	return out
}

func digestBytes(body []byte) string {
	sum := sha256.Sum256(body)
	return hex.EncodeToString(sum[:])
}
