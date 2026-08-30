package selfinstall

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const verifiedSlotSchema = "fak.selfinstall.verified-slot/v1"

// VerifiedTarget is the signed identity a downloaded executable must prove before it can
// enter a persistent activation slot.
type VerifiedTarget struct {
	MetadataGeneration uint64
	SourceCommit       string
	ArtifactDigest     string
	ArtifactSize       int64
	AppVersion         string
}

type verifiedSlotMetadata struct {
	Schema             string `json:"schema"`
	MetadataGeneration uint64 `json:"metadata_generation"`
	SourceCommit       string `json:"source_commit"`
	ArtifactDigest     string `json:"artifact_digest"`
	ArtifactSize       int64  `json:"artifact_size"`
	AppVersion         string `json:"app_version"`
}

// VerifyTarget checks byte identity first and executable provenance second. A candidate that
// merely runs is insufficient: its version and exact source commit must match the signed target.
func VerifyTarget(ctx context.Context, run Runner, candidate, repoRoot string, target VerifiedTarget) error {
	if err := validateVerifiedTarget(target); err != nil {
		return err
	}
	digest, size, err := fileSHA256(candidate)
	if err != nil {
		return fmt.Errorf("hash downloaded artifact: %w", err)
	}
	if size != target.ArtifactSize {
		return fmt.Errorf("artifact size mismatch: got %d want %d", size, target.ArtifactSize)
	}
	if !strings.EqualFold(digest, target.ArtifactDigest) {
		return fmt.Errorf("artifact SHA-256 mismatch")
	}
	identity, out, ok := smokeCandidate(ctx, run, repoRoot, candidate, target.SourceCommit)
	if !ok {
		return fmt.Errorf("artifact provenance/smoke failed: %s", trim(out))
	}
	if identity.AppVersion != target.AppVersion {
		return fmt.Errorf("artifact app version mismatch: got %q want %q", identity.AppVersion, target.AppVersion)
	}
	return nil
}

// StoreVerifiedSlot copies a verified target into an immutable generation+digest slot. Existing
// slots are re-verified and never overwritten, preserving prior verified rollback material.
func StoreVerifiedSlot(targetPath, candidate string, target VerifiedTarget) (string, error) {
	if err := validateVerifiedTarget(target); err != nil {
		return "", err
	}
	root := filepath.Clean(targetPath) + ".self-update-slots"
	dir := filepath.Join(root, fmt.Sprintf("g%020d-%s", target.MetadataGeneration, target.ArtifactDigest))
	artifactPath := filepath.Join(dir, filepath.Base(targetPath))
	metadataPath := filepath.Join(dir, "slot.json")
	expected := verifiedSlotMetadata{
		Schema: verifiedSlotSchema, MetadataGeneration: target.MetadataGeneration,
		SourceCommit: strings.ToLower(target.SourceCommit), ArtifactDigest: strings.ToLower(target.ArtifactDigest),
		ArtifactSize: target.ArtifactSize, AppVersion: target.AppVersion,
	}
	if data, err := os.ReadFile(metadataPath); err == nil {
		var existing verifiedSlotMetadata
		if json.Unmarshal(data, &existing) != nil || existing != expected {
			return "", fmt.Errorf("verified slot freeze mismatch at metadata generation %d", target.MetadataGeneration)
		}
		digest, size, err := fileSHA256(artifactPath)
		if err != nil || digest != expected.ArtifactDigest || size != expected.ArtifactSize {
			return "", fmt.Errorf("existing verified slot artifact is corrupt")
		}
		return artifactPath, nil
	} else if !os.IsNotExist(err) {
		return "", err
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	if digest, size, err := fileSHA256(artifactPath); err == nil {
		if digest != expected.ArtifactDigest || size != expected.ArtifactSize {
			return "", fmt.Errorf("unrecorded verified slot artifact is corrupt")
		}
	} else if !os.IsNotExist(err) {
		return "", err
	} else {
		tmp, err := stageCopy(candidate, artifactPath, "slot")
		if err != nil {
			return "", err
		}
		if err := os.Rename(tmp, artifactPath); err != nil {
			_ = os.Remove(tmp)
			return "", err
		}
	}
	data, err := json.Marshal(expected)
	if err != nil {
		return "", err
	}
	data = append(data, '\n')
	if err := atomicWriteFile(metadataPath, 0o600, strings.NewReader(string(data))); err != nil {
		return "", err
	}
	return artifactPath, nil
}

func validateVerifiedTarget(target VerifiedTarget) error {
	if target.MetadataGeneration == 0 {
		return fmt.Errorf("metadata generation must be positive")
	}
	if !validCommit(strings.ToLower(strings.TrimSpace(target.SourceCommit))) {
		return fmt.Errorf("artifact source commit is not a full Git object ID")
	}
	if !validSHA256(strings.ToLower(strings.TrimSpace(target.ArtifactDigest))) || target.ArtifactSize < 1 {
		return fmt.Errorf("artifact identity is invalid")
	}
	if strings.TrimSpace(target.AppVersion) == "" {
		return fmt.Errorf("artifact app version is required")
	}
	return nil
}
