package modelaccept

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// LadderEvidenceAdmission records the fail-closed integrity admission result for
// a readiness ladder packet. Digest details are populated for mismatches so an
// operator can identify the exact immutable artifact that was rejected.
type LadderEvidenceAdmission struct {
	Verdict        Verdict `json:"verdict"`
	Code           string  `json:"code,omitempty"`
	Path           string  `json:"path,omitempty"`
	ExpectedSHA256 string  `json:"expected_sha256,omitempty"`
	ActualSHA256   string  `json:"actual_sha256,omitempty"`
	Reason         string  `json:"reason,omitempty"`
}

type ladderChecksum struct {
	File   string `json:"file"`
	SHA256 string `json:"sha256"`
}

func verifyLadderEvidence(dir, manifest string) LadderEvidenceAdmission {
	hold := func(code, path, expected, actual, format string, args ...any) LadderEvidenceAdmission {
		return LadderEvidenceAdmission{Verdict: Hold, Code: code, Path: path, ExpectedSHA256: expected, ActualSHA256: actual, Reason: fmt.Sprintf(format, args...)}
	}
	if strings.TrimSpace(dir) == "" || strings.TrimSpace(manifest) == "" {
		return hold("ladder_evidence_configuration_missing", "", "", "", "ladder evidence directory and checksum manifest are both required")
	}
	root, err := filepath.Abs(dir)
	if err != nil {
		return hold("ladder_evidence_directory_unreadable", "", "", "", "resolve ladder evidence directory: %v", err)
	}
	manifestPath := manifest
	if !filepath.IsAbs(manifestPath) {
		manifestPath = filepath.Join(root, filepath.FromSlash(manifestPath))
	}
	manifestPath, err = filepath.Abs(manifestPath)
	if err != nil || !withinDirectory(root, manifestPath) {
		return hold("ladder_evidence_path_traversal", filepath.ToSlash(manifest), "", "", "checksum manifest path %q escapes ladder evidence directory", manifest)
	}
	resolvedRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		return hold("ladder_evidence_directory_unreadable", "", "", "", "resolve ladder evidence directory: %v", err)
	}
	resolvedManifest, err := filepath.EvalSymlinks(manifestPath)
	if err != nil {
		return hold("ladder_evidence_manifest_unreadable", filepath.ToSlash(manifest), "", "", "read checksum manifest %q: %v", manifest, err)
	}
	if !withinDirectory(resolvedRoot, resolvedManifest) {
		return hold("ladder_evidence_path_traversal", filepath.ToSlash(manifest), "", "", "checksum manifest path %q escapes ladder evidence directory", manifest)
	}
	f, err := os.Open(manifestPath)
	if err != nil {
		return hold("ladder_evidence_manifest_unreadable", filepath.ToSlash(manifest), "", "", "read checksum manifest %q: %v", manifest, err)
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	var entries []ladderChecksum
	if err := dec.Decode(&entries); err != nil {
		return hold("ladder_evidence_manifest_invalid", filepath.ToSlash(manifest), "", "", "decode checksum manifest %q: %v", manifest, err)
	}
	var trailing any
	if err := dec.Decode(&trailing); err != io.EOF {
		return hold("ladder_evidence_manifest_invalid", filepath.ToSlash(manifest), "", "", "checksum manifest %q contains trailing JSON", manifest)
	}
	if len(entries) == 0 {
		return hold("ladder_evidence_manifest_empty", filepath.ToSlash(manifest), "", "", "checksum manifest %q contains no artifacts", manifest)
	}
	seen := make(map[string]struct{}, len(entries))
	for _, entry := range entries {
		rel := filepath.Clean(filepath.FromSlash(entry.File))
		normalized := filepath.ToSlash(rel)
		if entry.File == "" || filepath.IsAbs(rel) || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			return hold("ladder_evidence_path_traversal", filepath.ToSlash(entry.File), entry.SHA256, "", "manifest artifact path %q escapes ladder evidence directory", entry.File)
		}
		if _, ok := seen[normalized]; ok {
			return hold("ladder_evidence_duplicate_path", normalized, entry.SHA256, "", "checksum manifest repeats artifact path %q", normalized)
		}
		seen[normalized] = struct{}{}
	}
	for _, entry := range entries {
		rel := filepath.Clean(filepath.FromSlash(entry.File))
		normalized := filepath.ToSlash(rel)
		if len(entry.SHA256) != sha256.Size*2 {
			return hold("ladder_evidence_digest_invalid", normalized, entry.SHA256, "", "artifact %q has an invalid SHA-256 digest", normalized)
		}
		if _, err := hex.DecodeString(entry.SHA256); err != nil {
			return hold("ladder_evidence_digest_invalid", normalized, entry.SHA256, "", "artifact %q has an invalid SHA-256 digest", normalized)
		}
		path := filepath.Join(root, rel)
		if !withinDirectory(root, path) {
			return hold("ladder_evidence_path_traversal", normalized, entry.SHA256, "", "manifest artifact path %q escapes ladder evidence directory", normalized)
		}
		resolvedPath, pathErr := filepath.EvalSymlinks(path)
		if pathErr == nil && !withinDirectory(resolvedRoot, resolvedPath) {
			return hold("ladder_evidence_path_traversal", normalized, entry.SHA256, "", "manifest artifact path %q escapes ladder evidence directory", normalized)
		}
		info, err := os.Lstat(path)
		if err != nil {
			return hold("ladder_evidence_artifact_unreadable", normalized, strings.ToLower(entry.SHA256), "", "read ladder evidence artifact %q: %v", normalized, err)
		}
		if !info.Mode().IsRegular() {
			return hold("ladder_evidence_artifact_unreadable", normalized, strings.ToLower(entry.SHA256), "", "ladder evidence artifact %q is not a regular file", normalized)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return hold("ladder_evidence_artifact_unreadable", normalized, strings.ToLower(entry.SHA256), "", "read ladder evidence artifact %q: %v", normalized, err)
		}
		actual := fmt.Sprintf("%x", sha256.Sum256(data))
		expected := strings.ToLower(entry.SHA256)
		if actual != expected {
			return hold("ladder_evidence_checksum_mismatch", normalized, expected, actual, "ladder evidence checksum mismatch for %q: expected %s, actual %s", normalized, expected, actual)
		}
	}
	manifestRel, err := filepath.Rel(root, manifestPath)
	if err != nil {
		return hold("ladder_evidence_manifest_unreadable", filepath.ToSlash(manifest), "", "", "resolve checksum manifest %q: %v", manifest, err)
	}
	err = filepath.WalkDir(root, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			return nil
		}
		rel, relErr := filepath.Rel(root, path)
		if relErr != nil {
			return relErr
		}
		normalized := filepath.ToSlash(rel)
		if rel == manifestRel {
			return nil
		}
		if _, ok := seen[normalized]; !ok {
			return fmt.Errorf("%s", normalized)
		}
		return nil
	})
	if err != nil {
		return hold("ladder_evidence_unlisted_artifact", filepath.ToSlash(err.Error()), "", "", "ladder evidence artifact %q is omitted from checksum manifest", filepath.ToSlash(err.Error()))
	}
	return LadderEvidenceAdmission{Verdict: Pass}
}

func withinDirectory(root, path string) bool {
	rel, err := filepath.Rel(root, path)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}
