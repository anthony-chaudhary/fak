package claimcheck

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"hash"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Fingerprint returns a content-addressed key for a finding evaluated against
// the current working-tree bytes of scopedPaths. It deliberately reads files
// from disk rather than from git, so dirty and untracked content participates
// in the key. Every input is required: an incomplete key fails closed instead
// of permitting reuse against an under-specified state.
func Fingerprint(headSHA string, scopedPaths []string, rulesDigest, evaluatorDigest string) (string, error) {
	if strings.TrimSpace(headSHA) == "" {
		return "", fmt.Errorf("claimcheck fingerprint: head SHA is required")
	}
	if len(scopedPaths) == 0 {
		return "", fmt.Errorf("claimcheck fingerprint: at least one scoped path is required")
	}
	if strings.TrimSpace(rulesDigest) == "" {
		return "", fmt.Errorf("claimcheck fingerprint: rules digest is required")
	}
	if strings.TrimSpace(evaluatorDigest) == "" {
		return "", fmt.Errorf("claimcheck fingerprint: evaluator digest is required")
	}

	paths := make([]string, 0, len(scopedPaths))
	seen := make(map[string]struct{}, len(scopedPaths))
	for _, raw := range scopedPaths {
		if strings.TrimSpace(raw) == "" {
			return "", fmt.Errorf("claimcheck fingerprint: scoped path is empty")
		}
		clean := filepath.Clean(raw)
		key := filepath.ToSlash(clean)
		if _, ok := seen[key]; ok {
			return "", fmt.Errorf("claimcheck fingerprint: duplicate scoped path %q", key)
		}
		seen[key] = struct{}{}
		paths = append(paths, clean)
	}
	sort.Slice(paths, func(i, j int) bool {
		return filepath.ToSlash(paths[i]) < filepath.ToSlash(paths[j])
	})

	h := sha256.New()
	writeFingerprintField(h, "fak.claimcheck.finding-fingerprint.v1")
	writeFingerprintField(h, headSHA)
	writeFingerprintField(h, rulesDigest)
	writeFingerprintField(h, evaluatorDigest)
	for _, path := range paths {
		info, err := os.Stat(path)
		if err != nil {
			return "", fmt.Errorf("claimcheck fingerprint: stat scoped path %q: %w", path, err)
		}
		if !info.Mode().IsRegular() {
			return "", fmt.Errorf("claimcheck fingerprint: scoped path %q is not a regular file", path)
		}
		content, err := os.ReadFile(path)
		if err != nil {
			return "", fmt.Errorf("claimcheck fingerprint: read scoped path %q: %w", path, err)
		}
		contentDigest := sha256.Sum256(content)
		writeFingerprintField(h, filepath.ToSlash(path))
		writeFingerprintField(h, hex.EncodeToString(contentDigest[:]))
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func writeFingerprintField(h hash.Hash, value string) {
	var size [8]byte
	binary.BigEndian.PutUint64(size[:], uint64(len(value)))
	_, _ = h.Write(size[:])
	_, _ = h.Write([]byte(value))
}
