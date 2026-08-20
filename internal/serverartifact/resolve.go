// Package serverartifact resolves a local model file to digest-bound identity
// facts and verifies that the file has not changed before launch handoff.
package serverartifact

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	// DigestAlgorithmSHA256 is the only digest algorithm accepted by this leaf.
	DigestAlgorithmSHA256 = "sha256"
	// DefaultMaxSizeBytes admits artifacts up to one pebibyte, including the
	// one-tebibyte operating-envelope floor, while retaining a finite ceiling.
	DefaultMaxSizeBytes int64 = 1 << 50
)

var (
	ErrInvalidReference  = errors.New("invalid artifact reference")
	ErrDisallowedLink    = errors.New("artifact path contains a link")
	ErrNotRegular        = errors.New("artifact is not a regular file")
	ErrDigestMismatch    = errors.New("artifact digest mismatch")
	ErrSizeLimit         = errors.New("artifact exceeds size limit")
	ErrArtifactChanged   = errors.New("artifact changed after verification")
	ErrInvalidResolution = errors.New("invalid artifact resolution")
)

// Reference declares the local file and required SHA-256 digest. MaxSizeBytes
// defaults to DefaultMaxSizeBytes when zero.
type Reference struct {
	Path         string
	SHA256       string
	MaxSizeBytes int64
}

// Identity contains the canonical, non-secret facts suitable for a server
// receipt. Digest is lowercase hexadecimal.
type Identity struct {
	CanonicalPath   string `json:"canonical_path"`
	SizeBytes       int64  `json:"size_bytes"`
	DigestAlgorithm string `json:"digest_algorithm"`
	Digest          string `json:"digest"`
}

// Resolution seals an accepted Identity to the filesystem snapshot that
// produced it. Call VerifyUnchanged immediately before launch handoff.
type Resolution struct {
	identity Identity
	snapshot fileSnapshot
	file     *os.File
}

// Identity returns a copy of the receipt-ready artifact facts.
func (r Resolution) Identity() Identity {
	return r.identity
}

// Close releases the verified file handle. Callers should keep the resolution
// open through VerifyUnchanged and close it after the launch handoff.
func (r Resolution) Close() error {
	if r.file == nil {
		return nil
	}
	return r.file.Close()
}

// DigestMismatchError reports both sides of an explicit SHA-256 mismatch.
type DigestMismatchError struct {
	Expected string
	Actual   string
}

func (e *DigestMismatchError) Error() string {
	return fmt.Sprintf("%v: expected %s, got %s", ErrDigestMismatch, e.Expected, e.Actual)
}

// Is makes DigestMismatchError match ErrDigestMismatch with errors.Is.
func (e *DigestMismatchError) Is(target error) bool {
	return target == ErrDigestMismatch
}

// SizeLimitError reports the observed size and configured ceiling.
type SizeLimitError struct {
	SizeBytes int64
	MaxBytes  int64
}

func (e *SizeLimitError) Error() string {
	return fmt.Sprintf("%v: %d bytes exceeds %d", ErrSizeLimit, e.SizeBytes, e.MaxBytes)
}

// Is makes SizeLimitError match ErrSizeLimit with errors.Is.
func (e *SizeLimitError) Is(target error) bool {
	return target == ErrSizeLimit
}

type fileSnapshot struct {
	info    os.FileInfo
	size    int64
	mode    os.FileMode
	modTime time.Time
}

// Resolve verifies one local regular file without following symbolic links,
// streams its SHA-256 digest under ctx, and returns a sealed resolution only
// when every declared constraint holds.
func Resolve(ctx context.Context, ref Reference) (Resolution, error) {
	if err := ctx.Err(); err != nil {
		return Resolution{}, err
	}
	canonical, expected, maxBytes, err := normalizeReference(ref)
	if err != nil {
		return Resolution{}, err
	}

	pathInfo, err := lstatPath(canonical)
	if err != nil {
		return Resolution{}, err
	}
	if !pathInfo.Mode().IsRegular() {
		return Resolution{}, fmt.Errorf("%w: %s", ErrNotRegular, canonical)
	}
	if pathInfo.Size() > maxBytes {
		return Resolution{}, &SizeLimitError{SizeBytes: pathInfo.Size(), MaxBytes: maxBytes}
	}

	f, err := os.Open(canonical)
	if err != nil {
		return Resolution{}, fmt.Errorf("open artifact: %w", err)
	}
	accepted := false
	defer func() {
		if !accepted {
			_ = f.Close()
		}
	}()

	openedInfo, err := f.Stat()
	if err != nil {
		return Resolution{}, fmt.Errorf("stat opened artifact: %w", err)
	}
	initial := snapshot(pathInfo)
	if !sameSnapshot(initial, openedInfo) {
		return Resolution{}, fmt.Errorf("%w: path changed while opening %s", ErrArtifactChanged, canonical)
	}

	actual, size, err := streamSHA256(ctx, f, maxBytes)
	if err != nil {
		return Resolution{}, err
	}
	afterOpen, err := f.Stat()
	if err != nil {
		return Resolution{}, fmt.Errorf("restat opened artifact: %w", err)
	}
	afterPath, err := lstatPath(canonical)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) || errors.Is(err, ErrDisallowedLink) {
			return Resolution{}, fmt.Errorf("%w: path changed while hashing %s: %v", ErrArtifactChanged, canonical, err)
		}
		return Resolution{}, err
	}
	if size != initial.size || !sameSnapshot(initial, afterOpen) || !sameSnapshot(initial, afterPath) {
		return Resolution{}, fmt.Errorf("%w: file changed while hashing %s", ErrArtifactChanged, canonical)
	}
	if actual != expected {
		return Resolution{}, &DigestMismatchError{Expected: expected, Actual: actual}
	}

	accepted = true
	return Resolution{
		identity: Identity{
			CanonicalPath:   canonical,
			SizeBytes:       size,
			DigestAlgorithm: DigestAlgorithmSHA256,
			Digest:          actual,
		},
		snapshot: initial,
		file:     f,
	}, nil
}

// VerifyUnchanged re-stats every path component and fails if the verified file
// was replaced or its size, mode, or modification time changed.
func (r Resolution) VerifyUnchanged(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if r.identity.CanonicalPath == "" || r.snapshot.info == nil || r.file == nil {
		return ErrInvalidResolution
	}
	heldInfo, err := r.file.Stat()
	if err != nil {
		return fmt.Errorf("%w: verified handle: %v", ErrInvalidResolution, err)
	}
	if !sameSnapshot(r.snapshot, heldInfo) {
		return fmt.Errorf("%w: %s", ErrArtifactChanged, r.identity.CanonicalPath)
	}
	info, err := lstatPath(r.identity.CanonicalPath)
	if err != nil {
		return fmt.Errorf("%w: %s: %v", ErrArtifactChanged, r.identity.CanonicalPath, err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if !sameSnapshot(r.snapshot, info) {
		return fmt.Errorf("%w: %s", ErrArtifactChanged, r.identity.CanonicalPath)
	}
	return nil
}

func normalizeReference(ref Reference) (canonical, expected string, maxBytes int64, err error) {
	if strings.TrimSpace(ref.Path) == "" {
		return "", "", 0, fmt.Errorf("%w: path is required", ErrInvalidReference)
	}
	digestBytes, decodeErr := hex.DecodeString(ref.SHA256)
	if decodeErr != nil || len(digestBytes) != sha256.Size {
		return "", "", 0, fmt.Errorf("%w: SHA-256 must be 64 hexadecimal characters", ErrInvalidReference)
	}
	maxBytes = ref.MaxSizeBytes
	if maxBytes == 0 {
		maxBytes = DefaultMaxSizeBytes
	}
	if maxBytes < 0 {
		return "", "", 0, fmt.Errorf("%w: max size must not be negative", ErrInvalidReference)
	}
	canonical, err = filepath.Abs(ref.Path)
	if err != nil {
		return "", "", 0, fmt.Errorf("%w: canonicalize path: %v", ErrInvalidReference, err)
	}
	return filepath.Clean(canonical), hex.EncodeToString(digestBytes), maxBytes, nil
}

// lstatPath checks every component from the volume root down. Opening a path
// after this check is paired with os.SameFile in Resolve to close replacement
// races without relying on platform-specific no-follow flags.
func lstatPath(path string) (os.FileInfo, error) {
	var components []string
	for current := path; ; {
		parent := filepath.Dir(current)
		if parent == current {
			break
		}
		components = append(components, current)
		current = parent
	}
	for i := len(components) - 1; i >= 0; i-- {
		info, err := os.Lstat(components[i])
		if err != nil {
			return nil, fmt.Errorf("lstat artifact path %s: %w", components[i], err)
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return nil, fmt.Errorf("%w: %s", ErrDisallowedLink, components[i])
		}
		if i > 0 && !info.IsDir() {
			return nil, fmt.Errorf("%w: path component %s is not a directory", ErrNotRegular, components[i])
		}
		if i == 0 {
			return info, nil
		}
	}
	return nil, fmt.Errorf("%w: empty canonical path", ErrInvalidReference)
}

func snapshot(info os.FileInfo) fileSnapshot {
	return fileSnapshot{info: info, size: info.Size(), mode: info.Mode(), modTime: info.ModTime()}
}

func sameSnapshot(want fileSnapshot, got os.FileInfo) bool {
	return os.SameFile(want.info, got) && want.size == got.Size() && want.mode == got.Mode() && want.modTime.Equal(got.ModTime())
}

func streamSHA256(ctx context.Context, r io.Reader, maxBytes int64) (string, int64, error) {
	h := sha256.New()
	buf := make([]byte, 1024*1024)
	var size int64
	for {
		if err := ctx.Err(); err != nil {
			return "", 0, err
		}
		n, readErr := r.Read(buf)
		if n > 0 {
			if int64(n) > maxBytes-size {
				return "", 0, &SizeLimitError{SizeBytes: size + int64(n), MaxBytes: maxBytes}
			}
			if _, err := h.Write(buf[:n]); err != nil {
				return "", 0, fmt.Errorf("hash artifact: %w", err)
			}
			size += int64(n)
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				break
			}
			return "", 0, fmt.Errorf("read artifact: %w", readErr)
		}
	}
	return hex.EncodeToString(h.Sum(nil)), size, nil
}
