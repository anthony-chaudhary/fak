package codexresume

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
)

const (
	ThreadProviderCodex         = "codex"
	WriterResourceHandleVersion = "writer-resource-v1"
	WriterResourceKind          = "codex-thread-writer-lock"
)

var (
	identityProviderPattern = regexp.MustCompile(`^[a-z][a-z0-9_-]*$`)
	canonicalUUIDPattern    = regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`)
)

// ThreadIdentity is the validated, provider-qualified identity of one persisted thread.
type ThreadIdentity struct {
	Provider string `json:"provider"`
	ID       string `json:"id"`
}

// NewThreadIdentity validates a provider-qualified thread identity.
func NewThreadIdentity(provider, id string) (ThreadIdentity, error) {
	identity := ThreadIdentity{Provider: provider, ID: id}
	if err := identity.Validate(); err != nil {
		return ThreadIdentity{}, err
	}
	return identity, nil
}

// NewCodexThreadIdentity validates a Codex thread UUID.
func NewCodexThreadIdentity(id string) (ThreadIdentity, error) {
	return NewThreadIdentity(ThreadProviderCodex, id)
}

// Validate rejects identities that are ambiguous or cannot be bound exactly.
func (identity ThreadIdentity) Validate() error {
	if !identityProviderPattern.MatchString(identity.Provider) {
		return fmt.Errorf("invalid thread identity provider %q", identity.Provider)
	}
	if !canonicalUUIDPattern.MatchString(identity.ID) {
		return fmt.Errorf("invalid thread identity UUID %q", identity.ID)
	}
	return nil
}

// WriterResourceHandle is a versioned identity for one canonical writer-lock resource.
type WriterResourceHandle struct {
	Version    string         `json:"version"`
	Kind       string         `json:"kind"`
	ResourceID string         `json:"resource_id"`
	Thread     ThreadIdentity `json:"thread"`
	LockPath   string         `json:"lock_path"`
}

// NewWriterResourceHandle binds a validated thread to one canonical lock resource.
func NewWriterResourceHandle(thread ThreadIdentity, lockPath string) (WriterResourceHandle, error) {
	if err := thread.Validate(); err != nil {
		return WriterResourceHandle{}, err
	}
	canonical, err := canonicalWriterLockPath(lockPath)
	if err != nil {
		return WriterResourceHandle{}, err
	}
	handle := WriterResourceHandle{
		Version:  WriterResourceHandleVersion,
		Kind:     WriterResourceKind,
		Thread:   thread,
		LockPath: canonical,
	}
	handle.ResourceID = writerResourceID(handle.Version, handle.Kind, handle.Thread, handle.LockPath)
	return handle, nil
}

// Validate detects unsupported versions and any identity/resource mismatch.
func (handle WriterResourceHandle) Validate() error {
	if handle.Version != WriterResourceHandleVersion {
		return fmt.Errorf("unsupported writer resource handle version %q", handle.Version)
	}
	if handle.Kind != WriterResourceKind {
		return fmt.Errorf("unsupported writer resource kind %q", handle.Kind)
	}
	if err := handle.Thread.Validate(); err != nil {
		return err
	}
	canonical, err := canonicalWriterLockPath(handle.LockPath)
	if err != nil {
		return err
	}
	if canonical != handle.LockPath {
		return errors.New("writer resource lock path is not canonical")
	}
	expected := writerResourceID(handle.Version, handle.Kind, handle.Thread, handle.LockPath)
	if handle.ResourceID != expected {
		return errors.New("writer resource ID does not match its thread identity and lock resource")
	}
	return nil
}

func canonicalWriterLockPath(lockPath string) (string, error) {
	if strings.TrimSpace(lockPath) == "" {
		return "", errors.New("writer lock path is required")
	}
	absolute, err := filepath.Abs(lockPath)
	if err != nil {
		return "", fmt.Errorf("canonicalize writer lock path: %w", err)
	}
	canonical := filepath.Clean(absolute)
	if runtime.GOOS == "windows" {
		canonical = strings.ToLower(canonical)
	}
	return canonical, nil
}

func writerResourceID(version, kind string, thread ThreadIdentity, lockPath string) string {
	digest := sha256.Sum256([]byte(strings.Join([]string{version, kind, thread.Provider, thread.ID, lockPath}, "\x00")))
	return "writer-resource-v1:" + hex.EncodeToString(digest[:16])
}
