package orchestration

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"
)

const (
	runStateSchema           = "fak.orchestration.run-state/v1"
	runStateDir              = "runs"
	admissionStoreDir        = "admissions"
	RunStateAuthorityVersion = "fak.orchestration.authority/v1"
)

// RunStateManifest binds one run to its owner-controlled admission store.
// The admission-store locator is derived canonically from the run ID.
type RunStateManifest struct {
	Schema           string `json:"schema"`
	RunID            string `json:"run_id"`
	AdmissionStore   string `json:"admission_store"`
	CreatedAt        string `json:"created_at"`
	AuthorityVersion string `json:"authority_version"`
}

// RunStateLocator resolves durable run state beneath one operator-trusted root.
type RunStateLocator struct {
	root   string
	maxAge time.Duration
	now    func() time.Time
}

// NewRunStateLocator opens the trusted state root. maxAge bounds manifest
// authority so abandoned run bindings cannot silently remain authoritative.
func NewRunStateLocator(root string, maxAge time.Duration) (*RunStateLocator, error) {
	return newRunStateLocator(root, maxAge, time.Now)
}

func newRunStateLocator(root string, maxAge time.Duration, now func() time.Time) (*RunStateLocator, error) {
	root = strings.TrimSpace(root)
	if root == "" {
		return nil, errors.New("run state root is required")
	}
	if maxAge <= 0 {
		return nil, errors.New("run state max age must be positive")
	}
	if err := os.MkdirAll(root, 0o700); err != nil {
		return nil, fmt.Errorf("create run state root: %w", err)
	}
	if err := os.Chmod(root, 0o700); err != nil {
		return nil, fmt.Errorf("secure run state root: %w", err)
	}
	resolved, err := filepath.EvalSymlinks(root)
	if err != nil {
		return nil, fmt.Errorf("resolve run state root: %w", err)
	}
	resolved, err = filepath.Abs(resolved)
	if err != nil {
		return nil, fmt.Errorf("make run state root absolute: %w", err)
	}
	dir := filepath.Join(resolved, runStateDir)
	if err := os.Mkdir(dir, 0o700); err != nil && !errors.Is(err, os.ErrExist) {
		return nil, fmt.Errorf("create run manifest directory: %w", err)
	}
	if err := rejectLink(dir); err != nil {
		return nil, fmt.Errorf("secure run manifest directory: %w", err)
	}
	if err := os.Chmod(dir, 0o700); err != nil {
		return nil, fmt.Errorf("secure run manifest directory: %w", err)
	}
	return &RunStateLocator{root: resolved, maxAge: maxAge, now: now}, nil
}

// Create publishes the canonical run binding atomically. A valid existing
// binding is the immutable winner, independent of the caller's current clock.
func (l *RunStateLocator) Create(runID string) (RunStateManifest, error) {
	if err := validatePathComponent("run ID", runID); err != nil {
		return RunStateManifest{}, err
	}
	createdAt := l.now().UTC()
	manifest := RunStateManifest{
		Schema:           runStateSchema,
		RunID:            runID,
		AdmissionStore:   canonicalAdmissionStore(runID),
		CreatedAt:        createdAt.Format(time.RFC3339Nano),
		AuthorityVersion: RunStateAuthorityVersion,
	}
	if err := l.validate(manifest); err != nil {
		return RunStateManifest{}, err
	}
	data, err := encodeRunStateManifest(manifest)
	if err != nil {
		return RunStateManifest{}, err
	}
	if err := l.secureManifestDir(); err != nil {
		return RunStateManifest{}, err
	}
	path := l.manifestPath(runID)
	tmp, err := os.CreateTemp(filepath.Dir(path), ".run-state-*")
	if err != nil {
		return RunStateManifest{}, fmt.Errorf("create run manifest temporary file: %w", err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	failed := true
	defer func() {
		if failed {
			tmp.Close()
		}
	}()
	if err := tmp.Chmod(0o600); err != nil {
		return RunStateManifest{}, fmt.Errorf("secure run manifest temporary file: %w", err)
	}
	if _, err := tmp.Write(data); err != nil {
		return RunStateManifest{}, fmt.Errorf("write run manifest: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		return RunStateManifest{}, fmt.Errorf("sync run manifest: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return RunStateManifest{}, fmt.Errorf("close run manifest: %w", err)
	}
	failed = false
	if err := os.Link(tmpName, path); err == nil {
		return manifest, nil
	} else if !errors.Is(err, os.ErrExist) {
		return RunStateManifest{}, fmt.Errorf("publish run manifest: %w", err)
	}
	existingData, err := os.ReadFile(path)
	if err != nil {
		return RunStateManifest{}, fmt.Errorf("read existing run manifest: %w", err)
	}
	existing, err := decodeRunStateManifest(existingData)
	if err != nil {
		return RunStateManifest{}, fmt.Errorf("validate existing run manifest: %w", err)
	}
	if err := validateRunStateIdentity(existing, runID); err != nil {
		return RunStateManifest{}, fmt.Errorf("conflicting run manifest already exists: %w", err)
	}
	return existing, nil
}

// Open strictly reads and validates the authoritative manifest for runID.
func (l *RunStateLocator) Open(runID string) (RunStateManifest, error) {
	if err := validatePathComponent("run ID", runID); err != nil {
		return RunStateManifest{}, err
	}
	if err := l.secureManifestDir(); err != nil {
		return RunStateManifest{}, err
	}
	path := l.manifestPath(runID)
	if err := rejectLink(path); err != nil {
		return RunStateManifest{}, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return RunStateManifest{}, fmt.Errorf("read run manifest: %w", err)
	}
	manifest, err := decodeRunStateManifest(data)
	if err != nil {
		return RunStateManifest{}, err
	}
	if manifest.RunID != runID {
		return RunStateManifest{}, errors.New("run manifest identity mismatch")
	}
	if err := l.validate(manifest); err != nil {
		return RunStateManifest{}, err
	}
	return manifest, nil
}

// Resolve constructs the existing admission store solely from the trusted root
// and validated manifest. admissionMaxAge is the successor receipt freshness.
func (l *RunStateLocator) Resolve(runID string, admissionMaxAge time.Duration) (*EffectSuccessorStore, error) {
	manifest, err := l.Open(runID)
	if err != nil {
		return nil, err
	}
	admissionsRoot := filepath.Join(l.root, admissionStoreDir)
	if err := ensureContainedDirectory(l.root, admissionsRoot); err != nil {
		return nil, fmt.Errorf("resolve admission store root: %w", err)
	}
	storePath := filepath.Join(l.root, filepath.FromSlash(manifest.AdmissionStore))
	if err := ensureContainedDirectory(admissionsRoot, storePath); err != nil {
		return nil, fmt.Errorf("resolve admission store: %w", err)
	}
	return OpenEffectSuccessorStore(storePath, admissionMaxAge)
}

func (l *RunStateLocator) manifestPath(runID string) string {
	return filepath.Join(l.root, runStateDir, runID+".json")
}

func (l *RunStateLocator) secureManifestDir() error {
	dir := filepath.Join(l.root, runStateDir)
	if err := rejectLink(dir); err != nil {
		return fmt.Errorf("validate run manifest directory: %w", err)
	}
	info, err := os.Stat(dir)
	if err != nil {
		return fmt.Errorf("stat run manifest directory: %w", err)
	}
	if !info.IsDir() {
		return errors.New("run manifest directory is not a directory")
	}
	return nil
}
func (l *RunStateLocator) validate(m RunStateManifest) error {
	if err := validateRunStateIdentity(m, m.RunID); err != nil {
		return err
	}
	created, err := time.Parse(time.RFC3339Nano, m.CreatedAt)
	if err != nil || created.IsZero() || created.Location() != time.UTC || created.Format(time.RFC3339Nano) != m.CreatedAt {
		return errors.New("created time must be canonical UTC RFC3339Nano")
	}
	now := l.now().UTC()
	if created.After(now) {
		return errors.New("run manifest created time is in the future")
	}
	if now.Sub(created) > l.maxAge {
		return errors.New("run manifest is stale")
	}
	return nil
}

func validateRunStateIdentity(m RunStateManifest, runID string) error {
	if m.Schema != runStateSchema {
		return errors.New("unsupported run manifest schema")
	}
	if err := validatePathComponent("run ID", m.RunID); err != nil {
		return err
	}
	if m.RunID != runID {
		return errors.New("manifest run ID does not match requested run ID")
	}
	if m.AdmissionStore != canonicalAdmissionStore(runID) {
		return errors.New("admission store locator does not match run ID")
	}
	if m.AuthorityVersion != RunStateAuthorityVersion {
		return errors.New("unsupported authority version")
	}
	created, err := time.Parse(time.RFC3339Nano, m.CreatedAt)
	if err != nil || created.IsZero() || created.Location() != time.UTC || created.Format(time.RFC3339Nano) != m.CreatedAt {
		return errors.New("created time must be canonical UTC RFC3339Nano")
	}
	return nil
}

func canonicalAdmissionStore(runID string) string {
	return path.Join(admissionStoreDir, runID)
}

func validatePathComponent(name, value string) error {
	if value == "" || value != strings.TrimSpace(value) || value == "." || value == ".." {
		return fmt.Errorf("%s must be a canonical path component", name)
	}
	if filepath.IsAbs(value) || filepath.VolumeName(value) != "" || strings.ContainsAny(value, `/\\`) || filepath.Base(value) != value {
		return fmt.Errorf("%s must be one relative path component", name)
	}
	return nil
}

func encodeRunStateManifest(m RunStateManifest) ([]byte, error) {
	data, err := json.Marshal(m)
	if err != nil {
		return nil, fmt.Errorf("encode run manifest: %w", err)
	}
	return append(data, '\n'), nil
}

func decodeRunStateManifest(data []byte) (RunStateManifest, error) {
	var m RunStateManifest
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&m); err != nil {
		return m, fmt.Errorf("decode run manifest: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return m, errors.New("decode run manifest: trailing content")
	}
	return m, nil
}

func rejectLink(path string) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return errors.New("symbolic link is not allowed")
	}
	return nil
}

func ensureContainedDirectory(root, candidate string) error {
	if info, err := os.Lstat(candidate); err == nil {
		if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
			return errors.New("admission store must be a real directory")
		}
	} else if errors.Is(err, os.ErrNotExist) {
		if err := os.Mkdir(candidate, 0o700); err != nil && !errors.Is(err, os.ErrExist) {
			return err
		}
	} else {
		return err
	}
	resolved, err := filepath.EvalSymlinks(candidate)
	if err != nil {
		return err
	}
	resolved, err = filepath.Abs(resolved)
	if err != nil {
		return err
	}
	rel, err := filepath.Rel(root, resolved)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) || filepath.IsAbs(rel) {
		return errors.New("admission store escapes trusted state root")
	}
	return nil
}
