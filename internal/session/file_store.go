package session

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
)

const descriptorFileVersion = "fak.session-descriptors.v1"

// fileStoreFaultHook is a test-only crash-boundary seam. Production leaves it nil.
// Tests install it only in isolated helper processes, so concurrent callers never
// share mutable hook state.
var fileStoreFaultHook func(stage string)

func fileStoreBoundary(stage string) {
	if fileStoreFaultHook != nil {
		fileStoreFaultHook(stage)
	}
}

// FileStore persists Descriptor rows into one JSON file. It is the production
// DescriptorStore for the live session registry: Put/Delete rewrite the small
// descriptor index, while List reads the current file back. The file is an index
// of drive state only, not a transcript.
type FileStore struct {
	mu   sync.Mutex
	path string
}

// NewFileStore returns a DescriptorStore backed by path.
func NewFileStore(path string) *FileStore {
	return &FileStore{path: path}
}

// CorruptDescriptorFileError reports malformed or unsupported descriptor-index
// content. Callers may recover by quarantining the index: descriptors are a
// rebuildable projection of live session state, not the session transcript.
// Cause carries the normalized, privacy-safe corruption class so recovery
// observability never has to echo descriptor contents (#4658).
type CorruptDescriptorFileError struct {
	Cause RecoveryCause
	Err   error
}

func (e *CorruptDescriptorFileError) Error() string { return e.Err.Error() }
func (e *CorruptDescriptorFileError) Unwrap() error { return e.Err }

// IsCorruptDescriptorFile reports whether err means the descriptor index was
// readable but its contents could not be trusted.
func IsCorruptDescriptorFile(err error) bool {
	var target *CorruptDescriptorFileError
	return errors.As(err, &target)
}

func corruptDescriptorFileError(cause RecoveryCause, err error) error {
	return &CorruptDescriptorFileError{Cause: cause, Err: err}
}

type descriptorFile struct {
	Version     string       `json:"version"`
	Descriptors []Descriptor `json:"descriptors"`
}

// Put writes one descriptor keyed by ID, replacing any prior row for that ID.
// The cross-process lock orders writers: for the same ID, the last Put that
// acquires the lock wins, regardless of the descriptor's embedded Rev value.
func (s *FileStore) Put(d Descriptor) error {
	if d.ID == "" {
		return errBlankDescriptorID
	}
	if s == nil || s.path == "" {
		return registryError("descriptor file path must be non-empty")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	unlock, err := lockDescriptorFile(s.path)
	if err != nil {
		return err
	}
	defer unlock()
	if err := cleanupDescriptorTemps(s.path); err != nil {
		return err
	}
	byID, err := s.loadLocked()
	if err != nil {
		return err
	}
	byID[d.ID] = d
	return s.saveLocked(byID)
}

// Delete removes id from the file. Deleting a missing id is a no-op.
func (s *FileStore) Delete(id string) error {
	if s == nil || s.path == "" {
		return registryError("descriptor file path must be non-empty")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	unlock, err := lockDescriptorFile(s.path)
	if err != nil {
		return err
	}
	defer unlock()
	if err := cleanupDescriptorTemps(s.path); err != nil {
		return err
	}
	byID, err := s.loadLocked()
	if err != nil {
		return err
	}
	if _, ok := byID[id]; !ok {
		return nil
	}
	delete(byID, id)
	return s.saveLocked(byID)
}

// List returns every descriptor currently persisted in the file.
func (s *FileStore) List() ([]Descriptor, error) {
	if s == nil || s.path == "" {
		return nil, registryError("descriptor file path must be non-empty")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	byID, err := s.loadLocked()
	if err != nil {
		return nil, err
	}
	out := make([]Descriptor, 0, len(byID))
	for _, d := range byID {
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out, nil
}

func (s *FileStore) loadLocked() (map[string]Descriptor, error) {
	if s == nil || s.path == "" {
		return nil, registryError("descriptor file path must be non-empty")
	}
	b, err := readDescriptorFile(s.path)
	if errors.Is(err, os.ErrNotExist) {
		return map[string]Descriptor{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read session descriptor file: %w", err)
	}
	if len(bytes.TrimSpace(b)) == 0 {
		return map[string]Descriptor{}, nil
	}
	var doc descriptorFile
	if err := json.Unmarshal(b, &doc); err != nil {
		return nil, corruptDescriptorFileError(RecoveryCauseDecode, fmt.Errorf("decode session descriptor file: %w", err))
	}
	if doc.Version != descriptorFileVersion {
		return nil, corruptDescriptorFileError(RecoveryCauseVersion, fmt.Errorf("unsupported session descriptor file version %q", doc.Version))
	}
	byID := make(map[string]Descriptor, len(doc.Descriptors))
	for _, d := range doc.Descriptors {
		if d.ID == "" {
			return nil, corruptDescriptorFileError(RecoveryCauseBlankID, errBlankDescriptorID)
		}
		byID[d.ID] = d
	}
	return byID, nil
}

func (s *FileStore) saveLocked(byID map[string]Descriptor) error {
	if s == nil || s.path == "" {
		return registryError("descriptor file path must be non-empty")
	}
	descs := make([]Descriptor, 0, len(byID))
	for _, d := range byID {
		descs = append(descs, d)
	}
	sort.Slice(descs, func(i, j int) bool { return descs[i].ID < descs[j].ID })
	doc := descriptorFile{Version: descriptorFileVersion, Descriptors: descs}

	dir := filepath.Dir(s.path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create session descriptor dir: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".session-descriptors-*.tmp")
	if err != nil {
		return fmt.Errorf("create session descriptor temp file: %w", err)
	}
	tmpName := tmp.Name()
	committed := false
	defer func() {
		if !committed {
			_ = os.Remove(tmpName)
		}
	}()

	enc := json.NewEncoder(tmp)
	enc.SetIndent("", "  ")
	if err := enc.Encode(doc); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("encode session descriptor file: %w", err)
	}
	fileStoreBoundary("encode")
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("flush session descriptor file: %w", err)
	}
	fileStoreBoundary("flush")
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close session descriptor file: %w", err)
	}
	fileStoreBoundary("close")
	if err := replaceFile(tmpName, s.path); err != nil {
		return err
	}
	committed = true
	return nil
}
