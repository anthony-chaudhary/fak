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

type descriptorFile struct {
	Version     string       `json:"version"`
	Descriptors []Descriptor `json:"descriptors"`
}

// Put writes one descriptor keyed by ID, replacing any prior row for that ID.
func (s *FileStore) Put(d Descriptor) error {
	if d.ID == "" {
		return errBlankDescriptorID
	}
	if s == nil || s.path == "" {
		return registryError("descriptor file path must be non-empty")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
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
	b, err := os.ReadFile(s.path)
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
		return nil, fmt.Errorf("decode session descriptor file: %w", err)
	}
	if doc.Version != descriptorFileVersion {
		return nil, fmt.Errorf("unsupported session descriptor file version %q", doc.Version)
	}
	byID := make(map[string]Descriptor, len(doc.Descriptors))
	for _, d := range doc.Descriptors {
		if d.ID == "" {
			return nil, errBlankDescriptorID
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
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close session descriptor file: %w", err)
	}
	if err := replaceFile(tmpName, s.path); err != nil {
		return err
	}
	committed = true
	return nil
}

func replaceFile(tmpName, path string) error {
	if err := os.Rename(tmpName, path); err == nil {
		return nil
	}
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("replace session descriptor file: %w", err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("replace session descriptor file: %w", err)
	}
	return nil
}
