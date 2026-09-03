package agentopt

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"sync"
	"time"
)

// MemoryCell represents an agent memory statement bound to file content provenance.
type MemoryCell struct {
	ID          string    `json:"id"`
	Statement   string    `json:"statement"`
	FilePath    string    `json:"file_path"`
	FileDigest  string    `json:"file_digest"`
	Stale       bool      `json:"stale"`
	LastChecked time.Time `json:"last_checked"`
}

// FileReaderFunc reads file bytes for a target path, returning an error if missing or unreadable.
type FileReaderFunc func(path string) ([]byte, error)

// ProvenanceMemoryStore maintains memory cells anchored to file content digests.
// It tracks modifications to underlying files and marks corresponding memory cells stale.
type ProvenanceMemoryStore struct {
	mu     sync.RWMutex
	cells  map[string]*MemoryCell
	order  []string
	nextID uint64
}

// NewProvenanceMemoryStore creates an empty ProvenanceMemoryStore.
func NewProvenanceMemoryStore() *ProvenanceMemoryStore {
	return &ProvenanceMemoryStore{
		cells: make(map[string]*MemoryCell),
		order: make([]string, 0),
	}
}

// ComputeContentDigest returns the hex-encoded SHA256 digest of content bytes.
func ComputeContentDigest(content []byte) string {
	h := sha256.Sum256(content)
	return hex.EncodeToString(h[:])
}

// StoreMemory registers a new memory statement bound to a file path and its current content.
func (s *ProvenanceMemoryStore) StoreMemory(statement, filePath, content string) *MemoryCell {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.nextID++
	id := fmt.Sprintf("mem-%06d", s.nextID)
	digest := ComputeContentDigest([]byte(content))

	cell := &MemoryCell{
		ID:          id,
		Statement:   statement,
		FilePath:    filePath,
		FileDigest:  digest,
		Stale:       false,
		LastChecked: time.Now(),
	}

	s.cells[id] = cell
	s.order = append(s.order, id)
	return cell
}

// InvalidateOnFileChange inspects memory cells bound to filePath and marks them stale if content changed.
// It returns the slice of IDs newly marked stale.
func (s *ProvenanceMemoryStore) InvalidateOnFileChange(filePath, newContent string) []string {
	s.mu.Lock()
	defer s.mu.Unlock()

	newDigest := ComputeContentDigest([]byte(newContent))
	now := time.Now()
	var invalidated []string

	for _, id := range s.order {
		cell := s.cells[id]
		if cell == nil || cell.FilePath != filePath {
			continue
		}
		cell.LastChecked = now
		if cell.FileDigest != newDigest {
			if !cell.Stale {
				cell.Stale = true
				invalidated = append(invalidated, cell.ID)
			}
		}
	}
	return invalidated
}

// InvalidateOnMutation is an alias for InvalidateOnFileChange to handle working tree modifications.
func (s *ProvenanceMemoryStore) InvalidateOnMutation(filePath, newContent string) []string {
	return s.InvalidateOnFileChange(filePath, newContent)
}

// InvalidateOnFileDeletion marks all memory cells associated with filePath as stale.
func (s *ProvenanceMemoryStore) InvalidateOnFileDeletion(filePath string) []string {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now()
	var invalidated []string

	for _, id := range s.order {
		cell := s.cells[id]
		if cell == nil || cell.FilePath != filePath {
			continue
		}
		cell.LastChecked = now
		if !cell.Stale {
			cell.Stale = true
			invalidated = append(invalidated, cell.ID)
		}
	}
	return invalidated
}

// CheckStaleness evaluates all memory cells against the provided file reader.
// Cells whose underlying files have changed digest or return a read error (e.g. deleted)
// are marked stale. Returns the list of IDs newly marked stale.
func (s *ProvenanceMemoryStore) CheckStaleness(fileReaderFn FileReaderFunc) []string {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now()
	var invalidated []string

	visitedPaths := make(map[string]struct {
		digest string
		err    error
	})

	for _, id := range s.order {
		cell := s.cells[id]
		if cell == nil {
			continue
		}

		stat, ok := visitedPaths[cell.FilePath]
		if !ok {
			content, err := fileReaderFn(cell.FilePath)
			if err != nil {
				stat = struct {
					digest string
					err    error
				}{digest: "", err: err}
			} else {
				stat = struct {
					digest string
					err    error
				}{digest: ComputeContentDigest(content), err: nil}
			}
			visitedPaths[cell.FilePath] = stat
		}

		cell.LastChecked = now
		if stat.err != nil || stat.digest != cell.FileDigest {
			if !cell.Stale {
				cell.Stale = true
				invalidated = append(invalidated, cell.ID)
			}
		}
	}

	return invalidated
}

// GetActiveMemories returns all memory cells that are not stale, in insertion order.
func (s *ProvenanceMemoryStore) GetActiveMemories() []MemoryCell {
	s.mu.RLock()
	defer s.mu.RUnlock()

	active := make([]MemoryCell, 0)
	for _, id := range s.order {
		cell := s.cells[id]
		if cell != nil && !cell.Stale {
			active = append(active, *cell)
		}
	}
	return active
}

// QueryMemories returns active memory cells matching the given query string in statement or file path.
// If query is empty, all active memories are returned.
func (s *ProvenanceMemoryStore) QueryMemories(query string) []MemoryCell {
	s.mu.RLock()
	defer s.mu.RUnlock()

	active := make([]MemoryCell, 0)
	q := strings.ToLower(strings.TrimSpace(query))

	for _, id := range s.order {
		cell := s.cells[id]
		if cell == nil || cell.Stale {
			continue
		}
		if q == "" ||
			strings.Contains(strings.ToLower(cell.Statement), q) ||
			strings.Contains(strings.ToLower(cell.FilePath), q) {
			active = append(active, *cell)
		}
	}
	return active
}

// GetAllMemories returns all stored memory cells regardless of staleness status.
func (s *ProvenanceMemoryStore) GetAllMemories() []MemoryCell {
	s.mu.RLock()
	defer s.mu.RUnlock()

	all := make([]MemoryCell, 0, len(s.order))
	for _, id := range s.order {
		if cell := s.cells[id]; cell != nil {
			all = append(all, *cell)
		}
	}
	return all
}

// GetMemory returns a copy of the memory cell with the specified ID if found.
func (s *ProvenanceMemoryStore) GetMemory(id string) (MemoryCell, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	cell, ok := s.cells[id]
	if !ok || cell == nil {
		return MemoryCell{}, false
	}
	return *cell, true
}
