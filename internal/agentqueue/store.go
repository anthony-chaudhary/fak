package agentqueue

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// Store persists one queue snapshot. Save acknowledges only after the new bytes
// have been flushed and atomically installed at Path.
type Store struct {
	Path string
}

func (s Store) Load() (Snapshot, error) {
	body, err := os.ReadFile(s.Path)
	if err != nil {
		return Snapshot{}, fmt.Errorf("agentqueue: read snapshot: %w", err)
	}
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.DisallowUnknownFields()
	var snap Snapshot
	if err := dec.Decode(&snap); err != nil {
		return Snapshot{}, fmt.Errorf("agentqueue: decode snapshot: %w", err)
	}
	var trailing any
	if err := dec.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			err = errors.New("trailing JSON value")
		}
		return Snapshot{}, fmt.Errorf("agentqueue: decode snapshot: %w", err)
	}
	if snap.Schema != Schema {
		return Snapshot{}, fmt.Errorf("agentqueue: unsupported schema %q", snap.Schema)
	}
	if _, err := Reconcile(snap); err != nil {
		return Snapshot{}, fmt.Errorf("agentqueue: invalid snapshot: %w", err)
	}
	return snap, nil
}

func (s Store) Save(snapshot Snapshot) error {
	if s.Path == "" {
		return errors.New("agentqueue: snapshot path is required")
	}
	if snapshot.Schema == "" {
		snapshot.Schema = Schema
	}
	if snapshot.Schema != Schema {
		return fmt.Errorf("agentqueue: unsupported schema %q", snapshot.Schema)
	}
	if _, err := Reconcile(snapshot); err != nil {
		return fmt.Errorf("agentqueue: invalid snapshot: %w", err)
	}
	body, err := json.MarshalIndent(snapshot, "", "  ")
	if err != nil {
		return fmt.Errorf("agentqueue: encode snapshot: %w", err)
	}
	body = append(body, '\n')

	dir := filepath.Dir(s.Path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("agentqueue: create snapshot directory: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".agentqueue-*.tmp")
	if err != nil {
		return fmt.Errorf("agentqueue: create snapshot temp: %w", err)
	}
	tmpPath := tmp.Name()
	committed := false
	defer func() {
		_ = tmp.Close()
		if !committed {
			_ = os.Remove(tmpPath)
		}
	}()
	if err := tmp.Chmod(0o600); err != nil {
		return fmt.Errorf("agentqueue: secure snapshot temp: %w", err)
	}
	if _, err := tmp.Write(body); err != nil {
		return fmt.Errorf("agentqueue: write snapshot temp: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		return fmt.Errorf("agentqueue: sync snapshot temp: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("agentqueue: close snapshot temp: %w", err)
	}
	if err := os.Rename(tmpPath, s.Path); err != nil {
		return fmt.Errorf("agentqueue: commit snapshot: %w", err)
	}
	committed = true
	// POSIX needs the directory entry flushed as well. Some platforms do not
	// support directory Sync; the atomically installed file is still valid.
	if parent, err := os.Open(dir); err == nil {
		_ = parent.Sync()
		_ = parent.Close()
	}
	return nil
}
