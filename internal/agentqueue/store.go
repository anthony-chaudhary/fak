package agentqueue

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

// Store persists one queue snapshot. Save acknowledges only after the new bytes
// have been flushed and atomically installed at Path.
type Store struct {
	Path string
}

// FileStore returns a Store backed by path.
func FileStore(path string) Store {
	return Store{Path: path}
}

// NewFileStore returns a Store backed by path.
func NewFileStore(path string) Store {
	return Store{Path: path}
}

var ErrGenerationConflict = errors.New("agentqueue: generation conflict")

// Reserve serializes the read-plan-write transition across processes. The
// expected generation is a compare-and-swap token, so a stale reconciler cannot
// reserve capacity from an observation another reconciler already changed.
func (s Store) Reserve(ctx context.Context, expectedGeneration string) (Receipt, Snapshot, error) {
	if s.Path == "" {
		return Receipt{}, Snapshot{}, errors.New("agentqueue: snapshot path is required")
	}
	lock, err := os.OpenFile(s.Path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return Receipt{}, Snapshot{}, fmt.Errorf("agentqueue: open reservation lock: %w", err)
	}
	defer lock.Close()
	for {
		err = flock.TryLock(lock)
		if err == nil {
			break
		}
		if !errors.Is(err, flock.ErrLockBusy) {
			return Receipt{}, Snapshot{}, fmt.Errorf("agentqueue: lock reservation: %w", err)
		}
		select {
		case <-ctx.Done():
			return Receipt{}, Snapshot{}, fmt.Errorf("agentqueue: reserve: %w", ctx.Err())
		case <-time.After(10 * time.Millisecond):
		}
	}
	defer flock.Unlock(lock)

	snapshot, err := s.Load()
	if err != nil {
		return Receipt{}, Snapshot{}, err
	}
	if snapshot.Generation != expectedGeneration {
		return Receipt{}, snapshot, fmt.Errorf("%w: expected %q, found %q", ErrGenerationConflict, expectedGeneration, snapshot.Generation)
	}
	receipt, err := Reconcile(snapshot)
	if err != nil {
		return Receipt{}, Snapshot{}, err
	}
	if len(receipt.Start) == 0 {
		return receipt, snapshot, nil
	}
	for _, start := range receipt.Start {
		snapshot.Attempts = append(snapshot.Attempts, Attempt{
			ID: start.IdempotencyKey, IntentID: start.IntentID, State: AttemptReserved,
		})
	}
	snapshot.Generation = nextGeneration(snapshot.Generation, receipt.Start)
	if err := s.Save(snapshot); err != nil {
		return Receipt{}, Snapshot{}, err
	}
	return receipt, snapshot, nil
}

func nextGeneration(current string, starts []StartAction) string {
	h := sha256.New()
	_, _ = h.Write([]byte(current))
	for _, start := range starts {
		_, _ = h.Write([]byte{0})
		_, _ = h.Write([]byte(start.IdempotencyKey))
	}
	return "gen:" + hex.EncodeToString(h.Sum(nil)[:16])
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

// ReconcileRestart loads the snapshot under the reservation lock, evaluates
// active attempts and leases for orphaned processes or stale leases, updates the
// snapshot, and atomically persists it to disk before releasing the lock.
func (s Store) ReconcileRestart(ctx context.Context, liveness ProcessLivenessChecker, opts RestartOptions) (RestartReconciliation, Snapshot, error) {
	if s.Path == "" {
		return RestartReconciliation{}, Snapshot{}, errors.New("agentqueue: snapshot path is required")
	}
	lock, err := os.OpenFile(s.Path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return RestartReconciliation{}, Snapshot{}, fmt.Errorf("agentqueue: open reservation lock: %w", err)
	}
	defer lock.Close()
	for {
		err = flock.TryLock(lock)
		if err == nil {
			break
		}
		if !errors.Is(err, flock.ErrLockBusy) {
			return RestartReconciliation{}, Snapshot{}, fmt.Errorf("agentqueue: lock reservation: %w", err)
		}
		select {
		case <-ctx.Done():
			return RestartReconciliation{}, Snapshot{}, fmt.Errorf("agentqueue: reconcile restart: %w", ctx.Err())
		case <-time.After(10 * time.Millisecond):
		}
	}
	defer flock.Unlock(lock)

	snapshot, err := s.Load()
	if err != nil {
		return RestartReconciliation{}, Snapshot{}, err
	}
	rec, updated, err := ReconcileRestart(snapshot, liveness, opts)
	if err != nil {
		return RestartReconciliation{}, snapshot, err
	}
	if err := s.Save(updated); err != nil {
		return RestartReconciliation{}, snapshot, err
	}
	return rec, updated, nil
}
