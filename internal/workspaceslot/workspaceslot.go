package workspaceslot

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

var (
	// ErrRingClosed is returned when operations are attempted on a closed SlotRing.
	ErrRingClosed = errors.New("workspaceslot: slot ring is closed")
	// ErrNoSlotsAvailable is returned by TryAcquire when all slots are currently leased.
	ErrNoSlotsAvailable = errors.New("workspaceslot: all slots are currently leased")
	// ErrInvalidSlot is returned when an unrecognized or nil slot is released.
	ErrInvalidSlot = errors.New("workspaceslot: invalid or unowned slot")
)

// Slot represents a single pre-allocated execution workspace slot.
type Slot struct {
	Index      int    `json:"index"`
	Name       string `json:"name"`
	Path       string `json:"path"`
	ScratchDir string `json:"scratch_dir"`
	RepoDir    string `json:"repo_dir"`
	SessionID  string `json:"session_id"`
	Leased     bool   `json:"leased"`
	BoundPIDs  []int  `json:"bound_pids"`
	mu         sync.Mutex
}

// BindPID binds an active child process PID to the slot so it will be terminated upon recycling.
func (s *Slot) BindPID(pid int) {
	if pid <= 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.BoundPIDs = append(s.BoundPIDs, pid)
}

// SlotRing manages a fixed-size ring buffer of pre-allocated workspace slots.
type SlotRing struct {
	baseDir  string
	capacity int
	slots    []*Slot
	freeCh   chan *Slot
	reaper   func(pid int) error
	mu       sync.Mutex
	closed   bool
}

// RingOption configures a SlotRing.
type RingOption func(*SlotRing)

// WithProcessReaper allows injecting a custom process tree reaper (defaults to toolprocgate.KillTree).
func WithProcessReaper(reaper func(pid int) error) RingOption {
	return func(r *SlotRing) {
		if reaper != nil {
			r.reaper = reaper
		}
	}
}

// NewSlotRing pre-allocates a fixed set of slots (slot-00 .. slot-K) under baseDir.
func NewSlotRing(baseDir string, capacity int, opts ...RingOption) (*SlotRing, error) {
	if capacity <= 0 {
		return nil, fmt.Errorf("workspaceslot: capacity must be > 0 (got %d)", capacity)
	}
	if baseDir == "" {
		return nil, errors.New("workspaceslot: baseDir cannot be empty")
	}

	if err := os.MkdirAll(baseDir, 0755); err != nil {
		return nil, fmt.Errorf("workspaceslot: create base dir %s: %w", baseDir, err)
	}

	ring := &SlotRing{
		baseDir:  baseDir,
		capacity: capacity,
		slots:    make([]*Slot, capacity),
		freeCh:   make(chan *Slot, capacity),
		reaper:   toolprocgate.KillTree,
	}

	for _, opt := range opts {
		opt(ring)
	}

	// Pre-allocate slots on disk once at initialization.
	for i := 0; i < capacity; i++ {
		name := fmt.Sprintf("slot-%02d", i)
		slotPath := filepath.Join(baseDir, name)
		scratchPath := filepath.Join(slotPath, "scratch")
		repoPath := filepath.Join(slotPath, "repo")

		if err := os.MkdirAll(scratchPath, 0755); err != nil {
			return nil, fmt.Errorf("workspaceslot: create scratch for %s: %w", name, err)
		}
		if err := os.MkdirAll(repoPath, 0755); err != nil {
			return nil, fmt.Errorf("workspaceslot: create repo for %s: %w", name, err)
		}

		slot := &Slot{
			Index:      i,
			Name:       name,
			Path:       slotPath,
			ScratchDir: scratchPath,
			RepoDir:    repoPath,
		}
		ring.slots[i] = slot
		ring.freeCh <- slot
	}

	return ring, nil
}

// BaseDir returns the root directory where pre-allocated slots reside.
func (r *SlotRing) BaseDir() string {
	return r.baseDir
}

// Capacity returns total slot count.
func (r *SlotRing) Capacity() int {
	return r.capacity
}

// AvailableCount returns the number of currently free slots.
func (r *SlotRing) AvailableCount() int {
	return len(r.freeCh)
}

// TryAcquire attempts to immediately acquire an available slot without blocking.
func (r *SlotRing) TryAcquire(sessionID string) (*Slot, error) {
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return nil, ErrRingClosed
	}
	r.mu.Unlock()

	select {
	case slot := <-r.freeCh:
		slot.mu.Lock()
		slot.Leased = true
		slot.SessionID = sessionID
		slot.BoundPIDs = nil
		slot.mu.Unlock()
		return slot, nil
	default:
		return nil, ErrNoSlotsAvailable
	}
}

// Acquire blocks until a slot is available or ctx is canceled.
func (r *SlotRing) Acquire(ctx context.Context, sessionID string) (*Slot, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return nil, ErrRingClosed
	}
	r.mu.Unlock()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case slot, ok := <-r.freeCh:
		if !ok {
			return nil, ErrRingClosed
		}
		slot.mu.Lock()
		slot.Leased = true
		slot.SessionID = sessionID
		slot.BoundPIDs = nil
		slot.mu.Unlock()
		return slot, nil
	}
}

// Release recycles a slot in-place and returns it to the free ring:
// 1. Kills all child processes bound to the session via toolprocgate.KillTree (or configured reaper).
// 2. High-speed deletion of scratch files inside slot.ScratchDir.
// 3. Cleans untracked/dirty files in slot.RepoDir (or git clean/reset if git repository exists).
// 4. Returns slot to available ring.
func (r *SlotRing) Release(slot *Slot) error {
	if slot == nil || slot.Index < 0 || slot.Index >= r.capacity {
		return ErrInvalidSlot
	}

	slot.mu.Lock()
	defer slot.mu.Unlock()

	if !slot.Leased {
		return nil // already released
	}

	// 1. Terminate all bound child processes.
	for _, pid := range slot.BoundPIDs {
		_ = r.reaper(pid)
	}
	slot.BoundPIDs = nil

	// 2. Fast in-place cleaning of ScratchDir.
	emptyDirectory(slot.ScratchDir)

	// 3. Fast in-place cleaning of RepoDir.
	cleanRepoDir(slot.RepoDir)

	slot.SessionID = ""
	slot.Leased = false

	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return ErrRingClosed
	}

	r.freeCh <- slot
	return nil
}

// Close closes the ring and cleans all slots.
func (r *SlotRing) Close() error {
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return nil
	}
	r.closed = true
	close(r.freeCh)
	r.mu.Unlock()

	// Terminate any live processes and clean directories
	for _, slot := range r.slots {
		slot.mu.Lock()
		for _, pid := range slot.BoundPIDs {
			_ = r.reaper(pid)
		}
		slot.BoundPIDs = nil
		emptyDirectory(slot.ScratchDir)
		emptyDirectory(slot.RepoDir)
		slot.mu.Unlock()
	}
	return nil
}

// emptyDirectory removes all entries inside dir while keeping dir itself.
func emptyDirectory(dir string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	for _, entry := range entries {
		path := filepath.Join(dir, entry.Name())
		_ = os.RemoveAll(path)
	}
}

// cleanRepoDir cleans untracked/dirty files in repoDir.
// If it's a git repo, runs git clean/checkout; otherwise empties untracked files.
func cleanRepoDir(repoDir string) {
	gitDir := filepath.Join(repoDir, ".git")
	if fi, err := os.Stat(gitDir); err == nil && (fi.IsDir() || !fi.IsDir()) {
		// Git repository present: reset and clean
		cmdClean := exec.Command("git", "-C", repoDir, "clean", "-ffxd")
		configureDispatchHelperCommand(cmdClean)
		_ = cmdClean.Run()
		cmdReset := exec.Command("git", "-C", repoDir, "reset", "--hard", "HEAD")
		configureDispatchHelperCommand(cmdReset)
		_ = cmdReset.Run()
		return
	}

	// Non-git: remove untracked artifacts inside repoDir
	emptyDirectory(repoDir)
}
