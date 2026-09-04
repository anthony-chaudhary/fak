package laneadmit

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
)

var (
	// ErrLaneOccupied reports that the requested lane is held or contended by another session.
	ErrLaneOccupied = errors.New("lane is occupied")

	// ErrLaneNotHeld reports that the specified lane is not held by the session.
	ErrLaneNotHeld = errors.New("lane not held by session")

	// ErrInvalidHolder reports that the holder ID is empty.
	ErrInvalidHolder = errors.New("invalid holder id")

	// ErrInvalidLane reports that the lane name is empty or invalid.
	ErrInvalidLane = errors.New("invalid lane")
)

// TransitionResult records the outcome of a dynamic lane transition.
type TransitionResult struct {
	Success    bool      `json:"success"`
	FromLane   string    `json:"from_lane"`
	ToLane     string    `json:"to_lane"`
	Reason     string    `json:"reason,omitempty"`
	AcquiredAt time.Time `json:"acquired_at,omitempty"`
}

type leaseEntry struct {
	sessionID  string
	lane       string
	acquiredAt time.Time
}

// DynamicLeaseManager coordinates dynamic lane leases and atomic transitions
// between lanes for autonomous workers.
type DynamicLeaseManager struct {
	mu     sync.Mutex
	tax    Taxonomy
	leases map[string]leaseEntry // keyed by foldLane(lane)
}

// TransitionManager is an alias for DynamicLeaseManager.
type TransitionManager = DynamicLeaseManager

// NewDynamicLeaseManager creates an initialized DynamicLeaseManager.
func NewDynamicLeaseManager() *DynamicLeaseManager {
	return &DynamicLeaseManager{
		leases: make(map[string]leaseEntry),
	}
}

// NewTransitionManager creates an initialized TransitionManager.
func NewTransitionManager() *TransitionManager {
	return NewDynamicLeaseManager()
}

// WithTaxonomy sets the lane taxonomy used for exclusivity and tree overlap checks.
func (m *DynamicLeaseManager) WithTaxonomy(tax Taxonomy) *DynamicLeaseManager {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.tax = tax
	return m
}

// Acquire claims a lane for sessionID if it is free and disjoint from all other sessions.
func (m *DynamicLeaseManager) Acquire(sessionID string, lane string) error {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return ErrInvalidHolder
	}
	canon := CanonicalLane(lane)
	if canon == "" {
		return ErrInvalidLane
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	key := foldLane(canon)
	if existing, ok := m.leases[key]; ok && existing.sessionID == sessionID {
		return nil
	}

	if conflict, otherHolder, _ := m.checkConflictLocked(sessionID, canon); conflict {
		return fmt.Errorf("lane %q is occupied by session %q: %w", canon, otherHolder, ErrLaneOccupied)
	}

	m.leases[key] = leaseEntry{
		sessionID:  sessionID,
		lane:       canon,
		acquiredAt: time.Now(),
	}
	return nil
}

// Release releases a lane held by sessionID.
func (m *DynamicLeaseManager) Release(sessionID string, lane string) error {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return ErrInvalidHolder
	}
	canon := CanonicalLane(lane)
	if canon == "" {
		return ErrInvalidLane
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	key := foldLane(canon)
	entry, ok := m.leases[key]
	if !ok || entry.sessionID != sessionID {
		return fmt.Errorf("lane %q is not held by session %q: %w", canon, sessionID, ErrLaneNotHeld)
	}

	delete(m.leases, key)
	return nil
}

// TransitionLane atomically transitions sessionID from fromLane to toLane.
// If toLane is occupied by another session, the transition is refused with
// ReasonCollisionRisk and fromLane remains held (no drop).
// If fromLane was not held by sessionID, returns an error or refusal.
func (m *DynamicLeaseManager) TransitionLane(sessionID string, fromLane string, toLane string) (TransitionResult, error) {
	sessionID = strings.TrimSpace(sessionID)
	canonFrom := CanonicalLane(fromLane)
	canonTo := CanonicalLane(toLane)

	res := TransitionResult{
		FromLane: canonFrom,
		ToLane:   canonTo,
	}

	if sessionID == "" {
		res.Reason = ErrInvalidHolder.Error()
		return res, ErrInvalidHolder
	}
	if canonFrom == "" {
		res.Reason = ErrInvalidLane.Error()
		return res, ErrInvalidLane
	}
	if canonTo == "" {
		res.Reason = ErrInvalidLane.Error()
		return res, ErrInvalidLane
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	fromKey := foldLane(canonFrom)
	fromEntry, ok := m.leases[fromKey]
	if !ok || fromEntry.sessionID != sessionID {
		res.Reason = ErrLaneNotHeld.Error()
		return res, ErrLaneNotHeld
	}

	toKey := foldLane(canonTo)
	if fromKey == toKey {
		now := time.Now()
		m.leases[toKey] = leaseEntry{
			sessionID:  sessionID,
			lane:       canonTo,
			acquiredAt: now,
		}
		res.Success = true
		res.AcquiredAt = now
		return res, nil
	}

	// Check if toLane conflicts with any other session's held leases.
	if conflict, _, reason := m.checkConflictLocked(sessionID, canonTo); conflict {
		res.Success = false
		res.Reason = reason
		return res, nil
	}

	// Atomically release fromLane and assign toLane.
	now := time.Now()
	delete(m.leases, fromKey)
	m.leases[toKey] = leaseEntry{
		sessionID:  sessionID,
		lane:       canonTo,
		acquiredAt: now,
	}

	res.Success = true
	res.AcquiredAt = now
	return res, nil
}

// Holder returns the session ID holding lane, if any.
func (m *DynamicLeaseManager) Holder(lane string) (string, bool) {
	canon := CanonicalLane(lane)
	if canon == "" {
		return "", false
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	entry, ok := m.leases[foldLane(canon)]
	if !ok {
		return "", false
	}
	return entry.sessionID, true
}

// IsHeld reports whether lane is currently held by any session.
func (m *DynamicLeaseManager) IsHeld(lane string) bool {
	_, ok := m.Holder(lane)
	return ok
}

// HeldLanes returns all canonical lanes held by sessionID, sorted.
func (m *DynamicLeaseManager) HeldLanes(sessionID string) []string {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	var out []string
	for _, entry := range m.leases {
		if entry.sessionID == sessionID {
			out = append(out, entry.lane)
		}
	}
	sort.Strings(out)
	return out
}

func (m *DynamicLeaseManager) checkConflictLocked(sessionID string, targetLane string) (bool, string, string) {
	for _, entry := range m.leases {
		if entry.sessionID == sessionID {
			continue
		}

		// 1. Same lane or hierarchical containment conflict.
		if LanesConflict(entry.lane, targetLane) {
			return true, entry.sessionID, ReasonCollisionRisk
		}

		// 2. Taxonomy-based exclusivity and tree overlap.
		if m.tax.Loaded {
			if m.tax.IsExclusive(targetLane) || m.tax.IsExclusive(entry.lane) {
				return true, entry.sessionID, ReasonCollisionRisk
			}

			treeTarget := m.tax.TreeFor(targetLane)
			treeEntry := m.tax.TreeFor(entry.lane)
			if len(treeTarget) > 0 && len(treeEntry) > 0 && dispatchorder.TreesOverlap(treeTarget, treeEntry) {
				return true, entry.sessionID, ReasonCollisionRisk
			}
		}
	}
	return false, "", ""
}
