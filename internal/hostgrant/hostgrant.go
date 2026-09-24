// Package hostgrant provides durable, host-local resource grants shared by
// independent fak processes.
package hostgrant

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

const schema = "fak.hostgrant/v1"

var (
	// ErrFull means the requested resources do not fit beside active grants.
	ErrFull = errors.New("hostgrant: capacity full")
	// ErrFenced means the supplied grant is absent or no longer names its live generation.
	ErrFenced = errors.New("hostgrant: stale or unknown grant")
	// ErrCapacityMismatch means this process configured a different capacity from the store.
	ErrCapacityMismatch = errors.New("hostgrant: capacity configuration mismatch")
)

// Vector is the independently accounted host resource demand of a grant.
type Vector struct {
	AgentSteps uint64 `json:"agent_steps"`
	Processes  uint64 `json:"processes"`
	Compilers  uint64 `json:"compilers"`
	ModelCalls uint64 `json:"model_calls"`
}

// Owner is the process identity currently responsible for a grant. StartedAt
// distinguishes PID reuse when callers obtain it from processalive.StartTime.
type Owner struct {
	ID        string    `json:"id"`
	PID       int       `json:"pid,omitempty"`
	StartedAt time.Time `json:"started_at,omitempty"`
}

// Request describes one idempotent admission attempt.
type Request struct {
	ID    string
	Owner Owner
	Cost  Vector
	TTL   time.Duration
}

// Grant is the fencing token returned by admission. ExpiresAt is evidence for
// reconciliation; expiry alone never releases or stops charging the grant.
type Grant struct {
	ID                 string    `json:"id"`
	Generation         uint64    `json:"generation"`
	Owner              Owner     `json:"owner"`
	Cost               Vector    `json:"cost"`
	ExpiresAt          time.Time `json:"expires_at"`
	UsesControlReserve bool      `json:"uses_control_reserve,omitempty"`
}

// Store is a durable host-local grant ledger. All mutations serialize through
// Path+".lock" and replace Path only after the new file has been flushed.
type Store struct {
	Path           string
	Capacity       Vector
	ControlReserve Vector
}

type diskState struct {
	Schema         string           `json:"schema"`
	Capacity       Vector           `json:"capacity"`
	ControlReserve Vector           `json:"control_reserve"`
	NextGeneration uint64           `json:"next_generation"`
	Grants         map[string]Grant `json:"grants"`
}

type admissionClass uint8

const (
	ordinaryAdmission admissionClass = iota
	trustedControlAdmission
)

// TryAcquire atomically reserves req.Cost. Repeating an active ID with the same
// owner and cost returns the existing grant, including after its advisory TTL.
func (s Store) TryAcquire(ctx context.Context, req Request) (Grant, error) {
	return s.tryAcquire(ctx, req, ordinaryAdmission)
}

// tryAcquire keeps the future trusted-control seam below the public package
// boundary. Current callers cannot choose a class: every Request is ordinary
// and therefore cannot spend ControlReserve.
func (s Store) tryAcquire(ctx context.Context, req Request, class admissionClass) (Grant, error) {
	if err := validateRequest(req); err != nil {
		return Grant{}, err
	}
	req.Owner = normalizeOwner(req.Owner)
	var out Grant
	err := s.withState(ctx, func(st *diskState) (bool, error) {
		usesControlReserve := class == trustedControlAdmission
		if current, ok := st.Grants[req.ID]; ok {
			if !sameOwner(current.Owner, req.Owner) || current.Cost != req.Cost || current.UsesControlReserve != usesControlReserve {
				return false, fmt.Errorf("%w: active id %q has different owner or cost", ErrFenced, req.ID)
			}
			out = current
			return false, nil
		}
		used, err := usedResources(st.Grants)
		if err != nil {
			return false, err
		}
		limit, err := admissionLimit(st, class)
		if err != nil {
			return false, err
		}
		charged := used.Ordinary
		if usesControlReserve {
			charged = used.Total
		}
		if !fits(limit, charged, req.Cost) || !fits(st.Capacity, used.Total, req.Cost) {
			return false, fmt.Errorf("%w: requested %+v, used %+v, admission capacity %+v", ErrFull, req.Cost, charged, limit)
		}
		generation, err := nextGeneration(st)
		if err != nil {
			return false, err
		}
		out = Grant{
			ID:                 req.ID,
			Generation:         generation,
			Owner:              req.Owner,
			Cost:               req.Cost,
			ExpiresAt:          time.Now().UTC().Add(req.TTL),
			UsesControlReserve: usesControlReserve,
		}
		st.Grants[req.ID] = out
		return true, nil
	})
	return out, err
}

// Transfer moves responsibility to owner and advances the fencing generation.
func (s Store) Transfer(ctx context.Context, grant Grant, owner Owner) (Grant, error) {
	if err := validateOwner(owner); err != nil {
		return Grant{}, err
	}
	owner = normalizeOwner(owner)
	var out Grant
	err := s.withState(ctx, func(st *diskState) (bool, error) {
		current, ok := st.Grants[grant.ID]
		if !ok || !sameGrant(current, grant) {
			return false, fmt.Errorf("%w: transfer %q generation %d", ErrFenced, grant.ID, grant.Generation)
		}
		generation, err := nextGeneration(st)
		if err != nil {
			return false, err
		}
		current.Generation = generation
		current.Owner = owner
		st.Grants[current.ID] = current
		out = current
		return true, nil
	})
	return out, err
}

// Release removes exactly the supplied generation. Missing or superseded
// generations are fenced so an old owner cannot release a transferred grant.
func (s Store) Release(ctx context.Context, grant Grant) error {
	return s.withState(ctx, func(st *diskState) (bool, error) {
		current, ok := st.Grants[grant.ID]
		if !ok || !sameGrant(current, grant) {
			return false, fmt.Errorf("%w: release %q generation %d", ErrFenced, grant.ID, grant.Generation)
		}
		delete(st.Grants, grant.ID)
		return true, nil
	})
}

func (s Store) withState(ctx context.Context, mutate func(*diskState) (bool, error)) error {
	if s.Path == "" {
		return errors.New("hostgrant: store path is required")
	}
	if err := validateReserve(s.Capacity, s.ControlReserve); err != nil {
		return err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := os.MkdirAll(filepath.Dir(s.Path), 0o700); err != nil {
		return fmt.Errorf("hostgrant: create store directory: %w", err)
	}
	lock, err := os.OpenFile(s.Path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return fmt.Errorf("hostgrant: open lock: %w", err)
	}
	defer lock.Close()
	for {
		err = flock.TryLock(lock)
		if err == nil {
			break
		}
		if !errors.Is(err, flock.ErrLockBusy) {
			return fmt.Errorf("hostgrant: acquire lock: %w", err)
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("hostgrant: acquire lock: %w", ctx.Err())
		case <-time.After(10 * time.Millisecond):
		}
	}
	defer flock.Unlock(lock)

	st, initialized, err := s.loadState()
	if err != nil {
		return err
	}
	if st.Capacity != s.Capacity || st.ControlReserve != s.ControlReserve {
		return fmt.Errorf("%w: configured capacity %+v reserve %+v, persisted capacity %+v reserve %+v", ErrCapacityMismatch, s.Capacity, s.ControlReserve, st.Capacity, st.ControlReserve)
	}
	changed, err := mutate(&st)
	if err != nil {
		if initialized {
			if writeErr := s.saveState(st); writeErr != nil {
				return errors.Join(err, writeErr)
			}
		}
		return err
	}
	if initialized || changed {
		return s.saveState(st)
	}
	return nil
}

func (s Store) loadState() (diskState, bool, error) {
	body, err := os.ReadFile(s.Path)
	if errors.Is(err, os.ErrNotExist) {
		return diskState{Schema: schema, Capacity: s.Capacity, ControlReserve: s.ControlReserve, NextGeneration: 1, Grants: map[string]Grant{}}, true, nil
	}
	if err != nil {
		return diskState{}, false, fmt.Errorf("hostgrant: read store: %w", err)
	}
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.DisallowUnknownFields()
	var st diskState
	if err := dec.Decode(&st); err != nil {
		return diskState{}, false, fmt.Errorf("hostgrant: decode store: %w", err)
	}
	var trailing any
	if err := dec.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			err = errors.New("trailing JSON value")
		}
		return diskState{}, false, fmt.Errorf("hostgrant: decode store: %w", err)
	}
	if st.Schema != schema {
		return diskState{}, false, fmt.Errorf("hostgrant: unsupported schema %q", st.Schema)
	}
	if st.NextGeneration == 0 {
		return diskState{}, false, errors.New("hostgrant: invalid zero next generation")
	}
	if err := validateReserve(st.Capacity, st.ControlReserve); err != nil {
		return diskState{}, false, fmt.Errorf("hostgrant: invalid persisted control reserve: %w", err)
	}
	if st.Grants == nil {
		st.Grants = map[string]Grant{}
	}
	seenGenerations := make(map[uint64]string, len(st.Grants))
	for id, grant := range st.Grants {
		if id == "" || grant.ID != id || grant.Generation == 0 {
			return diskState{}, false, fmt.Errorf("hostgrant: invalid persisted grant %q", id)
		}
		if grant.Generation >= st.NextGeneration {
			return diskState{}, false, fmt.Errorf("hostgrant: persisted grant %q generation %d is not below next generation %d", id, grant.Generation, st.NextGeneration)
		}
		if otherID, exists := seenGenerations[grant.Generation]; exists {
			return diskState{}, false, fmt.Errorf("hostgrant: persisted grants %q and %q share generation %d", otherID, id, grant.Generation)
		}
		seenGenerations[grant.Generation] = id
		if err := validateOwner(grant.Owner); err != nil {
			return diskState{}, false, fmt.Errorf("hostgrant: invalid persisted grant %q: %w", id, err)
		}
		if grant.Cost == (Vector{}) || grant.ExpiresAt.IsZero() {
			return diskState{}, false, fmt.Errorf("hostgrant: invalid persisted grant %q: zero cost or expiry", id)
		}
	}
	used, err := usedResources(st.Grants)
	if err != nil {
		return diskState{}, false, err
	}
	ordinaryCapacity, err := admissionLimit(&st, ordinaryAdmission)
	if err != nil {
		return diskState{}, false, err
	}
	if !fits(st.Capacity, Vector{}, used.Total) {
		return diskState{}, false, fmt.Errorf("hostgrant: persisted total usage %+v exceeds capacity %+v", used.Total, st.Capacity)
	}
	if !fits(ordinaryCapacity, Vector{}, used.Ordinary) {
		return diskState{}, false, fmt.Errorf("hostgrant: persisted ordinary usage %+v exceeds admission capacity %+v", used.Ordinary, ordinaryCapacity)
	}
	return st, false, nil
}

func (s Store) saveState(st diskState) error {
	body, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return fmt.Errorf("hostgrant: encode store: %w", err)
	}
	body = append(body, '\n')
	dir := filepath.Dir(s.Path)
	tmp, err := os.CreateTemp(dir, ".hostgrant-*.tmp")
	if err != nil {
		return fmt.Errorf("hostgrant: create temporary store: %w", err)
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
		return fmt.Errorf("hostgrant: secure temporary store: %w", err)
	}
	if _, err := tmp.Write(body); err != nil {
		return fmt.Errorf("hostgrant: write temporary store: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		return fmt.Errorf("hostgrant: flush temporary store: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("hostgrant: close temporary store: %w", err)
	}
	if err := os.Rename(tmpPath, s.Path); err != nil {
		return fmt.Errorf("hostgrant: install store: %w", err)
	}
	committed = true
	if parent, err := os.Open(dir); err == nil {
		_ = parent.Sync()
		_ = parent.Close()
	}
	return nil
}

func validateRequest(req Request) error {
	if req.ID == "" {
		return errors.New("hostgrant: request id is required")
	}
	if err := validateOwner(req.Owner); err != nil {
		return err
	}
	if req.Cost == (Vector{}) {
		return errors.New("hostgrant: request cost must charge at least one resource")
	}
	if req.TTL <= 0 {
		return errors.New("hostgrant: positive TTL is required")
	}
	return nil
}

func validateOwner(owner Owner) error {
	if owner.ID == "" {
		return errors.New("hostgrant: owner id is required")
	}
	if owner.PID <= 0 {
		return errors.New("hostgrant: owner PID must be positive")
	}
	if owner.StartedAt.IsZero() {
		return errors.New("hostgrant: owner process start time is required")
	}
	return nil
}

type resourceUsage struct {
	Total    Vector
	Ordinary Vector
}

func usedResources(grants map[string]Grant) (resourceUsage, error) {
	var used resourceUsage
	for _, grant := range grants {
		var ok bool
		used.Total, ok = add(used.Total, grant.Cost)
		if !ok {
			return resourceUsage{}, errors.New("hostgrant: persisted total resource usage overflows uint64")
		}
		if grant.UsesControlReserve {
			continue
		}
		used.Ordinary, ok = add(used.Ordinary, grant.Cost)
		if !ok {
			return resourceUsage{}, errors.New("hostgrant: persisted ordinary resource usage overflows uint64")
		}
	}
	return used, nil
}

func add(a, b Vector) (Vector, bool) {
	if math.MaxUint64-a.AgentSteps < b.AgentSteps || math.MaxUint64-a.Processes < b.Processes ||
		math.MaxUint64-a.Compilers < b.Compilers || math.MaxUint64-a.ModelCalls < b.ModelCalls {
		return Vector{}, false
	}
	return Vector{
		AgentSteps: a.AgentSteps + b.AgentSteps,
		Processes:  a.Processes + b.Processes,
		Compilers:  a.Compilers + b.Compilers,
		ModelCalls: a.ModelCalls + b.ModelCalls,
	}, true
}

func fits(capacity, used, requested Vector) bool {
	return used.AgentSteps <= capacity.AgentSteps && requested.AgentSteps <= capacity.AgentSteps-used.AgentSteps &&
		used.Processes <= capacity.Processes && requested.Processes <= capacity.Processes-used.Processes &&
		used.Compilers <= capacity.Compilers && requested.Compilers <= capacity.Compilers-used.Compilers &&
		used.ModelCalls <= capacity.ModelCalls && requested.ModelCalls <= capacity.ModelCalls-used.ModelCalls
}

func admissionLimit(st *diskState, class admissionClass) (Vector, error) {
	switch class {
	case ordinaryAdmission:
		return subtract(st.Capacity, st.ControlReserve)
	case trustedControlAdmission:
		return st.Capacity, nil
	default:
		return Vector{}, errors.New("hostgrant: invalid admission class")
	}
}

func validateReserve(capacity, reserve Vector) error {
	if _, err := subtract(capacity, reserve); err != nil {
		return fmt.Errorf("hostgrant: invalid control reserve: %w", err)
	}
	return nil
}

func subtract(capacity, reserve Vector) (Vector, error) {
	if reserve.AgentSteps > capacity.AgentSteps || reserve.Processes > capacity.Processes ||
		reserve.Compilers > capacity.Compilers || reserve.ModelCalls > capacity.ModelCalls {
		return Vector{}, errors.New("reserve exceeds capacity")
	}
	return Vector{
		AgentSteps: capacity.AgentSteps - reserve.AgentSteps,
		Processes:  capacity.Processes - reserve.Processes,
		Compilers:  capacity.Compilers - reserve.Compilers,
		ModelCalls: capacity.ModelCalls - reserve.ModelCalls,
	}, nil
}

func nextGeneration(st *diskState) (uint64, error) {
	if st.NextGeneration == 0 || st.NextGeneration == math.MaxUint64 {
		return 0, errors.New("hostgrant: generation space exhausted")
	}
	generation := st.NextGeneration
	st.NextGeneration++
	return generation, nil
}

func sameGrant(a, b Grant) bool {
	return a.ID == b.ID && a.Generation == b.Generation && sameOwner(a.Owner, b.Owner) && a.Cost == b.Cost && a.ExpiresAt.Equal(b.ExpiresAt) && a.UsesControlReserve == b.UsesControlReserve
}

func sameOwner(a, b Owner) bool {
	return a.ID == b.ID && a.PID == b.PID && a.StartedAt.Equal(b.StartedAt)
}

func normalizeOwner(owner Owner) Owner {
	if !owner.StartedAt.IsZero() {
		owner.StartedAt = owner.StartedAt.UTC()
	}
	return owner
}
