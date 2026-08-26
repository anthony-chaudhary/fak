package supervisionpolicy

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/flock"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// ErrFenced means another coordinator already advanced the durable epoch.
var ErrFenced = errors.New("coordinator epoch fenced")

// ProcessIdentity binds a child record to a run and process incarnation. Birth
// must come from an OS process-start identity (not a wall-clock observation).
type ProcessIdentity struct {
	RunID string `json:"run_id"`
	PID   int    `json:"pid"`
	Birth uint64 `json:"birth"`
}

func (p ProcessIdentity) valid() bool { return p.RunID != "" && p.PID > 0 && p.Birth != 0 }

// AdoptedChild is the durable control record transferred between coordinators.
// Restart is carried unchanged so takeover cannot reset the #9052 budget.
type AdoptedChild struct {
	ID                MemberID        `json:"id"`
	Identity          ProcessIdentity `json:"identity"`
	Owner             string          `json:"owner"`
	Epoch             uint64          `json:"epoch"`
	ReconnectDeadline time.Time       `json:"reconnect_deadline"`
	Progress          uint64          `json:"progress"`
	TerminalReceipt   EvidenceRef     `json:"terminal_receipt,omitempty"`
	Restart           ChildState      `json:"restart"`
	Quarantined       bool            `json:"quarantined,omitempty"`
}

// AdoptionState is the single durable fencing authority for one fleet.
type AdoptionState struct {
	Owner    string                    `json:"owner"`
	Epoch    uint64                    `json:"epoch"`
	Children map[MemberID]AdoptedChild `json:"children"`
}

type AdoptionDisposition string

const (
	AdoptionAdopted    AdoptionDisposition = "adopted"
	AdoptionReconciled AdoptionDisposition = "reconciled"
	AdoptionQuarantine AdoptionDisposition = "quarantined"
)

type AdoptionResult struct {
	Epoch    uint64                           `json:"epoch"`
	Children map[MemberID]AdoptionDisposition `json:"children"`
}

// IdentityVerifier independently checks that the recorded process incarnation
// is still live. Implementations must compare Birth as well as PID.
type IdentityVerifier func(ProcessIdentity) bool

// AdoptionStore serializes epoch changes through an atomic create lock and
// persists them with replace-by-rename. The lock path is deliberately separate
// from the state file so competing replacement coordinators cannot both win.
type AdoptionStore struct {
	Path        string
	LockTimeout time.Duration
}

func (s AdoptionStore) Load() (AdoptionState, error) {
	state := AdoptionState{Children: map[MemberID]AdoptedChild{}}
	data, err := os.ReadFile(s.Path)
	if errors.Is(err, os.ErrNotExist) {
		return state, nil
	}
	if err != nil {
		return state, err
	}
	if err := json.Unmarshal(data, &state); err != nil {
		return state, err
	}
	if state.Children == nil {
		state.Children = map[MemberID]AdoptedChild{}
	}
	return state, nil
}

// Initialize records the first coordinator and its children. Existing durable
// state is never overwritten, which makes accidental relaunch visible.
func (s AdoptionStore) Initialize(owner string, epoch uint64, children []AdoptedChild) error {
	return s.withLock(func() error {
		state, err := s.Load()
		if err != nil {
			return err
		}
		if state.Epoch != 0 {
			return ErrFenced
		}
		state.Owner, state.Epoch = owner, epoch
		for _, child := range children {
			if child.ID == "" || !child.Identity.valid() {
				return fmt.Errorf("invalid child identity for %q", child.ID)
			}
			child.Owner, child.Epoch = owner, epoch
			state.Children[child.ID] = child
		}
		return s.save(state)
	})
}

// Takeover performs discover -> durable epoch acquisition -> independent
// identity verification -> adopt/quarantine in one serialized transaction.
// Terminal receipts reconcile without relaunching or requiring a live PID.
func (s AdoptionStore) Takeover(owner string, expectedEpoch uint64, now time.Time, reconnectGrace time.Duration, verify IdentityVerifier) (AdoptionResult, error) {
	result := AdoptionResult{Children: map[MemberID]AdoptionDisposition{}}
	err := s.withLock(func() error {
		state, err := s.Load()
		if err != nil {
			return err
		}
		if owner == "" || state.Epoch != expectedEpoch {
			return ErrFenced
		}
		state.Epoch++
		state.Owner = owner
		result.Epoch = state.Epoch
		ids := make([]string, 0, len(state.Children))
		for id := range state.Children {
			ids = append(ids, string(id))
		}
		sort.Strings(ids)
		for _, raw := range ids {
			id := MemberID(raw)
			child := state.Children[id]
			switch {
			case child.TerminalReceipt != "":
				result.Children[id] = AdoptionReconciled
			case !child.Identity.valid() || verify == nil || !verify(child.Identity):
				child.Quarantined = true
				result.Children[id] = AdoptionQuarantine
			default:
				child.Owner = owner
				child.Epoch = state.Epoch
				child.ReconnectDeadline = now.Add(reconnectGrace)
				child.Quarantined = false
				result.Children[id] = AdoptionAdopted
			}
			state.Children[id] = child
		}
		return s.save(state)
	})
	return result, err
}

// CanCommand is the child-side fence check. A stale coordinator token stops
// authorizing control immediately after a replacement advances the epoch.
func (s AdoptionStore) CanCommand(owner string, epoch uint64, childID MemberID, now time.Time) (bool, error) {
	state, err := s.Load()
	if err != nil {
		return false, err
	}
	child, ok := state.Children[childID]
	if !ok || child.Quarantined || child.TerminalReceipt != "" {
		return false, nil
	}
	return state.Owner == owner && state.Epoch == epoch && child.Owner == owner && child.Epoch == epoch && !now.After(child.ReconnectDeadline), nil
}

func (s AdoptionStore) save(state AdoptionState) error {
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(s.Path), 0o755); err != nil {
		return err
	}
	tmp := s.Path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, s.Path)
}

func (s AdoptionStore) withLock(fn func() error) error {
	timeout := s.LockTimeout
	if timeout <= 0 {
		timeout = 2 * time.Second
	}
	deadline := time.Now().Add(timeout)
	lock, err := os.OpenFile(s.Path+".lock", os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return err
	}
	defer lock.Close()
	for {
		err = flock.TryLock(lock)
		if err == nil {
			defer flock.Unlock(lock)
			return fn()
		}
		if !errors.Is(err, flock.ErrLockBusy) {
			return err
		}
		if !time.Now().Before(deadline) {
			return fmt.Errorf("adoption lock timeout: %w", err)
		}
		time.Sleep(time.Millisecond)
	}
}
