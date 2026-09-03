// Package launchguard provides a host-local, cross-process circuit breaker for
// supervisors that launch detached agents or services.
package launchguard

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"syscall"
	"time"
)

// Outcome is the typed result of an admission attempt.
type Outcome string

const (
	Admitted        Outcome = "admitted"
	DuplicateActive Outcome = "duplicate-active"
	Backoff         Outcome = "backoff"
	Quarantined     Outcome = "quarantined"
	StaleRecovered  Outcome = "stale-recovered"
)

// Clock and Jitter make timing behavior deterministic in tests. Jitter receives
// the unjittered delay and must return a non-negative delay.
type Clock func() time.Time
type Jitter func(time.Duration) time.Duration

// Config defines one host-local launch budget store.
type Config struct {
	Dir         string
	MaxAttempts int
	Window      time.Duration
	BaseBackoff time.Duration
	MaxBackoff  time.Duration
	StaleAfter  time.Duration
	Cooldown    time.Duration
	Clock       Clock
	Jitter      Jitter
	PIDAlive    func(int) bool
	PID         int
}

// Guard owns launch budget state rooted at Config.Dir.
type Guard struct{ cfg Config }

// Decision describes an admission result without exposing the unhashed identity.
type Decision struct {
	Outcome    Outcome
	Identity   string
	RetryAfter time.Duration
	Status     Status
}

// Status is the operator-readable state for one stable identity.
type Status struct {
	Identity      string
	Attempts      int
	MaxAttempts   int
	WindowStart   time.Time
	LastFailure   time.Time
	LastSuccess   time.Time
	CooldownUntil time.Time
	Active        bool
	OwnerPID      int
	Quarantined   bool
	QuarantinedAt time.Time
}

// Lease represents an admitted launch. Finish must be called exactly once.
type Lease struct {
	guard    *Guard
	identity string
	token    string
	finished bool
}

type diskState struct {
	Attempts      []time.Time `json:"attempts,omitempty"`
	LastFailure   time.Time   `json:"last_failure,omitempty"`
	LastSuccess   time.Time   `json:"last_success,omitempty"`
	Quarantined   bool        `json:"quarantined,omitempty"`
	QuarantinedAt time.Time   `json:"quarantined_at,omitempty"`
}

type owner struct {
	PID       int       `json:"pid"`
	Token     string    `json:"token"`
	CreatedAt time.Time `json:"created_at"`
}

// New validates cfg and creates its state directory.
func New(cfg Config) (*Guard, error) {
	if cfg.Dir == "" || cfg.MaxAttempts <= 0 || cfg.Window <= 0 || cfg.BaseBackoff < 0 || cfg.StaleAfter <= 0 || cfg.Cooldown < 0 {
		return nil, errors.New("launchguard: dir, positive max attempts/window/stale age, and non-negative backoff/cooldown are required")
	}
	if cfg.MaxBackoff <= 0 {
		cfg.MaxBackoff = cfg.Window
	}
	if cfg.Clock == nil {
		cfg.Clock = time.Now
	}
	if cfg.Jitter == nil {
		cfg.Jitter = func(d time.Duration) time.Duration { return d }
	}
	if cfg.PIDAlive == nil {
		cfg.PIDAlive = processAlive
	}
	if cfg.PID == 0 {
		cfg.PID = os.Getpid()
	}
	if err := os.MkdirAll(cfg.Dir, 0o700); err != nil {
		return nil, fmt.Errorf("launchguard: create state directory: %w", err)
	}
	return &Guard{cfg: cfg}, nil
}

// StableIdentity returns only a SHA-256 digest of the supplied stable identity.
func StableIdentity(identity string) string {
	sum := sha256.Sum256([]byte(identity))
	return hex.EncodeToString(sum[:])
}

// Admit atomically admits one launch or returns a typed refusal. The caller must
// call Finish on a non-nil lease.
func (g *Guard) Admit(identity string) (Decision, *Lease, error) {
	id := StableIdentity(identity)
	now := g.cfg.Clock()
	recovered := false
	for attempts := 0; attempts < 3; attempts++ {
		o, err := g.createOwner(id, now)
		if err == nil {
			state, err := g.readState(id)
			if err != nil {
				g.releaseOwner(id, o.Token)
				return Decision{}, nil, err
			}
			var cooldownCutoff time.Time
			if g.cfg.Cooldown > 0 {
				cooldownCutoff = now.Add(-g.cfg.Cooldown)
			}
			state.prune(now.Add(-g.cfg.Window), cooldownCutoff)
			status := g.status(id, state, &o)
			if state.Quarantined {
				g.releaseOwner(id, o.Token)
				return Decision{Outcome: Quarantined, Identity: id, Status: g.status(id, state, nil)}, nil, nil
			}
			if g.cfg.Cooldown > 0 && !state.LastSuccess.IsZero() {
				if elapsed := now.Sub(state.LastSuccess); elapsed < g.cfg.Cooldown {
					g.releaseOwner(id, o.Token)
					retryAfter := g.cfg.Cooldown - elapsed
					return Decision{Outcome: DuplicateActive, Identity: id, RetryAfter: retryAfter, Status: g.status(id, state, nil)}, nil, nil
				}
			}
			if delay := g.backoff(state, now); delay > 0 {
				g.releaseOwner(id, o.Token)
				return Decision{Outcome: Backoff, Identity: id, RetryAfter: delay, Status: g.status(id, state, nil)}, nil, nil
			}
			if len(state.Attempts) >= g.cfg.MaxAttempts {
				state.Quarantined = true
				state.QuarantinedAt = now
				if err := g.writeState(id, state); err != nil {
					g.releaseOwner(id, o.Token)
					return Decision{}, nil, err
				}
				g.releaseOwner(id, o.Token)
				return Decision{Outcome: Quarantined, Identity: id, Status: g.status(id, state, nil)}, nil, nil
			}
			state.Attempts = append(state.Attempts, now)
			if err := g.writeState(id, state); err != nil {
				g.releaseOwner(id, o.Token)
				return Decision{}, nil, err
			}
			outcome := Admitted
			if recovered {
				outcome = StaleRecovered
			}
			status = g.status(id, state, &o)
			return Decision{Outcome: outcome, Identity: id, Status: status}, &Lease{guard: g, identity: id, token: o.Token}, nil
		}
		if !errors.Is(err, fs.ErrExist) {
			return Decision{}, nil, fmt.Errorf("launchguard: create owner: %w", err)
		}
		existing, stat, readErr := g.readOwner(id)
		if readErr != nil {
			return Decision{}, nil, readErr
		}
		age := now.Sub(existing.CreatedAt)
		if existing.CreatedAt.IsZero() {
			age = now.Sub(stat.ModTime())
		}
		if g.cfg.PIDAlive(existing.PID) || age < g.cfg.StaleAfter {
			state, stateErr := g.readState(id)
			if stateErr != nil {
				return Decision{}, nil, stateErr
			}
			var retryAfter time.Duration
			if age < g.cfg.StaleAfter {
				retryAfter = g.cfg.StaleAfter - age
			}
			return Decision{Outcome: DuplicateActive, Identity: id, RetryAfter: retryAfter, Status: g.status(id, state, &existing)}, nil, nil
		}
		if err := os.Remove(g.ownerPath(id)); err != nil && !errors.Is(err, fs.ErrNotExist) {
			return Decision{}, nil, fmt.Errorf("launchguard: recover stale owner: %w", err)
		}
		recovered = true
	}
	return Decision{}, nil, errors.New("launchguard: owner changed repeatedly during admission")
}

// Finish records success or failure and releases the active owner. Success
// clears the rolling attempt budget; failure starts backoff and may quarantine.
func (l *Lease) Finish(success bool) error {
	if l == nil || l.guard == nil {
		return errors.New("launchguard: nil lease")
	}
	if l.finished {
		return errors.New("launchguard: lease already finished")
	}
	l.finished = true
	g := l.guard
	state, err := g.readState(l.identity)
	if err != nil {
		return err
	}
	now := g.cfg.Clock()
	var cooldownCutoff time.Time
	if g.cfg.Cooldown > 0 {
		cooldownCutoff = now.Add(-g.cfg.Cooldown)
	}
	state.prune(now.Add(-g.cfg.Window), cooldownCutoff)
	if success {
		if g.cfg.Cooldown > 0 {
			state = diskState{LastSuccess: now}
		} else {
			state = diskState{}
		}
	} else {
		state.LastFailure = now
		if len(state.Attempts) >= g.cfg.MaxAttempts {
			state.Quarantined = true
			state.QuarantinedAt = now
		}
	}
	if err := g.writeState(l.identity, state); err != nil {
		return err
	}
	return g.releaseOwner(l.identity, l.token)
}

// Inspect returns current state without changing admission.
func (g *Guard) Inspect(identity string) (Status, error) {
	id := StableIdentity(identity)
	state, err := g.readState(id)
	if err != nil {
		return Status{}, err
	}
	now := g.cfg.Clock()
	var cooldownCutoff time.Time
	if g.cfg.Cooldown > 0 {
		cooldownCutoff = now.Add(-g.cfg.Cooldown)
	}
	state.prune(now.Add(-g.cfg.Window), cooldownCutoff)
	var active *owner
	if o, _, err := g.readOwner(id); err == nil {
		active = &o
	} else if !errors.Is(err, fs.ErrNotExist) {
		return Status{}, err
	}
	return g.status(id, state, active), nil
}

// Reset explicitly clears a non-active identity's budget and quarantine.
func (g *Guard) Reset(identity string) error {
	id := StableIdentity(identity)
	now := g.cfg.Clock()
	o, err := g.createOwner(id, now)
	if errors.Is(err, fs.ErrExist) {
		return errors.New("launchguard: cannot reset an active identity")
	}
	if err != nil {
		return fmt.Errorf("launchguard: acquire reset owner: %w", err)
	}
	defer g.releaseOwner(id, o.Token)
	return g.writeState(id, diskState{})
}

func (g *Guard) backoff(state diskState, now time.Time) time.Duration {
	if state.LastFailure.IsZero() || g.cfg.BaseBackoff == 0 {
		return 0
	}
	delay := g.cfg.BaseBackoff
	for i := 1; i < len(state.Attempts) && delay < g.cfg.MaxBackoff; i++ {
		if delay > g.cfg.MaxBackoff/2 {
			delay = g.cfg.MaxBackoff
			break
		}
		delay *= 2
	}
	if delay > g.cfg.MaxBackoff {
		delay = g.cfg.MaxBackoff
	}
	delay = g.cfg.Jitter(delay)
	if delay < 0 {
		delay = 0
	}
	remaining := state.LastFailure.Add(delay).Sub(now)
	if remaining < 0 {
		return 0
	}
	return remaining
}

func (s *diskState) prune(cutoff, cooldownCutoff time.Time) {
	first := 0
	for first < len(s.Attempts) && s.Attempts[first].Before(cutoff) {
		first++
	}
	s.Attempts = append([]time.Time(nil), s.Attempts[first:]...)
	if len(s.Attempts) == 0 && !s.Quarantined {
		s.LastFailure = time.Time{}
	}
	if !s.LastSuccess.IsZero() && !cooldownCutoff.IsZero() && s.LastSuccess.Before(cooldownCutoff) {
		s.LastSuccess = time.Time{}
	}
}

func (g *Guard) status(id string, state diskState, o *owner) Status {
	s := Status{
		Identity:      id,
		Attempts:      len(state.Attempts),
		MaxAttempts:   g.cfg.MaxAttempts,
		LastFailure:   state.LastFailure,
		LastSuccess:   state.LastSuccess,
		Quarantined:   state.Quarantined,
		QuarantinedAt: state.QuarantinedAt,
	}
	if len(state.Attempts) > 0 {
		s.WindowStart = state.Attempts[0]
	}
	if g.cfg.Cooldown > 0 && !state.LastSuccess.IsZero() {
		s.CooldownUntil = state.LastSuccess.Add(g.cfg.Cooldown)
	}
	if o != nil {
		s.Active = true
		s.OwnerPID = o.PID
	}
	return s
}

func (g *Guard) statePath(id string) string { return filepath.Join(g.cfg.Dir, id+".json") }
func (g *Guard) ownerPath(id string) string { return filepath.Join(g.cfg.Dir, id+".owner") }

func (g *Guard) createOwner(id string, now time.Time) (owner, error) {
	tokenBytes := make([]byte, 16)
	if _, err := rand.Read(tokenBytes); err != nil {
		return owner{}, fmt.Errorf("launchguard: owner token: %w", err)
	}
	o := owner{PID: g.cfg.PID, Token: hex.EncodeToString(tokenBytes), CreatedAt: now}
	f, err := os.OpenFile(g.ownerPath(id), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return owner{}, err
	}
	encErr := json.NewEncoder(f).Encode(o)
	syncErr := f.Sync()
	closeErr := f.Close()
	if encErr != nil || syncErr != nil || closeErr != nil {
		_ = os.Remove(g.ownerPath(id))
		return owner{}, errors.Join(encErr, syncErr, closeErr)
	}
	return o, nil
}

func (g *Guard) readOwner(id string) (owner, fs.FileInfo, error) {
	f, err := os.Open(g.ownerPath(id))
	if err != nil {
		return owner{}, nil, err
	}
	defer f.Close()
	stat, err := f.Stat()
	if err != nil {
		return owner{}, nil, err
	}
	var o owner
	if err := json.NewDecoder(f).Decode(&o); err != nil {
		// A competing creator may still be writing. Treat it as active and use
		// the file timestamp for stale-age decisions.
		return owner{}, stat, nil
	}
	return o, stat, nil
}

func (g *Guard) releaseOwner(id, token string) error {
	o, _, err := g.readOwner(id)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if o.Token != token {
		return errors.New("launchguard: owner token changed; refusing to release")
	}
	if err := os.Remove(g.ownerPath(id)); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("launchguard: release owner: %w", err)
	}
	return nil
}

func (g *Guard) readState(id string) (diskState, error) {
	data, err := os.ReadFile(g.statePath(id))
	if errors.Is(err, fs.ErrNotExist) {
		return diskState{}, nil
	}
	if err != nil {
		return diskState{}, fmt.Errorf("launchguard: read state: %w", err)
	}
	var state diskState
	if err := json.Unmarshal(data, &state); err != nil {
		return diskState{}, fmt.Errorf("launchguard: decode state: %w", err)
	}
	return state, nil
}

func (g *Guard) writeState(id string, state diskState) error {
	data, err := json.Marshal(state)
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(g.cfg.Dir, id+".tmp-")
	if err != nil {
		return fmt.Errorf("launchguard: create state temp: %w", err)
	}
	name := tmp.Name()
	ok := false
	defer func() {
		_ = tmp.Close()
		if !ok {
			_ = os.Remove(name)
		}
	}()
	if err := tmp.Chmod(0o600); err != nil {
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		return err
	}
	if err := tmp.Sync(); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(name, g.statePath(id)); err != nil {
		return fmt.Errorf("launchguard: replace state: %w", err)
	}
	ok = true
	return nil
}

func processAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	p, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return p.Signal(syscall.Signal(0)) == nil
}
