package loopmgr

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dormancy"
	"github.com/anthony-chaudhary/fak/internal/flock"
	"github.com/anthony-chaudhary/fak/internal/lifecycle"
)

const (
	SchemaEvent  = "fak.loop-event.v1"
	SchemaStatus = "fak.loop-status.v1"
)

type EventKind string

const (
	EventArmed     EventKind = "armed"
	EventFire      EventKind = "fire"
	EventAdmit     EventKind = "admit"
	EventStart     EventKind = "start"
	EventHeartbeat EventKind = "heartbeat"
	EventEnd       EventKind = "end"
	EventWitness   EventKind = "witness"
	EventNotify    EventKind = "notify"
)

type LoopState string

const (
	// Armed and Disabled are supervisor-only schedule states — a live served
	// sequence has no peer for either, so they are spelled here, not in the shared
	// leaf. The four common states are SOURCED from internal/lifecycle (the single
	// definition the served session shares) so the two machines cannot drift apart.
	StateArmed    LoopState = "armed"
	StateRunning  LoopState = lifecycle.TokenRunning
	StatePaused   LoopState = lifecycle.TokenPaused
	StateDraining LoopState = lifecycle.TokenDraining
	StateStopped  LoopState = lifecycle.TokenStopped
	StateDisabled LoopState = "disabled"
)

// Phase projects a LoopState onto the shared lifecycle skeleton. The bool is false
// for the supervisor-only extras (Armed/Disabled) and any unknown string — the
// projection is explicit about the extras, never a silent default. This is the
// supervisor half of the #912 "one machine" converter; internal/lifebridge
// composes it with the served-session half.
func (s LoopState) Phase() (lifecycle.Phase, bool) {
	switch s {
	case StateRunning:
		return lifecycle.Running, true
	case StatePaused:
		return lifecycle.Paused, true
	case StateDraining:
		return lifecycle.Draining, true
	case StateStopped:
		return lifecycle.Stopped, true
	}
	return 0, false
}

// LoopStateFromPhase lifts a shared lifecycle Phase into a LoopState. It is total
// over the four Phases (every shared state has a LoopState peer); an out-of-range
// Phase yields ("", false).
func LoopStateFromPhase(p lifecycle.Phase) (LoopState, bool) {
	switch p {
	case lifecycle.Running:
		return StateRunning, true
	case lifecycle.Paused:
		return StatePaused, true
	case lifecycle.Draining:
		return StateDraining, true
	case lifecycle.Stopped:
		return StateStopped, true
	}
	return "", false
}

type RunStatus string

const (
	StatusAdmitted           RunStatus = "admitted"
	StatusRefused            RunStatus = "refused"
	StatusRunning            RunStatus = "running"
	StatusClaimedDone        RunStatus = "claimed_done"
	StatusWitnessedDone      RunStatus = "witnessed_done"
	StatusWitnessRefused     RunStatus = "witness_refused"
	StatusWitnessUnavailable RunStatus = "witness_unavailable"
	StatusFailed             RunStatus = "failed"
	StatusCanceled           RunStatus = "canceled"
)

type EvidenceRef struct {
	Kind    string `json:"kind"`
	Ref     string `json:"ref"`
	Summary string `json:"summary,omitempty"`
	SHA256  string `json:"sha256,omitempty"`
}

type Event struct {
	Schema       string           `json:"schema"`
	Seq          uint64           `json:"seq"`
	TSUnixNano   int64            `json:"ts_unix_nano"`
	LoopID       string           `json:"loop_id"`
	RunID        string           `json:"run_id,omitempty"`
	Kind         EventKind        `json:"kind"`
	Source       string           `json:"source,omitempty"`
	Principal    string           `json:"principal,omitempty"`
	State        LoopState        `json:"state,omitempty"`
	Status       RunStatus        `json:"status,omitempty"`
	Reason       string           `json:"reason,omitempty"`
	Summary      string           `json:"summary,omitempty"`
	EvidenceRefs []EvidenceRef    `json:"evidence_refs,omitempty"`
	Metrics      map[string]int64 `json:"metrics,omitempty"`
	PrevHash     string           `json:"prev_hash,omitempty"`
	Hash         string           `json:"hash"`
}

type Option func(*appendConfig)

type appendConfig struct {
	clock func() time.Time
}

func WithClock(clock func() time.Time) Option {
	return func(cfg *appendConfig) {
		if clock != nil {
			cfg.clock = clock
		}
	}
}

// ErrLedgerBusy is returned by Append when the cross-process append lock could not
// be acquired within the bounded wait. It is deliberately fail-closed: a forked chain
// (two unserialized writers stamping the same seq + prev_hash) is worse than a
// retriable error, so a contended-out Append never falls back to an unlocked write.
// Callers (the one-shot `fak loop ...` producer) can retry.
var ErrLedgerBusy = errors.New("loopmgr: loop ledger append lock is busy")

// appendLockWait bounds how long Append polls for the append lock before failing with
// ErrLedgerBusy. A local single-line append holds the lock for microseconds, so this
// should essentially never elapse; it only bounds a pathological stuck holder.
const appendLockWait = 2 * time.Second

func Append(path string, ev Event, opts ...Option) (Event, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return Event{}, errors.New("loop ledger path is required")
	}
	cfg := appendConfig{clock: time.Now}
	for _, opt := range opts {
		opt(&cfg)
	}

	// Validate the caller's event before touching the lock (cheap, no I/O).
	if err := validateNewEvent(ev); err != nil {
		return Event{}, err
	}

	dir := filepath.Dir(path)
	if dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return Event{}, fmt.Errorf("create loop ledger dir: %w", err)
		}
	}

	// Cross-process critical section: hold an OS advisory lock on a sidecar
	// <path>.lock fd across the whole read-compute-write, so seq/prev_hash are
	// derived from the TRUE tail under exclusion and two processes cannot fork the
	// chain. Correctness rests on recomputing under the lock, NOT on the lock being
	// acquired within the budget — on timeout we fail (ErrLedgerBusy), never proceed.
	var out Event
	err := withLedgerLock(path, appendLockWait, func() error {
		existing, err := Load(path)
		if err != nil {
			return err
		}

		ev.Schema = SchemaEvent
		ev.Seq = uint64(len(existing) + 1)
		ev.TSUnixNano = cfg.clock().UTC().UnixNano()
		ev.PrevHash = ""
		if len(existing) > 0 {
			ev.PrevHash = existing[len(existing)-1].Hash
		}
		ev.Hash = hashEvent(ev)

		f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			return fmt.Errorf("open loop ledger: %w", err)
		}
		defer f.Close()

		line, err := json.Marshal(ev)
		if err != nil {
			return fmt.Errorf("marshal loop event: %w", err)
		}
		if _, err := f.Write(append(line, '\n')); err != nil {
			return fmt.Errorf("append loop event: %w", err)
		}
		out = ev
		return nil
	})
	if err != nil {
		return Event{}, err
	}
	return out, nil
}

// withLedgerLock runs fn while holding an exclusive cross-process advisory lock on
// <path>.lock. flock.TryLock is non-blocking, so it polls until the lock
// is free or wait elapses (then ErrLedgerBusy). The lock fd is closed on return, which
// also releases the OS lock (and the OS releases it if this process dies mid-write).
func withLedgerLock(path string, wait time.Duration, fn func() error) error {
	lockPath := path + ".lock"
	f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return fmt.Errorf("open loop ledger lock: %w", err)
	}
	defer f.Close()

	deadline := time.Now().Add(wait)
	for {
		lerr := flock.TryLock(f)
		if lerr == nil {
			break
		}
		if !errors.Is(lerr, flock.ErrLockBusy) {
			return fmt.Errorf("lock loop ledger: %w", lerr)
		}
		if time.Now().After(deadline) {
			return ErrLedgerBusy
		}
		time.Sleep(25 * time.Millisecond)
	}
	defer func() { _ = flock.Unlock(f) }()
	return fn()
}

func Load(path string) ([]Event, error) {
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("open loop ledger: %w", err)
	}
	defer f.Close()
	return loadReader(f)
}

func SnapshotFile(path string, now time.Time) (Status, error) {
	events, err := Load(path)
	if err != nil {
		return Status{}, err
	}
	st := Summarize(events, now)
	st.LedgerPath = path
	return st, nil
}

func Summarize(events []Event, now time.Time) Status {
	byLoop := map[string]*LoopSnapshot{}
	for _, ev := range events {
		loop := byLoop[ev.LoopID]
		if loop == nil {
			loop = &LoopSnapshot{LoopID: ev.LoopID}
			byLoop[ev.LoopID] = loop
		}
		loop.LastSeq = ev.Seq
		loop.LastEventUnixNano = ev.TSUnixNano
		loop.LastKind = ev.Kind
		if ev.RunID != "" {
			loop.CurrentRunID = ev.RunID
		}
		if ev.State != "" {
			loop.State = string(ev.State)
		}

		switch ev.Kind {
		case EventArmed:
			if loop.State == "" {
				loop.State = string(StateArmed)
			}
		case EventFire:
			loop.Fires++
			if loop.State == "" {
				loop.State = "fired"
			}
		case EventAdmit:
			if ev.Status == StatusRefused {
				loop.Refused++
				loop.ConsecutiveRefusals++
				loop.State = string(StatusRefused)
			} else {
				loop.Admitted++
				loop.ConsecutiveRefusals = 0
				if loop.State == "" {
					loop.State = string(StatusAdmitted)
				}
			}
			loop.setRun(ev, fallbackStatus(ev.Status, StatusAdmitted))
		case EventStart:
			loop.Started++
			loop.State = string(StateRunning)
			loop.setRun(ev, StatusRunning)
		case EventEnd:
			loop.Ended++
			status := fallbackStatus(ev.Status, StatusClaimedDone)
			loop.State = string(status)
			loop.setRun(ev, status)
		case EventWitness:
			status := fallbackStatus(ev.Status, StatusWitnessUnavailable)
			switch status {
			case StatusWitnessedDone:
				loop.Witnessed++
			case StatusWitnessRefused:
				loop.WitnessRefused++
			case StatusWitnessUnavailable:
				loop.WitnessUnavailable++
			}
			loop.State = string(status)
			loop.setRun(ev, status)
		case EventNotify:
			loop.Notifications++
		}
	}

	ids := make([]string, 0, len(byLoop))
	for id := range byLoop {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	loops := make([]LoopSnapshot, 0, len(ids))
	for _, id := range ids {
		loops = append(loops, *byLoop[id])
	}
	return Status{
		Schema:     SchemaStatus,
		TSUnixNano: now.UTC().UnixNano(),
		Loops:      loops,
	}
}

type Status struct {
	Schema     string         `json:"schema"`
	TSUnixNano int64          `json:"ts_unix_nano"`
	LedgerPath string         `json:"ledger_path,omitempty"`
	Loops      []LoopSnapshot `json:"loops"`
}

type LoopSnapshot struct {
	LoopID              string       `json:"loop_id"`
	State               string       `json:"state,omitempty"`
	LastSeq             uint64       `json:"last_seq"`
	LastEventUnixNano   int64        `json:"last_event_unix_nano,omitempty"`
	LastKind            EventKind    `json:"last_kind,omitempty"`
	CurrentRunID        string       `json:"current_run_id,omitempty"`
	Fires               uint64       `json:"fires"`
	Admitted            uint64       `json:"admitted"`
	Refused             uint64       `json:"refused"`
	ConsecutiveRefusals uint64       `json:"consecutive_refusals"`
	Started             uint64       `json:"started"`
	Ended               uint64       `json:"ended"`
	Witnessed           uint64       `json:"witnessed"`
	WitnessRefused      uint64       `json:"witness_refused"`
	WitnessUnavailable  uint64       `json:"witness_unavailable"`
	Notifications       uint64       `json:"notifications"`
	LastRun             *RunSnapshot `json:"last_run,omitempty"`
}

// LastActive exposes the loop's dormancy clock (issue #1179, epic #1178): the durable
// LastActiveAt stamp derived from the last ledger-event timestamp the fold already
// carries. From it a loop's dormancy band (warm/cool/cold/frozen/ancient) is derivable
// without I/O via snap.LastActive().HorizonAt(now) — the input the #1180 dormant-vs-stuck
// split and the Phase-3 durable-wake rungs (#1188) key on. A loop with no events yet
// yields the zero (unknown) Stamp, which buckets to Ancient. Pure: it reads only the
// already-recorded LastEventUnixNano, adds no field, and changes no ledger byte.
func (s LoopSnapshot) LastActive() dormancy.Stamp {
	return dormancy.FromUnixNano(s.LastEventUnixNano)
}

type RunSnapshot struct {
	RunID         string        `json:"run_id,omitempty"`
	Status        RunStatus     `json:"status,omitempty"`
	Reason        string        `json:"reason,omitempty"`
	Summary       string        `json:"summary,omitempty"`
	EvidenceRefs  []EvidenceRef `json:"evidence_refs,omitempty"`
	EndedUnixNano int64         `json:"ended_unix_nano,omitempty"`
}

func (s *LoopSnapshot) setRun(ev Event, status RunStatus) {
	runID := ev.RunID
	if runID == "" && s.LastRun != nil {
		runID = s.LastRun.RunID
	}
	s.LastRun = &RunSnapshot{
		RunID:         runID,
		Status:        status,
		Reason:        ev.Reason,
		Summary:       ev.Summary,
		EvidenceRefs:  append([]EvidenceRef(nil), ev.EvidenceRefs...),
		EndedUnixNano: ev.TSUnixNano,
	}
}

func loadReader(r io.Reader) ([]Event, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var out []Event
	var prev string
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var ev Event
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			return nil, fmt.Errorf("loop ledger line %d: decode: %w", lineNo, err)
		}
		if err := validateLoadedEvent(ev, uint64(len(out)+1), prev); err != nil {
			return nil, fmt.Errorf("loop ledger line %d: %w", lineNo, err)
		}
		prev = ev.Hash
		out = append(out, ev)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read loop ledger: %w", err)
	}
	return out, nil
}

// Integrity describes the first chain break a tolerant read encountered, if any. It
// is the structured form of "the strict reader would have aborted here": a console or
// other read-only consumer can render what was recovered and surface the break,
// instead of the whole pane going dark on a single forked/corrupt line.
type Integrity struct {
	Broken    bool   `json:"broken"`
	AtLine    int    `json:"at_line,omitempty"`
	AtSeq     uint64 `json:"at_seq,omitempty"`
	Reason    string `json:"reason,omitempty"`
	Recovered int    `json:"recovered_events"`
}

// LoadPrefix is the tolerant sibling of Load: it reads the longest valid chained
// prefix and, instead of aborting on the first integrity break (forked seq, bad
// prev_hash, tampered hash), STOPS and returns the events recovered so far plus an
// Integrity describing the break. err is reserved for true I/O / scanner faults, not
// chain breaks — a forked ledger yields (prefix, Integrity{Broken:true}, nil). This
// mirrors internal/journal's Verify (strict) vs ReadRows (tolerant) split; Load stays
// strict so the tamper-evidence guarantee is never weakened.
func LoadPrefix(path string) ([]Event, Integrity, error) {
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, Integrity{}, nil
	}
	if err != nil {
		return nil, Integrity{}, fmt.Errorf("open loop ledger: %w", err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var out []Event
	var prev string
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var ev Event
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			return out, Integrity{Broken: true, AtLine: lineNo, Reason: "decode: " + err.Error(), Recovered: len(out)}, nil
		}
		if verr := validateLoadedEvent(ev, uint64(len(out)+1), prev); verr != nil {
			return out, Integrity{Broken: true, AtLine: lineNo, AtSeq: ev.Seq, Reason: verr.Error(), Recovered: len(out)}, nil
		}
		prev = ev.Hash
		out = append(out, ev)
	}
	if err := scanner.Err(); err != nil {
		return out, Integrity{}, fmt.Errorf("read loop ledger: %w", err)
	}
	return out, Integrity{Recovered: len(out)}, nil
}

// SnapshotFilePartial is the tolerant sibling of SnapshotFile: it Summarizes the
// recovered prefix from LoadPrefix and carries out the Integrity break (if any). Used
// by the loops console so a forked/corrupt ledger renders the loops it could recover
// plus a break banner, instead of exiting blank.
func SnapshotFilePartial(path string, now time.Time) (Status, Integrity, error) {
	events, integ, err := LoadPrefix(path)
	if err != nil {
		return Status{}, integ, err
	}
	st := Summarize(events, now)
	st.LedgerPath = path
	return st, integ, nil
}

// validateEventCore checks the fields every loop event must satisfy regardless of
// whether it is newly minted or loaded from the ledger: a non-empty loop id and a
// known kind. Shared by validateNewEvent and validateLoadedEvent.
func validateEventCore(ev Event) error {
	if strings.TrimSpace(ev.LoopID) == "" {
		return errors.New("loop_id is required")
	}
	if !validKind(ev.Kind) {
		return fmt.Errorf("unknown loop event kind %q", ev.Kind)
	}
	return nil
}

func validateNewEvent(ev Event) error {
	if ev.Schema != "" && ev.Schema != SchemaEvent {
		return fmt.Errorf("schema = %q, want %q", ev.Schema, SchemaEvent)
	}
	return validateEventCore(ev)
}

func validateLoadedEvent(ev Event, wantSeq uint64, wantPrev string) error {
	if ev.Schema != SchemaEvent {
		return fmt.Errorf("schema = %q, want %q", ev.Schema, SchemaEvent)
	}
	if ev.Seq != wantSeq {
		return fmt.Errorf("seq = %d, want %d", ev.Seq, wantSeq)
	}
	if ev.PrevHash != wantPrev {
		return fmt.Errorf("prev_hash = %q, want %q", ev.PrevHash, wantPrev)
	}
	if err := validateEventCore(ev); err != nil {
		return err
	}
	if ev.Hash == "" {
		return errors.New("hash is required")
	}
	if got := hashEvent(ev); got != ev.Hash {
		return fmt.Errorf("hash = %q, want %q", ev.Hash, got)
	}
	return nil
}

func validKind(kind EventKind) bool {
	switch kind {
	case EventArmed, EventFire, EventAdmit, EventStart, EventHeartbeat, EventEnd, EventWitness, EventNotify:
		return true
	default:
		return false
	}
}

func fallbackStatus(status, fallback RunStatus) RunStatus {
	if status != "" {
		return status
	}
	return fallback
}

func hashEvent(ev Event) string {
	ev.Hash = ""
	b, err := json.Marshal(ev)
	if err != nil {
		panic(err)
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
