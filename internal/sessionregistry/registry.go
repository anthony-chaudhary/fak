// Package sessionregistry stores durable, inspectable parent/child execution lineage.
package sessionregistry

import (
	"bytes"
	"crypto/rand"
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

	"github.com/anthony-chaudhary/fak/internal/flock"
	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
)

const Schema = "fak-child-registration/1"

type State string

const (
	StateRegistered State = "registered"
	StateActive     State = "active"
	StateCompleted  State = "completed"
	StateFailed     State = "failed"
	StateCancelled  State = "cancelled"
	StateLost       State = "lost"
	StateReaped     State = "reaped"
	StateUnknown    State = "unknown"
)

type Identity struct {
	Runtime          string    `json:"runtime"`
	SessionID        string    `json:"session_id,omitempty"`
	ThreadID         string    `json:"thread_id,omitempty"`
	PID              int       `json:"pid,omitempty"`
	ProcessStartedAt time.Time `json:"process_started_at,omitempty"`
	HostID           string    `json:"host_id,omitempty"`
}

type Record struct {
	Schema               string    `json:"schema"`
	RegistrationID       string    `json:"registration_id"`
	ParentRegistrationID string    `json:"parent_registration_id,omitempty"`
	ParentAttemptID      string    `json:"parent_attempt_id,omitempty"`
	RootRegistrationID   string    `json:"root_registration_id"`
	RootOutcome          string    `json:"root_outcome,omitempty"`
	RootIssue            string    `json:"root_issue,omitempty"`
	TaskID               string    `json:"task_id,omitempty"`
	AttemptID            string    `json:"attempt_id"`
	ResumeOfAttemptID    string    `json:"resume_of_attempt_id,omitempty"`
	LaunchKind           string    `json:"launch_kind"`
	Scope                []string  `json:"scope,omitempty"`
	Lane                 string    `json:"lane,omitempty"`
	LeaseID              string    `json:"lease_id,omitempty"`
	Identity             Identity  `json:"identity"`
	State                State     `json:"state"`
	Reason               string    `json:"reason,omitempty"`
	WitnessRef           string    `json:"witness_ref,omitempty"`
	CreatedAt            time.Time `json:"created_at"`
	StartedAt            time.Time `json:"started_at,omitempty"`
	HeartbeatAt          time.Time `json:"heartbeat_at,omitempty"`
	TerminalAt           time.Time `json:"terminal_at,omitempty"`
}

type Event struct {
	Schema string    `json:"schema"`
	At     time.Time `json:"at"`
	Record Record    `json:"record"`
}

type NewInput struct {
	RegistrationID       string
	ParentRegistrationID string
	ParentAttemptID      string
	RootRegistrationID   string
	RootOutcome          string
	RootIssue            string
	TaskID               string
	AttemptID            string
	ResumeOfAttemptID    string
	LaunchKind           string
	Scope                []string
	Lane                 string
	LeaseID              string
	Runtime              string
	SessionID            string
	ThreadID             string
	HostID               string
	Now                  time.Time
}

type Store struct{ Path string }

func DefaultPath() string {
	if v := strings.TrimSpace(os.Getenv("FAK_SESSION_REGISTRY")); v != "" {
		return v
	}
	return sessionjournal.DefaultPath()
}

func (s Store) usesJournal() bool {
	return strings.TrimSpace(os.Getenv("FAK_SESSION_REGISTRY")) == "" && filepath.Clean(s.Path) == filepath.Clean(sessionjournal.DefaultPath())
}

func appendJournalRecord(r Record) error {
	carry := registrationCarry(r)
	kind := sessionjournal.KindBeat
	if r.State == StateRegistered {
		kind = sessionjournal.KindOpen
	}
	if isTerminal(r.State) {
		kind = sessionjournal.KindClose
	}
	return sessionjournal.Append("", sessionjournal.Event{
		Schema: sessionjournal.Schema, Kind: kind, ID: r.RegistrationID,
		PID: r.Identity.PID, TS: journalEventTime(r).Format(time.RFC3339Nano),
		Host: r.Identity.HostID, Agent: r.Identity.Runtime, Registration: &carry,
		Reason: r.Reason,
	})
}

func journalEventTime(r Record) time.Time {
	for _, at := range []time.Time{r.TerminalAt, r.HeartbeatAt, r.StartedAt, r.CreatedAt} {
		if !at.IsZero() {
			return at.UTC()
		}
	}
	return time.Now().UTC()
}

func registrationCarry(r Record) sessionjournal.RegistrationCarry {
	return sessionjournal.RegistrationCarry{
		RegistrationID: r.RegistrationID, ParentRegistrationID: r.ParentRegistrationID,
		ParentAttemptID: r.ParentAttemptID, RootRegistrationID: r.RootRegistrationID,
		RootOutcome: r.RootOutcome, RootIssue: r.RootIssue, TaskID: r.TaskID, AttemptID: r.AttemptID,
		ResumeOfAttemptID: r.ResumeOfAttemptID, LaunchKind: r.LaunchKind, Scope: append([]string(nil), r.Scope...),
		Lane: r.Lane, LeaseID: r.LeaseID, Runtime: r.Identity.Runtime, SessionID: r.Identity.SessionID,
		ThreadID: r.Identity.ThreadID, PID: r.Identity.PID, ProcessStartedAt: formatTime(r.Identity.ProcessStartedAt), HostID: r.Identity.HostID,
		State: string(r.State), Reason: r.Reason, WitnessRef: r.WitnessRef, CreatedAt: formatTime(r.CreatedAt),
		StartedAt: formatTime(r.StartedAt), HeartbeatAt: formatTime(r.HeartbeatAt), TerminalAt: formatTime(r.TerminalAt),
	}
}

func recordsFromJournal(events []sessionjournal.Event) []Record {
	sessions := sessionjournal.FoldEvents(events)
	out := make([]Record, 0, len(sessions))
	for _, folded := range sessions {
		if folded.Registration == nil {
			continue
		}
		r := recordFromCarry(*folded.Registration)
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].CreatedAt.Before(out[j].CreatedAt) })
	return out
}

func recordFromCarry(c sessionjournal.RegistrationCarry) Record {
	return Record{Schema: Schema, RegistrationID: c.RegistrationID, ParentRegistrationID: c.ParentRegistrationID,
		ParentAttemptID: c.ParentAttemptID, RootRegistrationID: c.RootRegistrationID, RootOutcome: c.RootOutcome,
		RootIssue: c.RootIssue, TaskID: c.TaskID, AttemptID: c.AttemptID, ResumeOfAttemptID: c.ResumeOfAttemptID,
		LaunchKind: c.LaunchKind, Scope: append([]string(nil), c.Scope...), Lane: c.Lane, LeaseID: c.LeaseID,
		Identity: Identity{Runtime: c.Runtime, SessionID: c.SessionID, ThreadID: c.ThreadID, PID: c.PID,
			ProcessStartedAt: parseTime(c.ProcessStartedAt), HostID: c.HostID}, State: State(c.State), Reason: c.Reason,
		WitnessRef: c.WitnessRef, CreatedAt: parseTime(c.CreatedAt), StartedAt: parseTime(c.StartedAt),
		HeartbeatAt: parseTime(c.HeartbeatAt), TerminalAt: parseTime(c.TerminalAt)}
}

func formatTime(t time.Time) string {
	if t.IsZero() {
		return ""
	}
	return t.UTC().Format(time.RFC3339Nano)
}
func parseTime(v string) time.Time { t, _ := time.Parse(time.RFC3339Nano, v); return t.UTC() }

func New(in NewInput) (Record, error) {
	now := in.Now.UTC()
	if now.IsZero() {
		now = time.Now().UTC()
	}
	id := strings.TrimSpace(in.RegistrationID)
	if id == "" {
		id = newID()
	}
	attempt := strings.TrimSpace(in.AttemptID)
	if attempt == "" {
		attempt = id
	}
	root := strings.TrimSpace(in.RootRegistrationID)
	if root == "" {
		if strings.TrimSpace(in.ParentRegistrationID) != "" {
			return Record{}, errors.New("root_registration_id is required for a child registration")
		}
		root = id
	}
	r := Record{Schema: Schema, RegistrationID: id, ParentRegistrationID: strings.TrimSpace(in.ParentRegistrationID), ParentAttemptID: strings.TrimSpace(in.ParentAttemptID), RootRegistrationID: root, RootOutcome: strings.TrimSpace(in.RootOutcome), RootIssue: strings.TrimSpace(in.RootIssue), TaskID: strings.TrimSpace(in.TaskID), AttemptID: attempt, ResumeOfAttemptID: strings.TrimSpace(in.ResumeOfAttemptID), LaunchKind: strings.TrimSpace(in.LaunchKind), Scope: compact(in.Scope), Lane: strings.TrimSpace(in.Lane), LeaseID: strings.TrimSpace(in.LeaseID), Identity: Identity{Runtime: strings.TrimSpace(in.Runtime), SessionID: strings.TrimSpace(in.SessionID), ThreadID: strings.TrimSpace(in.ThreadID), HostID: strings.TrimSpace(in.HostID)}, State: StateRegistered, CreatedAt: now}
	if err := Validate(r); err != nil {
		return Record{}, err
	}
	return r, nil
}

func Validate(r Record) error {
	if r.Schema != Schema {
		return fmt.Errorf("schema must be %q", Schema)
	}
	for n, v := range map[string]string{"registration_id": r.RegistrationID, "root_registration_id": r.RootRegistrationID, "attempt_id": r.AttemptID, "launch_kind": r.LaunchKind, "identity.runtime": r.Identity.Runtime} {
		if strings.TrimSpace(v) == "" {
			return fmt.Errorf("%s is required", n)
		}
	}
	if r.ParentRegistrationID == r.RegistrationID {
		return errors.New("registration cannot parent itself")
	}
	if r.ParentRegistrationID != "" && r.RootRegistrationID == r.RegistrationID {
		return errors.New("child root_registration_id cannot equal its own registration_id")
	}
	switch r.State {
	case StateRegistered, StateActive, StateCompleted, StateFailed, StateCancelled, StateLost, StateReaped:
	case StateUnknown:
		if strings.TrimSpace(r.Reason) == "" {
			return errors.New("unknown state requires a reason")
		}
	default:
		return fmt.Errorf("unsupported state %q", r.State)
	}
	if isTerminal(r.State) && r.TerminalAt.IsZero() {
		return errors.New("terminal state requires terminal_at")
	}
	return nil
}

func (s Store) Register(r Record) error {
	if s.usesJournal() {
		return appendJournalRecord(r)
	}
	return s.withLock(func() error { return s.registerLocked(r) })
}

func (s Store) registerLocked(r Record) error {
	records, err := s.readAllUnlocked()
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	for _, old := range records {
		if old.RegistrationID == r.RegistrationID {
			if sameImmutable(old, r) {
				return nil
			}
			return fmt.Errorf("registration_id %q already exists with different identity", r.RegistrationID)
		}
	}
	if r.ParentRegistrationID != "" {
		found := false
		for _, old := range records {
			if old.RegistrationID == r.ParentRegistrationID {
				found = true
				if old.RootRegistrationID != r.RootRegistrationID {
					return errors.New("parent and child root_registration_id differ")
				}
				break
			}
		}
		if !found {
			return fmt.Errorf("parent registration %q is not registered", r.ParentRegistrationID)
		}
	}
	return s.append(r)
}

func (s Store) Start(id string, pid int, started time.Time) (Record, error) {
	return s.update(id, func(r *Record) {
		r.State = StateActive
		r.Identity.PID = pid
		r.Identity.ProcessStartedAt = started.UTC()
		r.StartedAt = started.UTC()
		r.HeartbeatAt = started.UTC()
	})
}
func (s Store) Terminal(id string, state State, reason, witness string, at time.Time) (Record, error) {
	if !isTerminal(state) && state != StateUnknown {
		return Record{}, fmt.Errorf("state %q is not terminal", state)
	}
	return s.update(id, func(r *Record) {
		r.State = state
		r.Reason = strings.TrimSpace(reason)
		r.WitnessRef = strings.TrimSpace(witness)
		r.TerminalAt = at.UTC()
		r.HeartbeatAt = at.UTC()
	})
}
func (s Store) update(id string, fn func(*Record)) (Record, error) {
	if s.usesJournal() {
		rows, err := s.ReadAll()
		if err != nil {
			return Record{}, err
		}
		for _, row := range rows {
			if row.RegistrationID == id {
				fn(&row)
				if err := appendJournalRecord(row); err != nil {
					return Record{}, err
				}
				return row, nil
			}
		}
		return Record{}, fmt.Errorf("registration %s not found", id)
	}
	var updated Record
	err := s.withLock(func() error {
		var err error
		updated, err = s.updateLocked(id, fn)
		return err
	})
	return updated, err
}

func (s Store) updateLocked(id string, fn func(*Record)) (Record, error) {
	rows, err := s.readAllUnlocked()
	if err != nil {
		return Record{}, err
	}
	var r Record
	ok := false
	for i := len(rows) - 1; i >= 0; i-- {
		if rows[i].RegistrationID == id {
			r = rows[i]
			ok = true
			break
		}
	}
	if !ok {
		return Record{}, fmt.Errorf("registration %q not found", id)
	}
	fn(&r)
	if err := Validate(r); err != nil {
		return Record{}, err
	}
	return r, s.append(r)
}
func (s Store) append(r Record) error {
	if err := Validate(r); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(s.Path), 0o700); err != nil {
		return err
	}
	if err := s.repairIncompleteTail(); err != nil {
		return err
	}
	f, err := os.OpenFile(s.Path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	b, err := json.Marshal(Event{Schema: Schema, At: time.Now().UTC(), Record: r})
	if err != nil {
		return err
	}
	if _, err = f.Write(append(b, '\n')); err != nil {
		return err
	}
	return f.Sync()
}

func (s Store) repairIncompleteTail() error {
	data, err := os.ReadFile(s.Path)
	if errors.Is(err, os.ErrNotExist) || len(data) == 0 || data[len(data)-1] == '\n' {
		return nil
	}
	if err != nil {
		return err
	}
	start := bytes.LastIndexByte(data, '\n') + 1
	var event Event
	decodeErr := json.Unmarshal(data[start:], &event)
	switch {
	case decodeErr == nil:
		f, err := os.OpenFile(s.Path, os.O_APPEND|os.O_WRONLY, 0o600)
		if err != nil {
			return err
		}
		_, writeErr := f.Write([]byte{'\n'})
		closeErr := f.Close()
		if writeErr != nil {
			return writeErr
		}
		return closeErr
	case isUnexpectedEnd(decodeErr):
		return os.Truncate(s.Path, int64(start))
	default:
		return fmt.Errorf("decode registry: %w", decodeErr)
	}
}

func isUnexpectedEnd(err error) bool {
	return errors.Is(err, io.ErrUnexpectedEOF) || err != nil && err.Error() == "unexpected end of JSON input"
}

func (s Store) ReadAll() ([]Record, error) {
	if s.usesJournal() {
		return recordsFromJournal(sessionjournal.LoadFile(sessionjournal.DefaultPath())), nil
	}
	var rows []Record
	err := s.withLock(func() error {
		var err error
		rows, err = s.readAllUnlocked()
		return err
	})
	return rows, err
}

func (s Store) readAllUnlocked() ([]Record, error) {
	data, err := os.ReadFile(s.Path)
	if err != nil {
		return nil, err
	}
	latest := map[string]Record{}
	lines := bytes.Split(data, []byte{'\n'})
	terminated := len(data) == 0 || data[len(data)-1] == '\n'
	for i, line := range lines {
		if len(line) == 0 && i == len(lines)-1 && terminated {
			continue
		}
		var e Event
		if err := json.Unmarshal(line, &e); err != nil {
			// An append killed between writes leaves an unterminated JSON tail. It
			// never became a durable event, so retain the preceding valid ledger.
			if i == len(lines)-1 && !terminated && isUnexpectedEnd(err) {
				break
			}
			return nil, fmt.Errorf("decode registry: %w", err)
		}
		if err := Validate(e.Record); err != nil {
			return nil, err
		}
		latest[e.Record.RegistrationID] = e.Record
	}
	out := make([]Record, 0, len(latest))
	for _, r := range latest {
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].CreatedAt.Equal(out[j].CreatedAt) {
			return out[i].RegistrationID < out[j].RegistrationID
		}
		return out[i].CreatedAt.Before(out[j].CreatedAt)
	})
	return out, nil
}

func (s Store) withLock(fn func() error) error {
	if err := os.MkdirAll(filepath.Dir(s.Path), 0o700); err != nil {
		return err
	}
	f, err := os.OpenFile(s.Path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return fmt.Errorf("open registry lock: %w", err)
	}
	defer f.Close()
	deadline := time.Now().Add(5 * time.Second)
	for {
		err = flock.TryLock(f)
		if err == nil {
			break
		}
		if !errors.Is(err, flock.ErrLockBusy) {
			return fmt.Errorf("lock registry: %w", err)
		}
		if time.Now().After(deadline) {
			return errors.New("lock registry: timed out")
		}
		time.Sleep(10 * time.Millisecond)
	}
	defer func() { _ = flock.Unlock(f) }()
	return fn()
}

func Chain(rows []Record, id string) []Record {
	by := map[string]Record{}
	children := map[string][]Record{}
	for _, r := range rows {
		by[r.RegistrationID] = r
		children[r.ParentRegistrationID] = append(children[r.ParentRegistrationID], r)
	}
	start, ok := by[id]
	if !ok {
		return nil
	}
	root := start.RootRegistrationID
	if r, ok := by[root]; ok {
		start = r
	}
	var out []Record
	var walk func(Record)
	walk = func(r Record) {
		out = append(out, r)
		kids := children[r.RegistrationID]
		sort.Slice(kids, func(i, j int) bool { return kids[i].CreatedAt.Before(kids[j].CreatedAt) })
		for _, k := range kids {
			walk(k)
		}
	}
	walk(start)
	return out
}

type Query struct {
	RootIssue            string
	ParentRegistrationID string
	RegistrationID       string
	SessionID            string
	ThreadID             string
	PID                  int
	ProcessStartedAt     time.Time
	Lane                 string
	LeaseID              string
	WitnessRef           string
}

type Counts struct {
	Total                int            `json:"total"`
	Registered           int            `json:"registered"`
	Active               int            `json:"active"`
	Terminal             int            `json:"terminal"`
	Unknown              int            `json:"unknown"`
	UnregisteredObserved int            `json:"unregistered_observed"`
	ByState              map[State]int  `json:"by_state"`
	ByKind               map[string]int `json:"by_kind"`
}

type ObservedProcess struct {
	PID              int       `json:"pid"`
	ProcessStartedAt time.Time `json:"process_started_at"`
	Runtime          string    `json:"runtime"`
	SessionID        string    `json:"session_id,omitempty"`
	ThreadID         string    `json:"thread_id,omitempty"`
}

type UnregisteredObserved struct {
	Verdict string          `json:"verdict"`
	Reason  string          `json:"reason"`
	Process ObservedProcess `json:"process"`
}

func Filter(rows []Record, q Query) []Record {
	var out []Record
	for _, r := range rows {
		if q.RegistrationID != "" && r.RegistrationID != q.RegistrationID {
			continue
		}
		if q.RootIssue != "" && r.RootIssue != q.RootIssue {
			continue
		}
		if q.ParentRegistrationID != "" && r.ParentRegistrationID != q.ParentRegistrationID {
			continue
		}
		if q.SessionID != "" && r.Identity.SessionID != q.SessionID {
			continue
		}
		if q.ThreadID != "" && r.Identity.ThreadID != q.ThreadID {
			continue
		}
		if q.PID != 0 && r.Identity.PID != q.PID {
			continue
		}
		if !q.ProcessStartedAt.IsZero() && !r.Identity.ProcessStartedAt.Equal(q.ProcessStartedAt.UTC()) {
			continue
		}
		if q.Lane != "" && r.Lane != q.Lane {
			continue
		}
		if q.LeaseID != "" && r.LeaseID != q.LeaseID {
			continue
		}
		if q.WitnessRef != "" && r.WitnessRef != q.WitnessRef {
			continue
		}
		out = append(out, r)
	}
	return out
}

func Summarize(rows []Record, unregistered int) Counts {
	c := Counts{Total: len(rows), UnregisteredObserved: unregistered, ByState: map[State]int{}, ByKind: map[string]int{}}
	for _, r := range rows {
		c.ByState[r.State]++
		c.ByKind[r.LaunchKind]++
		switch {
		case r.State == StateRegistered:
			c.Registered++
		case r.State == StateActive:
			c.Active++
		case isTerminal(r.State):
			c.Terminal++
		case r.State == StateUnknown:
			c.Unknown++
		}
	}
	return c
}

func ReconcileObserved(rows []Record, observed []ObservedProcess) []UnregisteredObserved {
	var out []UnregisteredObserved
	for _, p := range observed {
		matched := false
		for _, r := range rows {
			if r.Identity.PID == p.PID && !r.Identity.ProcessStartedAt.IsZero() && r.Identity.ProcessStartedAt.Equal(p.ProcessStartedAt.UTC()) {
				matched = true
				break
			}
		}
		if !matched {
			out = append(out, UnregisteredObserved{Verdict: "UNREGISTERED_OBSERVED", Reason: "no registration matches pid plus process_start identity", Process: p})
		}
	}
	return out
}

func isTerminal(s State) bool {
	switch s {
	case StateCompleted, StateFailed, StateCancelled, StateLost, StateReaped:
		return true
	}
	return false
}
func sameImmutable(a, b Record) bool {
	return a.RegistrationID == b.RegistrationID && a.ParentRegistrationID == b.ParentRegistrationID && a.ParentAttemptID == b.ParentAttemptID && a.RootRegistrationID == b.RootRegistrationID && a.AttemptID == b.AttemptID && a.LaunchKind == b.LaunchKind && a.Identity.Runtime == b.Identity.Runtime
}
func compact(in []string) []string {
	var out []string
	seen := map[string]bool{}
	for _, v := range in {
		v = strings.TrimSpace(v)
		if v != "" && !seen[v] {
			seen[v] = true
			out = append(out, v)
		}
	}
	return out
}
func newID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err == nil {
		return "reg-" + hex.EncodeToString(b[:])
	}
	return fmt.Sprintf("reg-%d", time.Now().UnixNano())
}
