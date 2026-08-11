// Package sessionregistry stores durable, inspectable parent/child execution lineage.
package sessionregistry

import (
	"bufio"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
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

var appendMu sync.Mutex

func DefaultPath() string {
	if v := strings.TrimSpace(os.Getenv("FAK_SESSION_REGISTRY")); v != "" {
		return v
	}
	if d, err := os.UserConfigDir(); err == nil {
		return filepath.Join(d, "fak", "child-registrations.jsonl")
	}
	return filepath.Join(".fak", "child-registrations.jsonl")
}

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
	records, err := s.ReadAll()
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
	rows, err := s.ReadAll()
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
	appendMu.Lock()
	defer appendMu.Unlock()
	if err := os.MkdirAll(filepath.Dir(s.Path), 0o700); err != nil {
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
func (s Store) ReadAll() ([]Record, error) {
	f, err := os.Open(s.Path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	latest := map[string]Record{}
	sc := bufio.NewScanner(f)
	buf := make([]byte, 64*1024)
	sc.Buffer(buf, 4*1024*1024)
	for sc.Scan() {
		var e Event
		if err := json.Unmarshal(sc.Bytes(), &e); err != nil {
			return nil, fmt.Errorf("decode registry: %w", err)
		}
		if err := Validate(e.Record); err != nil {
			return nil, err
		}
		latest[e.Record.RegistrationID] = e.Record
	}
	if err := sc.Err(); err != nil {
		return nil, err
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
