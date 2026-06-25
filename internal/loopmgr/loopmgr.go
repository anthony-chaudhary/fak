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
	StateArmed    LoopState = "armed"
	StateRunning  LoopState = "running"
	StatePaused   LoopState = "paused"
	StateDraining LoopState = "draining"
	StateStopped  LoopState = "stopped"
	StateDisabled LoopState = "disabled"
)

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

func Append(path string, ev Event, opts ...Option) (Event, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return Event{}, errors.New("loop ledger path is required")
	}
	cfg := appendConfig{clock: time.Now}
	for _, opt := range opts {
		opt(&cfg)
	}

	existing, err := Load(path)
	if err != nil {
		return Event{}, err
	}
	if err := validateNewEvent(ev); err != nil {
		return Event{}, err
	}

	ev.Schema = SchemaEvent
	ev.Seq = uint64(len(existing) + 1)
	ev.TSUnixNano = cfg.clock().UTC().UnixNano()
	ev.PrevHash = ""
	if len(existing) > 0 {
		ev.PrevHash = existing[len(existing)-1].Hash
	}
	ev.Hash = hashEvent(ev)

	dir := filepath.Dir(path)
	if dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return Event{}, fmt.Errorf("create loop ledger dir: %w", err)
		}
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return Event{}, fmt.Errorf("open loop ledger: %w", err)
	}
	defer f.Close()

	line, err := json.Marshal(ev)
	if err != nil {
		return Event{}, fmt.Errorf("marshal loop event: %w", err)
	}
	if _, err := f.Write(append(line, '\n')); err != nil {
		return Event{}, fmt.Errorf("append loop event: %w", err)
	}
	return ev, nil
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
				loop.State = string(StatusRefused)
			} else {
				loop.Admitted++
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
	LoopID             string       `json:"loop_id"`
	State              string       `json:"state,omitempty"`
	LastSeq            uint64       `json:"last_seq"`
	LastEventUnixNano  int64        `json:"last_event_unix_nano,omitempty"`
	LastKind           EventKind    `json:"last_kind,omitempty"`
	CurrentRunID       string       `json:"current_run_id,omitempty"`
	Fires              uint64       `json:"fires"`
	Admitted           uint64       `json:"admitted"`
	Refused            uint64       `json:"refused"`
	Started            uint64       `json:"started"`
	Ended              uint64       `json:"ended"`
	Witnessed          uint64       `json:"witnessed"`
	WitnessRefused     uint64       `json:"witness_refused"`
	WitnessUnavailable uint64       `json:"witness_unavailable"`
	Notifications      uint64       `json:"notifications"`
	LastRun            *RunSnapshot `json:"last_run,omitempty"`
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

func validateNewEvent(ev Event) error {
	if ev.Schema != "" && ev.Schema != SchemaEvent {
		return fmt.Errorf("schema = %q, want %q", ev.Schema, SchemaEvent)
	}
	if strings.TrimSpace(ev.LoopID) == "" {
		return errors.New("loop_id is required")
	}
	if !validKind(ev.Kind) {
		return fmt.Errorf("unknown loop event kind %q", ev.Kind)
	}
	return nil
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
	if strings.TrimSpace(ev.LoopID) == "" {
		return errors.New("loop_id is required")
	}
	if !validKind(ev.Kind) {
		return fmt.Errorf("unknown loop event kind %q", ev.Kind)
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
