// Package sessionjournal is the crash-survivable session-registration journal: an
// append-only JSONL lifecycle log (open / beat / close, each boot-stamped) plus the
// pure fold that classifies every recorded session LIVE / CRASHED / STALE / CLOSED
// against the machine BOOT EPOCH — so a system-wide infra crash (a Windows-update
// reboot, or a WindowsTerminal 0xc0000005 that kills every terminal at one instant)
// can be recovered from: the fleet re-enumerated and the crashed set resumed.
//
// The keystone is the boot epoch (bootepoch_*.go). A session that started BEFORE the
// machine's current boot and was never cleanly closed cannot still be running — it
// died in the reboot. That single comparison is the machine-wide-crash detector, and
// it needs only a start time, which every recorded session already carries. It is the
// durable, authoritative input the resume pipeline reconstructs best-effort today
// (transcript scans + an irreversible-slug cwd recovery + a reboot-erased process
// scan). See docs/notes/CONCEPT-SESSION-CRASH-JOURNAL-BOOT-EPOCH-2026-07-09.md and
// epic #3784 (this is C1, #3785).
//
// The package is pure and stdlib-lean over an injected path: Append writes one line,
// FoldEvents folds the log to one Session per id, and Classify applies the boot-epoch
// verdict. It composes with — does not replace — internal/guardsessions (#3461) and
// internal/session's descriptor registry (#1197); unifying them is C7 (#3791).
package sessionjournal

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

// Schema tags each row so a forward-extended reader can tell a lifecycle event from any
// other JSONL that might share the file.
const Schema = "fak.sessionjournal.v1"

// FileName is the durable journal basename, and EnvPath overrides its full path.
const (
	FileName = "session-journal.jsonl"
	EnvPath  = "FAK_SESSION_JOURNAL"
)

// Kind is the lifecycle transition an event records.
type Kind string

const (
	KindOpen   Kind = "open"   // registration at session start
	KindBeat   Kind = "beat"   // heartbeat / liveness ping
	KindClose  Kind = "close"  // clean deregister at graceful exit
	KindRefuse Kind = "refuse" // source refused before capture; content-free audit row
)

// ErrSourceDenied reports that Append wrote a content-free refusal instead of
// the caller's event. It deliberately carries no source string.
var ErrSourceDenied = errors.New("sessionjournal: capture source refused: " + pathutil.CaptureRefusalReason)

// DriveCarry is the plain-data projection of a session's remaining drive-state — the
// budget / generation / objective axes that must survive a machine-wide reboot so a
// resumed process comes up at the RIGHT remaining allotment, not a fresh full one. Every
// field is a scalar (no internal state types) so this foundation leaf stays below the
// §5 no-upward-import fence; it is the crash-journal mirror of internal/resume.DriveCarry
// and the projection of internal/session.State's Budget / Generation / ObjectivePin axes.
// A nil *DriveCarry on an Event / Session means "no carry" — exactly today's behavior.
//
// This slice adds only the CHANNEL. Populating it from a live internal/session.State on
// the write path is the guard/cmd writer's job (kept out here to hold the stdlib fence),
// and the identity-map join + report render are the sibling E2 / E3 leaves.
type DriveCarry struct {
	TurnsLeft           int64  `json:"turns_left,omitempty"`
	TokensLeft          int64  `json:"tokens_left,omitempty"`
	ContextTokensLeft   int64  `json:"context_tokens_left,omitempty"`
	ContextTokensCap    int64  `json:"context_tokens_cap,omitempty"`
	SpendMicroCentsLeft int64  `json:"spend_micro_cents_left,omitempty"`
	SpendMicroCentsCap  int64  `json:"spend_micro_cents_cap,omitempty"`
	Generation          int    `json:"generation,omitempty"`
	ObjectivePinID      string `json:"objective_pin_id,omitempty"`
}

// Event is one appended lifecycle row. Forward-extensible: a row with extra keys still
// decodes, so later rungs can add fields without breaking this reader.
// RegistrationCarry is the discoverability/lineage projection persisted on the
// lifecycle journal. It keeps the journal package independent of sessionregistry
// while allowing that registry to rebuild its exact latest-record view.
type RegistrationCarry struct {
	RegistrationID       string   `json:"registration_id"`
	ParentRegistrationID string   `json:"parent_registration_id,omitempty"`
	ParentAttemptID      string   `json:"parent_attempt_id,omitempty"`
	RootRegistrationID   string   `json:"root_registration_id,omitempty"`
	RootOutcome          string   `json:"root_outcome,omitempty"`
	RootIssue            string   `json:"root_issue,omitempty"`
	TaskID               string   `json:"task_id,omitempty"`
	AttemptID            string   `json:"attempt_id,omitempty"`
	ResumeOfAttemptID    string   `json:"resume_of_attempt_id,omitempty"`
	LaunchKind           string   `json:"launch_kind,omitempty"`
	Scope                []string `json:"scope,omitempty"`
	Lane                 string   `json:"lane,omitempty"`
	LeaseID              string   `json:"lease_id,omitempty"`
	Runtime              string   `json:"runtime,omitempty"`
	SessionID            string   `json:"session_id,omitempty"`
	ThreadID             string   `json:"thread_id,omitempty"`
	PID                  int      `json:"pid,omitempty"`
	ProcessStartedAt     string   `json:"process_started_at,omitempty"`
	HostID               string   `json:"host_id,omitempty"`
	State                string   `json:"state,omitempty"`
	Reason               string   `json:"reason,omitempty"`
	WitnessRef           string   `json:"witness_ref,omitempty"`
	CreatedAt            string   `json:"created_at,omitempty"`
	StartedAt            string   `json:"started_at,omitempty"`
	HeartbeatAt          string   `json:"heartbeat_at,omitempty"`
	TerminalAt           string   `json:"terminal_at,omitempty"`
}

type Event struct {
	Schema       string             `json:"schema"`
	Kind         Kind               `json:"kind"`
	ID           string             `json:"id"` // session / trace id — the join + fold key
	TS           string             `json:"ts"` // RFC3339 UTC event time
	Boot         string             `json:"boot,omitempty"`
	PID          int                `json:"pid,omitempty"`
	ParentPID    int                `json:"parent_pid,omitempty"`
	Host         string             `json:"host,omitempty"`
	CWD          string             `json:"cwd,omitempty"`
	Model        string             `json:"model,omitempty"`
	Agent        string             `json:"agent,omitempty"`
	Account      string             `json:"account,omitempty"` // config dir / seat
	Argv         []string           `json:"argv,omitempty"`
	StartSHA     string             `json:"start_sha,omitempty"`
	Gateway      string             `json:"gateway,omitempty"`
	Drive        *DriveCarry        `json:"drive,omitempty"`         // remaining drive-state to resume at (nil = none)
	Registration *RegistrationCarry `json:"registration,omitempty"`  // latest discoverability/lineage record
	Reason       string             `json:"reason,omitempty"`        // close reason
	SourceDigest string             `json:"source_digest,omitempty"` // denied-source identity; raw source is never recorded
}

// DefaultPath resolves the journal path: the explicit override, otherwise the
// machine-global store shared by every account on the host.
func DefaultPath() string {
	if p := strings.TrimSpace(os.Getenv(EnvPath)); p != "" {
		return p
	}
	return hostJournalPath()
}

func bootMarkerPath() string { return filepath.Join(filepath.Dir(DefaultPath()), "boot-marker.json") }

// Append writes one event as a single JSONL line, creating the dir and file as needed.
// The append is one Write of one line, so concurrent session starts interleave at line
// granularity (O_APPEND) without corrupting rows — the same guarantee guardsessions.Record
// relies on. Best-effort by contract: a lifecycle append must never block the session, so
// callers log the returned error but proceed. An empty path resolves to DefaultPath.
// Cross-process flock hardening is C6 (#3790).
func Append(path string, ev Event) error {
	if ev.Schema == "" {
		ev.Schema = Schema
	}
	if match := pathutil.CheckCaptureSource(ev.CWD); match.Refused {
		refusal := Event{
			Schema:       Schema,
			Kind:         KindRefuse,
			ID:           ev.ID,
			TS:           ev.TS,
			Boot:         ev.Boot,
			Reason:       match.Reason,
			SourceDigest: match.SourceDigest,
		}
		if err := appendEvent(path, refusal); err != nil {
			return fmt.Errorf("%w (refusal audit unavailable: %v)", ErrSourceDenied, err)
		}
		return ErrSourceDenied
	}
	return appendEvent(path, ev)
}

func appendEvent(path string, ev Event) error {
	if strings.TrimSpace(path) == "" {
		path = DefaultPath()
	}
	return withJournalLock(path, func() error {
		if dir := filepath.Dir(path); dir != "" && dir != "." {
			if err := os.MkdirAll(dir, 0o755); err != nil {
				return err
			}
		}
		f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		if err != nil {
			return err
		}
		defer f.Close()
		b, err := json.Marshal(ev)
		if err != nil {
			return err
		}
		if _, err = f.Write(append(b, '\n')); err != nil {
			return err
		}
		return f.Sync()
	})
}

// ParseHealth is content-free integrity evidence for one journal scan.
// Counts never retain source lines, paths, arguments, or transcript content.
type ParseHealth struct {
	TotalRows       int    `json:"total_rows"`
	BlankRows       int    `json:"blank_rows"`
	AcceptedRows    int    `json:"accepted_rows"`
	MalformedRows   int    `json:"malformed_rows"`
	WrongSchemaRows int    `json:"wrong_schema_rows"`
	MissingIDRows   int    `json:"missing_id_rows"`
	ScanError       string `json:"scan_error,omitempty"`
	ReadError       string `json:"read_error,omitempty"`
}

// Degraded reports whether any nonblank source row could not enter the fold.
func (h ParseHealth) Degraded() bool {
	return h.MalformedRows > 0 || h.WrongSchemaRows > 0 || h.MissingIDRows > 0 || h.ScanError != "" || h.ReadError != ""
}

// ParseEventsReport scans JSONL into events and returns content-free rejection counts.
// Recovery remains tolerant: valid rows survive a torn tail, while callers can now expose
// that degradation rather than presenting a partial fold as silently complete.
func ParseEventsReport(content string) ([]Event, ParseHealth) {
	var out []Event
	var health ParseHealth
	sc := bufio.NewScanner(strings.NewReader(content))
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for sc.Scan() {
		health.TotalRows++
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			health.BlankRows++
			continue
		}
		var ev Event
		if json.Unmarshal([]byte(line), &ev) != nil {
			health.MalformedRows++
			continue
		}
		if ev.Schema != Schema {
			health.WrongSchemaRows++
			continue
		}
		if strings.TrimSpace(ev.ID) == "" {
			health.MissingIDRows++
			continue
		}
		out = append(out, ev)
		health.AcceptedRows++
	}
	if err := sc.Err(); err != nil {
		health.ScanError = "row_too_large"
	}
	return out, health
}

// ParseEvents preserves the tolerant recovery API while ParseEventsReport serves
// observability consumers that must distinguish complete from degraded folds.
func ParseEvents(content string) []Event {
	events, _ := ParseEventsReport(content)
	return events
}

// LoadFileReport reads and parses one journal. ReadError is a bounded class rather than
// the raw OS error, so paths and host details never cross the observability boundary.
func LoadFileReport(path string) ([]Event, ParseHealth) {
	b, err := os.ReadFile(path)
	if err != nil {
		kind := "unreadable"
		if os.IsNotExist(err) {
			kind = "not_found"
		}
		return nil, ParseHealth{ReadError: kind}
	}
	return ParseEventsReport(string(b))
}

// LoadFile preserves the recovery API: a missing/unreadable journal yields no events.
func LoadFile(path string) []Event {
	events, _ := LoadFileReport(path)
	return events
}

// Session is the folded lifecycle state of one recorded session — the input to Classify.
// EventsAsOf returns authoritative events whose valid RFC3339 timestamp is at or before
// the requested instant. Invalid timestamps are excluded rather than treated as ancient facts.
func EventsAsOf(events []Event, asOf time.Time) []Event {
	out := make([]Event, 0, len(events))
	for _, ev := range events {
		t, err := time.Parse(time.RFC3339, strings.TrimSpace(ev.TS))
		if err == nil && !t.After(asOf) {
			out = append(out, ev)
		}
	}
	return out
}

type Session struct {
	ID           string             `json:"id"`
	Boot         string             `json:"boot,omitempty"`
	PID          int                `json:"pid,omitempty"`
	ParentPID    int                `json:"parent_pid,omitempty"`
	Host         string             `json:"host,omitempty"`
	CWD          string             `json:"cwd,omitempty"`
	Model        string             `json:"model,omitempty"`
	Agent        string             `json:"agent,omitempty"`
	Account      string             `json:"account,omitempty"`
	Argv         []string           `json:"argv,omitempty"`
	StartSHA     string             `json:"start_sha,omitempty"`
	Gateway      string             `json:"gateway,omitempty"`
	StartedAt    time.Time          `json:"started_at"`
	LastSeen     time.Time          `json:"last_seen"`
	Closed       bool               `json:"closed"`
	CloseReason  string             `json:"close_reason,omitempty"`
	Drive        *DriveCarry        `json:"drive,omitempty"` // the newest carried drive-state (nil = none)
	Registration *RegistrationCarry `json:"registration,omitempty"`
}

// FoldEvents folds the lifecycle log to one Session per id, applying events in event-time
// order so a reopen after a close (a resumed handle) clears the closed flag and re-stamps
// the start. Rows sort newest-started first, which is what a "what was running" view wants.
func FoldEvents(events []Event) []Session {
	sorted := append([]Event(nil), events...)
	sort.SliceStable(sorted, func(i, j int) bool { return eventTime(sorted[i]).Before(eventTime(sorted[j])) })

	byID := map[string]*Session{}
	order := []string{}
	for _, ev := range sorted {
		if ev.Kind == KindRefuse {
			continue // an audit refusal is not a session lifecycle transition
		}
		t := eventTime(ev)
		s, ok := byID[ev.ID]
		if !ok {
			s = &Session{ID: ev.ID}
			byID[ev.ID] = s
			order = append(order, ev.ID)
		}
		switch ev.Kind {
		case KindOpen:
			// A (re)open re-stamps the start, refreshes provenance, and clears any
			// prior close — a resumed session is live again.
			s.StartedAt = t
			s.Closed = false
			s.CloseReason = ""
			applyProvenance(s, ev)
		case KindBeat:
			applyProvenance(s, ev)
		case KindClose:
			applyProvenance(s, ev)
			s.Closed = true
			s.CloseReason = ev.Reason
		}
		if t.After(s.LastSeen) {
			s.LastSeen = t
		}
	}

	out := make([]Session, 0, len(order))
	for _, id := range order {
		out = append(out, *byID[id])
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].StartedAt.After(out[j].StartedAt) })
	return out
}

// applyProvenance copies the non-empty provenance fields of an event onto the session, so
// a later beat carrying a refreshed pid or the open's rich fields both land.
func applyProvenance(s *Session, ev Event) {
	if ev.Boot != "" {
		s.Boot = ev.Boot
	}
	if ev.PID != 0 {
		s.PID = ev.PID
	}
	if ev.ParentPID != 0 {
		s.ParentPID = ev.ParentPID
	}
	if ev.Host != "" {
		s.Host = ev.Host
	}
	if ev.CWD != "" {
		s.CWD = ev.CWD
	}
	if ev.Model != "" {
		s.Model = ev.Model
	}
	if ev.Agent != "" {
		s.Agent = ev.Agent
	}
	if ev.Account != "" {
		s.Account = ev.Account
	}
	if len(ev.Argv) > 0 {
		s.Argv = ev.Argv
	}
	if ev.StartSHA != "" {
		s.StartSHA = ev.StartSHA
	}
	if ev.Gateway != "" {
		s.Gateway = ev.Gateway
	}
	if ev.Registration != nil {
		r := *ev.Registration
		r.Scope = append([]string(nil), ev.Registration.Scope...)
		s.Registration = &r
	}
	if ev.Drive != nil {
		// Last non-nil carry wins (same last-write-wins fold as the scalar fields
		// above); copy by value so the folded Session never aliases the event's pointer.
		d := *ev.Drive
		s.Drive = &d
	}
}

// eventTime parses an event's RFC3339 TS, zero on failure (an unstamped row folds as
// the epoch rather than crashing the sort).
func eventTime(ev Event) time.Time {
	if t, err := time.Parse(time.RFC3339, strings.TrimSpace(ev.TS)); err == nil {
		return t.UTC()
	}
	return time.Time{}
}

// Status is a session's liveness verdict.
type Status string

const (
	StatusLive    Status = "LIVE"
	StatusCrashed Status = "CRASHED"
	StatusStale   Status = "STALE"
	StatusClosed  Status = "CLOSED"
)

// The closed reason vocabulary attached to a verdict.
const (
	ReasonCleanExit     = "CLEAN_EXIT"
	ReasonMachineReboot = "MACHINE_REBOOT" // started before the current boot — died in the reboot
	ReasonPIDDead       = "PID_DEAD"       // same boot, but the process is gone
	ReasonStaleBeat     = "STALE_BEAT"     // same boot, no recent heartbeat — ambiguous
	ReasonLive          = "LIVE"
)

// ClassifyConfig parameterizes the fold so it is fully testable with injected inputs.
type ClassifyConfig struct {
	Now        time.Time          // wall clock for the stale-beat window
	BootTime   time.Time          // machine current boot instant; zero = unknown (skips MACHINE_REBOOT)
	StaleAfter time.Duration      // same-boot last-seen older than this -> STALE; 0 disables
	PIDAlive   func(pid int) bool // optional same-boot liveness check; nil = skip (foundation passes nil)
}

// Classified is a session plus its verdict.
type Classified struct {
	Session
	Status Status `json:"status"`
	Reason string `json:"reason"`
}

// Classify applies the boot-epoch verdict to each session. The order is deliberate:
// a clean close is definitive; then the machine-wide-reboot test (started before the
// current boot) — the one that survives a reboot with no live process to probe; then
// the optional same-boot PID check; then the stale-beat ambiguity; else LIVE.
func Classify(sessions []Session, cfg ClassifyConfig) []Classified {
	out := make([]Classified, 0, len(sessions))
	for _, s := range sessions {
		st, reason := classifyOne(s, cfg)
		out = append(out, Classified{Session: s, Status: st, Reason: reason})
	}
	return out
}

func classifyOne(s Session, cfg ClassifyConfig) (Status, string) {
	if s.Closed {
		reason := s.CloseReason
		if reason == "" {
			reason = ReasonCleanExit
		}
		return StatusClosed, reason
	}
	if !cfg.BootTime.IsZero() && !s.StartedAt.IsZero() && s.StartedAt.Before(cfg.BootTime) {
		return StatusCrashed, ReasonMachineReboot
	}
	if cfg.PIDAlive != nil && s.PID > 0 && !cfg.PIDAlive(s.PID) {
		return StatusCrashed, ReasonPIDDead
	}
	if cfg.StaleAfter > 0 && !s.LastSeen.IsZero() && !cfg.Now.IsZero() && cfg.Now.Sub(s.LastSeen) > cfg.StaleAfter {
		return StatusStale, ReasonStaleBeat
	}
	return StatusLive, ReasonLive
}

// Counts tallies verdicts by status for the report header.
func Counts(cs []Classified) map[Status]int {
	m := map[Status]int{}
	for _, c := range cs {
		m[c.Status]++
	}
	return m
}

// BootID is a stable per-boot equality key: the boot instant bucketed to 60s to absorb
// tick/NTP jitter, so every event within one boot shares an id while a genuine reboot
// (which moves the boot instant by the prior uptime — minutes to hours) gets a new one.
// Empty when the boot time is unknown. Exactness under laptop-sleep / NTP drift is C6.
func BootID(bootTime time.Time) string {
	if bootTime.IsZero() {
		return ""
	}
	return fmt.Sprintf("boot-%d", bootTime.Unix()/60)
}
