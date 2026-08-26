// Package serviceledger is the portable append-only observed-state event
// ledger for services (#4753, parent #4748, depends on #4749). Windows Event
// 1000 host-crash rows, SCM state changes, systemd journal entries, launchd
// exit messages, and fak's own supervisor receipts each prove one sensor with
// unrelated identifiers; this leaf folds them all into ONE versioned event
// schema (fak.service.events.v1) correlated by servicespec identity plus boot
// ID, manager invocation, PID, generation, lease token hash, checkpoint,
// request, and receipt — so an operator can answer whether a node crashed,
// rebooted, restarted, and resumed the same logical work.
//
// Exact-once and replay: every event carries a (source, source_uid) pair; the
// ledger refuses a second copy of the same pair, so re-ingesting the same
// native export (or replaying a partial one) is idempotent. Events without a
// native UID are keyed by a content digest, so identical synthetic events
// replay idempotently too.
//
// Redaction: Normalize redacts command-line secret values (token/password/
// key/credential assignments, bearer tokens) and private infrastructure
// identifiers (RFC1918 addresses, *.internal/*.corp/*.lan/*.local hosts) from
// the free-text detail before anything is persisted. Lease tokens are never
// stored — only HashLeaseToken digests.
//
// Live capture per native manager pipes the manager's own export into the
// fixture-tested adapters (adapters.go):
//
//	Windows:  wevtutil qe System /f:xml | fak service events --ingest windows-xml --file -
//	Windows scheduled tasks: wevtutil qe Microsoft-Windows-TaskScheduler/Operational /f:xml | fak service events --ingest windows-xml --file -
//	systemd:  journalctl -u UNIT -o json         | fak service events --ingest journald-json --file -
//	launchd:  log show --style ndjson ...        | fak service events --ingest launchd-ndjson --file -
//
// The parsers are witnessed against golden fixtures on every platform; the
// journald and launchd LIVE streams need Linux/macOS hosts and are exercised
// there by operators, not by this test suite.
package serviceledger

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

// EventSchemaV1 names the versioned append-only observed-event schema.
const EventSchemaV1 = "fak.service.events.v1"

// EventType is the closed vocabulary of observed-state changes (#4753 "Done
// when" list). No event is left untyped.
type EventType string

const (
	EventDesiredChange   EventType = "desired-change"   // operator/reconciler intent changed
	EventProcessExit     EventType = "process-exit"     // a run ended (classified by servicespec.ExitClass)
	EventWatchdogTimeout EventType = "watchdog-timeout" // liveness/readiness deadline missed
	EventManagerRestart  EventType = "manager-restart"  // the platform manager (re)launched the workload
	EventBootChange      EventType = "boot-change"      // host boot / incarnation changed
	EventLeaseFence      EventType = "lease-fence"      // a newer generation fenced prior owners
	EventReadiness       EventType = "readiness"        // observed phase report (servicespec.Phase)
	EventCheckpoint      EventType = "checkpoint"       // durable checkpoint written
	EventResume          EventType = "resume"           // workload resumed prior logical work
	EventCircuitOpen     EventType = "circuit-open"     // restart window exhausted; workload fenced
)

// AllEventTypes enumerates the vocabulary (stable order).
var AllEventTypes = []EventType{
	EventDesiredChange, EventProcessExit, EventWatchdogTimeout, EventManagerRestart,
	EventBootChange, EventLeaseFence, EventReadiness, EventCheckpoint, EventResume, EventCircuitOpen,
}

// Correlation carries every cross-sensor identifier the issue names beyond the
// servicespec identity, so unrelated sensors can be joined into one causal
// timeline. All fields are optional — a sensor reports what it knows.
type Correlation struct {
	BootID            string `json:"boot_id,omitempty"`
	ManagerInvocation string `json:"manager_invocation,omitempty"`
	PID               int    `json:"pid,omitempty"`
	Generation        int64  `json:"generation,omitempty"`
	LeaseTokenHash    string `json:"lease_token_hash,omitempty"`
	Checkpoint        string `json:"checkpoint,omitempty"`
	Request           string `json:"request,omitempty"`
	Receipt           string `json:"receipt,omitempty"`
	// Session is the resumed interactive session identity (#4756): which
	// desktop session lineage the workload re-entered after a bridge resume.
	Session string `json:"session,omitempty"`
}

// Event is one fak.service.events.v1 row. The identity axis reuses the landed
// servicespec.Identity; phase and exit vocabulary reuse servicespec so the
// desired/observed split of #4749 carries straight into the ledger.
type Event struct {
	Schema      string                   `json:"schema"`
	Seq         uint64                   `json:"seq,omitempty"` // ledger-assigned, monotonic
	Type        EventType                `json:"type"`
	AtUnixMS    int64                    `json:"at_unix_ms"`
	Source      string                   `json:"source"`               // sensor: windows-eventlog | journald | launchd | fak | ...
	SourceUID   string                   `json:"source_uid,omitempty"` // exact-once key within Source
	Identity    servicespec.Identity     `json:"identity"`
	Correlation Correlation              `json:"correlation"`
	Desired     servicespec.DesiredState `json:"desired,omitempty"` // EventDesiredChange
	Phase       servicespec.Phase        `json:"phase,omitempty"`   // EventReadiness
	Exit        *servicespec.ExitRecord  `json:"exit,omitempty"`    // EventProcessExit
	Detail      string                   `json:"detail,omitempty"`  // redacted free text from the sensor
}

// Normalize fills schema/workload defaults and redacts the free-text detail.
// Idempotent; called by Append before validation and persistence.
func (e *Event) Normalize() {
	if e.Schema == "" {
		e.Schema = EventSchemaV1
	}
	if e.Identity.Workload == "" {
		e.Identity.Workload = e.Identity.Service
	}
	e.Detail = Redact(e.Detail)
}

// Validate checks one normalized event against the v1 contract.
func (e *Event) Validate() error {
	if e.Schema != EventSchemaV1 {
		return fmt.Errorf("serviceledger: schema %q is not %q", e.Schema, EventSchemaV1)
	}
	ok := false
	for _, t := range AllEventTypes {
		if e.Type == t {
			ok = true
			break
		}
	}
	if !ok {
		return fmt.Errorf("serviceledger: unknown event type %q", e.Type)
	}
	if strings.TrimSpace(e.Identity.Node) == "" || strings.TrimSpace(e.Identity.Service) == "" {
		return errors.New("serviceledger: identity.node and identity.service are required")
	}
	if e.AtUnixMS <= 0 {
		return errors.New("serviceledger: at_unix_ms is required")
	}
	if e.Source == "" {
		return errors.New("serviceledger: source is required")
	}
	if e.Correlation.PID < 0 || e.Correlation.Generation < 0 {
		return errors.New("serviceledger: correlation pid/generation must be non-negative")
	}
	switch e.Type {
	case EventDesiredChange:
		if e.Desired != servicespec.DesiredRunning && e.Desired != servicespec.DesiredStopped {
			return fmt.Errorf("serviceledger: desired-change carries invalid desired %q", e.Desired)
		}
	case EventReadiness:
		valid := false
		for _, p := range servicespec.AllPhases {
			if e.Phase == p {
				valid = true
				break
			}
		}
		if !valid {
			return fmt.Errorf("serviceledger: readiness carries invalid phase %q", e.Phase)
		}
	case EventProcessExit:
		if e.Exit == nil {
			return errors.New("serviceledger: process-exit carries no exit record")
		}
		valid := false
		for _, c := range servicespec.AllExitClasses {
			if e.Exit.Class == c {
				valid = true
				break
			}
		}
		if !valid {
			return fmt.Errorf("serviceledger: process-exit carries invalid class %q", e.Exit.Class)
		}
	}
	return nil
}

// HashLeaseToken digests a raw lease token for Correlation.LeaseTokenHash.
// The ledger stores only this digest; the raw token never touches disk.
func HashLeaseToken(token string) string {
	sum := sha256.Sum256([]byte(token))
	return hex.EncodeToString(sum[:])
}

// dedupeKey is the exact-once key: (source, source_uid), falling back to a
// content digest when the sensor has no native UID.
func (e *Event) dedupeKey() string {
	uid := e.SourceUID
	if uid == "" {
		c := *e
		c.Seq = 0
		b, _ := json.Marshal(&c)
		sum := sha256.Sum256(b)
		uid = "content-" + hex.EncodeToString(sum[:])
	}
	return e.Source + "\x00" + uid
}

// Ledger is the append-only observed-event store: one JSON line per event,
// with an in-memory exact-once index rebuilt on Open.
type Ledger struct {
	mu       sync.Mutex
	path     string // empty = in-memory only
	recovery RecoveryReceipt
	seen     map[string]struct{}
	events   []Event
	nextSeq  uint64
}

// DefaultDir resolves the durable ledger directory.
func DefaultDir() string {
	if d := os.Getenv("FAK_SERVICE_LEDGER_DIR"); d != "" {
		return d
	}
	if d, err := os.UserConfigDir(); err == nil {
		return filepath.Join(d, "fak", "service-ledger")
	}
	return filepath.Join(os.TempDir(), "fak", "service-ledger")
}

// Memory returns an unpersisted ledger (tests, dry runs).
func Memory() *Ledger { return &Ledger{seen: map[string]struct{}{}} }

// RecoveryReceipt reports a bounded startup repair without exposing event contents.
type RecoveryReceipt struct {
	DiscardedTailBytes int64  `json:"discarded_tail_bytes,omitempty"`
	RecoveredSequence  uint64 `json:"recovered_sequence,omitempty"`
	CorruptionClass    string `json:"corruption_class,omitempty"`
	OperatorAction     string `json:"operator_action,omitempty"`
}

// Recovery returns the startup recovery receipt. A zero value means no repair was needed.
func (l *Ledger) Recovery() RecoveryReceipt { return l.recovery }

type diskRecord struct {
	Version  int             `json:"version"`
	Event    json.RawMessage `json:"event"`
	Checksum string          `json:"sha256"`
}

// Open loads (or lazily creates) the ledger under dir, replaying events.jsonl
// to rebuild the exact-once index. Malformed lines are refused, not skipped:
// an append-only ledger that silently drops rows cannot witness anything.
func Open(dir string) (*Ledger, error) {
	l := &Ledger{path: filepath.Join(dir, "events.jsonl"), seen: map[string]struct{}{}}
	f, err := os.Open(l.path)
	if errors.Is(err, os.ErrNotExist) {
		return l, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	line := 0
	var validBytes int64
	for sc.Scan() {
		line++
		rawBytes := append([]byte(nil), sc.Bytes()...)
		recordEnd := validBytes + int64(len(rawBytes)) + 1
		if recordEnd > info.Size() {
			break // final line lacks a commit newline: an unacknowledged torn tail
		}
		validBytes = recordEnd
		raw := strings.TrimSpace(string(rawBytes))
		if raw == "" {
			continue
		}
		e, err := decodeRecord([]byte(raw))
		if err != nil {
			return nil, fmt.Errorf("serviceledger: interior corruption in %s line %d: %w", l.path, line, err)
		}
		l.seen[e.dedupeKey()] = struct{}{}
		l.events = append(l.events, e)
		if e.Seq >= l.nextSeq {
			l.nextSeq = e.Seq
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if info.Size() > validBytes {
		discarded := info.Size() - validBytes
		if err := f.Close(); err != nil {
			return nil, err
		}
		if err := os.Truncate(l.path, validBytes); err != nil {
			return nil, fmt.Errorf("serviceledger: repair truncated tail: %w", err)
		}
		l.recovery = RecoveryReceipt{DiscardedTailBytes: discarded, RecoveredSequence: l.nextSeq, CorruptionClass: "truncated_tail", OperatorAction: "none; append was not acknowledged"}
	}
	return l, nil
}

// Append normalizes, validates, and durably appends one event. It returns the
// stored event and ingested=false (without error) when the (source, uid) pair
// was already ledgered — the exact-once / idempotent-replay contract.
func (l *Ledger) Append(ev Event) (Event, bool, error) {
	ev.Normalize()
	if err := ev.Validate(); err != nil {
		return Event{}, false, err
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	key := ev.dedupeKey()
	if _, dup := l.seen[key]; dup {
		return ev, false, nil
	}
	l.nextSeq++
	ev.Seq = l.nextSeq
	if l.path != "" {
		if err := l.persistLocked(ev); err != nil {
			l.nextSeq--
			return Event{}, false, err
		}
	}
	l.seen[key] = struct{}{}
	l.events = append(l.events, ev)
	return ev, true, nil
}

// AppendAll appends a batch (an adapter's parse), reporting how many rows were
// newly ingested versus already ledgered.
func (l *Ledger) AppendAll(evs []Event) (ingested, duplicates int, err error) {
	for _, ev := range evs {
		_, ok, e := l.Append(ev)
		if e != nil {
			return ingested, duplicates, e
		}
		if ok {
			ingested++
		} else {
			duplicates++
		}
	}
	return ingested, duplicates, nil
}

// Events returns a copy of all ledgered events in append order.
func (l *Ledger) Events() []Event {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make([]Event, len(l.events))
	copy(out, l.events)
	return out
}

func decodeRecord(raw []byte) (Event, error) {
	var framed diskRecord
	if err := json.Unmarshal(raw, &framed); err == nil && framed.Version != 0 {
		if framed.Version != 1 || len(framed.Event) == 0 || framed.Checksum == "" {
			return Event{}, fmt.Errorf("invalid record framing version %d", framed.Version)
		}
		sum := sha256.Sum256(framed.Event)
		if !strings.EqualFold(framed.Checksum, hex.EncodeToString(sum[:])) {
			return Event{}, errors.New("record checksum mismatch")
		}
		var ev Event
		if err := json.Unmarshal(framed.Event, &ev); err != nil {
			return Event{}, fmt.Errorf("framed event: %w", err)
		}
		return ev, nil
	}
	var ev Event // Read legacy unframed rows written before durability framing.
	if err := json.Unmarshal(raw, &ev); err != nil {
		return Event{}, err
	}
	return ev, nil
}

func syncParentDir(dir string) error {
	if runtime.GOOS == "windows" {
		return nil // NTFS FlushFileBuffers on the created file is the portable boundary.
	}
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}
func (l *Ledger) persistLocked(ev Event) error {
	if err := os.MkdirAll(filepath.Dir(l.path), 0o700); err != nil {
		return err
	}
	eventBytes, err := json.Marshal(&ev)
	if err != nil {
		return err
	}
	sum := sha256.Sum256(eventBytes)
	recordBytes, err := json.Marshal(diskRecord{Version: 1, Event: eventBytes, Checksum: hex.EncodeToString(sum[:])})
	if err != nil {
		return err
	}
	_, statErr := os.Stat(l.path)
	created := errors.Is(statErr, os.ErrNotExist)
	f, err := os.OpenFile(l.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	if _, err = f.Write(append(recordBytes, '\n')); err != nil {
		f.Close()
		return err
	}
	if err = f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	if created {
		return syncParentDir(filepath.Dir(l.path))
	}
	return nil
}

// Redaction: command-line secret values and private infrastructure
// identifiers must never be ledgered. Conservative by design — a redacted
// benign value costs nothing; a leaked secret cannot be unshipped.
var (
	reSecretKV  = regexp.MustCompile(`(?i)((?:--?|/)?[\w.-]*(?:token|secret|passw(?:or)?d|api[-_]?key|credential|auth)[\w.-]*\s*[=:]\s*)\S+`)
	reBearer    = regexp.MustCompile(`(?i)\b(bearer)\s+\S+`)
	rePrivHost  = regexp.MustCompile(`(?i)\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\.(?:internal|corp|lan|local|intranet)\b`)
	rePrivIPv4  = regexp.MustCompile(`\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b`)
	redactedTag = "[REDACTED]"
)

// Redact strips secret-looking command-line values, bearer/authorization
// tokens, private-suffix hostnames, and RFC1918 addresses from free text.
func Redact(s string) string {
	if s == "" {
		return s
	}
	// Bearer first: "Authorization: Bearer <tok>" must lose the token itself,
	// not just the scheme word, before the key=value pass runs.
	s = reBearer.ReplaceAllString(s, "${1} "+redactedTag)
	s = reSecretKV.ReplaceAllString(s, "${1}"+redactedTag)
	s = rePrivHost.ReplaceAllString(s, "[REDACTED-HOST]")
	s = rePrivIPv4.ReplaceAllString(s, "[REDACTED-IP]")
	return s
}
