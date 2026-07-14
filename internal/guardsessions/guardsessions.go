// Package guardsessions is the local, queryable INDEX of `fak guard` sessions — the
// durable answer to "which guard sessions are running (or recently ran) on this box, and
// how do I reference one?". Every `fak guard` launch appends one row (a short stable
// HANDLE, the trace id, the wrapped agent, pid, cwd, the audit-journal path, and the start
// time) to an append-only JSONL index; `fak guard sessions` folds and lists it, and
// Resolve turns a short prefix (of the handle OR the trace id) into the one session it
// names — so an operator can say `fak guard sessions <prefix>` instead of scraping Slack or
// grepping the outbox for the nonce.
//
// The package is pure and stdlib-only over an injected filesystem path: Record appends,
// Load folds the append-only log to the latest row per handle, and Resolve does the
// prefix/handle/trace match. It imports nothing internal, off the hot path — the guard
// start pays one small append, the query side reads the file.
package guardsessions

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Schema tags the index rows so a forward-extended reader can tell a guard-session row from
// any other JSONL that might share the file.
const Schema = "fak.guard-session.v1"

// IndexFileName is the durable index basename under the registry dir.
const IndexFileName = "guard_sessions.jsonl"

// Row is one recorded guard session. Only the fields the index needs are typed; a
// forward-extended row (extra keys) still decodes. Handle is the short, stable,
// human-referenceable id; TraceID is the guardTraceID the gateway uses; the rest is
// provenance an operator wants when picking a session to act on.
type Row struct {
	Schema    string `json:"schema"`
	Handle    string `json:"handle"`
	TraceID   string `json:"trace_id"`
	Agent     string `json:"agent"`
	PID       int    `json:"pid"`
	CWD       string `json:"cwd,omitempty"`
	AuditPath string `json:"audit,omitempty"`
	StartedAt string `json:"started_utc"`
	Nonce     string `json:"nonce,omitempty"`

	// Relaunch fields are the OS-independent contract consumed by the host-crash
	// actuator. Interactive distinguishes operator tabs from dispatcher-owned workers.
	Interactive  bool     `json:"interactive,omitempty"`
	ResumeHandle string   `json:"resume_handle,omitempty"`
	Command      []string `json:"command,omitempty"`
	WindowID     string   `json:"window_id,omitempty"`
	TabID        string   `json:"tab_id,omitempty"`
	GoalState    string   `json:"goal_state,omitempty"`
	LoopState    string   `json:"loop_state,omitempty"`
	EndedAt      string   `json:"ended_at,omitempty"`
}

// Handle derives a short, stable, human-referenceable id for a guard session from its
// trace id and start time. It is deterministic (same trace+start → same handle) and short
// enough to type, but seeded by the start instant so two sessions that reuse the same
// default trace id ("guard") on one box still get distinct handles. The form is "g" + the
// first 8 hex of sha256(traceID|unixNano) — a git-short-sha feel an operator can prefix-match.
func Handle(traceID string, startedAt time.Time) string {
	seed := fmt.Sprintf("%s|%d", strings.TrimSpace(traceID), startedAt.UTC().UnixNano())
	sum := sha256.Sum256([]byte(seed))
	return "g" + hex.EncodeToString(sum[:])[:8]
}

// NewRow builds a fully-populated index row, assigning the derived handle. startedAt is
// stamped UTC RFC3339; a zero time is treated as now by the caller before this (Record
// does not read the clock — it stays pure over its inputs).
func NewRow(traceID, agent string, pid int, cwd, auditPath, nonce string, startedAt time.Time) Row {
	return Row{
		Schema:    Schema,
		Handle:    Handle(traceID, startedAt),
		TraceID:   strings.TrimSpace(traceID),
		Agent:     strings.TrimSpace(agent),
		PID:       pid,
		CWD:       strings.TrimSpace(cwd),
		AuditPath: strings.TrimSpace(auditPath),
		StartedAt: startedAt.UTC().Format(time.RFC3339),
		Nonce:     strings.TrimSpace(nonce),
	}
}

// IndexPath is the index file under a registry dir.
func IndexPath(regDir string) string { return filepath.Join(regDir, IndexFileName) }

// NewInteractiveRow builds the durable relaunch specification for an operator-owned
// Guard session. Command is an argv vector, not a shell string, so the actuator can
// reconstruct it without lossy quoting or command injection.
func NewInteractiveRow(traceID, agent string, pid int, cwd, auditPath, nonce string, startedAt time.Time, command []string) Row {
	r := NewRow(traceID, agent, pid, cwd, auditPath, nonce, startedAt)
	r.Interactive = true
	r.ResumeHandle = r.Handle
	r.Command = append([]string(nil), command...)
	r.WindowID = strings.TrimSpace(os.Getenv("WT_WINDOW"))
	r.TabID = strings.TrimSpace(os.Getenv("WT_TAB_ID"))
	r.GoalState = strings.TrimSpace(os.Getenv("FAK_GOAL_ID"))
	r.LoopState = strings.TrimSpace(os.Getenv("FAK_LOOP_ID"))
	return r
}

// Ended returns a clean-exit tombstone. A host crash cannot execute this transition,
// so its latest row remains live for the resurrection watchdog.
func (r Row) Ended(at time.Time) Row {
	if at.IsZero() {
		at = time.Now()
	}
	r.EndedAt = at.UTC().Format(time.RFC3339Nano)
	return r
}

// Record appends one row to the index under regDir, creating the dir and file as needed.
// Best-effort by contract: a guard launch must never fail because its index append failed,
// so the caller ignores the returned error for anything but diagnostics. The append is a
// single Write of one line, so concurrent guard starts interleave at line granularity
// (O_APPEND) without corrupting rows.
func Record(regDir string, row Row) error {
	if row.Schema == "" {
		row.Schema = Schema
	}
	if err := os.MkdirAll(regDir, 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(IndexPath(regDir), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	b, err := json.Marshal(row)
	if err != nil {
		return err
	}
	_, err = f.Write(append(b, '\n'))
	return err
}

// Load folds the append-only index into the latest row per handle, newest-start first. A
// missing/unreadable file yields no rows (never an error): an absent index is simply no
// recorded sessions. Malformed lines and rows carrying the wrong schema are skipped, so a
// foreign line in the file can never surface as a fake session.
func Load(regDir string) []Row {
	return LoadFile(IndexPath(regDir))
}

// LoadFile is Load against an explicit path (the testable core).
func LoadFile(path string) []Row {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()

	// Fold to the latest row per handle: a later row for the same handle (a re-record)
	// wins, so the index stays correct even if a session is recorded more than once.
	latest := map[string]Row{}
	order := []string{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var r Row
		if json.Unmarshal([]byte(line), &r) != nil {
			continue
		}
		if r.Schema != Schema || strings.TrimSpace(r.Handle) == "" {
			continue
		}
		if _, seen := latest[r.Handle]; !seen {
			order = append(order, r.Handle)
		}
		latest[r.Handle] = r
	}
	rows := make([]Row, 0, len(order))
	for _, h := range order {
		rows = append(rows, latest[h])
	}
	// Newest start first (stable): the most recently launched session sorts to the top,
	// which is what an operator listing "what is running" wants to see.
	sort.SliceStable(rows, func(i, j int) bool {
		return startUnix(rows[i]) > startUnix(rows[j])
	})
	return rows
}

// startUnix parses a row's RFC3339 start into unix seconds, 0 on any parse failure (so an
// unstamped row sorts last rather than crashing the sort).
func startUnix(r Row) int64 {
	if t, err := time.Parse(time.RFC3339, strings.TrimSpace(r.StartedAt)); err == nil {
		return t.Unix()
	}
	return 0
}

// ResolveResult is the outcome of resolving a query string against the index.
type ResolveResult struct {
	// Row is the single matched session (valid only when Matched == 1).
	Row Row
	// Matched is how many sessions the query matched: 0 (none), 1 (unambiguous — Row is
	// set), or >1 (ambiguous — Candidates lists them so the caller can report the tie).
	Matched    int
	Candidates []Row
}

// Resolve turns a query into the one session it names. The match order is:
//  1. an EXACT handle or trace-id equality (a full id always wins, even if it is also a
//     prefix of a longer one), else
//  2. a case-insensitive PREFIX of the handle or the trace id.
//
// A query matching exactly one session returns Matched==1 with Row set; matching several
// returns Matched==len(Candidates) so the caller can print the ambiguity; matching none
// returns Matched==0. An empty query matches nothing. This is the "reference a specific
// session by a short prefix" resolver the goal asks for.
func Resolve(rows []Row, query string) ResolveResult {
	q := strings.TrimSpace(query)
	if q == "" {
		return ResolveResult{}
	}
	// Exact id wins outright.
	for _, r := range rows {
		if r.Handle == q || r.TraceID == q {
			return ResolveResult{Row: r, Matched: 1, Candidates: []Row{r}}
		}
	}
	lq := strings.ToLower(q)
	var hits []Row
	for _, r := range rows {
		if strings.HasPrefix(strings.ToLower(r.Handle), lq) ||
			(r.TraceID != "" && strings.HasPrefix(strings.ToLower(r.TraceID), lq)) {
			hits = append(hits, r)
		}
	}
	switch len(hits) {
	case 0:
		return ResolveResult{Matched: 0}
	case 1:
		return ResolveResult{Row: hits[0], Matched: 1, Candidates: hits}
	default:
		return ResolveResult{Matched: len(hits), Candidates: hits}
	}
}

// LiveInteractive returns actuator-ready sessions whose latest lifecycle row is not a
// clean-exit tombstone. Dispatched workers and incomplete legacy rows are excluded.
func LiveInteractive(rows []Row) []Row {
	out := make([]Row, 0, len(rows))
	for _, r := range rows {
		if r.Interactive && strings.TrimSpace(r.EndedAt) == "" && strings.TrimSpace(r.CWD) != "" && len(r.Command) > 0 && strings.TrimSpace(r.ResumeHandle) != "" {
			out = append(out, r)
		}
	}
	return out
}
