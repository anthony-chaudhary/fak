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
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
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
	HostRecovery bool     `json:"host_recovery,omitempty"`
	ResumeHandle string   `json:"resume_handle,omitempty"`
	Command      []string `json:"command,omitempty"`
	WindowID     string   `json:"window_id,omitempty"`
	TabID        string   `json:"tab_id,omitempty"`
	GoalState    string   `json:"goal_state,omitempty"`
	LoopState    string   `json:"loop_state,omitempty"`
	EndedAt      string   `json:"ended_at,omitempty"`

	// GatewayURL and Bearer publish the session's live loopback gateway so a second
	// process can discover and authenticate to it from the index alone, with no prior
	// port knowledge. Bearer is read-scoped: it admits status reads, not control. Both
	// are omitempty — a session that has not bound a gateway simply omits them.
	GatewayURL string `json:"gateway_url,omitempty"`
	Bearer     string `json:"bearer,omitempty"`
}

// WithGateway stamps the published loopback gateway URL and its read-scoped bearer onto
// the row, returning it for chaining off NewRow. It is the one seam that records how an
// operator (or a sibling process) reaches a live session's status endpoint.
func (r Row) WithGateway(url, bearer string) Row {
	r.GatewayURL = strings.TrimSpace(url)
	r.Bearer = strings.TrimSpace(bearer)
	return r
}

// GatewayPublishEpoch is the UTC instant the PRODUCER half of gateway discovery landed
// (#5400): from that build forward, a `fak guard` launch re-records its row with
// gateway_url (and the read-scoped bearer) stamped once its listener is actually serving.
// It exists so a READER can tell the two causes of a missing gateway_url APART instead of
// guessing at one: a row started BEFORE this instant was written by a fak that had no
// publisher at all, so the field could not have been set; a row started AFTER it and still
// missing the field means that session published nothing — it bound no gateway, or its
// re-record failed. Same empty field, opposite diagnoses.
var GatewayPublishEpoch = time.Date(2026, 7, 28, 0, 0, 0, 0, time.UTC)

// PredatesGatewayPublish reports whether the row started before the producer existed, so a
// missing GatewayURL on it is a build fact rather than a session fault. A row whose start
// time is absent or unparseable counts as predating: an unstamped start is a legacy shape,
// and attributing it to a live session's publish would be the same wrong guess this
// distinction removes.
func (r Row) PredatesGatewayPublish() bool {
	t, err := time.Parse(time.RFC3339, strings.TrimSpace(r.StartedAt))
	if err != nil {
		return true
	}
	return t.Before(GatewayPublishEpoch)
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
// HostRecoveryEnabled marks the row as eligible for automatic terminal-host
// resurrection. Existing rows default off so rollout cannot fan out across
// every historical interactive session.
func HostRecoveryEnabled(row Row) bool { return row.HostRecovery }

func NewInteractiveRow(traceID, agent string, pid int, cwd, auditPath, nonce string, startedAt time.Time, command []string, hostRecovery bool) Row {
	r := NewRow(traceID, agent, pid, cwd, auditPath, nonce, startedAt)
	r.Interactive = true
	r.HostRecovery = hostRecovery
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
	b, err := json.Marshal(row)
	if err != nil {
		f.Close()
		return err
	}
	if _, err := f.Write(append(b, '\n')); err != nil {
		f.Close()
		return err
	}
	// Close the append handle BEFORE compacting: the rewrite renames a temp file over the
	// index, which a lingering write handle would refuse on Windows (sharing violation).
	if err := f.Close(); err != nil {
		return err
	}
	// Best-effort, size-gated fold-and-rewrite AFTER the append has durably landed. The
	// index must not grow without bound, but a compaction failure must NEVER fail a guard
	// launch (Record's best-effort contract), so its result is deliberately discarded and
	// the append's success is what Record reports.
	_, _ = Compact(regDir)
	return nil
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
	rows, _ := foldReader(f)
	return rows
}

// foldReader is the SINGLE fold both the read path (LoadFile) and the rewrite path
// (CompactFile) share, so a compaction can never diverge from what a reader computes. It
// folds the append-only JSONL stream to the latest row per handle (a later row for the same
// handle — a re-record — wins), skipping blank/malformed/wrong-schema lines exactly as the
// query side does, then sorts newest-start first. rawLines counts every non-blank line it
// saw (valid or not), so a caller can weigh the append log against its folded footprint.
func foldReader(r io.Reader) (rows []Row, rawLines int) {
	latest := map[string]Row{}
	order := []string{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		rawLines++
		var row Row
		if json.Unmarshal([]byte(line), &row) != nil {
			continue
		}
		if row.Schema != Schema || strings.TrimSpace(row.Handle) == "" {
			continue
		}
		if _, seen := latest[row.Handle]; !seen {
			order = append(order, row.Handle)
		}
		latest[row.Handle] = row
	}
	rows = make([]Row, 0, len(order))
	for _, h := range order {
		rows = append(rows, latest[h])
	}
	// Newest start first (stable): the most recently launched session sorts to the top,
	// which is what an operator listing "what is running" wants to see.
	sort.SliceStable(rows, func(i, j int) bool {
		return startUnix(rows[i]) > startUnix(rows[j])
	})
	return rows, rawLines
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

// The compaction gate constants mirror the sibling resume_ledger.jsonl inline compaction
// (cmd/fak/resume_watchdog_runtime.go rwCompactResumeLedger, #3497): a cheap byte early-out
// keeps small files off the read path, and a redundancy check keeps a file already near its
// folded size byte-untouched. Held conservatively high so a rewrite is a rare, amortized
// event, not a per-append cost.
//
//   - compactMinBytes: matches #3497's 512 KiB FAK_RESUME_LEDGER_COMPACT_BYTES default. Below
//     this size the index is left alone with only an os.Stat spent (no read).
//   - compactMinLines / compactMultiple: rewrite only when the raw line count exceeds
//     max(compactMinLines, compactMultiple*distinctHandles) — i.e. the log carries several
//     superseded rows per live handle. A near-folded file (few dead rows) never rewrites.
const (
	compactMinBytes = 512 * 1024
	compactMinLines = 64
	compactMultiple = 4
)

// Compact runs the size-gated fold-and-rewrite over the guard-session index under regDir.
// It is the best-effort seam Record triggers after an append; a caller may also invoke it
// directly. Returns the number of superseded lines the fold collapsed (0 when the gate held).
func Compact(regDir string) (int, error) {
	return CompactFile(IndexPath(regDir))
}

// CompactFile rewrites the append-only guard-session index at path to exactly its folded
// LoadFile output — one row per handle, newest-start first — when the raw line count has
// outgrown its folded footprint. The rewrite is LOSSLESS by construction: it writes the
// same set every reader already computes (foldReader is the shared fold), so a superseded
// row it drops was already dead weight the read path ignored. It returns the number of
// superseded lines collapsed (0 when the gate held or nothing was rewritten) and never
// surfaces an internal I/O error as a failure — a compaction that cannot proceed simply
// leaves the append-only file in place.
//
// Concurrency: the box is shared — many `fak guard` launches append to this file
// concurrently — so a naive read-fold-rename can DROP a row appended between the snapshot
// read and the rename. This is handled the way #3497 relies on a conservative gate plus an
// atomic rename, and tightened here for the per-append trigger: (1) any bytes appended past
// the snapshot offset are carried verbatim into the rewrite before the rename, so a row that
// arrives during the fold survives (LoadFile re-folds them on the next read, so a duplicate
// or newer handle simply wins there); (2) the temp file has a unique name, so two guard
// launches that cross the gate at once cannot clobber each other's staging file. A RESIDUAL
// micro-window remains — a line appended after the tail re-read but before os.Rename is lost
// — and is NOT fully closed: it is bounded by the conservative gate (a rewrite is rare) and
// self-heals, since a dropped guard-session row re-records on the session's next lifecycle
// transition and the atomic rename never exposes a torn file to a reader.
func CompactFile(path string) (int, error) {
	st, err := os.Stat(path)
	if err != nil || st.Size() < compactMinBytes {
		return 0, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, nil
	}
	snap := int64(len(data))
	rows, rawLines := foldReader(bytes.NewReader(data))
	distinct := len(rows)
	limit := compactMinLines
	if m := compactMultiple * distinct; m > limit {
		limit = m
	}
	if rawLines <= limit {
		// Already near its folded size: leave the file BYTE-untouched.
		return 0, nil
	}

	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	for _, row := range rows {
		if err := enc.Encode(row); err != nil {
			return 0, nil
		}
	}
	// Carry any lines appended after the snapshot so a concurrent O_APPEND writer's row is
	// not lost to the rename. The tail is copied verbatim; LoadFile re-folds it on the next
	// read. A line appended after THIS read (before the rename) is the residual micro-window
	// documented above.
	if tf, terr := os.Open(path); terr == nil {
		if _, serr := tf.Seek(snap, io.SeekStart); serr == nil {
			if tail, rerr := io.ReadAll(tf); rerr == nil && len(bytes.TrimSpace(tail)) > 0 {
				buf.Write(tail)
			}
		}
		tf.Close()
	}

	tmp, err := os.CreateTemp(filepath.Dir(path), IndexFileName+".compact-*")
	if err != nil {
		return 0, nil
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(buf.Bytes()); err != nil {
		tmp.Close()
		_ = os.Remove(tmpName)
		return 0, nil
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpName)
		return 0, nil
	}
	if err := os.Rename(tmpName, path); err != nil {
		_ = os.Remove(tmpName)
		return 0, nil
	}
	return rawLines - distinct, nil
}
