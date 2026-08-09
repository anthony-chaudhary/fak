// Package sessiondiag classifies bounded, redacted evidence from Codex's local
// structured log and SQLite store. It never consumes or emits log bodies.
package sessiondiag

import (
	"strings"
	"time"
)

const Schema = "fak.sessiondiag.v1"

type Evidence struct {
	DBBasename       string `json:"db_basename"`
	ProcessCount     int64  `json:"process_count"`
	DBBytes          int64  `json:"db_bytes"`
	WALBytes         int64  `json:"wal_bytes"`
	PageSize         int64  `json:"page_size"`
	PageCount        int64  `json:"page_count"`
	FreelistPages    int64  `json:"freelist_pages"`
	RecentRows       int64  `json:"recent_rows"`
	QueueDrops       int64  `json:"queue_drops"`
	SlowWrites       int64  `json:"slow_writes"`
	ExplicitFailures int64  `json:"explicit_failures"`
	WindowSeconds    int64  `json:"window_seconds"`
	Integrity        string `json:"integrity"`
}

type Finding struct {
	Code     string `json:"code"`
	Severity string `json:"severity"`
	Count    int64  `json:"count,omitempty"`
	Detail   string `json:"detail"`
	Action   string `json:"action"`
}

type Report struct {
	Schema         string    `json:"schema"`
	Verdict        string    `json:"verdict"`
	Causality      string    `json:"causality"`
	ObservedAt     time.Time `json:"observed_at"`
	Evidence       Evidence  `json:"evidence"`
	Findings       []Finding `json:"findings"`
	ReadOnly       bool      `json:"read_only"`
	MutationNotice string    `json:"mutation_notice"`
}

func Classify(e Evidence, now time.Time) Report {
	r := Report{Schema: Schema, Verdict: "NO_FAULT_EVIDENCE", Causality: "not_established", ObservedAt: now.UTC(), Evidence: e, ReadOnly: true, MutationNotice: "diagnosis did not checkpoint, vacuum, compact, or otherwise mutate the Codex store"}
	add := func(code, sev string, count int64, detail, action string) {
		r.Findings = append(r.Findings, Finding{code, sev, count, detail, action})
	}
	if e.Integrity != "ok" && e.Integrity != "not_checked" && e.Integrity != "" {
		add("STORE_INTEGRITY_FAILURE", "critical", 1, "Codex log-store integrity check did not return ok", "stop writers, copy the DB/WAL/SHM set, and recover from the copy")
	}
	reclaim := e.FreelistPages * e.PageSize
	if e.DBBytes >= 512<<20 && reclaim >= 256<<20 && e.PageCount > 0 && e.FreelistPages*2 >= e.PageCount {
		add("LOG_STORE_RECLAIMABLE_PRESSURE", "warning", reclaim, "the log store is large and at least half its pages are reclaimable", "after all Codex processes stop, back up the DB/WAL/SHM set and compact the copy or use the documented shutdown-time procedure")
	}
	if e.WALBytes >= 128<<20 {
		add("LOG_WAL_PRESSURE", "warning", e.WALBytes, "the live write-ahead log is large", "reduce concurrent Codex writers; checkpoint or compact only after all Codex processes stop")
	}
	if e.SlowWrites > 0 {
		add("LOG_WRITE_CONTENTION", "warning", e.SlowWrites, "Codex recorded slow structured-log writes in the bounded window", "correlate the next abrupt exit with this timestamp window and writer count")
	}
	if e.QueueDrops > 0 {
		add("APP_SERVER_EVENT_LOSS", "error", e.QueueDrops, "Codex dropped in-process app-server events because its consumer queue was full", "reduce concurrent/high-volume sessions and preserve the next affected process UUID and timestamps")
	}
	if e.ExplicitFailures > 0 {
		add("EXPLICIT_PROCESS_FAILURE", "critical", e.ExplicitFailures, "Codex recorded an explicit panic, fatal runtime failure, or fak child exit", "inspect the matching process/thread crash record; this is direct failure evidence")
	}
	if len(r.Findings) == 0 {
		add("INSUFFICIENT_CRASH_EVIDENCE", "info", 0, "no explicit failure or configured pressure threshold was found in the bounded window", "rerun immediately after an abrupt exit and preserve OS event evidence")
	}
	if e.ExplicitFailures > 0 {
		r.Verdict = "EXPLICIT_FAILURE_EVIDENCE"
		r.Causality = "failure_recorded"
	} else if e.QueueDrops > 0 || e.SlowWrites > 0 || reclaim >= 256<<20 || e.WALBytes >= 128<<20 {
		r.Verdict = "CORRELATED_RUNTIME_PRESSURE"
	}
	if e.Integrity != "ok" && e.Integrity != "not_checked" && e.Integrity != "" {
		r.Verdict = "STORE_INTEGRITY_FAILURE"
	}
	return r
}

const IncidentSchema = "fak-session-incident/1"

type ExitKind string

const (
	ExitNone        ExitKind = "none"
	ExitFailure     ExitKind = "failure"
	ExitIntentional ExitKind = "intentional"
)

type IncidentInput struct {
	CapturedAt      time.Time
	ProcessID       int
	ProcessUUID     string
	ThreadID        string
	LastLogAt       time.Time
	ExitAt          time.Time
	ExitKind        ExitKind
	ExitCode        *int
	OSFailureEvent  bool
	QueueDropDelta  int64
	SlowWriteDelta  int64
	WriterCount     int64
	DBBytes         int64
	WALBytes        int64
	FreelistPages   int64
	ProcessObserved bool
}

type Incident struct {
	Schema         string `json:"schema"`
	Verdict        string `json:"verdict"`
	Causality      string `json:"causality"`
	CapturedAt     string `json:"captured_at"`
	ProcessID      int    `json:"process_id,omitempty"`
	ProcessUUID    string `json:"process_uuid,omitempty"`
	ThreadID       string `json:"thread_id,omitempty"`
	LastLogAt      string `json:"last_log_at,omitempty"`
	ExitAt         string `json:"exit_at,omitempty"`
	ExitCode       *int   `json:"exit_code,omitempty"`
	OSFailureEvent bool   `json:"os_failure_event"`
	QueueDropDelta int64  `json:"queue_drop_delta"`
	SlowWriteDelta int64  `json:"slow_write_delta"`
	WriterCount    int64  `json:"writer_count"`
	DBBytes        int64  `json:"db_bytes"`
	WALBytes       int64  `json:"wal_bytes"`
	FreelistPages  int64  `json:"freelist_pages"`
}

func CaptureIncident(in IncidentInput) Incident {
	out := Incident{
		Schema: IncidentSchema, CapturedAt: stamp(in.CapturedAt), ProcessID: in.ProcessID,
		ProcessUUID: safeID(in.ProcessUUID), ThreadID: safeID(in.ThreadID), LastLogAt: stamp(in.LastLogAt),
		ExitAt: stamp(in.ExitAt), ExitCode: in.ExitCode, OSFailureEvent: in.OSFailureEvent,
		QueueDropDelta: in.QueueDropDelta, SlowWriteDelta: in.SlowWriteDelta, WriterCount: in.WriterCount,
		DBBytes: in.DBBytes, WALBytes: in.WALBytes, FreelistPages: in.FreelistPages,
	}
	switch {
	case in.ExitKind == ExitFailure && (!in.ExitAt.IsZero() || in.OSFailureEvent || in.ExitCode != nil):
		out.Verdict, out.Causality = "DIRECT_PROCESS_FAILURE", "established"
	case in.ExitKind == ExitIntentional:
		out.Verdict, out.Causality = "INTENTIONAL_EXIT", "established"
	case in.ProcessObserved:
		out.Verdict, out.Causality = "HEALTHY_PROCESS", "not_applicable"
	case in.QueueDropDelta > 0 || in.SlowWriteDelta > 0 || in.WALBytes > 0:
		out.Verdict, out.Causality = "CORRELATED_RUNTIME_PRESSURE", "not_established"
	default:
		out.Verdict, out.Causality = "MISSING_EVIDENCE", "not_established"
	}
	return out
}

func stamp(t time.Time) string {
	if t.IsZero() {
		return ""
	}
	return t.UTC().Format(time.RFC3339Nano)
}

func safeID(s string) string {
	s = strings.TrimSpace(s)
	if len(s) > 128 {
		return ""
	}
	for _, r := range s {
		if !(r == '-' || r == '_' || r == '.' || r >= '0' && r <= '9' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z') {
			return ""
		}
	}
	return s
}
