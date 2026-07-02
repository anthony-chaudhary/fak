// Package sweep is the pure decision core of the manifest-free crash discovery: given
// every on-disk COPY of a recently-touched session (the same transcript can live under
// several account dirs after re-homes), it resolves the SUPERSET copy, reads the terminal
// turn, and buckets the session by the action it actually needs. It is the Go port of
// tools/resume_sweep.py's classify/sort/dedup core.
//
// # The gap it closes (inherited from the Python)
//
// The manifest-bound watcher only classifies sessions a registry already lists; a crash
// wave the manifest never recorded is invisible to it. The sweep is the discovery half:
// it reasons over the transcripts themselves. This leaf owns the three load-bearing rules:
//
//   - SUPERSET by uuid-set + last-ts, NOT file mtime — a re-capped resume rewrites only
//     the banner and bumps mtime on a stale PREFIX copy, so mtime picks the wrong file.
//   - The failure mode is adjudicated off the ERROR record ONLY (sessionsignals.
//     TerminalFailure), never the assistant prose; what the prose alone would have said
//     is still computed and surfaced (ProseDiverged) so an averted false positive is
//     visible instead of silent.
//   - A usage-limit crash is split by the reset clock into LIMIT_RESET_PASSED (resumable
//     now) vs LIMIT_RESET_FUTURE (wait), anchored on the banner's own timestamp.
//
// Pure by construction: the I/O shell (cmd/fak resume sweep) walks the account dirs,
// parses each .jsonl into Records, lists the live `claude --resume` processes, and reads
// the resume ledger; this leaf only classifies. Same inputs, same rows — no clock (now is
// supplied), no filesystem, no process census.
package sweep

import (
	"bufio"
	"encoding/json"
	"io"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionsignals"
)

// Record is the closed set of facts about ONE transcript line the sweep needs: identity
// (UUID), order (Timestamp), channel (Role / IsError), and the text. The shell extracts
// these; the leaf never sees the raw JSON.
type Record struct {
	UUID      string
	Timestamp string
	Role      string
	Text      string
	// IsError marks the injected error channel: a record whose type is "error" or that
	// carries isApiErrorMessage. Classification keys off these records ONLY.
	IsError bool
}

// Copy is one on-disk copy of a session's transcript: where it lives (which account and
// project slug) and its parsed records.
type Copy struct {
	Path    string
	Account string
	Project string
	Records []Record
}

// The closed bucket vocabulary, ordered by action urgency (see Order).
const (
	// BucketLimitResetPassed: a usage cap whose reset window has elapsed — resumable now.
	BucketLimitResetPassed = "LIMIT_RESET_PASSED"
	// BucketLimitResetFuture: a usage cap still inside its window — wait for the reset.
	BucketLimitResetFuture = "LIMIT_RESET_FUTURE"
	// BucketAPIErr: a transient 529/overload/transport error — resumable now.
	BucketAPIErr = "API_ERR"
	// BucketAuth: a login/credit/access wall — needs a human /login, NOT a resume.
	BucketAuth = "AUTH"
	// BucketLive: a running claude process is already driving the sid — leave it alone.
	BucketLive = "LIVE"
	// BucketOther: no error record at all — not a crash this sweep acts on.
	BucketOther = "OTHER"
)

// Row is the classified verdict for one session — the same fields the Python emitted, so
// a consumer of the machine record sees an unchanged shape.
type Row struct {
	SID             string `json:"sid"`
	Bucket          string `json:"bucket"`
	SupersetAccount string `json:"superset_account"`
	Project         string `json:"project"`
	IsSuperset      bool   `json:"is_superset"`
	NRecords        int    `json:"n_records"`
	Copies          int    `json:"copies"`
	Reset           string `json:"reset"`
	// Evidence is the clipped, non-forgeable error text that drove the bucket ("" for
	// LIVE/OTHER), and ProseDiverged flags that the final assistant prose ALONE would
	// have landed in a different bucket — a prose-only false positive the error-channel
	// discipline averted, surfaced for observability.
	Evidence      string `json:"evidence"`
	ProseDiverged bool   `json:"prose_diverged"`
	CWD           string `json:"cwd"`
	SupersetPath  string `json:"superset_path"`
	SeatOK        *bool  `json:"seat_ok,omitempty"`
}

// Classify buckets one session from its newest copy's terminal turn and resolves the
// superset copy. live is the set of sids a running `claude --resume` currently drives;
// now drives the reset past/future verdict.
func Classify(sid string, copies []Copy, live map[string]bool, now time.Time) Row {
	best := 0
	for i := 1; i < len(copies); i++ {
		bTS, iTS := lastTS(copies[best].Records), lastTS(copies[i].Records)
		// Superset = latest last-ts, then most records (NOT file mtime). ISO-8601 UTC
		// timestamps compare correctly as strings, the same order the Python relied on.
		if iTS > bTS || (iTS == bTS && len(copies[i].Records) > len(copies[best].Records)) {
			best = i
		}
	}
	bestCopy := copies[best]
	bestUUIDs := uuidSet(bestCopy.Records)
	isSuperset := true
	for _, c := range copies {
		for u := range uuidSet(c.Records) {
			if !bestUUIDs[u] {
				isSuperset = false
			}
		}
	}

	var lastAssistant, lastErr *Record
	for i := range bestCopy.Records {
		r := &bestCopy.Records[i]
		if r.Role == "assistant" && strings.TrimSpace(r.Text) != "" {
			lastAssistant = r
		}
		if r.IsError {
			lastErr = r
		}
	}
	errText, proseText := "", ""
	if lastErr != nil {
		errText = lastErr.Text
	}
	if lastAssistant != nil {
		proseText = lastAssistant.Text
	}
	// Adjudicate the failure mode off the error record ONLY — never the assistant prose.
	// The prose-alone verdict is computed purely for the ProseDiverged observability bit.
	kind, detail := sessionsignals.TerminalFailure(errText)
	proseKind, _ := sessionsignals.TerminalFailure(proseText)
	proseDiverged := proseKind != "" && proseKind != kind

	bucket, reset := BucketOther, ""
	switch {
	case live[sid]:
		bucket = BucketLive
	case kind == sessionsignals.FailureAuth:
		bucket = BucketAuth
	case kind == sessionsignals.FailureLimit:
		reset = detail
		anchor := time.Time{}
		if ts := lastTS(bestCopy.Records); ts != "" {
			if t, err := time.Parse(time.RFC3339, ts); err == nil {
				anchor = t
			}
		}
		passed, ok := sessionsignals.ResetPassed(detail, now, anchor)
		if ok && passed {
			bucket = BucketLimitResetPassed
		} else {
			// Unparseable reset strings are conservatively not-yet-passed.
			bucket = BucketLimitResetFuture
		}
	case kind == sessionsignals.FailureAPIErr:
		bucket = BucketAPIErr
	}

	evidence := ""
	if bucket != BucketLive && bucket != BucketOther {
		evidence = clip(errText, 90)
	}
	return Row{
		SID: sid, Bucket: bucket,
		SupersetAccount: bestCopy.Account, Project: bestCopy.Project,
		IsSuperset: isSuperset, NRecords: len(bestCopy.Records), Copies: len(copies),
		Reset: reset, Evidence: evidence, ProseDiverged: proseDiverged,
		SupersetPath: bestCopy.Path,
	}
}

// Order is the action-urgency sort rank of a bucket: resumable-now first, then waiting,
// then human-needed, then untouchable. Unknown buckets sort last.
func Order(bucket string) int {
	switch bucket {
	case BucketLimitResetPassed:
		return 0
	case BucketAPIErr:
		return 1
	case BucketLimitResetFuture:
		return 2
	case BucketAuth:
		return 3
	case BucketLive:
		return 4
	}
	return 9
}

// Sort orders rows by (bucket urgency, superset account, most records first) — stable, so
// equal keys keep their discovery order.
func Sort(rows []Row) {
	sort.SliceStable(rows, func(i, j int) bool {
		if a, b := Order(rows[i].Bucket), Order(rows[j].Bucket); a != b {
			return a < b
		}
		if rows[i].SupersetAccount != rows[j].SupersetAccount {
			return rows[i].SupersetAccount < rows[j].SupersetAccount
		}
		return rows[i].NRecords > rows[j].NRecords
	})
}

// RecentlyResumed returns the sids the resume ledger shows we (re)launched within the
// window. During an active resume pass a session's OLD copies still terminate on their
// pre-resume error, so without this the sweep re-flags work already in flight — reading
// the ledger (the record of what was actually launched) is the honest dedup. Malformed
// lines are skipped, matching the Python's tolerance of a hand-edited ledger.
func RecentlyResumed(r io.Reader, windowMin float64, now time.Time) map[string]bool {
	out := map[string]bool{}
	if r == nil {
		return out
	}
	cutoff := now.Add(-time.Duration(windowMin * float64(time.Minute)))
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var rec struct {
			TS      string `json:"ts"`
			Session string `json:"session"`
		}
		if json.Unmarshal([]byte(line), &rec) != nil || rec.TS == "" || rec.Session == "" {
			continue
		}
		t, err := time.Parse(time.RFC3339, rec.TS)
		if err != nil {
			continue
		}
		if !t.Before(cutoff) {
			out[rec.Session] = true
		}
	}
	return out
}

// lastTS is the timestamp of the newest record that carries one — the copy-ordering key.
func lastTS(recs []Record) string {
	for i := len(recs) - 1; i >= 0; i-- {
		if recs[i].Timestamp != "" {
			return recs[i].Timestamp
		}
	}
	return ""
}

// uuidSet is the set of non-empty record UUIDs in a copy — the subset test's currency.
func uuidSet(recs []Record) map[string]bool {
	out := make(map[string]bool, len(recs))
	for _, r := range recs {
		if r.UUID != "" {
			out[r.UUID] = true
		}
	}
	return out
}

var nonAlnumRE = regexp.MustCompile(`[^A-Za-z0-9]`)

// Slugify maps a directory path to the Claude Code project-slug form (every non-
// alphanumeric rune becomes '-'). The slug is lossy — '-' collapses both separators and
// real hyphens — which is exactly why CwdForSlug matches candidates by re-slugifying
// rather than string-reversing.
func Slugify(path string) string { return nonAlnumRE.ReplaceAllString(path, "-") }

// CwdForSlug recovers the real cwd for a project slug from a caller-supplied list of
// candidate directories (the shell enumerates plausible roots; this leaf just matches).
// Falls back to fallback when nothing matches.
func CwdForSlug(project string, candidates []string, fallback string) string {
	for _, d := range candidates {
		if Slugify(d) == project {
			return d
		}
	}
	return fallback
}

// clip returns a one-line, length-bounded evidence snippet (whitespace collapsed, then
// truncated with a "…"), mirroring the Python _clip.
func clip(text string, width int) string {
	s := strings.Join(strings.Fields(text), " ")
	if len(s) <= width {
		return s
	}
	// Truncate on a rune boundary so a multibyte char at the cut never splits.
	rs := []rune(s)
	if len(rs) <= width-1 {
		return s
	}
	return string(rs[:width-1]) + "…"
}
