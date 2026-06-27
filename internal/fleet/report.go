package fleet

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ReportSchema tags a per-box report — the PUBLIC/PRIVATE SEAM. The live control
// plane (the private Slack control-bridge to the lab boxes; see
// docs/dgx-slack-boundary.md and docs/private-comms-channel.md) emits one of these
// JSON files per box from live state; this public package reads, folds, renders, and
// scores them. The boundary is a DATA contract, not a code import: the private side
// never links this package and this package never knows the lab transport. A report
// carries only GENERIC operational state — a state word, a version, an age, a free
// note — never a host, a token, or a transcript.
//
// CONTRACT FOR THE PRODUCER (the private bridge, or `fak lab report`):
//   - Re-stamp age_sec on EVERY write. The reader floors age at the report file's
//     own mtime (so a frozen file from a dead bridge ages out and trips the stale
//     warn), but a non-updated age_sec is otherwise the bridge's only freshness word.
//   - Keep `note` pre-scrubbed: it is the one free-text field rendered verbatim, so a
//     stray lab hostname/channel/operator path there would leak into the public view.
const ReportSchema = "fak.fleet.report/v1"

// State is a box's coarse operational state. It is intentionally small and
// transport-neutral; a transport maps its own richer status onto these words.
type State string

const (
	StateLive     State = "live"     // serving / actively working
	StateIdle     State = "idle"     // up and reachable, no live work
	StateDraining State = "draining" // finishing in-flight work, not taking new
	StateDown     State = "down"     // reached, but reporting itself not serving
	StateUnknown  State = "unknown"  // no fresh report — the default when a box is silent
)

// Healthy reports whether a state counts toward fleet health (up and usable).
func (s State) Healthy() bool {
	return s == StateLive || s == StateIdle || s == StateDraining
}

// Known reports whether a state is one of the defined words. An unrecognized state
// from a newer transport is treated as unknown — never silently trusted as healthy.
func (s State) Known() bool {
	switch s {
	case StateLive, StateIdle, StateDraining, StateDown, StateUnknown:
		return true
	}
	return false
}

// Report is one box's current operational state — the seam schema. AgeSec is how
// long ago the box last reported (the transport stamps it); a large age means the
// box has gone quiet even if its last word was "live".
//
// ID and Err are tagged json:"-": identity is the roster's authority (never the
// wire), and Err is reader-owned (set when a box can't be reached or parsed), so a
// report file can never inject either field and flip the fold.
type Report struct {
	Schema  string  `json:"schema,omitempty"`
	ID      string  `json:"-"`
	State   State   `json:"state"`
	Version string  `json:"version,omitempty"`
	AgeSec  float64 `json:"age_sec,omitempty"`
	Note    string  `json:"note,omitempty"`
	Err     string  `json:"-"`
}

// Reachable reports whether a trustworthy report was obtained: no read error and a
// known, non-unknown state. A "down" report IS reachable — knowing a box is down is
// a real, useful observation; only silence (unknown) or an error is unreachable.
func (r Report) Reachable() bool {
	return r.Err == "" && r.State != StateUnknown && r.State.Known()
}

// ReadReports resolves one report per box, in roster order, using the FILE TRANSPORT:
// it reads <dir>/<box.Ref()>.json for each box. A missing or unreadable file is NOT
// fatal — that box gets an unknown report with Err set — because an operator view
// must never crash on one silent box. This is the public, offline/CI transport; the
// live Slack bridge is the private one that PRODUCES these files.
func ReadReports(dir string, ro Roster) []Report {
	out := make([]Report, len(ro.Boxes))
	for i, b := range ro.Boxes {
		out[i] = readOneReport(dir, b)
	}
	return out
}

func readOneReport(dir string, b Box) Report {
	key := b.Ref()
	// Endpoint is opaque to other transports, but for THIS (file) transport it names
	// a file inside the reports dir, so it must be a single safe segment. A typo'd or
	// escaping key reads as an error — never an out-of-tree file (path traversal).
	if !safeReportKey(key) {
		return Report{ID: b.ID, State: StateUnknown, Err: fmt.Sprintf("endpoint %q is not a file-safe report key", key)}
	}
	path := filepath.Join(dir, key+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		return Report{ID: b.ID, State: StateUnknown, Err: fmt.Sprintf("no report (%v)", rootErr(err))}
	}
	var r Report
	if err := json.Unmarshal(data, &r); err != nil {
		return Report{ID: b.ID, State: StateUnknown, Err: fmt.Sprintf("bad report json: %v", err)}
	}
	r.ID = b.ID // the roster is the identity authority; the file supplies only state.
	if r.Schema != "" && r.Schema != ReportSchema {
		// Mirror the roster's fail-loud schema guard: a future incompatible
		// fak.fleet.report/v2 must not be silently folded as v1.
		return Report{ID: b.ID, State: StateUnknown, Err: fmt.Sprintf("unsupported report schema %q (want %s)", r.Schema, ReportSchema)}
	}
	if !r.State.Known() {
		r.Err = fmt.Sprintf("unknown state %q", r.State)
		r.State = StateUnknown
	}
	// Freshness backstop: age_sec only ages a box if the bridge keeps re-stamping it,
	// so a dead bridge would leave a frozen "live, age 5s" file reading green forever.
	// Floor the age at the file's own mtime age so a stale file trips the stale warn.
	// (Reliable in the direct-write topology; an interposed rsync/scp that rewrites
	// mtime on sync would mask it — hence the re-stamp-every-write producer contract.)
	if fi, statErr := os.Stat(path); statErr == nil {
		if fileAge := time.Since(fi.ModTime()).Seconds(); fileAge > r.AgeSec {
			r.AgeSec = fileAge
		}
	}
	return r
}

// WriteReport writes one box's report into the reports dir using the same file
// transport ReadReports reads — <dir>/<id>.json. It is the PUBLIC producer half: the
// private Slack bridge is one producer, and `fak lab report` (a box self-reporting)
// is another, so the report-writing rule lives here next to the reader it must agree
// with. The id must be a file-safe report key (the same guard the reader applies), so
// a report can never escape the dir. age_sec is re-stamped to 0 on every write per the
// producer contract — this write IS the freshness event. Schema is forced to the
// current ReportSchema so a self-report is never mistaken for a future major.
func WriteReport(dir, id string, r Report) error {
	if !safeReportKey(id) {
		return fmt.Errorf("id %q is not a file-safe report key", id)
	}
	if !r.State.Known() {
		return fmt.Errorf("state %q is not one of live|idle|draining|down|unknown", r.State)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create reports dir: %w", err)
	}
	r.Schema = ReportSchema
	r.AgeSec = 0         // this write is the freshness event; the reader floors age at mtime anyway.
	r.ID, r.Err = "", "" // ID/Err are reader-owned and json:"-"; never serialize them.
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return fmt.Errorf("encode report: %w", err)
	}
	path := filepath.Join(dir, id+".json")
	if err := os.WriteFile(path, append(data, '\n'), 0o644); err != nil {
		return fmt.Errorf("write report: %w", err)
	}
	return nil
}

// safeReportKey reports whether a transport ref is safe to use as a FILE name in the
// file transport: a non-empty single path segment, no separators, no parent escape,
// no surrounding whitespace. Scoped to the file transport on purpose — Endpoint stays
// opaque for the private bridge (which may map it to a channel/session holding
// file-unsafe chars); only the file transport needs the key to be a clean filename.
func safeReportKey(ref string) bool {
	if ref == "" || ref != strings.TrimSpace(ref) {
		return false
	}
	if strings.ContainsAny(ref, "/\\") || strings.Contains(ref, "..") {
		return false
	}
	return filepath.Base(ref) == ref
}

// rootErr trims an *os.PathError down to its underlying cause for a compact message
// ("file does not exist" rather than the full path the operator already passed).
func rootErr(err error) error {
	var pe *os.PathError
	if errors.As(err, &pe) {
		return pe.Err
	}
	return err
}
