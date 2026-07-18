package fleet

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// ReportSchema tags a per-box report — the PUBLIC/PRIVATE SEAM. The live control
// plane (the private Slack control-bridge to the lab boxes; see
// docs/gpu-server-private-boundary.md and docs/private-comms-channel.md) emits one of these
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

// GPUStats is a box's COMPUTE-utilization reading — how busy its silicon is, NOT how
// full its memory is. It is deliberately orthogonal to readiness (State) and to
// cache-tier capacity (internal/engine/capacity_*, which measures byte fullness of
// HBM/DRAM/disk): an 8-GPU box can be memory-full on one GPU and idle on the other
// seven, and only this catches the wasted seven. It carries COUNTS and one percent —
// no host, no device serial, no vendor string — so it stays on the public/private
// seam like the rest of Report.
//
// A producer that cannot probe the GPU leaves Report.GPU nil, which the fold reads as
// unknown-utilization (no signal), NEVER as "0% = idle" — a false idle alarm on a box
// that simply can't measure would be worse than silence. UtilPct is the aggregate
// busy percent (0-100) across the box's GPUs when the producer has it; Busy/Total are
// the count of GPUs doing work vs present, the signal the waste crit keys on.
type GPUStats struct {
	Total   int `json:"total"`              // GPUs present on the box
	Busy    int `json:"busy"`               // GPUs actively executing (util above the producer's busy threshold)
	UtilPct int `json:"util_pct,omitempty"` // aggregate 0-100 busy percent across the box, when known
}

// InferenceStatus is the public, transport-neutral answer to "can this box take
// inference work right now?". It is separate from State: a box can be live for
// shell/dev work while its serving stack is warming or blocked.
type InferenceStatus string

const (
	InferenceReady    InferenceStatus = "ready"    // serving can take inference traffic now
	InferenceDegraded InferenceStatus = "degraded" // usable, but slower/partial/caveated
	InferenceWarming  InferenceStatus = "warming"  // loading or booting; not useful yet
	InferenceBlocked  InferenceStatus = "blocked"  // known blocker; operator action needed
	InferenceUnknown  InferenceStatus = "unknown"  // producer could not prove serving usefulness
)

func (s InferenceStatus) Known() bool {
	switch s {
	case InferenceReady, InferenceDegraded, InferenceWarming, InferenceBlocked, InferenceUnknown:
		return true
	default:
		return false
	}
}

func (s InferenceStatus) Useful() bool {
	return s == InferenceReady || s == InferenceDegraded
}

// InferenceStats is a scrubbed serving-usefulness report for one box. It names only
// generic serving facts: status, engine/model labels, throughput/latency numbers,
// and an optional generic reason. It must never carry a host, URL, channel id,
// token, raw transcript, or private filesystem path.
type InferenceStats struct {
	Status         InferenceStatus `json:"status"`                     // ready|degraded|warming|blocked|unknown
	Engine         string          `json:"engine,omitempty"`           // generic engine label, e.g. fak|vllm|sglang|llama
	Model          string          `json:"model,omitempty"`            // generic model label, never a private path
	OutputTPS      float64         `json:"output_tps,omitempty"`       // observed output tok/s when known
	ProbeLatencyMS float64         `json:"probe_latency_ms,omitempty"` // scrubbed end-to-end probe latency when known
	Reason         string          `json:"reason,omitempty"`           // short scrubbed reason class
}

// Report is one box's current operational state — the seam schema. AgeSec is how
// long ago the box last reported (the transport stamps it); a large age means the
// box has gone quiet even if its last word was "live".
//
// ID and Err are tagged json:"-": identity is the roster's authority (never the
// wire), and Err is reader-owned (set when a box can't be reached or parsed), so a
// report file can never inject either field and flip the fold.
//
// GPU is an OPTIONAL pointer: a box that cannot probe its GPUs omits it, and the fold
// treats a nil GPU as unknown-utilization (no waste signal), never as 0%-idle.
type Report struct {
	Schema    string          `json:"schema,omitempty"`
	ID        string          `json:"-"`
	State     State           `json:"state"`
	Version   string          `json:"version,omitempty"`
	AgeSec    float64         `json:"age_sec,omitempty"`
	Note      string          `json:"note,omitempty"`
	GPU       *GPUStats       `json:"gpu,omitempty"`
	Inference *InferenceStats `json:"inference,omitempty"`
	Err       string          `json:"-"`
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

// OrphanReports lists report files in dir whose key resolves to NO roster box — a
// producer wrote <key>.json under a name no box.Ref() matches. ReadReports only ever
// opens roster-keyed paths, so such a file is SILENTLY invisible to the fold: the box
// it was meant for reads `unknown` and the file itself is never mentioned. That gap is
// the difference between an operator seeing "0/N reachable, all down — recover the
// bridge" and seeing "the producer is writing under non-roster keys (a real host label
// vs the box's generic roster id)". Surfacing them turns a misleading outage into an
// actionable key-mismatch.
//
// Returns the sorted keys (basename without the .json suffix). A missing or unreadable
// reports dir yields no orphans and no error — that is the honest "no live reports yet"
// state the status command already degrades for. Only *.json files are considered, so
// a sidecar (targets, readiness) or subdir is not miscounted.
func OrphanReports(dir string, ro Roster) []string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	known := make(map[string]struct{}, len(ro.Boxes))
	for _, b := range ro.Boxes {
		known[b.Ref()] = struct{}{}
	}
	var orphans []string
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		key := strings.TrimSuffix(e.Name(), ".json")
		if _, ok := known[key]; !ok {
			orphans = append(orphans, key)
		}
	}
	sort.Strings(orphans)
	return orphans
}

// AdoptOrphanReports reconciles the report-key/roster join for the READINESS
// derivation: it reads each orphan report file (a key no box.Ref() matches — a
// producer keying by its own label instead of the box's generic roster id) and
// ADOPTS every file that parses as a valid, reachable current-schema report,
// returning those as extra reports keyed by their file stem, plus the keys that
// remain genuinely unreconcilable (unreadable, wrong schema, unknown state).
//
// This is what lets a live lab whose bridge writes under non-roster keys reach a
// determinate readiness verdict (#5065) WITHOUT weakening the fail-closed gate:
// an adopted report still passes every guard a roster-keyed one does (schema,
// state vocabulary, inference normalization, the mtime freshness floor), so a
// stale or junk file can never buy a false READY — it either ages out downstream
// or stays in the returned orphans list. After adoption, "reports-under-non-
// roster-keys" means exactly what it says: files under foreign keys that cannot
// be read as reports at all.
func AdoptOrphanReports(dir string, ro Roster) (adopted []Report, orphans []string) {
	for _, key := range OrphanReports(dir, ro) {
		if !safeReportKey(key) {
			orphans = append(orphans, key)
			continue
		}
		r := readReportFile(dir, key, key)
		if r.Reachable() {
			adopted = append(adopted, r)
		} else {
			orphans = append(orphans, key)
		}
	}
	return adopted, orphans
}

func readOneReport(dir string, b Box) Report {
	key := b.Ref()
	// Endpoint is opaque to other transports, but for THIS (file) transport it names
	// a file inside the reports dir, so it must be a single safe segment. A typo'd or
	// escaping key reads as an error — never an out-of-tree file (path traversal).
	if !safeReportKey(key) {
		return Report{ID: b.ID, State: StateUnknown, Err: fmt.Sprintf("endpoint %q is not a file-safe report key", key)}
	}
	return readReportFile(dir, key, b.ID)
}

// readReportFile reads and validates <dir>/<key>.json as a report owned by id.
// key must already be file-safe (the caller's guard). Shared by the roster-keyed
// read (id = the box's roster id) and orphan adoption (id = the file stem), so an
// adopted report passes exactly the guards a roster-keyed one does.
func readReportFile(dir, key, id string) Report {
	path := filepath.Join(dir, key+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		return Report{ID: id, State: StateUnknown, Err: fmt.Sprintf("no report (%v)", rootErr(err))}
	}
	var r Report
	if err := json.Unmarshal(data, &r); err != nil {
		return Report{ID: id, State: StateUnknown, Err: fmt.Sprintf("bad report json: %v", err)}
	}
	r.ID = id // the roster is the identity authority; the file supplies only state.
	if r.Schema != "" && r.Schema != ReportSchema {
		// Mirror the roster's fail-loud schema guard: a future incompatible
		// fak.fleet.report/v2 must not be silently folded as v1.
		return Report{ID: id, State: StateUnknown, Err: fmt.Sprintf("unsupported report schema %q (want %s)", r.Schema, ReportSchema)}
	}
	if !r.State.Known() {
		r.Err = fmt.Sprintf("unknown state %q", r.State)
		r.State = StateUnknown
	}
	if err := normalizeInference(r.Inference); err != nil {
		return Report{ID: id, State: StateUnknown, Err: err.Error()}
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
	if err := normalizeInference(r.Inference); err != nil {
		return err
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

func normalizeInference(inf *InferenceStats) error {
	if inf == nil {
		return nil
	}
	if inf.Status == "" {
		inf.Status = InferenceUnknown
	}
	if !inf.Status.Known() {
		return fmt.Errorf("unknown inference status %q", inf.Status)
	}
	if inf.OutputTPS < 0 {
		return fmt.Errorf("inference output_tps must be non-negative")
	}
	if inf.ProbeLatencyMS < 0 {
		return fmt.Errorf("inference probe_latency_ms must be non-negative")
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
