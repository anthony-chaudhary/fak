// Package journal is the durable, append-only, tamper-evident DECISION JOURNAL —
// the regulated-audit (AUD) surface of the trust floor. The kernel already
// fans out a lifecycle Event on every adjudication (abi.RegisterEmitter), but
// the three shipped emitters (harvest, vdso, ifc) are all IN-MEMORY: none is a
// durable record of "what did the kernel decide, when, over which bytes, and
// why". This package closes that gap.
//
// WHAT IT GUARANTEES.
//
//   - DURABLE: a persisting abi.Emitter writes one JSONL row per
//     EvDecide / EvDeny / EvResultDeny / EvQuarantine / EvVDSOHit (the vDSO-served hit included,
//     so a cache hit is audited exactly like an engine call). Rows are appended
//     and flushed per write, so a process crash loses nothing already returned to
//     the caller.
//   - TIME/SEQUENCE ANCHORED: the journal stamps its OWN monotonic Seq (1-based)
//     and a wall-clock timestamp on every row. The anchor lives in the row, not in
//     abi.Event — the frozen ABI is untouched.
//   - TAMPER-EVIDENT: every row carries the hash of the previous row's hash
//     chained with its own content (a hash chain / WORM ledger). Verify re-reads
//     the file and recomputes the chain; a single flipped byte breaks the link at
//     that row and fails the check. The journal does not PREVENT a privileged
//     edit, but it makes one DETECTABLE — the property an auditor underwrites.
//   - LIVE: in-process subscribers (and the gateway's /v1/fak/events stream) see
//     each row as it is committed; Recent serves a bounded tail without re-reading
//     the file.
//
// ENABLEMENT. The journal is off by default: writing to disk on every
// adjudication is a deployment choice, not something a benchmark or a unit test
// should pay for. Two ways to turn it on, both registering ONE persisting emitter
// against the frozen ABI:
//
//   - FAK_AUDIT_JOURNAL=/path/to/journal.jsonl — the package's init enables it at
//     boot (the FAK_IFC-style env toggle the rest of the kernel uses).
//   - journal.Enable(path) — the programmatic equivalent, for a front door (fak
//     guard) that decides AFTER init to default the audit trail on. Idempotent and
//     boot-env-respecting (see Enable).
//
// Unset and never Enabled => no emitter is registered and this package is inert.
package journal

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Row is one durable audit record. It is the on-disk JSONL schema AND the live
// stream element. Field order is the hash-chain pre-image order — do not reorder
// without bumping the chain (it would invalidate every existing journal).
type Row struct {
	Seq          uint64 `json:"seq"`          // monotonic 1-based order anchor
	TSUnixNano   int64  `json:"ts_unix_nano"` // wall-clock time anchor
	Kind         string `json:"kind"`         // DECIDE | DENY | RESULT_DENY | QUARANTINE | VDSO_HIT
	Tool         string `json:"tool,omitempty"`
	TraceID      string `json:"trace_id,omitempty"`
	Verdict      string `json:"verdict,omitempty"`
	Reason       string `json:"reason,omitempty"`
	By           string `json:"by,omitempty"`            // which adjudicator decided
	Taint        string `json:"taint,omitempty"`         // result provenance taint
	ArgsDigest   string `json:"args_digest,omitempty"`   // content hash of the call args
	ResultDigest string `json:"result_digest,omitempty"` // content hash of the result payload
	PrevHash     string `json:"prev_hash"`               // hash of the previous row ("" at genesis)
	Hash         string `json:"hash"`                    // chainHash(PrevHash, this row)

	// Correlation / bounded-disclosure fields — recorded and streamed, but NOT part
	// of the hash-chain pre-image (chainHash lists the chained fields explicitly, so
	// these are appended after Hash and EXISTING journals verify unchanged). The
	// tamper-evidence guarantee covers only the decision fields above; these are a
	// debugging convenience layered on top.
	CallSeq uint64 `json:"call_seq,omitempty"` // the kernel's per-call submission id (ToolCall.SeqNo): the join key tying a call's DECIDE to its later QUARANTINE
	Witness string `json:"witness,omitempty"`  // the bounded-disclosure claim the verdict surfaced (offending self-modify glob / tool.arg bound / require-witness claim)
}

// Journal is a hash-chained append-only ledger with an in-process live stream.
// The zero value is not usable; construct with Open (file-backed) or OpenMemory
// (in-memory, for tests of the stream/verify logic).
type Journal struct {
	mu        sync.Mutex
	bw        *bufio.Writer
	f         *os.File         // nil for an in-memory journal
	path      string           // file path ("" for an in-memory journal)
	seq       uint64           // last committed seq
	lastHash  string           // last committed row hash (the chain head)
	clock     func() time.Time // injectable for deterministic tests
	subs      map[int]chan Row // live subscribers (best-effort fan-out)
	nextSub   int
	recent    []Row // bounded tail for Recent (full history is on disk)
	maxRecent int
	dropped   uint64 // live-stream sends dropped (slow consumer)
	writeErr  uint64 // append failures (file-backed)
}

const defaultMaxRecent = 1024

// Open opens (creating if absent) a file-backed journal in append mode. If the
// file already holds rows, Open recovers the chain head (seq + last hash) so a
// restart CONTINUES the same tamper-evident chain instead of forking it. A
// corrupt tail is reported via Verify, not here — Open stays robust so a damaged
// log never bricks startup; the auditor runs Verify to learn the chain is broken.
func Open(path string) (*Journal, error) {
	// Recover the chain head from any existing content first.
	seq, last, err := recoverHead(path)
	if err != nil {
		return nil, err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, fmt.Errorf("journal: open %s: %w", path, err)
	}
	j := newJournal()
	j.f = f
	j.bw = bufio.NewWriter(f)
	j.path = path
	j.seq = seq
	j.lastHash = last
	return j, nil
}

// OpenMemory builds an in-memory journal (no file). Rows still chain, stream, and
// land in Recent; VerifyRows validates the chain over Recent without touching
// disk. Used by tests of the stream/verify logic.
func OpenMemory() *Journal { return newJournal() }

func newJournal() *Journal {
	return &Journal{
		clock:     time.Now,
		subs:      map[int]chan Row{},
		maxRecent: defaultMaxRecent,
	}
}

// Emit implements abi.Emitter: it is the kernel's per-event tap. It records every
// audit-relevant lifecycle event as a durable, chained row. It never blocks the
// kernel on a slow live consumer (best-effort fan-out) and never panics; a write
// failure is counted (WriteErrors) rather than propagated, since the fan-out
// contract is fire-and-forget.
func (j *Journal) Emit(ev abi.Event) {
	row, ok := rowFromEvent(ev)
	if !ok {
		return
	}
	j.append(row)
}

// append stamps the order/time anchor + chain hash and commits the row.
func (j *Journal) append(row Row) {
	j.mu.Lock()
	j.seq++
	row.Seq = j.seq
	row.TSUnixNano = j.clock().UnixNano()
	row.PrevHash = j.lastHash
	row.Hash = chainHash(row.PrevHash, row)
	j.lastHash = row.Hash

	if j.bw != nil {
		if err := writeRow(j.bw, row); err != nil {
			j.writeErr++
		}
	}
	// Bounded recent tail (full history is the file).
	j.recent = append(j.recent, row)
	if len(j.recent) > j.maxRecent {
		j.recent = j.recent[len(j.recent)-j.maxRecent:]
	}
	// Best-effort live fan-out: never block the kernel on a slow subscriber.
	for _, ch := range j.subs {
		select {
		case ch <- row:
		default:
			j.dropped++
		}
	}
	j.mu.Unlock()
}

// Subscribe returns a live channel of rows committed AFTER the call, plus a
// cancel that unsubscribes and closes the channel. Sends are best-effort: a
// subscriber that falls behind drops rows (counted in Dropped) — the FILE is the
// durable record, the stream is a convenience.
func (j *Journal) Subscribe() (<-chan Row, func()) {
	j.mu.Lock()
	id := j.nextSub
	j.nextSub++
	ch := make(chan Row, 256)
	j.subs[id] = ch
	j.mu.Unlock()
	var once sync.Once
	cancel := func() {
		once.Do(func() {
			j.mu.Lock()
			if c, ok := j.subs[id]; ok {
				delete(j.subs, id)
				close(c)
			}
			j.mu.Unlock()
		})
	}
	return ch, cancel
}

// Recent returns up to the last n committed rows (most recent last). n<=0 returns
// the whole bounded tail. It serves the gateway endpoint without re-reading disk.
func (j *Journal) Recent(n int) []Row {
	j.mu.Lock()
	defer j.mu.Unlock()
	if n <= 0 || n > len(j.recent) {
		n = len(j.recent)
	}
	out := make([]Row, n)
	copy(out, j.recent[len(j.recent)-n:])
	return out
}

// Stats reports the journal's live counters (head seq, dropped stream sends,
// write errors) for /healthz-style introspection.
func (j *Journal) Stats() (seq, dropped, writeErrors uint64) {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.seq, j.dropped, j.writeErr
}

// Path returns the file the journal appends to, or "" for an in-memory journal.
// It lets a caller that enabled the journal report WHERE the durable trail lives
// (the fak guard banner / exit summary) and what to run `fak audit verify` over.
func (j *Journal) Path() string {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.path
}

// ExportTo writes the journal as JSONL to w — the durable full history for a
// file-backed journal (re-read from disk after flushing buffered rows), or the
// bounded recent tail for an in-memory one. Returns the number of rows written.
// The output round-trips through Verify-style parsing, so an export of a sound
// journal is itself a sound journal.
func (j *Journal) ExportTo(w io.Writer) (int, error) {
	j.mu.Lock()
	if j.bw != nil {
		if err := j.bw.Flush(); err != nil {
			j.mu.Unlock()
			return 0, err
		}
	}
	path, mem := j.path, append([]Row(nil), j.recent...)
	j.mu.Unlock()

	if path == "" { // in-memory: export the recent tail
		n := 0
		for _, row := range mem {
			b, err := json.Marshal(row)
			if err != nil {
				return n, err
			}
			if _, err := w.Write(append(b, '\n')); err != nil {
				return n, err
			}
			n++
		}
		return n, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return 0, fmt.Errorf("journal: export %s: %w", path, err)
	}
	defer f.Close()
	n := 0
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		if len(sc.Bytes()) == 0 {
			continue
		}
		if _, err := w.Write(append(sc.Bytes(), '\n')); err != nil {
			return n, err
		}
		n++
	}
	return n, sc.Err()
}

// Flush pushes buffered bytes to the OS (durable across a process crash, not a
// power loss). Append already flushes per row; Flush is for explicit sync points.
func (j *Journal) Flush() error {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.bw != nil {
		return j.bw.Flush()
	}
	return nil
}

// Close flushes, fsyncs, and closes the file. Safe on an in-memory journal.
func (j *Journal) Close() error {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.bw != nil {
		if err := j.bw.Flush(); err != nil {
			return err
		}
	}
	if j.f != nil {
		_ = j.f.Sync()
		err := j.f.Close()
		j.f, j.bw = nil, nil
		return err
	}
	return nil
}

// writeRow appends one JSONL row and flushes it (per-row durability).
func writeRow(bw *bufio.Writer, row Row) error {
	b, err := json.Marshal(row)
	if err != nil {
		return err
	}
	if _, err := bw.Write(b); err != nil {
		return err
	}
	if err := bw.WriteByte('\n'); err != nil {
		return err
	}
	return bw.Flush()
}

// chainHash is the tamper-evident link: sha256 over the previous row's hash
// chained with this row's content fields (Seq..ResultDigest, in declaration
// order). PrevHash and Hash are excluded from the pre-image (PrevHash is the
// chained-in prefix; Hash is the output). A unit separator (0x1f) delimits fields
// so no concatenation collision is possible.
func chainHash(prev string, r Row) string {
	h := sha256.New()
	io.WriteString(h, prev)
	fmt.Fprintf(h, "\x1f%d\x1f%d\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s",
		r.Seq, r.TSUnixNano, r.Kind, r.Tool, r.TraceID, r.Verdict,
		r.Reason, r.By, r.Taint, r.ArgsDigest, r.ResultDigest)
	return hex.EncodeToString(h.Sum(nil))
}

// rowFromEvent projects a lifecycle Event into an audit row, returning false for
// a non-audit kind (EvSubmit/EvDispatch/EvComplete/EvRungLabel are operational,
// not decisions). Digests come from the frozen Ref.Digest (the content hash the
// vDSO + provenance already maintain) — the emitter never resolves blob bytes,
// so it stays cheap and leaks no payload into the log.
func rowFromEvent(ev abi.Event) (Row, bool) {
	var kind string
	switch ev.Kind {
	case abi.EvDecide:
		// A DENY decision is ALSO emitted as a dedicated EvDeny — the kernel pairs
		// them on every deny path (Decide: emit EvDecide then EvDeny; Submit: the
		// EvDecide at adjudication then an EvDeny in the deny/require-witness/escalate
		// branches). Recording the EvDecide row too would write the SAME deny twice
		// into the durable hash-chained journal, double-counting it in every consumer
		// that folds rows back — the `fak guard` exit summary's "decision(s) appended"
		// count and the guard-RSI verdict-quality metric (which keys on the `verdict`
		// field, so a DECIDE(DENY)+DENY pair counts as two denials). Record the
		// canonical decision ONCE: keep the DECIDE row for the non-deny outcomes
		// (ALLOW/TRANSFORM/REQUIRE_WITNESS) and let the paired EvDeny carry the deny.
		// A REQUIRE_WITNESS interim verdict is NOT a deny, so its DECIDE row is kept
		// and a later EvDeny records the resolved deny as a distinct, intended fact.
		if ev.Verdict != nil && ev.Verdict.Kind == abi.VerdictDeny {
			return Row{}, false
		}
		kind = "DECIDE"
	case abi.EvDeny:
		kind = "DENY"
	case abi.EvResultDeny:
		kind = "RESULT_DENY"
	case abi.EvQuarantine:
		kind = "QUARANTINE"
	case abi.EvVDSOHit:
		kind = "VDSO_HIT"
	default:
		return Row{}, false
	}
	row := Row{Kind: kind}
	if c := ev.Call; c != nil {
		row.Tool = c.Tool
		row.TraceID = c.TraceID
		row.ArgsDigest = refDigest(c.Args)
		row.CallSeq = c.SeqNo // join key: same call's DECIDE and QUARANTINE share it
	}
	if v := ev.Verdict; v != nil {
		row.Verdict = verdictName(v.Kind)
		row.Reason = abi.ReasonName(v.Reason)
		row.By = v.By
		row.Witness = witnessOf(v)
	}
	if r := ev.Result; r != nil {
		row.ResultDigest = refDigest(r.Payload)
		row.Taint = taintName(r.Payload.Taint)
	}
	return row, true
}

// refDigest is the audit identity of a Ref's bytes WITHOUT resolving them: the
// content hash if the backend stamped one, else a hash of the inline bytes, else
// empty. Never materializes a blob (no resolver dependency on the hot path).
// witnessOf extracts the bounded-disclosure claim a verdict surfaced — the
// offending self-modify glob / arg bound (WitnessPayload.Claim) or the
// require-witness gate's claim (Meta["claim"]). "" when the verdict disclosed
// nothing. This is the one forensic field the live wire carried but the durable
// row used to drop, leaving an audit unable to say WHICH glob/arg tripped a deny.
func witnessOf(v *abi.Verdict) string {
	if v == nil {
		return ""
	}
	if wp, ok := v.Payload.(abi.WitnessPayload); ok && wp.Claim != "" {
		return wp.Claim
	}
	return v.Meta["claim"]
}

func refDigest(r abi.Ref) string {
	if r.Digest != "" {
		return r.Digest
	}
	if r.Kind == abi.RefInline && len(r.Inline) > 0 {
		sum := sha256.Sum256(r.Inline)
		return "sha256:" + hex.EncodeToString(sum[:])
	}
	return ""
}

func verdictName(k abi.VerdictKind) string {
	switch k {
	case abi.VerdictAllow:
		return "ALLOW"
	case abi.VerdictDeny:
		return "DENY"
	case abi.VerdictTransform:
		return "TRANSFORM"
	case abi.VerdictQuarantine:
		return "QUARANTINE"
	case abi.VerdictRequireWitness:
		return "WITNESS"
	case abi.VerdictDefer:
		return "DEFER"
	case abi.VerdictIndeterminate:
		return "INDETERMINATE"
	}
	return fmt.Sprintf("K%d", k)
}

func taintName(t abi.TaintLabel) string {
	switch t {
	case abi.TaintTrusted:
		return "trusted"
	case abi.TaintTainted:
		return "tainted"
	case abi.TaintQuarantined:
		return "quarantined"
	}
	return ""
}

// recoverHead scans an existing journal to recover the chain head (last seq + last
// hash) so an append continues the same chain. A missing file is the genesis case
// (seq 0, empty hash). It does NOT validate the chain (that is Verify's job) so a
// damaged log never blocks startup.
func recoverHead(path string) (seq uint64, lastHash string, err error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, "", nil
		}
		return 0, "", fmt.Errorf("journal: stat %s: %w", path, err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var r Row
		if err := json.Unmarshal(line, &r); err != nil {
			// A torn final line (crash mid-write) is tolerated: stop at the last
			// well-formed row. Verify will catch genuine corruption.
			break
		}
		seq = r.Seq
		lastHash = r.Hash
	}
	if err := sc.Err(); err != nil {
		return 0, "", fmt.Errorf("journal: scan %s: %w", path, err)
	}
	return seq, lastHash, nil
}

// ReadRows reads all committed rows from a journal file, in order — the READ side
// of the durable log for a CONSUMER (a live guard-tail pane, an exporter) that wants
// the rows as data, not an integrity check (use Verify for that). It is deliberately
// robust for a live reader: a MISSING file is the empty journal (nil, nil) — tailing
// a not-yet-written journal is a valid "no rows yet" state, not an error — and a torn
// final line (a crash mid-append) is tolerated by stopping at the last well-formed
// row (mirroring recoverHead), so a reader never errors on a half-written tail.
// Genuine I/O errors (permission, a read fault) are returned. Verify, not ReadRows,
// is the surface that detects in-the-middle tampering.
func ReadRows(path string) ([]Row, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("journal: read %s: %w", path, err)
	}
	defer f.Close()
	var out []Row
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var r Row
		if err := json.Unmarshal(line, &r); err != nil {
			break // torn final line: stop at the last well-formed row (Verify catches real corruption)
		}
		out = append(out, r)
	}
	if err := sc.Err(); err != nil {
		return out, fmt.Errorf("journal: scan %s: %w", path, err)
	}
	return out, nil
}

// Verify re-reads a journal file and validates the hash chain end to end. It
// returns the number of rows checked and a non-nil error naming the FIRST broken
// link (a recomputed hash mismatch, a prev-hash discontinuity, or a sequence
// gap). A journal that passes Verify has not been edited since it was written.
func Verify(path string) (n int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, fmt.Errorf("journal: open %s: %w", path, err)
	}
	defer f.Close()
	return verifyReader(f)
}

func verifyReader(r io.Reader) (int, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	var (
		prev    string
		wantSeq uint64
		n       int
	)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var row Row
		if err := json.Unmarshal(line, &row); err != nil {
			return n, fmt.Errorf("journal: row %d: malformed JSON: %w", n+1, err)
		}
		wantSeq++
		next, err := verifyStep(prev, wantSeq, row)
		if err != nil {
			return n, err
		}
		prev = next
		n++
	}
	if err := sc.Err(); err != nil {
		return n, fmt.Errorf("journal: scan: %w", err)
	}
	return n, nil
}

// VerifyRows validates the hash chain over an in-memory slice (e.g. Recent() of
// an in-memory journal). Same checks as Verify, returning the first broken link.
func VerifyRows(rows []Row) (int, error) {
	var (
		prev    string
		wantSeq uint64
	)
	for i, row := range rows {
		wantSeq++
		next, err := verifyStep(prev, wantSeq, row)
		if err != nil {
			return i, err
		}
		prev = next
	}
	return len(rows), nil
}

// verifyStep checks one row against the running chain head + expected sequence and
// returns the new chain head. It is the single source of truth for "is this row
// authentic and in order", shared by file and in-memory verification.
func verifyStep(prev string, wantSeq uint64, row Row) (string, error) {
	if row.Seq != wantSeq {
		return "", fmt.Errorf("journal: sequence gap: seq=%d want %d", row.Seq, wantSeq)
	}
	if row.PrevHash != prev {
		return "", fmt.Errorf("journal: broken chain at seq %d: prev_hash=%q want %q", row.Seq, row.PrevHash, prev)
	}
	if got := chainHash(row.PrevHash, row); got != row.Hash {
		return "", fmt.Errorf("journal: tampered row at seq %d: hash=%q recomputed %q", row.Seq, row.Hash, got)
	}
	return row.Hash, nil
}

// ---------------------------------------------------------------------------
// Registered instance — opt-in via FAK_AUDIT_JOURNAL.
// ---------------------------------------------------------------------------

var (
	activeMu sync.Mutex
	active   *Journal
)

// Active returns the registered durable journal, or nil if none was enabled
// (FAK_AUDIT_JOURNAL unset at boot and no programmatic Enable). The gateway uses
// this to serve /v1/fak/events or 404.
func Active() *Journal {
	activeMu.Lock()
	defer activeMu.Unlock()
	return active
}

// Enable turns the durable decision journal ON at path AFTER init has run — the
// programmatic equivalent of FAK_AUDIT_JOURNAL, for a front door (fak guard) that
// decides to default the audit trail on. It creates the parent directory, opens
// (creating, or CONTINUING an existing chain) a file-backed journal, registers it
// as ONE persisting emitter against the frozen ABI, and returns it.
//
// It is IDEMPOTENT and boot-env-respecting: if a journal is already active
// (FAK_AUDIT_JOURNAL won at boot, or a prior Enable ran) Enable is a no-op that
// returns the existing journal — the emitter is never double-registered (the ABI
// has no unregister) and the first/boot enablement always wins. A genuine open
// failure is returned (never silently swallowed) so the caller can decide whether
// to proceed without the trail.
func Enable(path string) (*Journal, error) {
	activeMu.Lock()
	defer activeMu.Unlock()
	if active != nil {
		return active, nil
	}
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return nil, fmt.Errorf("journal: create dir %s: %w", dir, err)
		}
	}
	j, err := Open(path)
	if err != nil {
		return nil, err
	}
	active = j
	abi.RegisterEmitter(j)
	return j, nil
}

func init() {
	path := os.Getenv("FAK_AUDIT_JOURNAL")
	if path == "" {
		return // off unless a front door (fak guard) programmatically Enables it
	}
	if _, err := Enable(path); err != nil {
		// Fail loud but do not brick the kernel: a missing audit sidecar must not
		// stop adjudication (the in-memory counters still hold). An auditor who
		// requires fail-closed wires that as a separate posture (issue #12).
		fmt.Fprintf(os.Stderr, "fak: audit journal disabled — %v\n", err)
		return
	}
	fmt.Fprintf(os.Stderr, "fak: audit journal -> %s (durable, hash-chained)\n", path)
}
