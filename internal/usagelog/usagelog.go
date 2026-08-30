// Package usagelog is the durable, append-only, tamper-evident CLI-INVOCATION
// journal — the record of how `fak` ITSELF is used, the gap epic #1601 (child A,
// #1608) closes.
//
// THE GAP IT CLOSES. fak is an observability product: it gives operators a
// durable, tamper-evident record of how the agent it WRAPS behaves (the decision
// journal in internal/journal, the loop ledger, cache-value.jsonl). But it kept
// almost no durable record of how `fak` itself is invoked — main() switches on
// os.Args[1] and dispatches straight to a verb handler, recording nothing about
// WHICH verb ran, WHEN, with WHAT exit code, on WHAT host/version. "How is fak
// actually used, and which verbs error?" was unanswerable from any artifact. This
// package writes one durable row per top-level invocation so it becomes answerable.
//
// WHAT IT GUARANTEES. It mirrors the discipline of internal/journal (the decision
// journal) — same hash-chain, same per-row flush, same Verify contract — applied
// to a different, usage-shaped Row. The two are deliberately decoupled (this
// package imports nothing internal beyond the internal/flock lock primitive) so a
// usage trail can be reasoned about and verified on its own:
//
//   - DURABLE: one JSONL row is appended and flushed per Append, written at process
//     exit so the exit code is known. A crash loses only the in-flight invocation.
//   - SEQUENCE/TIME ANCHORED: the logger stamps its own monotonic Seq (1-based) and
//     a wall-clock timestamp on every row.
//   - ONE CHAIN ACROSS PROCESSES: concurrent fak invocations extend the SAME chain.
//     Append serializes head-recovery + write under a cross-process advisory lock
//     (a <path>.lock sidecar over the internal/flock primitive, the same critical
//     section loopmgr's ledger append uses) and re-reads the chain head from disk
//     inside that critical section — so two racing processes can never both stamp
//     the same seq off a stale head (the #2608 `sequence gap` CHAIN_BROKEN
//     artifact `fak audit usage` surfaced).
//   - TAMPER-EVIDENT: every row carries the hash of the previous row's hash chained
//     with its own content fields (a hash chain / WORM ledger). Verify re-reads the
//     file and recomputes the chain; a single flipped byte breaks the link at that
//     row and fails the check. A privileged edit is not PREVENTED but is made
//     DETECTABLE — the property an auditor underwrites. `fak audit verify` over a
//     usage.jsonl exercises the same chain idea.
//
// HONESTY FENCE. A usage row is an OBSERVED self-report of the `fak` PROCESS (verb,
// exit code, timing) — never a WITNESS of any downstream effect. Args are REDACTED
// by default: a row stores argc plus a salted args_digest, never raw argv (which
// can carry paths, `-m` messages, or tokens). Raw argv lands on disk only behind an
// explicit --full-args opt-in (Row.Args), and even then it is layered ON TOP of the
// chain (excluded from the hash pre-image), exactly like internal/journal's
// correlation fields — so existing journals verify unchanged whether or not argv is
// present.
//
// ENABLEMENT. On by default, like the audit journal; FAK_USAGE_LOG=off (see
// Enabled) turns it off so a sandbox, a benchmark, or a privacy-conscious operator
// can opt out with no disk writes at all.
package usagelog

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

// SchemaV1 is the on-disk schema tag stamped into every row's `schema` field, so a
// reader can tell a usage journal apart from any other JSONL ledger and refuse to
// fold a row whose schema it does not understand.
const SchemaV1 = "fak-usage-log/1"

// DiagnosticSchemaV1 identifies the optional structured diagnostic stream. It is
// deliberately distinct from SchemaV1: diagnostics are operational hints, not
// authoritative usage rows, receipts, or hash-chained evidence.
const DiagnosticSchemaV1 = "fak-kernel-diagnostic/1"

// DiagnosticLevel is the closed severity vocabulary for DiagnosticSink.
type DiagnosticLevel uint8

const (
	DiagnosticDebug DiagnosticLevel = iota + 1
	DiagnosticInfo
	DiagnosticWarn
	DiagnosticError
)

// LazyValue defers producing a diagnostic scalar until Emit has established that
// the level is enabled and the key is not configured as sensitive.
type LazyValue func() any

// DiagnosticSink emits optional, non-authoritative JSONL diagnostics. It never
// appends to or changes the usage journal; callers that need durable facts must
// continue to use Logger and the receipt/journal surfaces that own those facts.
// A nil writer disables the sink. DiagnosticSink is safe for concurrent use.
type DiagnosticSink struct {
	mu        sync.Mutex
	w         io.Writer
	minLevel  DiagnosticLevel
	sensitive map[string]struct{}
}

type diagnosticRecord struct {
	Schema string         `json:"schema"`
	Level  string         `json:"level"`
	Event  string         `json:"event"`
	Fields map[string]any `json:"fields"`
}

// NewDiagnosticSink constructs a diagnostic sink whose enabled levels are at or
// above minLevel. sensitiveKeys are replaced with "[REDACTED]" without evaluating
// a LazyValue supplied for that key. Empty keys and invalid levels are rejected.
func NewDiagnosticSink(w io.Writer, minLevel DiagnosticLevel, sensitiveKeys ...string) (*DiagnosticSink, error) {
	if !minLevel.valid() {
		return nil, fmt.Errorf("usagelog: invalid diagnostic minimum level %d", minLevel)
	}
	sensitive := make(map[string]struct{}, len(sensitiveKeys))
	for _, key := range sensitiveKeys {
		if strings.TrimSpace(key) == "" {
			return nil, errors.New("usagelog: diagnostic sensitive key is empty")
		}
		sensitive[key] = struct{}{}
	}
	return &DiagnosticSink{w: w, minLevel: minLevel, sensitive: sensitive}, nil
}

// Emit writes one deterministic structured JSON line. kv must be alternating
// non-empty string keys and supported JSON scalar values (or LazyValue closures
// returning such values). Malformed input, duplicate keys, closure panics, and
// unsupported values fail before serialization. Filtered and disabled calls
// return before inspecting values, so lazy closures cannot run on those paths.
func (d *DiagnosticSink) Emit(level DiagnosticLevel, event string, kv ...any) error {
	if d == nil {
		return errors.New("usagelog: nil diagnostic sink")
	}
	if !level.valid() {
		return fmt.Errorf("usagelog: invalid diagnostic level %d", level)
	}
	if d.w == nil || level < d.minLevel {
		return nil
	}
	if strings.TrimSpace(event) == "" {
		return errors.New("usagelog: diagnostic event is empty")
	}
	if len(kv)%2 != 0 {
		return fmt.Errorf("usagelog: diagnostic key/value count %d is odd", len(kv))
	}

	fields := make(map[string]any, len(kv)/2)
	for i := 0; i < len(kv); i += 2 {
		key, ok := kv[i].(string)
		if !ok {
			return fmt.Errorf("usagelog: diagnostic key at index %d has type %T, want string", i, kv[i])
		}
		if strings.TrimSpace(key) == "" {
			return fmt.Errorf("usagelog: diagnostic key at index %d is empty", i)
		}
		if _, exists := fields[key]; exists {
			return fmt.Errorf("usagelog: duplicate diagnostic key %q", key)
		}
		fields[key] = nil
	}
	for i := 0; i < len(kv); i += 2 {
		key := kv[i].(string)
		if _, redact := d.sensitive[key]; redact {
			fields[key] = "[REDACTED]"
			continue
		}
		value, err := diagnosticScalar(kv[i+1])
		if err != nil {
			return fmt.Errorf("usagelog: diagnostic key %q: %w", key, err)
		}
		fields[key] = value
	}

	line, err := json.Marshal(diagnosticRecord{
		Schema: DiagnosticSchemaV1,
		Level:  level.String(),
		Event:  event,
		Fields: fields,
	})
	if err != nil {
		return fmt.Errorf("usagelog: marshal diagnostic: %w", err)
	}
	line = append(line, '\n')
	d.mu.Lock()
	defer d.mu.Unlock()
	n, err := d.w.Write(line)
	if err != nil {
		return fmt.Errorf("usagelog: write diagnostic: %w", err)
	}
	if n != len(line) {
		return io.ErrShortWrite
	}
	return nil
}

func (l DiagnosticLevel) valid() bool {
	return l >= DiagnosticDebug && l <= DiagnosticError
}

// String returns the stable JSON spelling of a diagnostic level.
func (l DiagnosticLevel) String() string {
	switch l {
	case DiagnosticDebug:
		return "debug"
	case DiagnosticInfo:
		return "info"
	case DiagnosticWarn:
		return "warn"
	case DiagnosticError:
		return "error"
	default:
		return ""
	}
}

func diagnosticScalar(value any) (resolved any, err error) {
	if lazy, ok := value.(LazyValue); ok {
		if lazy == nil {
			return nil, errors.New("nil lazy value")
		}
		func() {
			defer func() {
				if recover() != nil {
					err = errors.New("lazy value panicked")
				}
			}()
			resolved = lazy()
		}()
		if err != nil {
			return nil, err
		}
		value = resolved
	}

	switch value := value.(type) {
	case nil, string, bool,
		int, int8, int16, int32, int64,
		uint, uint8, uint16, uint32, uint64:
		return value, nil
	case float32:
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("unsupported value %v", value)
		}
		return value, nil
	case float64:
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, fmt.Errorf("unsupported value %v", value)
		}
		return value, nil
	default:
		return nil, fmt.Errorf("unsupported value type %T", value)
	}
}

// Row is one durable usage record: the on-disk JSONL schema for a single top-level
// `fak <verb>` invocation. Field order of the CHAINED fields (Schema..PID, see
// chainHash) is the hash-chain pre-image order — do not reorder without bumping the
// chain (it would invalidate every existing journal). Args is appended AFTER Hash
// and is NOT part of the pre-image (a bounded-disclosure layer), so a redacted
// journal and a --full-args journal share the same chain math.
type Row struct {
	Schema     string `json:"schema"`                // SchemaV1
	Seq        uint64 `json:"seq"`                   // monotonic 1-based order anchor
	TSUnixNano int64  `json:"ts_unix_nano"`          // wall-clock time anchor (process exit)
	Verb       string `json:"verb"`                  // the top-level verb (os.Args[1]); "" for the no-verb help path
	Argc       int    `json:"argc"`                  // number of args AFTER the verb (len(os.Args)-2, clamped >=0)
	ArgsDigest string `json:"args_digest,omitempty"` // salted sha256 over argv — commits to the args without disclosing them
	ExitCode   int    `json:"exit_code"`             // process exit status
	DurationMS int64  `json:"duration_ms"`           // wall-clock duration of the invocation
	FakVersion string `json:"fak_version,omitempty"` // appversion.Current() of the running tree/binary
	Host       string `json:"host,omitempty"`        // os.Hostname()
	PID        int    `json:"pid"`                   // process id
	PrevHash   string `json:"prev_hash"`             // hash of the previous row ("" at genesis)
	Hash       string `json:"hash"`                  // chainHash(PrevHash, this row)

	// Args is the RAW argv, recorded ONLY when the operator opts in via --full-args.
	// It is a bounded-disclosure convenience layered on top of the tamper-evident
	// chain: it is appended after Hash and is NOT part of the hash pre-image, so its
	// presence or absence never changes the chain (mirrors internal/journal's
	// correlation fields). Empty/omitted in the default redacted mode.
	Args []string `json:"args,omitempty"`
}

// Logger is a hash-chained, append-only usage journal. The zero value is not
// usable; construct with Open. It is safe for concurrent Append calls both within
// a process (l.mu) and ACROSS processes: Append re-syncs the chain head from disk
// under a cross-process advisory lock, so concurrent fak invocations extend one
// chain instead of forking it (#2608).
type Logger struct {
	mu       sync.Mutex
	bw       *bufio.Writer
	f        *os.File
	path     string
	seq      uint64
	lastHash string
	end      int64            // journal byte offset the head was recovered at (stale-head check)
	lockWait time.Duration    // how long Append polls for the cross-process lock
	clock    func() time.Time // injectable for deterministic tests
}

// ErrJournalBusy is returned by Append when the cross-process journal lock could
// not be acquired within the wait window. The caller (recordUsage) treats a usage
// row as best-effort, so dropping the row is preferred over forking the chain by
// appending off a stale head.
var ErrJournalBusy = errors.New("usagelog: journal lock busy")

// defaultLockWait bounds how long Append polls for the cross-process lock. The
// critical section is one stat + (rarely) one file scan + one flushed write, so
// even a heavily contended journal clears in well under this.
const defaultLockWait = 2 * time.Second

// Open opens (creating if absent) a file-backed usage journal in append mode. If
// the file already holds rows, Open recovers the chain head (seq + last hash) so a
// restart CONTINUES the same tamper-evident chain instead of forking it. The parent
// directory is created. A corrupt tail is reported by Verify, not here — Open stays
// robust so a damaged log never bricks a CLI invocation.
func Open(path string) (*Logger, error) {
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return nil, fmt.Errorf("usagelog: create dir %s: %w", dir, err)
		}
	}
	seq, last, end, err := recoverHead(path)
	if err != nil {
		return nil, err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return nil, fmt.Errorf("usagelog: open %s: %w", path, err)
	}
	return &Logger{
		bw:       bufio.NewWriter(f),
		f:        f,
		path:     path,
		seq:      seq,
		lastHash: last,
		end:      end,
		lockWait: defaultLockWait,
		clock:    time.Now,
	}, nil
}

// Append stamps the schema/order/time anchor + chain hash onto the caller-built row
// and commits it (write + flush, for per-row durability). The caller fills the
// usage fields (Verb, Argc, ArgsDigest, ExitCode, DurationMS, FakVersion, Host,
// PID, and optionally Args); Append owns Schema, Seq, TSUnixNano (if unset),
// PrevHash, and Hash. It returns the committed row (with the stamped fields) so a
// caller or test can inspect exactly what landed on disk.
//
// Head-recovery + write is ONE cross-process critical section: Append holds an
// exclusive advisory lock on the <path>.lock sidecar and re-syncs seq/lastHash
// from disk before stamping, because the head recovered at Open time is stale the
// moment another fak invocation appends. Without this, two racing processes both
// stamp the same seq off the same recovered head, and Verify later reports the
// fork as `sequence gap: seq=N want N+1` (#2608).
func (l *Logger) Append(r Row) (Row, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	unlock, err := lockJournal(l.path, l.lockWait)
	if err != nil {
		return r, err
	}
	defer unlock()
	if err := l.refreshHeadLocked(); err != nil {
		return r, err
	}
	r.Schema = SchemaV1
	l.seq++
	r.Seq = l.seq
	if r.TSUnixNano == 0 {
		r.TSUnixNano = l.clock().UnixNano()
	}
	r.PrevHash = l.lastHash
	r.Hash = chainHash(r.PrevHash, r)
	n, err := writeRow(l.bw, r)
	if err != nil {
		return r, err
	}
	l.lastHash = r.Hash
	l.end += int64(n)
	return r, nil
}

// refreshHeadLocked re-syncs the chain head (seq + last hash) from disk when the
// journal changed since the head was last recovered — i.e. another process
// appended between our Open (or last Append) and now. The size==end fast path
// keeps the uncontended case at one stat. Caller must hold BOTH l.mu and the
// cross-process journal lock, so the head cannot move again before our write.
func (l *Logger) refreshHeadLocked() error {
	if st, err := os.Stat(l.path); err == nil && st.Size() == l.end {
		return nil // nothing landed since recovery: the cached head is still the tail
	}
	seq, last, end, err := recoverHead(l.path)
	if err != nil {
		return err
	}
	l.seq, l.lastHash, l.end = seq, last, end
	return nil
}

// lockJournal takes the cross-process exclusive advisory lock that guards the
// journal's chain head: a <path>.lock sidecar over internal/flock (mirroring
// loopmgr's ledger-append critical section). flock.TryLock is non-blocking, so it
// polls until the lock is free or wait elapses (then ErrJournalBusy — the caller
// drops the best-effort row rather than forking the chain). Closing the returned
// fd releases the OS lock, including when the holder process dies mid-write.
func lockJournal(path string, wait time.Duration) (unlock func(), err error) {
	lf, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, fmt.Errorf("usagelog: open journal lock: %w", err)
	}
	deadline := time.Now().Add(wait)
	for {
		lerr := flock.TryLock(lf)
		if lerr == nil {
			break
		}
		if !errors.Is(lerr, flock.ErrLockBusy) {
			_ = lf.Close()
			return nil, fmt.Errorf("usagelog: lock journal: %w", lerr)
		}
		if time.Now().After(deadline) {
			_ = lf.Close()
			return nil, ErrJournalBusy
		}
		time.Sleep(5 * time.Millisecond)
	}
	return func() {
		_ = flock.Unlock(lf)
		_ = lf.Close()
	}, nil
}

// Path returns the file the logger appends to — what to point `fak audit verify` at.
func (l *Logger) Path() string { return l.path }

// Close flushes, fsyncs, and closes the file.
func (l *Logger) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.bw != nil {
		if err := l.bw.Flush(); err != nil {
			return err
		}
	}
	if l.f != nil {
		_ = l.f.Sync()
		err := l.f.Close()
		l.f, l.bw = nil, nil
		return err
	}
	return nil
}

// writeRow appends one JSONL row and flushes it (per-row durability). It returns
// the number of bytes committed (line + newline) so the logger can advance its
// cached end-of-journal offset without an extra stat.
func writeRow(bw *bufio.Writer, row Row) (int, error) {
	b, err := json.Marshal(row)
	if err != nil {
		return 0, err
	}
	if _, err := bw.Write(b); err != nil {
		return 0, err
	}
	if err := bw.WriteByte('\n'); err != nil {
		return 0, err
	}
	return len(b) + 1, bw.Flush()
}

// chainHash is the tamper-evident link: sha256 over the previous row's hash chained
// with this row's content fields (Schema..PID, in declaration order). PrevHash and
// Hash are excluded from the pre-image (PrevHash is the chained-in prefix; Hash is
// the output), and so is the bounded-disclosure Args field. A unit separator (0x1f)
// delimits fields so no concatenation collision is possible. ArgsDigest is part of
// the pre-image, so the chain commits to WHICH args were used even though the raw
// argv is redacted.
func chainHash(prev string, r Row) string {
	h := sha256.New()
	io.WriteString(h, prev)
	fmt.Fprintf(h, "\x1f%s\x1f%d\x1f%d\x1f%s\x1f%d\x1f%s\x1f%d\x1f%d\x1f%s\x1f%s\x1f%d",
		r.Schema, r.Seq, r.TSUnixNano, r.Verb, r.Argc, r.ArgsDigest,
		r.ExitCode, r.DurationMS, r.FakVersion, r.Host, r.PID)
	return hex.EncodeToString(h.Sum(nil))
}

// Digest is the redaction primitive: a salted sha256 over the argv slice, returned
// as "sha256:<hex>". It COMMITS to the args (the same argv under the same salt
// always yields the same digest, so frequency analysis of a repeated command is
// possible for the salt holder) without DISCLOSING them (the raw bytes — paths,
// `-m` messages, tokens — never touch disk). An empty argv hashes to the salt
// alone, a stable non-empty value. A per-user persistent salt (see LoadOrCreateSalt)
// also defeats a dictionary/rainbow attack over guessable commands.
func Digest(salt []byte, args []string) string {
	h := sha256.New()
	h.Write(salt)
	for _, a := range args {
		h.Write([]byte{0x1f})
		io.WriteString(h, a)
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}

// recoverHead scans an existing journal to recover the chain head (last seq + last
// hash) so an append continues the same chain, plus the byte offset the head was
// recovered at (the stale-head check Append's critical section compares against
// the live file size). A missing file is the genesis case (seq 0, empty hash,
// offset 0). It does NOT validate the chain (that is Verify's job) so a damaged
// log never blocks a CLI invocation; a torn final line (a crash mid-write) is
// tolerated by stopping at the last well-formed row.
func recoverHead(path string) (seq uint64, lastHash string, end int64, err error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, "", 0, nil
		}
		return 0, "", 0, fmt.Errorf("usagelog: stat %s: %w", path, err)
	}
	defer f.Close()
	base, err := tailScanStart(f)
	if err != nil {
		return 0, "", 0, fmt.Errorf("usagelog: scan %s: %w", path, err)
	}
	seq, lastHash, end, err = scanHeadFrom(f, path, base)
	if err != nil {
		return 0, "", 0, err
	}
	if base > 0 && lastHash == "" {
		// The window opened AFTER the last row terminator (a single row longer than
		// the window, or a torn tail that swallowed it), so it yielded no head at
		// all. Re-scan from byte 0: returning the genesis head here would restart
		// seq at 1 and FORK the chain, the exact #2608 failure this function exists
		// to prevent.
		return scanHeadFrom(f, path, 0)
	}
	return seq, lastHash, end, nil
}

// recoverTailWindow bounds how far back from EOF recoverHead starts scanning. The
// chain head is the LAST row, so every row before the window can only be parsed and
// thrown away — but the journal is append-only and unbounded, so scanning from byte
// 0 made every `fak` invocation re-parse the whole invocation history of the machine.
// That is a per-spawn cost that grows without limit: at 21 MB it measured ~300 ms on
// the reference host, dwarfing every other part of startup, and every hook that
// shells `fak` paid it once per turn (#5626). 256 KiB holds thousands of rows; the
// guard in recoverHead covers the case where it does not hold even one.
const recoverTailWindow = 256 << 10

// tailScanStart returns the row boundary recoverHead should start scanning at: the
// byte after the first newline at or after EOF-recoverTailWindow, so a scan never
// starts mid-row. It returns 0 for a journal smaller than the window, or for a
// window holding no newline at all — both cases fall back to the original
// whole-file scan.
//
// Starting past the older rows means a torn line OLDER than the window no longer
// stops the scan. That changes only the already-corrupt case, and in the safer
// direction: the head becomes the true last row, so appends continue the live chain
// instead of forking off a truncated prefix. Detecting corruption was never this
// function's job — Verify owns that, and Open is deliberately robust so a damaged
// log cannot brick a CLI invocation.
func tailScanStart(f *os.File) (int64, error) {
	st, err := f.Stat()
	if err != nil {
		return 0, err
	}
	if st.Size() <= recoverTailWindow {
		return 0, nil // small journal: the whole-file scan already IS a tail scan
	}
	from := st.Size() - recoverTailWindow
	buf := make([]byte, recoverTailWindow)
	n, err := f.ReadAt(buf, from)
	if err != nil && !errors.Is(err, io.EOF) {
		return 0, err
	}
	i := bytes.IndexByte(buf[:n], '\n')
	if i < 0 {
		return 0, nil
	}
	return from + int64(i) + 1, nil
}

// scanHeadFrom runs the head-recovery scan from an absolute byte offset, reporting
// end in absolute file offsets so a caller (and Append's stale-head check) can
// compare it against the live file size regardless of where the scan started.
func scanHeadFrom(f *os.File, path string, base int64) (seq uint64, lastHash string, end int64, err error) {
	if _, err := f.Seek(base, io.SeekStart); err != nil {
		return 0, "", 0, fmt.Errorf("usagelog: scan %s: %w", path, err)
	}
	end = base
	br := bufio.NewReaderSize(f, 64*1024)
	for {
		chunk, rerr := br.ReadBytes('\n')
		if len(chunk) > 0 {
			if body := bytes.TrimSpace(chunk); len(body) > 0 {
				var r Row
				if err := json.Unmarshal(body, &r); err != nil {
					// Torn/malformed line: stop at the last well-formed row (Verify
					// catches real corruption). end stays at the offset BEFORE it.
					return seq, lastHash, end, nil
				}
				seq = r.Seq
				lastHash = r.Hash
			}
			end += int64(len(chunk)) // blank lines advance the offset without moving the head
		}
		if rerr == io.EOF {
			return seq, lastHash, end, nil
		}
		if rerr != nil {
			return 0, "", 0, fmt.Errorf("usagelog: scan %s: %w", path, rerr)
		}
	}
}

// ReadRows reads all committed rows from a usage journal, in order — the READ side
// for a consumer (the `fak usage` fold) that wants the rows as data, not an
// integrity check (use Verify for that). It is robust for a live reader: a MISSING
// file is the empty journal (nil, nil), and a torn final line is tolerated by
// stopping at the last well-formed row.
func ReadRows(path string) ([]Row, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("usagelog: read %s: %w", path, err)
	}
	defer f.Close()
	var out []Row
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var r Row
		if err := json.Unmarshal(line, &r); err != nil {
			break
		}
		out = append(out, r)
	}
	if err := sc.Err(); err != nil {
		return out, fmt.Errorf("usagelog: scan %s: %w", path, err)
	}
	return out, nil
}

// Verify re-reads a usage journal and validates the hash chain end to end. It
// returns the number of rows checked and a non-nil error naming the FIRST broken
// link (a recomputed-hash mismatch, a prev-hash discontinuity, or a sequence gap).
// A journal that passes Verify has not been edited since it was written.
func Verify(path string) (n int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, fmt.Errorf("usagelog: open %s: %w", path, err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	var (
		prev    string
		wantSeq uint64
	)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var row Row
		if err := json.Unmarshal(line, &row); err != nil {
			return n, fmt.Errorf("usagelog: row %d: malformed JSON: %w", n+1, err)
		}
		wantSeq++
		if row.Seq != wantSeq {
			return n, fmt.Errorf("usagelog: sequence gap: seq=%d want %d", row.Seq, wantSeq)
		}
		if row.PrevHash != prev {
			return n, fmt.Errorf("usagelog: broken chain at seq %d: prev_hash=%q want %q", row.Seq, row.PrevHash, prev)
		}
		if got := chainHash(row.PrevHash, row); got != row.Hash {
			return n, fmt.Errorf("usagelog: tampered row at seq %d: hash=%q recomputed %q", row.Seq, row.Hash, got)
		}
		prev = row.Hash
		n++
	}
	if err := sc.Err(); err != nil {
		return n, fmt.Errorf("usagelog: scan: %w", err)
	}
	return n, nil
}

// Enabled reports whether the usage journal should record this invocation. It is ON
// by default (like the audit journal) and OFF only when FAK_USAGE_LOG is set to
// "off" (case-insensitive) — the single opt-out a sandbox, benchmark, or
// privacy-conscious operator flips to write nothing at all.
func Enabled() bool {
	return !strings.EqualFold(strings.TrimSpace(os.Getenv("FAK_USAGE_LOG")), "off")
}

// DefaultPath is where the usage journal is appended when the operator names none:
// <user-config>/fak/usage.jsonl — a stable, per-user, cross-platform location
// appended across sessions so the tamper-evident chain CONTINUES rather than forking
// each run (mirroring guardDefaultAuditPath for the decision journal). Falls back to
// ".fak/usage.jsonl" under the working directory if no user config dir resolves.
func DefaultPath() string {
	if dir, err := os.UserConfigDir(); err == nil && strings.TrimSpace(dir) != "" {
		return filepath.Join(dir, "fak", "usage.jsonl")
	}
	return filepath.Join(".fak", "usage.jsonl")
}

// DefaultSaltPath is the per-user salt file sitting beside DefaultPath. The salt is
// what makes a redacted args_digest resistant to a dictionary attack over guessable
// commands while still letting the salt holder count how often an exact command ran.
func DefaultSaltPath() string {
	return filepath.Join(filepath.Dir(DefaultPath()), "usage.salt")
}

// LoadOrCreateSalt returns the persistent per-user redaction salt at path, creating
// it (32 random bytes, 0600) on first use. A persistent salt makes the same command
// hash to the same args_digest across invocations (so `fak usage` could count a
// repeated command) without ever making the salt — or the argv — guessable. The
// parent directory is created.
func LoadOrCreateSalt(path string) ([]byte, error) {
	if b, err := os.ReadFile(path); err == nil && len(b) > 0 {
		return b, nil
	}
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return nil, fmt.Errorf("usagelog: create salt dir %s: %w", dir, err)
		}
	}
	salt := make([]byte, 32)
	if _, err := rand.Read(salt); err != nil {
		return nil, fmt.Errorf("usagelog: generate salt: %w", err)
	}
	if err := os.WriteFile(path, salt, 0o600); err != nil {
		return nil, fmt.Errorf("usagelog: write salt %s: %w", path, err)
	}
	return salt, nil
}
