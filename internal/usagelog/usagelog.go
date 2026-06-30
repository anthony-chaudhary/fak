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
// package imports nothing internal) so a usage trail can be reasoned about and
// verified on its own:
//
//   - DURABLE: one JSONL row is appended and flushed per Append, written at process
//     exit so the exit code is known. A crash loses only the in-flight invocation.
//   - SEQUENCE/TIME ANCHORED: the logger stamps its own monotonic Seq (1-based) and
//     a wall-clock timestamp on every row.
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
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// SchemaV1 is the on-disk schema tag stamped into every row's `schema` field, so a
// reader can tell a usage journal apart from any other JSONL ledger and refuse to
// fold a row whose schema it does not understand.
const SchemaV1 = "fak-usage-log/1"

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
// usable; construct with Open. It is safe for concurrent Append calls (a single
// process rarely needs that, but the lock keeps the chain head consistent).
type Logger struct {
	mu       sync.Mutex
	bw       *bufio.Writer
	f        *os.File
	path     string
	seq      uint64
	lastHash string
	clock    func() time.Time // injectable for deterministic tests
}

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
	seq, last, err := recoverHead(path)
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
		clock:    time.Now,
	}, nil
}

// Append stamps the schema/order/time anchor + chain hash onto the caller-built row
// and commits it (write + flush, for per-row durability). The caller fills the
// usage fields (Verb, Argc, ArgsDigest, ExitCode, DurationMS, FakVersion, Host,
// PID, and optionally Args); Append owns Schema, Seq, TSUnixNano (if unset),
// PrevHash, and Hash. It returns the committed row (with the stamped fields) so a
// caller or test can inspect exactly what landed on disk.
func (l *Logger) Append(r Row) (Row, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	r.Schema = SchemaV1
	l.seq++
	r.Seq = l.seq
	if r.TSUnixNano == 0 {
		r.TSUnixNano = l.clock().UnixNano()
	}
	r.PrevHash = l.lastHash
	r.Hash = chainHash(r.PrevHash, r)
	l.lastHash = r.Hash
	if err := writeRow(l.bw, r); err != nil {
		return r, err
	}
	return r, nil
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
// hash) so an append continues the same chain. A missing file is the genesis case
// (seq 0, empty hash). It does NOT validate the chain (that is Verify's job) so a
// damaged log never blocks a CLI invocation; a torn final line (a crash mid-write)
// is tolerated by stopping at the last well-formed row.
func recoverHead(path string) (seq uint64, lastHash string, err error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, "", nil
		}
		return 0, "", fmt.Errorf("usagelog: stat %s: %w", path, err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var r Row
		if err := json.Unmarshal(line, &r); err != nil {
			break // torn final line: stop at the last well-formed row (Verify catches real corruption)
		}
		seq = r.Seq
		lastHash = r.Hash
	}
	if err := sc.Err(); err != nil {
		return 0, "", fmt.Errorf("usagelog: scan %s: %w", path, err)
	}
	return seq, lastHash, nil
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
