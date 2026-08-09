// Package sessionledger is the durable per-trace hash chain the gateway and the RSI
// loop append to at every turn boundary.
//
// Storage is an APPEND-ONLY JSONL log (`ledger.jsonl`): one record per line, one
// write per Append. It used to be a single `ledger.json` holding the whole state,
// re-marshalled and rewritten on EVERY append -- O(n^2) in both allocation and disk
// I/O. With the gateway persisting a full request body per turn that reached a
// 389 MB file, and every append re-serialized it: ~800 MB of transient garbage per
// turn per process, times one guard per live agent. Appending one bounded line
// instead keeps a turn's cost proportional to the turn, not to all history.
//
// Three bounds keep the log from becoming the same problem again:
//
//   - MaxContentBytes elides an oversized content blob down to a provenance stub
//     (byte count + sha256), so no single record can be large.
//   - MaxFileBytes rotates the log to `.1` so on-disk stays bounded.
//   - MaxNodes evicts oldest-first so in-memory stays bounded.
//
// Concurrency: several guards share one directory. Each Append is a single
// open(O_APPEND)/write/close of one bounded line, so writers interleave by whole
// records instead of clobbering each other's view the way the old whole-file
// rename did. A torn or malformed line is SKIPPED on read rather than failing the
// load -- a partial write must not make the whole ledger unreadable.
package sessionledger

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// Bounds. Exported so a caller (and the tests) can reason about them.
const (
	// MaxContentBytes is the largest content blob a single entry may retain
	// verbatim. Anything larger is replaced by an elision stub. A turn's request
	// body is the motivating case: it is provenance, not payload we need to keep.
	MaxContentBytes = 8 << 10 // 8 KiB

	// MaxFileBytes is the on-disk ceiling before the log rotates to `.1`.
	MaxFileBytes = 32 << 20 // 32 MiB

	// MaxNodes is the in-memory entry ceiling; oldest are evicted first.
	MaxNodes = 50_000
)

type Hash string

type Entry struct {
	Hash    Hash            `json:"hash"`
	Parent  Hash            `json:"parent,omitempty"`
	Kind    string          `json:"kind"`
	Content json.RawMessage `json:"content,omitempty"`
}

// record is one JSONL line. Kind == "" marks a head-only move (a Fork), which
// carries no new node.
type record struct {
	Trace   string          `json:"trace"`
	Hash    Hash            `json:"hash"`
	Parent  Hash            `json:"parent,omitempty"`
	Kind    string          `json:"kind,omitempty"`
	Content json.RawMessage `json:"content,omitempty"`
}

type Ledger struct {
	mu    sync.RWMutex
	path  string
	heads map[string]Hash
	nodes map[Hash]Entry
	order []Hash // insertion order, for oldest-first eviction
}

// LogName is the append-only log's filename within a ledger directory.
const LogName = "ledger.jsonl"

// legacyName is the pre-JSONL whole-state file. It is set aside on open, never
// loaded: by the time this shipped the live one was 389 MB, and loading it would
// reproduce the very blowup this package now avoids.
const legacyName = "ledger.json"

func Open(dir string) (*Ledger, error) {
	l := &Ledger{
		path:  filepath.Join(dir, LogName),
		heads: map[string]Hash{},
		nodes: map[Hash]Entry{},
	}
	retireLegacy(dir)
	f, err := os.Open(l.path)
	if errors.Is(err, os.ErrNotExist) {
		return l, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	// A record is bounded by MaxContentBytes plus framing; give the scanner room
	// for that plus slack, and skip any line longer rather than erroring.
	sc.Buffer(make([]byte, 0, 64<<10), MaxContentBytes+(64<<10))
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var r record
		if json.Unmarshal(line, &r) != nil {
			continue // torn or malformed line: skip, keep the rest of the log
		}
		l.applyRecord(r)
	}
	// A scanner error (an over-long line) leaves what we already read intact.
	return l, nil
}

// retireLegacy renames a pre-JSONL ledger.json aside exactly once, so a fresh
// process neither loads it nor trips over it. Best effort: a failure here must
// not stop the ledger from opening.
func retireLegacy(dir string) {
	old := filepath.Join(dir, legacyName)
	if _, err := os.Stat(old); err != nil {
		return
	}
	_ = os.Rename(old, freeLegacyName(old+".legacy"))
	_ = os.Remove(filepath.Join(dir, legacyName+".tmp"))
}

// freeLegacyName returns a destination that does not already exist. Retirement
// happens more than once in practice: while a mixed fleet is rolling, an older
// build keeps recreating ledger.json and each new process retires it again. A
// bare os.Rename OVERWRITES its destination on POSIX, which would silently
// destroy the previously retired file -- the one case this whole path exists to
// prevent. Falls back to the plain name only after an absurd number of tries,
// which cannot happen without something else being badly wrong.
func freeLegacyName(base string) string {
	if _, err := os.Stat(base); errors.Is(err, os.ErrNotExist) {
		return base
	}
	for i := 1; i < 10000; i++ {
		candidate := fmt.Sprintf("%s.%d", base, i)
		if _, err := os.Stat(candidate); errors.Is(err, os.ErrNotExist) {
			return candidate
		}
	}
	return base
}

// applyRecord folds one decoded line into state. Caller holds the lock (or is
// still constructing the ledger).
func (l *Ledger) applyRecord(r record) {
	if r.Kind != "" && r.Hash != "" {
		l.putNode(Entry{Hash: r.Hash, Parent: r.Parent, Kind: r.Kind, Content: r.Content})
	}
	if r.Trace != "" && r.Hash != "" {
		l.heads[r.Trace] = r.Hash
	}
}

// putNode inserts an entry and enforces MaxNodes, evicting oldest-first.
func (l *Ledger) putNode(e Entry) {
	if _, dup := l.nodes[e.Hash]; dup {
		return
	}
	l.nodes[e.Hash] = e
	l.order = append(l.order, e.Hash)
	for len(l.order) > MaxNodes {
		oldest := l.order[0]
		l.order = l.order[1:]
		delete(l.nodes, oldest)
	}
}

// underTest reports whether this is a `go test` binary. Only the test harness
// registers test.v on the default flag set, and it does so before any test runs.
// Checked via flag rather than importing testing, which would add test flags to
// the production CLI.
func underTest() bool { return flag.Lookup("test.v") != nil }

func DefaultDir() string {
	if d := os.Getenv("FAK_SESSION_LEDGER_DIR"); d != "" {
		return d
	}
	// A test binary must never open the OPERATOR's ledger. No test in this repo
	// sets FAK_SESSION_LEDGER_DIR, and the gateway suite reaches this seam through
	// handleAnthropicMessages -- so `go test ./internal/gateway/` was appending to
	// the real ledger under the user's config dir, and (with the legacy retirement
	// below) renaming their live file aside. Isolate per test binary instead.
	if underTest() {
		return filepath.Join(os.TempDir(), fmt.Sprintf("fak-test-session-ledger-%d", os.Getpid()))
	}
	if d, err := os.UserConfigDir(); err == nil {
		return filepath.Join(d, "fak", "session-ledger")
	}
	return filepath.Join(os.TempDir(), "fak", "session-ledger")
}

var (
	defaultOnce   sync.Once
	defaultLedger *Ledger
	defaultErr    error
)

func OpenDefault() (*Ledger, error) {
	defaultOnce.Do(func() { defaultLedger, defaultErr = Open(DefaultDir()) })
	return defaultLedger, defaultErr
}

func Memory() *Ledger { return &Ledger{heads: map[string]Hash{}, nodes: map[Hash]Entry{}} }

func digest(parent Hash, kind string, content []byte) Hash {
	h := sha256.New()
	h.Write([]byte(parent))
	h.Write([]byte{0})
	h.Write([]byte(kind))
	h.Write([]byte{0})
	h.Write(content)
	return Hash(hex.EncodeToString(h.Sum(nil)))
}

// Elide replaces an oversized JSON blob with a bounded provenance stub: how many
// bytes it was and the sha256 of those bytes. The chain still witnesses that the
// content existed and what it hashed to; it just stops carrying the payload.
func Elide(content []byte) json.RawMessage {
	sum := sha256.Sum256(content)
	stub := struct {
		Elided bool   `json:"elided"`
		Bytes  int    `json:"bytes"`
		SHA256 string `json:"sha256"`
	}{true, len(content), hex.EncodeToString(sum[:])}
	b, err := json.Marshal(stub)
	if err != nil { // unreachable for this struct
		return json.RawMessage(`{"elided":true}`)
	}
	return json.RawMessage(b)
}

func (l *Ledger) Append(trace, kind string, content []byte) (Entry, error) {
	if trace == "" || kind == "" {
		return Entry{}, errors.New("sessionledger: trace and kind are required")
	}
	if len(content) > 0 && !json.Valid(content) {
		return Entry{}, errors.New("sessionledger: content must be JSON")
	}
	c := bytes.Clone(content)
	if len(c) > MaxContentBytes {
		c = Elide(content)
	}

	l.mu.Lock()
	defer l.mu.Unlock()
	parent := l.heads[trace]
	e := Entry{Parent: parent, Kind: kind, Content: c}
	e.Hash = digest(parent, kind, c)
	l.putNode(e)
	l.heads[trace] = e.Hash
	return e, l.appendRecord(record{
		Trace: trace, Hash: e.Hash, Parent: e.Parent, Kind: e.Kind, Content: e.Content,
	})
}

// Fork creates a new trace head that points at the same immutable node as source.
// It only updates the head map; no entries or content bytes are copied.
func (l *Ledger) Fork(source, target string) (Hash, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if target == "" {
		return "", errors.New("sessionledger: target trace is required")
	}
	h, ok := l.heads[source]
	if !ok {
		return "", fmt.Errorf("sessionledger: source trace %q not found", source)
	}
	l.heads[target] = h
	// Kind "" ⇒ head-only move, no new node.
	return h, l.appendRecord(record{Trace: target, Hash: h})
}

func (l *Ledger) Head(trace string) Hash { l.mu.RLock(); defer l.mu.RUnlock(); return l.heads[trace] }
func (l *Ledger) NodeCount() int         { l.mu.RLock(); defer l.mu.RUnlock(); return len(l.nodes) }

// Chain walks trace's head back to its root, oldest-first. If an ancestor has been
// evicted by the MaxNodes bound the walk STOPS there and returns the surviving
// suffix rather than failing -- a bounded ledger legitimately forgets its oldest
// entries, and a truncated-but-true history beats an error. Verify still rejects
// such a suffix as unrooted, which is the honest answer for it.
func (l *Ledger) Chain(trace string) ([]Entry, error) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	h, ok := l.heads[trace]
	if !ok {
		return nil, fmt.Errorf("sessionledger: trace %q not found", trace)
	}
	// The walk itself is chainFromLocked (rewind.go), shared with ChainFrom so a rewound
	// suffix is replayed by exactly the same code that replays a live trace.
	return l.chainFromLocked(h)
}

func Verify(entries []Entry) error {
	var parent Hash
	for i, e := range entries {
		if e.Parent != parent {
			return fmt.Errorf("entry %d parent mismatch", i)
		}
		if got := digest(e.Parent, e.Kind, e.Content); got != e.Hash {
			return fmt.Errorf("entry %d hash mismatch", i)
		}
		parent = e.Hash
	}
	return nil
}

// appendRecord writes ONE line. Caller holds the write lock. Opening per append
// costs a syscall we can afford (a few per turn) and buys multi-process safety:
// each writer appends whole records to whatever inode currently holds the name,
// so a rotation by a sibling process cannot strand this one's writes in a file
// nobody reads.
func (l *Ledger) appendRecord(r record) error {
	if l.path == "" {
		return nil // Memory()
	}
	if err := os.MkdirAll(filepath.Dir(l.path), 0700); err != nil {
		return err
	}
	line, err := json.Marshal(r)
	if err != nil {
		return err
	}
	line = append(line, '\n')

	l.rotateIfLarge(len(line))

	f, err := os.OpenFile(l.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	if _, err = f.Write(line); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// rotateIfLarge renames the log to `.1` once it would exceed MaxFileBytes, so the
// directory holds at most two generations. Best effort: if the rename loses a race
// with a sibling process the next append simply lands in the fresh file.
func (l *Ledger) rotateIfLarge(incoming int) {
	st, err := os.Stat(l.path)
	if err != nil || st.Size()+int64(incoming) <= MaxFileBytes {
		return
	}
	_ = os.Rename(l.path, l.path+".1")
}
