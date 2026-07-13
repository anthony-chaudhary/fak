package sessionledger

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

type Hash string

type Entry struct {
	Hash    Hash            `json:"hash"`
	Parent  Hash            `json:"parent,omitempty"`
	Kind    string          `json:"kind"`
	Content json.RawMessage `json:"content,omitempty"`
}

type diskState struct {
	Heads map[string]Hash `json:"heads"`
	Nodes map[Hash]Entry  `json:"nodes"`
}

type Ledger struct {
	mu    sync.RWMutex
	path  string
	heads map[string]Hash
	nodes map[Hash]Entry
}

func Open(dir string) (*Ledger, error) {
	l := &Ledger{path: filepath.Join(dir, "ledger.json"), heads: map[string]Hash{}, nodes: map[Hash]Entry{}}
	b, err := os.ReadFile(l.path)
	if errors.Is(err, os.ErrNotExist) {
		return l, nil
	}
	if err != nil {
		return nil, err
	}
	var d diskState
	if err = json.Unmarshal(b, &d); err != nil {
		return nil, err
	}
	if d.Heads != nil {
		l.heads = d.Heads
	}
	if d.Nodes != nil {
		l.nodes = d.Nodes
	}
	return l, nil
}

func DefaultDir() string {
	if d := os.Getenv("FAK_SESSION_LEDGER_DIR"); d != "" {
		return d
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

func (l *Ledger) Append(trace, kind string, content []byte) (Entry, error) {
	if trace == "" || kind == "" {
		return Entry{}, errors.New("sessionledger: trace and kind are required")
	}
	if len(content) > 0 && !json.Valid(content) {
		return Entry{}, errors.New("sessionledger: content must be JSON")
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	parent := l.heads[trace]
	c := bytes.Clone(content)
	e := Entry{Parent: parent, Kind: kind, Content: c}
	e.Hash = digest(parent, kind, c)
	l.nodes[e.Hash] = e
	l.heads[trace] = e.Hash
	return e, l.saveLocked()
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
	return h, l.saveLocked()
}
func (l *Ledger) Head(trace string) Hash { l.mu.RLock(); defer l.mu.RUnlock(); return l.heads[trace] }
func (l *Ledger) NodeCount() int         { l.mu.RLock(); defer l.mu.RUnlock(); return len(l.nodes) }
func (l *Ledger) Chain(trace string) ([]Entry, error) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	h, ok := l.heads[trace]
	if !ok {
		return nil, fmt.Errorf("sessionledger: trace %q not found", trace)
	}
	var rev []Entry
	for h != "" {
		e, ok := l.nodes[h]
		if !ok {
			return nil, fmt.Errorf("sessionledger: missing node %s", h)
		}
		rev = append(rev, e)
		h = e.Parent
	}
	out := make([]Entry, len(rev))
	for i := range rev {
		out[len(rev)-1-i] = rev[i]
	}
	return out, nil
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
func (l *Ledger) saveLocked() error {
	if l.path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(l.path), 0700); err != nil {
		return err
	}
	b, err := json.Marshal(diskState{Heads: l.heads, Nodes: l.nodes})
	if err != nil {
		return err
	}
	tmp := l.path + ".tmp"
	if err = os.WriteFile(tmp, b, 0600); err != nil {
		return err
	}
	return os.Rename(tmp, l.path)
}
