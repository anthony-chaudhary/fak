// Package modelpack manages signed, resumable model artifacts and fixture-gated activation.
package modelpack

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Schema specifies the canonical metadata format identifier for signed model packs.
const Schema = "fak.model-pack-manifest/1"

// Chunk describes a content-addressed storage slice verified by its SHA-256 digest.
type Chunk struct {
	Digest string `json:"digest"`
	Size   int64  `json:"size"`
}

// Fixture defines a verification test vector executed during activation canary evaluation.
type Fixture struct {
	Name     string `json:"name"`
	Input    string `json:"input"`
	Expected string `json:"expected"`
}

// Manifest defines the complete specification of an installable model pack revision.
type Manifest struct {
	Schema    string    `json:"schema"`
	PackID    string    `json:"pack_id"`
	Revision  string    `json:"revision"`
	Chunks    []Chunk   `json:"chunks"`
	Fixtures  []Fixture `json:"fixtures,omitempty"`
	Signature string    `json:"signature,omitempty"`
}

// Event captures an immutable state transition recorded in the pack lifecycle history.
type Event struct {
	Sequence uint64    `json:"sequence"`
	At       time.Time `json:"at"`
	State    string    `json:"state"`
	PackID   string    `json:"pack_id"`
	Revision string    `json:"revision"`
	Detail   string    `json:"detail,omitempty"`
}

// Receipt acknowledges a verified state change witnessed by cryptographic digest and sequence.
type Receipt struct {
	Schema   string `json:"schema"`
	PackID   string `json:"pack_id"`
	Revision string `json:"revision"`
	State    string `json:"state"`
	Sequence uint64 `json:"sequence"`
	Digest   string `json:"digest"`
}

type state struct {
	Active        map[string]string `json:"active"`
	LastKnownGood map[string]string `json:"last_known_good"`
	Revoked       map[string]bool   `json:"revoked"`
	Events        []Event           `json:"events"`
}

// Manager orchestrates local model pack storage, download resumption, canary validation, and eviction.
type Manager struct {
	root string
	now  func() time.Time
	s    state
}

// Fetch represents a pluggable content-addressed chunk retrieval callback with byte offset support.
type Fetch func(digest string, offset int64, dst io.Writer) error

// Canary represents an evaluation hook invoked prior to promoting staged revisions to active state.
type Canary func(packDir string, fixtures []Fixture) error

// ErrSignature indicates that the manifest cryptographic signature could not be verified against the public key.
var ErrSignature = errors.New("modelpack: invalid manifest signature")

// ErrCorrupt indicates that a downloaded chunk payload failed hash verification against its declared digest.
var ErrCorrupt = errors.New("modelpack: chunk digest mismatch")

// ErrCapacity indicates that remaining storage budget is insufficient to satisfy the forecasted download size.
var ErrCapacity = errors.New("modelpack: insufficient reserved storage")

// ErrRevoked indicates that the requested pack revision has been marked invalid and must not be activated.
var ErrRevoked = errors.New("modelpack: revision revoked")

func canonical(m Manifest) ([]byte, error) { m.Signature = ""; return json.Marshal(m) }

// Sign serializes the manifest canonical structure and attaches an Ed25519 signature.
func Sign(m *Manifest, key ed25519.PrivateKey) error {
	b, e := canonical(*m)
	if e != nil {
		return e
	}
	m.Signature = hex.EncodeToString(ed25519.Sign(key, b))
	return nil
}

// Verify enforces cryptographic signature validity and schema correctness for a manifest.
//
// Key invariant: any manifest with an invalid signature or malformed chunk digest is rejected before storage is modified.
func Verify(m Manifest, key ed25519.PublicKey) error {
	if m.Schema != Schema || m.PackID == "" || m.Revision == "" {
		return fmt.Errorf("%w: invalid identity", ErrSignature)
	}
	sig, e := hex.DecodeString(m.Signature)
	if e != nil {
		return ErrSignature
	}
	b, e := canonical(m)
	if e != nil {
		return e
	}
	if !ed25519.Verify(key, b, sig) {
		return ErrSignature
	}
	for _, c := range m.Chunks {
		if len(c.Digest) != 64 || c.Size < 0 {
			return fmt.Errorf("%w: invalid chunk", ErrSignature)
		}
	}
	return nil
}

// Open loads or initializes persistent model pack management state from the specified root directory.
func Open(root string) (*Manager, error) {
	if err := os.MkdirAll(filepath.Join(root, "chunks"), 0755); err != nil {
		return nil, err
	}
	m := &Manager{root: root, now: func() time.Time { return time.Now().UTC() }, s: state{Active: map[string]string{}, LastKnownGood: map[string]string{}, Revoked: map[string]bool{}}}
	b, e := os.ReadFile(filepath.Join(root, "state.json"))
	if e == nil {
		if e = json.Unmarshal(b, &m.s); e != nil {
			return nil, e
		}
	} else if !os.IsNotExist(e) {
		return nil, e
	}
	if m.s.Active == nil {
		m.s.Active = map[string]string{}
	}
	if m.s.LastKnownGood == nil {
		m.s.LastKnownGood = map[string]string{}
	}
	if m.s.Revoked == nil {
		m.s.Revoked = map[string]bool{}
	}
	return m, nil
}

func (m *Manager) persist() error {
	b, e := json.MarshalIndent(m.s, "", "  ")
	if e != nil {
		return e
	}
	tmp := filepath.Join(m.root, "state.json.tmp")
	if e = os.WriteFile(tmp, b, 0644); e != nil {
		return e
	}
	return os.Rename(tmp, filepath.Join(m.root, "state.json"))
}

func key(id, rev string) string { return id + "@" + rev }

func (m *Manager) emit(man Manifest, stateName, detail string) (Receipt, error) {
	e := Event{Sequence: uint64(len(m.s.Events) + 1), At: m.now(), State: stateName, PackID: man.PackID, Revision: man.Revision, Detail: detail}
	m.s.Events = append(m.s.Events, e)
	if err := m.persist(); err != nil {
		return Receipt{}, err
	}
	raw, _ := json.Marshal(e)
	sum := sha256.Sum256(raw)
	return Receipt{Schema: "fak.model-pack-receipt/1", PackID: man.PackID, Revision: man.Revision, State: stateName, Sequence: e.Sequence, Digest: hex.EncodeToString(sum[:])}, nil
}

// Events returns a defensive copy of the sequential audit history recorded by the manager.
func (m *Manager) Events() []Event { return append([]Event(nil), m.s.Events...) }

// Active returns the currently activated revision identifier for the given pack ID.
func (m *Manager) Active(id string) string { return m.s.Active[id] }

// Forecast calculates the additional unacquired storage bytes required to complete pack acquisition.
//
// Assumption: partial download files accurately reflect already received byte counts on disk.
func (m *Manager) Forecast(man Manifest) int64 {
	var n int64
	for _, c := range man.Chunks {
		p := filepath.Join(m.root, "chunks", c.Digest)
		if st, e := os.Stat(p); e != nil || st.Size() != c.Size {
			part := p + ".part"
			off := int64(0)
			if st, e = os.Stat(part); e == nil {
				off = st.Size()
			}
			if off < c.Size {
				n += c.Size - off
			}
		}
	}
	return n
}

// Install verifies authority before fetching, resumes partial chunks, stages atomically,
// and activates only after the caller's task-fixture canary passes.
//
// Fail-closed guard: if verification fails, capacity is exceeded, or canary evaluation errors, staging is aborted cleanly.
func (m *Manager) Install(man Manifest, pub ed25519.PublicKey, capacity int64, fetch Fetch, canary Canary) (Receipt, error) {
	if err := Verify(man, pub); err != nil {
		r, _ := m.emit(man, "refused", "signature")
		return r, err
	}
	if m.s.Revoked[key(man.PackID, man.Revision)] {
		r, _ := m.emit(man, "refused", "revoked")
		return r, ErrRevoked
	}
	need := m.Forecast(man)
	if need > capacity {
		r, _ := m.emit(man, "refused", fmt.Sprintf("capacity need=%d available=%d", need, capacity))
		return r, ErrCapacity
	}
	m.emit(man, "reserved", fmt.Sprintf("bytes=%d", need))
	for _, c := range man.Chunks {
		final := filepath.Join(m.root, "chunks", c.Digest)
		if validFile(final, c) {
			continue
		}
		part := final + ".part"
		off := int64(0)
		if st, e := os.Stat(part); e == nil {
			off = st.Size()
			if off > c.Size {
				os.Remove(part)
				off = 0
			}
		}
		f, e := os.OpenFile(part, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if e != nil {
			return Receipt{}, e
		}
		e = fetch(c.Digest, off, f)
		ce := f.Close()
		if e != nil {
			r, _ := m.emit(man, "interrupted", c.Digest)
			return r, e
		}
		if ce != nil {
			return Receipt{}, ce
		}
		if !validFile(part, c) {
			r, _ := m.emit(man, "refused", "corrupt "+c.Digest)
			return r, ErrCorrupt
		}
		if e = os.Rename(part, final); e != nil {
			return Receipt{}, e
		}
	}
	stage := filepath.Join(m.root, "staging", man.PackID+"-"+man.Revision)
	os.RemoveAll(stage)
	if err := os.MkdirAll(stage, 0755); err != nil {
		return Receipt{}, err
	}
	b, _ := json.MarshalIndent(man, "", "  ")
	if err := os.WriteFile(filepath.Join(stage, "manifest.json"), b, 0644); err != nil {
		return Receipt{}, err
	}
	for _, c := range man.Chunks {
		if err := os.Link(filepath.Join(m.root, "chunks", c.Digest), filepath.Join(stage, c.Digest)); err != nil {
			return Receipt{}, err
		}
	}
	if canary != nil {
		if err := canary(stage, man.Fixtures); err != nil {
			os.RemoveAll(stage)
			r, _ := m.emit(man, "canary_failed", err.Error())
			return r, err
		}
	}
	dest := filepath.Join(m.root, "packs", man.PackID, man.Revision)
	if err := os.MkdirAll(filepath.Dir(dest), 0755); err != nil {
		return Receipt{}, err
	}
	os.RemoveAll(dest)
	if err := os.Rename(stage, dest); err != nil {
		return Receipt{}, err
	}
	if old := m.s.Active[man.PackID]; old != "" && old != man.Revision {
		m.s.LastKnownGood[man.PackID] = old
	}
	m.s.Active[man.PackID] = man.Revision
	return m.emit(man, "activated", "")
}
func validFile(path string, c Chunk) bool {
	f, e := os.Open(path)
	if e != nil {
		return false
	}
	defer f.Close()
	h := sha256.New()
	n, e := io.Copy(h, f)
	return e == nil && n == c.Size && hex.EncodeToString(h.Sum(nil)) == strings.ToLower(c.Digest)
}

// Revoke disables the designated pack revision and restores last-known-good state if active.
//
// Invariant: revoked revisions can never be activated, reinstalled, or retained as active state.
func (m *Manager) Revoke(id, rev string) (Receipt, error) {
	man := Manifest{PackID: id, Revision: rev}
	m.s.Revoked[key(id, rev)] = true
	if m.s.Active[id] == rev {
		if lkg := m.s.LastKnownGood[id]; lkg != "" && !m.s.Revoked[key(id, lkg)] {
			m.s.Active[id] = lkg
		} else {
			delete(m.s.Active, id)
		}
	}
	return m.emit(man, "revoked", "")
}

// Rollback reactivates the recorded last-known-good revision for the given pack identifier.
func (m *Manager) Rollback(id string) (Receipt, error) {
	rev := m.s.LastKnownGood[id]
	if rev == "" || m.s.Revoked[key(id, rev)] {
		return Receipt{}, errors.New("modelpack: no last-known-good revision")
	}
	old := m.s.Active[id]
	m.s.Active[id] = rev
	m.s.LastKnownGood[id] = old
	return m.emit(Manifest{PackID: id, Revision: rev}, "rolled_back", "")
}

// Evict removes inactive revisions oldest-name-first while retaining active and last-known-good revisions.
//
// Postcondition: active and last-known-good revisions remain untouched regardless of the byte eviction quota.
func (m *Manager) Evict(bytes int64) (int64, error) {
	root := filepath.Join(m.root, "packs")
	var paths []string
	_ = filepath.Walk(root, func(p string, info os.FileInfo, e error) error {
		if e == nil && info.IsDir() && p != root && filepath.Dir(filepath.Dir(p)) == root {
			paths = append(paths, p)
		}
		return nil
	})
	sort.Strings(paths)
	var freed int64
	for _, p := range paths {
		if freed >= bytes {
			break
		}
		id := filepath.Base(filepath.Dir(p))
		rev := filepath.Base(p)
		if m.s.Active[id] == rev || m.s.LastKnownGood[id] == rev {
			continue
		}
		var size int64
		_ = filepath.Walk(p, func(_ string, i os.FileInfo, e error) error {
			if e == nil && !i.IsDir() {
				size += i.Size()
			}
			return nil
		})
		if e := os.RemoveAll(p); e != nil {
			return freed, e
		}
		freed += size
		m.emit(Manifest{PackID: id, Revision: rev}, "evicted", fmt.Sprintf("bytes=%d", size))
	}
	return freed, nil
}
