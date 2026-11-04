package ctxmmu_test

import (
	"container/list"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// casFixture is the bounded content-addressed store the ctxmmu suite drives its
// page-out / CAS-pin tests against. It mirrors internal/blob.Store semantics
// (LRU eviction, refcounted digest pins, the 256-byte inline threshold) but
// lives inside the test binary so the package's import graph names no concrete
// backend — the MMU reaches it only through the abi.RegisterRegionBackend /
// RegisterPageOutBackend seams, exactly like production wiring.
type casFixture struct {
	mu       sync.Mutex
	blobs    map[string][]byte
	bytes    int64
	maxBytes int64
	pins     map[string]int
	lru      *list.List
	lruIndex map[string]*list.Element
}

var testCAS = newCasFixture()

func init() {
	// The test package's init runs after every imported package's init
	// (including blob's, pulled in transitively by the registrations benchmark
	// import), and both registries are last-wins, so the fixture is the
	// authoritative "blob" codec and region resolver for the whole suite.
	abi.RegisterRegionBackend(testCAS)
	abi.RegisterPageOutBackend("blob", testCAS)
}

func newCasFixture() *casFixture {
	return &casFixture{
		blobs:    make(map[string][]byte),
		maxBytes: 1 << 30,
		pins:     make(map[string]int),
		lru:      list.New(),
		lruIndex: make(map[string]*list.Element),
	}
}

// digestOf returns the canonical sha256 content address for b.
func digestOf(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// storeLocked records a fresh blob and evicts down to the byte bound. Pinned
// digests never sit in the LRU (Pin removes them), so only unpinned entries are
// pushed. Callers hold mu.
func (s *casFixture) storeLocked(d string, b []byte) {
	s.blobs[d] = append([]byte(nil), b...)
	s.bytes += int64(len(b))
	if s.pins[d] == 0 {
		s.lruIndex[d] = s.lru.PushFront(d)
	}
	s.evictExcessLocked()
}

// evictExcessLocked drops least-recently-used blobs until the resident bytes fit
// the bound. Callers hold mu.
func (s *casFixture) evictExcessLocked() {
	if s.maxBytes <= 0 {
		return
	}
	for s.bytes > s.maxBytes {
		el := s.lru.Back()
		if el == nil {
			return
		}
		d := el.Value.(string)
		s.lru.Remove(el)
		delete(s.lruIndex, d)
		if b, ok := s.blobs[d]; ok {
			s.bytes -= int64(len(b))
			delete(s.blobs, d)
		}
	}
}

// Put stores bytes in the CAS or inline and returns the corresponding
// addressable Ref, mirroring blob.Store.Put (dedup hits leave the LRU alone).
func (s *casFixture) Put(ctx context.Context, b []byte) (abi.Ref, error) {
	r := abi.Ref{Digest: digestOf(b), Len: int64(len(b)), Taint: abi.TaintTainted, Scope: abi.ScopeAgent}
	if len(b) <= 256 {
		r.Kind = abi.RefInline
		r.Inline = append([]byte(nil), b...)
		return r, nil
	}
	r.Kind = abi.RefBlob // RefInline is the zero value; the CAS path must set Kind explicitly
	d := r.Digest
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.blobs[d]; !ok {
		s.storeLocked(d, b)
	}
	return r, nil
}

// Resolve materializes the bytes a Ref points at, returning a copy on every hit.
func (s *casFixture) Resolve(ctx context.Context, r abi.Ref) ([]byte, error) {
	switch r.Kind {
	case abi.RefInline:
		return append([]byte(nil), r.Inline...), nil
	case abi.RefBlob, abi.RefRegion:
		s.mu.Lock()
		defer s.mu.Unlock()
		b, ok := s.blobs[r.Digest]
		if !ok {
			return nil, fmt.Errorf("casfixture: unknown digest %s", r.Digest)
		}
		return append([]byte(nil), b...), nil
	default:
		return nil, fmt.Errorf("casfixture: unknown RefKind %d", r.Kind)
	}
}

// PageOut offloads a Ref payload into the CAS and returns a handle Ref,
// preserving the input Ref's taint and scope exactly like blob.Store.PageOut.
func (s *casFixture) PageOut(ctx context.Context, r abi.Ref) (abi.Ref, error) {
	b, err := s.Resolve(ctx, r)
	if err != nil {
		return abi.Ref{}, err
	}
	d := digestOf(b)
	s.mu.Lock()
	if _, ok := s.blobs[d]; !ok {
		s.storeLocked(d, b)
	}
	s.mu.Unlock()
	return abi.Ref{Kind: abi.RefBlob, Digest: d, Len: int64(len(b)), Taint: r.Taint, Scope: r.Scope}, nil
}

// PageIn re-materializes a paged-out handle Ref into an inline Ref carrying a
// fresh copy of the resolved bytes.
func (s *casFixture) PageIn(ctx context.Context, h abi.Ref) (abi.Ref, error) {
	b, err := s.Resolve(ctx, h)
	if err != nil {
		return abi.Ref{}, err
	}
	return abi.Ref{Kind: abi.RefInline, Digest: h.Digest, Inline: b, Len: int64(len(b)), Taint: h.Taint, Scope: h.Scope}, nil
}

// Pin increments the refcounted protection preventing eviction of digest; the
// first pin pulls the digest out of the evictable LRU entirely.
func (s *casFixture) Pin(digest string) {
	if digest == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pins[digest]++
	if s.pins[digest] == 1 {
		if el, ok := s.lruIndex[digest]; ok {
			s.lru.Remove(el)
			delete(s.lruIndex, digest)
		}
	}
}

// Unpin decrements pin protection, restoring digest to evictable LRU status
// (pushed back to the front) when the count reaches zero.
func (s *casFixture) Unpin(digest string) {
	if digest == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	n := s.pins[digest]
	if n <= 0 {
		return
	}
	if n == 1 {
		delete(s.pins, digest)
		if _, ok := s.blobs[digest]; ok {
			s.lruIndex[digest] = s.lru.PushFront(digest)
		}
		s.evictExcessLocked()
		return
	}
	s.pins[digest] = n - 1
}

// MaxBytes reports the configured resident-footprint ceiling (0 = unbounded).
func (s *casFixture) MaxBytes() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.maxBytes
}

// SetMaxBytes updates the resident byte limit and immediately evicts eligible
// unpinned blobs down to it.
func (s *casFixture) SetMaxBytes(n int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.maxBytes = n
	s.evictExcessLocked()
}

// Bytes returns the total payload bytes currently resident in the CAS.
func (s *casFixture) Bytes() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.bytes
}

// Resolver returns the fixture itself as the abi.Resolver (it also satisfies
// abi.CASPinner via Pin/Unpin, so gated page-ins stay sound under eviction).
func (s *casFixture) Resolver() abi.Resolver { return s }

// Caps returns backend capabilities supported by the fixture.
func (s *casFixture) Caps() []abi.Capability { return nil }
