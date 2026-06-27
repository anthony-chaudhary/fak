// Package blobfs is a DURABLE, on-disk content-addressed store — the persistent
// sibling of internal/blob (the in-memory v0.1 default behind every abi.Ref).
// Where blob keeps every Ref's bytes in a process-lifetime map (gone on restart),
// blobfs writes them to a sharded directory tree under FAK_BLOB_DIR, so a
// paged-out / quarantined / cached payload SURVIVES a process bounce and is
// shareable across processes on the same host.
//
// It attaches to the FROZEN ABI exactly the way blob does — implements
// abi.Resolver (Put/Resolve), the optional abi.CASPinner (Pin/Unpin for
// GC-safety), and abi.PageOutBackend (PageOut/PageIn) — but it registers its
// page-out codec under a NEW id ("blobfs"), NOT the default "blob", so it
// COEXISTS with the in-memory codec rather than silently replacing it (the keyed
// page-out registry is plural by design; the architest singleton gate pins id
// "blob" to package blob). A consumer opts into durable page-out by selecting the
// "blobfs" codec (the context-MMU reads FAK_PAGEOUT_BACKEND); the storedrv router
// composes blobfs as its durable tier so the SAME content-addressed namespace
// spans memory and disk.
//
// Content addressing is the load-bearing property shared with blob: the sha256
// digest IS the file name, so a byte-identical payload is stored exactly once and
// a digest written by one process resolves in another. Writes are crash-safe — a
// temp file fsync'd then atomically renamed into place — so a torn write never
// leaves a corrupt blob under a valid digest. Small payloads stay inline on the
// Ref (RefInline, durable by virtue of living in the Ref itself), avoiding a disk
// round-trip on the hot path, exactly as blob does.
//
// ENABLEMENT. blobfs is OPT-IN and inert by default: writing every Ref to disk is
// a deployment choice, not something a unit test or a benchmark should pay for.
// Set FAK_BLOB_DIR=/path/to/store and the package's init registers a durable
// page-out codec under id "blobfs"; unset, no codec is registered and the package
// is inert (the FAK_AUDIT_JOURNAL-style env toggle the rest of the kernel uses).
package blobfs

import (
	"container/list"
	"context"
	"encoding/hex"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/blob"
)

// InlineMax mirrors blob.InlineMax: a payload this small or smaller is returned
// inline on the Ref (durable in the Ref itself) instead of touching disk.
const InlineMax = blob.InlineMax

// digestHexLen is the length of a sha256 content address in hex (32 bytes).
const digestHexLen = 64

// tmpPrefix names the temp files an in-flight Put creates inside a shard dir; the
// Open scan skips any name carrying it so a crashed Put's leftover is never
// counted as (or resolved as) a blob.
const tmpPrefix = ".blobfs-tmp-"

// Store is a concurrency-safe, durable, content-addressed blob store rooted at a
// directory. The on-disk layout is sharded by the first two hex bytes of the
// digest (dir/<aa>/<bb>/<digest>) so no single directory holds an unbounded fan.
//
// An in-memory index (digest -> size) is seeded by a one-time scan at Open so
// dedup, the byte-budget GC, and the Len/Bytes taps work across a restart without
// re-reading payloads. The GC is pin-aware: a digest a live holder will resolve
// later (a held quarantine handle, a router cache entry) is Pin'd and never the
// thing deleted; everything else is evictable FIFO once the resident footprint
// exceeds maxBytes. maxBytes <= 0 disables GC (durable append-only).
type Store struct {
	root string

	mu       sync.Mutex
	index    map[string]int64         // digest -> on-disk byte size
	bytes    int64                    // total bytes resident on disk (O(1) tap)
	maxBytes int64                    // 0 => unbounded (no GC)
	pins     map[string]int           // digest -> pin count (>0 => protected, kept OUT of order)
	order    *list.List               // evictable (unpinned) digests; back = oldest
	orderIdx map[string]*list.Element // digest -> its order element (unpinned, resident)

	puts    int64
	hits    int64 // Put of an already-present digest (content dedup)
	resolv  int64
	evicted int64 // digests deleted by the byte budget
}

// New opens (creating if absent) a durable store rooted at dir, seeding the index
// from any blobs already on disk so a restart continues the same store. The byte
// budget comes from FAK_BLOB_DIR_MAX_BYTES (0/unset => unbounded).
func New(dir string) (*Store, error) { return NewWithBudget(dir, maxBytesFromEnv()) }

// NewWithBudget is New with an explicit resident-byte budget (a non-positive
// budget disables GC). It is the seam the GC-regression test uses with a small
// bound; the bound is pin-aware, so a pinned digest is never the thing deleted.
func NewWithBudget(dir string, maxBytes int64) (*Store, error) {
	if dir == "" {
		return nil, fmt.Errorf("blobfs: empty store directory")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("blobfs: create store dir %s: %w", dir, err)
	}
	s := &Store{
		root:     dir,
		index:    map[string]int64{},
		maxBytes: maxBytes,
		pins:     map[string]int{},
		order:    list.New(),
		orderIdx: map[string]*list.Element{},
	}
	if err := s.scan(); err != nil {
		return nil, err
	}
	return s, nil
}

func maxBytesFromEnv() int64 {
	if v, ok := os.LookupEnv("FAK_BLOB_DIR_MAX_BYTES"); ok {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil && n >= 0 {
			return n
		}
	}
	return 0 // durable stores default to unbounded; disk is cheaper than RAM
}

// scan walks the store tree once and seeds the index + footprint from the digests
// already on disk. It reads only directory entries (names + sizes), never payload
// bytes, so the cost is one stat per resident blob — the durable-store analogue of
// journal.recoverHead. A file whose name is not a 64-hex digest (a temp leftover,
// a stray) is ignored, so a crashed Put never pollutes the index.
func (s *Store) scan() error {
	return filepath.WalkDir(s.root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		name := d.Name()
		if !isDigest(name) {
			return nil // temp file, stray, or partial — not a committed blob
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		s.index[name] = info.Size()
		s.bytes += info.Size()
		s.orderIdx[name] = s.order.PushFront(name)
		return nil
	})
}

// isDigest reports whether name is a lowercase 64-char hex sha256 — the canonical
// blob file name. Anything else (a temp prefix, a shard dir, a stray) is skipped.
func isDigest(name string) bool {
	if len(name) != digestHexLen {
		return false
	}
	_, err := hex.DecodeString(name)
	return err == nil
}

// pathFor returns the sharded on-disk path for a digest: root/<aa>/<bb>/<digest>.
// Two levels of 2-hex sharding cap any single directory's fan at 256 entries for
// the shard dirs and spread blobs across 65536 leaf dirs.
func (s *Store) pathFor(digest string) string {
	return filepath.Join(s.root, digest[0:2], digest[2:4], digest)
}

// Put stores b and returns an addressable Ref. Small payloads (<= InlineMax) are
// returned inline and never touch disk; larger ones are written content-addressed
// and a byte-identical payload already on disk is a dedup hit (no rewrite). The
// write is crash-safe: bytes land in a temp file that is fsync'd and atomically
// renamed into place, so a digest path either does not exist or holds the whole
// payload — never a torn prefix.
func (s *Store) Put(ctx context.Context, b []byte) (abi.Ref, error) {
	r, inline := blob.PreparePut(b)
	if inline {
		return r, nil
	}
	if err := s.commit(ctx, r.Digest, b); err != nil {
		return abi.Ref{}, err
	}
	return r, nil
}

// commit stores b on disk under its digest unconditionally (content-addressed,
// idempotent: a digest already resident is a dedup hit). It is the shared store
// path behind Put's large branch AND PageOut — PageOut must persist even a small
// body (a 50-byte quarantined injection string still has to page out to a handle),
// so it cannot reuse Put's inline shortcut.
func (s *Store) commit(ctx context.Context, d string, b []byte) error {
	s.mu.Lock()
	atomic.AddInt64(&s.puts, 1)
	if _, ok := s.index[d]; ok {
		atomic.AddInt64(&s.hits, 1)
		s.mu.Unlock()
		return nil
	}
	s.mu.Unlock()

	// Write OUTSIDE the lock (disk I/O is slow; the rename is the commit point).
	if err := s.writeBlob(d, b); err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	// A concurrent commit of the same digest may have landed while we wrote; the
	// rename is idempotent (same bytes) so only the first to index it counts.
	if _, ok := s.index[d]; !ok {
		s.index[d] = int64(len(b))
		s.bytes += int64(len(b))
		if s.pins[d] == 0 {
			s.orderIdx[d] = s.order.PushFront(d)
		}
		s.evictLocked()
	}
	return nil
}

// writeBlob writes b to its sharded path via a temp file + atomic rename. The
// rename is the durability commit: an interrupted write leaves only a temp file
// (skipped by scan and reclaimable), never a corrupt blob under a valid digest.
func (s *Store) writeBlob(digest string, b []byte) error {
	final := s.pathFor(digest)
	dir := filepath.Dir(final)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("blobfs: shard dir %s: %w", dir, err)
	}
	if _, err := os.Stat(final); err == nil {
		return nil // already committed by a peer process/goroutine — dedup
	}
	tmp, err := os.CreateTemp(dir, tmpPrefix+"*")
	if err != nil {
		return fmt.Errorf("blobfs: temp file in %s: %w", dir, err)
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(b); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return fmt.Errorf("blobfs: write %s: %w", tmpName, err)
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return fmt.Errorf("blobfs: fsync %s: %w", tmpName, err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("blobfs: close %s: %w", tmpName, err)
	}
	if err := os.Rename(tmpName, final); err != nil {
		os.Remove(tmpName)
		// A racing peer may have created final between our Stat and Rename; that is
		// a dedup win, not an error (same content addresses to the same bytes).
		if _, statErr := os.Stat(final); statErr == nil {
			return nil
		}
		return fmt.Errorf("blobfs: commit %s: %w", final, err)
	}
	return nil
}

// Resolve materializes the bytes a Ref points at. Inline Refs carry their own
// bytes; RefBlob/RefRegion read the sharded file by digest. The read is lock-free:
// a committed blob is immutable (content-addressed), and the GC never deletes a
// pinned digest, so a live holder that pinned its handle always resolves.
func (s *Store) Resolve(ctx context.Context, r abi.Ref) ([]byte, error) {
	switch r.Kind {
	case abi.RefInline:
		return append([]byte(nil), r.Inline...), nil
	case abi.RefBlob, abi.RefRegion:
		atomic.AddInt64(&s.resolv, 1)
		b, err := os.ReadFile(s.pathFor(r.Digest))
		if err != nil {
			if os.IsNotExist(err) {
				return nil, fmt.Errorf("blobfs: unknown digest %s", r.Digest)
			}
			return nil, fmt.Errorf("blobfs: resolve %s: %w", r.Digest, err)
		}
		return b, nil
	default:
		return nil, fmt.Errorf("blobfs: unknown RefKind %d", r.Kind)
	}
}

// PageOut moves a (possibly inline) Ref's bytes onto disk and returns a handle Ref
// carrying no inline bytes — the durable analogue of blob.PageOut. After PageOut
// the handle resolves from disk, so a quarantined/cold result paged out through
// blobfs survives a process restart.
func (s *Store) PageOut(ctx context.Context, r abi.Ref) (abi.Ref, error) {
	b, err := s.Resolve(ctx, r)
	if err != nil {
		return abi.Ref{}, err
	}
	// Persist unconditionally (commit, not Put): page-out's contract is "bytes out
	// of context behind a pointer", so even a small body must land on disk and the
	// handle must not re-inline it — otherwise the RefBlob handle would not resolve.
	d := blob.Digest(b)
	if err := s.commit(ctx, d, b); err != nil {
		return abi.Ref{}, err
	}
	return abi.Ref{Kind: abi.RefBlob, Digest: d, Len: int64(len(b)), Taint: r.Taint, Scope: r.Scope}, nil
}

// PageIn re-materializes a paged-out handle Ref into an inline Ref.
func (s *Store) PageIn(ctx context.Context, handle abi.Ref) (abi.Ref, error) {
	return blob.PageIn(ctx, s, handle)
}

// Pin protects a digest from GC for as long as a live holder will resolve it
// (abi.CASPinner). Refcounted, so a digest shared by several holders survives
// until the last Unpin. A no-op for the empty digest. Safe before or after Put.
func (s *Store) Pin(digest string) {
	if digest == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pins[digest]++
	if s.pins[digest] == 1 {
		if el, ok := s.orderIdx[digest]; ok {
			s.order.Remove(el)
			delete(s.orderIdx, digest)
		}
	}
}

// Unpin releases one Pin; when the last holder unpins, the digest becomes
// evictable again (re-entered at the order front if still resident). A no-op if
// not pinned (abi.CASPinner).
func (s *Store) Unpin(digest string) {
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
		if _, ok := s.index[digest]; ok {
			s.orderIdx[digest] = s.order.PushFront(digest)
		}
		s.evictLocked()
		return
	}
	s.pins[digest] = n - 1
}

// evictLocked deletes unpinned blobs (oldest first) until the on-disk footprint is
// within maxBytes. Pinned digests are never in the order list, so they are never
// deleted; if only pinned digests remain, the footprint legitimately exceeds the
// bound and the loop stops (it bounds the leak, not the live working set). A
// delete that fails on disk is dropped from the index regardless (the file may be
// gone already); the byte accounting tracks the index, not the filesystem. Caller
// holds s.mu.
func (s *Store) evictLocked() {
	if s.maxBytes <= 0 {
		return
	}
	for s.bytes > s.maxBytes {
		el := s.order.Back()
		if el == nil {
			return // everything resident is pinned (live) — nothing safe to delete
		}
		d := el.Value.(string)
		s.order.Remove(el)
		delete(s.orderIdx, d)
		if sz, ok := s.index[d]; ok {
			s.bytes -= sz
			delete(s.index, d)
			_ = os.Remove(s.pathFor(d))
			atomic.AddInt64(&s.evicted, 1)
		}
	}
}

// Stats reports store activity (puts, dedup hits, resolves) for KPI taps.
func (s *Store) Stats() (puts, dedupHits, resolves int64) {
	return atomic.LoadInt64(&s.puts), atomic.LoadInt64(&s.hits), atomic.LoadInt64(&s.resolv)
}

// Resident reports the current resident store size (blob count, total bytes) and
// the lifetime count of blobs deleted by the byte budget — the durable analogue of
// blob.Resident.
func (s *Store) Resident() (blobCount int, bytes, evicted int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.index), s.bytes, atomic.LoadInt64(&s.evicted)
}

// Root reports the store's on-disk root directory.
func (s *Store) Root() string { return s.root }

// ----------------------------------------------------------------------------
// ABI registration: an OPT-IN durable page-out codec under id "blobfs".
// ----------------------------------------------------------------------------

var active *Store

// Active returns the registered durable store, or nil if FAK_BLOB_DIR was unset at
// boot (the package is inert). The storedrv router uses this to compose blobfs as
// its durable tier without re-opening the directory.
func Active() *Store { return active }

func init() {
	dir := os.Getenv("FAK_BLOB_DIR")
	if dir == "" {
		return // off by default: no codec registered, package inert
	}
	s, err := New(dir)
	if err != nil {
		// Fail loud but do not brick the kernel: a missing durable sidecar must not
		// stop adjudication (the in-memory blob store still serves). An operator who
		// requires durable storage learns from the stderr line that it did not load.
		fmt.Fprintf(os.Stderr, "fak: durable blob store disabled — %v\n", err)
		return
	}
	active = s
	abi.RegisterPageOutBackend("blobfs", pageOutBackend{s})
	fmt.Fprintf(os.Stderr, "fak: durable blob store -> %s (content-addressed, id=blobfs)\n", dir)
}

// pageOutBackend adapts *Store to abi.PageOutBackend for the keyed registry.
type pageOutBackend struct{ s *Store }

// PageOut persists a Ref's bytes to the durable store and returns a bytes-absent
// handle (abi.PageOutBackend, delegating to the underlying Store).
func (b pageOutBackend) PageOut(ctx context.Context, r abi.Ref) (abi.Ref, error) {
	return b.s.PageOut(ctx, r)
}

// PageIn re-materializes a paged-out handle from the durable store (abi.PageOutBackend).
func (b pageOutBackend) PageIn(ctx context.Context, h abi.Ref) (abi.Ref, error) {
	return b.s.PageIn(ctx, h)
}
