package ctxmmu_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// mockAsyncDiskBackend simulates an asynchronous CAS / disk backend where writeback
// to disk is deferred / in-flight until Flush() is called. Before Flush(), PageIn
// fails with ErrDiskWritePending (reproducing the immediate restore race in #10018).
type mockAsyncDiskBackend struct {
	mu       sync.Mutex
	pending  map[string][]byte
	onDisk   map[string][]byte
	pageIns  int64
	pageOuts int64
}

var errDiskWritePending = errors.New("mockcas: disk writeback in flight; not yet on disk")

func newMockAsyncDiskBackend() *mockAsyncDiskBackend {
	return &mockAsyncDiskBackend{
		pending: make(map[string][]byte),
		onDisk:  make(map[string][]byte),
	}
}

func (b *mockAsyncDiskBackend) Caps() []abi.Capability { return nil }

func (b *mockAsyncDiskBackend) PageOut(ctx context.Context, r abi.Ref) (abi.Ref, error) {
	atomic.AddInt64(&b.pageOuts, 1)
	body := r.Inline
	sum := sha256.Sum256(body)
	d := hex.EncodeToString(sum[:])

	b.mu.Lock()
	defer b.mu.Unlock()
	// Staged as pending writeback; NOT yet committed to disk!
	b.pending[d] = append([]byte(nil), body...)
	return abi.Ref{
		Kind:   abi.RefBlob,
		Digest: d,
		Len:    int64(len(body)),
		Taint:  r.Taint,
		Scope:  r.Scope,
	}, nil
}

func (b *mockAsyncDiskBackend) PageIn(ctx context.Context, h abi.Ref) (abi.Ref, error) {
	atomic.AddInt64(&b.pageIns, 1)
	b.mu.Lock()
	defer b.mu.Unlock()

	// If only pending in writeback queue and not flushed to disk, fail with error
	if _, ok := b.onDisk[h.Digest]; !ok {
		return abi.Ref{}, errDiskWritePending
	}

	data := b.onDisk[h.Digest]
	return abi.Ref{
		Kind:   abi.RefInline,
		Digest: h.Digest,
		Inline: append([]byte(nil), data...),
		Len:    int64(len(data)),
		Taint:  h.Taint,
		Scope:  h.Scope,
	}, nil
}

func (b *mockAsyncDiskBackend) Flush() {
	b.mu.Lock()
	defer b.mu.Unlock()
	for d, data := range b.pending {
		b.onDisk[d] = data
	}
	b.pending = make(map[string][]byte)
}

func computeHexSHA256(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// TestPagedStore_ImmediateRestoreBeforeDiskFlush proves that when an oversize result is paged out,
// an immediate restore request via ResolvePagedRef / ResolvePaged succeeds deterministically
// from in-memory staging even when the background CAS disk writeback has not yet completed (#10018).
func TestPagedStore_ImmediateRestoreBeforeDiskFlush(t *testing.T) {
	ctx := context.Background()

	const codecID = "async_disk_sim"
	backend := newMockAsyncDiskBackend()
	abi.RegisterPageOutBackend(codecID, backend)

	t.Setenv("FAK_PAGEOUT_BACKEND", codecID)
	m := ctxmmu.New()

	// 1. Create an oversize benign result (> 4096 bytes)
	originalContent := bytes.Repeat([]byte("alpha-bravo-charlie-delta-echo-foxtrot-"), 150) // ~6000 bytes
	c := call("fetch_large_artifact")
	r := result(c, originalContent)

	// 2. Admit through MMU: should trigger page-out to pointer stub
	v := m.Admit(ctx, c, r)
	if v.Kind != abi.VerdictTransform {
		t.Fatalf("expected VerdictTransform for oversize result, got %v", v.Kind)
	}

	tp, ok := v.Payload.(abi.TransformPayload)
	if !ok {
		t.Fatalf("payload is not TransformPayload: %T", v.Payload)
	}

	stubBytes := resolveBody(t, ctx, tp.NewArgs)
	var stub map[string]any
	if err := json.Unmarshal(stubBytes, &stub); err != nil {
		t.Fatalf("unmarshal stub: %v, raw=%s", err, stubBytes)
	}
	if paged, _ := stub["_paged"].(bool); !paged {
		t.Fatalf("expected _paged=true, got %+v", stub)
	}
	ref, _ := stub["ref"].(string)
	if ref == "" {
		t.Fatalf("expected non-empty ref in stub")
	}

	// 3. Confirm that the mock backend's disk writeback is STILL PENDING (disk PageIn fails)
	_, diskErr := backend.PageIn(ctx, abi.Ref{Kind: abi.RefBlob, Digest: ref})
	if !errors.Is(diskErr, errDiskWritePending) {
		t.Fatalf("expected backend PageIn to fail with errDiskWritePending before flush, got %v", diskErr)
	}

	// 4. IMMEDIATE RESTORE BEFORE DISK FLUSH:
	// Both m.ResolvePagedRef and ctxmmu.ResolvePaged must succeed via in-memory staging fallback!
	restoredDirect, okDirect := m.ResolvePagedRef(ctx, ref)
	if !okDirect {
		t.Fatalf("m.ResolvePagedRef failed to restore paged ref %s before disk flush", ref)
	}
	if !bytes.Equal(restoredDirect, originalContent) {
		t.Fatalf("m.ResolvePagedRef restored bytes mismatch: got %d bytes, want %d bytes", len(restoredDirect), len(originalContent))
	}

	restoredGlobal, okGlobal := ctxmmu.ResolvePaged(ctx, ref)
	if !okGlobal {
		t.Fatalf("ctxmmu.ResolvePaged failed to restore paged ref %s before disk flush", ref)
	}
	if !bytes.Equal(restoredGlobal, originalContent) {
		t.Fatalf("ctxmmu.ResolvePaged restored bytes mismatch: got %d bytes, want %d bytes", len(restoredGlobal), len(originalContent))
	}

	// 5. Also verify with "sha256:" prefix
	restoredPrefixed, okPrefixed := m.ResolvePagedRef(ctx, "sha256:"+ref)
	if !okPrefixed || !bytes.Equal(restoredPrefixed, originalContent) {
		t.Fatalf("m.ResolvePagedRef failed with sha256: prefix")
	}

	// 6. Now simulate background CAS writeback completion (Flush to disk)
	backend.Flush()

	// Verify disk PageIn now succeeds
	if _, err := backend.PageIn(ctx, abi.Ref{Kind: abi.RefBlob, Digest: ref}); err != nil {
		t.Fatalf("backend PageIn should succeed after Flush: %v", err)
	}

	// And restore continues to succeed seamlessly
	restoredAfterFlush, okAfter := m.ResolvePagedRef(ctx, ref)
	if !okAfter || !bytes.Equal(restoredAfterFlush, originalContent) {
		t.Fatalf("m.ResolvePagedRef failed after disk flush")
	}
}

// TestPagedStore_UnknownRefRefused proves that unpaged or non-existent digests return false.
func TestPagedStore_UnknownRefRefused(t *testing.T) {
	ctx := context.Background()
	m := ctxmmu.New()

	unknown := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	if _, ok := m.ResolvePagedRef(ctx, unknown); ok {
		t.Fatalf("expected ResolvePagedRef to return false for unknown digest")
	}
	if _, ok := ctxmmu.ResolvePaged(ctx, unknown); ok {
		t.Fatalf("expected ResolvePaged to return false for unknown digest")
	}
	if _, ok := ctxmmu.GetStagedPagedRef(unknown); ok {
		t.Fatalf("expected GetStagedPagedRef to return false for unknown digest")
	}
}

// TestPagedStore_TamperedRefRefused proves that staged entries with corrupted/tampered bytes
// that no longer hash to their digest address fail closed and return false.
func TestPagedStore_TamperedRefRefused(t *testing.T) {
	store := ctxmmu.NewPagedStore(1024*1024, 100)

	validData := []byte("original uncorrupted payload content")
	sum := sha256.Sum256(validData)
	digest := hex.EncodeToString(sum[:])

	// Stage with correct content
	store.Stage(digest, validData)

	// Verify intact
	got, ok := store.Get(digest)
	if !ok || !bytes.Equal(got, validData) {
		t.Fatalf("initial Get failed")
	}

	// Manually stage mismatching content under same digest (tamper simulation)
	store.Stage(digest, []byte("tampered and modified content bytes"))

	// Get must detect hash mismatch and refuse
	corrupted, okCorrupt := store.Get(digest)
	if okCorrupt {
		t.Fatalf("expected Get to refuse tampered payload, got %q", string(corrupted))
	}
}

// TestPagedStore_EvictionAndCapacityBounds proves that the staging store is strictly bounded
// and evicts oldest unpinned entries on byte or entry capacity overflow.
func TestPagedStore_EvictionAndCapacityBounds(t *testing.T) {
	// Small budget: max 2000 bytes, max 3 entries
	store := ctxmmu.NewPagedStore(2000, 3)

	payload1 := bytes.Repeat([]byte("A"), 600)
	d1 := computeHexSHA256(payload1)

	payload2 := bytes.Repeat([]byte("B"), 600)
	d2 := computeHexSHA256(payload2)

	payload3 := bytes.Repeat([]byte("C"), 600)
	d3 := computeHexSHA256(payload3)

	store.Stage(d1, payload1)
	store.Stage(d2, payload2)
	store.Stage(d3, payload3)

	if store.Len() != 3 {
		t.Fatalf("store.Len = %d, want 3", store.Len())
	}

	// Access d1 so d2 becomes the oldest entry
	if _, ok := store.Get(d1); !ok {
		t.Fatalf("Get d1 failed")
	}

	// Add payload4 (600 bytes) -> total would be 2400 > 2000 and 4 entries > 3 entries
	payload4 := bytes.Repeat([]byte("D"), 600)
	d4 := computeHexSHA256(payload4)
	store.Stage(d4, payload4)

	// Oldest unpinned (d2) must be evicted!
	if _, ok := store.Get(d2); ok {
		t.Fatalf("expected d2 to be evicted on overflow")
	}
	// d1, d3, d4 must remain
	if _, ok := store.Get(d1); !ok {
		t.Fatalf("d1 should remain")
	}
	if _, ok := store.Get(d4); !ok {
		t.Fatalf("d4 should remain")
	}

	_, _, _, evicted := store.Stats()
	if evicted < 1 {
		t.Fatalf("expected at least 1 eviction in stats, got %d", evicted)
	}
}

// TestPagedStore_PinPreventsEviction proves that pinned entries are protected from LRU eviction.
func TestPagedStore_PinPreventsEviction(t *testing.T) {
	// Max 2 entries
	store := ctxmmu.NewPagedStore(10000, 2)

	p1 := []byte("payload one")
	d1 := computeHexSHA256(p1)
	p2 := []byte("payload two")
	d2 := computeHexSHA256(p2)
	p3 := []byte("payload three")
	d3 := computeHexSHA256(p3)

	store.Stage(d1, p1)
	store.Pin(d1) // Pin d1

	store.Stage(d2, p2)
	// Now add d3. Without pin, d1 would be evicted. With pin on d1, d2 must be evicted instead!
	store.Stage(d3, p3)

	if _, ok := store.Get(d1); !ok {
		t.Fatalf("pinned entry d1 was evicted")
	}
	if _, ok := store.Get(d2); ok {
		t.Fatalf("unpinned entry d2 should have been evicted instead of pinned d1")
	}
	if _, ok := store.Get(d3); !ok {
		t.Fatalf("new entry d3 should be present")
	}

	// Unpin d1 and add d4 -> now d1 can be evicted
	store.Unpin(d1)
	p4 := []byte("payload four")
	d4 := computeHexSHA256(p4)
	store.Stage(d4, p4)

	if _, ok := store.Get(d1); ok {
		t.Fatalf("unpinned entry d1 should now be evictable")
	}
}

// TestPagedStore_QuarantineUnclearedRefused proves that a quarantined result with pending writeback
// still enforces the witness clear gate before allowing PageIn.
func TestPagedStore_QuarantineUnclearedRefused(t *testing.T) {
	ctx := context.Background()
	const codecID = "async_quarantine_sim"
	backend := newMockAsyncDiskBackend()
	abi.RegisterPageOutBackend(codecID, backend)
	t.Setenv("FAK_PAGEOUT_BACKEND", codecID)

	m := ctxmmu.New()

	secretData := append([]byte("sk-"), bytes.Repeat([]byte("x"), 300)...)
	c := call("read_credentials")
	r := result(c, secretData)

	v := m.Admit(ctx, c, r)
	if v.Kind != abi.VerdictQuarantine {
		t.Fatalf("expected VerdictQuarantine, got %v", v.Kind)
	}
	id := r.Meta["quarantine_id"]
	if id == "" {
		t.Fatal("missing quarantine_id")
	}

	// PageIn without Clear must be refused (trust gate invariant)
	if _, err := m.PageIn(ctx, id); err == nil {
		t.Fatal("expected PageIn before Clear to be refused")
	}

	// Clear the witness gate
	m.Clear(id)

	// PageIn after Clear must succeed immediately from staging even before disk writeback
	restored, err := m.PageIn(ctx, id)
	if err != nil {
		t.Fatalf("PageIn after Clear failed: %v", err)
	}
	if !bytes.Equal(restored, secretData) {
		t.Fatalf("restored quarantine bytes mismatch")
	}
}

// TestPagedStore_ConcurrentAccess verifies thread-safety under concurrent stage/get/evict.
func TestPagedStore_ConcurrentAccess(t *testing.T) {
	store := ctxmmu.NewPagedStore(64*1024, 20)
	var wg sync.WaitGroup

	const numWorkers = 16
	const numOps = 100

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for i := 0; i < numOps; i++ {
				data := []byte(fmt.Sprintf("worker-%d-iteration-%d-data-payload", workerID, i))
				d := computeHexSHA256(data)

				store.Stage(d, data)
				if i%3 == 0 {
					store.Pin(d)
				}
				if got, ok := store.Get(d); ok && !bytes.Equal(got, data) {
					t.Errorf("data mismatch in worker %d", workerID)
				}
				if i%3 == 0 {
					store.Unpin(d)
				}
			}
		}(w)
	}

	wg.Wait()
}
