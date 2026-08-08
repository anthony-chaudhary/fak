package blobfs

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/blob"
)

func payload(n int, fill byte) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = fill
	}
	return b
}

// TestPutResolveRoundTrip is the core durable-CAS contract: a stored payload
// resolves byte-identically through the same store.
func TestPutResolveRoundTrip(t *testing.T) {
	ctx := context.Background()
	s, err := New(t.TempDir())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	want := payload(4096, 'x') // > InlineMax, lands on disk
	r, err := s.Put(ctx, want)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	if r.Kind != abi.RefBlob {
		t.Fatalf("want RefBlob for a large payload, got kind %d", r.Kind)
	}
	if r.Digest != blob.Digest(want) {
		t.Fatalf("Ref digest %q != content digest %q", r.Digest, blob.Digest(want))
	}
	got, err := s.Resolve(ctx, r)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("resolved bytes differ from stored")
	}
}

// TestInlineSmallPayloadNeverTouchesDisk proves a <=InlineMax payload is carried
// inline on the Ref (durable in the Ref itself) and writes no file.
func TestInlineSmallPayloadNeverTouchesDisk(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s, err := New(dir)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	small := payload(InlineMax, 's')
	r, err := s.Put(ctx, small)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	if r.Kind != abi.RefInline {
		t.Fatalf("want RefInline for a <=InlineMax payload, got kind %d", r.Kind)
	}
	if !bytes.Equal(r.Inline, small) {
		t.Fatalf("inline Ref does not carry the bytes")
	}
	if count, _, _ := s.Resident(); count != 0 {
		t.Fatalf("inline payload should not be resident on disk, got %d blobs", count)
	}
}

// TestContentDedup proves a byte-identical payload is stored exactly once.
func TestContentDedup(t *testing.T) {
	ctx := context.Background()
	s, err := New(t.TempDir())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	b := payload(1024, 'd')
	if _, err := s.Put(ctx, b); err != nil {
		t.Fatalf("Put 1: %v", err)
	}
	if _, err := s.Put(ctx, b); err != nil {
		t.Fatalf("Put 2: %v", err)
	}
	count, _, _ := s.Resident()
	if count != 1 {
		t.Fatalf("content dedup failed: %d resident blobs, want 1", count)
	}
	_, hits, _ := s.Stats()
	if hits != 1 {
		t.Fatalf("want 1 dedup hit, got %d", hits)
	}
	stats := s.ByteStats()
	if stats.NovelBytes != int64(len(b)) {
		t.Fatalf("want %d novel bytes, got %d", len(b), stats.NovelBytes)
	}
	if stats.DedupedBytesAvoided != int64(len(b)) {
		t.Fatalf("want %d deduped bytes avoided, got %d", len(b), stats.DedupedBytesAvoided)
	}
}

// TestRestartSurvival is the reason blobfs exists: a digest written by one store
// resolves through a FRESH store opened on the same directory — the property the
// in-memory blob CAS does not have.
func TestRestartSurvival(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()

	s1, err := New(dir)
	if err != nil {
		t.Fatalf("New 1: %v", err)
	}
	want := payload(8192, 'p')
	r, err := s1.Put(ctx, want)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}

	// Simulate a process restart: a brand-new Store over the same directory.
	s2, err := New(dir)
	if err != nil {
		t.Fatalf("New 2 (restart): %v", err)
	}
	if count, _, _ := s2.Resident(); count != 1 {
		t.Fatalf("restart scan did not recover the blob: %d resident, want 1", count)
	}
	got, err := s2.Resolve(ctx, r)
	if err != nil {
		t.Fatalf("Resolve after restart: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("bytes did not survive the restart")
	}
}

// TestScanIgnoresTempLeftovers proves a crashed Put's temp file is never counted
// as (or resolved as) a committed blob on reopen.
func TestScanIgnoresTempLeftovers(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s, err := New(dir)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := s.Put(ctx, payload(2048, 'r')); err != nil {
		t.Fatalf("Put: %v", err)
	}
	// Drop a temp leftover inside a shard dir, as a torn write would.
	shard := filepath.Join(dir, "ab", "cd")
	if err := os.MkdirAll(shard, 0o755); err != nil {
		t.Fatalf("mkdir shard: %v", err)
	}
	if err := os.WriteFile(filepath.Join(shard, tmpPrefix+"crash"), []byte("garbage"), 0o644); err != nil {
		t.Fatalf("write temp: %v", err)
	}
	s2, err := New(dir)
	if err != nil {
		t.Fatalf("New (reopen): %v", err)
	}
	if count, _, _ := s2.Resident(); count != 1 {
		t.Fatalf("temp leftover polluted the index: %d resident, want 1", count)
	}
}

// TestPageOutPageIn proves the durable page-out codec contract: a handle carries no
// inline bytes, and page-in re-materializes them.
func TestPageOutPageIn(t *testing.T) {
	ctx := context.Background()
	s, err := New(t.TempDir())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	body := payload(5000, 'q')
	inline := abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body))}
	handle, err := s.PageOut(ctx, inline)
	if err != nil {
		t.Fatalf("PageOut: %v", err)
	}
	if handle.Kind != abi.RefBlob || len(handle.Inline) != 0 {
		t.Fatalf("page-out handle must be a bytes-absent RefBlob, got kind %d inline=%d", handle.Kind, len(handle.Inline))
	}
	back, err := s.PageIn(ctx, handle)
	if err != nil {
		t.Fatalf("PageIn: %v", err)
	}
	if !bytes.Equal(back.Inline, body) {
		t.Fatalf("page-in bytes differ from paged-out body")
	}
}

// TestPageOutSurvivesRestart proves a quarantined/cold result paged out through
// blobfs is recoverable in a fresh process — the gap the in-memory codec leaves.
func TestPageOutSurvivesRestart(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s1, err := New(dir)
	if err != nil {
		t.Fatalf("New 1: %v", err)
	}
	body := payload(3000, 'z')
	handle, err := s1.PageOut(ctx, abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body))})
	if err != nil {
		t.Fatalf("PageOut: %v", err)
	}
	s2, err := New(dir)
	if err != nil {
		t.Fatalf("New 2: %v", err)
	}
	back, err := s2.PageIn(ctx, handle)
	if err != nil {
		t.Fatalf("PageIn after restart: %v", err)
	}
	if !bytes.Equal(back.Inline, body) {
		t.Fatalf("paged-out bytes did not survive restart")
	}
}

// TestPinProtectsFromGC proves the byte-budget GC never deletes a pinned digest,
// even when it is the oldest entry — the vDSO/MMU soundness invariant on disk.
func TestPinProtectsFromGC(t *testing.T) {
	ctx := context.Background()
	// Budget holds ~2 of the 1KiB-ish blobs; pin the FIRST so it is the oldest.
	s, err := NewWithBudget(t.TempDir(), 3000)
	if err != nil {
		t.Fatalf("NewWithBudget: %v", err)
	}
	first, err := s.Put(ctx, payload(1024, 0))
	if err != nil {
		t.Fatalf("Put first: %v", err)
	}
	s.Pin(first.Digest)
	for i := 1; i < 10; i++ {
		if _, err := s.Put(ctx, payload(1024, byte(i))); err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
	}
	if _, _, evicted := s.Resident(); evicted == 0 {
		t.Fatalf("expected GC to evict under a tight budget, evicted=0")
	}
	if _, err := s.Resolve(ctx, first); err != nil {
		t.Fatalf("pinned (oldest) digest was evicted by GC: %v", err)
	}
}

// TestEnvBudget proves FAK_BLOB_DIR_MAX_BYTES bounds the resident footprint.
func TestEnvBudget(t *testing.T) {
	t.Setenv("FAK_BLOB_DIR_MAX_BYTES", "4096")
	ctx := context.Background()
	s, err := New(t.TempDir()) // reads the env budget at construction
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for i := 0; i < 50; i++ {
		if _, err := s.Put(ctx, payload(1024, byte(i))); err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
	}
	if _, b, _ := s.Resident(); b > 4096 {
		t.Fatalf("env budget not honored: resident=%d bytes, want <= 4096", b)
	}
}

// TestFileNameIsDigest proves the on-disk file name is the content address — the
// property that makes a digest written by one process resolvable in another.
func TestFileNameIsDigest(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s, err := New(dir)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	b := payload(1500, 'n')
	r, err := s.Put(ctx, b)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	want := filepath.Join(dir, r.Digest[0:2], r.Digest[2:4], r.Digest)
	if _, err := os.Stat(want); err != nil {
		t.Fatalf("expected blob at sharded digest path %s: %v", want, err)
	}
}

// TestResolverInterface pins blobfs.Store as a usable abi.Resolver + CASPinner so a
// router can compose it as a tier.
func TestResolverInterface(t *testing.T) {
	var _ abi.Resolver = (*Store)(nil)
	var _ abi.CASPinner = (*Store)(nil)
	var _ abi.PageOutBackend = pageOutBackend{}
}

// TestResolveRefusesCorruptedOnDiskBlob is the fail-closed content-addressing guard:
// blobfs is a SHARED, durable directory ("shareable across processes on the same
// host"), so a committed file is exposed to bit-rot and to a foreign process writing
// wrong content under a valid digest name. Content addressing is only sound if a read
// verifies the address — bytes that no longer hash to the requested digest must be
// REFUSED, never returned, the same guard blobhttp enforces for its untrusted remote.
// The in-memory blob store gets this for free (its bytes are keyed by their own digest
// in RAM); blobfs reads a file BY NAME, so the name/content bond must be re-checked.
// Pre-fix Resolve handed back the corrupt bytes; post-fix it errors "digest mismatch".
func TestResolveRefusesCorruptedOnDiskBlob(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s, err := New(dir)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	want := payload(4096, 'x') // > InlineMax → lands on disk under its content digest
	r, err := s.Put(ctx, want)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	if r.Kind != abi.RefBlob {
		t.Fatalf("precondition: want RefBlob so Resolve reads from disk, got kind %d", r.Kind)
	}

	// Corrupt the committed file in place: SAME digest file name, DIFFERENT bytes —
	// exactly what silent bit-rot or a foreign write under a valid digest looks like.
	corrupt := payload(4096, 'z')
	if blob.Digest(corrupt) == r.Digest {
		t.Fatalf("test setup: corrupt payload must not hash to the original digest")
	}
	onDisk := filepath.Join(dir, r.Digest[0:2], r.Digest[2:4], r.Digest)
	if err := os.WriteFile(onDisk, corrupt, 0o644); err != nil {
		t.Fatalf("corrupt on-disk blob: %v", err)
	}

	// The read must fail-closed: refuse the mismatched bytes rather than leak corruption
	// back into the caller as if it were the addressed content.
	got, err := s.Resolve(ctx, r)
	if err == nil {
		t.Fatalf("Resolve returned corrupt bytes (len=%d) for digest %s; want a digest-mismatch refusal", len(got), r.Digest)
	}
	if !strings.Contains(err.Error(), "digest mismatch") {
		t.Fatalf("Resolve error = %q, want it to mention 'digest mismatch'", err)
	}
}
