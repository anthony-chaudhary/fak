package l3kv

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// stagerKV is a test abi.KVBackend that ALSO implements SpanStager: it serves span
// bytes from an in-memory map keyed by [from,n]. Its own StageSpan/RestoreSpan are
// the no-op default shape (OK with no bytes / MISS) so a test can prove that
// wrapping it with l3kv is what adds real durable movement — the refute guard.
type stagerKV struct {
	spans            map[[2]int][]byte
	n                int
	restorePositions int
	restoreErr       error
	installed        []byte
}

func (m *stagerKV) Len() int                    { return m.n }
func (m *stagerKV) Prefill(ids []int) []float32 { return nil }
func (m *stagerKV) Evict(from, n int) int       { return n }
func (m *stagerKV) ModelID() string             { return "mock" }
func (m *stagerKV) StageSpan(context.Context, string, int, int) (abi.KVResidency, error) {
	return abi.KVResidency{Outcome: abi.KVResidencyOK}, nil // no-op default: OK, BytesMoved=0
}
func (m *stagerKV) RestoreSpan(_ context.Context, digest string) (abi.KVResidency, error) {
	return abi.KVResidency{Outcome: abi.KVResidencyMiss, Digest: digest}, nil
}
func (m *stagerKV) StageSpanBytes(from, n int) ([]byte, error) {
	b, ok := m.spans[[2]int{from, n}]
	if !ok {
		return nil, fmt.Errorf("no span [%d,%d)", from, n)
	}
	return b, nil
}
func (m *stagerKV) RestoreSpanBytes(payload []byte) (int, error) {
	if m.restoreErr != nil {
		return 0, m.restoreErr
	}
	m.installed = append([]byte(nil), payload...)
	if m.restorePositions > 0 {
		return m.restorePositions, nil
	}
	return 1, nil
}

// bareKV is a test abi.KVBackend that does NOT implement SpanStager — the honest-fence
// case: wrapping it must yield a typed FAULT on stage, never a silent OK-drop.
type bareKV struct{}

func (bareKV) Len() int                    { return 0 }
func (bareKV) Prefill(ids []int) []float32 { return nil }
func (bareKV) Evict(from, n int) int       { return n }
func (bareKV) ModelID() string             { return "bare" }
func (bareKV) StageSpan(_ context.Context, digest string, _, n int) (abi.KVResidency, error) {
	return abi.KVResidency{Outcome: abi.KVResidencyOK, Digest: digest, Positions: n}, nil
}
func (bareKV) RestoreSpan(_ context.Context, digest string) (abi.KVResidency, error) {
	return abi.KVResidency{Outcome: abi.KVResidencyMiss, Digest: digest}, nil
}

// digestA is a valid span-digest-shaped key (hex sha256 length).
const digestA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

func spanBytes(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i*7 + 3)
	}
	return b
}

// TestStageThenRestoreRoundTripsBitExact is the #1472 witness: StageSpan(digest)
// then RestoreSpan(digest) round-trips the span through a durable store, and the
// bytes the store returns are bit-exact to the bytes staged (max|delta|=0).
func TestStageThenRestoreRoundTripsBitExact(t *testing.T) {
	ctx := context.Background()
	payload := spanBytes(4096)
	mock := &stagerKV{spans: map[[2]int][]byte{{10, 20}: payload}, n: 100, restorePositions: 20}
	store := newMemStore()
	b := New(mock, store)

	res, err := b.StageSpan(ctx, digestA, 10, 20)
	if err != nil {
		t.Fatalf("StageSpan err: %v", err)
	}
	if res.Outcome != abi.KVResidencyOK {
		t.Fatalf("StageSpan outcome = %v, want OK (%s)", res.Outcome, res.Reason)
	}
	if res.BytesMoved != int64(len(payload)) {
		t.Fatalf("StageSpan BytesMoved = %d, want %d", res.BytesMoved, len(payload))
	}
	if res.Positions != 20 {
		t.Fatalf("StageSpan Positions = %d, want 20", res.Positions)
	}

	got, found, err := store.Get(ctx, digestA)
	if err != nil || !found {
		t.Fatalf("store.Get found=%v err=%v", found, err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatal("staged bytes are not bit-exact in the store (max|delta| != 0)")
	}

	rr, err := b.RestoreSpan(ctx, digestA)
	if err != nil {
		t.Fatalf("RestoreSpan err: %v", err)
	}
	if rr.Outcome != abi.KVResidencyOK {
		t.Fatalf("RestoreSpan outcome = %v, want OK (%s)", rr.Outcome, rr.Reason)
	}
	if rr.BytesMoved != int64(len(payload)) {
		t.Fatalf("RestoreSpan BytesMoved = %d, want %d", rr.BytesMoved, len(payload))
	}
	if rr.Positions != 20 || !bytes.Equal(mock.installed, payload) {
		t.Fatalf("RestoreSpan positions=%d installed=%d, want 20/%d", rr.Positions, len(mock.installed), len(payload))
	}
}

func TestRestoreRequiresSuccessfulLiveInstall(t *testing.T) {
	ctx := context.Background()
	payload := spanBytes(64)
	store := newMemStore()
	if err := store.Put(ctx, digestA, payload); err != nil {
		t.Fatal(err)
	}
	without := New(bareKV{}, store)
	if got, _ := without.RestoreSpan(ctx, digestA); got.Outcome != abi.KVResidencyFault {
		t.Fatalf("no installer outcome=%v, want FAULT", got.Outcome)
	}
	failing := New(&stagerKV{restoreErr: errors.New("invalid span image")}, store)
	if got, _ := failing.RestoreSpan(ctx, digestA); got.Outcome != abi.KVResidencyFault || got.BytesMoved != 0 {
		t.Fatalf("failed install=%+v, want zero-byte FAULT", got)
	}
}

// TestDiskStoreSurvivesReopen proves the durable tier survives a process restart:
// a span staged by one *diskStore is bit-exact readable by a second one opened over
// the same directory (the restart the in-process default cannot do).
func TestDiskStoreSurvivesReopen(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	payload := spanBytes(9000)

	st1, err := newDiskStore(dir)
	if err != nil {
		t.Fatalf("newDiskStore: %v", err)
	}
	if err := st1.Put(ctx, digestA, payload); err != nil {
		t.Fatalf("Put: %v", err)
	}

	st2, err := newDiskStore(dir) // simulate a restart: a fresh store over the same dir
	if err != nil {
		t.Fatalf("reopen newDiskStore: %v", err)
	}
	got, found, err := st2.Get(ctx, digestA)
	if err != nil || !found {
		t.Fatalf("after reopen: found=%v err=%v", found, err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatal("payload not bit-exact after reopen")
	}
}

// TestTamperedRecordIsFault proves the fail-closed integrity guard: a corrupted
// on-disk record is a typed FAULT at restore, never a silent wrong hit.
func TestTamperedRecordIsFault(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store, err := newDiskStore(dir)
	if err != nil {
		t.Fatalf("newDiskStore: %v", err)
	}
	payload := spanBytes(2048)
	if err := store.Put(ctx, digestA, payload); err != nil {
		t.Fatalf("Put: %v", err)
	}
	// Flip the last byte on disk — a payload byte, past the record header — so the
	// sha256 no longer matches its payload (an integrity fault, not a format fault).
	p := filepath.Join(dir, digestA)
	raw, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("read record: %v", err)
	}
	raw[len(raw)-1] ^= 0xFF
	if err := os.WriteFile(p, raw, 0o644); err != nil {
		t.Fatalf("rewrite record: %v", err)
	}

	if _, found, err := store.Get(ctx, digestA); err == nil {
		t.Fatalf("tampered store.Get returned no error (found=%v) — integrity guard missed", found)
	}

	mock := &stagerKV{spans: map[[2]int][]byte{}, n: 0}
	b := New(mock, store)
	rr, _ := b.RestoreSpan(ctx, digestA)
	if rr.Outcome != abi.KVResidencyFault {
		t.Fatalf("RestoreSpan over tampered record = %v, want FAULT", rr.Outcome)
	}
}

// TestUnknownDigestIsMiss proves a never-staged span is a typed MISS (recompute,
// but told), not a FAULT and not a silent hit.
func TestUnknownDigestIsMiss(t *testing.T) {
	ctx := context.Background()
	b := New(&stagerKV{spans: map[[2]int][]byte{}}, newMemStore())
	rr, err := b.RestoreSpan(ctx, digestA)
	if err != nil {
		t.Fatalf("RestoreSpan err: %v", err)
	}
	if rr.Outcome != abi.KVResidencyMiss {
		t.Fatalf("RestoreSpan for never-staged digest = %v, want MISS", rr.Outcome)
	}
}

// TestNoByteSourceIsHonestFault is the honest fence: wrapping a backend that does
// not expose a span byte-source yields a typed FAULT on stage — NOT the in-process
// default's silent OK-that-drops-the-span.
func TestNoByteSourceIsHonestFault(t *testing.T) {
	ctx := context.Background()
	b := New(bareKV{}, newMemStore())
	res, err := b.StageSpan(ctx, digestA, 0, 5)
	if err != nil {
		t.Fatalf("StageSpan err: %v", err)
	}
	if res.Outcome != abi.KVResidencyFault {
		t.Fatalf("StageSpan over a stager-less backend = %v, want FAULT", res.Outcome)
	}
	// And nothing was written: a restore is a clean MISS, not a phantom hit.
	if rr, _ := b.RestoreSpan(ctx, digestA); rr.Outcome != abi.KVResidencyMiss {
		t.Fatalf("RestoreSpan after a FAULTed stage = %v, want MISS", rr.Outcome)
	}
}

// TestRefuteWrapIsTheWire is the refute guard: the UNWRAPPED backend's StageSpan is
// the no-op default (OK, BytesMoved=0) and its RestoreSpan is a MISS — so the real
// durable movement (OK, BytesMoved>0, retrievable) is the l3kv WIRE, not a behavior
// the inner backend already had.
func TestRefuteWrapIsTheWire(t *testing.T) {
	ctx := context.Background()
	payload := spanBytes(1500)
	mock := &stagerKV{spans: map[[2]int][]byte{{0, 8}: payload}, n: 8}

	// Unwrapped: the default no-op shape.
	if raw, _ := mock.StageSpan(ctx, digestA, 0, 8); raw.BytesMoved != 0 {
		t.Fatalf("unwrapped StageSpan BytesMoved = %d, want 0 (no-op default)", raw.BytesMoved)
	}
	if raw, _ := mock.RestoreSpan(ctx, digestA); raw.Outcome != abi.KVResidencyMiss {
		t.Fatalf("unwrapped RestoreSpan = %v, want MISS", raw.Outcome)
	}

	// Wrapped: real bytes moved AND retrievable.
	b := New(mock, newMemStore())
	res, _ := b.StageSpan(ctx, digestA, 0, 8)
	if res.Outcome != abi.KVResidencyOK || res.BytesMoved != int64(len(payload)) {
		t.Fatalf("wrapped StageSpan = {%v, %d}, want {OK, %d}", res.Outcome, res.BytesMoved, len(payload))
	}
	if rr, _ := b.RestoreSpan(ctx, digestA); rr.Outcome != abi.KVResidencyOK {
		t.Fatalf("wrapped RestoreSpan = %v, want OK", rr.Outcome)
	}
}

// TestDelegatesLocalOps proves the wrapper is transparent for the local ops the
// KV-MMU drives (Len/Evict/ModelID), so wrapping does not disturb decode.
func TestDelegatesLocalOps(t *testing.T) {
	mock := &stagerKV{spans: map[[2]int][]byte{}, n: 42}
	b := New(mock, newMemStore())
	if b.Len() != 42 {
		t.Fatalf("Len = %d, want 42", b.Len())
	}
	if b.Evict(0, 5) != 5 {
		t.Fatalf("Evict = %d, want 5", b.Evict(0, 5))
	}
	if b.ModelID() != "mock" {
		t.Fatalf("ModelID = %q, want mock", b.ModelID())
	}
}

// TestFactoryFailsClosed proves the factory rejects a session value the in-process
// backend does not own (never enforce against a mis-constructed backend).
func TestFactoryFailsClosed(t *testing.T) {
	if _, ok := Factory(newMemStore())(123); ok {
		t.Fatal("Factory accepted a non-session value; want ok=false")
	}
}

// TestInvalidKeyRejected guards the on-disk filename against traversal.
func TestInvalidKeyRejected(t *testing.T) {
	ctx := context.Background()
	store, err := newDiskStore(t.TempDir())
	if err != nil {
		t.Fatalf("newDiskStore: %v", err)
	}
	for _, k := range []string{"", "../escape", "a/b", "not-hex!!"} {
		if err := store.Put(ctx, k, []byte("x")); err == nil {
			t.Fatalf("Put accepted invalid key %q", k)
		}
	}
}

// TestRecordCarriesVersionHeader proves the on-disk record is the self-describing
// envelope (#3395): the raw file opens with the readable magic and the format
// version, ahead of the sha256 the integrity guard already used.
func TestRecordCarriesVersionHeader(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store, err := newDiskStore(dir)
	if err != nil {
		t.Fatalf("newDiskStore: %v", err)
	}
	payload := spanBytes(777)
	if err := store.Put(ctx, digestA, payload); err != nil {
		t.Fatalf("Put: %v", err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, digestA))
	if err != nil {
		t.Fatalf("read record: %v", err)
	}
	if len(raw) != recordHeaderLen+len(payload) {
		t.Fatalf("record len = %d, want header(%d)+payload(%d)=%d", len(raw), recordHeaderLen, len(payload), recordHeaderLen+len(payload))
	}
	if string(raw[:len(recordMagic)]) != recordMagic {
		t.Fatalf("record magic = %q, want %q", raw[:len(recordMagic)], recordMagic)
	}
	if ver := binary.BigEndian.Uint16(raw[len(recordMagic) : len(recordMagic)+2]); ver != recordVersion {
		t.Fatalf("record version = %d, want %d", ver, recordVersion)
	}
	// And the envelope round-trips: Get strips the header and returns the exact bytes.
	got, found, err := store.Get(ctx, digestA)
	if err != nil || !found {
		t.Fatalf("Get found=%v err=%v", found, err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatal("payload not bit-exact through the versioned envelope")
	}
}

// TestUnknownMagicIsFault proves a record without the l3kv magic — an old headerless
// record, bit-rot, or a foreign file under a valid-looking key — is refused as a
// structured fault, not parsed. It surfaces as a RestoreSpan FAULT (fail-closed),
// never a wrong hit.
func TestUnknownMagicIsFault(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store, err := newDiskStore(dir)
	if err != nil {
		t.Fatalf("newDiskStore: %v", err)
	}
	// The pre-#3395 record shape (SHA256(payload) || payload, no magic) is exactly the
	// "unknown magic" case now — write one directly under a valid key.
	payload := spanBytes(64)
	sum := sha256.Sum256(payload)
	legacy := append(append([]byte(nil), sum[:]...), payload...)
	if err := os.WriteFile(filepath.Join(dir, digestA), legacy, 0o644); err != nil {
		t.Fatalf("write legacy record: %v", err)
	}
	_, found, err := store.Get(ctx, digestA)
	if err == nil {
		t.Fatalf("Get accepted a headerless record (found=%v) — magic gate missed", found)
	}
	if !errors.Is(err, errBadMagic) {
		t.Fatalf("Get err = %v, want errBadMagic", err)
	}
	rr, _ := New(&stagerKV{spans: map[[2]int][]byte{}}, store).RestoreSpan(ctx, digestA)
	if rr.Outcome != abi.KVResidencyFault {
		t.Fatalf("RestoreSpan over unknown-magic record = %v, want FAULT", rr.Outcome)
	}
}

// TestUnsupportedVersionIsFault proves the forward-compat gate: a well-formed l3kv
// record whose version this build does not know is refused as a structured fault —
// even when its sha256 matches its payload (so it is the VERSION rung that trips,
// not the integrity rung).
func TestUnsupportedVersionIsFault(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store, err := newDiskStore(dir)
	if err != nil {
		t.Fatalf("newDiskStore: %v", err)
	}
	payload := spanBytes(128)
	sum := sha256.Sum256(payload)
	var ver [2]byte
	binary.BigEndian.PutUint16(ver[:], recordVersion+42) // a future version this build cannot read
	rec := append([]byte(nil), recordMagic...)
	rec = append(rec, ver[:]...)
	rec = append(rec, sum[:]...) // valid integrity sum, so only the version is wrong
	rec = append(rec, payload...)
	if err := os.WriteFile(filepath.Join(dir, digestA), rec, 0o644); err != nil {
		t.Fatalf("write future-version record: %v", err)
	}
	_, found, err := store.Get(ctx, digestA)
	if err == nil {
		t.Fatalf("Get accepted a future-version record (found=%v) — version gate missed", found)
	}
	if !errors.Is(err, errUnsupportedVersion) {
		t.Fatalf("Get err = %v, want errUnsupportedVersion", err)
	}
}
