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
	"sync"
)

// Store is the durable residency tier a demoted KV span lands in. It is keyed by
// the span's residency DIGEST — the identity the KV-MMU stages and later restores a
// span by — NOT by a hash of the payload. That distinction is load-bearing: a
// content-addressed CAS (internal/blobfs) keys bytes by SHA256(bytes), but a
// RestoreSpan arrives with the span digest and no payload, so the residency store
// must look up by that digest. Hence a span-keyed durable K/V rather than a bare
// blobfs Resolve. The interface is small on purpose so implementations slot in
// behind it without touching the backend: routerStore (routerstore.go) is the
// registered default — the storedrv router with a durable span→content manifest —
// and diskStore below is the self-contained flat-file stand-in the witness suite
// also exercises.
type Store interface {
	// Put durably records payload under key (the span residency digest). It is
	// atomic and crash-safe: after Put returns nil, a Get(key) — in this process or
	// a later one after a restart — yields the same payload or a typed fault, never
	// a torn prefix.
	Put(ctx context.Context, key string, payload []byte) error
	// Get returns the payload staged under key. found=false is a clean MISS (the
	// span was never staged, or the tier reaped it) — the caller recomputes but is
	// TOLD. A non-nil error is a FAULT: an I/O failure, or an integrity failure
	// (the bytes no longer hash to what was written — corrupt or tampered), which is
	// refused rather than returned.
	Get(ctx context.Context, key string) (payload []byte, found bool, err error)
}

// Durable record envelope (v1). A self-describing header lets a reader detect the
// on-disk format instead of inferring it from length, so a future format change is
// a refused unknown-version FAULT rather than a silently mis-parsed payload (#3395).
// Layout:
//
//	magic(4) || version(uint16 BE) || SHA256(payload)(32) || payload
//
// The sha256 keeps its integrity role (the CRC slot the sessionimage bundle already
// has); the magic+version add the forward-compat discriminant the raw KV record
// lacked. Get refuses a record whose magic or version it does not recognize with a
// typed fault (errBadMagic / errUnsupportedVersion), fail-closed — never a wrong hit
// nor a torn read masquerading as a clean miss.
const (
	recordMagic   = "L3KV"    // 4-byte readable magic prefix identifying an l3kv record
	recordVersion = uint16(1) // on-disk record format version
)

// recordHeaderLen is the fixed prefix ahead of the payload: magic + uint16 version +
// the sha256 integrity sum. len of a string constant and sha256.Size are both
// constants, so this folds at compile time.
const recordHeaderLen = len(recordMagic) + 2 + sha256.Size

var (
	// errBadMagic is the fault for bytes that do not begin with recordMagic — an old
	// headerless record, a foreign file dropped under a valid-looking key, or bit-rot
	// in the prefix. Refused, never parsed.
	errBadMagic = errors.New("unrecognized record magic — not an l3kv record (corrupt or foreign format)")
	// errUnsupportedVersion is the fault for a well-formed l3kv record whose version
	// this build does not read — the forward-compat gate itself.
	errUnsupportedVersion = errors.New("unsupported record version")
)

// diskStore is a crash-safe, restart-surviving durable K/V rooted at a directory,
// keyed by span digest — the self-contained flat-file Store (no router, no
// manifest; one record file per span). The registered backend now composes
// routerStore instead, but this implementation stays as the zero-composition
// stand-in and the on-disk witness for the versioned record envelope above: a
// read re-verifies the payload against its own hash and refuses corrupt/tampered
// or unknown-format bytes (the "verify, don't trust" admission guard, fail-closed).
// Writes commit via a temp file + fsync + atomic rename, the same durability point
// internal/blobfs uses — an interrupted write leaves only a temp file, never a half
// record under a valid key.
type diskStore struct{ dir string }

func newDiskStore(dir string) (*diskStore, error) {
	if dir == "" {
		return nil, fmt.Errorf("l3kv: empty store directory")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("l3kv: create store dir %s: %w", dir, err)
	}
	return &diskStore{dir: dir}, nil
}

// validKey guards the on-disk filename: a span digest is a lowercase/uppercase hex
// SHA256 string, so anything else — a path separator, a "..", an empty key — is
// rejected before it can escape the store directory.
func validKey(key string) bool {
	if key == "" || len(key) > 128 {
		return false
	}
	for i := 0; i < len(key); i++ {
		c := key[i]
		isHex := (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')
		if !isHex {
			return false
		}
	}
	return true
}

func (s *diskStore) path(key string) string { return filepath.Join(s.dir, key) }

func (s *diskStore) Put(ctx context.Context, key string, payload []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if !validKey(key) {
		return fmt.Errorf("l3kv: invalid span key %q", key)
	}
	sum := sha256.Sum256(payload)
	buf := make([]byte, 0, recordHeaderLen+len(payload))
	buf = append(buf, recordMagic...)
	var ver [2]byte
	binary.BigEndian.PutUint16(ver[:], recordVersion)
	buf = append(buf, ver[:]...)
	buf = append(buf, sum[:]...)
	buf = append(buf, payload...)
	return atomicWrite(s.path(key), buf)
}

func (s *diskStore) Get(ctx context.Context, key string) ([]byte, bool, error) {
	if err := ctx.Err(); err != nil {
		return nil, false, err
	}
	if !validKey(key) {
		return nil, false, fmt.Errorf("l3kv: invalid span key %q", key)
	}
	b, err := os.ReadFile(s.path(key))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil // clean MISS: never staged / reaped
		}
		return nil, false, err // FAULT: I/O error
	}
	if len(b) < recordHeaderLen {
		return nil, false, fmt.Errorf("l3kv: truncated record for %s (%d bytes < %d-byte header)", key, len(b), recordHeaderLen)
	}
	if string(b[:len(recordMagic)]) != recordMagic {
		return nil, false, fmt.Errorf("l3kv: %w for %s", errBadMagic, key)
	}
	if ver := binary.BigEndian.Uint16(b[len(recordMagic) : len(recordMagic)+2]); ver != recordVersion {
		return nil, false, fmt.Errorf("l3kv: %w %d for %s (this build reads v%d)", errUnsupportedVersion, ver, key, recordVersion)
	}
	sum, payload := b[len(recordMagic)+2:recordHeaderLen], b[recordHeaderLen:]
	got := sha256.Sum256(payload)
	if !bytes.Equal(sum, got[:]) {
		return nil, false, fmt.Errorf("l3kv: integrity check failed for %s (corrupt or tampered)", key)
	}
	return payload, true, nil
}

// atomicWrite writes b to final via a temp file that is fsync'd and atomically
// renamed into place — the rename is the durability commit. Mirrors
// internal/blobfs writeBlob's crash-safe shape.
func atomicWrite(final string, b []byte) error {
	dir := filepath.Dir(final)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("l3kv: shard dir %s: %w", dir, err)
	}
	tmp, err := os.CreateTemp(dir, ".l3kv-*")
	if err != nil {
		return fmt.Errorf("l3kv: temp file in %s: %w", dir, err)
	}
	name := tmp.Name()
	if _, err := tmp.Write(b); err != nil {
		tmp.Close()
		os.Remove(name)
		return fmt.Errorf("l3kv: write %s: %w", name, err)
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		os.Remove(name)
		return fmt.Errorf("l3kv: fsync %s: %w", name, err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(name)
		return fmt.Errorf("l3kv: close %s: %w", name, err)
	}
	if err := os.Rename(name, final); err != nil {
		os.Remove(name)
		return fmt.Errorf("l3kv: commit %s: %w", final, err)
	}
	return nil
}

// memStore is a process-lifetime, in-memory Store — the S0 stand-in the seam study
// calls for. It is durable only within a process (no restart survival); the
// witness suite uses diskStore for the restart-survival and tamper cases.
type memStore struct {
	mu sync.Mutex
	m  map[string][]byte
}

func newMemStore() *memStore { return &memStore{m: map[string][]byte{}} }

func (s *memStore) Put(_ context.Context, key string, payload []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.m[key] = append([]byte(nil), payload...)
	return nil
}

func (s *memStore) Get(_ context.Context, key string) ([]byte, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	b, ok := s.m[key]
	if !ok {
		return nil, false, nil
	}
	return append([]byte(nil), b...), true, nil
}
