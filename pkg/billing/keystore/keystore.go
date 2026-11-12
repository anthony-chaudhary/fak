package keystore

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// KeystoreVersionV1 is the store-file format version this package writes
// and accepts. A file declaring any other version is refused as
// ErrMalformedPayload.
const KeystoreVersionV1 = "fak-billing-keystore/v1"

// Closed key-material constants. The prefix marks fak-minted credentials on
// the wire; the random part is 32 bytes of crypto/rand rendered as lowercase
// hex, so a full key is keyFormatPrefix followed by 64 hex characters.
const (
	keyFormatPrefix  = "sk-fak-"
	keyRandomBytes   = 32
	displayPrefixLen = 12
	sha256HexLen     = 64
	storeFilePerm    = 0o600
)

// Sentinel errors. Argument-shape faults wrap ErrInvalidArgument; store-file
// faults wrap ErrCorruptStore; the rest name one specific fault. Callers
// match with errors.Is against the specific sentinel (or a root).
var (
	// ErrInvalidArgument is the root sentinel for rejected call arguments.
	ErrInvalidArgument = errors.New("keystore: invalid argument")

	// ErrEmptyTenantID marks a Mint without a tenant.
	ErrEmptyTenantID = fmt.Errorf("%w: tenant id is required", ErrInvalidArgument)

	// ErrEmptyPath marks an Open without a store path.
	ErrEmptyPath = fmt.Errorf("%w: store path is required", ErrInvalidArgument)

	// ErrEmptyKey marks a Validate or Revoke without a key.
	ErrEmptyKey = fmt.Errorf("%w: key is required", ErrInvalidArgument)

	// ErrUnknownKey is returned when a key or prefix matches no stored key.
	// Callers map it to 401.
	ErrUnknownKey = errors.New("keystore: unknown key")

	// ErrKeyRevoked is returned when the matched key is already revoked.
	// Callers map it to 403.
	ErrKeyRevoked = errors.New("keystore: key revoked")

	// ErrAmbiguousKey is returned when a Revoke by display prefix matches
	// more than one stored key: the prefix is short by design, so two keys
	// can share it. Revoke by full key is never ambiguous.
	ErrAmbiguousKey = errors.New("keystore: prefix matches multiple keys")

	// ErrCorruptStore is the root sentinel for a store file that is not a
	// well-formed store of this format.
	ErrCorruptStore = errors.New("keystore: corrupt store")

	// ErrMalformedPayload marks a payload-shape fault: undecodable bytes, a
	// field outside the v1 schema, trailing data, or a version stamp this
	// package does not write.
	ErrMalformedPayload = fmt.Errorf("%w: payload outside the v1 schema", ErrCorruptStore)

	// ErrInvalidRecord marks a record that parses but violates the closed
	// record contract: bad prefix shape, bad hash shape, empty tenant, zero
	// creation time, or a duplicated hash.
	ErrInvalidRecord = fmt.Errorf("%w: record violates the closed record contract", ErrCorruptStore)
)

// KeyRecord is one stored key: the at-rest form of a minted credential.
//
// It holds NO plaintext: Prefix is a short display fragment and Hash is the
// SHA-256 digest of the full key, so the plaintext is unrecoverable from a
// record. RevokedAt being nil means the key is active.
type KeyRecord struct {
	// Prefix is the first displayPrefixLen characters of the full key,
	// for operator display only. It is not secret and is not unique: two
	// keys may share a prefix.
	Prefix string `json:"prefix"`

	// Hash is the lowercase-hex SHA-256 digest of the full key. It is the
	// lookup key for Validate and the durable identity of the record.
	Hash string `json:"hash"`

	// TenantID is the opaque registry identifier of the owning tenant.
	// Carried verbatim; this package neither mints nor interprets it.
	TenantID string `json:"tenant_id"`

	// CreatedAt is when the key was minted, RFC3339 UTC.
	CreatedAt time.Time `json:"created_at"`

	// RevokedAt is when the key was revoked; nil means the key is active.
	RevokedAt *time.Time `json:"revoked_at,omitempty"`
}

// Validate checks the closed record contract: a well-formed display prefix
// (the key marker plus lowercase hex, as Mint produces it), a well-formed
// SHA-256 digest, a non-empty tenant, and a non-zero creation time. On the
// first violated rule it returns a wrapped ErrInvalidRecord.
func (r KeyRecord) Validate() error {
	tail := ""
	if len(r.Prefix) >= len(keyFormatPrefix) {
		tail = r.Prefix[len(keyFormatPrefix):]
	}
	if len(r.Prefix) != displayPrefixLen || !strings.HasPrefix(r.Prefix, keyFormatPrefix) || !isLowerHex(tail) {
		return fmt.Errorf("%w: prefix %q must be %d characters: %q followed by lowercase hex",
			ErrInvalidRecord, r.Prefix, displayPrefixLen, keyFormatPrefix)
	}
	if len(r.Hash) != sha256HexLen || !isLowerHex(r.Hash) {
		return fmt.Errorf("%w: hash %q must be %d lowercase hex characters",
			ErrInvalidRecord, r.Hash, sha256HexLen)
	}
	if r.TenantID == "" {
		return fmt.Errorf("%w: tenant_id is required", ErrInvalidRecord)
	}
	if r.CreatedAt.IsZero() {
		return fmt.Errorf("%w: created_at is required", ErrInvalidRecord)
	}
	return nil
}

// Store is a durable tenant-key registry.
//
// All exported methods are safe for concurrent use. A nil *Store fails
// closed with ErrInvalidArgument (or an empty slice) rather than panicking.
type Store struct {
	mu sync.RWMutex

	path   string
	keys   []KeyRecord // insertion (mint) order
	byHash map[string]int
}

// Mint generates a new high-entropy key bound to tenantID and stores it.
//
// The returned string is the FULL plaintext key: sk-fak- followed by 32
// bytes of crypto/rand as lowercase hex. It is delivered exactly once, at
// mint time; the store persists only its SHA-256 digest and a short display
// prefix, so the plaintext is unrecoverable afterwards. The record is a
// snapshot: mutating it does not affect the store.
//
// Two keys may share a display prefix; Revoke disambiguates by full key or
// refuses with ErrAmbiguousKey.
func (s *Store) Mint(tenantID string) (string, *KeyRecord, error) {
	if s == nil {
		return "", nil, ErrInvalidArgument
	}
	if strings.TrimSpace(tenantID) == "" {
		return "", nil, ErrEmptyTenantID
	}
	raw := make([]byte, keyRandomBytes)
	if _, err := rand.Read(raw); err != nil {
		return "", nil, fmt.Errorf("keystore: generate key material: %w", err)
	}
	key := keyFormatPrefix + hex.EncodeToString(raw)

	rec := KeyRecord{
		Prefix:    key[:displayPrefixLen],
		Hash:      sha256Hex(key),
		TenantID:  tenantID,
		CreatedAt: nowUTC(),
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	next := make([]KeyRecord, len(s.keys)+1)
	copy(next, s.keys)
	next[len(next)-1] = rec
	if err := s.persistLocked(next); err != nil {
		return "", nil, err
	}
	s.keys = next
	s.byHash[rec.Hash] = len(s.keys) - 1
	cp := rec
	return key, &cp, nil
}

// Validate reports whether key is a stored, active key, returning its
// record. An unknown key is ErrUnknownKey; a stored-but-revoked key is
// ErrKeyRevoked. Callers map those to 401/403.
//
// Timing: the lookup is a map probe keyed by the SHA-256 digest of the
// supplied key. The probe input is a 256-bit digest of caller-controlled
// data, so anything timing-distinguishable in the probe correlates with
// that digest, not with any stored secret: learning "this digest is
// present" is exactly the verdict Validate returns anyway, and turning a
// present digest into a usable key requires a SHA-256 preimage. No
// plaintext key material is ever compared, hashed, or touched on this
// path, so a plain map lookup over digests is acceptable here.
func (s *Store) Validate(key string) (*KeyRecord, error) {
	if s == nil {
		return nil, ErrInvalidArgument
	}
	if strings.TrimSpace(key) == "" {
		return nil, ErrEmptyKey
	}
	s.mu.RLock()
	defer s.mu.RUnlock()

	i, ok := s.byHash[sha256Hex(key)]
	if !ok {
		return nil, ErrUnknownKey
	}
	r := s.keys[i]
	if r.RevokedAt != nil {
		return nil, ErrKeyRevoked
	}
	return &r, nil
}

// Revoke revokes the stored key identified by prefixOrKey — either a full
// key or a display prefix — durably marking it revoked.
//
// A full-key match (by SHA-256 digest) takes precedence over a prefix
// match, so a full key is never ambiguous even when another key shares its
// display prefix. A prefix matching several keys is ErrAmbiguousKey; no
// match is ErrUnknownKey; revoking an already-revoked key is
// ErrKeyRevoked.
func (s *Store) Revoke(prefixOrKey string) error {
	if s == nil {
		return ErrInvalidArgument
	}
	if strings.TrimSpace(prefixOrKey) == "" {
		return ErrEmptyKey
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	idx, err := s.locateLocked(prefixOrKey)
	if err != nil {
		return err
	}
	if s.keys[idx].RevokedAt != nil {
		return ErrKeyRevoked
	}
	now := nowUTC()
	next := make([]KeyRecord, len(s.keys))
	copy(next, s.keys)
	next[idx].RevokedAt = &now
	if err := s.persistLocked(next); err != nil {
		return err
	}
	s.keys = next
	return nil
}

// List returns a copy of every stored record, active and revoked, in stable
// order: CreatedAt, then Prefix.
func (s *Store) List() []KeyRecord {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := make([]KeyRecord, len(s.keys))
	copy(out, s.keys)
	sortByKeyOrder(out)
	return out
}

// TenantKeys returns a copy of the records bound to tenantID, in List
// order. An empty or unknown tenant yields an empty slice.
func (s *Store) TenantKeys(tenantID string) []KeyRecord {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := make([]KeyRecord, 0, len(s.keys))
	for _, r := range s.keys {
		if r.TenantID == tenantID {
			out = append(out, r)
		}
	}
	sortByKeyOrder(out)
	return out
}

// locateLocked resolves prefixOrKey to a record index under the rules of
// Revoke. Callers hold mu.
func (s *Store) locateLocked(prefixOrKey string) (int, error) {
	if i, ok := s.byHash[sha256Hex(prefixOrKey)]; ok {
		return i, nil
	}
	found := -1
	for i, r := range s.keys {
		if r.Prefix == prefixOrKey {
			if found != -1 {
				return -1, ErrAmbiguousKey
			}
			found = i
		}
	}
	if found == -1 {
		return -1, ErrUnknownKey
	}
	return found, nil
}

// sortByKeyOrder orders records by CreatedAt, then Prefix. Records that
// agree on both keep their relative order under sort.SliceStable, so the
// result is deterministic for a given store.
func sortByKeyOrder(keys []KeyRecord) {
	sort.SliceStable(keys, func(i, j int) bool {
		if !keys[i].CreatedAt.Equal(keys[j].CreatedAt) {
			return keys[i].CreatedAt.Before(keys[j].CreatedAt)
		}
		return keys[i].Prefix < keys[j].Prefix
	})
}

// nowUTC returns the current instant as a UTC wall-clock time with NO
// monotonic clock reading, so an in-memory timestamp equals its decoded
// form exactly (== and .Equal agree) and encodings stay byte-stable.
func nowUTC() time.Time {
	return time.Unix(0, time.Now().UnixNano()).UTC()
}

// sha256Hex renders the SHA-256 digest of s as lowercase hex.
func sha256Hex(s string) string {
	d := sha256.Sum256([]byte(s))
	return hex.EncodeToString(d[:])
}

// isLowerHex reports whether s is entirely lowercase hexadecimal.
func isLowerHex(s string) bool {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}
