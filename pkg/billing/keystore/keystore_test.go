package keystore

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"sync"
	"testing"
)

// keyRe is the closed shape of a minted key: sk-fak- + 64 lowercase hex.
var keyRe = regexp.MustCompile(`^sk-fak-[0-9a-f]{64}$`)

// validKey renders seed as a syntactically valid full key, for fixtures.
func validKey(seed string) string {
	d := sha256.Sum256([]byte(seed))
	return keyFormatPrefix + hex.EncodeToString(d[:])
}

func openStore(t *testing.T) (*Store, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "store.json")
	s, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	return s, path
}

func TestMint(t *testing.T) {
	t.Parallel()
	s, path := openStore(t)

	for _, tenant := range []string{"", "   ", "\t"} {
		if _, _, err := s.Mint(tenant); !errors.Is(err, ErrEmptyTenantID) {
			t.Errorf("Mint(%q): error = %v, want ErrEmptyTenantID", tenant, err)
		}
	}

	key, rec, err := s.Mint("tenant-a")
	if err != nil {
		t.Fatalf("Mint: %v", err)
	}
	if !keyRe.MatchString(key) {
		t.Errorf("key %q does not match %s", key, keyRe)
	}
	if rec.Prefix != key[:displayPrefixLen] {
		t.Errorf("record prefix = %q, want %q", rec.Prefix, key[:displayPrefixLen])
	}
	if rec.Hash != sha256Hex(key) {
		t.Errorf("record hash = %q, want %q", rec.Hash, sha256Hex(key))
	}
	if rec.TenantID != "tenant-a" {
		t.Errorf("record tenant = %q, want tenant-a", rec.TenantID)
	}
	if rec.CreatedAt.IsZero() || rec.CreatedAt.Location().String() != "UTC" {
		t.Errorf("record created_at = %v, want non-zero UTC", rec.CreatedAt)
	}
	if rec.RevokedAt != nil {
		t.Errorf("fresh record revoked_at = %v, want nil", rec.RevokedAt)
	}

	// Hash at rest: the file holds the digest and the prefix, never the
	// plaintext key.
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read store file after mint: %v", err)
	}
	if bytes.Contains(data, []byte(key)) {
		t.Error("store file contains the plaintext key; only its digest may be persisted")
	}
	if !bytes.Contains(data, []byte(rec.Hash)) || !bytes.Contains(data, []byte(rec.Prefix)) {
		t.Error("store file missing the digest or the display prefix")
	}

	key2, _, err := s.Mint("tenant-b")
	if err != nil {
		t.Fatalf("second Mint: %v", err)
	}
	if key2 == key {
		t.Error("two Mints returned the same key")
	}
	if got := len(s.List()); got != 2 {
		t.Errorf("List after two mints = %d records, want 2", got)
	}
}

func TestValidate(t *testing.T) {
	t.Parallel()
	s, _ := openStore(t)

	active, _, err := s.Mint("tenant-a")
	if err != nil {
		t.Fatalf("Mint: %v", err)
	}
	revoked, _, err := s.Mint("tenant-b")
	if err != nil {
		t.Fatalf("Mint: %v", err)
	}
	if err := s.Revoke(revoked); err != nil {
		t.Fatalf("Revoke: %v", err)
	}

	cases := []struct {
		name    string
		key     string
		wantErr error
		tenant  string
	}{
		{"active", active, nil, "tenant-a"},
		{"revoked", revoked, ErrKeyRevoked, ""},
		{"unknown well-shaped", validKey("not-minted"), ErrUnknownKey, ""},
		{"empty", "", ErrEmptyKey, ""},
		{"whitespace", "   ", ErrEmptyKey, ""},
		{"wrong length", keyFormatPrefix + "abc", ErrUnknownKey, ""},
		{"uppercase hex", keyFormatPrefix + "ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF01", ErrUnknownKey, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec, err := s.Validate(tc.key)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("Validate: error = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("Validate: %v", err)
			}
			if rec.TenantID != tc.tenant {
				t.Errorf("record tenant = %q, want %q", rec.TenantID, tc.tenant)
			}
			if rec.Hash != sha256Hex(tc.key) {
				t.Errorf("record hash = %q, want %q", rec.Hash, sha256Hex(tc.key))
			}
		})
	}
}

func TestRevoke(t *testing.T) {
	t.Parallel()
	s, _ := openStore(t)

	key, rec, err := s.Mint("tenant-a")
	if err != nil {
		t.Fatalf("Mint: %v", err)
	}

	if err := s.Revoke(""); !errors.Is(err, ErrEmptyKey) {
		t.Errorf("Revoke(\"\"): error = %v, want ErrEmptyKey", err)
	}
	if err := s.Revoke(validKey("ghost")); !errors.Is(err, ErrUnknownKey) {
		t.Errorf("Revoke(unknown): error = %v, want ErrUnknownKey", err)
	}

	// By full key.
	if err := s.Revoke(key); err != nil {
		t.Fatalf("Revoke(full key): %v", err)
	}
	// Already revoked: by key and by prefix.
	if err := s.Revoke(key); !errors.Is(err, ErrKeyRevoked) {
		t.Errorf("re-Revoke(full key): error = %v, want ErrKeyRevoked", err)
	}
	if err := s.Revoke(rec.Prefix); !errors.Is(err, ErrKeyRevoked) {
		t.Errorf("re-Revoke(prefix): error = %v, want ErrKeyRevoked", err)
	}
	if _, err := s.Validate(key); !errors.Is(err, ErrKeyRevoked) {
		t.Errorf("Validate after revoke: error = %v, want ErrKeyRevoked", err)
	}
	listed := s.List()
	if len(listed) != 1 || listed[0].RevokedAt == nil {
		t.Errorf("List after revoke = %+v, want one revoked record", listed)
	}

	// By prefix, on a fresh key.
	key2, rec2, err := s.Mint("tenant-b")
	if err != nil {
		t.Fatalf("Mint: %v", err)
	}
	if err := s.Revoke(rec2.Prefix); err != nil {
		t.Fatalf("Revoke(prefix): %v", err)
	}
	if _, err := s.Validate(key2); !errors.Is(err, ErrKeyRevoked) {
		t.Errorf("Validate after prefix revoke: error = %v, want ErrKeyRevoked", err)
	}
}

func TestRevokeAmbiguousPrefix(t *testing.T) {
	t.Parallel()
	// Two stored keys share one display prefix; a prefix-only Revoke must
	// refuse, while each full key still resolves unambiguously.
	dir := t.TempDir()
	path := filepath.Join(dir, "store.json")
	k1, k2 := validKey("alpha"), validKey("beta")
	writeFixture(t, path, []KeyRecord{
		fixtureRecord(k1[:displayPrefixLen], sha256Hex(k1), "tenant-a"),
		fixtureRecord(k1[:displayPrefixLen], sha256Hex(k2), "tenant-b"),
	})
	s, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	if err := s.Revoke(k1[:displayPrefixLen]); !errors.Is(err, ErrAmbiguousKey) {
		t.Errorf("Revoke(shared prefix): error = %v, want ErrAmbiguousKey", err)
	}
	if err := s.Revoke(k1); err != nil {
		t.Fatalf("Revoke(full key k1): %v", err)
	}
	if _, err := s.Validate(k1); !errors.Is(err, ErrKeyRevoked) {
		t.Errorf("k1 should be revoked: %v", err)
	}
	if _, err := s.Validate(k2); err != nil {
		t.Errorf("k2 must stay active: %v", err)
	}
}

func TestNilStore(t *testing.T) {
	t.Parallel()
	var s *Store
	if _, _, err := s.Mint("t"); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil Mint: error = %v, want ErrInvalidArgument", err)
	}
	if _, err := s.Validate("k"); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil Validate: error = %v, want ErrInvalidArgument", err)
	}
	if err := s.Revoke("k"); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil Revoke: error = %v, want ErrInvalidArgument", err)
	}
	if got := s.List(); got != nil {
		t.Errorf("nil List = %v, want nil", got)
	}
	if got := s.TenantKeys("t"); got != nil {
		t.Errorf("nil TenantKeys = %v, want nil", got)
	}
}

func TestConcurrentUse(t *testing.T) {
	t.Parallel()
	s, _ := openStore(t)

	const workers, perWorker = 4, 10
	var (
		mu   sync.Mutex
		keys = make([]string, 0, workers*perWorker)
		wg   sync.WaitGroup
	)
	start := make(chan struct{})
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			<-start
			for i := 0; i < perWorker; i++ {
				key, _, err := s.Mint("tenant-" + string(rune('a'+w)))
				if err != nil {
					t.Errorf("concurrent Mint: %v", err)
					return
				}
				mu.Lock()
				keys = append(keys, key)
				mu.Unlock()
				// Read traffic interleaved with the writes.
				if _, err := s.Validate(key); err != nil {
					t.Errorf("concurrent Validate: %v", err)
					return
				}
				_ = s.List()
				_ = s.TenantKeys("tenant-x")
			}
		}(w)
	}
	close(start)
	wg.Wait()

	if got := len(s.List()); got != len(keys) {
		t.Fatalf("List = %d records, want %d", got, len(keys))
	}
	seen := make(map[string]bool, len(keys))
	for _, k := range keys {
		if seen[k] {
			t.Fatalf("duplicate key minted: %s", k)
		}
		seen[k] = true
		if _, err := s.Validate(k); err != nil {
			t.Errorf("Validate(%s...): %v", k[:12], err)
		}
	}
}
