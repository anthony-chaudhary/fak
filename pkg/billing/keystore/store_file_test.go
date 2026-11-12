package keystore

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

var fixtureTime = time.Date(2026, 11, 1, 12, 0, 0, 0, time.UTC)

// fixtureRecord builds a contract-valid record with a fixed creation time.
func fixtureRecord(prefix, hash, tenant string) KeyRecord {
	return KeyRecord{
		Prefix:    prefix,
		Hash:      hash,
		TenantID:  tenant,
		CreatedAt: fixtureTime,
	}
}

// writeFixture writes records as a well-formed v1 store file at path.
func writeFixture(t *testing.T, path string, records []KeyRecord) {
	t.Helper()
	sf := storeFile{Version: KeystoreVersionV1, Keys: records}
	b, err := json.Marshal(&sf)
	if err != nil {
		t.Fatalf("marshal fixture: %v", err)
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
}

// baseStoreJSON is one well-formed store file holding a single record for
// validKey("alpha"); tamper cases mutate it.
func baseStoreJSON() string {
	k := validKey("alpha")
	rec := fmt.Sprintf(`{"prefix":%q,"hash":%q,"tenant_id":"tenant-a","created_at":"2026-11-01T12:00:00Z"}`,
		k[:displayPrefixLen], sha256Hex(k))
	return fmt.Sprintf(`{"version":%q,"keys":[%s]}`, KeystoreVersionV1, rec)
}

func TestOpenMissingFile(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "absent.json")
	s, err := Open(path)
	if err != nil {
		t.Fatalf("Open(missing): %v", err)
	}
	if got := s.List(); len(got) != 0 {
		t.Errorf("List on missing-file store = %d records, want 0", len(got))
	}
	if _, err := s.Validate(validKey("x")); !errors.Is(err, ErrUnknownKey) {
		t.Errorf("Validate on empty store: error = %v, want ErrUnknownKey", err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Errorf("Open(missing) created the file; want no file until the first mutation (stat err = %v)", err)
	}

	if _, _, err := s.Mint("tenant-a"); err != nil {
		t.Fatalf("Mint: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("store file absent after first mutation: %v", err)
	}
	if _, err := decodeStore(data); err != nil {
		t.Errorf("file written by Mint does not decode: %v", err)
	}
}

func TestPersistenceAcrossReopen(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "store.json")
	s, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	kA1, _, _ := s.Mint("tenant-a")
	kA2, _, _ := s.Mint("tenant-a")
	kB, _, _ := s.Mint("tenant-b")
	if err := s.Revoke(kB); err != nil {
		t.Fatalf("Revoke: %v", err)
	}
	before := s.List()

	s2, err := Open(path)
	if err != nil {
		t.Fatalf("re-Open: %v", err)
	}
	after := s2.List()
	if len(before) != len(after) {
		t.Fatalf("reopen lost records: %d -> %d", len(before), len(after))
	}
	for i := range before {
		if !recordEqual(before[i], after[i]) {
			t.Errorf("record %d differs across reopen:\n before %+v\n after  %+v", i, before[i], after[i])
		}
	}
	if _, err := s2.Validate(kA1); err != nil {
		t.Errorf("Validate(kA1) after reopen: %v", err)
	}
	if _, err := s2.Validate(kA2); err != nil {
		t.Errorf("Validate(kA2) after reopen: %v", err)
	}
	if _, err := s2.Validate(kB); !errors.Is(err, ErrKeyRevoked) {
		t.Errorf("Validate(kB) after reopen: error = %v, want ErrKeyRevoked", err)
	}
	if got := s2.TenantKeys("tenant-a"); len(got) != 2 {
		t.Errorf("TenantKeys(tenant-a) after reopen = %d records, want 2", len(got))
	}
}

func TestTamperedStoreFiles(t *testing.T) {
	t.Parallel()
	base := baseStoreJSON()
	cases := []struct {
		name    string
		mutate  func(string) string
		wantErr error
	}{
		{"garbage bytes", func(s string) string { return "{not json" }, ErrMalformedPayload},
		{"empty file", func(s string) string { return "" }, ErrMalformedPayload},
		{"unknown top-level field", func(s string) string {
			return `{"version":"fak-billing-keystore/v1","keys":[],"extra":1}`
		}, ErrMalformedPayload},
		{"unknown record field", func(s string) string {
			return `{"version":"fak-billing-keystore/v1","keys":[{"prefix":"sk-fak-aaaaa","hash":"` +
				sha256Hex(validKey("alpha")) + `","tenant_id":"t","created_at":"2026-11-01T12:00:00Z","note":"x"}]}`
		}, ErrMalformedPayload},
		{"trailing data", func(s string) string { return s + `{"version":"fak-billing-keystore/v1"}` }, ErrMalformedPayload},
		{"wrong version", func(s string) string {
			return `{"version":"fak-billing-keystore/v2","keys":[]}`
		}, ErrMalformedPayload},
		{"keys not an array", func(s string) string {
			return `{"version":"fak-billing-keystore/v1","keys":"nope"}`
		}, ErrMalformedPayload},
		{"short hash", func(s string) string {
			return replaceFirst(base, `"hash":"`, `"hash":"`+sha256Hex(validKey("alpha"))[:63])
		}, ErrInvalidRecord},
		{"uppercase hash", func(s string) string {
			h := sha256Hex(validKey("alpha"))
			up := h
			for i, c := range up {
				if c >= 'a' && c <= 'f' {
					up = up[:i] + string(c-'a'+'A') + up[i+1:]
				}
			}
			return replaceFirst(base, sha256Hex(validKey("alpha")), up)
		}, ErrInvalidRecord},
		{"non-hex hash", func(s string) string {
			return replaceFirst(base, sha256Hex(validKey("alpha")), "g"+sha256Hex(validKey("alpha"))[1:])
		}, ErrInvalidRecord},
		{"empty tenant", func(s string) string { return replaceFirst(base, `"tenant_id":"tenant-a"`, `"tenant_id":""`) }, ErrInvalidRecord},
		{"zero created_at", func(s string) string {
			return replaceFirst(base, `"created_at":"2026-11-01T12:00:00Z"`, `"created_at":"0001-01-01T00:00:00Z"`)
		}, ErrInvalidRecord},
		{"bad prefix marker", func(s string) string { return replaceFirst(base, `"prefix":"sk-fak-`, `"prefix":"pk-fak-`) }, ErrInvalidRecord},
		{"short prefix", func(s string) string {
			return replaceFirst(base, fmt.Sprintf(`"prefix":"%s"`, validKey("alpha")[:displayPrefixLen]),
				fmt.Sprintf(`"prefix":"%s"`, validKey("alpha")[:displayPrefixLen-1]))
		}, ErrInvalidRecord},
		{"duplicate hash", func(s string) string {
			k := validKey("alpha")
			rec := fmt.Sprintf(`{"prefix":%q,"hash":%q,"tenant_id":"tenant-a","created_at":"2026-11-01T12:00:00Z"}`,
				k[:displayPrefixLen], sha256Hex(k))
			return fmt.Sprintf(`{"version":%q,"keys":[%s,%s]}`, KeystoreVersionV1, rec, rec)
		}, ErrInvalidRecord},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "store.json")
			raw := tc.mutate(base)
			if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
				t.Fatalf("write tampered file: %v", err)
			}
			_, err := Open(path)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("Open: error = %v, want %v", err, tc.wantErr)
			}
			if !errors.Is(err, ErrCorruptStore) {
				t.Fatalf("Open: error %v does not wrap ErrCorruptStore", err)
			}
			// The file must be left untouched.
			got, rerr := os.ReadFile(path)
			if rerr != nil {
				t.Fatalf("read back tampered file: %v", rerr)
			}
			if !bytes.Equal(got, []byte(raw)) {
				t.Error("Open modified the rejected store file")
			}
		})
	}
}

// recordEqual compares two records by value: times with .Equal, and
// RevokedAt through its referent, since a decoded record carries fresh
// pointers.
func recordEqual(a, b KeyRecord) bool {
	if a.Prefix != b.Prefix || a.Hash != b.Hash || a.TenantID != b.TenantID ||
		!a.CreatedAt.Equal(b.CreatedAt) {
		return false
	}
	if (a.RevokedAt == nil) != (b.RevokedAt == nil) {
		return false
	}
	return a.RevokedAt == nil || a.RevokedAt.Equal(*b.RevokedAt)
}

// replaceFirst swaps the first occurrence of old for new in s.
func replaceFirst(s, old, new string) string {
	i := index(s, old)
	if i < 0 {
		panic("replaceFirst: needle not found")
	}
	return s[:i] + new + s[i+len(old):]
}

func index(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}

func TestTamperedHashDegradesToUnknown(t *testing.T) {
	t.Parallel()
	// A flipped hash character still parses (64 lowercase hex), so Open
	// succeeds; the original key simply no longer validates.
	dir := t.TempDir()
	path := filepath.Join(dir, "store.json")
	k := validKey("alpha")
	writeFixture(t, path, []KeyRecord{fixtureRecord(k[:displayPrefixLen], sha256Hex(k), "tenant-a")})
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	flipped := replaceFirst(string(raw), sha256Hex(k), flipHexChar(sha256Hex(k)))
	if err := os.WriteFile(path, []byte(flipped), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}

	s, err := Open(path)
	if err != nil {
		t.Fatalf("Open with tampered hash: %v", err)
	}
	if _, err := s.Validate(k); !errors.Is(err, ErrUnknownKey) {
		t.Errorf("Validate(tampered): error = %v, want ErrUnknownKey", err)
	}
}

// flipHexChar changes the first hex character of h to a different hex
// character, so the result stays well-formed but differs from h.
func flipHexChar(h string) string {
	alt := byte('0')
	if h[0] == '0' {
		alt = '1'
	}
	out := make([]byte, len(h))
	copy(out, h)
	out[0] = alt
	return string(out)
}

func TestListOrderAndTenantFilter(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "store.json")
	ha, hb, hc := sha256Hex(validKey("a")), sha256Hex(validKey("b")), sha256Hex(validKey("c"))
	early := fixtureTime.Add(-time.Hour)
	bRec := fixtureRecord("sk-fak-bbbbb", hb, "tenant-2")
	bRec.CreatedAt = early
	// Insertion order deliberately disagrees with the expected List order.
	writeFixture(t, path, []KeyRecord{
		fixtureRecord("sk-fak-aaaaa", ha, "tenant-1"), // 12:00
		bRec, // 11:00 (earlier)
		fixtureRecord("sk-fak-fffff", hc, "tenant-1"), // 12:00
	})
	s, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	got := s.List()
	wantPrefixes := []string{"sk-fak-bbbbb", "sk-fak-aaaaa", "sk-fak-fffff"}
	if len(got) != len(wantPrefixes) {
		t.Fatalf("List = %d records, want %d", len(got), len(wantPrefixes))
	}
	for i, p := range wantPrefixes {
		if got[i].Prefix != p {
			t.Errorf("List[%d].Prefix = %q, want %q", i, got[i].Prefix, p)
		}
	}
	tenants := s.TenantKeys("tenant-1")
	if len(tenants) != 2 || tenants[0].Prefix != "sk-fak-aaaaa" || tenants[1].Prefix != "sk-fak-fffff" {
		t.Errorf("TenantKeys(tenant-1) = %+v, want the two tenant-1 records in order", tenants)
	}
	if got := s.TenantKeys("nobody"); len(got) != 0 {
		t.Errorf("TenantKeys(nobody) = %d records, want 0", len(got))
	}
	if got := s.TenantKeys(""); len(got) != 0 {
		t.Errorf("TenantKeys(\"\") = %d records, want 0", len(got))
	}
}

func TestEncodeDecodeRoundTrip(t *testing.T) {
	t.Parallel()
	k1, k2 := validKey("one"), validKey("two")
	recs := []KeyRecord{
		fixtureRecord(k1[:displayPrefixLen], sha256Hex(k1), "tenant-a"),
		fixtureRecord(k2[:displayPrefixLen], sha256Hex(k2), "tenant-b"),
	}
	revoked := fixtureTime.Add(5 * time.Minute)
	recs[1].RevokedAt = &revoked

	b, err := encodeStore(recs)
	if err != nil {
		t.Fatalf("encodeStore: %v", err)
	}
	sf, err := decodeStore(b)
	if err != nil {
		t.Fatalf("decodeStore round trip: %v", err)
	}
	if sf.Version != KeystoreVersionV1 || len(sf.Keys) != len(recs) {
		t.Fatalf("round-trip shape = %+v", sf)
	}
	// encodeStore emits List order (CreatedAt, then Prefix), so compare
	// against the same records in that order.
	want := make([]KeyRecord, len(recs))
	copy(want, recs)
	sortByKeyOrder(want)
	for i := range want {
		if !recordEqual(sf.Keys[i], want[i]) {
			t.Errorf("round-trip record %d = %+v, want %+v", i, sf.Keys[i], want[i])
		}
	}

	// Same logical state, different insertion order, identical bytes.
	b2, err := encodeStore([]KeyRecord{recs[1], recs[0]})
	if err != nil {
		t.Fatalf("encodeStore (reordered): %v", err)
	}
	if !bytes.Equal(b, b2) {
		t.Error("encoding is not byte-stable under reordering of the same logical state")
	}
}

func TestKeyRecordValidateContract(t *testing.T) {
	t.Parallel()
	valid := fixtureRecord("sk-fak-aaaaa", sha256Hex(validKey("v")), "tenant-x")
	cases := []struct {
		name    string
		mutate  func(KeyRecord) KeyRecord
		wantErr bool
	}{
		{"valid", func(r KeyRecord) KeyRecord { return r }, false},
		{"short prefix", func(r KeyRecord) KeyRecord { r.Prefix = r.Prefix[:11]; return r }, true},
		{"long prefix", func(r KeyRecord) KeyRecord { r.Prefix = r.Prefix + "a"; return r }, true},
		{"wrong marker", func(r KeyRecord) KeyRecord { r.Prefix = "pk-" + r.Prefix[3:]; return r }, true},
		{"non-hex prefix tail", func(r KeyRecord) KeyRecord { r.Prefix = "sk-fak-zzzzz"; return r }, true},
		{"uppercase prefix tail", func(r KeyRecord) KeyRecord { r.Prefix = "sk-fak-AAAAA"; return r }, true},
		{"short hash", func(r KeyRecord) KeyRecord { r.Hash = r.Hash[:63]; return r }, true},
		{"uppercase hash", func(r KeyRecord) KeyRecord { r.Hash = "ABCDEF" + r.Hash[6:]; return r }, true},
		{"empty tenant", func(r KeyRecord) KeyRecord { r.TenantID = ""; return r }, true},
		{"zero created_at", func(r KeyRecord) KeyRecord { r.CreatedAt = time.Time{}; return r }, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := tc.mutate(valid)
			err := r.Validate()
			if tc.wantErr {
				if !errors.Is(err, ErrInvalidRecord) {
					t.Fatalf("Validate: error = %v, want ErrInvalidRecord", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("Validate: unexpected error: %v", err)
			}
		})
	}
}
