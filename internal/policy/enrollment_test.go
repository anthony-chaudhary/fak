package policy

// enrollment_test.go is the acceptance suite for the org TRUST ANCHOR (W5 of epic
// #5315, issue #5323). orgenvelope_test.go proves the pure verifier refuses a bad
// key when the CALLER hands it the right one; orgledger_test.go proves
// anti-rollback survives the caller forgetting. This file proves the remaining
// link: WHERE the right key comes from, and that nothing can swap it out quietly.
//
// The load-bearing cases are all negative, because every one of them is a way the
// trust anchor could be defeated without breaking any signature:
//
//   - OPT-IN        — with no enrollment on disk the org plane is INERT, and the
//     anchor it hands back still refuses a perfectly-signed envelope rather than
//     failing open. Absent enrollment must be today's behavior, not a hole.
//   - PIN           — only the pinned key verifies. A foreign-key envelope, and a
//     pinned-key envelope carrying a foreign issuer, are both refused.
//   - NO SILENT RE-PIN — a second enroll to a different org, or to the SAME org
//     under a different root key (the MitM enroll-endpoint attack), is REFUSED,
//     and the refusal is INERT: the on-disk anchor is byte-identical afterwards.
//     Re-pinning requires an explicit revoke.
//   - FAIL CLOSED   — a tampered or truncated store is REFUSED, never degraded to
//     "not enrolled", because a silent degrade to un-enrolled discards whatever
//     tightening the org floor was carrying.
//   - DEVICE BINDING — the recorded device identity is what target selectors are
//     matched against, and an unrecognized selector matches NOTHING.
//
// Keys are generated per test from fixed seeds via the suite's testKey helper — no
// secret is ever embedded in the repo — and no test reads the wall clock.

import (
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// enrollPath returns a fresh, non-existent enrollment path under a per-test temp
// dir (nested one level down so the store's own MkdirAll is exercised).
func enrollPath(t *testing.T) string {
	t.Helper()
	return filepath.Join(t.TempDir(), "org", "enrollment.json")
}

// enrollReq is a complete, valid enrollment request pinning pub.
func enrollReq(pub ed25519.PublicKey) OrgEnrollRequest {
	return OrgEnrollRequest{
		OrgURL:   "https://policy.acme-corp.example/fak",
		Issuer:   "acme-corp",
		RootKey:  pub,
		DeviceID: "device-7f3a",
		User:     "jane@acme-corp.example",
		Groups:   []string{"eng", "sre"},
		Now:      fixedNow,
	}
}

// reasonOf extracts the closed-vocabulary refusal code from an enrollment error,
// so a test asserts the REASON rather than error prose.
func reasonOf(t *testing.T, err error) abi.ReasonCode {
	t.Helper()
	if err == nil {
		t.Fatalf("expected a refusal, got nil error")
	}
	var ee *EnvelopeError
	if !errors.As(err, &ee) {
		t.Fatalf("expected *EnvelopeError, got %T: %v", err, err)
	}
	return ee.Reason
}

// --- OPT-IN: absent enrollment is inert, and inert is not fail-open ------------

// TestOrgTrustAnchorAbsentIsInertNotFailOpen is the epic's central invariant: an
// un-enrolled fak behaves exactly as it does today. A missing store is NOT an
// error (that would make the org plane mandatory), but the anchor it yields must
// still refuse a real, correctly-signed envelope — so a caller that ignores the
// "not enrolled" signal cannot accidentally trust central policy.
func TestOrgTrustAnchorAbsentIsInertNotFailOpen(t *testing.T) {
	path := enrollPath(t)

	opts, enrolled, err := OrgTrustAnchor(path, fixedNow, "0.9.0")
	if err != nil {
		t.Fatalf("absent enrollment must not be an error, got %v", err)
	}
	if enrolled {
		t.Fatalf("absent enrollment reported as enrolled")
	}
	if len(opts.RootPublicKey) != 0 {
		t.Fatalf("un-enrolled anchor carries a root key of %d bytes; must carry none", len(opts.RootPublicKey))
	}

	// A genuinely valid envelope, signed by a real key, must STILL be refused —
	// there is no pinned anchor to prove it against.
	pub, priv := testKey(t, 0x11)
	_ = pub
	raw := liveEnv(t, priv, "acme-corp", 5)
	if _, err := VerifyEnvelope(raw, opts); err == nil {
		t.Fatalf("un-enrolled anchor ACCEPTED a signed envelope; the org plane must be inert without enrollment")
	} else if got := reasonOf(t, err); got != abi.ReasonTrustViolation {
		t.Fatalf("un-enrolled refusal reason = %s, want TRUST_VIOLATION", abi.ReasonName(got))
	}

	// And the probe must not have created the store as a side effect.
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("reading an absent enrollment created %s; the opt-in path must not write", path)
	}
}

// TestLoadOrgEnrollmentAbsentIsNotEnrolled proves the missing-file zero state is
// distinguishable from a broken one at the load seam too.
func TestLoadOrgEnrollmentAbsentIsNotEnrolled(t *testing.T) {
	e, enrolled, err := LoadOrgEnrollment(enrollPath(t))
	if err != nil {
		t.Fatalf("absent enrollment must load as the zero state, got %v", err)
	}
	if enrolled {
		t.Fatalf("absent enrollment reported enrolled")
	}
	if e.OrgURL != "" || len(e.RootKeys) != 0 || e.DeviceID != "" {
		t.Fatalf("absent enrollment returned a populated record: %+v", e)
	}
}

// --- PIN: the anchor round-trips, and ONLY it verifies -------------------------

// TestEnrollOrgPinsTrustAnchorAndDeviceIdentity is the DoD spine: enroll writes a
// pinned root key, the org URL, and a device identity, and all three survive a
// reload from disk.
func TestEnrollOrgPinsTrustAnchorAndDeviceIdentity(t *testing.T) {
	path := enrollPath(t)
	pub, _ := testKey(t, 0x21)

	got, err := EnrollOrg(path, enrollReq(pub))
	if err != nil {
		t.Fatalf("enroll: %v", err)
	}
	if got.Schema != OrgEnrollmentSchema {
		t.Fatalf("schema = %q, want %q", got.Schema, OrgEnrollmentSchema)
	}

	loaded, enrolled, err := LoadOrgEnrollment(path)
	if err != nil || !enrolled {
		t.Fatalf("reload after enroll: enrolled=%v err=%v", enrolled, err)
	}
	if loaded.OrgURL != "https://policy.acme-corp.example/fak" {
		t.Fatalf("org url = %q", loaded.OrgURL)
	}
	if loaded.Issuer != "acme-corp" {
		t.Fatalf("issuer = %q", loaded.Issuer)
	}
	if loaded.DeviceID != "device-7f3a" {
		t.Fatalf("device id = %q", loaded.DeviceID)
	}
	if loaded.EnrolledAt != fixedNow.Unix() {
		t.Fatalf("enrolled_at = %d, want %d", loaded.EnrolledAt, fixedNow.Unix())
	}
	keys, err := loaded.RootPublicKeys()
	if err != nil {
		t.Fatalf("decode pinned keys: %v", err)
	}
	if len(keys) != 1 || !keys[0].Equal(pub) {
		t.Fatalf("pinned key set does not round-trip to the enrolled root key")
	}

	// The store holds a PUBLIC key only. No private key material may ever be
	// persisted here, so assert the file cannot hold a full Ed25519 private key.
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read store: %v", err)
	}
	for _, blob := range keyBlobsIn(t, raw) {
		if len(blob) >= ed25519.PrivateKeySize {
			t.Fatalf("enrollment store holds a %d-byte key blob; only 32-byte PUBLIC keys belong here", len(blob))
		}
	}
}

// keyBlobsIn decodes every base64 string value in the stored record, so the test
// above can prove no private-key-sized blob was persisted.
func keyBlobsIn(t *testing.T, raw []byte) [][]byte {
	t.Helper()
	var generic map[string]any
	if err := json.Unmarshal(raw, &generic); err != nil {
		t.Fatalf("store is not JSON: %v", err)
	}
	var out [][]byte
	var walk func(any)
	walk = func(v any) {
		switch tv := v.(type) {
		case string:
			if b, err := base64.StdEncoding.DecodeString(tv); err == nil && len(b) > 0 {
				out = append(out, b)
			}
		case []any:
			for _, x := range tv {
				walk(x)
			}
		case map[string]any:
			for _, x := range tv {
				walk(x)
			}
		}
	}
	walk(generic)
	return out
}

// TestOrgTrustAnchorVerifiesOnlyThePinnedKey is the core security property: the
// verifier reached through an enrollment accepts an envelope from the pinned key
// and refuses one from any other key — no operator flag in the loop.
func TestOrgTrustAnchorVerifiesOnlyThePinnedKey(t *testing.T) {
	path := enrollPath(t)
	pub, priv := testKey(t, 0x31)
	foreignPub, foreignPriv := testKey(t, 0x32)
	if pub.Equal(foreignPub) {
		t.Fatalf("test setup: the two keys must differ")
	}
	if _, err := EnrollOrg(path, enrollReq(pub)); err != nil {
		t.Fatalf("enroll: %v", err)
	}

	opts, enrolled, err := OrgTrustAnchor(path, fixedNow, "0.9.0")
	if err != nil || !enrolled {
		t.Fatalf("trust anchor: enrolled=%v err=%v", enrolled, err)
	}

	// Pinned key: accepted.
	if _, err := VerifyEnvelope(liveEnv(t, priv, "acme-corp", 5), opts); err != nil {
		t.Fatalf("envelope signed by the PINNED key was refused: %v", err)
	}

	// Foreign key: refused. This is the whole point of pinning.
	if _, err := VerifyEnvelope(liveEnv(t, foreignPriv, "acme-corp", 5), opts); err == nil {
		t.Fatalf("envelope signed by a NON-PINNED key was ACCEPTED")
	} else if got := reasonOf(t, err); got != abi.ReasonTrustViolation {
		t.Fatalf("foreign-key refusal reason = %s, want TRUST_VIOLATION", abi.ReasonName(got))
	}

	// Pinned key but a foreign ISSUER: refused, because enrollment pins the
	// issuer too. A compromised signer cannot re-badge itself as another org.
	if _, err := VerifyEnvelope(liveEnv(t, priv, "evil-corp", 5), opts); err == nil {
		t.Fatalf("envelope with a foreign ISSUER was ACCEPTED; enrollment must pin the issuer")
	} else if got := reasonOf(t, err); got != abi.ReasonTrustViolation {
		t.Fatalf("foreign-issuer refusal reason = %s, want TRUST_VIOLATION", abi.ReasonName(got))
	}
}

// --- NO SILENT RE-PIN ----------------------------------------------------------

// TestEnrollOrgRefusesSilentRepin covers the DoD's explicit requirement and the
// attack under it. Re-pointing an enrolled device at another org — or at the same
// org under a different root key, which is what a MitM'd enroll endpoint would
// serve — must be REFUSED rather than silently replacing the anchor.
func TestEnrollOrgRefusesSilentRepin(t *testing.T) {
	path := enrollPath(t)
	pub, _ := testKey(t, 0x41)
	otherPub, _ := testKey(t, 0x42)

	first, err := EnrollOrg(path, enrollReq(pub))
	if err != nil {
		t.Fatalf("first enroll: %v", err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read store: %v", err)
	}

	cases := []struct {
		name   string
		mutate func(r *OrgEnrollRequest)
	}{
		{"different org", func(r *OrgEnrollRequest) { r.OrgURL = "https://policy.evil-corp.example/fak" }},
		{"different root key", func(r *OrgEnrollRequest) { r.RootKey = otherPub }},
		{"different issuer", func(r *OrgEnrollRequest) { r.Issuer = "evil-corp" }},
		{"different device id", func(r *OrgEnrollRequest) { r.DeviceID = "device-0000" }},
		{"widened groups", func(r *OrgEnrollRequest) { r.Groups = []string{"eng", "sre", "admin"} }},
		{"different user", func(r *OrgEnrollRequest) { r.User = "root@acme-corp.example" }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := enrollReq(pub)
			tc.mutate(&req)
			if _, err := EnrollOrg(path, req); err == nil {
				t.Fatalf("re-enroll with a %s was ACCEPTED; a re-pin must require an explicit revoke", tc.name)
			} else if got := reasonOf(t, err); got != abi.ReasonTrustViolation {
				t.Fatalf("re-pin refusal reason = %s, want TRUST_VIOLATION", abi.ReasonName(got))
			}

			// A REFUSAL IS INERT: the anchor on disk is untouched.
			after, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("read store after refusal: %v", err)
			}
			if string(after) != string(before) {
				t.Fatalf("a refused re-pin MUTATED the stored anchor\nbefore: %s\nafter:  %s", before, after)
			}
		})
	}

	// The identical request is idempotent — re-running enroll is not an attack,
	// and it must not re-stamp the original enrollment time.
	again, err := EnrollOrg(path, enrollReq(pub))
	if err != nil {
		t.Fatalf("identical re-enroll must be idempotent, got %v", err)
	}
	if again.EnrolledAt != first.EnrolledAt {
		t.Fatalf("idempotent re-enroll re-stamped enrolled_at (%d -> %d)", first.EnrolledAt, again.EnrolledAt)
	}
}

// TestRevokeOrgEnrollmentClearsTheAnchor proves the sanctioned re-pin path: revoke,
// then enroll elsewhere. After a revoke the org plane is inert again.
func TestRevokeOrgEnrollmentClearsTheAnchor(t *testing.T) {
	path := enrollPath(t)
	pub, _ := testKey(t, 0x51)
	otherPub, otherPriv := testKey(t, 0x52)

	if _, err := EnrollOrg(path, enrollReq(pub)); err != nil {
		t.Fatalf("enroll: %v", err)
	}
	revoked, err := RevokeOrgEnrollment(path)
	if err != nil {
		t.Fatalf("revoke: %v", err)
	}
	if !revoked {
		t.Fatalf("revoke reported nothing to revoke while enrolled")
	}

	// Inert again: exactly the un-enrolled posture.
	_, enrolled, err := OrgTrustAnchor(path, fixedNow, "0.9.0")
	if err != nil || enrolled {
		t.Fatalf("after revoke: enrolled=%v err=%v, want inert", enrolled, err)
	}

	// Revoking twice is a no-op, not an error.
	again, err := RevokeOrgEnrollment(path)
	if err != nil {
		t.Fatalf("second revoke: %v", err)
	}
	if again {
		t.Fatalf("second revoke claimed to revoke an absent enrollment")
	}

	// And now re-enrolling elsewhere is permitted — the explicit path.
	req := enrollReq(otherPub)
	req.OrgURL = "https://policy.other-corp.example/fak"
	req.Issuer = "other-corp"
	if _, err := EnrollOrg(path, req); err != nil {
		t.Fatalf("enroll after revoke: %v", err)
	}
	opts, enrolled, err := OrgTrustAnchor(path, fixedNow, "0.9.0")
	if err != nil || !enrolled {
		t.Fatalf("re-enroll anchor: enrolled=%v err=%v", enrolled, err)
	}
	if _, err := VerifyEnvelope(liveEnv(t, otherPriv, "other-corp", 5), opts); err != nil {
		t.Fatalf("envelope from the newly pinned org was refused: %v", err)
	}
}

// --- FAIL CLOSED ---------------------------------------------------------------

// TestLoadOrgEnrollmentTamperedFailsClosed is the counterpart of the ledger's
// broken-store rule. Degrading a damaged or edited store to "not enrolled" would
// let anyone with write access silently drop whatever tightening the org floor
// carried — so every problem except ABSENCE is a refusal.
func TestLoadOrgEnrollmentTamperedFailsClosed(t *testing.T) {
	pub, _ := testKey(t, 0x61)
	foreignPub, _ := testKey(t, 0x62)

	cases := []struct {
		name   string
		mutate func(t *testing.T, path string)
		reason abi.ReasonCode
	}{
		{
			name: "swapped root key",
			mutate: func(t *testing.T, path string) {
				rewriteStore(t, path, func(m map[string]any) {
					m["root_keys"] = []any{base64.StdEncoding.EncodeToString(foreignPub)}
				})
			},
			reason: abi.ReasonTrustViolation,
		},
		{
			name: "swapped org url",
			mutate: func(t *testing.T, path string) {
				rewriteStore(t, path, func(m map[string]any) {
					m["org_url"] = "https://policy.evil-corp.example/fak"
				})
			},
			reason: abi.ReasonTrustViolation,
		},
		{
			name: "swapped device id",
			mutate: func(t *testing.T, path string) {
				rewriteStore(t, path, func(m map[string]any) { m["device_id"] = "device-0000" })
			},
			reason: abi.ReasonTrustViolation,
		},
		{
			name: "dropped checksum",
			mutate: func(t *testing.T, path string) {
				rewriteStore(t, path, func(m map[string]any) { delete(m, "sum") })
			},
			reason: abi.ReasonTrustViolation,
		},
		{
			name: "wrong schema",
			mutate: func(t *testing.T, path string) {
				rewriteStore(t, path, func(m map[string]any) { m["schema"] = "fak-org-enrollment/v99" })
			},
			reason: abi.ReasonTrustViolation,
		},
		{
			name: "truncated file",
			mutate: func(t *testing.T, path string) {
				raw, err := os.ReadFile(path)
				if err != nil {
					t.Fatalf("read: %v", err)
				}
				if err := os.WriteFile(path, raw[:len(raw)/2], 0o600); err != nil {
					t.Fatalf("truncate: %v", err)
				}
			},
			reason: abi.ReasonTrustViolation,
		},
		{
			name: "oversized file",
			mutate: func(t *testing.T, path string) {
				if err := os.WriteFile(path, []byte(strings.Repeat("x", 1<<20)), 0o600); err != nil {
					t.Fatalf("write: %v", err)
				}
			},
			reason: abi.ReasonOversize,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := enrollPath(t)
			if _, err := EnrollOrg(path, enrollReq(pub)); err != nil {
				t.Fatalf("enroll: %v", err)
			}
			tc.mutate(t, path)

			if _, enrolled, err := LoadOrgEnrollment(path); err == nil {
				t.Fatalf("a %s store LOADED (enrolled=%v); a damaged anchor must fail closed", tc.name, enrolled)
			} else if got := reasonOf(t, err); got != tc.reason {
				t.Fatalf("%s refusal reason = %s, want %s", tc.name, abi.ReasonName(got), abi.ReasonName(tc.reason))
			}

			// The anchor seam must refuse too — never hand back a usable
			// VerifyOptions, and never silently report "not enrolled".
			opts, enrolled, err := OrgTrustAnchor(path, fixedNow, "0.9.0")
			if err == nil {
				t.Fatalf("OrgTrustAnchor accepted a %s store (enrolled=%v)", tc.name, enrolled)
			}
			if len(opts.RootPublicKey) != 0 {
				t.Fatalf("a refused anchor still carried a root key")
			}
		})
	}
}

// rewriteStore applies mutate to the stored record as generic JSON and writes it
// back — a hand-edit, exactly as an attacker with file access would make it.
func rewriteStore(t *testing.T, path string, mutate func(map[string]any)) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read store: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("decode store: %v", err)
	}
	mutate(m)
	out, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("encode store: %v", err)
	}
	if err := os.WriteFile(path, out, 0o600); err != nil {
		t.Fatalf("write store: %v", err)
	}
}

// TestEnrollOrgRefusesMalformedAnchor proves a bad anchor never reaches the disk.
// A store holding an unusable key would only fail later, at verify time, where
// the failure looks like a policy problem instead of an enrollment one.
func TestEnrollOrgRefusesMalformedAnchor(t *testing.T) {
	pub, _ := testKey(t, 0x71)

	cases := []struct {
		name   string
		mutate func(r *OrgEnrollRequest)
	}{
		{"no root key", func(r *OrgEnrollRequest) { r.RootKey = nil }},
		{"short root key", func(r *OrgEnrollRequest) { r.RootKey = ed25519.PublicKey(make([]byte, 16)) }},
		{"all-zero root key", func(r *OrgEnrollRequest) { r.RootKey = ed25519.PublicKey(make([]byte, ed25519.PublicKeySize)) }},
		{"no org url", func(r *OrgEnrollRequest) { r.OrgURL = "" }},
		{"blank org url", func(r *OrgEnrollRequest) { r.OrgURL = "   " }},
		{"no issuer", func(r *OrgEnrollRequest) { r.Issuer = "" }},
		{"no device id", func(r *OrgEnrollRequest) { r.DeviceID = "" }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := enrollPath(t)
			req := enrollReq(pub)
			tc.mutate(&req)
			if _, err := EnrollOrg(path, req); err == nil {
				t.Fatalf("enroll accepted %s", tc.name)
			}
			if _, err := os.Stat(path); !os.IsNotExist(err) {
				t.Fatalf("a refused enroll (%s) still wrote %s", tc.name, path)
			}
		})
	}
}

// --- DEVICE IDENTITY BINDING ---------------------------------------------------

// TestOrgEnrollmentDeviceIdentityBinds proves the recorded identity is what a
// signed target selector is resolved against, and that an unknown selector shape
// matches NOTHING — a targeting rule fak cannot parse must never be read as
// "applies to everyone".
func TestOrgEnrollmentDeviceIdentityBinds(t *testing.T) {
	path := enrollPath(t)
	pub, _ := testKey(t, 0x81)
	if _, err := EnrollOrg(path, enrollReq(pub)); err != nil {
		t.Fatalf("enroll: %v", err)
	}
	e, _, err := LoadOrgEnrollment(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	cases := []struct {
		target string
		want   bool
	}{
		{"", true},
		{"*", true},
		{"device:*", true},
		{"device:device-7f3a", true},
		{"device:device-0000", false},
		{"user:jane@acme-corp.example", true},
		{"user:root@acme-corp.example", false},
		{"group:eng", true},
		{"group:sre", true},
		{"group:admin", false},
		{"group:*", true},
		// Fail closed on anything the selector grammar does not cover.
		{"tenant:acme", false},
		{"device-7f3a", false},
		{"DEVICE:device-7f3a", false},
		{"group:", false},
		{":", false},
	}
	for _, tc := range cases {
		if got := e.MatchesTarget(tc.target); got != tc.want {
			t.Errorf("MatchesTarget(%q) = %v, want %v", tc.target, got, tc.want)
		}
	}

	// An enrollment with no groups must not match a group selector.
	bare := OrgEnrollment{DeviceID: "d1"}
	if bare.MatchesTarget("group:eng") {
		t.Errorf("an enrollment with no groups matched group:eng")
	}
	// A zero enrollment must not be targetable by a wildcard either: with no
	// device identity there is nothing to bind a grant to.
	var zero OrgEnrollment
	if zero.MatchesTarget("*") {
		t.Errorf("a zero (un-enrolled) record matched the wildcard target")
	}
}

// TestOrgEnrollmentPathIsOverridable proves the store location is resolvable
// without a hard-coded home, so a test or an operator can point it somewhere
// explicit. The env var carries a PATH, never key material.
func TestOrgEnrollmentPathIsOverridable(t *testing.T) {
	want := filepath.Join(t.TempDir(), "custom-enrollment.json")
	t.Setenv(OrgEnrollmentPathEnv, want)
	if got := OrgEnrollmentPath(); got != want {
		t.Fatalf("OrgEnrollmentPath() = %q, want %q", got, want)
	}

	t.Setenv(OrgEnrollmentPathEnv, "")
	got := OrgEnrollmentPath()
	if got == "" {
		t.Fatalf("OrgEnrollmentPath() returned empty with no override")
	}
	if !strings.HasSuffix(filepath.ToSlash(got), "fak/org-enrollment.json") &&
		!strings.HasSuffix(filepath.ToSlash(got), ".fak/org-enrollment.json") {
		t.Fatalf("default enrollment path %q does not land under a fak config dir", got)
	}
}

// TestOrgTrustAnchorCarriesRunningFacts proves the anchor forwards the ambient
// facts the pure verifier refuses to read for itself, so an enrolled caller still
// gets the freshness and min_version gates.
func TestOrgTrustAnchorCarriesRunningFacts(t *testing.T) {
	path := enrollPath(t)
	pub, priv := testKey(t, 0x91)
	if _, err := EnrollOrg(path, enrollReq(pub)); err != nil {
		t.Fatalf("enroll: %v", err)
	}

	// The invariant is that the anchor FORWARDS the caller's running version
	// verbatim — it never substitutes one of its own. Asserting the round trip
	// rather than today's release string keeps this from failing on a version
	// bump that changed nothing about the behaviour under test.
	const running = "0.9.0"
	opts, _, err := OrgTrustAnchor(path, fixedNow, running)
	if err != nil {
		t.Fatalf("anchor: %v", err)
	}
	if !opts.Now.Equal(fixedNow) {
		t.Fatalf("anchor Now = %v, want %v", opts.Now, fixedNow)
	}
	if opts.RunningVersion != running {
		t.Fatalf("anchor RunningVersion = %q, want the caller's %q forwarded unchanged", opts.RunningVersion, running)
	}
	if opts.ExpectedIssuer != "acme-corp" {
		t.Fatalf("anchor ExpectedIssuer = %q, want the enrolled issuer", opts.ExpectedIssuer)
	}

	// An expired envelope from the pinned key is still refused: pinning the key
	// does not disable the freshness gate.
	stale := envAt(t, priv, "acme-corp", 5, fixedNow.Unix()-7200, fixedNow.Unix()-3600)
	if _, err := VerifyEnvelope(stale, opts); err == nil {
		t.Fatalf("an expired envelope from the pinned key was accepted")
	} else if got := reasonOf(t, err); got != abi.ReasonUnwitnessed {
		t.Fatalf("expired refusal reason = %s, want UNWITNESSED", abi.ReasonName(got))
	}

	// A binary too old for the envelope is refused as well.
	old, _, err := OrgTrustAnchor(path, fixedNow, "0.1.0")
	if err != nil {
		t.Fatalf("anchor: %v", err)
	}
	if _, err := VerifyEnvelope(liveEnv(t, priv, "acme-corp", 5), old); err == nil {
		t.Fatalf("an envelope requiring a newer binary was accepted")
	} else if got := reasonOf(t, err); got != abi.ReasonPolicyBlock {
		t.Fatalf("min_version refusal reason = %s, want POLICY_BLOCK", abi.ReasonName(got))
	}
}

func TestEnrollOrgPersistsOptInAuditEndpoint(t *testing.T) {
	path := enrollPath(t)
	_, err := EnrollOrg(path, OrgEnrollRequest{OrgURL: "https://org.example/policy", Issuer: "org.example", RootKey: func() ed25519.PublicKey { p, _ := testKey(t, 0xa1); return p }(), DeviceID: "dev-1", AuditURL: "https://audit.example/receipts", Now: time.Unix(100, 0)})
	if err != nil {
		t.Fatal(err)
	}
	got, ok, err := LoadOrgEnrollment(path)
	if err != nil || !ok {
		t.Fatalf("load ok=%v err=%v", ok, err)
	}
	if got.AuditURL != "https://audit.example/receipts" {
		t.Fatalf("audit_url=%q", got.AuditURL)
	}
}
func TestEnrollOrgRejectsMalformedAuditEndpoint(t *testing.T) {
	_, err := EnrollOrg(enrollPath(t), OrgEnrollRequest{OrgURL: "https://org.example/policy", Issuer: "org.example", RootKey: func() ed25519.PublicKey { p, _ := testKey(t, 0xa1); return p }(), DeviceID: "dev-1", AuditURL: "file:///secret", Now: time.Unix(100, 0)})
	if err == nil {
		t.Fatal("expected malformed audit URL refusal")
	}
}
