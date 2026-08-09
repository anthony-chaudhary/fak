package main

import (
	"bytes"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/policy"
)

// cmd/fak/enroll_test.go — the acceptance bar for `fak enroll` (#5323).
//
// The library half already proves the store's fail-closed rules. What is testable ONLY
// here is the operator-facing contract: that a refusal is visible in the exit code, that
// an un-enrolled box SAYS it is inert instead of printing nothing, and — the load-bearing
// one — that a damaged anchor never renders like an absent one.

func enrollTestKey(t *testing.T, seed byte) (ed25519.PublicKey, string) {
	t.Helper()
	raw := make([]byte, ed25519.PublicKeySize)
	for i := range raw {
		raw[i] = seed + byte(i)
	}
	return ed25519.PublicKey(raw), base64.StdEncoding.EncodeToString(raw)
}

// runEnrollCapture drives one invocation and hands back both streams, so a test can assert
// on WHICH stream a message landed on. That distinction is the contract: refusals belong on
// stderr, and the JSON ledger must be alone on stdout or no consumer can parse it.
func runEnrollCapture(t *testing.T, argv ...string) (rc int, stdout, stderr string) {
	t.Helper()
	var out, errb bytes.Buffer
	rc = runEnroll(&out, &errb, argv)
	return rc, out.String(), errb.String()
}

func TestEnrollPinsAnchorAndDeviceIdentity(t *testing.T) {
	store := filepath.Join(t.TempDir(), "org-enrollment.json")
	_, b64 := enrollTestKey(t, 7)

	rc, stdout, stderr := runEnrollCapture(t,
		"--path", store,
		"--org", "https://policy.acme.example/fak",
		"--root-key", b64,
		"--device", "node-a",
		"--user", "jane",
		"--groups", "eng, sre",
	)
	if rc != 0 {
		t.Fatalf("enroll rc = %d, want 0 (stderr: %s)", rc, stderr)
	}
	for _, want := range []string{"enrolled:", "https://policy.acme.example/fak", "node-a", "jane", "eng, sre", "sha256:"} {
		if !strings.Contains(stdout, want) {
			t.Errorf("enroll stdout missing %q:\n%s", want, stdout)
		}
	}

	rec, enrolled, err := policy.LoadOrgEnrollment(store)
	if err != nil || !enrolled {
		t.Fatalf("store not readable after enroll: enrolled=%v err=%v", enrolled, err)
	}
	if rec.DeviceID != "node-a" || rec.User != "jane" {
		t.Errorf("identity not bound: device=%q user=%q", rec.DeviceID, rec.User)
	}
	if len(rec.Groups) != 2 || rec.Groups[0] != "eng" || rec.Groups[1] != "sre" {
		t.Errorf("groups not split/trimmed: %#v", rec.Groups)
	}
	// The issuer default is derived from the org URL host, not left empty — an empty
	// pinned issuer would accept a signature from any identity.
	if rec.Issuer != "policy.acme.example" {
		t.Errorf("default issuer = %q, want policy.acme.example", rec.Issuer)
	}
	// The private half must never be inferable from what we print or store.
	if strings.Contains(stdout, b64) {
		t.Errorf("enroll printed the raw key blob instead of a fingerprint:\n%s", stdout)
	}
}

// TestEnrollRefusesSilentRepin is the DoD line: a second enroll pointing somewhere else is
// refused, the on-disk anchor is untouched, and --revoke is the only route through.
func TestEnrollRefusesSilentRepin(t *testing.T) {
	store := filepath.Join(t.TempDir(), "org-enrollment.json")
	_, first := enrollTestKey(t, 1)
	_, second := enrollTestKey(t, 200)

	if rc, _, stderr := runEnrollCapture(t, "--path", store,
		"--org", "https://good.example/fak", "--root-key", first, "--device", "node-a"); rc != 0 {
		t.Fatalf("first enroll rc = %d, want 0 (stderr: %s)", rc, stderr)
	}
	before, err := os.ReadFile(store)
	if err != nil {
		t.Fatalf("read store: %v", err)
	}

	rc, stdout, stderr := runEnrollCapture(t, "--path", store,
		"--org", "https://evil.example/fak", "--root-key", second, "--device", "node-a")
	if rc != 1 {
		t.Fatalf("re-pin rc = %d, want 1 (a refusal must not read as success)", rc)
	}
	if !strings.Contains(stderr, "already enrolled") || !strings.Contains(stderr, "--revoke") {
		t.Errorf("refusal does not name the state or the cure:\n%s", stderr)
	}
	if strings.Contains(stdout, "enrolled:") {
		t.Errorf("refusal still printed a success line on stdout:\n%s", stdout)
	}

	after, err := os.ReadFile(store)
	if err != nil {
		t.Fatalf("read store after refusal: %v", err)
	}
	if !bytes.Equal(before, after) {
		t.Fatalf("refused re-pin still mutated the store:\nbefore=%s\nafter=%s", before, after)
	}

	// Re-enrolling with the SAME facts is idempotent, not a re-pin: a config-management
	// loop that runs this verb every hour must not start refusing on the second hour.
	if rc, _, stderr := runEnrollCapture(t, "--path", store,
		"--org", "https://good.example/fak", "--root-key", first, "--device", "node-a"); rc != 0 {
		t.Fatalf("idempotent re-enroll rc = %d, want 0 (stderr: %s)", rc, stderr)
	}

	// --revoke is the sanctioned route, and it must actually open it.
	if rc, _, stderr := runEnrollCapture(t, "--path", store, "--revoke"); rc != 0 {
		t.Fatalf("revoke rc = %d, want 0 (stderr: %s)", rc, stderr)
	}
	if rc, _, stderr := runEnrollCapture(t, "--path", store,
		"--org", "https://evil.example/fak", "--root-key", second, "--device", "node-a"); rc != 0 {
		t.Fatalf("enroll after revoke rc = %d, want 0 (stderr: %s)", rc, stderr)
	}
}

func TestEnrollRequiresAnExplicitRootKey(t *testing.T) {
	store := filepath.Join(t.TempDir(), "org-enrollment.json")

	rc, _, stderr := runEnrollCapture(t, "--path", store, "--org", "https://acme.example/fak")
	if rc != 1 {
		t.Fatalf("keyless enroll rc = %d, want 1", rc)
	}
	if !strings.Contains(stderr, "--root-key is required") || !strings.Contains(stderr, "#5321") {
		t.Errorf("refusal does not name the missing flag and the gap that owns it:\n%s", stderr)
	}
	// Nothing may reach disk: a store with no usable anchor would report the box as
	// enrolled while authenticating nothing.
	if _, err := os.Stat(store); !os.IsNotExist(err) {
		t.Fatalf("keyless enroll wrote a store anyway (stat err = %v)", err)
	}
}

func TestEnrollRefusesUnusableRootKey(t *testing.T) {
	dir := t.TempDir()
	store := filepath.Join(dir, "org-enrollment.json")
	short := filepath.Join(dir, "short.key")
	if err := os.WriteFile(short, []byte(base64.StdEncoding.EncodeToString([]byte("too-short"))), 0o600); err != nil {
		t.Fatalf("write key file: %v", err)
	}

	for name, key := range map[string]string{
		"not base64":  "!!!!not-base64!!!!",
		"wrong size":  short,
		"empty file":  filepath.Join(dir, "missing.key"),
		"zero length": "",
	} {
		t.Run(name, func(t *testing.T) {
			rc, _, stderr := runEnrollCapture(t, "--path", store,
				"--org", "https://acme.example/fak", "--root-key", key)
			if rc != 1 {
				t.Fatalf("rc = %d, want 1 (stderr: %s)", rc, stderr)
			}
			if _, err := os.Stat(store); !os.IsNotExist(err) {
				t.Fatalf("bad key still produced a store (stat err = %v)", err)
			}
		})
	}
}

// TestEnrollStatusSaysInertNotSilent guards the property the whole opt-in design rests on:
// an un-enrolled box is INERT, and it says so. Silence would be indistinguishable from a
// verb that crashed before printing.
func TestEnrollStatusSaysInertNotSilent(t *testing.T) {
	store := filepath.Join(t.TempDir(), "org-enrollment.json")

	rc, stdout, stderr := runEnrollCapture(t, "--path", store)
	if rc != 0 {
		t.Fatalf("status rc = %d, want 0 — not enrolled is a posture, not an error (stderr: %s)", rc, stderr)
	}
	for _, want := range []string{"not enrolled", "INERT", "fak enroll --org"} {
		if !strings.Contains(stdout, want) {
			t.Errorf("status stdout missing %q:\n%s", want, stdout)
		}
	}
}

// TestEnrollStatusRefusesDamagedStore is the fail-closed line. A store that will not load
// must NOT be reported as "not enrolled" — that is the exact shape where a check's absence
// renders identically to a check that passed (epic #5601).
func TestEnrollStatusRefusesDamagedStore(t *testing.T) {
	dir := t.TempDir()
	_, b64 := enrollTestKey(t, 3)

	cases := map[string]func(t *testing.T, store string){
		"garbage": func(t *testing.T, store string) {
			if err := os.WriteFile(store, []byte("{not json"), 0o600); err != nil {
				t.Fatalf("write: %v", err)
			}
		},
		"tampered sum": func(t *testing.T, store string) {
			if rc, _, se := runEnrollCapture(t, "--path", store,
				"--org", "https://acme.example/fak", "--root-key", b64, "--device", "node-a"); rc != 0 {
				t.Fatalf("seed enroll rc = %d (%s)", rc, se)
			}
			b, err := os.ReadFile(store)
			if err != nil {
				t.Fatalf("read: %v", err)
			}
			var raw map[string]any
			if err := json.Unmarshal(b, &raw); err != nil {
				t.Fatalf("decode: %v", err)
			}
			raw["org_url"] = "https://attacker.example/fak" // sum now stale
			edited, err := json.Marshal(raw)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			if err := os.WriteFile(store, edited, 0o600); err != nil {
				t.Fatalf("write: %v", err)
			}
		},
	}

	for name, seed := range cases {
		t.Run(name, func(t *testing.T) {
			store := filepath.Join(dir, strings.ReplaceAll(name, " ", "-")+".json")
			seed(t, store)

			rc, stdout, stderr := runEnrollCapture(t, "--path", store)
			if rc != 1 {
				t.Fatalf("damaged-store status rc = %d, want 1", rc)
			}
			if strings.Contains(stdout, "not enrolled") {
				t.Errorf("damaged store rendered as not-enrolled — that is the fail-open:\n%s", stdout)
			}
			if !strings.Contains(stderr, "--revoke") {
				t.Errorf("refusal does not name the cure:\n%s", stderr)
			}

			// Enrolling over a store we could not read must be refused too, or a
			// corrupted anchor could be laundered into a fresh valid one.
			if rc, _, _ := runEnrollCapture(t, "--path", store,
				"--org", "https://acme.example/fak", "--root-key", b64); rc != 1 {
				t.Errorf("enroll over a damaged store rc = %d, want 1", rc)
			}
			// --revoke is the documented cure and must work on a store that will not load.
			if rc, _, se := runEnrollCapture(t, "--path", store, "--revoke"); rc != 0 {
				t.Fatalf("revoke of a damaged store rc = %d, want 0 (%s)", rc, se)
			}
			if _, err := os.Stat(store); !os.IsNotExist(err) {
				t.Fatalf("revoke left the damaged store behind (stat err = %v)", err)
			}
		})
	}
}

func TestEnrollRevokeIsIdempotentAndSaysSo(t *testing.T) {
	store := filepath.Join(t.TempDir(), "org-enrollment.json")

	rc, stdout, stderr := runEnrollCapture(t, "--path", store, "--revoke")
	if rc != 0 {
		t.Fatalf("revoke-when-absent rc = %d, want 0 (stderr: %s)", rc, stderr)
	}
	if !strings.Contains(stdout, "nothing to revoke") {
		t.Errorf("revoke-when-absent was silent about the no-op:\n%s", stdout)
	}

	_, b64 := enrollTestKey(t, 11)
	if rc, _, se := runEnrollCapture(t, "--path", store,
		"--org", "https://acme.example/fak", "--root-key", b64, "--device", "node-a"); rc != 0 {
		t.Fatalf("enroll rc = %d (%s)", rc, se)
	}
	rc, stdout, stderr = runEnrollCapture(t, "--path", store, "--revoke")
	if rc != 0 {
		t.Fatalf("revoke rc = %d, want 0 (stderr: %s)", rc, stderr)
	}
	if !strings.Contains(stdout, "INERT") {
		t.Errorf("revoke did not report the resulting posture:\n%s", stdout)
	}
	if _, err := os.Stat(store); !os.IsNotExist(err) {
		t.Fatalf("store survived revoke (stat err = %v)", err)
	}
}

// TestEnrollJSONKeysAreAlwaysPresent pins the #5299 house rule: a consumer must never have
// to read an ABSENT key as a state. Both postures carry the same key set.
func TestEnrollJSONKeysAreAlwaysPresent(t *testing.T) {
	store := filepath.Join(t.TempDir(), "org-enrollment.json")
	_, b64 := enrollTestKey(t, 23)

	decode := func(t *testing.T, s string) map[string]any {
		t.Helper()
		var m map[string]any
		if err := json.Unmarshal([]byte(s), &m); err != nil {
			t.Fatalf("stdout is not JSON (%v):\n%s", err, s)
		}
		return m
	}
	keys := []string{"store", "enrolled", "org_url", "issuer", "device_id", "user", "groups", "enrolled_at", "key_fingerprints"}

	rc, stdout, stderr := runEnrollCapture(t, "--path", store, "--json")
	if rc != 0 {
		t.Fatalf("status --json rc = %d (stderr: %s)", rc, stderr)
	}
	absent := decode(t, stdout)
	for _, k := range keys {
		if _, ok := absent[k]; !ok {
			t.Errorf("not-enrolled JSON is missing key %q: %v", k, absent)
		}
	}
	if absent["enrolled"] != false {
		t.Errorf("not-enrolled JSON says enrolled=%v", absent["enrolled"])
	}
	if got, ok := absent["groups"].([]any); !ok || got == nil {
		t.Errorf("groups must be [] and never null when absent: %#v", absent["groups"])
	}

	if rc, _, se := runEnrollCapture(t, "--path", store,
		"--org", "https://acme.example/fak", "--root-key", b64, "--device", "node-a"); rc != 0 {
		t.Fatalf("enroll rc = %d (%s)", rc, se)
	}
	rc, stdout, stderr = runEnrollCapture(t, "--path", store, "--json")
	if rc != 0 {
		t.Fatalf("enrolled status --json rc = %d (stderr: %s)", rc, stderr)
	}
	present := decode(t, stdout)
	for _, k := range keys {
		if _, ok := present[k]; !ok {
			t.Errorf("enrolled JSON is missing key %q: %v", k, present)
		}
	}
	if present["enrolled"] != true {
		t.Errorf("enrolled JSON says enrolled=%v", present["enrolled"])
	}
	fps, _ := present["key_fingerprints"].([]any)
	if len(fps) != 1 {
		t.Fatalf("key_fingerprints = %#v, want exactly one", present["key_fingerprints"])
	}
	if fp, _ := fps[0].(string); !strings.HasPrefix(fp, "sha256:") || strings.Contains(fp, b64) {
		t.Errorf("fingerprint is not a sha256 digest of the key: %q", fp)
	}
}

// TestEnrollPathHonoursTheEnvOverride keeps the CLI on the library's path resolution rather
// than a second, drifting copy of it.
func TestEnrollPathHonoursTheEnvOverride(t *testing.T) {
	store := filepath.Join(t.TempDir(), "env-store.json")
	t.Setenv("FAK_ORG_ENROLLMENT_PATH", store)
	_, b64 := enrollTestKey(t, 31)

	if rc, _, se := runEnrollCapture(t, "--org", "https://acme.example/fak",
		"--root-key", b64, "--device", "node-a"); rc != 0 {
		t.Fatalf("enroll rc = %d (%s)", rc, se)
	}
	if _, err := os.Stat(store); err != nil {
		t.Fatalf("enroll did not honour the env override: %v", err)
	}
	rc, stdout, _ := runEnrollCapture(t)
	if rc != 0 || !strings.Contains(stdout, store) {
		t.Errorf("status did not resolve to the overridden store (rc=%d):\n%s", rc, stdout)
	}
}

func TestEnrollUsageErrorIsExitTwo(t *testing.T) {
	rc, _, _ := runEnrollCapture(t, "--no-such-flag")
	if rc != 2 {
		t.Fatalf("unknown flag rc = %d, want 2 (usage), distinct from 1 (refusal)", rc)
	}
}

func TestOrgURLHost(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"https://policy.acme.example/fak", "policy.acme.example"},
		{"https://policy.acme.example:8443/fak", "policy.acme.example"},
		{"http://user:pw@policy.acme.example/fak?x=1", "policy.acme.example"},
		{"policy.acme.example", "policy.acme.example"},
		{"https://[2001:db8::1]:8443/fak", "2001:db8::1"},
		{"", ""},
	} {
		if got := orgURLHost(tc.in); got != tc.want {
			t.Errorf("orgURLHost(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
