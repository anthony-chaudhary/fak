// policy_org_check_test.go pins the CLI seam that issue #5320 opened onto the
// `fak-org-policy/v1` signed envelope: `fak policy --check <envelope.json>`.
//
// The envelope and its verifier (internal/policy/orgenvelope.go) are already proven
// in their own package. What is proven HERE is the part only the command surface can
// break: that the verb ROUTES an envelope to policy.VerifyEnvelope at all, that it
// routes on the file's schema shape rather than its name, that the operator's
// --org-key is the only trust anchor it will use, that the plain-manifest path it has
// always had is untouched, and  -  the load-bearing one  -  that every refusal comes
// back as an error with a closed-vocabulary reason and NO rendered floor. A CLI that
// printed the wrapped floor beside a failed signature would be the exact fail-open the
// envelope exists to prevent, and it would be invisible to any test of the verifier.
package main

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
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// The fixed clock the window checks are made against, plus a window that brackets it.
// Fixed rather than time.Now()-relative so a freshness refusal is a property of the
// table, not of when the suite happened to run.
const (
	orgTestNow       = int64(1_750_000_000)
	orgTestNotBefore = int64(1_700_000_000)
	orgTestExpires   = int64(1_800_000_000)
)

// orgTestManifest is the wrapped `fak-policy/v1` body. Two exact allows make the
// rendered floor identifiable in the report without depending on the whole summary.
const orgTestManifest = `{"version":"fak-policy/v1","allow":["Read","Grep"]}`

// orgTestKeys mints a throwaway org root keypair. Every key in this file is generated
// in-process: no fixture on disk, no network, and nothing that could outlive the test.
func orgTestKeys(t *testing.T) (ed25519.PublicKey, ed25519.PrivateKey) {
	t.Helper()
	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("generate org root key: %v", err)
	}
	return pub, priv
}

// orgTestEnvelope is a well-formed, in-window envelope carrying orgTestManifest. Each
// test mutates the one field it is about, so the field under test is the only
// difference from a case that is known to verify.
func orgTestEnvelope() policy.OrgEnvelope {
	return policy.OrgEnvelope{
		Issuer:    "acme-security",
		Version:   7,
		NotBefore: orgTestNotBefore,
		Expires:   orgTestExpires,
		Target:    "fleet/laptops",
		Body:      json.RawMessage(orgTestManifest),
	}
}

// writeOrgEnvelope signs e with priv and writes the transmittable JSON under name in
// dir, returning the path. name is a parameter because filename-independence is itself
// one of the properties under test.
func writeOrgEnvelope(t *testing.T, dir, name string, e policy.OrgEnvelope, priv ed25519.PrivateKey) string {
	t.Helper()
	signed, err := policy.SignEnvelope(e, priv)
	if err != nil {
		t.Fatalf("sign envelope: %v", err)
	}
	raw, err := signed.Marshal()
	if err != nil {
		t.Fatalf("marshal envelope: %v", err)
	}
	return writeOrgFile(t, dir, name, string(raw))
}

func writeOrgFile(t *testing.T, dir, name, body string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
	return path
}

// orgTestOpts is what the verb assembles from --org-key plus the process clock and
// build version: the pinned root key, a clock inside the fixture window, and a running
// version new enough that min_version is not the thing under test.
func orgTestOpts(pub ed25519.PublicKey) orgCheckOptions {
	return orgCheckOptions{
		Key:            base64.StdEncoding.EncodeToString(pub),
		Now:            time.Unix(orgTestNow, 0),
		RunningVersion: "9.9.9",
	}
}

// wantEnvelopeRefusal asserts the whole shape of a fail-closed outcome: an error, NO
// report (an unverified floor must never reach stdout), the exact closed-vocabulary
// Reason and Detail, and a message that actually names the reason  -  that message is
// what `fak policy` prints to stderr before exiting non-zero.
func wantEnvelopeRefusal(t *testing.T, report string, err error, reason abi.ReasonCode, detail string) {
	t.Helper()
	if err == nil {
		t.Fatalf("want a refusal, got a report:\n%s", report)
	}
	if report != "" {
		t.Fatalf("a refused envelope must render no floor, got:\n%s", report)
	}
	var ee *policy.EnvelopeError
	if !errors.As(err, &ee) {
		t.Fatalf("want *policy.EnvelopeError, got %T: %v", err, err)
	}
	if ee.Reason != reason || ee.Detail != detail {
		t.Fatalf("want %s(%s), got %s(%s)", abi.ReasonName(reason), detail, abi.ReasonName(ee.Reason), ee.Detail)
	}
	if !strings.Contains(err.Error(), abi.ReasonName(reason)) {
		t.Fatalf("stderr text must name the closed-vocabulary reason %s, got %q", abi.ReasonName(reason), err.Error())
	}
}

// TestPolicyCheckVerifiesOrgEnvelopeAndPrintsTheWrappedFloor is the positive case: a
// correctly signed, in-window envelope verifies and the report carries the provenance
// an operator needs to decide whether to apply it (issuer, monotonic version, validity
// window, target) followed by the floor itself.
func TestPolicyCheckVerifiesOrgEnvelopeAndPrintsTheWrappedFloor(t *testing.T) {
	pub, priv := orgTestKeys(t)
	path := writeOrgEnvelope(t, t.TempDir(), "org.json", orgTestEnvelope(), priv)

	report, err := checkPolicyFile(path, orgTestOpts(pub))
	if err != nil {
		t.Fatalf("a well-formed envelope must verify, got: %v", err)
	}
	for _, want := range []string{
		policy.OrgEnvelopeVersion,
		"issuer             : acme-security",
		"envelope version   : 7",
		"target             : fleet/laptops",
		"2023-11-14T22:13:20Z", // not_before, rendered in UTC
		"2027-01-15T08:00:00Z", // expires
		"allow (exact)      : 2 tool(s)",
	} {
		if !strings.Contains(report, want) {
			t.Fatalf("report is missing %q:\n%s", want, report)
		}
	}
}

// TestPolicyCheckRefusesOrgEnvelopeSignedByAnotherKey is the authenticity gate. The
// envelope is structurally perfect and inside its window; only the signing key is
// foreign. Verifying it against the org root key must refuse TRUST_VIOLATION rather
// than fall back to "well, the body parses".
func TestPolicyCheckRefusesOrgEnvelopeSignedByAnotherKey(t *testing.T) {
	pub, _ := orgTestKeys(t)
	_, impostor := orgTestKeys(t)
	path := writeOrgEnvelope(t, t.TempDir(), "org.json", orgTestEnvelope(), impostor)

	report, err := checkPolicyFile(path, orgTestOpts(pub))
	wantEnvelopeRefusal(t, report, err, abi.ReasonTrustViolation, "signature")
}

// TestPolicyCheckRefusesOrgEnvelopeFromAnotherIssuer covers the other half of auth:
// --org-issuer pins WHO may issue, so a validly signed envelope naming a different
// issuer is still refused.
func TestPolicyCheckRefusesOrgEnvelopeFromAnotherIssuer(t *testing.T) {
	pub, priv := orgTestKeys(t)
	e := orgTestEnvelope()
	e.Issuer = "acme-marketing"
	path := writeOrgEnvelope(t, t.TempDir(), "org.json", e, priv)

	opts := orgTestOpts(pub)
	opts.ExpectedIssuer = "acme-security"
	report, err := checkPolicyFile(path, opts)
	wantEnvelopeRefusal(t, report, err, abi.ReasonTrustViolation, "issuer")
}

// TestPolicyCheckRefusesExpiredOrgEnvelope is the freshness gate on the late side: a
// genuinely-signed envelope whose window has closed cannot be witnessed as current, so
// it must not be allowed to set the floor.
func TestPolicyCheckRefusesExpiredOrgEnvelope(t *testing.T) {
	pub, priv := orgTestKeys(t)
	e := orgTestEnvelope()
	e.Expires = orgTestNow - 1
	path := writeOrgEnvelope(t, t.TempDir(), "org.json", e, priv)

	report, err := checkPolicyFile(path, orgTestOpts(pub))
	wantEnvelopeRefusal(t, report, err, abi.ReasonUnwitnessed, "expired")
}

// TestPolicyCheckRefusesNotYetValidOrgEnvelope is the freshness gate on the early
// side  -  the pre-dated envelope, staged before its window opens.
func TestPolicyCheckRefusesNotYetValidOrgEnvelope(t *testing.T) {
	pub, priv := orgTestKeys(t)
	e := orgTestEnvelope()
	e.NotBefore = orgTestNow + 1
	path := writeOrgEnvelope(t, t.TempDir(), "org.json", e, priv)

	report, err := checkPolicyFile(path, orgTestOpts(pub))
	wantEnvelopeRefusal(t, report, err, abi.ReasonUnwitnessed, "not_before")
}

// TestPolicyCheckRefusesRolledBackOrgEnvelope is the anti-rollback gate. The replayed
// envelope is authentic and in-window  -  that is exactly what makes the attack work  -
// but its version is below what this local has already accepted, so re-admitting it
// could only ever widen the floor back to an older, more permissive one.
func TestPolicyCheckRefusesRolledBackOrgEnvelope(t *testing.T) {
	pub, priv := orgTestKeys(t)
	e := orgTestEnvelope()
	e.Version = 3
	path := writeOrgEnvelope(t, t.TempDir(), "org.json", e, priv)

	opts := orgTestOpts(pub)
	opts.SeenVersion = 7
	report, err := checkPolicyFile(path, opts)
	wantEnvelopeRefusal(t, report, err, abi.ReasonUnwitnessed, "rollback")
}

// TestPolicyCheckRefusesOrgEnvelopeBlockedByMinVersion pins the applicability gate the
// report's "min fak version" line describes: a binary older than the envelope demands
// is blocked from applying it.
func TestPolicyCheckRefusesOrgEnvelopeBlockedByMinVersion(t *testing.T) {
	pub, priv := orgTestKeys(t)
	e := orgTestEnvelope()
	e.MinVersion = "9.9.10"
	path := writeOrgEnvelope(t, t.TempDir(), "org.json", e, priv)

	report, err := checkPolicyFile(path, orgTestOpts(pub))
	wantEnvelopeRefusal(t, report, err, abi.ReasonPolicyBlock, "min_version")
}

// TestPolicyCheckOrgEnvelopeWithoutKeyRefusesRatherThanSkipsTheCheck is the fail-loud
// case that has no verifier-side twin: with no --org-key there is no trust anchor at
// all. The verb must say so and refuse, never quietly print the wrapped floor as if an
// unsigned read were a verification.
func TestPolicyCheckOrgEnvelopeWithoutKeyRefusesRatherThanSkipsTheCheck(t *testing.T) {
	_, priv := orgTestKeys(t)
	path := writeOrgEnvelope(t, t.TempDir(), "org.json", orgTestEnvelope(), priv)

	report, err := checkPolicyFile(path, orgCheckOptions{Now: time.Unix(orgTestNow, 0), RunningVersion: "9.9.9"})
	if err == nil {
		t.Fatalf("a keyless envelope check must refuse, got a report:\n%s", report)
	}
	if report != "" {
		t.Fatalf("a keyless envelope check must render no floor, got:\n%s", report)
	}
	if !strings.Contains(err.Error(), "--org-key") {
		t.Fatalf("the refusal must name the missing flag, got %q", err.Error())
	}
}

// TestPolicyCheckRejectsAWrongSizedOrgKey pins the other trust-anchor failure: a key
// that decodes but is not an Ed25519 public key is refused before verification, not
// padded or truncated into one.
func TestPolicyCheckRejectsAWrongSizedOrgKey(t *testing.T) {
	_, priv := orgTestKeys(t)
	path := writeOrgEnvelope(t, t.TempDir(), "org.json", orgTestEnvelope(), priv)

	opts := orgTestOpts(nil)
	opts.Key = base64.StdEncoding.EncodeToString([]byte("too-short"))
	report, err := checkPolicyFile(path, opts)
	if err == nil {
		t.Fatalf("a wrong-sized org key must refuse, got a report:\n%s", report)
	}
	if !strings.Contains(err.Error(), "Ed25519 public key") {
		t.Fatalf("the refusal must explain the key size, got %q", err.Error())
	}
}

// TestPolicyCheckAcceptsAnOrgKeyFromAFile covers the ergonomic half of --org-key: the
// operator keeps the root key in a file rather than pasting 44 base64 characters onto
// a command line, and the same envelope verifies either way.
func TestPolicyCheckAcceptsAnOrgKeyFromAFile(t *testing.T) {
	dir := t.TempDir()
	pub, priv := orgTestKeys(t)
	path := writeOrgEnvelope(t, dir, "org.json", orgTestEnvelope(), priv)
	keyPath := writeOrgFile(t, dir, "org-root.pub", base64.StdEncoding.EncodeToString(pub)+"\n")

	opts := orgTestOpts(pub)
	opts.Key = keyPath
	report, err := checkPolicyFile(path, opts)
	if err != nil {
		t.Fatalf("an org key read from a file must verify the same envelope, got: %v", err)
	}
	if !strings.Contains(report, "issuer             : acme-security") {
		t.Fatalf("report is missing the verified provenance:\n%s", report)
	}
}

// TestPolicyCheckPlainManifestKeepsTheManifestPath is the no-regression witness: a
// plain `fak-policy/v1` manifest still takes policy.LoadRuntime and still prints the
// exact line this verb has always printed  -  no key, no envelope machinery, no change.
func TestPolicyCheckPlainManifestKeepsTheManifestPath(t *testing.T) {
	path := writeOrgFile(t, t.TempDir(), "floor.json", orgTestManifest)

	report, err := checkPolicyFile(path, orgCheckOptions{})
	if err != nil {
		t.Fatalf("a plain manifest must still validate: %v", err)
	}
	if !strings.Contains(report, "(manifest valid; every deny cites a closed-vocabulary reason)") {
		t.Fatalf("the manifest report changed:\n%s", report)
	}
	if !strings.Contains(report, "allow (exact)      : 2 tool(s)") {
		t.Fatalf("the manifest floor summary is missing:\n%s", report)
	}
}

// TestPolicyCheckInvalidManifestStillFailsLoud proves the envelope arm did not swallow
// the manifest arm's errors: a body citing a reason outside the closed vocabulary is
// still a hard failure with no floor printed.
func TestPolicyCheckInvalidManifestStillFailsLoud(t *testing.T) {
	path := writeOrgFile(t, t.TempDir(), "floor.json", `{"version":"fak-policy/v1","deny":{"rm":"NOT_A_REASON"}}`)

	report, err := checkPolicyFile(path, orgCheckOptions{})
	if err == nil {
		t.Fatalf("an unknown deny reason must refuse, got a report:\n%s", report)
	}
	if report != "" {
		t.Fatalf("an invalid manifest must render no floor, got:\n%s", report)
	}
}

// TestPolicyCheckMissingFileFailsLoud keeps the oldest behaviour of all: an absent
// --check target is an error, not an empty floor.
func TestPolicyCheckMissingFileFailsLoud(t *testing.T) {
	path := filepath.Join(t.TempDir(), "absent.json")

	if report, err := checkPolicyFile(path, orgCheckOptions{}); err == nil {
		t.Fatalf("a missing manifest must refuse, got a report:\n%s", report)
	}
}

// TestPolicyCheckRoutesByShapeNotFilename is the routing contract stated directly: the
// same bytes must be treated as an envelope no matter what the file is called, and a
// manifest called "envelope.json" must NOT be pushed down the signed path. A filename
// is not a trust input, so it must not select a verifier.
func TestPolicyCheckRoutesByShapeNotFilename(t *testing.T) {
	dir := t.TempDir()
	pub, priv := orgTestKeys(t)

	// An envelope wearing a manifest's name still gets verified: with no key supplied
	// it refuses for a MISSING KEY, which only the envelope arm can say.
	envAsManifest := writeOrgEnvelope(t, dir, "policy-manifest.json", orgTestEnvelope(), priv)
	if _, err := checkPolicyFile(envAsManifest, orgCheckOptions{}); err == nil || !strings.Contains(err.Error(), "--org-key") {
		t.Fatalf("an envelope named like a manifest must still take the envelope path, got %v", err)
	}
	// ... and with the key it verifies.
	if _, err := checkPolicyFile(envAsManifest, orgTestOpts(pub)); err != nil {
		t.Fatalf("an envelope named like a manifest must verify: %v", err)
	}

	// A manifest wearing an envelope's name still takes the manifest path.
	manifestAsEnv := writeOrgFile(t, dir, "org-envelope.json", orgTestManifest)
	report, err := checkPolicyFile(manifestAsEnv, orgTestOpts(pub))
	if err != nil {
		t.Fatalf("a manifest named like an envelope must still validate as a manifest: %v", err)
	}
	if !strings.Contains(report, "(manifest valid;") {
		t.Fatalf("a manifest named like an envelope took the wrong path:\n%s", report)
	}
}

// TestPolicyIsOrgEnvelopeDiscriminatesTheTwoSchemas pins the sniff itself at the byte
// level, including the near misses: the signature pair alone is not enough (a manifest
// never has one, but a truncated envelope might have only half), and the manifest's
// STRING `version` schema tag is what keeps the two apart.
func TestPolicyIsOrgEnvelopeDiscriminatesTheTwoSchemas(t *testing.T) {
	for _, tc := range []struct {
		name string
		raw  string
		want bool
	}{
		{"plain manifest", orgTestManifest, false},
		{"empty manifest", `{}`, false},
		{"envelope", `{"issuer":"a","alg":"ed25519","sig":"AA==","version":4,"body":{}}`, true},
		{"envelope without a version counter", `{"issuer":"a","alg":"ed25519","sig":"AA==","body":{}}`, true},
		{"signature without an alg", `{"sig":"AA==","version":4}`, false},
		{"alg without a signature", `{"alg":"ed25519","version":4}`, false},
		{"manifest schema tag wins over a stray sig", `{"version":"fak-policy/v1","alg":"ed25519","sig":"AA=="}`, false},
		{"not JSON at all", `not json`, false},
		{"a JSON array", `[]`, false},
	} {
		if got := isOrgEnvelope([]byte(tc.raw)); got != tc.want {
			t.Errorf("isOrgEnvelope(%s) = %v, want %v", tc.name, got, tc.want)
		}
	}
}
