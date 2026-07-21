package policy

import (
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// testKey derives a DETERMINISTIC Ed25519 keypair from a fixed 32-byte seed, so
// the whole suite is reproducible and never touches a random source or the clock
// (the verifier takes `now` as a parameter). Two distinct seeds give two distinct
// orgs, used for the wrong-key case.
func testKey(t *testing.T, seedByte byte) (ed25519.PublicKey, ed25519.PrivateKey) {
	t.Helper()
	seed := make([]byte, ed25519.SeedSize)
	for i := range seed {
		seed[i] = seedByte
	}
	priv := ed25519.NewKeyFromSeed(seed)
	return priv.Public().(ed25519.PublicKey), priv
}

// baseBody is a small but non-trivial, valid fak-policy/v1 manifest body.
func baseBody(t *testing.T) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(Manifest{
		Version: Version,
		Allow:   []string{"search_web"},
		Deny:    map[string]string{"delete_account": "POLICY_BLOCK"},
	})
	if err != nil {
		t.Fatalf("marshal body: %v", err)
	}
	return b
}

// fixedNow is the clock every test passes in — well inside the default window.
var fixedNow = time.Unix(1_700_000_000, 0)

// goodEnvelope builds and signs a valid envelope with the given signing key.
func goodEnvelope(t *testing.T, priv ed25519.PrivateKey) OrgEnvelope {
	t.Helper()
	e := OrgEnvelope{
		Issuer:     "acme-corp",
		Version:    5,
		NotBefore:  fixedNow.Unix() - 3600,
		Expires:    fixedNow.Unix() + 3600,
		MinVersion: "0.5.0",
		Target:     "group:eng",
		Body:       baseBody(t),
	}
	signed, err := SignEnvelope(e, priv)
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	return signed
}

func defaultOpts(pub ed25519.PublicKey) VerifyOptions {
	return VerifyOptions{
		RootPublicKey:      pub,
		ExpectedIssuer:     "acme-corp",
		Now:                fixedNow,
		HighestSeenVersion: 5,
		RunningVersion:     "0.9.0",
	}
}

func marshal(t *testing.T, e OrgEnvelope) []byte {
	t.Helper()
	raw, err := e.Marshal()
	if err != nil {
		t.Fatalf("marshal envelope: %v", err)
	}
	return raw
}

// assertReason asserts the verify error is an *EnvelopeError with the wanted
// closed-vocabulary reason and detail token.
func assertReason(t *testing.T, err error, want abi.ReasonCode, wantDetail string) {
	t.Helper()
	if err == nil {
		t.Fatalf("expected failure with %s/%s, got nil (verified)", abi.ReasonName(want), wantDetail)
	}
	var ee *EnvelopeError
	if !errors.As(err, &ee) {
		t.Fatalf("error is not *EnvelopeError: %T %v", err, err)
	}
	if ee.Reason != want {
		t.Fatalf("reason = %s, want %s (detail %q)", abi.ReasonName(ee.Reason), abi.ReasonName(want), ee.Detail)
	}
	if wantDetail != "" && ee.Detail != wantDetail {
		t.Fatalf("detail = %q, want %q (reason %s)", ee.Detail, wantDetail, abi.ReasonName(ee.Reason))
	}
}

func TestOrgEnvelopeGoodVerifies(t *testing.T) {
	pub, priv := testKey(t, 1)
	raw := marshal(t, goodEnvelope(t, priv))

	v, err := VerifyEnvelope(raw, defaultOpts(pub))
	if err != nil {
		t.Fatalf("good envelope failed to verify: %v", err)
	}
	if v.Envelope.Issuer != "acme-corp" {
		t.Fatalf("issuer = %q", v.Envelope.Issuer)
	}
	if v.Envelope.Version != 5 {
		t.Fatalf("version = %d", v.Envelope.Version)
	}
	// The inner manifest resolved through the standard loader.
	if !v.Runtime.Adjudicator.Allow["search_web"] {
		t.Fatalf("inner floor did not allow search_web: %+v", v.Runtime.Adjudicator.Allow)
	}
	if _, ok := v.Runtime.Adjudicator.Deny["delete_account"]; !ok {
		t.Fatalf("inner floor did not deny delete_account")
	}
}

func TestOrgEnvelopeRoundTripSameVersionAccepted(t *testing.T) {
	// Re-verifying the SAME version as highest-seen is not a rollback.
	pub, priv := testKey(t, 1)
	raw := marshal(t, goodEnvelope(t, priv))
	opts := defaultOpts(pub)
	opts.HighestSeenVersion = 5 // equal to envelope version
	if _, err := VerifyEnvelope(raw, opts); err != nil {
		t.Fatalf("equal-version envelope should verify: %v", err)
	}
}

func TestOrgEnvelopeBadSignature(t *testing.T) {
	pub, priv := testKey(t, 1)
	e := goodEnvelope(t, priv)
	// Flip the signature to a valid-length but wrong value.
	sig, _ := base64.StdEncoding.DecodeString(e.Sig)
	sig[0] ^= 0xFF
	e.Sig = base64.StdEncoding.EncodeToString(sig)
	assertReason(t, mustVerifyErr(t, marshal(t, e), defaultOpts(pub)), abi.ReasonTrustViolation, "signature")
}

func TestOrgEnvelopeWrongKey(t *testing.T) {
	// Signed by org A, verified against org B's root key → signature fails.
	_, privA := testKey(t, 1)
	pubB, _ := testKey(t, 2)
	raw := marshal(t, goodEnvelope(t, privA))
	assertReason(t, mustVerifyErr(t, raw, defaultOpts(pubB)), abi.ReasonTrustViolation, "signature")
}

func TestOrgEnvelopeWrongIssuer(t *testing.T) {
	// Correctly signed, but the claimed issuer is not the expected one.
	pub, priv := testKey(t, 1)
	e := goodEnvelope(t, priv)
	e.Issuer = "evil-corp"
	signed, err := SignEnvelope(e, priv) // re-sign so the signature is valid over the wrong issuer
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	assertReason(t, mustVerifyErr(t, marshal(t, signed), defaultOpts(pub)), abi.ReasonTrustViolation, "issuer")
}

func TestOrgEnvelopeExpired(t *testing.T) {
	pub, priv := testKey(t, 1)
	raw := marshal(t, goodEnvelope(t, priv))
	opts := defaultOpts(pub)
	opts.Now = fixedNow.Add(48 * time.Hour) // past Expires
	assertReason(t, mustVerifyErr(t, raw, opts), abi.ReasonUnwitnessed, "expired")
}

func TestOrgEnvelopeNotYetValid(t *testing.T) {
	pub, priv := testKey(t, 1)
	raw := marshal(t, goodEnvelope(t, priv))
	opts := defaultOpts(pub)
	opts.Now = fixedNow.Add(-48 * time.Hour) // before NotBefore
	assertReason(t, mustVerifyErr(t, raw, opts), abi.ReasonUnwitnessed, "not_before")
}

func TestOrgEnvelopeRolledBackVersion(t *testing.T) {
	pub, priv := testKey(t, 1)
	raw := marshal(t, goodEnvelope(t, priv)) // version 5
	opts := defaultOpts(pub)
	opts.HighestSeenVersion = 9 // we have already accepted a newer version
	assertReason(t, mustVerifyErr(t, raw, opts), abi.ReasonUnwitnessed, "rollback")
}

func TestOrgEnvelopeMinVersionTooHigh(t *testing.T) {
	pub, priv := testKey(t, 1)
	e := goodEnvelope(t, priv)
	e.MinVersion = "2.0.0"
	signed, _ := SignEnvelope(e, priv)
	opts := defaultOpts(pub)
	opts.RunningVersion = "1.4.0" // older than min_version
	assertReason(t, mustVerifyErr(t, marshal(t, signed), opts), abi.ReasonPolicyBlock, "min_version")
}

func TestOrgEnvelopeMinVersionUnparseableFailsClosed(t *testing.T) {
	pub, priv := testKey(t, 1)
	e := goodEnvelope(t, priv)
	e.MinVersion = "1.0.0"
	signed, _ := SignEnvelope(e, priv)
	opts := defaultOpts(pub)
	opts.RunningVersion = "dev" // cannot prove it satisfies min_version → closed
	assertReason(t, mustVerifyErr(t, marshal(t, signed), opts), abi.ReasonPolicyBlock, "min_version")
}

func TestOrgEnvelopeMalformedJSON(t *testing.T) {
	pub, _ := testKey(t, 1)
	assertReason(t, mustVerifyErr(t, []byte(`{"issuer":`), defaultOpts(pub)), abi.ReasonMalformed, "decode")
}

func TestOrgEnvelopeUnknownFieldRejected(t *testing.T) {
	pub, priv := testKey(t, 1)
	raw := marshal(t, goodEnvelope(t, priv))
	// Inject an unknown top-level key → fail-loud, matching loader discipline.
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		t.Fatal(err)
	}
	obj["issuers"] = json.RawMessage(`"typo"`)
	tampered, _ := json.Marshal(obj)
	assertReason(t, mustVerifyErr(t, tampered, defaultOpts(pub)), abi.ReasonMalformed, "decode")
}

func TestOrgEnvelopeMalformedBody(t *testing.T) {
	// A body carrying an unknown deny reason must fail through the SAME manifest
	// parse path (ToRuntime) as an on-disk manifest would.
	pub, priv := testKey(t, 1)
	e := OrgEnvelope{
		Issuer:    "acme-corp",
		Version:   5,
		NotBefore: fixedNow.Unix() - 10,
		Expires:   fixedNow.Unix() + 10,
		Body:      json.RawMessage(`{"deny":{"x":"NOT_A_REAL_REASON"}}`),
	}
	signed, _ := SignEnvelope(e, priv)
	assertReason(t, mustVerifyErr(t, marshal(t, signed), defaultOpts(pub)), abi.ReasonMalformed, "body")
}

func TestOrgEnvelopeWrongAlg(t *testing.T) {
	pub, priv := testKey(t, 1)
	e := goodEnvelope(t, priv)
	e.Alg = "rsa" // unsupported
	assertReason(t, mustVerifyErr(t, marshal(t, e), defaultOpts(pub)), abi.ReasonMalformed, "alg")
}

func TestOrgEnvelopeInvertedWindow(t *testing.T) {
	pub, priv := testKey(t, 1)
	e := goodEnvelope(t, priv)
	e.NotBefore = fixedNow.Unix() + 3600
	e.Expires = fixedNow.Unix() - 3600
	signed, _ := SignEnvelope(e, priv)
	assertReason(t, mustVerifyErr(t, marshal(t, signed), defaultOpts(pub)), abi.ReasonMalformed, "window")
}

func TestOrgEnvelopeOversize(t *testing.T) {
	pub, priv := testKey(t, 1)
	raw := marshal(t, goodEnvelope(t, priv))
	opts := defaultOpts(pub)
	opts.MaxBytes = 8 // absurdly small budget
	assertReason(t, mustVerifyErr(t, raw, opts), abi.ReasonOversize, "size")
}

func TestOrgEnvelopeMissingRootKey(t *testing.T) {
	_, priv := testKey(t, 1)
	raw := marshal(t, goodEnvelope(t, priv))
	opts := defaultOpts(nil) // no trust anchor
	opts.RootPublicKey = nil
	assertReason(t, mustVerifyErr(t, raw, opts), abi.ReasonTrustViolation, "no_root_key")
}

func TestOrgEnvelopeBadSignatureEncoding(t *testing.T) {
	pub, priv := testKey(t, 1)
	e := goodEnvelope(t, priv)
	e.Sig = "!!!not-base64!!!"
	assertReason(t, mustVerifyErr(t, marshal(t, e), defaultOpts(pub)), abi.ReasonMalformed, "sig")
}

// mustVerifyErr runs VerifyEnvelope and returns the error, failing if it
// unexpectedly succeeded.
func mustVerifyErr(t *testing.T, raw []byte, opts VerifyOptions) error {
	t.Helper()
	if _, err := VerifyEnvelope(raw, opts); err != nil {
		return err
	}
	t.Fatalf("expected verification to fail, but it succeeded")
	return nil
}

// TestOrgEnvelopeReasonsAreClosedVocabulary guards that every reason this
// verifier can emit is a member of the closed refusal vocabulary.
func TestOrgEnvelopeReasonsAreClosedVocabulary(t *testing.T) {
	for _, code := range []abi.ReasonCode{
		abi.ReasonMalformed, abi.ReasonOversize, abi.ReasonTrustViolation,
		abi.ReasonUnwitnessed, abi.ReasonPolicyBlock,
	} {
		name := abi.ReasonName(code)
		if _, ok := abi.ReasonByName(name); !ok {
			t.Fatalf("reason %s is not in the closed vocabulary", name)
		}
		if strings.HasPrefix(name, "REASON_") {
			t.Fatalf("reason code %d has no stable name", code)
		}
	}
}
