package deletioncert

import (
	"crypto/ed25519"
	"testing"
)

// fakeJournal is a JournalVerifier backed by a fixed seq->(prev,hash) map — it
// stands in for internal/journal so this package stays a leaf in tests too. A
// real journal-backed verifier is exercised in the cmd/deletioncert demo.
type fakeJournal map[uint64][2]string // seq -> {prevHash, hash}

func (f fakeJournal) AnchorRow(seq uint64) (string, string, bool) {
	v, ok := f[seq]
	if !ok {
		return "", "", false
	}
	return v[0], v[1], true
}

func newKey(t *testing.T) ed25519.PrivateKey {
	t.Helper()
	// Deterministic seed so tests never read randomness (the package never does).
	seed := make([]byte, ed25519.SeedSize)
	for i := range seed {
		seed[i] = byte(i + 1)
	}
	return ed25519.NewKeyFromSeed(seed)
}

const anchorResultDigest = "sha256:deadbeefcafe" // the journal row's content digest

func baseCert() Certificate {
	return Certificate{
		// Subject left empty on purpose: Mint derives it from Anchor.ResultDigest,
		// and Verify re-enforces the equality. A caller that sets a mismatching
		// Subject is treated as a relabel attempt (TestSubjectRelabelRejected).
		ModelPath:    "gqa-rope",
		CodeCommit:   "abc1234",
		WitnessName:  "commit:abc123",
		Span:         Span{From: 5, Len: 3},
		EvictedCount: 3,
		Equivalence:  Equivalence{Claim: "max|Δ|=0 vs never-seen", MaxAbsDelta: 0, RunID: "selfcheck-1"},
		Anchor:       Anchor{Seq: 7, PrevHash: "prev7", Hash: "hash7", ResultDigest: anchorResultDigest},
		JournalHead:  "hash7",
		TrustEpoch:   2,
		IssuedAtUnix: 1_700_000_000,
	}
}

func mintBase(t *testing.T) (Certificate, ed25519.PrivateKey) {
	t.Helper()
	priv := newKey(t)
	c, err := Mint(priv, baseCert())
	if err != nil {
		t.Fatalf("Mint: %v", err)
	}
	return c, priv
}

func goodJournal() fakeJournal { return fakeJournal{7: {"prev7", "hash7"}} }

func TestMintVerifyRoundTrip(t *testing.T) {
	c, _ := mintBase(t)
	r := Verify(c, goodJournal())
	if !r.Valid {
		t.Fatalf("round-trip not valid: %+v", r)
	}
	if !r.SignatureOK || !r.AnchorOK || !r.AnchorBound || !r.SubjectBound || !r.EquivalenceOK {
		t.Errorf("expected all rungs green, got %+v", r)
	}
	if !r.SelfAttested {
		t.Errorf("v1 cert with no ExternalAnchor must report SelfAttested=true")
	}
}

// TestSubjectRelabelRejected: a certificate whose Subject does not equal the
// anchor row's content digest is a relabel attempt — a genuine past row pointed
// at a different subject. The signature still verifies (the issuer signed the
// mismatch), but the subject-binding rung must reject it.
func TestSubjectRelabelRejected(t *testing.T) {
	priv := newKey(t)
	bc := baseCert()
	bc.Subject = "sha256:some-other-data" // != Anchor.ResultDigest
	c, err := Mint(priv, bc)
	if err != nil {
		t.Fatalf("Mint: %v", err)
	}
	r := Verify(c, goodJournal())
	if r.Valid || r.SubjectBound {
		t.Errorf("relabeled subject should fail: %+v", r)
	}
	if !r.SignatureOK || !r.AnchorBound {
		t.Errorf("sig+anchor should pass; only subject binding fails: %+v", r)
	}
}

func TestMintDefaultsScopeAndSubject(t *testing.T) {
	c, _ := mintBase(t)
	if c.Scope != "inference-working-set+agent-memory" {
		t.Errorf("default scope = %q", c.Scope)
	}
	if c.Subject == "" {
		t.Errorf("Subject should be auto-derived when empty")
	}
	if c.Schema != SchemaVersion {
		t.Errorf("schema = %q", c.Schema)
	}
}

// TestTamperDetected flips one field of a minted certificate and asserts Verify
// rejects it. The signature covers the whole pre-image, so ANY field change must
// break verification — this is the property a certificate exists to provide.
func TestTamperDetected(t *testing.T) {
	tamper := map[string]func(*Certificate){
		"evicted_count": func(c *Certificate) { c.EvictedCount = 99 },
		"span_from":     func(c *Certificate) { c.Span.From = 0 },
		"span_len":      func(c *Certificate) { c.Span.Len = 100 },
		"witness":       func(c *Certificate) { c.WitnessName = "commit:forged" },
		"scope":         func(c *Certificate) { c.Scope = "everything-including-weights" },
		"subject":       func(c *Certificate) { c.Subject = "sha256:forged" },
		"trust_epoch":   func(c *Certificate) { c.TrustEpoch = 999 },
		"anchor_seq":    func(c *Certificate) { c.Anchor.Seq = 8 },
		"anchor_hash":   func(c *Certificate) { c.Anchor.Hash = "hashX" },
		"equiv_run":     func(c *Certificate) { c.Equivalence.RunID = "other" },
		"issued_at":     func(c *Certificate) { c.IssuedAtUnix = 0 },
	}
	for name, mut := range tamper {
		t.Run(name, func(t *testing.T) {
			c, _ := mintBase(t)
			mut(&c)
			r := Verify(c, goodJournal())
			if r.Valid {
				t.Errorf("tampered field %q passed verification: %+v", name, r)
			}
			if r.SignatureOK {
				t.Errorf("tampered field %q should break the signature", name)
			}
		})
	}
}

// TestNonBitExactRejected: a certificate recording any drift is not a
// provable-deletion receipt and must fail the equivalence rung.
func TestNonBitExactRejected(t *testing.T) {
	priv := newKey(t)
	bc := baseCert()
	bc.Equivalence.MaxAbsDelta = 1e-6 // composed-rotation drift, not bit-exact
	c, err := Mint(priv, bc)
	if err != nil {
		t.Fatalf("Mint: %v", err)
	}
	r := Verify(c, goodJournal())
	if r.Valid || r.EquivalenceOK {
		t.Errorf("non-bit-exact cert should be rejected, got %+v", r)
	}
	if r.SignatureOK != true {
		t.Errorf("signature should still verify (only equivalence fails): %+v", r)
	}
}

// TestAnchorAbsent: the anchor row is missing / the chain is broken at it.
func TestAnchorAbsent(t *testing.T) {
	c, _ := mintBase(t)
	r := Verify(c, fakeJournal{}) // empty journal: seq 7 absent
	if r.Valid || r.AnchorBound {
		t.Errorf("missing anchor should fail: %+v", r)
	}
	if !r.SignatureOK || !r.EquivalenceOK {
		t.Errorf("sig+equiv should pass before anchor rung: %+v", r)
	}
	if !r.AnchorOK {
		// AnchorOK must be false (row absent)
	} else {
		t.Errorf("AnchorOK should be false for an absent row")
	}
}

// TestAnchorHashMismatch: the row exists but its hashes don't match the cert —
// the certificate was minted against a DIFFERENT journal (or the journal was
// rewritten). Fail closed.
func TestAnchorHashMismatch(t *testing.T) {
	c, _ := mintBase(t)
	j := fakeJournal{7: {"prevDIFF", "hashDIFF"}}
	r := Verify(c, j)
	if r.Valid || r.AnchorBound {
		t.Errorf("hash-mismatched anchor should fail: %+v", r)
	}
	if !r.AnchorOK {
		t.Errorf("row exists so AnchorOK should be true; binding is what fails: %+v", r)
	}
}

// TestNilVerifierFailsClosed: skipping the journal rung is not "valid" — the
// whole point is the anchor is re-checkable, so a nil verifier is not a pass.
func TestNilVerifierFailsClosed(t *testing.T) {
	c, _ := mintBase(t)
	r := Verify(c, nil)
	if r.Valid {
		t.Errorf("nil verifier must not yield Valid: %+v", r)
	}
	if !r.SignatureOK || !r.EquivalenceOK {
		t.Errorf("sig+equiv still checkable without a journal: %+v", r)
	}
}

func TestParseRejectsUnknownSchema(t *testing.T) {
	c, _ := mintBase(t)
	b, err := c.Marshal()
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	// Round-trips cleanly.
	if _, err := Parse(b); err != nil {
		t.Fatalf("Parse of own output failed: %v", err)
	}
	// A future/foreign schema is refused, not silently accepted.
	bad := []byte(`{"schema":"fak.deletioncert/v999"}`)
	if _, err := Parse(bad); err == nil {
		t.Errorf("Parse should reject unknown schema")
	}
}

func TestExternalAnchorClearsSelfAttested(t *testing.T) {
	priv := newKey(t)
	bc := baseCert()
	bc.ExternalAnchor = ExternalAnchor{Kind: "rfc3161", Proof: "deadbeef"}
	c, err := Mint(priv, bc)
	if err != nil {
		t.Fatalf("Mint: %v", err)
	}
	r := Verify(c, goodJournal())
	if !r.Valid {
		t.Fatalf("cert with external anchor should still verify: %+v", r)
	}
	if r.SelfAttested {
		t.Errorf("a cert carrying an ExternalAnchor must report SelfAttested=false")
	}
}

func TestMintRejectsBadKey(t *testing.T) {
	if _, err := Mint(ed25519.PrivateKey{1, 2, 3}, baseCert()); err == nil {
		t.Errorf("Mint should reject an undersized key")
	}
}
