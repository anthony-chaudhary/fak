package kvint2eval

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// canonicalDigest hashes the CANONICAL LF form of a testdata source.
//
// A provenance pin has to name ONE byte sequence to be checkable at all, and the only sequence
// every checkout agrees on is the one git stores: `git show HEAD:<path> | sha256sum` and a POSIX
// working copy both produce it, while a Windows checkout under core.autocrlf=true holds the same
// source with CRLF and hashes differently. .gitattributes pins *.cu to eol=lf for exactly the
// reason it already pins *.golden. Normalizing here as well keeps the test honest on a checkout
// made BEFORE that pin, so a stale line ending reports as nothing at all rather than as tampering.
func canonicalDigest(t *testing.T, name string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(bytes.ReplaceAll(raw, []byte("\r\n"), []byte("\n")))
	return hex.EncodeToString(sum[:])
}

func observedRequest(t *testing.T) Request {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "l4-observed.json"))
	if err != nil {
		t.Fatal(err)
	}
	var req Request
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatal(err)
	}
	return req
}

// TestProducerPinMatchesTheProducerItNames re-reads the producer and rehashes it.
//
// Evaluate checks only that a pin is 64 hex characters (digestOK); witnessDigest then covers the
// record with a digest OF ITSELF. Neither one ever opens the file the pin refers to, so a
// provenance digest naming NO artifact in the tree satisfies every other test in this package.
// That is precisely how #6260 shipped `653f025b23...` as the producer digest for a producer whose
// bytes hash to `b6c5b44379...` — an internally consistent record making an externally false
// claim. Rehashing the real bytes is the independently read effect the acceptance clause asks for,
// and it fails on drift in EITHER direction: editing the pin or editing the producer.
func TestProducerPinMatchesTheProducerItNames(t *testing.T) {
	want := canonicalDigest(t, "l4_producer.cu")
	req := observedRequest(t)

	// The artifact IS the producer's output slice, and the model is synthesized deterministically
	// by that same producer from seed 6260 rather than being a separate downloadable file, so the
	// producer source is the provenance of both. They therefore carry one digest, and the record
	// says so only if both are checked — pinning just the artifact would leave the model free to
	// drift back to an unbacked value.
	if req.Artifact.SHA256 != want {
		t.Errorf("artifact pin does not match testdata/l4_producer.cu:\n got  %s\n want %s", req.Artifact.SHA256, want)
	}
	if req.Model.SHA256 != want {
		t.Errorf("model pin does not match testdata/l4_producer.cu:\n got  %s\n want %s", req.Model.SHA256, want)
	}
}

// TestPublishedProducerDigestMatchesTheProducer binds the USER-FACING claim to the same bytes.
//
// The false digest in #6260 was not only in the fixture: docs/research/quantization/int2-kv-rotation.md
// prints "Producer SHA-256 is <hex>" to a reader who has no way to tell that no file in the tree
// hashes to it. A pin that only the fixture asserts leaves the prose free to drift, so the doc is
// checked against the producer directly.
func TestPublishedProducerDigestMatchesTheProducer(t *testing.T) {
	want := canonicalDigest(t, "l4_producer.cu")
	path := filepath.Join("..", "..", "docs", "research", "quantization", "int2-kv-rotation.md")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), want) {
		t.Errorf("%s does not publish the producer digest %s that testdata/l4_producer.cu actually hashes to", path, want)
	}
}

// TestObservedWitnessDigestCoversTheCorrectedPins re-derives the record's self-digest.
//
// witness_sha256 is computed over the marshaled Request, so correcting a provenance pin
// necessarily invalidates it. Recording the corrected pin WITHOUT recomputing the witness would
// trade a false provenance claim for a record that Evaluate refuses as DigestChanged; this pins
// the two as one fact so neither can be repaired alone.
func TestObservedWitnessDigestCoversTheCorrectedPins(t *testing.T) {
	req := observedRequest(t)
	got, err := witnessDigest(req)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.EqualFold(got, req.WitnessSHA256) {
		t.Errorf("witness_sha256 does not cover the record:\n got  %s\n want %s", req.WitnessSHA256, got)
	}
}
