package modelpack

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestVerifyManifestInvariants verifies all structural and cryptographic constraints on manifests.
func TestVerifyManifestInvariants(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	payload := []byte("payload-bytes-for-invariants")
	sum := sha256.Sum256(payload)
	validManifest := Manifest{
		Schema:   Schema,
		PackID:   "model-alpha",
		Revision: "rev-1",
		Chunks: []Chunk{
			{Digest: hex.EncodeToString(sum[:]), Size: int64(len(payload))},
		},
		Fixtures: []Fixture{
			{Name: "smoke", Input: "in", Expected: "out"},
		},
	}
	if err := Sign(&validManifest, priv); err != nil {
		t.Fatal(err)
	}

	// 1. Valid manifest verifies cleanly.
	if err := Verify(validManifest, pub); err != nil {
		t.Fatalf("expected valid manifest to pass verify, got: %v", err)
	}

	// 2. Invariant: Schema must equal canonical Schema constant.
	badSchema := validManifest
	badSchema.Schema = "fak.unrecognized-manifest/1"
	if err := Verify(badSchema, pub); !errors.Is(err, ErrSignature) {
		t.Fatalf("expected ErrSignature for invalid schema, got: %v", err)
	}

	// 3. Invariant: PackID must not be empty.
	emptyPack := validManifest
	emptyPack.PackID = ""
	if err := Verify(emptyPack, pub); !errors.Is(err, ErrSignature) {
		t.Fatalf("expected ErrSignature for empty PackID, got: %v", err)
	}

	// 4. Invariant: Revision must not be empty.
	emptyRev := validManifest
	emptyRev.Revision = ""
	if err := Verify(emptyRev, pub); !errors.Is(err, ErrSignature) {
		t.Fatalf("expected ErrSignature for empty Revision, got: %v", err)
	}

	// 5. Invariant: Signature must be valid hex.
	badHexSig := validManifest
	badHexSig.Signature = "not-a-hex-signature-string!"
	if err := Verify(badHexSig, pub); !errors.Is(err, ErrSignature) {
		t.Fatalf("expected ErrSignature for malformed hex signature, got: %v", err)
	}

	// 6. Invariant: Forged signature (tampered revision) must be rejected.
	tampered := validManifest
	tampered.Revision = "rev-tampered"
	if err := Verify(tampered, pub); !errors.Is(err, ErrSignature) {
		t.Fatalf("expected ErrSignature for forged signature, got: %v", err)
	}

	// 7. Invariant: Chunk digest length must be exactly 64 characters (SHA-256 hex).
	shortDigest := validManifest
	shortDigest.Chunks = []Chunk{{Digest: strings.Repeat("a", 63), Size: 10}}
	_ = Sign(&shortDigest, priv)
	if err := Verify(shortDigest, pub); !errors.Is(err, ErrSignature) {
		t.Fatalf("expected ErrSignature for short digest, got: %v", err)
	}

	longDigest := validManifest
	longDigest.Chunks = []Chunk{{Digest: strings.Repeat("a", 65), Size: 10}}
	_ = Sign(&longDigest, priv)
	if err := Verify(longDigest, pub); !errors.Is(err, ErrSignature) {
		t.Fatalf("expected ErrSignature for long digest, got: %v", err)
	}

	// 8. Invariant: Chunk size must be non-negative.
	negativeSize := validManifest
	negativeSize.Chunks = []Chunk{{Digest: hex.EncodeToString(sum[:]), Size: -1}}
	_ = Sign(&negativeSize, priv)
	if err := Verify(negativeSize, pub); !errors.Is(err, ErrSignature) {
		t.Fatalf("expected ErrSignature for negative chunk size, got: %v", err)
	}
}

// TestForecastInvariants verifies storage forecasting across missing, partial, and cached chunks.
func TestForecastInvariants(t *testing.T) {
	root := t.TempDir()
	mgr, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}

	c1Digest := strings.Repeat("1", 64)
	c2Digest := strings.Repeat("2", 64)
	man := Manifest{
		Schema:   Schema,
		PackID:   "forecast-pack",
		Revision: "v1",
		Chunks: []Chunk{
			{Digest: c1Digest, Size: 100},
			{Digest: c2Digest, Size: 200},
		},
	}

	// 1. Initial state: both chunks completely missing -> forecast is 300.
	if got := mgr.Forecast(man); got != 300 {
		t.Fatalf("expected forecast 300, got %d", got)
	}

	// 2. Partial chunk: write 40 bytes to c1.part -> forecast should be (100 - 40) + 200 = 260.
	c1PartPath := filepath.Join(root, "chunks", c1Digest+".part")
	if err := os.WriteFile(c1PartPath, make([]byte, 40), 0644); err != nil {
		t.Fatal(err)
	}
	if got := mgr.Forecast(man); got != 260 {
		t.Fatalf("expected forecast 260 with 40-byte part, got %d", got)
	}

	// 3. Complete chunk: write full 100 bytes to final chunk path -> forecast should be 200.
	c1FinalPath := filepath.Join(root, "chunks", c1Digest)
	if err := os.WriteFile(c1FinalPath, make([]byte, 100), 0644); err != nil {
		t.Fatal(err)
	}
	if got := mgr.Forecast(man); got != 200 {
		t.Fatalf("expected forecast 200 with c1 complete, got %d", got)
	}

	// 4. Corrupted chunk size on disk (e.g. 50 bytes instead of 100) -> treated as needing refetch.
	if err := os.WriteFile(c1FinalPath, make([]byte, 50), 0644); err != nil {
		t.Fatal(err)
	}
	// With c1FinalPath having 50 bytes (wrong size), and c1PartPath having 40 bytes,
	// Forecast will see final chunk size != c1.Size, check part file (40 bytes), and need (100 - 40) + 200 = 260.
	if got := mgr.Forecast(man); got != 260 {
		t.Fatalf("expected forecast 260 with invalid final chunk size, got %d", got)
	}
}

// TestRevocationInvariants verifies fail-closed security when revoking active and inactive revisions.
func TestRevocationInvariants(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	root := t.TempDir()
	mgr, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}

	p1 := []byte("pack-payload-v1")
	p2 := []byte("pack-payload-v2")
	m1 := signedManifest(t, priv, "model-x", "v1", p1)
	m2 := signedManifest(t, priv, "model-x", "v2", p2)

	// Install v1 then v2.
	if _, err := mgr.Install(m1, pub, 1000, source(p1), nil); err != nil {
		t.Fatal(err)
	}
	if _, err := mgr.Install(m2, pub, 1000, source(p2), nil); err != nil {
		t.Fatal(err)
	}

	if mgr.Active("model-x") != "v2" {
		t.Fatalf("expected v2 active, got %q", mgr.Active("model-x"))
	}

	// Revoking v1 (the last-known-good) should leave v2 active.
	if _, err := mgr.Revoke("model-x", "v1"); err != nil {
		t.Fatal(err)
	}
	if mgr.Active("model-x") != "v2" {
		t.Fatalf("expected v2 to remain active after revoking v1, got %q", mgr.Active("model-x"))
	}

	// Attempting rollback should fail because v1 is revoked.
	if _, err := mgr.Rollback("model-x"); err == nil {
		t.Fatal("expected rollback to fail when LKG is revoked")
	}

	// Revoking active v2 should notice LKG (v1) is also revoked, leaving no active model.
	if _, err := mgr.Revoke("model-x", "v2"); err != nil {
		t.Fatal(err)
	}
	if got := mgr.Active("model-x"); got != "" {
		t.Fatalf("expected empty active model after revoking all, got %q", got)
	}

	// Attempting to reinstall revoked v1 must fail closed with ErrRevoked.
	if _, err := mgr.Install(m1, pub, 1000, source(p1), nil); !errors.Is(err, ErrRevoked) {
		t.Fatalf("expected ErrRevoked on reinstalling revoked pack, got %v", err)
	}
}

// TestEvictionProtectionInvariants verifies that active and LKG revisions are strictly retained.
func TestEvictionProtectionInvariants(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	root := t.TempDir()
	mgr, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}

	data0 := []byte("model-data-revision-000")
	data1 := []byte("model-data-revision-001")
	data2 := []byte("model-data-revision-002")

	m0 := signedManifest(t, priv, "guard-model", "r0", data0)
	m1 := signedManifest(t, priv, "guard-model", "r1", data1)
	m2 := signedManifest(t, priv, "guard-model", "r2", data2)

	// Install r0, r1, r2 sequentially: r2 is active, r1 is LKG, r0 is inactive.
	if _, err := mgr.Install(m0, pub, 1000, source(data0), nil); err != nil {
		t.Fatal(err)
	}
	if _, err := mgr.Install(m1, pub, 1000, source(data1), nil); err != nil {
		t.Fatal(err)
	}
	if _, err := mgr.Install(m2, pub, 1000, source(data2), nil); err != nil {
		t.Fatal(err)
	}

	if mgr.Active("guard-model") != "r2" {
		t.Fatalf("expected r2 active, got %s", mgr.Active("guard-model"))
	}

	// Request eviction of 1 byte: only r0 should be eligible.
	freed, err := mgr.Evict(1)
	if err != nil {
		t.Fatal(err)
	}
	if freed == 0 {
		t.Fatal("expected at least some bytes freed by evicting r0")
	}

	// Check disk state: r0 removed, r1 and r2 present.
	r0Path := filepath.Join(root, "packs", "guard-model", "r0")
	r1Path := filepath.Join(root, "packs", "guard-model", "r1")
	r2Path := filepath.Join(root, "packs", "guard-model", "r2")

	if _, err := os.Stat(r0Path); !os.IsNotExist(err) {
		t.Fatal("expected r0 to be evicted from disk")
	}
	if _, err := os.Stat(r1Path); err != nil {
		t.Fatalf("expected LKG r1 to remain on disk: %v", err)
	}
	if _, err := os.Stat(r2Path); err != nil {
		t.Fatalf("expected active r2 to remain on disk: %v", err)
	}

	// Additional eviction request should free 0 bytes because remaining packs are protected.
	freedMore, err := mgr.Evict(1000)
	if err != nil {
		t.Fatal(err)
	}
	if freedMore != 0 {
		t.Fatalf("expected 0 bytes freed when only protected revisions exist, got %d", freedMore)
	}
}

// TestReceiptAndJournalInvariants verifies receipt schema, monotonic sequencing, and digest fidelity.
func TestReceiptAndJournalInvariants(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	root := t.TempDir()
	mgr, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}

	payload := []byte("journal-test-payload")
	m := signedManifest(t, priv, "journal-pack", "rev-1", payload)

	receipt, err := mgr.Install(m, pub, 1000, source(payload), nil)
	if err != nil {
		t.Fatal(err)
	}

	// 1. Invariant: Receipt schema is canonical.
	if receipt.Schema != "fak.model-pack-receipt/1" {
		t.Fatalf("unexpected receipt schema: %q", receipt.Schema)
	}
	if receipt.State != "activated" {
		t.Fatalf("expected state activated, got %q", receipt.State)
	}
	if receipt.PackID != "journal-pack" || receipt.Revision != "rev-1" {
		t.Fatalf("receipt identity mismatch: %+v", receipt)
	}

	// 2. Invariant: Events are monotonically sequenced and match receipt digests.
	events := mgr.Events()
	if len(events) < 2 {
		t.Fatalf("expected at least 2 events (reserved + activated), got %d", len(events))
	}

	for i, ev := range events {
		wantSeq := uint64(i + 1)
		if ev.Sequence != wantSeq {
			t.Fatalf("event %d sequence = %d, want %d", i, ev.Sequence, wantSeq)
		}
	}

	// Check that the receipt digest matches the SHA-256 of the last event JSON.
	lastEvent := events[len(events)-1]
	raw, err := json.Marshal(lastEvent)
	if err != nil {
		t.Fatal(err)
	}
	expectedDigest := hex.EncodeToString(sha256Sum(raw))
	if receipt.Digest != expectedDigest {
		t.Fatalf("receipt digest %q != expected %q", receipt.Digest, expectedDigest)
	}

	// 3. Invariant: Journal survives reload through Open.
	reloaded, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	reloadedEvents := reloaded.Events()
	if len(reloadedEvents) != len(events) {
		t.Fatalf("reloaded events count %d != original %d", len(reloadedEvents), len(events))
	}
}

func sha256Sum(b []byte) []byte {
	h := sha256.Sum256(b)
	return h[:]
}

// TestRollbackInvariants verifies atomic state swap between active and LKG revisions.
func TestRollbackInvariants(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	root := t.TempDir()
	mgr, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}

	// Rollback on fresh manager with no LKG returns error.
	if _, err := mgr.Rollback("no-such-pack"); err == nil {
		t.Fatal("expected error on rollback with no LKG")
	}

	p1 := []byte("p1-bytes")
	p2 := []byte("p2-bytes")
	m1 := signedManifest(t, priv, "swap-pack", "v1", p1)
	m2 := signedManifest(t, priv, "swap-pack", "v2", p2)

	if _, err := mgr.Install(m1, pub, 500, source(p1), nil); err != nil {
		t.Fatal(err)
	}
	if _, err := mgr.Install(m2, pub, 500, source(p2), nil); err != nil {
		t.Fatal(err)
	}

	// Initial: active = v2, LKG = v1
	if mgr.Active("swap-pack") != "v2" {
		t.Fatalf("active = %q, want v2", mgr.Active("swap-pack"))
	}

	// First rollback: active becomes v1, LKG becomes v2
	receipt1, err := mgr.Rollback("swap-pack")
	if err != nil {
		t.Fatal(err)
	}
	if receipt1.State != "rolled_back" || mgr.Active("swap-pack") != "v1" {
		t.Fatalf("after first rollback: state=%q, active=%q", receipt1.State, mgr.Active("swap-pack"))
	}

	// Second rollback: swaps back to v2
	receipt2, err := mgr.Rollback("swap-pack")
	if err != nil {
		t.Fatal(err)
	}
	if receipt2.State != "rolled_back" || mgr.Active("swap-pack") != "v2" {
		t.Fatalf("after second rollback: state=%q, active=%q", receipt2.State, mgr.Active("swap-pack"))
	}
}
