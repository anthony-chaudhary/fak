package leaseref

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func initTestRealGitRepo(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	dir := t.TempDir()
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "test@example.com"},
		{"config", "user.name", "tester"},
	} {
		c := exec.Command("git", args...)
		c.Dir = dir
		if out, err := c.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	return dir
}

// TestStrictLeaseSnapshotRejectsCorruptRef is the primary regression unit test for #11847.
// It verifies on a real Git fixture that:
//  1. An empty valid store succeeds and returns empty records.
//  2. A store with a valid lease and session/contract refs succeeds, preserving namespace classification.
//  3. Injecting a malformed lease ref causes StrictSnapshot to fail, propagating the offending ref identity.
//  4. Legacy tolerant List and Live remain fully compatible, skipping the corrupt ref.
func TestStrictLeaseSnapshotRejectsCorruptRef(t *testing.T) {
	dir := initTestRealGitRepo(t)
	ctx := context.Background()
	s := NewInDir(dir)

	// 1. An empty valid store succeeds.
	emptyRecs, err := s.StrictSnapshot(ctx)
	if err != nil {
		t.Fatalf("StrictSnapshot on empty store failed: %v", err)
	}
	if len(emptyRecs) != 0 {
		t.Fatalf("StrictSnapshot on empty store returned %d records, want 0", len(emptyRecs))
	}
	emptyList, err := s.StrictList(ctx)
	if err != nil {
		t.Fatalf("StrictList on empty store failed: %v", err)
	}
	if len(emptyList) != 0 {
		t.Fatalf("StrictList on empty store returned %d records, want 0", len(emptyList))
	}
	pkgEmptySnap, err := StrictSnapshot(ctx, s)
	if err != nil || len(pkgEmptySnap) != 0 {
		t.Fatalf("package StrictSnapshot on empty store = %v, %v", pkgEmptySnap, err)
	}
	pkgEmptyList, err := StrictList(ctx, s)
	if err != nil || len(pkgEmptyList) != 0 {
		t.Fatalf("package StrictList on empty store = %v, %v", pkgEmptyList, err)
	}

	// 2. Add a valid lease.
	validRec := Record{
		ID:         "valid-lane",
		TreeGlobs:  []string{"internal/leaseref/**"},
		Holder:     "worker-node:session-1",
		AcquiredAt: time.Now().Unix(),
		TTLSeconds: 600,
	}
	if _, err := s.Acquire(ctx, validRec); err != nil {
		t.Fatalf("Acquire valid lease: %v", err)
	}

	// 3. Add session, contract, and intent refs to prove namespace classification isolation.
	sess := SessionDescriptor{
		ID:        "worker-1",
		Host:      "worker-node",
		UpdatedAt: time.Now().Unix(),
		TTLSecs:   600,
	}
	if _, err := s.PublishSession(ctx, sess); err != nil {
		t.Fatalf("PublishSession: %v", err)
	}

	contract := ContractRecord{
		TicketID:   "ticket-101",
		Holder:     "worker-node:session-1",
		TTLSeconds: 600,
	}
	if _, err := s.AcquireContract(ctx, contract); err != nil {
		t.Fatalf("AcquireContract: %v", err)
	}

	intent := IntentRecord{
		Target:     "#2155",
		Holder:     "worker-node:session-1",
		TTLSeconds: 600,
	}
	claimedIntent, _, err := s.ClaimIntent(ctx, intent, time.Now())
	if err != nil {
		t.Fatalf("ClaimIntent: %v", err)
	}

	// Verify namespace classification predicates.
	validRef := validRec.Ref()
	sessRef := sess.Ref()
	contractRef := contract.Ref()
	intentRef := claimedIntent.Ref()

	if !IsLeaseRef(validRef) {
		t.Fatalf("IsLeaseRef(%q) = false, want true", validRef)
	}
	if IsLeaseRef(sessRef) {
		t.Fatalf("IsLeaseRef(%q) = true, want false (session ref must not be classified as lease)", sessRef)
	}
	if IsLeaseRef(contractRef) {
		t.Fatalf("IsLeaseRef(%q) = true, want false (contract ref must not be classified as lease)", contractRef)
	}
	if IsLeaseRef(intentRef) {
		t.Fatalf("IsLeaseRef(%q) = true, want false (intent ref must not be classified as lease)", intentRef)
	}
	if !IsSessionRef(sessRef) {
		t.Fatalf("IsSessionRef(%q) = false, want true", sessRef)
	}
	if !IsContractRef(contractRef) {
		t.Fatalf("IsContractRef(%q) = false, want true", contractRef)
	}
	if !IsIntentRef(intentRef) {
		t.Fatalf("IsIntentRef(%q) = false, want true", intentRef)
	}

	// Before corruption, StrictSnapshot must return only the valid lease.
	preSnap, err := s.StrictSnapshot(ctx)
	if err != nil {
		t.Fatalf("StrictSnapshot with valid lease + session + contract + intent failed: %v", err)
	}
	if len(preSnap) != 1 || preSnap[0].ID != "valid-lane" {
		t.Fatalf("StrictSnapshot = %+v, want exactly [valid-lane]", preSnap)
	}

	// 4. Inject a malformed lease ref pointing to invalid JSON bytes.
	c := exec.Command("git", "hash-object", "-w", "--stdin")
	c.Dir = dir
	c.Stdin = strings.NewReader("malformed-lease-not-json {{{")
	out, err := c.CombinedOutput()
	if err != nil {
		t.Fatalf("git hash-object: %v\n%s", err, out)
	}
	corruptBlobSHA := strings.TrimSpace(string(out))

	corruptRef := refPrefix + "corrupt-lane"
	c = exec.Command("git", "update-ref", corruptRef, corruptBlobSHA)
	c.Dir = dir
	if out, err := c.CombinedOutput(); err != nil {
		t.Fatalf("git update-ref %s: %v\n%s", corruptRef, err, out)
	}

	if !IsLeaseRef(corruptRef) {
		t.Fatalf("IsLeaseRef(%q) = false, want true", corruptRef)
	}

	// 5. Strict snapshot must now FAIL and identify the offending ref.
	_, err = s.StrictSnapshot(ctx)
	if err == nil {
		t.Fatalf("StrictSnapshot succeeded on store containing malformed lease ref %s", corruptRef)
	}
	if !IsCorruptRef(err) {
		t.Fatalf("IsCorruptRef(err) = false, want true for: %v", err)
	}
	var cre *CorruptRefError
	if !errors.As(err, &cre) {
		t.Fatalf("errors.As(*CorruptRefError) failed, got error type %T (%v)", err, err)
	}
	if cre.Ref != corruptRef {
		t.Fatalf("CorruptRefError.Ref = %q, want %q", cre.Ref, corruptRef)
	}
	if !strings.Contains(err.Error(), corruptRef) {
		t.Fatalf("error message %q must contain offending ref %q", err.Error(), corruptRef)
	}

	// Check that StrictList fails identically.
	_, err = s.StrictList(ctx)
	if err == nil {
		t.Fatalf("StrictList succeeded on store containing malformed lease ref %s", corruptRef)
	}
	if !IsCorruptRef(err) {
		t.Fatalf("StrictList error must satisfy IsCorruptRef: %v", err)
	}

	// Check package-level functions.
	if _, err := StrictSnapshot(ctx, s); !IsCorruptRef(err) {
		t.Fatalf("package StrictSnapshot error = %v, want IsCorruptRef", err)
	}
	if _, err := StrictList(ctx, s); !IsCorruptRef(err) {
		t.Fatalf("package StrictList error = %v, want IsCorruptRef", err)
	}

	// Check StrictLive fails as well.
	if _, _, err := s.StrictLive(ctx, time.Now()); !IsCorruptRef(err) {
		t.Fatalf("StrictLive error = %v, want IsCorruptRef", err)
	}

	// 6. Legacy tolerant List and Live must remain compatible (skipping corrupt ref).
	legacy, err := s.List(ctx)
	if err != nil {
		t.Fatalf("legacy List failed on corrupt ref: %v", err)
	}
	if len(legacy) != 1 || legacy[0].ID != "valid-lane" {
		t.Fatalf("legacy List = %+v, want exactly 1 valid record [valid-lane]", legacy)
	}

	live, expired, err := s.Live(ctx, time.Now())
	if err != nil {
		t.Fatalf("legacy Live failed on corrupt ref: %v", err)
	}
	if len(live) != 1 || live[0].ID != "valid-lane" {
		t.Fatalf("legacy Live = %+v (expired=%v), want exactly 1 live record [valid-lane]", live, expired)
	}
}

// TestStrictLeaseSnapshotRejectsMissingBlobRef verifies that a ref pointing to a missing
// git object fails strict snapshot with CorruptRefError naming the ref, while legacy List skips it.
func TestStrictLeaseSnapshotRejectsMissingBlobRef(t *testing.T) {
	dir := initTestRealGitRepo(t)
	ctx := context.Background()
	s := NewInDir(dir)

	validRec := Record{
		ID:         "valid-keep",
		TreeGlobs:  []string{"pkg/**"},
		Holder:     "worker:1",
		AcquiredAt: time.Now().Unix(),
		TTLSeconds: 600,
	}
	if _, err := s.Acquire(ctx, validRec); err != nil {
		t.Fatalf("Acquire valid lease: %v", err)
	}

	// Create a missing blob ref by writing directly to loose ref file with a non-existent object SHA.
	missingRefName := "refs/fak/locks/missing-lane"
	looseRefPath := filepath.Join(dir, ".git", "refs", "fak", "locks", "missing-lane")
	if err := os.MkdirAll(filepath.Dir(looseRefPath), 0755); err != nil {
		t.Fatalf("MkdirAll loose ref dir: %v", err)
	}
	fakeSHA := "0123456789abcdef0123456789abcdef01234567\n"
	if err := os.WriteFile(looseRefPath, []byte(fakeSHA), 0644); err != nil {
		t.Fatalf("WriteFile loose ref: %v", err)
	}

	// StrictSnapshot must fail and identify missing-lane.
	_, err := s.StrictSnapshot(ctx)
	if err == nil {
		t.Fatalf("StrictSnapshot succeeded with missing blob ref %s", missingRefName)
	}
	var cre *CorruptRefError
	if !errors.As(err, &cre) {
		t.Fatalf("errors.As(*CorruptRefError) failed for missing blob error: %v", err)
	}
	if cre.Ref != missingRefName {
		t.Fatalf("cre.Ref = %q, want %q", cre.Ref, missingRefName)
	}

	// Tolerant List must skip the missing blob ref and return only valid-keep.
	legacy, err := s.List(ctx)
	if err != nil {
		t.Fatalf("legacy List failed on missing blob ref: %v", err)
	}
	if len(legacy) != 1 || legacy[0].ID != "valid-keep" {
		t.Fatalf("legacy List = %+v, want [valid-keep]", legacy)
	}
}

// TestStrictLeaseSnapshotRejectsIDMismatch verifies that a lease blob whose internal ID
// does not match the ref name is rejected by strict snapshot.
func TestStrictLeaseSnapshotRejectsIDMismatch(t *testing.T) {
	dir := initTestRealGitRepo(t)
	ctx := context.Background()
	s := NewInDir(dir)

	// Create a record with ID "claimed-id"
	rec := Record{
		ID:         "claimed-id",
		TreeGlobs:  []string{"pkg/**"},
		Holder:     "worker:1",
		AcquiredAt: time.Now().Unix(),
		TTLSeconds: 600,
	}
	b, err := json.Marshal(rec)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	// Write blob
	c := exec.Command("git", "hash-object", "-w", "--stdin")
	c.Dir = dir
	c.Stdin = strings.NewReader(string(b))
	out, err := c.CombinedOutput()
	if err != nil {
		t.Fatalf("hash-object: %v\n%s", err, out)
	}
	sha := strings.TrimSpace(string(out))

	// Point a ref with a DIFFERENT name: refs/fak/locks/ref-id
	refPath := refPrefix + "ref-id"
	c = exec.Command("git", "update-ref", refPath, sha)
	c.Dir = dir
	if out, err := c.CombinedOutput(); err != nil {
		t.Fatalf("update-ref: %v\n%s", err, out)
	}

	// StrictSnapshot must reject the ID mismatch
	_, err = s.StrictSnapshot(ctx)
	if err == nil {
		t.Fatalf("StrictSnapshot succeeded on mismatched lease ID")
	}
	var cre *CorruptRefError
	if !errors.As(err, &cre) {
		t.Fatalf("errors.As(*CorruptRefError) failed: %v", err)
	}
	if cre.Ref != refPath {
		t.Fatalf("cre.Ref = %q, want %q", cre.Ref, refPath)
	}
}

// TestStrictLiveAndLiveSnapshot verifies partitioning into live vs expired records.
func TestStrictLiveAndLiveSnapshot(t *testing.T) {
	dir := initTestRealGitRepo(t)
	ctx := context.Background()
	s := NewInDir(dir)

	now := time.Now()
	// Active lease (TTL 3600)
	active := Record{
		ID:         "active-lane",
		TreeGlobs:  []string{"active/**"},
		Holder:     "w:1",
		AcquiredAt: now.Unix(),
		TTLSeconds: 3600,
	}
	// Expired lease (acquired long ago, TTL 10)
	past := Record{
		ID:         "past-lane",
		TreeGlobs:  []string{"past/**"},
		Holder:     "w:2",
		AcquiredAt: now.Unix() - 100,
		TTLSeconds: 10,
	}
	if _, err := s.Acquire(ctx, active); err != nil {
		t.Fatalf("Acquire active: %v", err)
	}
	if _, err := s.Acquire(ctx, past); err != nil {
		t.Fatalf("Acquire past: %v", err)
	}

	live, expired, err := s.StrictLive(ctx, now)
	if err != nil {
		t.Fatalf("StrictLive: %v", err)
	}
	if len(live) != 1 || live[0].ID != "active-lane" {
		t.Fatalf("live = %+v, want [active-lane]", live)
	}
	if len(expired) != 1 || expired[0] != "past-lane" {
		t.Fatalf("expired = %v, want [past-lane]", expired)
	}

	res, err := s.StrictSnapshotWithResult(ctx, now)
	if err != nil {
		t.Fatalf("StrictSnapshotWithResult: %v", err)
	}
	if len(res.Records) != 2 || len(res.Live) != 1 || len(res.Expired) != 1 {
		t.Fatalf("res = %+v, want 2 records, 1 live, 1 expired", res)
	}
}

// TestStrictSnapshotNilStore verifies that calling strict snapshot methods or
// package-level helpers on a nil Store returns an error rather than panicking.
func TestStrictSnapshotNilStore(t *testing.T) {
	ctx := context.Background()
	var s *Store

	if _, err := s.StrictSnapshot(ctx); err == nil {
		t.Fatal("s.StrictSnapshot on nil store must fail, got nil err")
	}
	if _, err := StrictSnapshot(ctx, s); err == nil {
		t.Fatal("StrictSnapshot on nil store must fail, got nil err")
	}
	if _, err := StrictList(ctx, s); err == nil {
		t.Fatal("StrictList on nil store must fail, got nil err")
	}
	if _, _, err := StrictLive(ctx, s, time.Now()); err == nil {
		t.Fatal("StrictLive on nil store must fail, got nil err")
	}
}
