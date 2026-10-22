package leaseref

import (
	"context"
	"errors"
	"maps"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime integration est=2s lane=default
func TestStrictSnapshotExactWaiver(t *testing.T) {
	t.Parallel()
	dir := initTestRealGitRepo(t)
	put := func(id, data string) string {
		t.Helper()
		oid := reservedNamespaceGit(t, dir, data, "hash-object", "-w", "--stdin")
		reservedNamespaceGit(t, dir, "", "update-ref", refPrefix+id, oid)
		return oid
	}
	legacy := put("epoch-1106", "1\n")
	put("epoch-backup", `{"id":"epoch-backup","tree_globs":["protected/**"],"ttl_seconds":0}`)
	put("expired", `{"id":"expired","acquired_unix":1,"ttl_seconds":1}`)
	w := SnapshotWaiver{Ref: refPrefix + "epoch-1106", ExpectedOID: legacy, Reason: "explicit opaque legacy exception", Evidence: "fixture review"}
	store := NewInDir(dir)
	ctx, now := context.Background(), time.Unix(1800000000, 0)
	before := reservedNamespaceRefs(t, dir)
	live, expired, err := store.StrictLiveSnapshotWithWaivers(ctx, now, []SnapshotWaiver{w})
	if err != nil || len(live) != 1 || live[0].ID != "epoch-backup" || len(expired) != 1 || expired[0] != "expired" {
		t.Fatalf("snapshot = %v, %v, %v", live, expired, err)
	}
	if !maps.Equal(before, reservedNamespaceRefs(t, dir)) {
		t.Fatal("snapshot changed refs")
	}
	if _, err := store.StrictSnapshot(ctx); !IsCorruptRef(err) {
		t.Fatalf("ordinary strict reader unexpectedly changed: %v", err)
	}
	fail := func(name string, s *Store, ws []SnapshotWaiver) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			before := reservedNamespaceRefs(t, dir)
			live, expired, err := s.StrictLiveSnapshotWithWaivers(ctx, now, ws)
			if err == nil || live != nil || expired != nil {
				t.Fatalf("expected closed refusal, got %v, %v, %v", live, expired, err)
			}
			if !maps.Equal(before, reservedNamespaceRefs(t, dir)) {
				t.Fatal("refusal changed refs")
			}
		})
	}
	fail("unapproved", store, nil)
	fail("duplicate", store, []SnapshotWaiver{w, w})
	bad := w
	bad.ExpectedOID = strings.Repeat("a", 40)
	fail("changed-oid", store, []SnapshotWaiver{bad})
	bad = w
	bad.Ref = refPrefix + "missing"
	fail("missing-ref", store, []SnapshotWaiver{bad})
	bad = w
	bad.Evidence = " "
	fail("missing-evidence", store, []SnapshotWaiver{bad})
	bad = w
	bad.Ref = "refs/heads/main"
	fail("outside-locks", store, []SnapshotWaiver{bad})
	bad = w
	bad.ExpectedOID = legacy[:12]
	fail("abbreviated-oid", store, []SnapshotWaiver{bad})
	put("second-unknown", "2\n")
	fail("second-unknown", store, []SnapshotWaiver{w})
	reservedNamespaceGit(t, dir, "", "update-ref", "-d", refPrefix+"second-unknown")
	for _, data := range []string{`null`, `{"id":"epoch-1106"}`, `{"id":"other"}`, `{"id":3}`, `{"id":`, `not-json`} {
		w.ExpectedOID = put("epoch-1106", data)
		fail("cannot-waive-"+data, store, []SnapshotWaiver{w})
	}
	w.ExpectedOID = put("epoch-1106", "1\n")
	tree := reservedNamespaceGit(t, dir, "", "mktree")
	reservedNamespaceGit(t, dir, "", "update-ref", w.Ref, tree)
	w.ExpectedOID = tree
	fail("nonblob", store, []SnapshotWaiver{w})
	w.ExpectedOID = put("epoch-1106", "1\n")
	for _, stage := range []string{"enumeration", "blob-type", "blob-read", "second-enumeration"} {
		count := 0
		run := func(ctx context.Context, dir string, args ...string) (string, int, error) {
			original := args
			if args[0] == "--no-replace-objects" {
				args = args[1:]
			}
			if args[0] == "for-each-ref" {
				count++
			}
			reject := stage == "enumeration" && args[0] == "for-each-ref" || stage == "second-enumeration" && args[0] == "for-each-ref" && count == 2 || stage == "blob-type" && args[0] == "cat-file" && args[1] == "-t" || stage == "blob-read" && args[0] == "cat-file" && args[1] == "blob"
			if reject {
				return "", 128, nil
			}
			return gitRunner(ctx, dir, original...)
		}
		fail(stage, NewWithRunner(run, dir), []SnapshotWaiver{w})
	}
	// Pinning is observable: every payload load must use a full OID, not a ref.
	run := func(ctx context.Context, dir string, args ...string) (string, int, error) {
		original := args
		if args[0] == "--no-replace-objects" {
			args = args[1:]
		}
		if args[0] == "cat-file" && !waiverObjectID(args[2]) {
			return "", 1, errors.New("mutable payload read")
		}
		return gitRunner(ctx, dir, original...)
	}
	if _, _, err := NewWithRunner(run, dir).StrictLiveSnapshotWithWaivers(ctx, now, []SnapshotWaiver{w}); err != nil {
		t.Fatal(err)
	}
	// Replacement refs must not substitute different bytes for an accepted OID.
	replacement := reservedNamespaceGit(t, dir, `{"id":"epoch-1106"}`, "hash-object", "-w", "--stdin")
	replacementRef := "refs/replace/" + w.ExpectedOID
	reservedNamespaceGit(t, dir, "", "update-ref", replacementRef, replacement)
	if _, _, err := store.StrictLiveSnapshotWithWaivers(ctx, now, []SnapshotWaiver{w}); err != nil {
		t.Fatalf("replacement altered pinned object: %v", err)
	}
	if got := reservedNamespaceGit(t, dir, "", "rev-parse", replacementRef); got != replacement {
		t.Fatal("replacement ref changed")
	}
	// A mapping changed after immutable reads must be refused, without undoing it.
	changed := false
	run = func(ctx context.Context, dir string, args ...string) (string, int, error) {
		original := args
		if args[0] == "--no-replace-objects" {
			args = args[1:]
		}
		if args[0] == "cat-file" && args[1] == "blob" && !changed {
			changed = true
			put("epoch-1106", "2\n")
		}
		return gitRunner(ctx, dir, original...)
	}
	if _, _, err := NewWithRunner(run, dir).StrictLiveSnapshotWithWaivers(ctx, now, []SnapshotWaiver{w}); err == nil || !strings.Contains(err.Error(), "changed during") {
		t.Fatalf("ref race = %v", err)
	}
}

// fak-test:runtime integration est=100ms lane=default
func TestStrictSnapshotWaiverFullGitRef(t *testing.T) {
	t.Parallel()
	dir := initTestRealGitRepo(t)
	ref := refPrefix + "epoch-anthony-chaudhary/fak#12582"
	reservedNamespaceGit(t, dir, "", "check-ref-format", ref)
	oid := reservedNamespaceGit(t, dir, "1\n", "hash-object", "-w", "--stdin")
	reservedNamespaceGit(t, dir, "", "update-ref", ref, oid)
	w := SnapshotWaiver{Ref: ref, ExpectedOID: oid, Reason: "exact opaque exception", Evidence: "fixture review"}
	store := NewInDir(dir)
	before := reservedNamespaceRefs(t, dir)
	live, expired, err := store.StrictLiveSnapshotWithWaivers(context.Background(), time.Now(), []SnapshotWaiver{w})
	if err != nil || len(live) != 0 || len(expired) != 0 {
		t.Fatalf("full-ref waiver = %v, %v, %v", live, expired, err)
	}
	if !maps.Equal(before, reservedNamespaceRefs(t, dir)) {
		t.Fatal("full-ref waiver changed refs")
	}
	// Full Git identity never relaxes ordinary Record ID validation or permits a
	// record-shaped payload to be hidden by a new exact waiver.
	oid = reservedNamespaceGit(t, dir, `{"id":"epoch-anthony-chaudhary/fak#12582"}`, "hash-object", "-w", "--stdin")
	reservedNamespaceGit(t, dir, "", "update-ref", ref, oid)
	w.ExpectedOID = oid
	for _, waivers := range [][]SnapshotWaiver{nil, {w}} {
		if _, _, err := store.StrictLiveSnapshotWithWaivers(context.Background(), time.Now(), waivers); err == nil {
			t.Fatal("invalid Record ID admitted")
		}
	}
	for _, suffix := range []string{"", "x//y", ".hidden/x", "x.lock/y", "x..y", "x@{y", "x*", "x.", "x\n", "epoch/v1/x", "contract/v1/x", "session-x", "intent-x", "contract-x"} {
		w.Ref = refPrefix + suffix
		if err := ValidateSnapshotWaivers([]SnapshotWaiver{w}); err == nil {
			t.Fatalf("invalid or reserved ref admitted: %q", w.Ref)
		}
	}
}
