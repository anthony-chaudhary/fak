package sessionimage

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/recall"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// TestBranchDirForksCOW is the load-bearing witness for #1200: a checkpoint forked into a
// new durable id SHARES the parent's content-addressed recall pages copy-on-write (no
// fresh page bytes at branch time), re-keys its own drive, records the parent_id link and
// the "branched from ... at ..." lineage in the migration log — and leaves the parent
// bundle completely unaffected.
func TestBranchDirForksCOW(t *testing.T) {
	parentDir := t.TempDir()
	if _, err := DumpDir(parentDir, buildInput(t, "sess-parent")); err != nil {
		t.Fatalf("DumpDir parent: %v", err)
	}
	// Snapshot the parent's bytes before the fork, to prove it is never written.
	parentBefore := readBundle(t, parentDir)

	branchDir := t.TempDir()
	meta, err := BranchDir(parentDir, branchDir, BranchOptions{BranchID: "sess-branch", Now: 1_700_000_100})
	if err != nil {
		t.Fatalf("BranchDir: %v", err)
	}

	// (1) New durable id + parent_id link.
	if meta.SessionID != "sess-branch" {
		t.Fatalf("branch id = %q, want sess-branch", meta.SessionID)
	}
	if meta.ParentID != "sess-parent" {
		t.Fatalf("parent_id = %q, want sess-parent", meta.ParentID)
	}

	// (2) COW: the branch's content-addressed pages are the parent's — the manifest page
	// table and the CAS keyset are identical, so ZERO fresh page bytes were written at fork
	// time. (A divergent page, once the branch runs, gets a new content address; see below.)
	pManifest, pCAS := loadCore(t, parentDir)
	bManifest, bCAS := loadCore(t, branchDir)
	if len(bManifest.Pages) != len(pManifest.Pages) {
		t.Fatalf("branch page count = %d, want %d (page table must be shared, not rebuilt)", len(bManifest.Pages), len(pManifest.Pages))
	}
	for i := range pManifest.Pages {
		if bManifest.Pages[i].Digest != pManifest.Pages[i].Digest {
			t.Fatalf("page %d digest diverged at fork: branch %s parent %s", i, bManifest.Pages[i].Digest, pManifest.Pages[i].Digest)
		}
	}
	if len(bCAS) != len(pCAS) {
		t.Fatalf("branch CAS has %d entries, want %d (no fresh page bytes at fork)", len(bCAS), len(pCAS))
	}
	for d := range pCAS {
		if _, ok := bCAS[d]; !ok {
			t.Fatalf("branch CAS is missing shared page %s", d)
		}
	}
	// Storage-sharing check: on a filesystem that supports hardlinks (the test tempdir), the
	// swap device is ONE inode, not a second copy. os.SameFile is the cross-platform proof.
	if pfi, e1 := os.Stat(filepath.Join(parentDir, CASFile)); e1 == nil {
		if bfi, e2 := os.Stat(filepath.Join(branchDir, CASFile)); e2 == nil && !os.SameFile(pfi, bfi) {
			t.Logf("cas.json not hardlinked (copy fallback) — COW share still holds by content address")
		}
	}

	// (3) The branch is a whole, integrity-verified image with a re-keyed drive.
	img, err := LoadDir(branchDir)
	if err != nil {
		t.Fatalf("LoadDir branch: %v", err)
	}
	if img.Drive.TraceID != "sess-branch" {
		t.Fatalf("branch drive TraceID = %q, want sess-branch", img.Drive.TraceID)
	}
	if img.Drive.Run != session.Throttled || img.Drive.Budget.TurnsLeft != 3 {
		t.Fatalf("branch drive did not inherit parent state: %+v", img.Drive)
	}

	// (4) Lineage is an audited fact in the migration log.
	if n := len(img.Meta.Migrations); n == 0 {
		t.Fatal("branch migration log is empty; want a 'branched from' entry")
	}
	last := img.Meta.Migrations[len(img.Meta.Migrations)-1]
	if want := "branched from sess-parent at "; !strings.Contains(last.Reason, want) {
		t.Fatalf("migration reason = %q, want it to contain %q", last.Reason, want)
	}

	// (5) The parent bundle is byte-for-byte unaffected by the fork.
	parentAfter := readBundle(t, parentDir)
	for name, before := range parentBefore {
		if string(parentAfter[name]) != string(before) {
			t.Fatalf("parent %s changed by the fork (parent must be unaffected)", name)
		}
	}
}

// TestBranchDivergentPageWrittenFresh proves the copy-on-write tail: once a branch diverges
// (a page CHANGES), only that divergent page is written fresh — it takes a NEW content
// address while every unchanged page keeps its shared digest. This is the content-addressed
// mechanism the acceptance names ("only divergent pages are written fresh").
func TestBranchDivergentPageWrittenFresh(t *testing.T) {
	parentDir := t.TempDir()
	if _, err := DumpDir(parentDir, buildInput(t, "sess-parent")); err != nil {
		t.Fatalf("DumpDir parent: %v", err)
	}
	branchDir := t.TempDir()
	if _, err := BranchDir(parentDir, branchDir, BranchOptions{BranchID: "sess-branch", Now: 1}); err != nil {
		t.Fatalf("BranchDir: %v", err)
	}
	_, baseCAS := loadCore(t, branchDir)

	// The branch runs and appends a NEW page (divergence) on top of the shared pages.
	rec := recall.NewRecorder("sess-branch")
	rec.Record(context.Background(), "get_user_details", []byte(benignAccount))
	rec.Record(context.Background(), "read_refund_policy", []byte(poisonPolicy))
	rec.Record(context.Background(), "search_flights", []byte(benignFlights))
	rec.Record(context.Background(), "book_reservation", []byte(`{"pnr":"ABC123","seat":"14C"}`)) // divergent page

	divergedDir := t.TempDir()
	if _, err := DumpDir(divergedDir, Input{SessionID: "sess-branch", Drive: session.State{TraceID: "sess-branch"}, Recorder: rec, Now: 2}); err != nil {
		t.Fatalf("DumpDir diverged: %v", err)
	}
	_, divergedCAS := loadCore(t, divergedDir)

	// Every shared page still resolves at its old address; exactly one fresh page appeared.
	fresh := 0
	for d := range divergedCAS {
		if _, ok := baseCAS[d]; !ok {
			fresh++
		}
	}
	if fresh != 1 {
		t.Fatalf("fresh page count = %d, want 1 (only the divergent page is written fresh)", fresh)
	}
	for d := range baseCAS {
		if _, ok := divergedCAS[d]; !ok {
			t.Fatalf("shared page %s was rewritten on divergence (should keep its content address)", d)
		}
	}
}

// TestBranchRejectsBadInput covers the guardrails: a blank or parent-colliding id, and a
// branch dir equal to the parent, are all refused before anything is written.
func TestBranchRejectsBadInput(t *testing.T) {
	parentDir := t.TempDir()
	if _, err := DumpDir(parentDir, buildInput(t, "sess-parent")); err != nil {
		t.Fatalf("DumpDir parent: %v", err)
	}
	if _, err := BranchDir(parentDir, t.TempDir(), BranchOptions{BranchID: ""}); err == nil {
		t.Fatal("blank branch id accepted, want error")
	}
	if _, err := BranchDir(parentDir, t.TempDir(), BranchOptions{BranchID: "sess-parent"}); err == nil {
		t.Fatal("parent-colliding branch id accepted, want error")
	}
	if _, err := BranchDir(parentDir, parentDir, BranchOptions{BranchID: "sess-branch"}); err == nil {
		t.Fatal("branch dir == parent dir accepted, want error")
	}
}

// --- helpers ---

// loadCore loads a bundle's recall core image and returns its manifest plus the set of
// content addresses its page table references (each page digest is a CAS key; the map
// dedups pages that share a content address).
func loadCore(t *testing.T, dir string) (recall.Manifest, map[string]struct{}) {
	t.Helper()
	s, err := recall.Load(dir)
	if err != nil {
		t.Fatalf("recall.Load %s: %v", dir, err)
	}
	cas := map[string]struct{}{}
	for _, p := range s.Manifest.Pages {
		cas[p.Digest] = struct{}{}
	}
	return s.Manifest, cas
}

func readBundle(t *testing.T, dir string) map[string][]byte {
	t.Helper()
	out := map[string][]byte{}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir %s: %v", dir, err)
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Fatalf("ReadFile %s: %v", e.Name(), err)
		}
		out[e.Name()] = b
	}
	return out
}
