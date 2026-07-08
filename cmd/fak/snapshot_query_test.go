package main

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/recall"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionimage"
)

// TestSnapshotQueryAnswersRealSessionImage is the #1529 witness: the first-class
// `fak snapshot query` path answers a content query against a REAL (non-demo) session
// image. It dumps an image with a benign account page and a poisoned refund-policy page
// through the SHIPPED write-time recall gate, reloads it through the exact loader the
// verb uses (openSessionImage -> integrity-verified LoadDir), and runs the O(1) recall
// working-set query. The benign page answers the query; the poisoned page is NEVER in
// the working set (cold-path correct). The test fails before the verb's query core
// exists and passes after.
func TestSnapshotQueryAnswersRealSessionImage(t *testing.T) {
	const id = "airline-real"
	ctx := context.Background()

	// A real, gated session: one benign account result + one poisoned policy doc. The
	// poison mirrors the shipped adversarial fixture (snapDemoPoison), so the write-time
	// gate quarantines it exactly as in production.
	rec := recall.NewRecorder(id)
	rec.Record(ctx, "get_user_details", []byte(snapDemoBenign))   // step 0 benign
	rec.Record(ctx, "read_refund_policy", []byte(snapDemoPoison)) // step 1 POISON -> quarantined

	imgDir := filepath.Join(t.TempDir(), "image")
	if _, err := sessionimage.DumpDir(imgDir, sessionimage.Input{
		SessionID: id,
		Drive:     session.State{TraceID: id, Run: session.Running},
		Recorder:  rec,
		Model:     "model-A", Host: "laptop", Now: 1_700_000_000,
	}); err != nil {
		t.Fatalf("DumpDir a real session image: %v", err)
	}

	// Reload through the verb's own loader (NOT a demo binary), then query.
	img, cleanup, err := openSessionImage(imgDir)
	if err != nil {
		t.Fatalf("openSessionImage: %v", err)
	}
	defer cleanup()
	hits, stats, hasContent, err := querySessionImage(img, "what refund fee did the account show?", 3)
	if err != nil {
		t.Fatalf("querySessionImage: %v", err)
	}

	if !hasContent {
		t.Fatal("expected a queryable core image, got drive-only")
	}
	if stats.Benign != 1 || stats.Quarantined != 1 {
		t.Fatalf("page accounting benign=%d quarantined=%d, want 1/1", stats.Benign, stats.Quarantined)
	}
	if len(hits) == 0 {
		t.Fatal("working set is empty — expected the benign account page to answer the refund-fee query")
	}
	// Cold-path correctness: the quarantined poison page is never a candidate, so no hit
	// carries the poison marker and none descends from the sealed refund-policy page.
	for _, h := range hits {
		if strings.Contains(h.Descriptor, "ignore previous instructions") {
			t.Fatalf("poisoned content leaked into the working set: %+v", h)
		}
	}
	// The benign account page (step 0) is the top hit for the refund-fee query.
	if hits[0].Step != 0 {
		t.Fatalf("top hit step=%d, want the benign account page at step 0", hits[0].Step)
	}
	if hits[0].Bytes != len(snapDemoBenign) {
		t.Fatalf("top hit bytes=%d, want the byte-identical benign account page (%d)", hits[0].Bytes, len(snapDemoBenign))
	}
}

// TestSnapshotQueryDriveOnlyImageHasNoCoreImage proves a freshly-minted drive-only image
// (no recall core image) reports has_core_image=false rather than faulting — the verb
// degrades gracefully instead of pretending a query ran.
func TestSnapshotQueryDriveOnlyImageHasNoCoreImage(t *testing.T) {
	const id = "drive-only"
	imgDir := filepath.Join(t.TempDir(), "image")
	if _, err := sessionimage.DumpDir(imgDir, sessionimage.Input{
		SessionID: id,
		Drive:     session.State{TraceID: id, Run: session.Running},
	}); err != nil {
		t.Fatalf("DumpDir drive-only image: %v", err)
	}
	img, cleanup, err := openSessionImage(imgDir)
	if err != nil {
		t.Fatalf("openSessionImage: %v", err)
	}
	defer cleanup()
	hits, _, hasContent, err := querySessionImage(img, "anything", 3)
	if err != nil {
		t.Fatalf("querySessionImage: %v", err)
	}
	if hasContent {
		t.Fatal("drive-only image should report no core image")
	}
	if len(hits) != 0 {
		t.Fatalf("drive-only image returned %d hit(s), want 0", len(hits))
	}
}

// TestOpenSessionImageArchiveReapsTempDir witnesses the #3298 fix on the .faksession
// archive path: openSessionImage unpacks the archive into a `fak-snap-query-*` temp dir
// and returns a cleanup func; the dir exists while the (lazily demand-paged) image is
// live and is reaped when the caller runs cleanup — no orphan on the success path. An
// isolated TMP root keeps the assertion immune to the live fleet's temp-dir churn.
func TestOpenSessionImageArchiveReapsTempDir(t *testing.T) {
	tmpRoot := t.TempDir()
	t.Setenv("TMPDIR", tmpRoot) // openSessionImage's os.MkdirTemp("", ...) resolves here
	t.Setenv("TMP", tmpRoot)
	t.Setenv("TEMP", tmpRoot)

	const id = "reap-witness"
	imgDir := filepath.Join(tmpRoot, "image")
	if _, err := sessionimage.DumpDir(imgDir, sessionimage.Input{
		SessionID: id,
		Drive:     session.State{TraceID: id, Run: session.Running},
	}); err != nil {
		t.Fatalf("DumpDir: %v", err)
	}
	archive := filepath.Join(tmpRoot, id+".faksession")
	if err := sessionimage.PackFile(imgDir, archive); err != nil {
		t.Fatalf("PackFile: %v", err)
	}

	unpackGlob := filepath.Join(tmpRoot, "fak-snap-query-*")
	if n := globCount(t, unpackGlob); n != 0 {
		t.Fatalf("precondition: %d unpack dir(s) already present", n)
	}
	img, cleanup, err := openSessionImage(archive)
	if err != nil {
		t.Fatalf("openSessionImage(archive): %v", err)
	}
	if img == nil || cleanup == nil {
		t.Fatal("archive path returned a nil image or nil cleanup")
	}
	if n := globCount(t, unpackGlob); n != 1 {
		t.Fatalf("while image live: want 1 unpack dir, got %d", n)
	}
	cleanup()
	if n := globCount(t, unpackGlob); n != 0 {
		t.Fatalf("leak: %d unpack dir(s) survived cleanup", n)
	}
}

func globCount(t *testing.T, pattern string) int {
	t.Helper()
	m, err := filepath.Glob(pattern)
	if err != nil {
		t.Fatalf("glob %s: %v", pattern, err)
	}
	return len(m)
}
