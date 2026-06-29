package recall

import (
	"reflect"
	"testing"
)

// durablePage builds a benign, durable, present-and-matching page admitted under a live
// witness — every required #783 axis holds, so its syndrome is reusable. (Local twin of
// page_syndrome_test.go's cleanPage, parameterised on step so an image can hold several.)
func durablePage(step int, body []byte) Page {
	return Page{
		Step:       step,
		Role:       "read_memory",
		Digest:     Digest(body),
		Len:        int64(len(body)),
		Durability: "durable",
		Witness:    "etag:v1",
		TrustEpoch: 3,
	}
}

// TestScrubSyndrome_ReportsReusableVsFailedEvidence is the #784 acceptance: a syndrome
// patrol over a set of persisted pages reports each page's #783 syndrome — clean pages
// reusable, and a page that went stale/corrupt since write flagged with its failed
// evidence axis. Two distinct staleness vectors are exercised against the SAME image:
//
//   - a digest erasure: the swap device rotted so the bytes under a page's address no
//     longer hash to it (the CAS-rot vector);
//   - a witness refutation: a page witnessed-and-current at persist whose source has
//     since been refuted in the live ledger (the trust-ledger vector, which fails BOTH
//     the witness and trust-epoch axes).
//
// It uses the oracle-injected core so the refutation is pinned without mutating the
// package-global vdso.Default (a singleton other tests in this package read).
func TestScrubSyndrome_ReportsReusableVsFailedEvidence(t *testing.T) {
	cleanBody := []byte("the durable, witnessed, present body")
	clean := durablePage(0, cleanBody)

	// A second clean page that is unwitnessed but durable — also reusable, proving the
	// witness axes do not block a page that makes no witness claim.
	unwitBody := []byte("a durable preference with no external witness")
	unwit := Page{Step: 1, Role: "read_memory", Digest: Digest(unwitBody), Len: int64(len(unwitBody)), Durability: "durable"}

	// A page whose CAS blob rotted since persist: it addresses origBody's digest, but the
	// stored bytes have a flipped byte, so they no longer hash — the digest axis fails.
	origBody := []byte("the original authoritative bytes")
	rotD := Digest(origBody)
	rotBlob := append([]byte(nil), origBody...)
	rotBlob[0] ^= 0xFF
	rotted := durablePage(2, origBody) // Digest=rotD, but CAS will hold the rotted blob

	// A page admitted under a witness that is refuted in TODAY's ledger.
	revBody := []byte("sourced under a since-refuted witness")
	revoked := durablePage(3, revBody)
	revoked.Witness = "scrub-784:revoked"

	m := Manifest{
		Version:   ManifestVersion,
		SessionID: "s-scrub-syndrome",
		Pages:     []Page{clean, unwit, rotted, revoked},
		Cleared:   map[string]bool{},
	}
	cas := map[string][]byte{
		clean.Digest:   cleanBody,
		unwit.Digest:   unwitBody,
		rotD:           rotBlob, // rotted: present but does not hash to rotD
		revoked.Digest: revBody,
	}
	ora := fakeOracle{revoked: map[string]bool{"scrub-784:revoked": true}}

	rep := scrubSyndromeImage("/img/s-scrub-syndrome", m, cas, ora)

	if rep.Pages != 4 {
		t.Fatalf("want 4 pages surveyed, got %d", rep.Pages)
	}
	if rep.Reusable != 2 || rep.Failed != 2 {
		t.Fatalf("want 2 reusable + 2 failed, got reusable=%d failed=%d (%+v)", rep.Reusable, rep.Failed, rep.Rows)
	}

	byStep := map[int]PageReusability{}
	for _, r := range rep.Rows {
		byStep[r.Step] = r
	}

	// The two clean pages are reusable with NO failed evidence.
	for _, step := range []int{0, 1} {
		r := byStep[step]
		if !r.Reusable {
			t.Errorf("step %d should be reusable; failed=%v", step, r.Failed)
		}
		if len(r.Failed) != 0 {
			t.Errorf("a reusable page must name no failed evidence, step %d got %v", step, r.Failed)
		}
	}

	// The rotted page fails exactly the digest axis.
	if r := byStep[2]; r.Reusable || !reflect.DeepEqual(r.Failed, []string{"digest"}) {
		t.Errorf("rotted page should fail the digest axis only, got reusable=%v failed=%v", r.Reusable, r.Failed)
	}

	// The revoked-witness page fails BOTH the witness and its temporal companion, the
	// trust-epoch axis — in EvidenceAxis order (witness before trust_epoch).
	if r := byStep[3]; r.Reusable || !reflect.DeepEqual(r.Failed, []string{"witness", "trust_epoch"}) {
		t.Errorf("revoked-witness page should fail witness+trust_epoch, got reusable=%v failed=%v", r.Reusable, r.Failed)
	}

	// The per-axis summary rolls those failures up: digest once, witness once, trust_epoch once.
	wantAxis := map[string]int{"digest": 1, "witness": 1, "trust_epoch": 1}
	if !reflect.DeepEqual(rep.FailedByAxis, wantAxis) {
		t.Fatalf("FailedByAxis = %v, want %v", rep.FailedByAxis, wantAxis)
	}

	// Each failing row carries a non-empty reason on its blocking axis — never a bare
	// "bad" with no witness of WHICH piece an operator must repair.
	for _, step := range []int{2, 3} {
		for _, e := range byStep[step].Syndrome.FailedEvidence() {
			if e.Reason == "" {
				t.Errorf("step %d failed axis %s must explain why (empty reason)", step, e.Axis)
			}
		}
	}
}

// TestScrubSyndrome_IsPureReadOverBodies proves the pass mutates nothing: the persisted
// page bodies (CAS) and the page table are byte-identical after a scrub, even when the
// image contains a failing page. ScrubSyndrome reports/flags only — any seal/tombstone
// action stays in the active Scrub seam.
func TestScrubSyndrome_IsPureReadOverBodies(t *testing.T) {
	dir := t.TempDir()

	cleanBody := []byte("durable healthy body")
	clean := durablePage(0, cleanBody)

	// A page that has gone stale (turn-class durability => fails the durability axis) so the
	// report has something to flag, proving purity holds for failing pages too.
	staleBody := []byte("a turn-class page that was never durable enough to re-enter context")
	stale := Page{Step: 1, Role: "status", Digest: Digest(staleBody), Len: int64(len(staleBody)), Durability: "turn"}
	stale = stampSyndrome(stale)
	clean = stampSyndrome(clean)

	cas := map[string][]byte{clean.Digest: cleanBody, stale.Digest: staleBody}
	m := Manifest{Version: ManifestVersion, SessionID: "s-pure",
		Pages: []Page{clean, stale}, Cleared: map[string]bool{}}
	if err := writeImage(dir, m, cas); err != nil {
		t.Fatal(err)
	}

	// Snapshot the on-disk bytes BEFORE the scrub.
	beforeM, beforeCAS, err := loadImageRaw(dir)
	if err != nil {
		t.Fatal(err)
	}
	beforeSnap := deepCopyManifest(beforeM)
	beforeBytes := copyCAS(beforeCAS)

	rep, err := ScrubSyndrome(dir)
	if err != nil {
		t.Fatalf("scrub: %v", err)
	}
	if rep.Reusable != 1 || rep.Failed != 1 {
		t.Fatalf("want one reusable + one failed, got %+v", rep)
	}
	// The stale page is flagged on the durability axis specifically.
	for _, r := range rep.Rows {
		if r.Step == 1 && !reflect.DeepEqual(r.Failed, []string{"durability"}) {
			t.Fatalf("stale turn-class page should fail the durability axis, got %v", r.Failed)
		}
	}

	// PURITY: re-read the image and prove the page table + every CAS blob is byte-identical.
	afterM, afterCAS, err := loadImageRaw(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(beforeSnap, deepCopyManifest(afterM)) {
		t.Fatal("ScrubSyndrome mutated the persisted manifest — it must be a pure read")
	}
	if !reflect.DeepEqual(beforeBytes, afterCAS) {
		t.Fatal("ScrubSyndrome mutated the persisted CAS bytes — it must be a pure read")
	}
}

// TestScrubSyndrome_EmptyImageReportsEmpty proves an image with no pages scrubs to an
// empty, well-formed report (zero pages, zero reusable, zero failed, no rows) — not an
// error and not a nil-deref.
func TestScrubSyndrome_EmptyImageReportsEmpty(t *testing.T) {
	dir := t.TempDir()
	if err := writeImage(dir, Manifest{Version: ManifestVersion, SessionID: "s-empty",
		Pages: nil, Cleared: map[string]bool{}}, map[string][]byte{}); err != nil {
		t.Fatal(err)
	}
	rep, err := ScrubSyndrome(dir)
	if err != nil {
		t.Fatalf("scrub empty image: %v", err)
	}
	if rep.Pages != 0 || rep.Reusable != 0 || rep.Failed != 0 {
		t.Fatalf("empty image must report zero counts, got %+v", rep)
	}
	if len(rep.Rows) != 0 {
		t.Fatalf("empty image must report no rows, got %v", rep.Rows)
	}
	if rep.InputDir != dir {
		t.Fatalf("report should record its input dir %q, got %q", dir, rep.InputDir)
	}
}
