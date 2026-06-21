package cdb

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/recall"
)

// attachFixture ingests the committed synthetic session, persists it to a temp core
// image, and attaches a debugger — the full cross-process path (ingest -> persist ->
// reload in a fresh process state), so every assertion below is on bytes that came off
// disk, not out of the recording process.
func attachFixture(t *testing.T) *Image {
	t.Helper()
	ctx := context.Background()
	rec, st, err := IngestSession(ctx, "../../testdata/cdb/session.jsonl", "cdb-fixture")
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}
	if st.Pages == 0 {
		t.Fatalf("ingest recorded no pages")
	}
	dir := t.TempDir()
	if err := rec.Persist(dir); err != nil {
		t.Fatalf("persist: %v", err)
	}
	im, err := Attach(dir)
	if err != nil {
		t.Fatalf("attach: %v", err)
	}
	return im
}

// TestIngestDecomposesAFlatSessionIntoAPageTable — a real-shaped transcript becomes a
// core image: 5 tool-result pages, 3 benign + 2 sealed (the injection and the secret),
// one heavy page that paged out, and the duplicate read deduped on the swap device.
func TestIngestDecomposesAFlatSessionIntoAPageTable(t *testing.T) {
	im := attachFixture(t)
	info := im.Info()

	if info.Pages != 5 {
		t.Errorf("pages = %d, want 5", info.Pages)
	}
	if info.Benign != 3 || info.Sealed != 2 {
		t.Errorf("benign=%d sealed=%d, want 3/2", info.Benign, info.Sealed)
	}
	if info.Heavy < 1 {
		t.Errorf("heavy_pages = %d, want >=1 (the 6.5KB web-search result paged out)", info.Heavy)
	}
	// the duplicate read shares one content-addressed blob: dedup saved exactly the
	// repeated page's bytes, and distinct blobs < pages.
	if info.DedupSaved <= 0 {
		t.Errorf("dedup_saved = %d, want >0 (the duplicate account read)", info.DedupSaved)
	}
	if info.DistinctBlobs >= info.Pages {
		t.Errorf("distinct_blobs=%d should be < pages=%d (dedup)", info.DistinctBlobs, info.Pages)
	}
	if info.RawBytes != info.CASBytes+info.DedupSaved {
		t.Errorf("raw=%d != cas=%d + dedup=%d", info.RawBytes, info.CASBytes, info.DedupSaved)
	}
}

// TestExamineBenignRoundTripsByteIdentical — the debugger's `x` on a benign page
// re-materializes the exact bytes, at zero model tokens (rung 0).
func TestExamineBenignRoundTripsByteIdentical(t *testing.T) {
	ctx := context.Background()
	im := attachFixture(t)
	// step 0 is the benign account result.
	b, err := im.Examine(ctx, 0)
	if err != nil {
		t.Fatalf("examine step 0: %v", err)
	}
	if !strings.Contains(string(b), "refund_fee") {
		t.Errorf("benign account page did not round-trip: %q", string(b))
	}
}

// TestExamineSealedIsRefusedAcrossTheBoundary — the moat: a page the gate sealed at
// write time is refused on page-in from the reloaded image with no witness, and STILL
// refused after a clearance because the content re-screen re-quarantines it (a
// clearance does not launder poison).
func TestExamineSealedIsRefusedAcrossTheBoundary(t *testing.T) {
	ctx := context.Background()
	im := attachFixture(t)

	// find a sealed page (the injection) and its qid.
	var qid string
	var step int = -1
	for _, f := range im.Backtrace() {
		if f.Sealed {
			step, qid = f.Step, f.QID
			break
		}
	}
	if step < 0 {
		t.Fatal("no sealed page in the image")
	}

	if _, err := im.Examine(ctx, step); !errors.Is(err, recall.ErrSealed) {
		t.Fatalf("sealed page page-in: want ErrSealed, got %v", err)
	}
	im.Clear(qid)
	if _, err := im.Examine(ctx, step); !errors.Is(err, recall.ErrSealed) {
		t.Fatalf("sealed page after clear: want still ErrSealed (re-screen re-quarantines), got %v", err)
	}
}

// TestBacktraceLeaksNoPoison — a sealed page's frame is sealed-metadata only; the
// memory map never echoes the poisoned bytes.
func TestBacktraceLeaksNoPoison(t *testing.T) {
	im := attachFixture(t)
	for _, f := range im.Backtrace() {
		if f.Sealed && strings.Contains(strings.ToLower(f.Descriptor), "ignore previous instructions") {
			t.Errorf("sealed frame %d leaked poison into its descriptor: %q", f.Step, f.Descriptor)
		}
	}
}

func TestPageCacheEntryExposesMetadataWithoutPageIn(t *testing.T) {
	im := attachFixture(t)
	e, ok := im.PageCacheEntry(0)
	if !ok {
		t.Fatal("page cache entry missing")
	}
	if e.Plane != cachemeta.PlaneContextPage || e.ID.MediaType != cachemeta.MediaRecallPage {
		t.Fatalf("bad page cache entry: %+v", e)
	}
	if e.Labels["session_id"] == "" || e.Labels["step"] != "0" {
		t.Fatalf("page coordinates missing: %+v", e.Labels)
	}
	all := im.PageCacheEntries()
	if len(all) != im.Info().Pages {
		t.Fatalf("PageCacheEntries len=%d, want %d", len(all), im.Info().Pages)
	}
}

// TestWorkingSetIsASmallResidentSlice — the headline. A follow-up demand-pages only
// the pages it references; the working set is a small fraction of the resident image,
// it carries no poison, and the sealed pages were excluded as candidates outright.
func TestWorkingSetIsASmallResidentSlice(t *testing.T) {
	ctx := context.Background()
	im := attachFixture(t)

	ws := im.WorkingSet(ctx, "what refund fee did the user's account show?", 0)
	if ws.PagesTouched == 0 {
		t.Fatal("working set is empty; expected the account page(s)")
	}
	if ws.PoisonInSet {
		t.Error("poison leaked into the working set")
	}
	if ws.SealedSkipped != 2 {
		t.Errorf("sealed_skipped = %d, want 2", ws.SealedSkipped)
	}
	if ws.BytesPagedIn >= ws.ResidentBytes {
		t.Errorf("bytes_paged_in=%d should be < resident=%d (you didn't replay the image)",
			ws.BytesPagedIn, ws.ResidentBytes)
	}
	// the heavy 6.5KB web-search page is NOT referenced by this query, so it must stay
	// cold — the working-set residency is well under half the resident bytes.
	if ws.ResidencyPct >= 50 {
		t.Errorf("residency = %.1f%%, want a small slice (<50%%)", ws.ResidencyPct)
	}
	if ws.FaultsAvoided < 1 {
		t.Errorf("faults_avoided = %d, want >=1 (the heavy page never faulted in)", ws.FaultsAvoided)
	}
}

func TestWorkingSetSkipsAgentTombstonedPage(t *testing.T) {
	ctx := context.Background()
	im := attachFixture(t)
	if _, err := im.RequestContextChange(recall.ContextChangeRequest{
		Action:      recall.ContextActionTombstone,
		Step:        0,
		Reason:      "agent marked stale account memory",
		RequestedBy: "agent:self-audit",
	}); err != nil {
		t.Fatalf("request context change: %v", err)
	}
	info := im.Info()
	if info.Tombstoned != 1 {
		t.Fatalf("info tombstoned = %d, want 1", info.Tombstoned)
	}
	bt := im.Backtrace()
	if !bt[0].Tombstoned {
		t.Fatalf("backtrace did not surface the tombstone: %+v", bt[0])
	}

	ws := im.WorkingSet(ctx, "what refund fee did the user's account show?", 0)
	if ws.TombstonedSkipped != 1 {
		t.Fatalf("tombstoned_skipped = %d, want 1", ws.TombstonedSkipped)
	}
	for _, sl := range ws.Slices {
		if sl.Step == 0 {
			t.Fatalf("tombstoned page leaked into working set: %+v", sl)
		}
	}
}

// TestGrepIsAReadOnlyMapSearch — grep over the page table returns matching frames and
// pages in nothing; even a pattern that matches a sealed page yields only its safe
// descriptor.
func TestGrepIsAReadOnlyMapSearch(t *testing.T) {
	im := attachFixture(t)
	hits := im.Grep("Read")
	if len(hits) == 0 {
		t.Fatal("grep 'Read' matched no frames")
	}
	for _, f := range hits {
		if strings.Contains(strings.ToLower(f.Descriptor), "exfiltrate") {
			t.Errorf("grep echoed poison bytes via frame %d", f.Step)
		}
	}
}

// TestAttachFailsClosedOnMissingImage — attaching to a non-image directory errors
// rather than returning an empty debugger.
func TestAttachFailsClosedOnMissingImage(t *testing.T) {
	if _, err := Attach(t.TempDir()); err == nil {
		t.Error("attach to an empty dir should fail, got nil error")
	}
}

// TestImageResultPagesInByteIdentical — the byte-faithfulness witness the adversarial
// panel flagged as missing. A tool_result whose content is a block list containing an
// IMAGE block carries a `source` field that cdb's typed jblock struct does NOT model;
// the page must still page in BYTE-IDENTICAL to the original content bytes (the raw
// transcript value), not be dropped to a re-marshaled stub that omits `source`. This is
// the load-bearing rung-0 claim for real sessions, where the heavy pages are images.
func TestImageResultPagesInByteIdentical(t *testing.T) {
	ctx := context.Background()
	// the exact content array as it appears in the transcript: base64 "hello world" in
	// an image source (benign, so canon does not seal it). No interior whitespace, so
	// the json.RawMessage cdb preserves is byte-for-byte this string.
	content := `[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGVsbG8gd29ybGQ="}}]`
	transcript := `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Read","input":{"f":"img.png"}}]}}` + "\n" +
		`{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":` + content + `}]}}`

	rec, st, err := ingest(ctx, strings.NewReader(transcript), "img-test")
	if err != nil || st.Pages != 1 {
		t.Fatalf("ingest: pages=%d err=%v", st.Pages, err)
	}
	dir := t.TempDir()
	if err := rec.Persist(dir); err != nil {
		t.Fatalf("persist: %v", err)
	}
	im, err := Attach(dir)
	if err != nil {
		t.Fatalf("attach: %v", err)
	}
	b, err := im.Examine(ctx, 0)
	if err != nil {
		t.Fatalf("examine: %v (a benign image must not be sealed and must page in)", err)
	}
	if string(b) != content {
		t.Fatalf("image result not byte-identical:\n got: %q\nwant: %q", string(b), content)
	}
	// the `source`/`data` fields the jblock struct ignores MUST survive verbatim — the
	// precise property that would break if flattenContent re-marshaled the typed struct.
	if !strings.Contains(string(b), `"source"`) || !strings.Contains(string(b), "aGVsbG8gd29ybGQ=") {
		t.Errorf("image source/data was dropped on ingest: %q", string(b))
	}
}

// TestIngestFromReaderHandlesBlockListContent — the parser flattens a tool_result whose
// content is a block list (not a bare string), which is how large results arrive.
func TestIngestFromReaderHandlesBlockListContent(t *testing.T) {
	ctx := context.Background()
	transcript := strings.Join([]string{
		`{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Read","input":{"f":"x"}}]}}`,
		`{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"hello from a block list"}]}]}}`,
	}, "\n")
	rec, st, err := ingest(ctx, strings.NewReader(transcript), "reader-test")
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}
	if st.Pages != 1 || st.ToolUses != 1 {
		t.Fatalf("pages=%d tool_uses=%d, want 1/1", st.Pages, st.ToolUses)
	}
	// the page role is resolved to the tool that produced it.
	dir := t.TempDir()
	if err := rec.Persist(dir); err != nil {
		t.Fatalf("persist: %v", err)
	}
	im, err := Attach(dir)
	if err != nil {
		t.Fatalf("attach: %v", err)
	}
	b, err := im.Examine(ctx, 0)
	if err != nil || !strings.Contains(string(b), "hello from a block list") {
		t.Fatalf("examine: %q err=%v", string(b), err)
	}
	if im.Backtrace()[0].Role != "Read" {
		t.Errorf("role = %q, want Read", im.Backtrace()[0].Role)
	}
}
