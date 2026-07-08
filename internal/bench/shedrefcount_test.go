package bench

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// TestShedRefcountBench is the re-runnable witness for issue #3096: the shed-span
// reference-count detector. It asserts the issue's acceptance on both halves —
// a real-shaped transcript with a KNOWN shed-then-reference lands the span
// refcount > 0 at shed and fires USE_AFTER_FREE, and a CLEAN session lands every
// shed span at refcount 0 — and pins the report to a checked-in golden artifact.
func TestShedRefcountBench(t *testing.T) {
	r := BuildShedRefcountReport()
	if r.Provenance.Kind != ProvenanceSimulated {
		t.Fatalf("fixture provenance = %q, want %q", r.Provenance.Kind, ProvenanceSimulated)
	}
	if r.Schema != ShedRefcountSchema {
		t.Fatalf("schema = %q, want %q", r.Schema, ShedRefcountSchema)
	}

	// Acceptance half 1 — the known shed-then-reference: three shed spans, two of
	// them use-after-free, discriminated per span.
	if r.ShedSpans != 3 {
		t.Fatalf("shed spans = %d, want 3", r.ShedSpans)
	}
	byID := map[string]SpanRefcount{}
	for _, s := range r.Spans {
		byID[s.SpanID] = s
	}

	// The originating-task span was re-read (turn 5) AND cited (turn 7) after the
	// shed at turn 3 → refcount 2, use-after-free.
	if got := byID["sha-originating-task"]; got.Refcount != 2 || !got.UseAfterFree {
		t.Errorf("sha-originating-task refcount=%d uaf=%v; want 2 / true", got.Refcount, got.UseAfterFree)
	} else {
		if got.RefsByKind[RefReread] != 1 || got.RefsByKind[RefCite] != 1 {
			t.Errorf("sha-originating-task refs_by_kind = %v; want reread:1 cite:1", got.RefsByKind)
		}
		// refcount > 0 measured against the shed turn — the issue's "> 0 at shed time".
		if got.ShedTurn != 3 {
			t.Errorf("sha-originating-task shed_turn = %d; want 3", got.ShedTurn)
		}
	}

	// The tool-result schema span was pinned (turn 6) after the shed → refcount 1,
	// use-after-free via the pin path (not just re-read).
	if got := byID["sha-toolresult-schema"]; got.Refcount != 1 || !got.UseAfterFree || got.RefsByKind[RefPin] != 1 {
		t.Errorf("sha-toolresult-schema refcount=%d uaf=%v pin=%d; want 1 / true / 1",
			got.Refcount, got.UseAfterFree, got.RefsByKind[RefPin])
	}

	// The scratch-note span was shed and never referenced → refcount 0, safe. The
	// detector does NOT flag every shed, only the still-live ones.
	if got := byID["sha-scratch-note"]; got.Refcount != 0 || got.UseAfterFree {
		t.Errorf("sha-scratch-note refcount=%d uaf=%v; want 0 / false", got.Refcount, got.UseAfterFree)
	}

	if r.UseAfterFreeCount != 2 || r.Signal != SignalUseAfterFree || r.Clean {
		t.Fatalf("uaf_count=%d signal=%q clean=%v; want 2 / %q / false",
			r.UseAfterFreeCount, r.Signal, r.Clean, SignalUseAfterFree)
	}

	// Acceptance half 2 — a clean session: every shed span at refcount 0, no signal.
	clean := BuildShedRefcountReportFor(DefaultCleanTranscript())
	if clean.ShedSpans != 2 {
		t.Fatalf("clean shed spans = %d, want 2", clean.ShedSpans)
	}
	for _, s := range clean.Spans {
		if s.Refcount != 0 || s.UseAfterFree {
			t.Errorf("clean session span %s refcount=%d uaf=%v; want 0 / false", s.SpanID, s.Refcount, s.UseAfterFree)
		}
	}
	if clean.UseAfterFreeCount != 0 || clean.Signal != "" || !clean.Clean {
		t.Fatalf("clean uaf_count=%d signal=%q clean=%v; want 0 / \"\" / true",
			clean.UseAfterFreeCount, clean.Signal, clean.Clean)
	}

	// Over-count guard: the clean session's re-read at turn 1 is BEFORE the shed at
	// turn 4, so it is live use and must not be counted; the pin at turn 6 names a
	// span that was never shed and must be ignored. Both are already witnessed by
	// the refcount-0 assertions above, but assert the config span's shed turn to be
	// explicit that the pre-shed reference was seen and correctly excluded.
	for _, s := range clean.Spans {
		if s.SpanID == "sha-config" && s.ShedTurn != 4 {
			t.Errorf("sha-config shed_turn = %d; want 4", s.ShedTurn)
		}
	}

	// Golden: the committed detector artifact. Regenerate with UPDATE_GOLDEN=1.
	got, err := r.JSON()
	if err != nil {
		t.Fatalf("marshal report: %v", err)
	}
	golden := filepath.Join("testdata", "shedrefcount_report.json")
	if os.Getenv("UPDATE_GOLDEN") != "" {
		if err := os.WriteFile(golden, append(got, '\n'), 0o644); err != nil {
			t.Fatalf("write golden: %v", err)
		}
		t.Logf("updated golden %s", golden)
		return
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("read golden (run with UPDATE_GOLDEN=1 to create): %v", err)
	}
	if !bytes.Equal(bytes.TrimRight(want, "\n"), bytes.TrimRight(got, "\n")) {
		t.Errorf("report drifted from golden %s; re-run with UPDATE_GOLDEN=1 if intended", golden)
	}
}
