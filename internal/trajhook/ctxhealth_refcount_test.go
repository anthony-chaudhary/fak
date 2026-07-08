package trajhook

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

// shedTurn is a corpus row that sheds a span; refTurn references one; plainTurn is
// neither. They build the small transcripts the detector folds, mirroring how the
// gateway compaction path would stamp the OPEN Labels channel on a live session.
func shedTurn(trace string, seq int, span string) trajectory.Turn {
	return trajectory.Turn{TraceID: trace, Seq: seq, Tool: "context_compact",
		Labels: map[string]string{LabelShedSpan: span}}
}
func refTurn(trace string, seq int, span, kind string) trajectory.Turn {
	l := map[string]string{LabelRefSpan: span}
	if kind != "" {
		l[LabelRefKind] = kind
	}
	return trajectory.Turn{TraceID: trace, Seq: seq, Tool: "read", Labels: l}
}
func plainTurn(trace string, seq int) trajectory.Turn {
	return trajectory.Turn{TraceID: trace, Seq: seq, Tool: "read"}
}

// TestShedThenReferencedFiresUseAfterFree is the issue #3096 acceptance: a transcript
// with a KNOWN shed-then-reference shows the shed span's refcount > 0 and the
// USE_AFTER_FREE signal fires.
func TestShedThenReferencedFiresUseAfterFree(t *testing.T) {
	corpus := []trajectory.Turn{
		plainTurn("dirty", 1),
		refTurn("dirty", 2, "sha256:abc", "reread"), // referenced WHILE resident — not a UAF
		shedTurn("dirty", 3, "sha256:abc"),          // span shed here
		plainTurn("dirty", 4),
		refTurn("dirty", 5, "sha256:abc", "reread"), // referenced AFTER shed — the use-after-free
	}

	stats := ShedSpanRefcounts(corpus)
	if len(stats) != 1 {
		t.Fatalf("stats = %+v, want exactly one shed span", stats)
	}
	s := stats[0]
	if s.SpanID != "sha256:abc" || s.ShedSeq != 3 {
		t.Fatalf("stat identity = %+v, want span sha256:abc shed at 3", s)
	}
	// Only the seq-5 reference is after the seq-3 shed; the seq-2 resident read is not counted.
	if s.RefCount != 1 || !s.UseAfterFree {
		t.Fatalf("refcount = %d uaf = %v, want 1 / true (later ref only)", s.RefCount, s.UseAfterFree)
	}
	if !reflect.DeepEqual(s.RefSeqs, []int{5}) {
		t.Fatalf("ref seqs = %v, want [5] (the post-shed reference)", s.RefSeqs)
	}

	findings := ShedRefcount()(corpus)
	if len(findings) != 1 {
		t.Fatalf("findings = %+v, want one USE_AFTER_FREE", findings)
	}
	f := findings[0]
	if f.Label != LabelUseAfterFree || f.Score != 1 || f.TraceID != "dirty" || f.Seq != 3 || f.Related != "sha256:abc" {
		t.Fatalf("finding = %+v, want USE_AFTER_FREE score 1 trace dirty seq 3 related sha256:abc", f)
	}
}

// TestCleanSessionAllShedSpansRefcountZero is the other half of the acceptance: a
// clean session (spans shed and never referenced afterward) shows every shed span at
// refcount 0 and fires NO signal.
func TestCleanSessionAllShedSpansRefcountZero(t *testing.T) {
	corpus := []trajectory.Turn{
		refTurn("clean", 1, "sha256:one", "cited"), // resident reference, before the shed
		shedTurn("clean", 2, "sha256:one"),
		shedTurn("clean", 3, "sha256:two"),
		plainTurn("clean", 4),
		plainTurn("clean", 5), // nobody re-references a shed span
	}

	stats := ShedSpanRefcounts(corpus)
	if len(stats) != 2 {
		t.Fatalf("stats = %+v, want two shed spans", stats)
	}
	for _, s := range stats {
		if s.RefCount != 0 || s.UseAfterFree {
			t.Fatalf("clean shed span %q refcount = %d uaf = %v, want 0 / false", s.SpanID, s.RefCount, s.UseAfterFree)
		}
	}
	if findings := ShedRefcount()(corpus); len(findings) != 0 {
		t.Fatalf("clean session findings = %+v, want none", findings)
	}
}

// TestShedRefcountCountsMultipleLaterRefsAndKinds: refcount increments per later
// reference and records the distinct reference kinds (re-read/cite/pin).
func TestShedRefcountCountsMultipleLaterRefsAndKinds(t *testing.T) {
	corpus := []trajectory.Turn{
		shedTurn("t", 1, "sha256:x"),
		refTurn("t", 2, "sha256:x", "reread"),
		refTurn("t", 3, "sha256:x", "pin"),
		refTurn("t", 4, "sha256:x", "cited"),
	}
	stats := ShedSpanRefcounts(corpus)
	if len(stats) != 1 {
		t.Fatalf("stats = %+v, want one", stats)
	}
	s := stats[0]
	if s.RefCount != 3 {
		t.Fatalf("refcount = %d, want 3 later references", s.RefCount)
	}
	if !reflect.DeepEqual(s.RefKinds, []string{"cited", "pin", "reread"}) {
		t.Fatalf("ref kinds = %v, want sorted [cited pin reread]", s.RefKinds)
	}
	if f := ShedRefcount()(corpus); len(f) != 1 || f[0].Score != 3 {
		t.Fatalf("finding = %+v, want one with score 3", f)
	}
}

// TestShedRefcountDeterministicAcrossTraceOrder: the fold is a pure function of the
// corpus, independent of the input slice's turn order, and separates traces.
func TestShedRefcountDeterministicAcrossTraceOrder(t *testing.T) {
	inOrder := []trajectory.Turn{
		shedTurn("a", 1, "sha256:a"),
		refTurn("a", 2, "sha256:a", "reread"),
		shedTurn("b", 1, "sha256:b"),
		plainTurn("b", 2),
	}
	shuffled := []trajectory.Turn{
		plainTurn("b", 2),
		refTurn("a", 2, "sha256:a", "reread"),
		shedTurn("b", 1, "sha256:b"),
		shedTurn("a", 1, "sha256:a"),
	}
	got1 := ShedSpanRefcounts(inOrder)
	got2 := ShedSpanRefcounts(shuffled)
	if !reflect.DeepEqual(got1, got2) {
		t.Fatalf("order-dependent output:\n in-order = %+v\n shuffled = %+v", got1, got2)
	}
	// Trace a has the use-after-free; trace b is clean.
	if len(got1) != 2 {
		t.Fatalf("stats = %+v, want two shed spans across the two traces", got1)
	}
	if got1[0].TraceID != "a" || !got1[0].UseAfterFree || got1[1].TraceID != "b" || got1[1].UseAfterFree {
		t.Fatalf("per-trace verdicts wrong: %+v", got1)
	}
}

// TestShedRefcountEmptyAndNoShedIsNoOp: a corpus that sheds nothing (or is empty) is
// a clean no-op — the additive-no-regression posture for every session that never
// stamps a shed label.
func TestShedRefcountEmptyAndNoShedIsNoOp(t *testing.T) {
	if s := ShedSpanRefcounts(nil); len(s) != 0 {
		t.Fatalf("nil corpus stats = %+v, want empty", s)
	}
	noShed := []trajectory.Turn{plainTurn("t", 1), refTurn("t", 2, "sha256:z", "reread")}
	if s := ShedSpanRefcounts(noShed); len(s) != 0 {
		t.Fatalf("no-shed corpus stats = %+v, want empty (a ref to a never-shed span is not a UAF)", s)
	}
	if f := ShedRefcount()(noShed); len(f) != 0 {
		t.Fatalf("no-shed findings = %+v, want none", f)
	}
}
