package trajectory

import (
	"bytes"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/simhash"
)

// embedFor is a test convenience over simhash.Embed.
func embedFor(s string) []float32 { return simhash.Embed(s) }

// mkCall builds a ToolCall with a trace id, tool, and producer-stamped query in the
// OPEN Meta channel — the shape the gateway/agent loop emits.
func mkCall(trace, tool, query string) *abi.ToolCall {
	return &abi.ToolCall{
		TraceID: trace,
		Tool:    tool,
		Meta:    map[string]string{"query": query},
		Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(query)},
	}
}

func allowVerdict() *abi.Verdict { return &abi.Verdict{Kind: abi.VerdictAllow} }
func denyVerdict() *abi.Verdict  { return &abi.Verdict{Kind: abi.VerdictDeny} }

// TestRecorderFoldsTrace — the core: a sequence of events folds into ordered,
// 1-based turns under the right trace, with query and verdict carried through.
func TestRecorderFoldsTrace(t *testing.T) {
	r := New()
	r.Emit(abi.Event{Kind: abi.EvDecide, Call: mkCall("trace-1", "search_kb", "find refund policy"), Verdict: allowVerdict()})
	r.Emit(abi.Event{Kind: abi.EvDeny, Call: mkCall("trace-1", "refund_payment", "refund the customer"), Verdict: denyVerdict()})

	turns := r.Trace("trace-1")
	if len(turns) != 2 {
		t.Fatalf("turns = %d, want 2", len(turns))
	}
	if turns[0].Seq != 1 || turns[1].Seq != 2 {
		t.Fatalf("seq not 1-based contiguous: %d, %d", turns[0].Seq, turns[1].Seq)
	}
	if turns[0].Query != "find refund policy" || turns[0].Tool != "search_kb" || turns[0].Verdict != "ALLOW" {
		t.Fatalf("turn 0 wrong: %+v", turns[0])
	}
	if turns[1].Verdict != "DENY" {
		t.Fatalf("turn 1 verdict = %q, want DENY", turns[1].Verdict)
	}
	if turns[0].ArgsDigest == "" {
		t.Fatalf("args digest not derived from inline ref")
	}
}

// TestOperationalKindsSkipped — Submit/Dispatch/Complete (no verdict, no analysis
// signal) do not become turns; only decision-bearing kinds do.
func TestOperationalKindsSkipped(t *testing.T) {
	r := New()
	r.Emit(abi.Event{Kind: abi.EvSubmit, Call: mkCall("t", "x", "q")})
	r.Emit(abi.Event{Kind: abi.EvDispatch, Call: mkCall("t", "x", "q")})
	if r.Len() != 0 {
		t.Fatalf("operational kinds recorded: len = %d", r.Len())
	}
}

// TestFieldsEnrichment — the OPEN Event.Fields channel carries cost/timestamp/
// materialization the producer stamps, and the recorder folds them in.
func TestFieldsEnrichment(t *testing.T) {
	r := New()
	r.Emit(abi.Event{
		Kind:    abi.EvDecide,
		Call:    mkCall("t", "tool", "the query"),
		Verdict: allowVerdict(),
		Fields: map[string]any{
			"tokens":       1200,
			"bytes":        int64(4096),
			"ts_unix_nano": int64(1700000000000000000),
			"materialized": "FAULT",
			"cache_hit":    false,
		},
	})
	tn := r.Trace("t")[0]
	if tn.TokenEstimate != 1200 || tn.Bytes != 4096 || tn.TSUnixNano != 1700000000000000000 || tn.Materialized != "FAULT" {
		t.Fatalf("fields not folded: %+v", tn)
	}
}

// TestFieldsQueryOverridesMeta — Fields["query"] wins over Meta["query"] (the
// producer's late enrichment is authoritative).
func TestFieldsQueryOverridesMeta(t *testing.T) {
	r := New()
	c := mkCall("t", "tool", "meta query")
	r.Emit(abi.Event{Kind: abi.EvDecide, Call: c, Verdict: allowVerdict(), Fields: map[string]any{"query": "fields query"}})
	if got := r.Trace("t")[0].Query; got != "fields query" {
		t.Fatalf("query = %q, want fields query", got)
	}
}

// TestEmbedQueries — with embedding on, each turn carries a deterministic simhash
// vector over its query; with it off, no vector.
func TestEmbedQueries(t *testing.T) {
	on := New().EmbedQueries(true)
	on.Emit(abi.Event{Kind: abi.EvDecide, Call: mkCall("t", "x", "delete the database"), Verdict: allowVerdict()})
	if len(on.Trace("t")[0].QueryEmbedding) == 0 {
		t.Fatalf("embedding not stamped when EmbedQueries(true)")
	}

	off := New()
	off.Emit(abi.Event{Kind: abi.EvDecide, Call: mkCall("t", "x", "delete the database"), Verdict: allowVerdict()})
	if len(off.Trace("t")[0].QueryEmbedding) != 0 {
		t.Fatalf("embedding stamped when EmbedQueries(false)")
	}
}

// TestVDSOHitIsCacheHit — a vDSO-served call is recorded as a cache hit even with no
// Verdict object, so cache-reuse analysis sees it.
func TestVDSOHitIsCacheHit(t *testing.T) {
	r := New()
	r.Emit(abi.Event{Kind: abi.EvVDSOHit, Call: mkCall("t", "pure_tool", "2+2")})
	tn := r.Trace("t")[0]
	if !tn.CacheHit || tn.Materialized != "HIT" || tn.Verdict != "VDSO_HIT" {
		t.Fatalf("vDSO hit not recorded as cache hit: %+v", tn)
	}
}

// TestMaxPerTrace — the per-trace cap drops oldest turns and renumbers Seq so it
// stays 1-based contiguous.
func TestMaxPerTrace(t *testing.T) {
	r := New().MaxPerTrace(2)
	for i := 0; i < 5; i++ {
		r.Emit(abi.Event{Kind: abi.EvDecide, Call: mkCall("t", "x", "q"), Verdict: allowVerdict()})
	}
	turns := r.Trace("t")
	if len(turns) != 2 {
		t.Fatalf("cap not enforced: len = %d, want 2", len(turns))
	}
	if turns[0].Seq != 1 || turns[1].Seq != 2 {
		t.Fatalf("seq not renumbered after cap: %d, %d", turns[0].Seq, turns[1].Seq)
	}
}

// TestExportImportRoundTrip — the corpus survives JSONL export+import: same turns,
// same per-trace grouping, in order. This is the contract a `fak traj` verb relies on.
func TestExportImportRoundTrip(t *testing.T) {
	r := New().EmbedQueries(true)
	r.Emit(abi.Event{Kind: abi.EvDecide, Call: mkCall("a", "t1", "first query"), Verdict: allowVerdict()})
	r.Emit(abi.Event{Kind: abi.EvDeny, Call: mkCall("a", "t2", "second query"), Verdict: denyVerdict()})
	r.Emit(abi.Event{Kind: abi.EvDecide, Call: mkCall("b", "t3", "third query"), Verdict: allowVerdict()})

	var buf bytes.Buffer
	n, err := r.ExportTo(&buf)
	if err != nil || n != 3 {
		t.Fatalf("export n=%d err=%v, want 3 rows", n, err)
	}

	r2, m, err := ImportFrom(&buf)
	if err != nil || m != 3 {
		t.Fatalf("import m=%d err=%v, want 3", m, err)
	}
	if got := r2.Traces(); len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("trace order not preserved: %v", got)
	}
	if a := r2.Trace("a"); len(a) != 2 || a[1].Verdict != "DENY" {
		t.Fatalf("trace a not round-tripped: %+v", a)
	}
}

// TestIndexSimilarity — the corpus Index finds the near-duplicate past query first,
// the one call the gardening skill makes. This wires trajectory to simhash end to end.
func TestIndexSimilarity(t *testing.T) {
	r := New()
	r.Emit(abi.Event{Kind: abi.EvDecide, Call: mkCall("a", "x", "delete all users from the table"), Verdict: allowVerdict()})
	r.Emit(abi.Event{Kind: abi.EvDecide, Call: mkCall("b", "x", "what time is the meeting"), Verdict: allowVerdict()})

	ix := r.Index()
	if ix.Len() != 2 {
		t.Fatalf("index len = %d, want 2", ix.Len())
	}
	got := ix.TopK(embedFor("remove every user from the table"), 1)
	if len(got) != 1 || got[0].ID != "a:1" {
		t.Fatalf("near-duplicate not ranked first: %+v", got)
	}
}
