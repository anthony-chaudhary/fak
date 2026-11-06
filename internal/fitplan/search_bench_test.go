package fitplan

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// BenchmarkSearch times the T04 joint model-fit search, the package's headline
// production entry point (the `fak serve --fit auto` path), on the T04 golden
// fixture: an 8B-class dense model on a discrete 16 GiB pool, f32+q8 tiers,
// 4k prompt / 512 output, target concurrency 2.
func BenchmarkSearch(b *testing.B) {
	in := searchInput(16 * gib)
	rep := Search(in)
	if rep.Status != StatusFits || rep.Chosen == nil {
		b.Fatalf("fixture sanity: status %s chosen %+v refusal %+v", rep.Status, rep.Chosen, rep.Refusal)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep = Search(in)
		if rep.Chosen == nil {
			b.Fatalf("chosen went nil: %+v", rep)
		}
	}
}

// BenchmarkPlan times the shared T01 planner pricing entry point on two
// KV-dtype candidates at the golden fixture's pool: the production path that
// prices every candidate against the roofline and chooses the best eligible row.
func BenchmarkPlan(b *testing.B) {
	in := baseInput(16*gib, cand(compute.KVPrecisionF32, 4096, true), cand(compute.KVPrecisionQ8, 4096, false))
	rep := Plan(in)
	if rep.Chosen == nil && rep.Refusal == nil {
		b.Fatalf("fixture sanity: no chosen row and no refusal: %+v", rep)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep = Plan(in)
		if rep.Chosen == nil && rep.Refusal == nil {
			b.Fatalf("chosen went nil: %+v", rep)
		}
	}
}

// BenchmarkSearchWithSLO times the SLO-gated T04 search: the golden fixture with
// TPOT/TTFT ceilings, which gate B* down through maxBatchAtRate and filter rows
// by their chunked-prefill TTFT. The fixture's predicted TTFT is ~4.16 s, so the
// ceiling sits just above it: tightening it to 2000 ms refuses every row.
func BenchmarkSearchWithSLO(b *testing.B) {
	in := searchInput(16 * gib)
	in.SLO = SearchSLO{TPOTMS: 50, TTFTMS: 5000}
	rep := Search(in)
	if rep.Status != StatusFits || rep.Chosen == nil {
		b.Fatalf("fixture sanity: status %s chosen %+v refusal %+v", rep.Status, rep.Chosen, rep.Refusal)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep = Search(in)
		if rep.Chosen == nil {
			b.Fatalf("chosen went nil: %+v", rep)
		}
	}
}
