package adjudicator

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

// readClassCalls are allowed, read-shaped calls that classify to classRead — the calls
// DefaultRungProfile elides write-only rungs for.
func readClassCalls() []*abi.ToolCall {
	return []*abi.ToolCall{
		inlineCall("get_user_details", `{"user_id":"u1"}`),
		inlineCall("search_flights", `{"from":"SFO","to":"JFK"}`),
		inlineCall("list_all_airports", `{}`),
		inlineCall("calculate", `{"expr":"1+1"}`),
	}
}

// TestReadProfilePreservesVerdict is the correctness guard behind the benchmark: the
// read-class profile must return the SAME verdict as the byte-identical baseline for
// every read-class call.
func TestReadProfilePreservesVerdict(t *testing.T) {
	ctx := context.Background()
	base := New(DefaultPolicy())                // nil Profile (HEAD floor)
	prof := New(DefaultPolicyWithReadProfile()) // read-class elision
	for _, c := range readClassCalls() {
		gb := base.Adjudicate(ctx, c)
		gp := prof.Adjudicate(ctx, c)
		if gb.Kind != gp.Kind || gb.Reason != gp.Reason || gb.By != gp.By {
			t.Errorf("tool %q: read profile %v/%s != baseline %v/%s", c.Tool,
				gp.Kind, abi.ReasonName(gp.Reason), gb.Kind, abi.ReasonName(gb.Reason))
		}
	}
}

// BenchmarkDecideReadClass is the #667 latency arm: it measures read-class adjudication
// under the byte-identical baseline (nil Profile) vs the read-class profile. The
// profile is opt-in because it trades one riskClass computation for skipping rungs
// that are inert for classRead; this benchmark records that cost instead of claiming
// it is always a win. Run with
//
//	go test ./internal/adjudicator -bench BenchmarkDecideReadClass -benchmem
func BenchmarkDecideReadClass(b *testing.B) {
	ctx := context.Background()
	calls := readClassCalls()

	b.Run("baseline_nil_profile", func(b *testing.B) {
		a := New(DefaultPolicy())
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = a.Adjudicate(ctx, calls[i%len(calls)])
		}
	})

	b.Run("read_profile", func(b *testing.B) {
		a := New(DefaultPolicyWithReadProfile())
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = a.Adjudicate(ctx, calls[i%len(calls)])
		}
	})
}
