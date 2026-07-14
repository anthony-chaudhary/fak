package adjudicator

import (
	"context"
	"regexp"
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

// BenchmarkDecideCapitalizedTool is the #4007 allocation arm: Claude Code tool
// names are capitalized ("Bash", "Read", "WebFetch"), so a case-insensitive shape
// probe that re-lowers the name per prefix pays an allocation on every probe.
// This benchmark counts those allocations on the default floor for exactly that
// name shape; the read-class arms above use lower-case names where ToLower
// returns its input unallocated and cannot see the regression. Run with
//
//	go test ./internal/adjudicator -bench BenchmarkDecideCapitalizedTool -benchmem
func BenchmarkDecideCapitalizedTool(b *testing.B) {
	ctx := context.Background()
	calls := []*abi.ToolCall{
		inlineCall("Bash", `{"command":"ls"}`),
		inlineCall("Read", `{"file_path":"notes.txt"}`),
		inlineCall("WebFetch", `{"url":"https://example.com/x"}`),
	}
	a := New(DefaultPolicy())
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Adjudicate(ctx, calls[i%len(calls)])
	}
}

// BenchmarkDecideArgPredicateCanonicalized is the #2407 latency arm: it times
// the ArgDenyRegex rung with the canonicalization stage against a baseline
// with no ArgPredicates configured, so a regression that made
// canonicalizeArgValue expensive (e.g. an accidental allocation-heavy path)
// shows up here rather than only in aggregate. Run with
//
//	go test ./internal/adjudicator -bench BenchmarkDecideArgPredicateCanonicalized -benchmem
func BenchmarkDecideArgPredicateCanonicalized(b *testing.B) {
	ctx := context.Background()
	call := inlineCall("read_credential_file", `{"path":"~/.ssh/id_rsa"}`)

	b.Run("no_arg_predicates", func(b *testing.B) {
		a := New(Policy{Allow: map[string]bool{"read_credential_file": true}})
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = a.Adjudicate(ctx, call)
		}
	})

	b.Run("canonicalized_arg_deny_regex", func(b *testing.B) {
		a := New(Policy{
			Allow: map[string]bool{"read_credential_file": true},
			ArgPredicates: []ArgPredicate{{
				Tool: "read_credential_file", Arg: "path", Kind: ArgDenyRegex,
				Re: regexp.MustCompile(`\.ssh/id_(rsa|ed25519)$`), Reason: abi.ReasonPolicyBlock,
			}},
		})
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = a.Adjudicate(ctx, call)
		}
	})
}
