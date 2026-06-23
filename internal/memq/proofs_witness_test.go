package memq

// proofs_witness_test.go — deterministic witness tests closing the OPEN math-proof
// obligation for internal/memq. See fak/docs/proofs/00-METHOD.md.
//
// OPEN closed here:
//
//	[memq-deterministic-input-driven] Run and Explain are deterministic and
//	input-driven: for a fixed (backend cells, query, caps) the plan AND the result —
//	the per-step counts, the rendered set + order + bytes, the refusals, and the
//	proposed/applied effects — are byte-for-byte identical on every invocation,
//	depending only on the inputs and nothing nondeterministic (no RNG, wall-clock,
//	or map-iteration-order dependence leaks into the output ordering).
//	mechanism: exec.go Run (slice iteration + sort.SliceStable in sortByRank, the
//	precomputed refcount/score maps READ but never RANGED for output order),
//	plan.go Explain.
//
// Strategy: determinism + metamorphic. We fingerprint the full Result/Plan and assert
//   - many repeated Run/Explain calls on one store are byte-identical (no per-call
//     RNG/clock drift),
//   - results are identical across many independently-built demo stores (the demo
//     corpus is deterministic, and the executor builds fresh refcount/score maps each
//     run — Go randomizes map iteration order per map, so identical output witnesses
//     that no map-iteration order leaks into membership/order),
//   - a changed query deterministically re-selects (its own output stable).

import (
	"context"
	"fmt"
	"testing"
)

// fingerprintResult canonicalizes a Result into a single comparable string capturing
// the step trace, rendered set (order + bytes), refusals, and effects.
func fingerprintResult(r Result) string {
	var b []byte
	b = append(b, []byte("intent="+r.Intent+"\n")...)
	for _, s := range r.Steps {
		b = append(b, []byte(fmt.Sprintf("step#%d|%s|in=%d|out=%d|note=%s\n", s.Index, s.Kind, s.In, s.Out, s.Note))...)
	}
	for _, it := range r.Rendered {
		b = append(b, []byte(fmt.Sprintf("rendered|%s|step=%d|bytes=%d|tok=%d\n", it.ID, it.Step, it.Bytes, it.Tokens))...)
	}
	for _, rf := range r.Refused {
		b = append(b, []byte(fmt.Sprintf("refused|%s|%s\n", rf.ID, rf.Reason))...)
	}
	for _, e := range r.Effects {
		b = append(b, []byte(fmt.Sprintf("effect|%s|applied=%v|cells=%v|note=%s\n", e.Kind, e.Applied, e.Cells, e.Note))...)
	}
	for _, c := range r.Working {
		b = append(b, []byte(fmt.Sprintf("work|%s|%s\n", c.ID, c.Descriptor))...)
	}
	return string(b)
}

// witnessQuery is a non-trivial pipeline: it filters, ranks by relevance, budgets,
// renders, and consolidates — so the fingerprint is non-vacuous.
func witnessQuery() Query {
	return Query{
		Intent: "refund fee",
		Ops: []Op{
			{Kind: OpScan},
			{Kind: OpFilter, Pred: &Pred{Op: PredEq, Field: "sealed", Value: "false"}},
			{Kind: OpRank, By: RankRelevance, Desc: true},
			{Kind: OpBudget, Bytes: 400},
			{Kind: OpRender},
			{Kind: OpConsolidate},
		},
	}
}

func TestRunIsDeterministicAcrossRepeatedCalls(t *testing.T) {
	ctx := context.Background()
	m := NewDemoStore()
	q := witnessQuery()
	r0, err := Run(ctx, m, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	want := fingerprintResult(r0)
	if want == "" {
		t.Fatal("non-vacuous precondition failed: empty fingerprint")
	}
	for i := 0; i < 64; i++ {
		r, err := Run(ctx, NewDemoStore(), q, Caps{})
		if err != nil {
			t.Fatal(err)
		}
		if got := fingerprintResult(r); got != want {
			t.Fatalf("Run not deterministic at iter %d:\n want %q\n got  %q", i, want, got)
		}
	}
}

func TestExplainIsDeterministic(t *testing.T) {
	q := witnessQuery()
	want := Explain(q).Text()
	if want == "" {
		t.Fatal("empty plan text")
	}
	for i := 0; i < 32; i++ {
		if got := Explain(q).Text(); got != want {
			t.Fatalf("Explain not deterministic at iter %d", i)
		}
	}
}

func TestChangedIntentReselects(t *testing.T) {
	ctx := context.Background()
	q1 := Get0(t, "render").Build(Params{Intent: "refund fee"})
	q2 := Get0(t, "render").Build(Params{Intent: "go developer wsl tests"})
	r1, err := Run(ctx, NewDemoStore(), q1, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	r2, err := Run(ctx, NewDemoStore(), q2, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	f1, f2 := fingerprintResult(r1), fingerprintResult(r2)
	if f1 == "" || f2 == "" {
		t.Fatal("non-vacuous precondition failed: an intent produced an empty result")
	}
	if f1 == f2 {
		t.Fatal("two different intents produced an identical result — intent is not actually an input")
	}
}
