package memq

import (
	"context"
	"reflect"
	"testing"
)

// TestMultiIntentAmaxRecallsMinoritySignal is the #4020 acceptance witness: a cell
// relevant to intent #2 but not intent #1 is recalled under amax with both intents,
// and NOT recalled under the single-intent query — the GQA group-reduce serving N
// consumers where a point intent serves one.
func TestMultiIntentAmaxRecallsMinoritySignal(t *testing.T) {
	ctx := context.Background()
	m := NewMemStore()
	m.Add("billing", "tool_result", DurabilitySession, []byte("refund fee dispute opened"), false)              // cell:0 — intent #1 only (overlap 3)
	m.Add("billing", "tool_result", DurabilitySession, []byte("refund fee dispute escalated"), false)           // cell:1 — intent #1 only (overlap 3)
	m.Add("billing", "tool_result", DurabilitySession, []byte("refund fee dispute resolved"), false)            // cell:2 — intent #1 only (overlap 3)
	m.Add("secops", "tool_result", DurabilitySession, []byte("rotate expiring vault credentials today"), false) // cell:3 — intent #2 only (overlap 4)
	intent1, intent2 := "refund fee dispute", "rotate expiring vault credentials"
	ops := []Op{{Kind: OpScan}, {Kind: OpRank, By: RankRelevance, Desc: true}, {Kind: OpLimit, K: 3}}

	single, err := Run(ctx, m, Query{Intent: intent1, Ops: ops}, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	if got := ids(single.Working); !reflect.DeepEqual(got, []string{"cell:0", "cell:1", "cell:2"}) {
		t.Fatalf("single-intent top-3 = %v, want [cell:0 cell:1 cell:2] (cell:3 starved)", got)
	}

	multi, err := Run(ctx, m, Query{Intent: intent1, Intents: []string{intent1, intent2}, ScoreAgg: ScoreAggAmax, Ops: ops}, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	if got := ids(multi.Working); !reflect.DeepEqual(got, []string{"cell:3", "cell:0", "cell:1"}) {
		t.Fatalf("amax multi-intent top-3 = %v, want [cell:3 cell:0 cell:1] (minority signal first)", got)
	}
}

// TestMultiIntentAggOpMatters pins the upstream tradeoff the knob exposes: under mean
// (and sum) a cell mildly relevant to BOTH intents outranks a cell strongly relevant
// to only one (the minority signal is diluted); under amax the single-consumer cell
// wins. Per-intent overlaps by construction: cell:0 = (2, 2); cell:1 = (0, 3).
func TestMultiIntentAggOpMatters(t *testing.T) {
	ctx := context.Background()
	m := NewMemStore()
	m.Add("ops", "tool_result", DurabilitySession, []byte("refund fee memo plus rotate vault note"), false) // cell:0 — 2 tokens per intent
	m.Add("secops", "tool_result", DurabilitySession, []byte("rotate vault credentials now"), false)        // cell:1 — 3 tokens, intent B only
	intents := []string{"refund fee dispute", "rotate vault credentials"}
	ops := []Op{{Kind: OpScan}, {Kind: OpRank, By: RankRelevance, Desc: true}, {Kind: OpLimit, K: 1}}

	cases := []struct {
		agg  string
		want string
	}{
		{"", "cell:0"},           // empty = mean: 2+2 beats 0+3
		{ScoreAggMean, "cell:0"}, // mean dilutes the minority signal
		{ScoreAggSum, "cell:0"},  // sum agrees with mean under a constant intent count
		{ScoreAggAmax, "cell:1"}, // amax keeps it: max(0,3) beats max(2,2)
	}
	for _, tc := range cases {
		r, err := Run(ctx, m, Query{Intents: intents, ScoreAgg: tc.agg, Ops: ops}, Caps{})
		if err != nil {
			t.Fatalf("agg %q: %v", tc.agg, err)
		}
		if len(r.Working) != 1 || r.Working[0].ID != tc.want {
			t.Errorf("agg %q selected %v, want [%s]", tc.agg, ids(r.Working), tc.want)
		}
	}
}

// TestMultiIntentDefaultPathsByteIdentical pins the default-off contract: (a) a
// single-element Intents produces a Result deeply equal to today's Intent path under
// every agg op, and (b) ScoreAgg without Intents is inert — the gate is Intents alone.
func TestMultiIntentDefaultPathsByteIdentical(t *testing.T) {
	ctx := context.Background()
	base := Get0(t, "recall").Build(Params{Intent: "refund fee", K: 3})
	want, err := Run(ctx, NewDemoStore(), base, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	for _, agg := range []string{"", ScoreAggMean, ScoreAggAmax, ScoreAggSum} {
		q := base
		q.Intents = []string{base.Intent}
		q.ScoreAgg = agg
		got, err := Run(ctx, NewDemoStore(), q, Caps{})
		if err != nil {
			t.Fatalf("agg %q: %v", agg, err)
		}
		if !reflect.DeepEqual(want, got) {
			t.Errorf("Intents={%q} with agg %q diverged from the single-Intent result", base.Intent, agg)
		}
	}
	q := base
	q.ScoreAgg = ScoreAggAmax
	got, err := Run(ctx, NewDemoStore(), q, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(want, got) {
		t.Error("ScoreAgg without Intents changed the result — the gate must be Intents alone")
	}
}

// TestValidateRefusesUnknownScoreAgg: an unknown reduction op never runs — the same
// fail-closed posture as an unknown op kind.
func TestValidateRefusesUnknownScoreAgg(t *testing.T) {
	if err := Validate(Query{ScoreAgg: "median"}); err == nil {
		t.Error("score_agg \"median\" validated but should be refused fail-closed")
	}
	for _, agg := range []string{"", ScoreAggMean, ScoreAggAmax, ScoreAggSum} {
		if err := Validate(Query{ScoreAgg: agg, Intents: []string{"a", "b"}}); err != nil {
			t.Errorf("score_agg %q refused: %v", agg, err)
		}
	}
}
