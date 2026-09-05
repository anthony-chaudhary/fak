package tooltrend

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/toolrollup"
)

// call is a tiny constructor so the table rows read as (tool, out-tokens, ok).
func call(tool string, out int, ok bool) toolrollup.ToolCall {
	return toolrollup.ToolCall{Tool: tool, TokensOut: out, OK: ok}
}

func TestSizeClass(t *testing.T) {
	cases := []struct {
		tokens int
		want   string
	}{
		{-1, "empty"},
		{0, "empty"},
		{1, "small"},
		{99, "small"},
		{100, "medium"},
		{999, "medium"},
		{1000, "large"},
		{9999, "large"},
		{10000, "xlarge"},
		{1 << 20, "xlarge"},
	}
	for _, c := range cases {
		if got := SizeClass(c.tokens); got != c.want {
			t.Errorf("SizeClass(%d) = %q, want %q", c.tokens, got, c.want)
		}
	}
}

func TestFoldEmpty(t *testing.T) {
	tr := Fold(nil)
	if tr.Schema != Schema {
		t.Errorf("schema = %q, want %q", tr.Schema, Schema)
	}
	if tr.Buckets != 0 {
		t.Errorf("buckets = %d, want 0", tr.Buckets)
	}
	// Slices must be non-nil so a JSON consumer sees [] not null.
	if tr.Points == nil || tr.ToolMovers == nil || tr.ShapeMovers == nil {
		t.Fatalf("empty fold returned a nil slice: %+v", tr)
	}
	if len(tr.Points) != 0 || len(tr.ToolMovers) != 0 || len(tr.ShapeMovers) != 0 {
		t.Errorf("empty fold not empty: %+v", tr)
	}
}

func TestFoldSingleBucketHasNoMovers(t *testing.T) {
	tr := Fold([]Bucket{{
		Label: "s1",
		Calls: []toolrollup.ToolCall{call("read", 50, true), call("read", 50, true), call("bash", 500, false)},
	}})
	if len(tr.Points) != 1 {
		t.Fatalf("points = %d, want 1", len(tr.Points))
	}
	if len(tr.ToolMovers) != 0 || len(tr.ShapeMovers) != 0 {
		t.Errorf("single bucket should have no movers, got tool=%v shape=%v", tr.ToolMovers, tr.ShapeMovers)
	}
	p := tr.Points[0]
	if p.Calls != 3 {
		t.Errorf("calls = %d, want 3", p.Calls)
	}
	// tool mix: read 2/3, bash 1/3
	if got := p.ToolMix["read"]; !almost(got, 2.0/3.0) {
		t.Errorf("tool_mix[read] = %v, want 0.666...", got)
	}
	if got := p.ToolMix["bash"]; !almost(got, 1.0/3.0) {
		t.Errorf("tool_mix[bash] = %v, want 0.333...", got)
	}
	// shape mix: 2 small (50), 1 medium (500)
	if got := p.ShapeMix["small"]; !almost(got, 2.0/3.0) {
		t.Errorf("shape_mix[small] = %v, want 0.666...", got)
	}
	if got := p.ShapeMix["medium"]; !almost(got, 1.0/3.0) {
		t.Errorf("shape_mix[medium] = %v, want 0.333...", got)
	}
	// error rate: 1 of 3 failed
	if !almost(p.ErrorRate, 1.0/3.0) {
		t.Errorf("error_rate = %v, want 0.333...", p.ErrorRate)
	}
}

func TestFoldMoversFirstToLast(t *testing.T) {
	// First session leans on read; last session leans on bash. Between the two,
	// read falls and bash rises, and a newly-appearing tool (edit) rises from 0.
	first := Bucket{Label: "old", Calls: []toolrollup.ToolCall{
		call("read", 50, true), call("read", 50, true), call("read", 50, true), call("bash", 50, true),
	}} // read 0.75, bash 0.25
	mid := Bucket{Label: "mid", Calls: []toolrollup.ToolCall{call("read", 50, true), call("bash", 50, true)}}
	last := Bucket{Label: "new", Calls: []toolrollup.ToolCall{
		call("bash", 50, true), call("bash", 50, true), call("bash", 50, true), call("edit", 50, true),
	}} // bash 0.75, edit 0.25, read 0.0

	tr := Fold([]Bucket{first, mid, last})
	if tr.Buckets != 3 || len(tr.Points) != 3 {
		t.Fatalf("buckets/points = %d/%d, want 3/3", tr.Buckets, len(tr.Points))
	}

	byKey := map[string]Move{}
	for _, m := range tr.ToolMovers {
		byKey[m.Key] = m
	}
	// read: 0.75 -> 0.00, down, |0.75|
	if m := byKey["read"]; m.Direction != "down" || !almost(m.From, 0.75) || !almost(m.To, 0) || !almost(m.AbsChange, 0.75) {
		t.Errorf("read mover = %+v, want down 0.75->0.00", m)
	}
	// bash: 0.25 -> 0.75, up, |0.50|
	if m := byKey["bash"]; m.Direction != "up" || !almost(m.From, 0.25) || !almost(m.To, 0.75) || !almost(m.Delta, 0.5) {
		t.Errorf("bash mover = %+v, want up 0.25->0.75", m)
	}
	// edit: 0.00 -> 0.25, up (appeared)
	if m := byKey["edit"]; m.Direction != "up" || !almost(m.From, 0) || !almost(m.To, 0.25) {
		t.Errorf("edit mover = %+v, want up 0.00->0.25", m)
	}

	// Ranking: read (0.75) is the biggest absolute mover and must sort first.
	if tr.ToolMovers[0].Key != "read" {
		t.Errorf("first tool mover = %q, want read (biggest abs change)", tr.ToolMovers[0].Key)
	}
	// movers sorted by abs change desc
	for i := 1; i < len(tr.ToolMovers); i++ {
		if tr.ToolMovers[i-1].AbsChange < tr.ToolMovers[i].AbsChange {
			t.Errorf("movers not sorted by abs change desc at %d: %+v", i, tr.ToolMovers)
		}
	}
}

func TestFoldFlatKeysDropped(t *testing.T) {
	// Identical first and last mix: read stays 0.5, bash stays 0.5 -> no movers.
	b := Bucket{Calls: []toolrollup.ToolCall{call("read", 50, true), call("bash", 50, true)}}
	tr := Fold([]Bucket{b, b})
	if len(tr.ToolMovers) != 0 {
		t.Errorf("flat mix should yield no tool movers, got %+v", tr.ToolMovers)
	}
	if len(tr.ShapeMovers) != 0 {
		t.Errorf("flat mix should yield no shape movers, got %+v", tr.ShapeMovers)
	}
}

func TestFoldTopKCap(t *testing.T) {
	// First bucket: five distinct tools each 0.2. Last bucket: a different five,
	// each 0.2. That is ten movers (five fall to 0, five rise from 0); cap at 3.
	first := Bucket{Calls: []toolrollup.ToolCall{
		call("a", 50, true), call("b", 50, true), call("c", 50, true), call("d", 50, true), call("e", 50, true),
	}}
	last := Bucket{Calls: []toolrollup.ToolCall{
		call("v", 50, true), call("w", 50, true), call("x", 50, true), call("y", 50, true), call("z", 50, true),
	}}
	tr := FoldTopK([]Bucket{first, last}, 3)
	if len(tr.ToolMovers) != 3 {
		t.Errorf("topK=3 should cap movers at 3, got %d", len(tr.ToolMovers))
	}
	// topK<=0 keeps all ten.
	all := FoldTopK([]Bucket{first, last}, 0)
	if len(all.ToolMovers) != 10 {
		t.Errorf("topK=0 should keep all movers, got %d", len(all.ToolMovers))
	}
}

func TestFoldDeterministic(t *testing.T) {
	buckets := []Bucket{
		{Label: "1", Calls: []toolrollup.ToolCall{call("read", 50, true), call("bash", 2000, false)}},
		{Label: "2", Calls: []toolrollup.ToolCall{call("bash", 2000, true), call("bash", 50, true), call("edit", 200, true)}},
	}
	a := Fold(buckets)
	b := Fold(buckets)
	if !reflect.DeepEqual(a, b) {
		t.Errorf("Fold not deterministic:\n a=%+v\n b=%+v", a, b)
	}
}

func almost(got, want float64) bool {
	d := got - want
	return d < 1e-9 && d > -1e-9
}

func BenchmarkFold(b *testing.B) {
	buckets := makeSyntheticBuckets(10, 50)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = Fold(buckets)
	}
}

func BenchmarkFoldTopK(b *testing.B) {
	buckets := makeSyntheticBuckets(20, 100)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = FoldTopK(buckets, 5)
	}
}

func BenchmarkFoldSingleBucket(b *testing.B) {
	calls := make([]toolrollup.ToolCall, 100)
	tools := []string{"read", "edit", "bash", "grep", "glob"}
	for i := range calls {
		calls[i] = toolrollup.ToolCall{
			Tool:      tools[i%len(tools)],
			TokensOut: (i * 37) % 15000,
			OK:        i%10 != 0,
		}
	}
	bucket := Bucket{Label: "bench-session", Calls: calls}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = Fold([]Bucket{bucket})
	}
}

func BenchmarkSizeClass(b *testing.B) {
	tokens := []int{-5, 0, 42, 250, 4500, 50000}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = SizeClass(tokens[i%len(tokens)])
	}
}

func makeSyntheticBuckets(numBuckets, callsPerBucket int) []Bucket {
	tools := []string{"read", "edit", "bash", "grep", "glob", "write"}
	buckets := make([]Bucket, numBuckets)
	for i := 0; i < numBuckets; i++ {
		calls := make([]toolrollup.ToolCall, callsPerBucket)
		for j := 0; j < callsPerBucket; j++ {
			toolIdx := (j + i) % len(tools)
			calls[j] = toolrollup.ToolCall{
				Tool:      tools[toolIdx],
				TokensOut: ((j + 1) * (i + 1) * 73) % 20000,
				OK:        (j+i)%8 != 0,
			}
		}
		buckets[i] = Bucket{
			Label: fmt.Sprintf("session-%d", i),
			Calls: calls,
		}
	}
	return buckets
}
