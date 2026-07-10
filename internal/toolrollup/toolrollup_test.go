package toolrollup

import (
	"bytes"
	"encoding/json"
	"math"
	"reflect"
	"strings"
	"testing"
)

// sampleCorpus is 6 synthetic tool calls across 3 tools, with a mix of successes and
// errors, chosen so every aggregate has a hand-checkable value.
//
//	Read: 3 calls, in {100,200,300} out {50,100,150} dur {10,20,30}, 1 error
//	Bash: 2 calls, in {40,60}       out {20,30}       dur {5,15},     1 error
//	Grep: 1 call,  in {10}          out {5}           dur {2},        0 errors
func sampleCorpus() []ToolCall {
	return []ToolCall{
		{Tool: "Read", TokensIn: 100, TokensOut: 50, DurationMS: 10, OK: true},
		{Tool: "Bash", TokensIn: 40, TokensOut: 20, DurationMS: 5, OK: true},
		{Tool: "Read", TokensIn: 200, TokensOut: 100, DurationMS: 20, OK: true},
		{Tool: "Grep", TokensIn: 10, TokensOut: 5, DurationMS: 2, OK: true},
		{Tool: "Read", TokensIn: 300, TokensOut: 150, DurationMS: 30, OK: false},
		{Tool: "Bash", TokensIn: 60, TokensOut: 30, DurationMS: 15, OK: false},
	}
}

func almostEqual(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestRollup_Aggregates(t *testing.T) {
	got := Rollup(sampleCorpus())

	// Deterministic order: call count desc, then name asc.
	wantOrder := []string{"Read", "Bash", "Grep"}
	if len(got) != len(wantOrder) {
		t.Fatalf("got %d stats, want %d: %+v", len(got), len(wantOrder), got)
	}
	for i, name := range wantOrder {
		if got[i].Tool != name {
			t.Fatalf("order[%d] = %q, want %q (full: %+v)", i, got[i].Tool, name, got)
		}
	}

	want := map[string]ToolStat{
		"Read": {
			Tool: "Read", Calls: 3,
			TotalTokensIn: 600, MeanTokensIn: 200,
			TotalTokensOut: 300, MeanTokensOut: 100,
			TotalDuration: 60, MeanDuration: 20,
			Errors: 1, ErrorRate: 1.0 / 3.0, Share: 3.0 / 6.0,
		},
		"Bash": {
			Tool: "Bash", Calls: 2,
			TotalTokensIn: 100, MeanTokensIn: 50,
			TotalTokensOut: 50, MeanTokensOut: 25,
			TotalDuration: 20, MeanDuration: 10,
			Errors: 1, ErrorRate: 0.5, Share: 2.0 / 6.0,
		},
		"Grep": {
			Tool: "Grep", Calls: 1,
			TotalTokensIn: 10, MeanTokensIn: 10,
			TotalTokensOut: 5, MeanTokensOut: 5,
			TotalDuration: 2, MeanDuration: 2,
			Errors: 0, ErrorRate: 0, Share: 1.0 / 6.0,
		},
	}

	for _, g := range got {
		w := want[g.Tool]
		if g.Calls != w.Calls || g.Errors != w.Errors {
			t.Errorf("%s counts: calls=%d errors=%d, want calls=%d errors=%d", g.Tool, g.Calls, g.Errors, w.Calls, w.Errors)
		}
		if g.TotalTokensIn != w.TotalTokensIn || g.TotalTokensOut != w.TotalTokensOut || g.TotalDuration != w.TotalDuration {
			t.Errorf("%s totals: in=%d out=%d dur=%d, want in=%d out=%d dur=%d",
				g.Tool, g.TotalTokensIn, g.TotalTokensOut, g.TotalDuration, w.TotalTokensIn, w.TotalTokensOut, w.TotalDuration)
		}
		if !almostEqual(g.MeanTokensIn, w.MeanTokensIn) || !almostEqual(g.MeanTokensOut, w.MeanTokensOut) || !almostEqual(g.MeanDuration, w.MeanDuration) {
			t.Errorf("%s means: in=%v out=%v dur=%v, want in=%v out=%v dur=%v",
				g.Tool, g.MeanTokensIn, g.MeanTokensOut, g.MeanDuration, w.MeanTokensIn, w.MeanTokensOut, w.MeanDuration)
		}
		if !almostEqual(g.ErrorRate, w.ErrorRate) {
			t.Errorf("%s error rate = %v, want %v", g.Tool, g.ErrorRate, w.ErrorRate)
		}
		if !almostEqual(g.Share, w.Share) {
			t.Errorf("%s share = %v, want %v", g.Tool, g.Share, w.Share)
		}
	}

	// Shares partition the corpus: they sum to 1.
	var shareSum float64
	for _, g := range got {
		shareSum += g.Share
	}
	if !almostEqual(shareSum, 1.0) {
		t.Errorf("shares sum to %v, want 1.0", shareSum)
	}
}

// TestRollup_TieBreakByName asserts the secondary sort key: equal call counts break
// by tool name ascending, so the ranking is total (never map-order dependent).
func TestRollup_TieBreakByName(t *testing.T) {
	recs := []ToolCall{
		{Tool: "beta", OK: true},
		{Tool: "alpha", OK: true},
		{Tool: "gamma", OK: true},
	}
	got := Rollup(recs)
	wantOrder := []string{"alpha", "beta", "gamma"}
	for i, name := range wantOrder {
		if got[i].Tool != name {
			t.Fatalf("tie-break order[%d] = %q, want %q (full: %+v)", i, got[i].Tool, name, got)
		}
	}
}

func TestRollup_Empty(t *testing.T) {
	got := Rollup(nil)
	if got == nil {
		t.Fatal("Rollup(nil) returned a nil slice; want empty non-nil")
	}
	if len(got) != 0 {
		t.Fatalf("Rollup(nil) = %+v, want empty", got)
	}
}

func TestReadCorpus_RoundTrip(t *testing.T) {
	orig := sampleCorpus()

	// Write the corpus as JSONL (one JSON record per line), the on-disk shape.
	var buf bytes.Buffer
	for _, r := range orig {
		b, err := json.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}
		buf.Write(b)
		buf.WriteByte('\n')
	}

	got, err := ReadCorpus(&buf)
	if err != nil {
		t.Fatalf("ReadCorpus: %v", err)
	}
	if !reflect.DeepEqual(got, orig) {
		t.Fatalf("round-trip mismatch:\n got %+v\nwant %+v", got, orig)
	}

	// And the fold over the read-back corpus equals the fold over the originals.
	if !reflect.DeepEqual(Rollup(got), Rollup(orig)) {
		t.Fatal("Rollup over read-back corpus differs from Rollup over originals")
	}
}

func TestReadCorpus_BlankLinesTolerated(t *testing.T) {
	jsonl := `{"tool":"Read","input_tokens":100,"output_tokens":50,"duration_ms":10,"ok":true}

{"tool":"Bash","input_tokens":40,"output_tokens":20,"duration_ms":5,"ok":true}
`
	got, err := ReadCorpus(strings.NewReader(jsonl))
	if err != nil {
		t.Fatalf("ReadCorpus with blank lines: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d records, want 2: %+v", len(got), got)
	}
	if got[0].Tool != "Read" || got[1].Tool != "Bash" {
		t.Fatalf("unexpected records: %+v", got)
	}
}

func TestReadCorpus_MalformedRecordErrors(t *testing.T) {
	jsonl := `{"tool":"Read","ok":true}
{"tool":"Bash", not json}
`
	if _, err := ReadCorpus(strings.NewReader(jsonl)); err == nil {
		t.Fatal("expected an error on a malformed record, got nil")
	}
}

func TestReadCorpus_Empty(t *testing.T) {
	got, err := ReadCorpus(strings.NewReader(""))
	if err != nil {
		t.Fatalf("ReadCorpus(empty): %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("ReadCorpus(empty) = %+v, want empty", got)
	}
}

func TestToolRollupSchema(t *testing.T) {
	if ToolRollupSchema != "fak.toolrollup.v1" {
		t.Fatalf("ToolRollupSchema = %q, want fak.toolrollup.v1", ToolRollupSchema)
	}
}

// TestRender_Smoke asserts the human table prints a header plus one data row per
// tool, in the fold's deterministic order (most-used first), so a report reads
// stably. It checks the header columns and that the busiest tool (Read, 3 calls)
// renders on the first data line.
func TestRender_Smoke(t *testing.T) {
	var buf bytes.Buffer
	Render(&buf, Rollup(sampleCorpus()))

	lines := strings.Split(strings.TrimRight(buf.String(), "\n"), "\n")
	if len(lines) != 4 { // 1 header + 3 tools
		t.Fatalf("Render produced %d lines, want 4 (header + 3 tools):\n%s", len(lines), buf.String())
	}
	header := lines[0]
	for _, col := range []string{"TOOL", "CALLS", "SHARE%", "ERR%", "MEAN-IN", "MEAN-OUT", "MEAN-MS"} {
		if !strings.Contains(header, col) {
			t.Errorf("header missing column %q: %q", col, header)
		}
	}
	// First data row is the most-used tool: Read (3 calls).
	if !strings.HasPrefix(strings.TrimSpace(lines[1]), "Read") {
		t.Errorf("first data row = %q, want it to start with Read", lines[1])
	}
	if !strings.Contains(lines[1], "3") {
		t.Errorf("Read row should show 3 calls: %q", lines[1])
	}
}

// TestRender_EmptyHonest asserts an empty rollup prints just the header (no panic,
// no data rows) — the honest-empty contract mirrored from the fold.
func TestRender_EmptyHonest(t *testing.T) {
	var buf bytes.Buffer
	Render(&buf, Rollup(nil))
	lines := strings.Split(strings.TrimRight(buf.String(), "\n"), "\n")
	if len(lines) != 1 {
		t.Fatalf("Render(empty) produced %d lines, want 1 (header only):\n%s", len(lines), buf.String())
	}
	if !strings.Contains(lines[0], "TOOL") {
		t.Fatalf("empty render missing header: %q", lines[0])
	}
}
