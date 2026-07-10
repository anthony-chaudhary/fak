package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

// writeCorpusFile marshals turns as one-JSON-Turn-per-line JSONL (the shape
// trajectory.ImportFrom reads) into a temp file and returns its path.
func writeCorpusFile(t *testing.T, turns []trajectory.Turn) string {
	t.Helper()
	var b strings.Builder
	enc := json.NewEncoder(&b)
	for _, tn := range turns {
		if err := enc.Encode(tn); err != nil {
			t.Fatalf("encode turn: %v", err)
		}
	}
	path := filepath.Join(t.TempDir(), "corpus.jsonl")
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	return path
}

func tk(trace string, seq int, tool string, tokens int, verdict string) trajectory.Turn {
	return trajectory.Turn{TraceID: trace, Seq: seq, Tool: tool, TokenEstimate: tokens, Verdict: verdict}
}

// A corpus with two sessions: the first leans on read, the second on bash, plus a
// non-tool turn that must be counted in Turns but excluded from the tool folds.
func demoCorpus(t *testing.T) string {
	return writeCorpusFile(t, []trajectory.Turn{
		{TraceID: "t1", Seq: 0, Query: "do the thing"}, // non-tool turn
		tk("t1", 1, "read", 50, "ALLOW"),
		tk("t1", 2, "read", 50, "ALLOW"),
		tk("t1", 3, "read", 50, "ALLOW"),
		tk("t1", 4, "bash", 50, "DENY"),
		tk("t2", 1, "bash", 50, "ALLOW"),
		tk("t2", 2, "bash", 50, "ALLOW"),
		tk("t2", 3, "bash", 50, "ALLOW"),
		tk("t2", 4, "edit", 50, "ALLOW"),
	})
}

func TestCmdTrajReportJSON(t *testing.T) {
	path := demoCorpus(t)
	out := captureStdout(t, func() { cmdTrajReport([]string{"--corpus", path, "--json"}) })

	var rep trajReport
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("unmarshal report: %v\n%s", err, out)
	}
	if rep.Schema != trajReportSchema {
		t.Errorf("schema = %q, want %q", rep.Schema, trajReportSchema)
	}
	if rep.Turns != 9 {
		t.Errorf("turns = %d, want 9 (incl. the non-tool turn)", rep.Turns)
	}
	if rep.ToolTurns != 8 {
		t.Errorf("tool_turns = %d, want 8", rep.ToolTurns)
	}
	if rep.Sessions != 2 {
		t.Errorf("sessions = %d, want 2", rep.Sessions)
	}

	// Rollup: read and bash each have 4 calls across the corpus.
	byTool := map[string]int{}
	for _, s := range rep.Tools {
		byTool[s.Tool] = s.Calls
	}
	if byTool["read"] != 3 || byTool["bash"] != 4 || byTool["edit"] != 1 {
		t.Errorf("tool calls = %v, want read:3 bash:4 edit:1", byTool)
	}

	// Trend: first session (t1) read-heavy, last (t2) bash-heavy.
	moves := map[string]string{} // key -> direction
	for _, m := range rep.Trend.ToolMovers {
		moves[m.Key] = m.Direction
	}
	if moves["read"] != "down" {
		t.Errorf("read should trend down, movers=%+v", rep.Trend.ToolMovers)
	}
	if moves["bash"] != "up" {
		t.Errorf("bash should trend up, movers=%+v", rep.Trend.ToolMovers)
	}
	if moves["edit"] != "up" {
		t.Errorf("edit should trend up (appeared), movers=%+v", rep.Trend.ToolMovers)
	}

	// First bucket error rate: 1 DENY of 4 tool calls.
	if len(rep.Trend.Points) != 2 {
		t.Fatalf("trend points = %d, want 2", len(rep.Trend.Points))
	}
	if got := rep.Trend.Points[0].ErrorRate; got < 0.24 || got > 0.26 {
		t.Errorf("first-session error rate = %v, want ~0.25", got)
	}

	// Transitions never span a session boundary: read->read must be present.
	sawReadRead := false
	for _, e := range rep.Transitions {
		if e.From == "read" && e.To == "read" {
			sawReadRead = true
		}
	}
	if !sawReadRead {
		t.Errorf("expected a read->read transition, got %+v", rep.Transitions)
	}
}

func TestCmdTrajReportText(t *testing.T) {
	path := demoCorpus(t)
	out := captureStdout(t, func() { cmdTrajReport([]string{"--corpus", path}) })
	for _, want := range []string{"tool calls", "top tools", "top transitions", "tool-mix movers", "output-shape movers"} {
		if !strings.Contains(out, want) {
			t.Errorf("text report missing %q\n%s", want, out)
		}
	}
}

func TestTurnOK(t *testing.T) {
	cases := map[string]bool{
		"":           true,
		"ALLOW":      true,
		"allow":      true,
		"WITNESS":    true,
		"DENY":       false,
		" quarantine ": false,
		"BLOCK":      false,
		"ERROR":      false,
		"FAULT":      false,
	}
	for verdict, want := range cases {
		if got := turnOK(verdict); got != want {
			t.Errorf("turnOK(%q) = %v, want %v", verdict, got, want)
		}
	}
}

func TestGroupByTraceSortsBySeq(t *testing.T) {
	turns := []trajectory.Turn{
		tk("t1", 3, "c", 0, "ALLOW"),
		tk("t1", 1, "a", 0, "ALLOW"),
		tk("t2", 1, "x", 0, "ALLOW"),
		tk("t1", 2, "b", 0, "ALLOW"),
	}
	order, byTrace := groupByTrace(turns)
	if len(order) != 2 || order[0] != "t1" || order[1] != "t2" {
		t.Fatalf("order = %v, want [t1 t2] (first-seen)", order)
	}
	got := []string{}
	for _, tn := range byTrace["t1"] {
		got = append(got, tn.Tool)
	}
	if strings.Join(got, "") != "abc" {
		t.Errorf("t1 tools = %v, want a,b,c sorted by seq", got)
	}
}
