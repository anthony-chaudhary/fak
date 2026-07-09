package atif

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

func turn(trace string, seq int, tool, verdict string) trajectory.Turn {
	return trajectory.Turn{TraceID: trace, Seq: seq, Tool: tool, Verdict: verdict, Labels: map[string]string{}}
}

func TestFromTurns_SingleTraceMapsStepsInOrder(t *testing.T) {
	turns := []trajectory.Turn{
		turn("t1", 2, "Read", "ALLOW"),
		turn("t1", 1, "Bash", "ALLOW"),
		turn("t1", 3, "Write", "DENY"),
	}
	b := FromTurns(turns, Options{})
	if b.SchemaVersion != SchemaVersion {
		t.Fatalf("schema = %q, want %q", b.SchemaVersion, SchemaVersion)
	}
	if len(b.Trajectories) != 1 {
		t.Fatalf("want 1 root trajectory, got %d", len(b.Trajectories))
	}
	tr := b.Trajectories[0]
	if tr.TrajectoryID != "t1" || tr.StepCount != 3 {
		t.Fatalf("trajectory id=%q stepCount=%d", tr.TrajectoryID, tr.StepCount)
	}
	// Steps must be seq-sorted and index-monotonic from 0.
	wantSeq := []int{1, 2, 3}
	for i, s := range tr.Steps {
		if s.Index != i {
			t.Errorf("step %d has index %d", i, s.Index)
		}
		if s.Seq != wantSeq[i] {
			t.Errorf("step %d seq = %d, want %d", i, s.Seq, wantSeq[i])
		}
		if s.ToolUseID != "t1:"+itoa(s.Seq) {
			t.Errorf("step %d tool_use_id = %q", i, s.ToolUseID)
		}
	}
	if tr.Steps[2].Type != "decision" {
		t.Errorf("DENY step type = %q, want decision", tr.Steps[2].Type)
	}
}

func TestFromTurns_RedactionDefaultOmitsQueryAndLabels(t *testing.T) {
	x := turn("t1", 1, "Bash", "ALLOW")
	x.Query = "secret prompt text"
	x.Labels["model"] = "claude-opus-4-8"

	// Default: redacted.
	b := FromTurns([]trajectory.Turn{x}, Options{})
	s := b.Trajectories[0].Steps[0]
	if s.Query != "" {
		t.Errorf("redacted export leaked query %q", s.Query)
	}
	if s.Labels != nil {
		t.Errorf("redacted export leaked labels %v", s.Labels)
	}
	if b.Trajectories[0].Metadata != nil {
		t.Errorf("redacted export leaked metadata %v", b.Trajectories[0].Metadata)
	}

	// Full-fidelity: included.
	bf := FromTurns([]trajectory.Turn{x}, Options{FullFidelity: true})
	sf := bf.Trajectories[0].Steps[0]
	if sf.Query != "secret prompt text" {
		t.Errorf("full-fidelity query = %q", sf.Query)
	}
	if sf.Labels["model"] != "claude-opus-4-8" {
		t.Errorf("full-fidelity labels = %v", sf.Labels)
	}
	if bf.Trajectories[0].Metadata["model"] != "claude-opus-4-8" {
		t.Errorf("full-fidelity metadata = %v", bf.Trajectories[0].Metadata)
	}
}

func TestFromTurns_SubagentNestingAndBackref(t *testing.T) {
	parentSpawn := turn("parent", 5, "Agent", "ALLOW")
	child := turn("child", 1, "Bash", "ALLOW")
	child.Labels["parent_tool_use_id"] = "parent:5" // names the exact spawning step

	turns := []trajectory.Turn{
		turn("parent", 1, "Read", "ALLOW"),
		parentSpawn,
		child,
	}
	b := FromTurns(turns, Options{})
	if len(b.Trajectories) != 1 {
		t.Fatalf("child should nest, not be a root; got %d roots", len(b.Trajectories))
	}
	root := b.Trajectories[0]
	if len(root.Subagents) != 1 || root.Subagents[0].TrajectoryID != "child" {
		t.Fatalf("subagent not attached: %+v", root.Subagents)
	}
	// The spawning step (parent:5) must back-reference the child trajectory.
	var spawn *Step
	for i := range root.Steps {
		if root.Steps[i].ToolUseID == "parent:5" {
			spawn = &root.Steps[i]
		}
	}
	if spawn == nil || spawn.SubagentRef != "child" {
		t.Fatalf("spawn step missing subagent ref: %+v", spawn)
	}
}

func TestFromTurns_DanglingParentStaysRoot(t *testing.T) {
	child := turn("child", 1, "Bash", "ALLOW")
	child.Labels["parent_trace_id"] = "ghost" // parent not in corpus
	b := FromTurns([]trajectory.Turn{child}, Options{})
	if len(b.Trajectories) != 1 || b.Trajectories[0].TrajectoryID != "child" {
		t.Fatalf("dangling-parent child must survive as a root: %+v", b.Trajectories)
	}
}

func TestFromTurns_CompactionCounted(t *testing.T) {
	c := turn("t1", 2, "compact", "WITNESS")
	c.Labels["compaction"] = "true"
	turns := []trajectory.Turn{turn("t1", 1, "Read", "ALLOW"), c}
	b := FromTurns(turns, Options{})
	if got := b.Trajectories[0].CompactionEvents; got != 1 {
		t.Fatalf("compaction events = %d, want 1", got)
	}
}

func TestWriteBundle_RoundTrips(t *testing.T) {
	b := FromTurns([]trajectory.Turn{turn("t1", 1, "Bash", "ALLOW")}, Options{})
	var buf bytes.Buffer
	if err := WriteBundle(&buf, b); err != nil {
		t.Fatal(err)
	}
	var back Bundle
	if err := json.Unmarshal(buf.Bytes(), &back); err != nil {
		t.Fatalf("emitted JSON does not parse: %v", err)
	}
	if back.SchemaVersion != SchemaVersion || len(back.Trajectories) != 1 {
		t.Fatalf("round-trip lost data: %+v", back)
	}
}

func TestSplitToolUseID(t *testing.T) {
	cases := []struct {
		in    string
		trace string
		seq   int
		ok    bool
	}{
		{"parent:5", "parent", 5, true},
		{"a:b:12", "a:b", 12, true},
		{"bare", "", 0, false},
		{"trailing:", "", 0, false},
		{"nan:x", "", 0, false},
	}
	for _, c := range cases {
		tr, seq, ok := splitToolUseID(c.in)
		if ok != c.ok || tr != c.trace || seq != c.seq {
			t.Errorf("splitToolUseID(%q) = (%q,%d,%v), want (%q,%d,%v)", c.in, tr, seq, ok, c.trace, c.seq, c.ok)
		}
	}
}
