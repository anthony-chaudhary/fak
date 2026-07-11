package findingsink

import (
	"bufio"
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
)

func TestStdoutSinkPlansEveryFindingAndMutatesNothing(t *testing.T) {
	var buf bytes.Buffer
	sink := StdoutSink{W: &buf}
	findings := []Finding{
		{Key: "k1", Title: "first gap", Summary: "do the first thing"},
		{Key: "k2", Title: "second gap"},
	}
	rep, err := sink.Emit(findings, EmitOptions{})
	if err != nil {
		t.Fatalf("Emit: %v", err)
	}
	if rep.Mode != "dry-run" || rep.Planned != 2 || len(rep.Rows) != 2 {
		t.Fatalf("report = %+v, want dry-run/planned 2/2 rows", rep)
	}
	out := buf.String()
	for _, want := range []string{"k1", "first gap", "do the first thing", "k2"} {
		if !strings.Contains(out, want) {
			t.Errorf("stdout missing %q; got:\n%s", want, out)
		}
	}
}

func readLedger(t *testing.T, path string) []ledgerRecord {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open ledger: %v", err)
	}
	defer f.Close()
	var recs []ledgerRecord
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var rec ledgerRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			t.Fatalf("bad ledger line %q: %v", line, err)
		}
		recs = append(recs, rec)
	}
	return recs
}

// The durable local-db sink is the point of the seam: a re-run must UPSERT by Key, not append a
// duplicate, and must preserve records not in the current batch.
func TestLocalDBSinkUpsertsByKeyAndConverges(t *testing.T) {
	root := t.TempDir()
	sink := LocalDBSink{}
	opt := EmitOptions{Live: true, Dir: root}
	path := filepath.Join(root, ".fak", "checkpoint-findings.jsonl")

	// First run: two findings both created.
	rep, err := sink.Emit([]Finding{
		{Key: "a", Title: "A", Grade: "D"},
		{Key: "b", Title: "B"},
	}, opt)
	if err != nil {
		t.Fatalf("first Emit: %v", err)
	}
	if rep.Created != 2 || rep.Updated != 0 {
		t.Fatalf("first run created/updated = %d/%d, want 2/0", rep.Created, rep.Updated)
	}
	if got := readLedger(t, path); len(got) != 2 {
		t.Fatalf("ledger has %d records after first run, want 2", len(got))
	}

	// Second run: 'a' changed (update, count bumps), 'c' new (create), 'b' omitted (must persist).
	rep, err = sink.Emit([]Finding{
		{Key: "a", Title: "A2", Grade: "F"},
		{Key: "c", Title: "C"},
	}, opt)
	if err != nil {
		t.Fatalf("second Emit: %v", err)
	}
	if rep.Created != 1 || rep.Updated != 1 {
		t.Fatalf("second run created/updated = %d/%d, want 1/1", rep.Created, rep.Updated)
	}

	recs := readLedger(t, path)
	byKey := map[string]ledgerRecord{}
	for _, r := range recs {
		byKey[r.Key] = r
	}
	if len(recs) != 3 {
		t.Fatalf("ledger has %d records after upsert, want 3 (a,b,c): %+v", len(recs), recs)
	}
	if a := byKey["a"]; a.Title != "A2" || a.Grade != "F" || a.Count != 2 {
		t.Errorf("record a = %+v, want title A2 grade F count 2", a)
	}
	if b := byKey["b"]; b.Title != "B" || b.Count != 1 {
		t.Errorf("record b not preserved across a batch that omitted it: %+v", b)
	}
	if c := byKey["c"]; c.Title != "C" || c.Count != 1 {
		t.Errorf("record c = %+v, want title C count 1", c)
	}

	// First-seen order must be stable: a, b, c.
	gotOrder := []string{recs[0].Key, recs[1].Key, recs[2].Key}
	wantOrder := []string{"a", "b", "c"}
	for i := range wantOrder {
		if gotOrder[i] != wantOrder[i] {
			t.Fatalf("ledger order = %v, want first-seen %v", gotOrder, wantOrder)
		}
	}
}

func TestLocalDBSinkDryRunTouchesNoDisk(t *testing.T) {
	root := t.TempDir()
	sink := LocalDBSink{}
	path := filepath.Join(root, ".fak", "checkpoint-findings.jsonl")
	_, err := sink.Emit([]Finding{{Key: "a", Title: "A"}}, EmitOptions{Live: false, Dir: root})
	if err != nil {
		t.Fatalf("Emit: %v", err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("dry-run wrote a ledger at %s (err=%v), want none", path, err)
	}
}

// FromActionItem keeps the rich item attached so the github sink emits a detailed issue, while
// projecting the neutral fields the local/stdout sinks read.
func TestFromActionItemProjectsAndAttaches(t *testing.T) {
	ai := dogfoodissues.ActionItem{
		Key: "checkpoint-debt/x", Title: "x gap", Grade: "D", DebtName: "checkpoint_debt",
		DebtCount: 1, NextAction: "fix x", CurrentState: "x is broken",
		Paths: []string{"internal/x/**"}, Labels: []string{"checkpoint-debt"},
	}
	f := FromActionItem(ai)
	if f.Key != ai.Key || f.Title != ai.Title || f.Grade != ai.Grade || f.Summary != ai.NextAction {
		t.Fatalf("projection lost fields: %+v", f)
	}
	if f.issue == nil || f.issue.CurrentState != "x is broken" {
		t.Fatalf("rich item not attached: %+v", f.issue)
	}
	// A finding with no attached item still synthesizes a usable issue.
	bare := Finding{Key: "k", Title: "t", Summary: "s", Body: "b"}
	got := bare.actionItem("ev")
	if got.Key != "k" || got.NextAction != "s" || got.CurrentState != "b" || got.EvidencePath != "ev" {
		t.Fatalf("synthesized item wrong: %+v", got)
	}
}

// richItem is a fully-specified, dispatchable issue candidate (the shape a real producer emits).
func richItem(key, title string) dogfoodissues.ActionItem {
	return dogfoodissues.ActionItem{
		Key: key, Title: title, SourceProbe: "checkpoint-scorecard", ScoreName: "checkpoint_gap",
		Grade: "D", DebtName: "checkpoint_debt", DebtCount: 1, EvidencePath: "fak checkpoint-scorecard --json",
		NextAction: "Add a durable, resumable WIP store to internal/x.", Finding: key,
		ParentRef:    "fak checkpoint-scorecard",
		CurrentState: "Subsystem x cannot resume its work-in-progress after a mid-task crash.",
		WhyNow:       "Crash-recovery keeps in-flight work from evaporating silently.",
		WorkingSpine: "Persist in-flight state to a durable append-only store and add a resume path.",
		WorkUnit:     "leaf", ExpectedSteps: 4,
		Assumptions:    []string{"The subsystem names its intended durable store."},
		ConfusionRisks: []string{"Do not satisfy the probe with a comment or a test."},
		Coordination:   []string{"One issue owns one gap."},
		Trigger:        "Checkpoint scorecard reports x missing crash recovery.",
		BatchPolicy:    "One issue per gap; reruns update by marker.",
		InScope:        "Persist in-flight state and add a resume path.",
		OutOfScope:     "Do not change the scorecard roster to clear the gap.",
		DoneCondition:  "A re-run no longer lists the " + key + " gap.",
		Witness:        "fak checkpoint-scorecard --json",
		AcceptanceGate: "go build ./... && go test ./internal/checkpointscore",
		Paths:          []string{"internal/x/**"},
		Labels:         []string{"checkpoint-debt", "tech-debt"},
		BoundaryNotes:  []string{"Public subsystem-source evidence only."},
		ClosureBinding: "Resolving commit cites #N and carries a (fak <leaf>) trailer.",
	}
}

// GitHubSink dry-run is pure (no gh): a fully-specified finding plans as create, while an
// under-specified one is routed to skipped rather than emitted as a vague public issue.
func TestGitHubSinkDryRunPlansRichAndSkipsBare(t *testing.T) {
	sink := GitHubSink{}
	rep, err := sink.Emit(FromActionItems([]dogfoodissues.ActionItem{
		richItem("checkpoint-debt/a", "checkpoint: x does not persist resumable WIP state"),
		richItem("checkpoint-debt/b", "checkpoint: y does not persist resumable WIP state"),
	}), EmitOptions{Live: false})
	if err != nil {
		t.Fatalf("Emit: %v", err)
	}
	if rep.Mode != "dry-run" || rep.Planned != 2 {
		t.Fatalf("report = %+v, want dry-run planned 2", rep)
	}
	for _, row := range rep.Rows {
		if row.Action != "create" {
			t.Errorf("row %s action = %q, want create", row.Key, row.Action)
		}
	}

	// A bare finding is under-specified: it must be skipped, never emitted as a vague issue.
	bare, err := sink.Emit([]Finding{{Key: "checkpoint-debt/bare", Title: "bare"}}, EmitOptions{Live: false})
	if err != nil {
		t.Fatalf("bare Emit: %v", err)
	}
	if bare.Planned != 0 || bare.Skipped != 1 {
		t.Fatalf("bare report = %+v, want planned 0 skipped 1", bare)
	}
}
