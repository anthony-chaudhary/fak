package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fixtureForTest(t *testing.T) Fixture {
	t.Helper()
	f, err := BuildFixture()
	if err != nil {
		t.Fatal(err)
	}
	return f
}

func TestRoundTripPortableFixture(t *testing.T) {
	f := fixtureForTest(t)
	path := filepath.Join(t.TempDir(), "fixture.json")
	if err := WriteFixture(path, f); err != nil {
		t.Fatal(err)
	}
	got, err := ReadFixture(path)
	if err != nil {
		t.Fatal(err)
	}
	a, _ := json.Marshal(f)
	b, _ := json.Marshal(got)
	if string(a) != string(b) {
		t.Fatal("JSON round trip changed fixture")
	}
}

func TestScrubReportAndNoLeakage(t *testing.T) {
	f := fixtureForTest(t)
	b, _ := json.Marshal(f)
	for _, forbidden := range []string{"sk-live-DO-NOT-SERIALIZE", "alex@example.com"} {
		if strings.Contains(string(b), forbidden) {
			t.Fatalf("serialized secret survived: %s", forbidden)
		}
	}
	if f.ScrubReport.Replacements != 3 || len(f.ScrubReport.Paths) != 3 {
		t.Fatalf("scrub report = %+v", f.ScrubReport)
	}
	path := filepath.Join(t.TempDir(), "fixture.json")
	if err := WriteFixture(path, f); err != nil {
		t.Fatal(err)
	}
	written, _ := os.ReadFile(path)
	if secretPattern.Match(written) || emailPattern.Match(written) {
		t.Fatal("sensitive pattern survived written fixture")
	}
}

func TestReplayAcrossThreeProducts(t *testing.T) {
	f := fixtureForTest(t)
	got, outcome, err := Replay(f)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"reference", "ops-console", "pickup-card"} {
		if _, ok := got[name]; !ok {
			t.Fatalf("missing %s projection", name)
		}
	}
	if outcome != f.ExpectedOutcome {
		t.Fatalf("outcome %+v, want %+v", outcome, f.ExpectedOutcome)
	}
	if err := Compare(f.Expected, got, Strict, f.Nondeterminism); err != nil {
		t.Fatal(err)
	}
}

func TestStrictAndTolerantModes(t *testing.T) {
	f := fixtureForTest(t)
	actual, _, _ := Replay(f)
	q := actual["reference"]
	q.Meta["observed_at"] = "2099-01-01T00:00:00Z"
	actual["reference"] = q
	if Compare(f.Expected, actual, Strict, f.Nondeterminism) == nil {
		t.Fatal("strict comparison accepted time change")
	}
	if err := Compare(f.Expected, actual, Tolerant, f.Nondeterminism); err != nil {
		t.Fatalf("tolerant comparison rejected declared time field: %v", err)
	}
	q = actual["reference"]
	q.Status = "failed"
	actual["reference"] = q
	if Compare(f.Expected, actual, Tolerant, f.Nondeterminism) == nil {
		t.Fatal("tolerant comparison hid semantic status change")
	}
}

func TestSeededNondeterminism(t *testing.T) {
	f := fixtureForTest(t)
	a, _, _ := Replay(f)
	b, _, _ := Replay(f)
	if err := Compare(a, b, Strict, f.Nondeterminism); err != nil {
		t.Fatal(err)
	}
	f.Nondeterminism[1].Seed++
	c, _, _ := Replay(f)
	if Compare(a, c, Strict, f.Nondeterminism) == nil {
		t.Fatal("changed random seed produced identical strict output")
	}
	if err := Compare(a, c, Tolerant, f.Nondeterminism); err != nil {
		t.Fatalf("declared random field was not tolerated: %v", err)
	}
}

func TestBehavioralMutationCaught(t *testing.T) {
	f := fixtureForTest(t)
	var p map[string]any
	if err := json.Unmarshal(f.Events[2].Payload, &p); err != nil {
		t.Fatal(err)
	}
	p["text"] = "Order was canceled."
	f.Events[2].Payload, _ = json.Marshal(p)
	actual, _, err := Replay(f)
	if err != nil {
		t.Fatal(err)
	}
	if Compare(f.Expected, actual, Tolerant, f.Nondeterminism) == nil {
		t.Fatal("behavioral mutation was hidden")
	}
}

func TestProjectionMutationCaught(t *testing.T) {
	f := fixtureForTest(t)
	actual, _, _ := Replay(f)
	q := actual["pickup-card"]
	q.Lines[0] = "Order 42 — canceled"
	actual["pickup-card"] = q
	if Compare(f.Expected, actual, Tolerant, f.Nondeterminism) == nil {
		t.Fatal("UI projection mutation was hidden")
	}
}
