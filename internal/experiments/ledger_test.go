package experiments

import (
	"strings"
	"testing"
)

func TestExperimentCells(t *testing.T) {
	e := Experiment{
		ID:       "test-1",
		Owner:    "alice",
		Host:     "host1",
		Models:   []string{"qwen2.5-1.5b", "qwen2.5-7b"},
		Backends: []string{"cpu", "cuda"},
		Started:  "2026-06-27T10:00:00Z",
	}
	cells := e.Cells()
	if len(cells) != 4 {
		t.Fatalf("expected 4 cells, got %d", len(cells))
	}
}

func TestExperimentOverlaps(t *testing.T) {
	e1 := Experiment{
		ID:       "exp-1",
		Owner:    "alice",
		Host:     "host1",
		Models:   []string{"qwen2.5-1.5b"},
		Backends: []string{"cpu"},
		Started:  "2026-06-27T10:00:00Z",
	}
	e2 := Experiment{
		ID:       "exp-2",
		Owner:    "bob",
		Host:     "host2",
		Models:   []string{"qwen2.5-1.5b"},
		Backends: []string{"cuda"},
		Started:  "2026-06-27T10:00:00Z",
	}
	e3 := Experiment{
		ID:       "exp-3",
		Owner:    "carol",
		Host:     "host3",
		Models:   []string{"qwen2.5-1.5b"},
		Backends: []string{"cpu"},
		Started:  "2026-06-27T10:00:00Z",
	}
	if e1.Overlaps(e2) {
		t.Error("e1 and e2 should not overlap")
	}
	if !e1.Overlaps(e3) {
		t.Error("e1 and e3 should overlap")
	}
}

func TestParseLedger(t *testing.T) {
	content := `{"id":"exp-1","owner":"alice","host":"host1","models":["qwen2.5-1.5b"],"backends":["cpu"],"started":"2026-06-27T10:00:00Z","artifact_path":"exp1.json"}
{"id":"exp-2","owner":"bob","host":"host2","models":["qwen2.5-7b"],"backends":["cuda"],"started":"2026-06-27T10:00:00Z","artifact_path":"exp2.json"}
`
	rows := ParseLedger(content)
	if len(rows) != 2 {
		t.Fatalf("expected 2 rows, got %d", len(rows))
	}
	if rows[0].ID != "exp-1" {
		t.Errorf("expected exp-1, got %s", rows[0].ID)
	}
}

func TestFindOverlaps(t *testing.T) {
	exps := []Experiment{
		{
			ID:       "exp-1",
			Owner:    "alice",
			Host:     "host1",
			Models:   []string{"qwen2.5-1.5b"},
			Backends: []string{"cpu"},
			Started:  "2026-06-27T10:00:00Z",
		},
		{
			ID:       "exp-2",
			Owner:    "bob",
			Host:     "host2",
			Models:   []string{"qwen2.5-7b"},
			Backends: []string{"cuda"},
			Started:  "2026-06-27T10:00:00Z",
		},
		{
			ID:       "exp-3",
			Owner:    "carol",
			Host:     "host3",
			Models:   []string{"qwen2.5-1.5b"},
			Backends: []string{"cpu"},
			Started:  "2026-06-27T10:00:00Z",
		},
	}
	overlaps := FindOverlaps(exps, []string{"qwen2.5-1.5b"}, []string{"cpu"})
	if len(overlaps) != 2 {
		t.Fatalf("expected 2 overlaps, got %d", len(overlaps))
	}
}

func TestAppendLedgerLine(t *testing.T) {
	exp := Experiment{
		ID:           "test-1",
		Owner:        "alice",
		Host:         "host1",
		Models:       []string{"qwen2.5-1.5b"},
		Backends:     []string{"cpu"},
		Started:      "2026-06-27T10:00:00Z",
		ArtifactPath: "test.json",
	}
	line, err := AppendLedgerLine(exp)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(line, `"id":"test-1"`) {
		t.Errorf("line missing id: %s", line)
	}
}
