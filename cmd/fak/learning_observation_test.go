package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/learningobservation"
)

func TestLearningObservationGolden(t *testing.T) {
	storePath := filepath.Join(t.TempDir(), "observations.json")
	add := func(kind, source, content, outcome string) string {
		t.Helper()
		args := []string{"add", "--store", storePath, "--kind", kind, "--source", source, "--content", content}
		if outcome != "" {
			args = append(args, "--outcome", outcome)
		}
		var out, errb bytes.Buffer
		if code := runLearningObservation(&out, &errb, args); code != 0 {
			t.Fatalf("add exit=%d stderr=%s", code, errb.String())
		}
		store, err := learningobservation.Load(storePath)
		if err != nil {
			t.Fatal(err)
		}
		for _, record := range store.Records {
			if record.Source == source {
				return record.ID
			}
		}
		t.Fatalf("record for %s missing; output=%s", source, out.String())
		return ""
	}
	link := func(from, relation, to string) (int, string) {
		t.Helper()
		var out, errb bytes.Buffer
		code := runLearningObservation(&out, &errb, []string{"link", "--store", storePath, "--from", from, "--relation", relation, "--to", to})
		return code, errb.String()
	}
	trace := func(candidate string) string {
		t.Helper()
		var out, errb bytes.Buffer
		if code := runLearningObservation(&out, &errb, []string{"trace", "--store", storePath, "--candidate", candidate}); code != 0 {
			t.Fatalf("trace exit=%d stderr=%s", code, errb.String())
		}
		return strings.TrimSpace(out.String())
	}

	keptCandidate := add("candidate", "candidate://keep", "keep bounded retry", "")
	keptWitness := add("witness", "replay://keep-1", "independent replay passed", "")
	keptVerdict := add("verdict", "verdict://keep", "admission retained", "kept")
	if code, errb := link(keptCandidate, "tested-by", keptWitness); code != 0 {
		t.Fatal(errb)
	}
	if code, errb := link(keptWitness, "kept-as", keptVerdict); code != 0 {
		t.Fatal(errb)
	}

	rejectedCandidate := add("candidate", "candidate://reject", "unbounded retry", "")
	rejectedWitness := add("witness", "replay://reject-1", "independent replay regressed", "")
	rejectedVerdict := add("verdict", "verdict://reject", "admission reverted", "rejected")
	if code, errb := link(rejectedCandidate, "tested-by", rejectedWitness); code != 0 {
		t.Fatal(errb)
	}
	if code, errb := link(rejectedWitness, "rejected-as", rejectedVerdict); code != 0 {
		t.Fatal(errb)
	}

	code, denial := link(rejectedCandidate, "tested-by", "lo_missing")
	if code == 0 || !strings.Contains(denial, "dangling id") {
		t.Fatalf("dangling code=%d stderr=%q", code, denial)
	}
	got := trace(keptCandidate) + "\n" + trace(rejectedCandidate) + "\n" + strings.TrimSpace(denial) + "\n"
	want, err := os.ReadFile(filepath.Join("testdata", "learning_observation.golden"))
	if err != nil {
		t.Fatal(err)
	}
	if got != string(want) {
		t.Fatalf("golden mismatch\n--- got\n%s--- want\n%s", got, want)
	}
}

func TestLearningObservationDuplicateAndConflict(t *testing.T) {
	storePath := filepath.Join(t.TempDir(), "store.json")
	run := func(content string) (int, string, string) {
		var out, errb bytes.Buffer
		code := runLearningObservation(&out, &errb, []string{"add", "--store", storePath, "--kind", "observation", "--source", "trajectory://7", "--content", content})
		return code, out.String(), errb.String()
	}
	code, first, errb := run("tool recovered")
	if code != 0 {
		t.Fatal(errb)
	}
	code, second, errb := run("tool   recovered")
	if code != 0 || first == second || !strings.Contains(second, `"created":false`) {
		t.Fatalf("duplicate code=%d first=%s second=%s stderr=%s", code, first, second, errb)
	}
	code, _, errb = run("tool failed")
	if code == 0 || !strings.Contains(errb, "conflicting content") {
		t.Fatalf("conflict code=%d stderr=%s", code, errb)
	}
}
