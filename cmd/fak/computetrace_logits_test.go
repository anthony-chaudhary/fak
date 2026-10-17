package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/logitoracle"
)

type logitFixtureSession struct {
	n      int
	offset float32
}

func (s *logitFixtureSession) row() []float32 {
	out := make([]float32, 8)
	for i := range out {
		out[i] = float32((i*7+s.n*3)%11) + s.offset*float32(i)
	}
	return out
}

func (s *logitFixtureSession) PrefillNoLogits(ids []int) { s.n += len(ids) }
func (s *logitFixtureSession) Prefill(ids []int) []float32 {
	s.n += len(ids)
	return s.row()
}
func (s *logitFixtureSession) Step(int) []float32 {
	s.n++
	return s.row()
}

func writeLogitFixture(t *testing.T, dir string, offset float32) {
	t.Helper()
	_, err := logitoracle.Dump([]logitoracle.PromptSpec{{Name: "p", IDs: []int{1, 2, 3}, Positions: 4}}, logitoracle.DumpOptions{
		Dir: dir, Label: filepath.Base(dir),
		NewSession: func() (logitoracle.Session, func(), error) { return &logitFixtureSession{offset: offset}, nil, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
}

// fak-test:justify why=regression when=changed:cmd/fak/computetrace*.go
// fak-test:runtime fast est=20ms lane=default
func TestComputeTraceLogitsCompareExitCodes(t *testing.T) {
	root := t.TempDir()
	ref, near, far := filepath.Join(root, "ref"), filepath.Join(root, "near"), filepath.Join(root, "far")
	writeLogitFixture(t, ref, 0)
	writeLogitFixture(t, near, 1e-7)
	writeLogitFixture(t, far, 0.5)

	var out, errOut bytes.Buffer
	report := filepath.Join(root, "report.json")
	if code := runComputeTrace(&out, &errOut, []string{"logits", "compare", "--ref", ref, "--cand", near, "--json", "--out", report}); code != 0 {
		t.Fatalf("near compare exit=%d stderr=%s", code, errOut.String())
	}
	var rep logitoracle.Report
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatal(err)
	}
	if rep.Schema != logitoracle.ReportSchema || rep.Verdict != logitoracle.VerdictPass || rep.Prompts[0].Compared != 4 {
		t.Fatalf("report %+v", rep)
	}
	out.Reset()
	if code := runComputeTrace(&out, &errOut, []string{"logits", "compare", "--ref", ref, "--cand", far}); code != 1 {
		t.Fatalf("far compare exit=%d", code)
	}
	if !strings.Contains(out.String(), "verdict="+logitoracle.VerdictFail) {
		t.Fatalf("text report %q", out.String())
	}
	if code := runComputeTrace(&out, &errOut, []string{"logits", "compare", "--ref", ref}); code != 2 {
		t.Fatalf("missing --cand exit=%d", code)
	}
	if code := runComputeTrace(&out, &errOut, []string{"logits", "dump"}); code != 2 {
		t.Fatalf("bare dump exit=%d", code)
	}
}
