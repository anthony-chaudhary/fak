package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/logitoracle"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
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

// A host reference of a checkpoint larger than RAM (DeepSeek-V4.1 Flash Q2_K on
// CPU) needs the serve --cpu-offload-experts arm and a bounded KV budget; the
// dump must thread both to the serve loader instead of hard-coding false/0.
func TestComputeTraceLogitsDumpThreadsOffloadAndContext(t *testing.T) {
	dir := t.TempDir()
	prompts := filepath.Join(dir, "p.json")
	if err := os.WriteFile(prompts, []byte(`[{"name":"P1","ids":[3085],"positions":1}]`), 0o644); err != nil {
		t.Fatal(err)
	}
	var gotOffload bool
	gotContext := -1
	prev := logitDumpLoadModel
	logitDumpLoadModel = func(_ string, _ compute.Backend, offload bool, ctx int, _ *ggufload.ExpertShard, _ int, _ *serveFitBudget) (*fakmodel.Model, bool, *gateway.ModelLoadProfile, gateway.StartupPhase) {
		gotOffload, gotContext = offload, ctx
		return nil, false, nil, gateway.StartupPhase{}
	}
	defer func() { logitDumpLoadModel = prev }()
	var stdout, stderr bytes.Buffer
	code := runComputeTraceLogitsDump(&stdout, &stderr, []string{"--gguf", filepath.Join(dir, "m.gguf"), "--prompts", prompts,
		"--out", filepath.Join(dir, "out"), "--backend", "cpu", "--cpu-offload-experts", "--native-context-tokens", "512"})
	if code != 1 || !gotOffload || gotContext != 512 {
		t.Fatalf("code=%d offload=%v context=%d stderr=%s", code, gotOffload, gotContext, stderr.String())
	}
}
