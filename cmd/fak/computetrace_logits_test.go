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

// fak-test:justify why=regression when=changed:cmd/fak/computetrace*.go
// fak-test:runtime fast est=20ms lane=default
func TestComputeTraceLogitsCompareKLBaselineGate(t *testing.T) {
	root := t.TempDir()
	ref, twin, near, far := filepath.Join(root, "ref"), filepath.Join(root, "twin"), filepath.Join(root, "near"), filepath.Join(root, "far")
	writeLogitFixture(t, ref, 0)
	writeLogitFixture(t, twin, 0.002)
	writeLogitFixture(t, near, 0.002)
	writeLogitFixture(t, far, 0.01)
	var out, errOut bytes.Buffer
	pair := filepath.Join(root, "pair.json")
	if code := runComputeTrace(&out, &errOut, []string{"logits", "compare", "--ref", ref, "--cand", twin, "--json", "--out", pair}); code != 0 {
		t.Fatalf("baseline pair exit=%d stderr=%s", code, errOut.String())
	}
	run := func(cand string, extra ...string) (int, logitoracle.Report) {
		out.Reset()
		code := runComputeTrace(&out, &errOut, append([]string{"logits", "compare", "--ref", ref, "--cand", cand, "--json", "--kl-baseline", pair}, extra...))
		var rep logitoracle.Report
		if code != 2 && out.Len() > 0 {
			if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
				t.Fatal(err)
			}
		}
		return code, rep
	}
	// A candidate as noisy as the baseline pair passes; one well past the floor fails on KL alone.
	if code, rep := run(near); code != 0 || rep.KLNoise.Verdict != logitoracle.KLNoiseWithin || rep.Thresholds.KLNoise == nil || rep.KLNoise.Multiple != 2 {
		t.Fatalf("near: exit=%d kl=%+v stderr=%s", code, rep.KLNoise, errOut.String())
	}
	code, rep := run(far, "--kl-multiple", "1.5")
	if code != 1 || rep.KLNoise.Verdict != logitoracle.KLNoiseExceeded || rep.Verdict != logitoracle.VerdictFail || rep.KLNoise.Multiple != 1.5 {
		t.Fatalf("far: exit=%d verdict=%s kl=%+v", code, rep.Verdict, rep.KLNoise)
	}
	// A bit-identical pair has no noise to scale by; a missing report is refused before comparing.
	same := filepath.Join(root, "same.json")
	if code := runComputeTrace(&out, &errOut, []string{"logits", "compare", "--ref", ref, "--cand", ref, "--json", "--out", same}); code != 0 {
		t.Fatalf("identical pair exit=%d", code)
	}
	for _, bad := range []string{same, filepath.Join(root, "missing.json"), filepath.Join(ref, "manifest.json")} {
		errOut.Reset()
		out.Reset()
		if code := runComputeTrace(&out, &errOut, []string{"logits", "compare", "--ref", ref, "--cand", near, "--kl-baseline", bad}); code != 1 || out.Len() != 0 || errOut.Len() == 0 {
			t.Fatalf("baseline %s: exit=%d out=%q", bad, code, out.String())
		}
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

func TestConfigureLogitDumpSessionCarriesCPUOffloadExperts(t *testing.T) {
	for _, offload := range []bool{false, true} {
		s := &fakmodel.Session{}
		configureLogitDumpSession(s, true, offload)
		if !s.Quant || !s.Q4K {
			t.Fatalf("offload=%v: Quant=%v Q4K=%v, want both true", offload, s.Quant, s.Q4K)
		}
		if s.CPUOffloadExperts != offload {
			t.Fatalf("CPUOffloadExperts=%v, want %v: the dump must free ring-less offload stages per call as serve does", s.CPUOffloadExperts, offload)
		}
	}
}
