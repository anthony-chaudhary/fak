package model

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// withV41ActivationTraceEnv points the cached FAK_V41_ACTIVATION_TRACE lookup at
// path for one test.
func withV41ActivationTraceEnv(t *testing.T, path string) {
	t.Helper()
	prev := v41ActivationTraceEnvPath
	v41ActivationTraceEnvPath = func() string { return path }
	t.Cleanup(func() { v41ActivationTraceEnvPath = prev })
}

func TestV41ActivationTraceEnv(t *testing.T) {
	// The reduced fixture admits one layer; a second needs an Engram row source.
	const layers = 1
	m := v41ReducedModel(t)
	ids := []int{1, 3, 5}
	H := m.Cfg.HiddenSize
	base := m.Forward(ids)

	t.Run("unset writes nothing", func(t *testing.T) {
		withV41ActivationTraceEnv(t, "")
		if finish := m.v41ActivationTraceFromEnv(ids); finish != nil {
			t.Fatal("an unset variable must return no finisher")
		}
	})

	t.Run("set writes a complete per-layer trace with residual streams", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "v41-trace.json")
		withV41ActivationTraceEnv(t, path)
		got := m.Forward(ids)
		if !v41LogitsEqual(base, got) {
			t.Fatal("env tracing changed the forward logits")
		}
		if computetrace.Enabled() {
			t.Fatal("the env recorder stayed enabled after the forward")
		}
		f, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		a, err := computetrace.Read(f)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(a.InputTokens, ids) || a.Dropped != 0 || a.DroppedSampleValues != 0 {
			t.Fatalf("tokens=%v dropped=%d dropped_samples=%d", a.InputTokens, a.Dropped, a.DroppedSampleValues)
		}
		if want := layers * 5 * len(ids); len(a.Events) != want {
			t.Fatalf("events=%d, want %d (layers x 5 stages x tokens)", len(a.Events), want)
		}
		residual := 0
		for _, e := range a.Events {
			if e.Phase != "residual_streams" {
				continue
			}
			residual++
			if e.Shapes[0][0] != 4*H || !slices.Equal(e.SampleIndices, computetrace.ActivationSampleIndices(4*H, 8)) {
				t.Fatalf("residual layer=%d token=%d shape=%v indices=%v", e.Layer, e.Token, e.Shapes, e.SampleIndices)
			}
			// The reduced graph carries stream 0 as x and zero side streams, and
			// its per-layer hidden is flatten(x) after the layer.
			hidden := got.Hidden[e.Layer+1][e.Token*H : (e.Token+1)*H]
			for i, j := range e.SampleIndices {
				want := float32(0)
				if j < H {
					want = hidden[j]
				}
				if e.SampleValues[i] != want {
					t.Fatalf("residual layer=%d token=%d j=%d = %v, want %v", e.Layer, e.Token, j, e.SampleValues[i], want)
				}
			}
		}
		if residual != layers*len(ids) {
			t.Fatalf("residual events=%d, want %d", residual, layers*len(ids))
		}

		self := computetrace.ActivationErrorProfile(a, a, computetrace.ActivationTolerance{})
		if self.Verdict != computetrace.TraceEqual || len(self.Rows) != layers*5 {
			t.Fatalf("self profile verdict=%s rows=%d", self.Verdict, len(self.Rows))
		}
	})

	t.Run("an explicitly enabled recorder is not clobbered", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "unused.json")
		withV41ActivationTraceEnv(t, path)
		rec, disable := computetrace.Enable(4096, "run", "req")
		defer disable()
		_ = m.Forward(ids)
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("env trace written while another recorder was enabled: %v", err)
		}
		if n := len(rec.Artifact().Events); n == 0 {
			t.Fatal("the explicit recorder lost the forward's events")
		}
	})
}

// TestV41ActivationTraceEnvKernelEvents is the strix3 regression: per-kernel
// matmul events share the process recorder and used to spend the env budget,
// so only layers 0-6 of a 41-layer, 7-token prefill survived.
func TestV41ActivationTraceEnvKernelEvents(t *testing.T) {
	const layers, tokens = 41, 7
	m := v41ReducedModel(t)
	m.Cfg.NumLayers = layers
	ids := make([]int, tokens)
	path := filepath.Join(t.TempDir(), "v41-trace.json")
	withV41ActivationTraceEnv(t, path)

	finish := m.v41ActivationTraceFromEnv(ids)
	if finish == nil {
		t.Fatal("env recorder not enabled")
	}
	tracer := m.v41ActivationTracer(ids)
	stages := []string{v41TraceStageQLatent, v41TraceStageKVLatent, v41TraceStageAttnOut, v41TraceStageMoESum}
	row := make([]float32, 64)
	for l := 0; l < layers; l++ {
		for _, stage := range stages {
			for tok := 0; tok < tokens; tok++ {
				for k := 0; k < 8; k++ {
					computetrace.Record(computetrace.Event{Operation: "matmul", Phase: "kernel"})
				}
				tracer.record(l, tok, stage, row)
			}
		}
		for tok := 0; tok < tokens; tok++ {
			tracer.recordStreams(l, tok, [][]float32{row, row, row, row})
		}
	}
	if err := finish(nil); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	a, err := computetrace.Read(f)
	if err != nil {
		t.Fatal(err)
	}
	if want := layers * 5 * tokens; len(a.Events) != want || a.Dropped != 0 || a.DroppedSampleValues != 0 {
		t.Fatalf("events=%d dropped=%d dropped_samples=%d, want %d/0/0", len(a.Events), a.Dropped, a.DroppedSampleValues, want)
	}
	seen := map[[2]int]int{}
	for _, e := range a.Events {
		if e.Operation != v41ActivationTraceOperation {
			t.Fatalf("env trace retained a %q event", e.Operation)
		}
		seen[[2]int{e.Layer, e.Token}]++
	}
	for l := 0; l < layers; l++ {
		for tok := 0; tok < tokens; tok++ {
			if seen[[2]int{l, tok}] != 5 {
				t.Fatalf("layer=%d token=%d has %d stages, want 5", l, tok, seen[[2]int{l, tok}])
			}
		}
	}
}
