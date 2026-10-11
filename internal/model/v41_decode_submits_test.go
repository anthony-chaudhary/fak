package model

// Per-step execution-counter witness harness for V4.1 decode (#13820 one command
// buffer per step, #13821 device residency). The device run lives in
// v41_decode_submits_vulkan_test.go; this file holds the shared harness, the
// layer-sweep fixture and the cpu-ref half of the witness.

import (
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const (
	v41DecodeSubmitsSteps     = 4
	v41DecodeSubmitsCPUTol    = 2e-3
	v41DecodeSubmitsEngramRow = 408
	v41DecodeSubmitsSliding   = 128
)

type v41DecodeSubmitsWindow struct {
	phase     string
	step      int
	tokens    int
	wall      time.Duration
	token     int
	logits    []float32
	route     string
	delta     compute.BackendExecutionObservation
	liveBytes uint64
}

type v41DecodeSubmitsTrace struct {
	observed bool
	windows  []v41DecodeSubmitsWindow
	decode   compute.BackendExecutionObservation
	run      compute.BackendExecutionObservation
}

func v41DecodeSubmitsPrompt(n, vocab int) []int {
	prompt := make([]int, n)
	for i := range prompt {
		prompt[i] = 1 + (i*3)%(vocab-1)
	}
	return prompt
}

// v41DecodeSubmitsMeasure prefills prompt and greedily decodes steps tokens,
// bracketing the prefill and every decode step with a backend execution
// snapshot, and logs each window as it closes. A nil backend records wall time
// and logits only.
func v41DecodeSubmitsMeasure(t *testing.T, label string, s *Session, be compute.Backend, prompt []int, steps int) v41DecodeSubmitsTrace {
	t.Helper()
	tr := v41DecodeSubmitsTrace{observed: be != nil}
	record := func(w v41DecodeSubmitsWindow) {
		tr.windows = append(tr.windows, w)
		v41DecodeSubmitsLog(t, label, w)
	}
	snap := func(at string) compute.BackendExecutionSnapshot {
		if !tr.observed {
			return compute.BackendExecutionSnapshot{}
		}
		got, ok, err := compute.CaptureBackendExecutionSnapshot(be)
		if err != nil || !ok {
			t.Fatalf("backend execution snapshot %s: available=%t err=%v", at, ok, err)
		}
		return got
	}
	delta := func(at string, before, after compute.BackendExecutionSnapshot) compute.BackendExecutionObservation {
		if !tr.observed {
			return compute.BackendExecutionObservation{}
		}
		got, err := compute.BackendExecutionDelta(before, after)
		if err != nil {
			t.Fatalf("backend execution delta %s: %v", at, err)
		}
		return got
	}

	incremental := 0
	prior := v41IncrementalStepProbe
	v41IncrementalStepProbe = func() {
		incremental++
		if prior != nil {
			prior()
		}
	}
	defer func() { v41IncrementalStepProbe = prior }()

	runStart := snap("before prefill")
	began := time.Now()
	logits := append([]float32(nil), s.Prefill(prompt)...)
	wall := time.Since(began)
	afterPrefill := snap("after prefill")
	record(v41DecodeSubmitsWindow{
		phase: "prefill", tokens: len(prompt), wall: wall, token: argmaxF32(logits), logits: logits,
		route: "full", delta: delta("prefill", runStart, afterPrefill), liveBytes: afterPrefill.DeviceAllocationLiveBytes,
	})
	prev := afterPrefill
	for i := 0; i < steps; i++ {
		next := tr.windows[len(tr.windows)-1].token
		seen := incremental
		began = time.Now()
		logits = append([]float32(nil), s.Step(next)...)
		wall = time.Since(began)
		after := snap(fmt.Sprintf("after decode step %d", i))
		route := "refold"
		if incremental > seen {
			route = "incremental"
		}
		record(v41DecodeSubmitsWindow{
			phase: "decode", step: i, tokens: 1, wall: wall, token: argmaxF32(logits), logits: logits,
			route: route, delta: delta(fmt.Sprintf("decode step %d", i), prev, after), liveBytes: after.DeviceAllocationLiveBytes,
		})
		prev = after
	}
	tr.decode = delta("decode total", afterPrefill, prev)
	tr.run = delta("run total", runStart, prev)
	return tr
}

// v41DecodeSubmitsAdd sums every uint64 counter field, so a counter added to
// compute.BackendCounterSnapshot later is covered by the split check unchanged.
func v41DecodeSubmitsAdd(a, b compute.BackendCounterSnapshot) compute.BackendCounterSnapshot {
	out := reflect.ValueOf(&a).Elem()
	in := reflect.ValueOf(b)
	for i := 0; i < out.NumField(); i++ {
		if f := out.Field(i); f.Kind() == reflect.Uint64 {
			f.SetUint(f.Uint() + in.Field(i).Uint())
		}
	}
	return a
}

// v41DecodeSubmitsCheck asserts the stable invariants only: finite logits per
// window and, when counters are observed, present per-step device work and a
// per-step split that sums exactly to the decode and run totals. It never
// bounds a count, a byte volume or a wall time.
func v41DecodeSubmitsCheck(t *testing.T, tr v41DecodeSubmitsTrace) {
	t.Helper()
	for _, w := range tr.windows {
		if len(w.logits) == 0 {
			t.Fatalf("%s step %d returned no logits", w.phase, w.step)
		}
		for j, v := range w.logits {
			if !finite32(v) {
				t.Fatalf("%s step %d logit %d nonfinite", w.phase, w.step, j)
			}
		}
	}
	if !tr.observed {
		return
	}
	var decode, run compute.BackendCounterSnapshot
	for _, w := range tr.windows {
		if !w.delta.TransferCountersObserved {
			t.Errorf("%s step %d: transfer counters not observed", w.phase, w.step)
		}
		run = v41DecodeSubmitsAdd(run, w.delta.Counters)
		if w.phase != "decode" {
			continue
		}
		decode = v41DecodeSubmitsAdd(decode, w.delta.Counters)
		if c := w.delta.Counters; c.ComputeDispatches == 0 || c.DispatchSubmits == 0 {
			t.Errorf("decode step %d recorded no device work: compute_dispatches=%d dispatch_submits=%d", w.step, c.ComputeDispatches, c.DispatchSubmits)
		}
	}
	if decode != tr.decode.Counters {
		t.Errorf("per-step decode split does not sum to the decode total:\n split=%+v\n total=%+v", decode, tr.decode.Counters)
	}
	if run != tr.run.Counters {
		t.Errorf("prefill+decode split does not sum to the run total:\n split=%+v\n total=%+v", run, tr.run.Counters)
	}
}

func v41DecodeSubmitsLog(t *testing.T, label string, w v41DecodeSubmitsWindow) {
	t.Helper()
	c := w.delta.Counters
	t.Logf("v41_decode_submits %s phase=%s step=%d tokens=%d route=%s wall_ms=%.3f submits=%d dispatches=%d q4k_dispatches=%d h2d_bytes=%d h2d_count=%d d2h_bytes=%d d2h_count=%d d2d_copies=%d d2d_bytes=%d home_hits=%d home_admissions=%d live_bytes=%d token=%d",
		label, w.phase, w.step, w.tokens, w.route, float64(w.wall.Microseconds())/1000,
		c.DispatchSubmits, c.ComputeDispatches, c.Q4KMatmulDispatches, c.H2DBytes, c.H2DCount, c.D2HBytes, c.D2HCount,
		c.D2DCopies, c.D2DBytes, c.TensorHomeHits, c.TensorHomeAdmissions, w.liveBytes, w.token)
}

// v41DecodeSubmitsSteady averages decode steps after the first, which can carry
// one-time session binding work.
func v41DecodeSubmitsSteady(tr v41DecodeSubmitsTrace) (wallMS float64, c [6]float64, n int) {
	for _, w := range tr.windows {
		if w.phase != "decode" || w.step == 0 {
			continue
		}
		d := w.delta.Counters
		wallMS += float64(w.wall.Microseconds()) / 1000
		for i, v := range []uint64{d.DispatchSubmits, d.ComputeDispatches, d.H2DBytes, d.H2DCount, d.D2HBytes, d.D2HCount} {
			c[i] += float64(v)
		}
		n++
	}
	if n == 0 {
		return 0, c, 0
	}
	for i := range c {
		c[i] /= float64(n)
	}
	return wallMS / float64(n), c, n
}

// v41DecodeSubmitsLayeredFixture extends the one-layer full-geometry Engram
// model to `layers` copies of layer 0 with Engram on layer 0 only, the same
// construction as v41IncrementalEngramFixture. A layer sweep then separates the
// per-layer cost from the head and the Engram add-on. Its 128-row window is the
// pinned config's sliding_window, so decode past 128 tokens stays on the
// incremental ring; v41IncrementalEngramFixture's global window refolds there.
func v41DecodeSubmitsLayeredFixture(t *testing.T, layers int, compressed bool) *Model {
	t.Helper()
	m, _ := v41FullEngramModel(t)
	const prefix = "model.layers.0."
	for l := 1; l < layers; l++ {
		for name, meta := range m.manifest {
			if strings.HasPrefix(name, prefix) {
				m.manifest[layerName(l, name[len(prefix):])] = meta
			}
		}
	}
	m.Cfg.NumLayers = layers
	m.Cfg.Window = make([]int, layers)
	for l := range m.Cfg.Window {
		m.Cfg.Window[l] = v41DecodeSubmitsSliding
	}
	d := m.Cfg.DeepSeekV41
	d.CompressRatios = make([]int, layers)
	d.KVSourceLayerIDs = nil
	d.IndexSourceLayerIDs = nil
	d.EngramLayerIDs = []int{0}
	d.EngramNumEmbeddings = []int{v41DecodeSubmitsEngramRow}
	if compressed {
		H := m.Cfg.HiddenSize
		var tensors []synthTensor
		for l := 0; l < layers; l++ {
			d.CompressRatios[l] = 2
			tensors = append(tensors,
				synthTensor{layerName(l, "attn.compressor.wkv.weight"), []int{v41KVLoraRank, H}},
				synthTensor{layerName(l, "attn.compressor.wgate.weight"), []int{v41KVLoraRank, H}},
				synthTensor{layerName(l, "attn.compressor.norm.weight"), []int{v41KVLoraRank}},
			)
		}
		man, raw := synthBuildRaw(tensors, func(name string, next func() float32) float32 {
			if hasSuffix(name, "compressor.norm.weight") {
				return 1
			}
			return synthMatmulFill(name, next)
		})
		for name, meta := range man {
			meta.Offset += len(m.raw)
			m.manifest[name] = meta
		}
		m.raw = append(m.raw, raw...)
	}
	layout := V41EngramLayout{
		TokenMap: make([]uint32, m.Cfg.VocabSize), CompressedVocab: uint32(m.Cfg.VocabSize), PadID: 0,
		Rows: []uint32{v41DecodeSubmitsEngramRow}, Multipliers: [][]uint64{{17, 13, 19, 11}},
		Primes: [][]uint32{make([]uint32, 24)}, MaxNgramSize: 4, HeadsPerNgram: 8,
	}
	for i := range layout.TokenMap {
		layout.TokenMap[i] = uint32(i)
	}
	for i := range layout.Primes[0] {
		layout.Primes[0][i] = 17
	}
	src := &v41EngramMemorySource{packed: v41EngramTestPackedRows(v41DecodeSubmitsEngramRow), rows: v41DecodeSubmitsEngramRow}
	if err := m.wireV41Engram(layout, []V41EngramRowSource{src}, int64(V41EngramPackedRowBytes)); err != nil {
		t.Fatal(err)
	}
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("layer-sweep fixture admission (layers=%d compressed=%t): %v", layers, compressed, err)
	}
	return m
}

// v41DecodeSubmitsFixtures names every fixture the witness measures: the two
// v41IncrementalEngramFixture variants (Engram on layers 0 and 2) that
// TestV41DeviceForwardGreedyParityVulkan decodes, and a 1/3-layer sweep with
// Engram on layer 0 only.
type v41DecodeSubmitsFixture struct {
	name  string
	build func(t *testing.T) *Model
}

func v41DecodeSubmitsFixtures() []v41DecodeSubmitsFixture {
	var out []v41DecodeSubmitsFixture
	for _, compressed := range []bool{false, true} {
		out = append(out, v41DecodeSubmitsFixture{fmt.Sprintf("engram02_layers3_compressed_%t", compressed), func(t *testing.T) *Model {
			m, _ := v41IncrementalEngramFixture(t, compressed)
			return m
		}})
		for _, layers := range []int{1, 3} {
			out = append(out, v41DecodeSubmitsFixture{fmt.Sprintf("engram0_layers%d_compressed_%t", layers, compressed), func(t *testing.T) *Model {
				return v41DecodeSubmitsLayeredFixture(t, layers, compressed)
			}})
		}
	}
	return out
}

// TestV41DecodeSubmitsCPURef is the cpu-ref half of the #13820/#13821 witness:
// the measured harness drives a host session whose greedy decode must match
// the cacheless forward oracle step by step on every fixture.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=5s lane=default
func TestV41DecodeSubmitsCPURef(t *testing.T) {
	for _, fx := range v41DecodeSubmitsFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			ref, m := fx.build(t), fx.build(t)
			defer func() {
				for _, mm := range []*Model{ref, m} {
					if err := mm.CloseWeights(); err != nil {
						t.Error(err)
					}
				}
			}()
			prompt := v41DecodeSubmitsPrompt(4, m.Cfg.VocabSize)
			s := &Session{M: m}
			defer s.Close()
			tr := v41DecodeSubmitsMeasure(t, fx.name+" backend=host ctx=4", s, nil, prompt, v41DecodeSubmitsSteps)
			v41DecodeSubmitsCheck(t, tr)
			history := append([]int(nil), prompt...)
			for _, w := range tr.windows {
				want := v41IncrementalEngramCold(t, ref, history)
				var scale, worst float64
				for j, v := range want {
					scale = math.Max(scale, math.Abs(float64(v)))
					worst = math.Max(worst, math.Abs(float64(w.logits[j]-v)))
				}
				if rel := worst / math.Max(1, scale); rel > v41DecodeSubmitsCPUTol {
					t.Errorf("%s step %d logits rel=%g > %g", w.phase, w.step, rel, v41DecodeSubmitsCPUTol)
				}
				if got, wantTok := w.token, argmaxF32(want); got != wantTok {
					t.Errorf("%s step %d greedy token=%d want %d", w.phase, w.step, got, wantTok)
				}
				if w.phase == "decode" && w.route != "incremental" {
					t.Logf("%s step %d took the %s route", w.phase, w.step, w.route)
				}
				history = append(history, w.token)
			}
		})
	}
}
