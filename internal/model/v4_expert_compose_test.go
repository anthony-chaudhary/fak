package model

import (
	"encoding/binary"
	"errors"
	"math"
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestComposeV4RoutedExpertsMatchesIndependentResidentOracle(t *testing.T) {
	const layer = 7
	weights := map[int]map[string][]float32{
		2: {
			"w1": {1, -2, 0.5, 1, -1, 0.25},
			"w2": {0.75, -1, 2, -0.5, 0.25, 1.5},
			"w3": {-0.5, 1.5, 2, -1, 0.75, 0.5},
		},
		5: {
			"w1": {-1.25, 0.5, 1.5, -0.75, 0.25, 2},
			"w2": {1, 0.5, -1.5, -2, 0.75, 0.25},
			"w3": {2, -0.25, -1, 1.25, 0.5, -2},
		},
	}
	tensors := make(map[string]tinySTTensor)
	for expert, matrices := range weights {
		for projection, values := range matrices {
			shape := []int{3, 2}
			if projection == "w2" {
				shape = []int{2, 3}
			}
			name := v4ComposeTensorName(layer, expert, projection)
			tensors[name] = tinySTTensor{dtype: "F32", shape: shape, data: f32TestBytes(values)}
		}
	}
	source, reads := newV4ComposeFixtureSource(t, tensors)
	plan, err := source.planV4ExpertBatch(layer, []int{2, 5}, 6*24)
	if err != nil {
		t.Fatal(err)
	}
	be := compute.Default()
	ring := newPagedRing(be, 2*24)
	decode := func(tensor v4ExpertTensor) (compute.Tensor, error) {
		values := make([]float32, len(tensor.Bytes)/4)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(tensor.Bytes[4*i:]))
		}
		return compute.NewF32(be, tensor.Shape, values), nil
	}
	stager, err := newV4ExpertStager(source, ring, plan, compute.F32, decode)
	if err != nil {
		t.Fatal(err)
	}
	xHost := []float32{0.75, -1.25}
	x := be.Upload(compute.NewF32(be, []int{2}, xHost), compute.F32)
	defer be.Free(x)
	routes := []v4RoutedExpert{{Expert: 2, Weight: 0.7}, {Expert: 5, Weight: 0.3}}
	got, err := composeV4RoutedExperts(layer, routes, x, 0, stager)
	if err != nil {
		t.Fatal(err)
	}
	want := residentV4CompositionOracle(xHost, routes, weights)
	if len(got) != len(want) {
		t.Fatalf("output length=%d want=%d", len(got), len(want))
	}
	for i := range got {
		if delta := math.Abs(float64(got[i] - want[i])); delta > 1e-6 {
			t.Fatalf("output[%d]=%.9g want %.9g delta %.3g", i, got[i], want[i], delta)
		}
	}
	stats := stager.Stats()
	if reads.tensorReads != 6 || stats.SourceReads != 6 || stats.SourceBytes != 6*24 || stats.PageIn != 6 {
		t.Fatalf("source/stage evidence reads=%d stats=%+v", reads.tensorReads, stats)
	}
	// Each projection is now staged one GEMM ahead of its demand (#13044), so the demand
	// matMul finds it resident: all six page-ins are HITS on the demand path, not misses.
	// The page-in count, read count and eviction count are unchanged from the on-demand
	// path — the same six weights are read, uploaded and evicted — only the demand's view
	// of them moves from miss to hit, which is exactly the overlap the fuse buys.
	if stats.Hits != 6 || stats.Evictions != 4 || stats.PeakResidentBytes != 2*24 {
		t.Fatalf("bounded ring stats=%+v, want hits=6 evictions=4 peak=48", stats)
	}
	if ring.used() > ring.budget() {
		t.Fatalf("ring used=%d exceeds budget=%d", ring.used(), ring.budget())
	}
}

func TestComposeV4RoutedExpertsAppliesOfficialAsymmetricSwiGLULimit(t *testing.T) {
	const (
		layer = 3
		limit = float32(10)
	)
	// Identity w1/w3 expose every clamp branch directly. w2 sums pairs so all
	// four dimensions affect the observed output: gate below/above the limit and
	// up below/above each symmetric bound.
	tensors := map[string]tinySTTensor{
		v4ComposeTensorName(layer, 0, "w1"): {dtype: "F32", shape: []int{4, 4}, data: f32TestBytes([]float32{
			1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
		})},
		v4ComposeTensorName(layer, 0, "w2"): {dtype: "F32", shape: []int{4, 4}, data: f32TestBytes([]float32{
			1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1,
		})},
		v4ComposeTensorName(layer, 0, "w3"): {dtype: "F32", shape: []int{4, 4}, data: f32TestBytes([]float32{
			0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0,
		})},
	}
	source, reads := newV4ComposeFixtureSource(t, tensors)
	plan, err := source.planV4ExpertBatch(layer, []int{0}, 3*64)
	if err != nil {
		t.Fatal(err)
	}
	be := compute.Default()
	decode := func(tensor v4ExpertTensor) (compute.Tensor, error) {
		values := make([]float32, len(tensor.Bytes)/4)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(tensor.Bytes[4*i:]))
		}
		return compute.NewF32(be, tensor.Shape, values), nil
	}
	stager, err := newV4ExpertStager(source, newPagedRing(be, 3*64), plan, compute.F32, decode)
	if err != nil {
		t.Fatal(err)
	}
	input := []float32{-20, 20, 30, -30}
	x := be.Upload(compute.NewF32(be, []int{4}, input), compute.F32)
	defer be.Free(x)
	got, err := composeV4RoutedExperts(layer, []v4RoutedExpert{{0, 1}}, x, limit, stager)
	if err != nil {
		t.Fatal(err)
	}

	gate := append([]float32(nil), input...)
	up := []float32{input[2], input[3], input[0], input[1]}
	middle := make([]float32, len(gate))
	for i := range middle {
		g := gate[i]
		if g > limit {
			g = limit
		}
		u := up[i]
		if u < -limit {
			u = -limit
		}
		if u > limit {
			u = limit
		}
		middle[i] = float32(float64(g)/(1+math.Exp(-float64(g)))) * u
	}
	w2 := tensors[v4ComposeTensorName(layer, 0, "w2")].data
	_ = w2 // matrix values are intentionally restated for an independent oracle.
	want := scalarMatVec([]float32{1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1}, 4, 4, middle)
	for i := range want {
		if delta := math.Abs(float64(got[i] - want[i])); delta > 1e-5 {
			t.Fatalf("limited output[%d]=%.9g want %.9g delta %.3g", i, got[i], want[i], delta)
		}
	}
	// Symmetrically clamping the negative gate would produce a materially
	// different first component; this assertion specifically witnesses the
	// official upper-only gate clamp.
	symmetricGate := float32(-float64(limit) / (1 + math.Exp(float64(limit))))
	symmetricFirst := symmetricGate*(-limit) + v4SiLU(limit)*(-limit)
	if math.Abs(float64(got[0]-symmetricFirst)) < 1e-3 {
		t.Fatalf("output accidentally matches symmetric gate clamp: got=%g symmetric=%g", got[0], symmetricFirst)
	}

	readsBefore, usedBefore := reads.tensorReads, stager.ring.used()
	for _, bad := range []float32{-1, float32(math.NaN()), float32(math.Inf(1))} {
		if _, err := composeV4RoutedExperts(layer, []v4RoutedExpert{{0, 1}}, x, bad, stager); !errors.Is(err, ErrV4ExpertCompose) {
			t.Fatalf("limit %v error=%v, want ErrV4ExpertCompose", bad, err)
		}
	}
	if reads.tensorReads != readsBefore || stager.ring.used() != usedBefore {
		t.Fatalf("invalid limits mutated source/ring: reads %d->%d used %d->%d", readsBefore, reads.tensorReads, usedBefore, stager.ring.used())
	}
}

func TestComposeV4RoutedExpertsRejectsInvalidInputs(t *testing.T) {
	const layer = 1
	tensors := map[string]tinySTTensor{
		v4ComposeTensorName(layer, 0, "w1"): {dtype: "F32", shape: []int{1, 1}, data: f32TestBytes([]float32{1})},
		v4ComposeTensorName(layer, 0, "w2"): {dtype: "F32", shape: []int{1, 1}, data: f32TestBytes([]float32{1})},
		v4ComposeTensorName(layer, 0, "w3"): {dtype: "F32", shape: []int{1, 1}, data: f32TestBytes([]float32{1})},
	}
	source, _ := newV4ComposeFixtureSource(t, tensors)
	plan, err := source.planV4ExpertBatch(layer, []int{0}, 12)
	if err != nil {
		t.Fatal(err)
	}
	be := compute.Default()
	stager, err := newV4ExpertStager(source, newPagedRing(be, 12), plan, compute.F32, func(tensor v4ExpertTensor) (compute.Tensor, error) {
		return compute.NewF32(be, tensor.Shape, []float32{1}), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	x := be.Upload(compute.NewF32(be, []int{1}, []float32{1}), compute.F32)
	defer be.Free(x)
	cases := []struct {
		name   string
		layer  int
		routes []v4RoutedExpert
		x      compute.Tensor
	}{
		{"empty", layer, nil, x},
		{"wrong-layer", layer + 1, []v4RoutedExpert{{0, 1}}, x},
		{"unselected", layer, []v4RoutedExpert{{3, 1}}, x},
		{"duplicate", layer, []v4RoutedExpert{{0, 0.5}, {0, 0.5}}, x},
		{"nan-weight", layer, []v4RoutedExpert{{0, float32(math.NaN())}}, x},
		{"matrix-input", layer, []v4RoutedExpert{{0, 1}}, be.Upload(compute.NewF32(be, []int{1, 1}, []float32{1}), compute.F32)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := composeV4RoutedExperts(tc.layer, tc.routes, tc.x, 0, stager); !errors.Is(err, ErrV4ExpertCompose) {
				t.Fatalf("error=%v, want ErrV4ExpertCompose", err)
			}
		})
	}
	be.Free(cases[len(cases)-1].x)

	// A malformed complete selection is refused before any source IO or ring
	// mutation, even when the malformed expert appears after a valid one.
	badTensors := map[string]tinySTTensor{
		v4ComposeTensorName(layer, 0, "w1"): {dtype: "F32", shape: []int{1, 1}, data: f32TestBytes([]float32{1})},
		v4ComposeTensorName(layer, 0, "w2"): {dtype: "F32", shape: []int{1, 1}, data: f32TestBytes([]float32{1})},
		v4ComposeTensorName(layer, 0, "w3"): {dtype: "F32", shape: []int{1, 1}, data: f32TestBytes([]float32{1})},
		v4ComposeTensorName(layer, 1, "w1"): {dtype: "F32", shape: []int{2, 1}, data: f32TestBytes([]float32{1, 1})},
		v4ComposeTensorName(layer, 1, "w2"): {dtype: "F32", shape: []int{1, 1}, data: f32TestBytes([]float32{1})},
		v4ComposeTensorName(layer, 1, "w3"): {dtype: "F32", shape: []int{2, 1}, data: f32TestBytes([]float32{1, 1})},
	}
	badSource, badReads := newV4ComposeFixtureSource(t, badTensors)
	badPlan, err := badSource.planV4ExpertBatch(layer, []int{0, 1}, 36)
	if err != nil {
		t.Fatal(err)
	}
	badRing := newPagedRing(be, 36)
	badStager, err := newV4ExpertStager(badSource, badRing, badPlan, compute.F32, func(tensor v4ExpertTensor) (compute.Tensor, error) {
		return compute.NewF32(be, tensor.Shape, make([]float32, len(tensor.Bytes)/4)), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := composeV4RoutedExperts(layer, []v4RoutedExpert{{0, 0.5}, {1, 0.5}}, x, 0, badStager); !errors.Is(err, ErrV4ExpertCompose) {
		t.Fatalf("dimension error=%v, want ErrV4ExpertCompose", err)
	}
	if badReads.tensorReads != 0 || badRing.used() != 0 {
		t.Fatalf("invalid composition mutated source/ring: reads=%d used=%d", badReads.tensorReads, badRing.used())
	}
}

// v4OverlapBackend models a device whose host->device transfer lands while OTHER compute
// runs: any pending fence is satisfied the moment a MatMul completes. That is the physical
// behaviour staging a projection a GEMM ahead exists to exploit, and it is what makes the
// fuse observable rather than merely asserted — a transfer issued before a GEMM is Done by
// the time its own demand arrives (overlap), while one issued with no GEMM in between must
// be waited on (blocking). MatMul also flags any GEMM against a not-yet-landed weight, so a
// skipped fence fails the test rather than quietly multiplying stale bytes.
type v4OverlapBackend struct {
	compute.Backend
	inflight      map[compute.Buffer]bool // weights whose transfer has not yet landed
	unfencedGEMMs int
	fences        int
}

func newV4OverlapBackend() *v4OverlapBackend {
	return &v4OverlapBackend{Backend: compute.Default(), inflight: map[compute.Buffer]bool{}}
}

func (b *v4OverlapBackend) Name() string { return "vulkan-test-overlap" }

func (b *v4OverlapBackend) UploadAsync(t compute.Tensor, as compute.Dtype) (compute.Tensor, compute.Fence) {
	out := b.Backend.Upload(t, as)
	if buf := out.Buf(); buf != nil {
		b.inflight[buf] = true
	}
	b.fences++
	return out, &v4OverlapFence{b: b, buf: out.Buf()}
}

type v4OverlapFence struct {
	b      *v4OverlapBackend
	buf    compute.Buffer
	landed bool
}

func (f *v4OverlapFence) Done() bool { return f.landed || !f.b.inflight[f.buf] }
func (f *v4OverlapFence) Wait() {
	if !f.b.inflight[f.buf] {
		f.landed = true
		return
	}
	f.landed = true
	delete(f.b.inflight, f.buf)
}

func (b *v4OverlapBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	if buf := w.Buf(); buf != nil && b.inflight[buf] {
		b.unfencedGEMMs++
	}
	out := b.Backend.MatMul(w, x)
	// Completing a GEMM lets every in-flight transfer land, modelling overlap.
	for buf := range b.inflight {
		delete(b.inflight, buf)
	}
	return out
}

// TestComposeV4RoutedExpertsStagesAheadOnAnAsyncBackend is the #13044 witness: on a
// backend whose transfers land while other compute runs, staging a projection a GEMM ahead
// of its demand lets that transfer land underneath the work between them, so the fuse is
// observable as overlap rather than merely asserted. The backend flags any GEMM run against
// a weight whose own transfer had not landed, so a skipped fence fails the test instead of
// silently producing wrong numbers; the outputs are checked against the resident oracle.
func TestComposeV4RoutedExpertsStagesAheadOnAnAsyncBackend(t *testing.T) {
	const layer = 9
	weights := map[int]map[string][]float32{
		1: {
			"w1": {1, -2, 0.5, 1, -1, 0.25},
			"w2": {0.75, -1, 2, -0.5, 0.25, 1.5},
			"w3": {-0.5, 1.5, 2, -1, 0.75, 0.5},
		},
	}
	tensors := make(map[string]tinySTTensor)
	for expert, matrices := range weights {
		for projection, values := range matrices {
			shape := []int{3, 2}
			if projection == "w2" {
				shape = []int{2, 3}
			}
			tensors[v4ComposeTensorName(layer, expert, projection)] = tinySTTensor{dtype: "F32", shape: shape, data: f32TestBytes(values)}
		}
	}
	source, _ := newV4ComposeFixtureSource(t, tensors)
	plan, err := source.planV4ExpertBatch(layer, []int{1}, 3*24)
	if err != nil {
		t.Fatal(err)
	}
	be := newV4OverlapBackend()
	ring := newPagedRing(be, 3*24)
	decode := func(tensor v4ExpertTensor) (compute.Tensor, error) {
		values := make([]float32, len(tensor.Bytes)/4)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(tensor.Bytes[4*i:]))
		}
		return compute.NewF32(be, tensor.Shape, values), nil
	}
	stager, err := newV4ExpertStager(source, ring, plan, compute.F32, decode)
	if err != nil {
		t.Fatal(err)
	}
	xHost := []float32{0.75, -1.25}
	x := be.Upload(compute.NewF32(be, []int{2}, xHost), compute.F32)
	defer be.Free(x)
	routes := []v4RoutedExpert{{Expert: 1, Weight: 1}}
	got, err := composeV4RoutedExperts(layer, routes, x, 0, stager)
	if err != nil {
		t.Fatal(err)
	}
	if be.unfencedGEMMs != 0 {
		t.Fatalf("%d GEMM(s) ran against a weight whose transfer had not landed", be.unfencedGEMMs)
	}
	if be.fences == 0 {
		t.Fatal("no transfer went through UploadAsync; the async path was never taken")
	}
	if ring.asyncOverlapped == 0 {
		t.Fatalf("asyncOverlapped=0 with %d async transfers: staging ahead must land at least one transfer under the work between them (waited=%d)", be.fences, ring.asyncWaited)
	}
	want := residentV4CompositionOracle(xHost, routes, weights)
	for i := range want {
		if delta := math.Abs(float64(got[i] - want[i])); delta > 1e-6 {
			t.Fatalf("output[%d]=%.9g want %.9g delta %.3g", i, got[i], want[i], delta)
		}
	}
}

func v4ComposeTensorName(layer, expert int, projection string) string {
	return "model.layers." + strconv.Itoa(layer) + ".ffn.experts." + strconv.Itoa(expert) + "." + projection + ".weight"
}

func newV4ComposeFixtureSource(t *testing.T, tensors map[string]tinySTTensor) (*v4ExpertSource, *v4ExpertSourceReaderAt) {
	t.Helper()
	buf := tinySafetensorsBytes(t, tensors)
	rr := &v4ExpertSourceReaderAt{data: buf}
	sf, err := newSafetensorsFile(rr, int64(len(buf)), nil)
	if err != nil {
		t.Fatal(err)
	}
	rr.dataBase = sf.dataBase
	source, err := newV4ExpertSource(sf)
	if err != nil {
		t.Fatal(err)
	}
	return source, rr
}

// residentV4CompositionOracle deliberately does not call the production stager,
// backend MatMul, or SiLU helper. It is an independent row-major scalar oracle.
func residentV4CompositionOracle(x []float32, routes []v4RoutedExpert, weights map[int]map[string][]float32) []float32 {
	out := make([]float32, len(x))
	for _, route := range routes {
		w := weights[route.Expert]
		gate := scalarMatVec(w["w1"], 3, 2, x)
		up := scalarMatVec(w["w3"], 3, 2, x)
		middle := make([]float32, 3)
		for i := range middle {
			sigmoid := float32(1 / (1 + math.Exp(-float64(gate[i]))))
			middle[i] = route.Weight * gate[i] * sigmoid * up[i]
		}
		projected := scalarMatVec(w["w2"], 2, 3, middle)
		for i := range out {
			out[i] += projected[i]
		}
	}
	return out
}

func scalarMatVec(matrix []float32, rows, cols int, x []float32) []float32 {
	out := make([]float32, rows)
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			out[row] += matrix[row*cols+col] * x[col]
		}
	}
	return out
}
