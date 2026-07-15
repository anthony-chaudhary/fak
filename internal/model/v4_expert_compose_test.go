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
	got, err := composeV4RoutedExperts(layer, routes, x, stager)
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
	if stats.Hits != 0 || stats.Evictions != 4 || stats.PeakResidentBytes != 2*24 {
		t.Fatalf("bounded ring stats=%+v, want hits=0 evictions=4 peak=48", stats)
	}
	if ring.used() > ring.budget() {
		t.Fatalf("ring used=%d exceeds budget=%d", ring.used(), ring.budget())
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
			if _, err := composeV4RoutedExperts(tc.layer, tc.routes, tc.x, stager); !errors.Is(err, ErrV4ExpertCompose) {
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
	if _, err := composeV4RoutedExperts(layer, []v4RoutedExpert{{0, 0.5}, {1, 0.5}}, x, badStager); !errors.Is(err, ErrV4ExpertCompose) {
		t.Fatalf("dimension error=%v, want ErrV4ExpertCompose", err)
	}
	if badReads.tensorReads != 0 || badRing.used() != 0 {
		t.Fatalf("invalid composition mutated source/ring: reads=%d used=%d", badReads.tensorReads, badRing.used())
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
