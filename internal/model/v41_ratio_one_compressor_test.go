package model

import (
	"errors"
	"reflect"
	"testing"
)

// The helper's projection/normalization boundary is exercised directly.
// Ratio-one shared-source admission stays closed until shared-source ownership,
// prefill/Step publication and restore are implemented together. The F32 fixture
// carries BF16-representable projection inputs/weights; it is not an artifact,
// physical BF16 GEMM or FP4 cache qualification.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=20ms lane=default
func TestV41RatioOneCompressorRows(t *testing.T) {
	t.Parallel()
	const hidden, width = 3, 2
	const eps = float32(1e-5)
	wkvName := layerName(0, "attn.compressor.wkv.weight")
	normName := layerName(0, "attn.compressor.norm.weight")
	manifest, raw := synthBuildRaw([]synthTensor{
		{wkvName, []int{width, hidden}},
		{normName, []int{width}},
	}, synthMatmulFill)
	m := &Model{Cfg: Config{NumLayers: 1, HiddenSize: hidden, HeadDim: width, RMSNormEps: float64(eps)}, manifest: manifest, raw: raw}
	weights := []float32{1.0078125, 0.25, -0.5, -0.5, 1.5, 0.125}
	gain := []float32{0.75, -1.25}
	v41WriteTensorF32(t, m, wkvName, weights)
	v41WriteTensorF32(t, m, normName, gain)
	// No wgate tensor exists. Ordinary already-projected KV is deliberately
	// different from the pre-attention carriers the compressor must project.
	inputs := [][]float32{{1.0078125, 0.50390625, -0.1259765625}, {1.5, -2, 0.25}}
	ordinaryKV := [][]float32{{101, 102}, {103, 104}}
	wantProjected := make([][]float32, len(inputs))
	want := make([][]float32, len(inputs))
	for i, row := range inputs {
		wantProjected[i] = []float32{
			float32(float32(row[0]*weights[0]+row[1]*weights[1]) + row[2]*weights[2]),
			float32(float32(row[0]*weights[3]+row[1]*weights[4]) + row[2]*weights[5]),
		}
		want[i] = v41CompressorNormRefTail(wantProjected[i], gain, eps, "")
	}
	host, err := m.v41CompressedRows(0, 1, ordinaryKV, inputs)
	if err != nil || !reflect.DeepEqual(host, want) {
		t.Fatalf("ratio-one host rows = %v, %v; want %v", host, err, want)
	}

	var events []string
	projected, normalized := 0, 0
	project := func(layer int, leaf string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
		if layer != 0 || leaf != "attn.compressor.wkv.weight" || out != width || in != hidden || rows != 1 || projected >= len(inputs) || !reflect.DeepEqual(panel, inputs[projected]) {
			t.Fatalf("unexpected projection: layer=%d leaf=%s shape=(%d,%d,%d) input=%v", layer, leaf, out, in, rows, panel)
		}
		events = append(events, "wkv")
		row := append([]float32(nil), wantProjected[projected]...)
		projected++
		return row, v41ProjectionHandled, nil
	}
	normalize := func(layer int, row, learnedGain []float32, epsilon float32) ([]float32, error) {
		if layer != 0 || normalized >= len(inputs) || !reflect.DeepEqual(row, wantProjected[normalized]) || !reflect.DeepEqual(learnedGain, gain) || epsilon != eps {
			t.Fatalf("unexpected normalization: layer=%d row=%v gain=%v epsilon=%g", layer, row, learnedGain, epsilon)
		}
		events = append(events, "norm")
		normalized++
		return v41CompressorNormRefTail(row, learnedGain, epsilon, ""), nil
	}
	selected, err := m.v41CompressedRowsWithOperations(0, 1, ordinaryKV, inputs, project, normalize)
	if err != nil || !reflect.DeepEqual(selected, want) || !reflect.DeepEqual(events, []string{"wkv", "norm", "wkv", "norm"}) {
		t.Fatalf("selected rows = %v, %v, events=%v; want %v with one projection/norm per token", selected, err, events, want)
	}
	// Ratio zero must touch neither the compressor projection nor its norm.
	bypass, err := m.v41CompressedRowsWithOperations(0, 0, ordinaryKV, inputs, project, normalize)
	if err != nil || !reflect.DeepEqual(bypass, ordinaryKV) || projected != len(inputs) || normalized != len(inputs) {
		t.Fatalf("ratio-zero bypass changed: %v, %v, projection=%d norm=%d", bypass, err, projected, normalized)
	}
	failure := errors.New("selected ratio-one operation failure")
	faultProject := func(int, string, []float32, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
		return nil, v41ProjectionError, failure
	}
	if rows, err := m.v41CompressedRowsWithOperations(0, 1, ordinaryKV, inputs, faultProject, normalize); rows != nil || !errors.Is(err, failure) || normalized != len(inputs) {
		t.Fatalf("projection failure did not stop before normalization: %v, %v", rows, err)
	}
	normFailures := 0
	faultNorm := func(int, []float32, []float32, float32) ([]float32, error) {
		normFailures++
		return nil, failure
	}
	if rows, err := m.v41CompressedRowsWithOperations(0, 1, ordinaryKV, inputs, nil, faultNorm); rows != nil || !errors.Is(err, failure) || normFailures != 1 {
		t.Fatalf("normalization failure returned partial rows or retried: %v, %v, calls=%d", rows, err, normFailures)
	}
	// The completed helper does not lift ratio-one shared-source admission.
	m.Cfg.DeepSeekV41 = &DeepSeekV41Config{CompressRatios: []int{1}, KVSourceLayerIDs: []int{0}}
	if err := m.v41KVSourceForwardAdmitted(); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("ratio-one shared source must remain fenced, got %v", err)
	}
}
