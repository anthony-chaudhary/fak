package model

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestV4ExpertStagerUsesRingHitsEvictionAndReload(t *testing.T) {
	const (
		experts     = 6
		out         = 2
		in          = 2
		weightBytes = int64(out * in * 4)
	)
	tensors := make(map[string]tinySTTensor, experts)
	weights := make(map[string][]float32, experts)
	for expert := 0; expert < experts; expert++ {
		name := fmt.Sprintf("model.layers.0.ffn.experts.%d.w1.weight", expert)
		w := []float32{float32(expert + 1), 0.25, -0.5, float32(expert + 2)}
		weights[name] = w
		tensors[name] = tinySTTensor{dtype: "F32", shape: []int{out, in}, data: f32TestBytes(w)}
	}
	buf := tinySafetensorsBytes(t, tensors)
	rr := &v4ExpertSourceReaderAt{data: buf}
	sf, err := newSafetensorsFile(rr, int64(len(buf)), nil)
	if err != nil {
		t.Fatalf("newSafetensorsFile: %v", err)
	}
	rr.dataBase = sf.dataBase
	source, err := newV4ExpertSource(sf)
	if err != nil {
		t.Fatalf("newV4ExpertSource: %v", err)
	}
	plan, err := source.planV4ExpertBatch(0, []int{0, 1, 2, 3, 4, 5}, 6*weightBytes)
	if err != nil {
		t.Fatalf("planV4ExpertBatch: %v", err)
	}

	be := compute.Default()
	ring := newPagedRing(be, 2*weightBytes)
	decodeCalls := 0
	decode := func(tensor v4ExpertTensor) (compute.Tensor, error) {
		decodeCalls++
		if tensor.Dtype != "F32" || len(tensor.Bytes) != int(weightBytes) {
			return compute.Tensor{}, fmt.Errorf("unexpected fixture tensor %s", tensor.Name)
		}
		w := make([]float32, len(tensor.Bytes)/4)
		for i := range w {
			w[i] = math.Float32frombits(binary.LittleEndian.Uint32(tensor.Bytes[i*4:]))
		}
		return compute.NewF32(be, tensor.Shape, w), nil
	}
	stager, err := newV4ExpertStager(source, ring, plan, compute.F32, decode)
	if err != nil {
		t.Fatalf("newV4ExpertStager: %v", err)
	}
	x := be.Upload(compute.NewF32(be, []int{in}, []float32{0.75, -1.25}), compute.F32)
	run := func(name string) {
		t.Helper()
		got, err := stager.matMul(name, x)
		if err != nil {
			t.Fatalf("stage %s: %v", name, err)
		}
		want := ringResident(t, be, out, in, weights[name], x)
		if !ringEqual(got, want) {
			t.Fatalf("%s staged output %v != resident %v", name, got, want)
		}
	}

	names := []string{
		"model.layers.0.ffn.experts.0.w1.weight",
		"model.layers.0.ffn.experts.1.w1.weight",
		"model.layers.0.ffn.experts.2.w1.weight",
	}
	run(names[0])
	run(names[1])
	readsBeforeHit, decodesBeforeHit := rr.tensorReads, decodeCalls
	run(names[1])
	if rr.tensorReads != readsBeforeHit || decodeCalls != decodesBeforeHit {
		t.Fatalf("cache hit performed source/decode work: reads %d->%d decodes %d->%d", readsBeforeHit, rr.tensorReads, decodesBeforeHit, decodeCalls)
	}
	run(names[2]) // evicts names[0]
	run(names[0]) // deterministic reload

	stats := stager.Stats()
	if stats.SourceReads != 4 || stats.SourceBytes != 4*weightBytes || stats.PageIn != 4 || stats.Hits != 1 || stats.Evictions != 2 {
		t.Fatalf("stats=%+v, want reads/bytes/pageIn/hits/evict=4/%d/4/1/2", stats, 4*weightBytes)
	}
	if stats.PeakResidentBytes != 2*weightBytes || ring.used() > ring.budget() {
		t.Fatalf("peak/used/budget=%d/%d/%d, want peak %d and bounded", stats.PeakResidentBytes, ring.used(), ring.budget(), 2*weightBytes)
	}
}

func TestV4ExpertStagerRejectsUnselectedAndDecodeErrorsWithoutAdmission(t *testing.T) {
	tensors := map[string]tinySTTensor{
		"model.layers.0.ffn.experts.0.w1.weight": {dtype: "F32", shape: []int{1, 1}, data: f32TestBytes([]float32{2})},
		"model.layers.0.ffn.experts.1.w1.weight": {dtype: "F32", shape: []int{1, 1}, data: f32TestBytes([]float32{3})},
	}
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
	plan, err := source.planV4ExpertBatch(0, []int{0}, 4)
	if err != nil {
		t.Fatal(err)
	}
	be := compute.Default()
	ring := newPagedRing(be, 4)
	decodeErr := errors.New("fixture decode refused")
	stager, err := newV4ExpertStager(source, ring, plan, compute.F32, func(v4ExpertTensor) (compute.Tensor, error) {
		return compute.Tensor{}, decodeErr
	})
	if err != nil {
		t.Fatal(err)
	}
	x := be.Upload(compute.NewF32(be, []int{1}, []float32{1}), compute.F32)
	if _, err := stager.matMul("model.layers.0.ffn.experts.1.w1.weight", x); !errors.Is(err, ErrV4ExpertNotSelected) {
		t.Fatalf("unselected error=%v, want ErrV4ExpertNotSelected", err)
	}
	if rr.tensorReads != 0 {
		t.Fatalf("unselected request performed %d source reads", rr.tensorReads)
	}
	selected := "model.layers.0.ffn.experts.0.w1.weight"
	if _, err := stager.matMul(selected, x); !errors.Is(err, decodeErr) {
		t.Fatalf("decode error=%v, want fixture error", err)
	}
	if ring.isResident(selected) || ring.used() != 0 || ring.pageIn != 0 {
		t.Fatalf("failed decode admitted resident: resident=%v used=%d pageIn=%d", ring.isResident(selected), ring.used(), ring.pageIn)
	}
}
