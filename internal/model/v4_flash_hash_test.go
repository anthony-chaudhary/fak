package model

import (
	"encoding/binary"
	"errors"
	"math"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestDeepSeekV4FlashHashRouteMatchesIndependentOracle(t *testing.T) {
	logits := make([]float32, 256)
	for i := range logits {
		logits[i] = float32(i%13)/3 - 2
	}
	ids := []int{255, 1, 3, 5, 7, 9}
	const scale = float32(1.5)

	got, err := v4HashRoute(logits, ids, scale)
	if err != nil {
		t.Fatalf("256-expert Flash route rejected: %v", err)
	}
	var denom float64
	for _, id := range ids {
		denom += math.Sqrt(math.Log1p(math.Exp(float64(logits[id]))))
	}
	for i, id := range ids {
		if got[i].expert != id {
			t.Fatalf("pick %d expert=%d want %d", i, got[i].expert, id)
		}
		want := float32(math.Sqrt(math.Log1p(math.Exp(float64(logits[id])))) / denom * float64(scale))
		if math.Abs(float64(got[i].weight-want)) > 2e-6 {
			t.Fatalf("expert %d weight=%g want=%g", id, got[i].weight, want)
		}
	}
}

func TestDeepSeekV4FlashHashRouteRejectsWidthAndExpertIDMismatches(t *testing.T) {
	validIDs := []int{0, 1, 2, 3, 4, 255}
	for _, width := range []int{255, 257} {
		t.Run("logits_width_"+itoa(width), func(t *testing.T) {
			if _, err := v4HashRoute(make([]float32, width), validIDs, 1.5); err == nil {
				t.Fatalf("admitted unsupported logits width %d", width)
			}
		})
	}
	badIDs := []int{0, 1, 2, 3, 4, 256}
	if _, err := v4HashRoute(make([]float32, 256), badIDs, 1.5); err == nil {
		t.Fatal("admitted expert id 256 for a 256-expert Flash route")
	}
}

// TestDeepSeekV4FlashExpertRuntimeHashPrimitive is a 32-dimensional synthetic
// component witness. It proves the admitted Flash profile selects and evaluates
// six FP4 experts; it makes no full-model geometry or token-generation claim.
func TestDeepSeekV4FlashExpertRuntimeHashPrimitive(t *testing.T) {
	restoreSpecs := useTinyV4RuntimeQuantSpecs()
	defer restoreSpecs()
	dir, weights := writeV4RuntimeFixture(t)
	_, cfg := readDeepSeekV4FlashConfig(t)
	be := compute.Default()
	runtime, err := newV4ExpertRuntime(dir, cfg, be, 16384, 2)
	if err != nil {
		t.Fatalf("construct Flash primitive runtime: %v", err)
	}
	defer runtime.Close()

	xHost := make([]float32, 32)
	for i := range xHost {
		xHost[i] = float32((i%5)-2) / 2
	}
	x := be.Upload(compute.NewF32(be, []int{32}, xHost), compute.F32)
	defer be.Free(x)
	logits := make([]float32, 256)
	for i := range logits {
		logits[i] = float32(i%9) / 2
	}

	got, err := runtime.forwardHash(0, 0, logits, x)
	if err != nil {
		t.Fatalf("Flash hash primitive forward: %v", err)
	}
	ids := []int{0, 1, 2, 3, 4, 5}
	picks, err := v4HashRoute(logits, ids, float32(cfg.RoutedScalingFactor))
	if err != nil {
		t.Fatal(err)
	}
	want := runtimeResidentOracle(0, picks, xHost, weights, float32(cfg.SwigluLimit))
	if !closeSlice(got, want, 2e-5) {
		t.Fatalf("Flash hash primitive=%v want=%v", got, want)
	}
	stats := runtime.Stats()
	if stats.HashReadCount != 1 || stats.SourceReads == 0 {
		t.Fatalf("primitive did not read one hash row and selected experts: %+v", stats)
	}
}

// TestDeepSeekV4FlashHashRejectsOutOfRangeRowBeforeExpertRead uses the same
// 32-dimensional synthetic fixture and proves an invalid hash ID is rejected
// before any expert tensor payload is read.
func TestDeepSeekV4FlashHashRejectsOutOfRangeRowBeforeExpertRead(t *testing.T) {
	restoreSpecs := useTinyV4RuntimeQuantSpecs()
	defer restoreSpecs()
	dir, _ := writeV4RuntimeFixture(t)
	data := make([]byte, v4HashVocab*v4HashTopK*8)
	for token := 0; token < v4HashVocab; token++ {
		for slot := 0; slot < v4HashTopK; slot++ {
			binary.LittleEndian.PutUint64(data[(token*v4HashTopK+slot)*8:], uint64(slot))
		}
	}
	binary.LittleEndian.PutUint64(data, 256)
	name := v4HashTensorName(0)
	writeV4RawShard(t, filepath.Join(dir, "hash0.safetensors"), map[string]tinySTTensor{
		name: {dtype: "I64", shape: []int{v4HashVocab, v4HashTopK}, data: data},
	})

	_, cfg := readDeepSeekV4FlashConfig(t)
	be := compute.Default()
	runtime, err := newV4ExpertRuntime(dir, cfg, be, 16384, 2)
	if err != nil {
		t.Fatalf("construct Flash primitive runtime: %v", err)
	}
	defer runtime.Close()
	x := be.Upload(compute.NewF32(be, []int{32}, make([]float32, 32)), compute.F32)
	defer be.Free(x)

	_, err = runtime.forwardHash(0, 0, make([]float32, 256), x)
	if err == nil || !errors.Is(err, ErrV4ExpertRuntime) {
		t.Fatalf("out-of-range Flash hash row error=%v, want ErrV4ExpertRuntime", err)
	}
	stats := runtime.Stats()
	if stats.HashReadCount != 1 || stats.SourceReads != 0 {
		t.Fatalf("out-of-range hash row reached expert payloads: %+v", stats)
	}
}
