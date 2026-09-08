//go:build cuda && cgo

package compute

import (
	"os"
	"slices"
	"testing"
)

const cudaPLEHashRequiredEnv = "FAK_CUDA_PLE_HASH_REQUIRED"

func qwen4ExpPLEHashCUDAOrSkip(t *testing.T) *cudaBackend {
	t.Helper()
	be, ok := Lookup("cuda")
	if !ok || be == nil || be.Name() != "cuda" {
		if os.Getenv(cudaPLEHashRequiredEnv) == "1" {
			t.Fatalf("%s=1: real CUDA PLE hash fixture required", cudaPLEHashRequiredEnv)
		}
		t.Skip("exact CUDA backend is unavailable")
	}
	cuda, ok := be.(*cudaBackend)
	if !ok {
		t.Fatalf("registered CUDA backend has type %T", be)
	}
	return cuda
}

func TestQwen4ExpPLEFusedHashCUDA(t *testing.T) {
	cuda := qwen4ExpPLEHashCUDAOrSkip(t)
	_, liveBefore := cuda.cudaAllocationCountsForTest()
	spec := Qwen4ExpPLEHashSpec{
		NGramSize: 3, HeadsPerNGram: 8, BoundaryToken: 151643,
		Multipliers: []int64{0x657f9d1268b45d3, -0x61c8864680b583eb, 0x6a09e667f3bcc909},
		HeadVocabSizes: []int64{
			2500009, 2500027, 2500049, 2500051, 2500073, 2500103, 2500111, 2500121,
			2500133, 2500151, 2500157, 2500193, 2500199, 2500217, 2500231, 2500289,
		},
		HeadOffsets: []int64{
			0, 2500009, 5000036, 7500085, 10000136, 12500209, 15000312, 17500423,
			20000544, 22500677, 25000828, 27500985, 30001178, 32501377, 35001594, 37501825,
		},
	}
	batch := Qwen4ExpPLEHashBatch{
		InputIDs:        []int64{7, 151643, 11, -19, 23, 29, 31, 37, 41, 43},
		SequenceLengths: []int{1, 4, 2, 3},
		NGramContext: []int64{
			151643, 151643,
			2, 3,
			151643, 5,
			47, 53,
		},
	}
	want, err := Qwen4ExpPLERowIDsReference(batch, spec)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := cuda.PrepareQwen4ExpPLEHash(batch, spec)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := plan.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})
	receipt, err := plan.Dispatch()
	if err != nil {
		t.Fatal(err)
	}
	if receipt.KernelLaunches != 1 || receipt.HostToDeviceBytes != 0 || receipt.DeviceToHostBytes != 0 {
		t.Fatalf("dispatch receipt = %+v, want one launch and zero transfers", receipt)
	}
	got, err := plan.ReadRows()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, want) {
		t.Fatalf("CUDA row ids differ\n got %v\nwant %v", got, want)
	}
	if err := plan.Close(); err != nil {
		t.Fatal(err)
	}
	_, liveAfter := cuda.cudaAllocationCountsForTest()
	if liveAfter != liveBefore {
		t.Fatalf("live CUDA allocations after close = %d, before = %d", liveAfter, liveBefore)
	}
}

func TestQwen4ExpPLEFusedHashCUDAEmptyBatchDoesNotLaunch(t *testing.T) {
	cuda := qwen4ExpPLEHashCUDAOrSkip(t)
	spec := Qwen4ExpPLEHashSpec{
		NGramSize: 2, HeadsPerNGram: 1, BoundaryToken: 0,
		Multipliers: []int64{3, 5}, HeadVocabSizes: []int64{7}, HeadOffsets: []int64{0},
	}
	plan, err := cuda.PrepareQwen4ExpPLEHash(Qwen4ExpPLEHashBatch{}, spec)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := plan.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	}()
	receipt, err := plan.Dispatch()
	if err != nil {
		t.Fatal(err)
	}
	if receipt != (Qwen4ExpPLEHashDispatchReceipt{}) {
		t.Fatalf("empty dispatch receipt = %+v, want zero", receipt)
	}
	rows, err := plan.ReadRows()
	if err != nil {
		t.Fatal(err)
	}
	if rows == nil || len(rows) != 0 {
		t.Fatalf("empty rows = %#v, want non-nil empty", rows)
	}
}
