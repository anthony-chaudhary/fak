package ggufload

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// gguf_glm_retain_test.go — the #3078/#3197 GLM-5.2 self-speculation substrate scaffold: the
// model.RetainMTP flag flips the MTP ("nextn") head from DROPPED (default, byte-identical to the
// historical load) to RETAINED (its bytes accounted by the fit estimators). This is the inverse
// of internal/model/glm_test.go's TestGLMDropsMtpAndVisualTensorsAtLoad, on the GGUF side.

// TestGLMMoeDsaSkipGGUFTensorHonorsRetainMTP pins the flag-gated DROP predicate: the MTP head is
// dropped by default and retained when RetainMTP is set, while the vision tower is ALWAYS dropped.
func TestGLMMoeDsaSkipGGUFTensorHonorsRetainMTP(t *testing.T) {
	nextn := "blk.78.nextn.eh_proj.weight"
	vision := "v.blk.0.attn_q.weight"
	kept := "blk.0.attn_k_b.weight"

	// Default (flag OFF): MTP + vision dropped, byte-identical to the historical behavior.
	if !glmMoeDsaSkipGGUFTensor(nextn) {
		t.Fatalf("RetainMTP=off: skip(%q)=false, want true (MTP dropped by default)", nextn)
	}
	if !glmMoeDsaSkipGGUFTensor(vision) {
		t.Fatalf("RetainMTP=off: skip(%q)=false, want true (vision always dropped)", vision)
	}
	if glmMoeDsaSkipGGUFTensor(kept) {
		t.Fatalf("RetainMTP=off: skip(%q)=true, want false (forward tensor kept)", kept)
	}

	// Flag ON: MTP retained; vision still dropped; forward tensor still kept.
	defer func() { model.RetainMTP = false }()
	model.RetainMTP = true
	if glmMoeDsaSkipGGUFTensor(nextn) {
		t.Fatalf("RetainMTP=on: skip(%q)=true, want false (MTP retained)", nextn)
	}
	if !glmMoeDsaSkipGGUFTensor(vision) {
		t.Fatalf("RetainMTP=on: skip(%q)=false, want true (vision still dropped)", vision)
	}
	if glmMoeDsaSkipGGUFTensor(kept) {
		t.Fatalf("RetainMTP=on: skip(%q)=true, want false (forward tensor kept)", kept)
	}
}

// TestGLMMoeDsaSkipGGUFTensorHonorsRetainVision is the vision twin of the RetainMTP test: the
// CLIP image tower (v.*/mm.*) is dropped from byte-accounting by default and RETAINED (counted)
// when model.RetainVision is set (#4029). The two retention flags are independent — the vision
// flag leaves the MTP head at its own default and never affects forward tensors.
func TestGLMMoeDsaSkipGGUFTensorHonorsRetainVision(t *testing.T) {
	vision := "v.blk.0.attn_q.weight"
	proj := "mm.0.weight"
	nextn := "blk.78.nextn.eh_proj.weight"
	kept := "blk.0.attn_k_b.weight"

	// Default (flag OFF): vision dropped, byte-identical to the historical behavior.
	if !glmMoeDsaSkipGGUFTensor(vision) {
		t.Fatalf("RetainVision=off: skip(%q)=false, want true (vision dropped by default)", vision)
	}

	// Flag ON: both vision namespaces retained; MTP head still at its own default; forward kept.
	defer func() { model.RetainVision = false }()
	model.RetainVision = true
	if glmMoeDsaSkipGGUFTensor(vision) {
		t.Fatalf("RetainVision=on: skip(%q)=true, want false (vision tower retained)", vision)
	}
	if glmMoeDsaSkipGGUFTensor(proj) {
		t.Fatalf("RetainVision=on: skip(%q)=true, want false (projector retained)", proj)
	}
	if !glmMoeDsaSkipGGUFTensor(nextn) {
		t.Fatalf("RetainVision=on: skip(%q)=false, want true (MTP head still dropped by its own flag)", nextn)
	}
	if glmMoeDsaSkipGGUFTensor(kept) {
		t.Fatalf("RetainVision=on: skip(%q)=true, want false (forward tensor kept)", kept)
	}
}

// TestGLMMoeDsaMTPOrVisionTensorIgnoresRetainMTP pins the loader-safety contract: the ungated
// union that the materializing loaders + the CPU-offload classifier key on ALWAYS reports the
// MTP head and vision tower, regardless of RetainMTP — because the GGUF MTP head has no canonical
// slot to materialize into yet, so retaining its bytes must never route it into CanonicalTensorName
// (which would reject a real GLM-5.2 checkpoint).
func TestGLMMoeDsaMTPOrVisionTensorIgnoresRetainMTP(t *testing.T) {
	defer func() { model.RetainMTP = false }()
	for _, retain := range []bool{false, true} {
		model.RetainMTP = retain
		if !glmMoeDsaMTPOrVisionTensor("blk.78.nextn.eh_proj.weight") {
			t.Fatalf("RetainMTP=%v: union(nextn)=false, want true (loader always drops from materialization)", retain)
		}
		if !glmMoeDsaMTPOrVisionTensor("v.blk.0.attn_q.weight") {
			t.Fatalf("RetainMTP=%v: union(vision)=false, want true", retain)
		}
		if glmMoeDsaMTPOrVisionTensor("blk.0.attn_k_b.weight") {
			t.Fatalf("RetainMTP=%v: union(forward tensor)=true, want false", retain)
		}
	}
}

// TestEstimateCPUOffloadRetainsMTPBytes proves the byte-accounting delta: with RetainMTP OFF the
// MTP head counts nowhere (byte-identical to today); with RetainMTP ON it is accounted as a device
// weight (never an expert), so the device-weight total grows by exactly the head's payload bytes
// and the host-offload total is unchanged.
func TestEstimateCPUOffloadRetainsMTPBytes(t *testing.T) {
	const nextnBytes = int64(1<<20) * 4 // (1<<20) f32 elems
	newWS := func() *WeightSource {
		f := &File{
			Metadata: map[string]Value{
				"general.architecture": {Type: TypeString, Value: "glm-dsa"},
			},
			Tensors: []TensorInfo{
				{Name: "token_embd.weight", Dims: []uint64{256}, Type: TensorF32},           // device, 1024 B
				{Name: "blk.0.ffn_gate_inp.weight", Dims: []uint64{128}, Type: TensorF32},   // router device, 512 B
				{Name: "blk.0.attn_k_b.weight", Dims: []uint64{64}, Type: TensorF32},        // KV-b half device, 256 B
				{Name: "blk.0.ffn_gate_shexp.weight", Dims: []uint64{512}, Type: TensorF32}, // shared expert host, 2048 B
				{Name: "blk.0.ffn_gate_exps.weight", Dims: []uint64{1024}, Type: TensorF32}, // routed expert blob host, 4096 B
				{Name: "blk.78.nextn.eh_proj.weight", Dims: []uint64{1 << 20}, Type: TensorF32},
			},
		}
		ws, err := NewWeightSource(f, nil, 0)
		if err != nil {
			t.Fatalf("NewWeightSource: %v", err)
		}
		return ws
	}

	// RetainMTP OFF (default): the nextn head counts nowhere.
	off, err := newWS().EstimateCPUOffloadExpertsMemoryPlan()
	if err != nil {
		t.Fatalf("EstimateCPUOffloadExpertsMemoryPlan (off): %v", err)
	}
	offBy := off.ByClass()
	if got, want := offBy[compute.MemoryWeights], int64(1024+512+256); got != want {
		t.Fatalf("RetainMTP=off device weights = %d, want %d (MTP excluded)", got, want)
	}
	if got, want := offBy[compute.MemoryOffload], int64(2048+4096); got != want {
		t.Fatalf("RetainMTP=off host offload = %d, want %d", got, want)
	}

	// RetainMTP ON: the nextn head is accounted as a device weight; host offload unchanged.
	defer func() { model.RetainMTP = false }()
	model.RetainMTP = true
	on, err := newWS().EstimateCPUOffloadExpertsMemoryPlan()
	if err != nil {
		t.Fatalf("EstimateCPUOffloadExpertsMemoryPlan (on): %v", err)
	}
	onBy := on.ByClass()
	if got, want := onBy[compute.MemoryWeights], int64(1024+512+256)+nextnBytes; got != want {
		t.Fatalf("RetainMTP=on device weights = %d, want %d (MTP retained)", got, want)
	}
	if got, want := onBy[compute.MemoryOffload], int64(2048+4096); got != want {
		t.Fatalf("RetainMTP=on host offload = %d, want %d (unchanged)", got, want)
	}
	if delta := onBy[compute.MemoryWeights] - offBy[compute.MemoryWeights]; delta != nextnBytes {
		t.Fatalf("device-weight delta = %d, want the MTP head payload %d", delta, nextnBytes)
	}
}
