package main

import (
	"bytes"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// serve_streamed_offload_test.go — fak#13121 acceptance at the serve-arm seam. The MoE
// --cpu-offload-experts planner charged the whole routed-expert payload host-resident, so a
// DeepSeek-V4.1 Flash Q2_K checkpoint (183.25 GiB of experts against a 62 GiB Halo) refused before
// any load. The streamed policy charges a BOUNDED resident working set instead and faults the rest
// from the staged shards, so the serve arm must SELECT it exactly when the full charge cannot be
// host-resident AND the checkpoint tier can stage the routed slabs.

// serveStreamedSynthWeightSource is a Q2_K routed-expert MoE fixture (the DeepSeek-V4.1 Flash
// encoding) whose fused slabs are 3-D [E, out, in] and therefore stageable by the checkpoint tier.
// Header only: NewWeightSource indexes the directory without reading a payload. GGUF stores dims
// reversed, so Dims {in, out, E} yields shape [E, out, in].
func serveStreamedSynthWeightSource(t *testing.T) *ggufload.WeightSource {
	t.Helper()
	const (
		experts = 4
		hidden  = 256
	)
	// One Q2_K slab is experts*(hidden/256)*84 bytes; the offsets below are absolute into the blob.
	slabBytes := int64(experts) * int64(hidden/256) * 84
	f := &ggufload.File{
		Metadata: map[string]ggufload.Value{
			"general.architecture":                         {Type: ggufload.TypeString, Value: "glm_moe_dsa"},
			"glm_moe_dsa.context_length":                   {Type: ggufload.TypeUint64, Value: uint64(16)},
			"glm_moe_dsa.embedding_length":                 {Type: ggufload.TypeUint64, Value: uint64(32)},
			"glm_moe_dsa.block_count":                      {Type: ggufload.TypeUint64, Value: uint64(2)},
			"glm_moe_dsa.feed_forward_length":              {Type: ggufload.TypeUint64, Value: uint64(64)},
			"glm_moe_dsa.attention.head_count":             {Type: ggufload.TypeUint64, Value: uint64(4)},
			"glm_moe_dsa.attention.head_count_kv":          {Type: ggufload.TypeUint64, Value: uint64(2)},
			"glm_moe_dsa.attention.layer_norm_rms_epsilon": {Type: ggufload.TypeFloat32, Value: float32(1e-5)},
			"glm_moe_dsa.rope.freq_base":                   {Type: ggufload.TypeFloat32, Value: float32(10000)},
			"glm_moe_dsa.expert_count":                     {Type: ggufload.TypeUint64, Value: uint64(experts)},
			"glm_moe_dsa.expert_used_count":                {Type: ggufload.TypeUint64, Value: uint64(2)},
			"glm_moe_dsa.expert_feed_forward_length":       {Type: ggufload.TypeUint64, Value: uint64(hidden)},
			"tokenizer.ggml.eos_token_id":                  {Type: ggufload.TypeUint32, Value: uint32(2)},
		},
		Tensors: []ggufload.TensorInfo{
			{Name: "token_embd.weight", Dims: []uint64{256}, Type: ggufload.TensorF32, FileOffset: 0},
			{Name: "blk.0.ffn_gate_exps.weight", Dims: []uint64{hidden, hidden, experts}, Type: ggufload.TensorQ2_K, FileOffset: 1024},
			{Name: "blk.0.ffn_up_exps.weight", Dims: []uint64{hidden, hidden, experts}, Type: ggufload.TensorQ2_K, FileOffset: 1024 + slabBytes},
			{Name: "blk.0.ffn_down_exps.weight", Dims: []uint64{hidden, hidden, experts}, Type: ggufload.TensorQ2_K, FileOffset: 1024 + 2*slabBytes},
		},
	}
	// A real in-memory shard so FusedExpertTensors can resolve each slab's owning reader (the
	// checkpoint tier reads through it).
	blob := make([]byte, 1024+3*slabBytes)
	ws, err := ggufload.NewWeightSource(f, bytes.NewReader(blob), int64(len(blob)))
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

// The routed slabs of the fixture must be stageable by the checkpoint tier — this is the artifact
// half of the streamed decision, so if it is false the feature is silently inert on the exact
// artifact it exists for.
func TestServeStreamedExpertSlabsAreStageable(t *testing.T) {
	ws := serveStreamedSynthWeightSource(t)
	if !serveStreamedExpertsCapable(ws) {
		t.Fatal("fixture routed slabs are not stageable; the streamed decision has no fault path to arm")
	}
}

// A routed set that fits the host budget keeps the RESIDENT arm byte-for-byte (P4 preserved): the
// streamed policy must not fire when the full charge is host-resident.
func TestServeCPUOffloadStreamedArmNotSelectedWhenResidentFits(t *testing.T) {
	ws := serveStreamedSynthWeightSource(t)
	// A huge host budget: the full routed charge fits, so streaming is unnecessary.
	bigFit := serveFitBudget{Base: 1 << 40, Headroom: 0}
	plan, streamed, err := serveStreamedCPUOffloadPlan(ws, 1, 0, bigFit)
	if err != nil {
		t.Fatalf("serveStreamedCPUOffloadPlan: %v", err)
	}
	if streamed {
		t.Fatalf("streamed policy fired although the full routed charge fits the host budget; plan=%+v", plan)
	}
	resident, err := serveGGUFCPUOffloadMemoryPlan(ws, 1, 0, bigFit)
	if err != nil {
		t.Fatalf("resident plan: %v", err)
	}
	if plan.HostTotal() != resident.HostTotal() {
		t.Fatalf("non-streamed plan host total %d, want the resident plan %d (must be byte-identical)", plan.HostTotal(), resident.HostTotal())
	}
}

// The core arm witness: when the full routed charge cannot be host-resident AND the slabs are
// stageable, the streamed policy is selected, the bounded resident bound is charged, and the device
// dense side is UNCHANGED against the resident plan's device side.
func TestServeCPUOffloadStreamedArm(t *testing.T) {
	ws := serveStreamedSynthWeightSource(t)
	if !serveStreamedExpertsCapable(ws) {
		t.Fatal("fixture routed slabs are not stageable; the streamed fault path has no work here")
	}

	resident, err := serveGGUFCPUOffloadMemoryPlan(ws, 1, 0, serveFitBudget{})
	if err != nil {
		t.Fatalf("resident plan: %v", err)
	}
	if resident.HostTotal() <= 0 {
		t.Fatal("fixture must host-scope the routed experts (HostTotal>0)")
	}

	// A host budget far below the routed charge forces the streamed decision.
	tinyFit := serveFitBudget{Base: 512, Headroom: 0}
	plan, streamed, err := serveStreamedCPUOffloadPlan(ws, 1, 0, tinyFit)
	if err != nil {
		t.Fatalf("serveStreamedCPUOffloadPlan: %v", err)
	}
	if !streamed {
		t.Fatalf("streamed policy did not fire with a bounded host budget %d against a routed set %d", tinyFit.Base, resident.HostTotal())
	}
	if plan.HostTotal() > tinyFit.avail() {
		t.Fatalf("streamed host total %d exceeds the bounded resident budget %d", plan.HostTotal(), tinyFit.avail())
	}
	if plan.DeviceTotal() != resident.DeviceTotal() {
		t.Fatalf("streamed device total %d, want the unchanged dense side %d", plan.DeviceTotal(), resident.DeviceTotal())
	}
	byDetail := map[string]int64{}
	for _, d := range plan {
		byDetail[d.Detail] += d.Bytes
	}
	if byDetail["gguf-host-expert-offload-streamed"] == 0 {
		t.Fatalf("streamed plan carries no gguf-host-expert-offload-streamed row; plan=%+v", plan)
	}
}
