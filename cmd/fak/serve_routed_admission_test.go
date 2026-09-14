package main

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// serve_routed_admission_test.go — issue #1475 acceptance at the serve-arm seam: a mixed-Q2_K/Q3_K
// routed-expert MoE artifact with NO Q4_K tensor must select the --cpu-offload-experts arm when
// offload intent is set (not the device-resident Q4_K arm and not the Q8 arm), and the routed
// experts must be host-scoped in the resulting plan. An unadmitted expert encoding refuses by the
// named ggufload.ErrRoutedExpertEncodingUnqualified key. Header only: NewWeightSource indexes the
// tensor directory without touching a payload.

// Serve-layer routed-expert inventory: glm-dsa metadata with Q2_K gate/up and Q3_K down expert
// blobs (NO Q4_K anywhere) plus a dense F32 token_embd. There is no Q4_K tensor, so
// ClassifyTensorQuant(...).Q4KResident is false and only the OFFLOAD intent can reach the
// cpu-offload-experts arm.
func serveSynthRoutedExpertWeightSource(t *testing.T, downType ggufload.TensorType) *ggufload.WeightSource {
	t.Helper()
	f := &ggufload.File{
		Metadata: map[string]ggufload.Value{
			"general.architecture":                     {Type: ggufload.TypeString, Value: "glm-dsa"},
			"glm-dsa.context_length":                   {Type: ggufload.TypeUint64, Value: uint64(16)},
			"glm-dsa.embedding_length":                 {Type: ggufload.TypeUint64, Value: uint64(32)},
			"glm-dsa.block_count":                      {Type: ggufload.TypeUint64, Value: uint64(2)},
			"glm-dsa.feed_forward_length":              {Type: ggufload.TypeUint64, Value: uint64(64)},
			"glm-dsa.attention.head_count":             {Type: ggufload.TypeUint64, Value: uint64(4)},
			"glm-dsa.attention.head_count_kv":          {Type: ggufload.TypeUint64, Value: uint64(2)},
			"glm-dsa.attention.layer_norm_rms_epsilon": {Type: ggufload.TypeFloat32, Value: float32(1e-5)},
			"glm-dsa.rope.freq_base":                   {Type: ggufload.TypeFloat32, Value: float32(10000)},
			"glm-dsa.expert_count":                     {Type: ggufload.TypeUint64, Value: uint64(4)},
			"glm-dsa.expert_used_count":                {Type: ggufload.TypeUint64, Value: uint64(2)},
			"glm-dsa.expert_feed_forward_length":       {Type: ggufload.TypeUint64, Value: uint64(64)},
			"tokenizer.ggml.eos_token_id":              {Type: ggufload.TypeUint32, Value: uint32(2)},
		},
		Tensors: []ggufload.TensorInfo{
			{Name: "token_embd.weight", Dims: []uint64{256}, Type: ggufload.TensorF32},
			{Name: "blk.0.ffn_gate_exps.weight", Dims: []uint64{1024}, Type: ggufload.TensorQ2_K},
			{Name: "blk.0.ffn_up_exps.weight", Dims: []uint64{1024}, Type: ggufload.TensorQ2_K},
			{Name: "blk.0.ffn_down_exps.weight", Dims: []uint64{1024}, Type: downType},
		},
	}
	ws, err := ggufload.NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

// A mixed-Q2_K/Q3_K artifact with NO Q4_K selects the cpu-offload-experts arm ONLY when offload
// intent is set; without it the arm is the Q8 dequant arm. The offload arm's plan host-scopes the
// routed experts (HostTotal>0).
func TestServeDeviceLoadArmSelectsCPUOffloadForPackedExpertArtifact(t *testing.T) {
	ws := serveSynthRoutedExpertWeightSource(t, ggufload.TensorQ3_K)
	be := serveCapBackend{Backend: compute.Default(), total: 8 << 30, free: 8 << 30, known: true, uploadDtype: true}

	if quant := ggufload.ClassifyTensorQuant(ws.File.Tensors); quant.Q4KResident {
		t.Fatalf("fixture must carry no Q4_K tensor, got Q4KResident=true inventory=%s", quant.Inventory)
	}

	if got := resolveDeviceServeLoadArm(ws, be, false, true); got != serveLoadArmCPUOffloadExperts {
		t.Fatalf("offload intent on a qualified packed-expert artifact = %q, want %q", got, serveLoadArmCPUOffloadExperts)
	}
	if got := resolveDeviceServeLoadArm(ws, be, false, false); got != serveLoadArmQuantProfileQ8 {
		t.Fatalf("no offload intent = %q, want the Q8 arm %q (offload intent is required)", got, serveLoadArmQuantProfileQ8)
	}

	plan, err := serveGGUFWeightMemoryPlanForArm(ws, serveLoadArmCPUOffloadExperts)
	if err != nil {
		t.Fatalf("serveGGUFWeightMemoryPlanForArm(cpu-offload-experts): %v", err)
	}
	if plan.HostTotal() <= 0 {
		t.Fatalf("cpu-offload-experts plan HostTotal = %d, want >0 (routed experts host-scoped); plan=%+v", plan.HostTotal(), plan)
	}
	if got, want := plan.ByClass()[compute.MemoryOffload], int64((1024/256)*84*2+(1024/256)*110); got != want {
		t.Fatalf("host offload = %d, want the routed set %d", got, want)
	}
}

// A backend WITHOUT the offload intent must NOT get the offload arm even though the artifact is
// qualified — the arm is driven by the operator flag, not the artifact alone.
func TestServeHostLoadArmRequiresOffloadIntent(t *testing.T) {
	ws := serveSynthRoutedExpertWeightSource(t, ggufload.TensorQ3_K)
	if got := resolveHostServeLoadArm(ws, false, true); got != serveLoadArmCPUOffloadExperts {
		t.Fatalf("host offload intent on a qualified artifact = %q, want %q", got, serveLoadArmCPUOffloadExperts)
	}
	if got := resolveHostServeLoadArm(ws, false, false); got == serveLoadArmCPUOffloadExperts {
		t.Fatalf("host arm without offload intent = %q, must not be the offload arm", got)
	}
}

// The serve-layer negative: the SAME fixture with F16 routed experts makes
// serveArtifactCPUOffloadExperts refuse with the named key, never a silent partial load.
func TestServeArtifactCPUOffloadExpertsRefusesUnadmittedEncoding(t *testing.T) {
	ws := serveSynthRoutedExpertWeightSource(t, ggufload.TensorF16)
	ok, err := serveArtifactCPUOffloadExperts(ws)
	if ok {
		t.Fatal("F16 routed experts must not be admitted for host offload")
	}
	if err == nil {
		t.Fatal("serveArtifactCPUOffloadExperts must return the named refusal for F16 experts, got nil")
	}
	if !errors.Is(err, ggufload.ErrRoutedExpertEncodingUnqualified) {
		t.Fatalf("error %v, want errors.Is ggufload.ErrRoutedExpertEncodingUnqualified", err)
	}
	if !strings.Contains(err.Error(), "blk.0.ffn_down_exps.weight") {
		t.Fatalf("refusal %q must name the offending tensor", err.Error())
	}
}
