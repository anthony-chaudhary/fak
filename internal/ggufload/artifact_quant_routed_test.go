package ggufload

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// artifact_quant_routed_test.go — issue #1475 acceptance: the admission planner and the serve arm
// selector share ONE routed-expert encoding predicate, so a mixed-Q2_K/Q3_K MoE inventory is
// admitted for host offload and an unadmitted encoding refuses by the NAMED key instead of
// silently charging a transcoded (f32-inflated) footprint or selecting the device-resident arm.
// The WeightSource is built from a SYNTHETIC File (header only, no weights on disk), so this is
// pure header arithmetic and runs under -short.

// routedExpertQ2KQ3KFixture is the qualified mixed-encoding inventory: dense token_embd F32
// (device) + three routed-expert blobs (host) whose encodings are all in the enumerated admit-set
// (Q2_K gate/up, Q3_K down). Synth dims 1024 elems per blob are exact multiples of the 256-element
// k-quant super-block, so the payload math is fixed:
//   - Q2_K: 1024/256 * 84  = 336 B each (gate, up)
//   - Q3_K: 1024/256 * 110 = 440 B (down)
//   - F32 : 256 * 4        = 1024 B (token_embd)
const (
	routedQ2KBlobBytes = int64((1024 / 256) * 84)  // 336
	routedQ3KBlobBytes = int64((1024 / 256) * 110) // 440
	routedDenseF32     = int64(256 * 4)            // 1024
)

func routedExpertEncodingFixture(t *testing.T, downType TensorType) *WeightSource {
	t.Helper()
	f := &File{
		Metadata: synthGLMMeta(4),
		Tensors: []TensorInfo{
			{Name: "token_embd.weight", Dims: []uint64{256}, Type: TensorF32},
			{Name: "blk.0.ffn_gate_exps.weight", Dims: []uint64{1024}, Type: TensorQ2_K},
			{Name: "blk.0.ffn_up_exps.weight", Dims: []uint64{1024}, Type: TensorQ2_K},
			{Name: "blk.0.ffn_down_exps.weight", Dims: []uint64{1024}, Type: downType},
		},
	}
	ws, err := NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

// AdmittedRoutedExpertEncoding is an ENUMERATED admit-set: the 13 supported k-quant/IQ encodings
// are true, and everything else — including F16/BF16, whose 2 B/weight would expand to 4 B/weight
// on the raw-resident host pool — is refused rather than silently treated as device-scoped.
func TestAdmittedRoutedExpertEncodingIsEnumerated(t *testing.T) {
	for _, typ := range []TensorType{
		TensorQ2_K, TensorQ3_K, TensorQ4_K, TensorQ5_K, TensorQ6_K, TensorQ8_0,
		TensorIQ3_XXS, TensorIQ2_XXS, TensorIQ2_XS, TensorIQ1_S, TensorIQ2_S,
		TensorIQ1_M, TensorIQ4_XS,
	} {
		if !AdmittedRoutedExpertEncoding(typ) {
			t.Fatalf("AdmittedRoutedExpertEncoding(%s) = false, want true (admitted)", typ)
		}
	}
	for _, typ := range []TensorType{TensorF16, TensorBF16, TensorF32, TensorMXFP4, TensorQ4_0} {
		if AdmittedRoutedExpertEncoding(typ) {
			t.Fatalf("AdmittedRoutedExpertEncoding(%s) = true, want false (not in admit-set)", typ)
		}
	}
}

// A mixed-Q2_K/Q3_K routed-expert inventory (deepseek41/glm-dsa MoE) is QUALIFIED: the refusal is
// nil, and the offload plan host-scopes EXACTLY the routed experts while the dense tensor stays
// device-scoped. This is the "admission plan stored-vs-transformed" invariant: the planner charges
// the raw stored bytes the loader will hold, never an inflated transformed footprint.
func TestRoutedExpertMixedQ2KQ3KAdmittedAndHostScoped(t *testing.T) {
	ws := routedExpertEncodingFixture(t, TensorQ3_K)
	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if err := RoutedExpertEncodingRefusal(cfg, ws.File.Tensors); err != nil {
		t.Fatalf("mixed Q2_K/Q3_K routed experts must be admitted for host offload, got %v", err)
	}
	ok, typ, name := RoutedExpertResidencyQualified(cfg, ws.File.Tensors)
	if !ok || typ != 0 || name != "" {
		t.Fatalf("RoutedExpertResidencyQualified = (%v,%v,%q), want (true, 0, \"\")", ok, typ, name)
	}

	plan, err := ws.EstimateCPUOffloadExpertsMemoryPlan()
	if err != nil {
		t.Fatalf("EstimateCPUOffloadExpertsMemoryPlan: %v", err)
	}
	hostWant := routedQ2KBlobBytes + routedQ2KBlobBytes + routedQ3KBlobBytes
	if got := plan.HostTotal(); got != hostWant {
		t.Fatalf("HostTotal = %d, want routed experts only %d (%d Q2_K gate + %d Q2_K up + %d Q3_K down)",
			got, hostWant, routedQ2KBlobBytes, routedQ2KBlobBytes, routedQ3KBlobBytes)
	}
	if got := plan.DeviceTotal(); got != routedDenseF32 {
		t.Fatalf("DeviceTotal = %d, want the dense token_embd only %d", got, routedDenseF32)
	}
	if got := plan.Total(); got != hostWant+routedDenseF32 {
		t.Fatalf("Total = %d, want all bytes %d", got, hostWant+routedDenseF32)
	}

	// Every host-scoped row is a compute.MemoryOffload; the dense row stays device weights.
	var hostRows, deviceRows int
	for _, d := range plan {
		switch {
		case d.Scope == compute.MemoryScopeHost:
			hostRows++
			if d.Class != compute.MemoryOffload {
				t.Fatalf("host row = %+v, want MemoryOffload", d)
			}
			if d.Detail != "gguf-host-expert-offload" {
				t.Fatalf("host row detail = %q, want gguf-host-expert-offload", d.Detail)
			}
		case d.Scope == compute.MemoryScopeDevice:
			deviceRows++
			if d.Class != compute.MemoryWeights || d.Detail != "gguf-device-dense-load" {
				t.Fatalf("device row = %+v, want device-scoped dense weights", d)
			}
		}
	}
	if hostRows == 0 || deviceRows == 0 {
		t.Fatalf("plan = %+v, want both a host offload row and a device dense row", plan)
	}
}

// The named-refusal invariant: an unadmitted routed-expert encoding (F16 — 2 B/weight expanding to
// 4 B/weight) makes BOTH the predicate share key and the planner refuse with
// ErrRoutedExpertEncodingUnqualified, naming the offending tensor. Never a silent partial load.
func TestRoutedExpertUnadmittedEncodingRefusesByName(t *testing.T) {
	ws := routedExpertEncodingFixture(t, TensorF16)
	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	const offend = "blk.0.ffn_down_exps.weight"

	err = RoutedExpertEncodingRefusal(cfg, ws.File.Tensors)
	if err == nil {
		t.Fatal("F16 routed expert must be refused, got nil")
	}
	if !errors.Is(err, ErrRoutedExpertEncodingUnqualified) {
		t.Fatalf("RoutedExpertEncodingRefusal error %v, want errors.Is ErrRoutedExpertEncodingUnqualified", err)
	}
	if !strings.Contains(err.Error(), offend) {
		t.Fatalf("refusal %q must name the offending tensor %s", err.Error(), offend)
	}

	_, err = ws.EstimateCPUOffloadExpertsMemoryPlan()
	if err == nil {
		t.Fatal("EstimateCPUOffloadExpertsMemoryPlan must refuse an F16 routed expert, got nil")
	}
	if !errors.Is(err, ErrRoutedExpertEncodingUnqualified) {
		t.Fatalf("planner error %v, want errors.Is ErrRoutedExpertEncodingUnqualified", err)
	}
	if !strings.Contains(err.Error(), offend) {
		t.Fatalf("planner refusal %q must name the offending tensor %s", err.Error(), offend)
	}
}
