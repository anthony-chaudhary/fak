package model

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/computetrace"
	"github.com/anthony-chaudhary/fak/internal/stepobs"
)

// A self-contained CPU fixture for real Session construction, full-axis cold
// normalization, provenance and cleanup. DeviceMemory only selects software
// bindings; this type cannot pass compute's sealed Vulkan input-borrow gate.
// The existing scalar mHC equations are reproduced over owned CPU buffers.
// This is a metadata/parity witness, not an independent mHC arithmetic oracle.
type v41InputServingRecorder struct {
	compute.Backend
	owner   sync.Mutex
	weight  func() compute.Tensor
	inspect func()
	shapes  [][]int
	fault   error
}

func (b *v41InputServingRecorder) ForwardExecutionOwner() *sync.Mutex { return &b.owner }
func (b *v41InputServingRecorder) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory, c.UploadDtype = true, true
	return c
}
func (b *v41InputServingRecorder) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return compute.BackendSupportsDeviceWeightDtype(b.Backend, dt)
}
func (b *v41InputServingRecorder) owned(shape []int, values []float32) compute.Tensor {
	return compute.NewF32(b, append([]int(nil), shape...), append([]float32(nil), values...))
}
func (b *v41InputServingRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, _ compute.MemoryClass, _ string) compute.Tensor {
	if dt != compute.F32 {
		panic("input-scope fixture requires F32")
	}
	return b.owned(x.Shape, b.Backend.Read(x))
}
func (b *v41InputServingRecorder) RMSNorm(x, w compute.Tensor, eps float32) compute.Tensor {
	if b.weight != nil && w.Buf() == b.weight().Buf() {
		b.shapes = append(b.shapes, append([]int(nil), x.Shape...))
		if b.inspect != nil {
			b.inspect()
		}
		if b.fault != nil {
			panic(b.fault)
		}
	}
	raw := b.Backend.RMSNorm(x, w, eps)
	out := b.owned(raw.Shape, b.Backend.Read(raw))
	b.Backend.Free(raw)
	return out
}
func (b *v41InputServingRecorder) SupportsV41AttentionBF16(n int) bool { return n > 0 }
func (b *v41InputServingRecorder) SupportsV41BF16Publish(n int) bool   { return n > 0 }
func (b *v41InputServingRecorder) V41BF16Publish(x compute.Tensor, site uint32) (compute.Tensor, error) {
	return b.V41AttentionBF16(x, site)
}
func (b *v41InputServingRecorder) V41AttentionBF16(x compute.Tensor, site uint32) (compute.Tensor, error) {
	values := append([]float32(nil), b.Backend.Read(x)...)
	for i, value := range values {
		// Quotient/remainder nearest-even copyback, independent of the runtime helper.
		bits := math.Float32bits(value)
		upper, lower := uint16(bits>>16), uint16(bits)
		if lower > 0x8000 || (lower == 0x8000 && upper&1 != 0) {
			upper++
		}
		rounded := uint32(upper) << 16
		if rounded&0x7f800000 == 0x7f800000 {
			return compute.Tensor{}, &compute.V41BF16CheckedArithmeticError{Site: site, Index: i}
		}
		values[i] = math.Float32frombits(rounded)
	}
	return b.owned(x.Shape, values), nil
}
func (b *v41InputServingRecorder) SupportsV41MHC() bool { return true }
func (b *v41InputServingRecorder) operand(x compute.Tensor, shape ...int) []float32 {
	if (x.Backend() != b && x.Backend() != b.Backend) || x.Dtype != compute.F32 || !reflect.DeepEqual(x.Shape, shape) {
		panic("input-scope fixture mHC shape")
	}
	return b.Backend.Read(x)
}
func (b *v41InputServingRecorder) V41MHCInitialStreams(x compute.Tensor, rows, hidden int) (compute.Tensor, error) {
	in := b.operand(x, rows, hidden)
	out := make([]float32, rows*4*hidden)
	for row := 0; row < rows; row++ {
		for stream := 0; stream < 4; stream++ {
			for dim := 0; dim < hidden; dim++ {
				out[(row*4+stream)*hidden+dim] = v41RoundBF16(in[row*hidden+dim])
			}
		}
	}
	return b.owned([]int{rows, 4, hidden}, out), nil
}
func (b *v41InputServingRecorder) V41MHCSplit(mixes, scale, base compute.Tensor, rows, iters int, eps float32) (compute.Tensor, error) {
	m, sc, ba := b.operand(mixes, rows, 24), b.operand(scale, 3), b.operand(base, 24)
	out := make([]float32, 0, rows*24)
	for row := 0; row < rows; row++ {
		mix, err := v41MHCSplit(m[row*24:(row+1)*24], sc, ba, 4, iters, eps)
		if err != nil {
			return compute.Tensor{}, err
		}
		out = append(append(append(out, mix.pre...), mix.post...), mix.comb...)
	}
	return b.owned([]int{rows, 24}, out), nil
}
func (b *v41InputServingRecorder) V41MHCRMSSplit(raw, streams, params compute.Tensor, rows, hidden, iters int, eps, rmsEps float32) (compute.Tensor, error) {
	m, s, p := b.operand(raw, rows, 24), b.operand(streams, rows, 4, hidden), b.operand(params, 27)
	width := 4 * hidden
	out := make([]float32, 0, rows*24)
	for row := 0; row < rows; row++ {
		var squares float32
		for _, value := range s[row*width : (row+1)*width] {
			squares += float32(value * value)
		}
		inv := float32(1 / math.Sqrt(float64(squares/float32(width)+rmsEps)))
		scaled := make([]float32, 24)
		for i := range scaled {
			scaled[i] = float32(m[row*24+i] * inv)
		}
		mix, err := v41MHCSplit(scaled, p[:3], p[3:], 4, iters, eps)
		if err != nil {
			return compute.Tensor{}, err
		}
		out = append(append(append(out, mix.pre...), mix.post...), mix.comb...)
	}
	return b.owned([]int{rows, 24}, out), nil
}
func (b *v41InputServingRecorder) V41MHCPre(streams, coeffs compute.Tensor, rows, hidden int) (compute.Tensor, error) {
	s, k := b.operand(streams, rows, 4, hidden), b.operand(coeffs, rows, 24)
	out := make([]float32, rows*hidden)
	for row := 0; row < rows; row++ {
		for dim := 0; dim < hidden; dim++ {
			var acc float32
			for stream := 0; stream < 4; stream++ {
				acc += float32(k[row*24+stream] * s[(row*4+stream)*hidden+dim])
			}
			out[row*hidden+dim] = v41RoundBF16(acc)
		}
	}
	return b.owned([]int{rows, hidden}, out), nil
}
func (b *v41InputServingRecorder) V41MHCPost(x, streams, coeffs compute.Tensor, rows, hidden int) (compute.Tensor, error) {
	in, s, k := b.operand(x, rows, hidden), b.operand(streams, rows, 4, hidden), b.operand(coeffs, rows, 24)
	out := make([]float32, rows*4*hidden)
	for row := 0; row < rows; row++ {
		for dst := 0; dst < 4; dst++ {
			for dim := 0; dim < hidden; dim++ {
				var acc float32
				for src := 0; src < 4; src++ {
					acc += float32(k[row*24+8+src*4+dst] * s[(row*4+src)*hidden+dim])
				}
				acc += float32(k[row*24+4+dst] * v41RoundBF16(in[row*hidden+dim]))
				out[(row*4+dst)*hidden+dim] = v41RoundBF16(acc)
			}
		}
	}
	return b.owned([]int{rows, 4, hidden}, out), nil
}

// Build the same bounded full-axis shape as the original full-normalization
// fixture, using the real F32 loader and Session admission. HeadDim remains 512,
// unlike the reduced prefix-coverage fixture. No production source is replaced.
func v41InputServingModel(t *testing.T) *Model {
	t.Helper()
	raw, err := os.ReadFile("testdata/deepseek_v41_flash_config.json")
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	if hex.EncodeToString(sum[:]) != "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879" {
		t.Fatal("official config fixture changed")
	}
	var cfg Config
	if err = json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	cfg.ModelType = "deepseek41"
	cfg.NumLayers, cfg.HiddenSize, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim = 1, 64, 1, 1, 512
	cfg.QKNopeHeadDim, cfg.QKRopeHeadDim, cfg.QLoraRank = 448, 64, 32
	cfg.OLoraRank, cfg.OGroups, cfg.MoEIntermediateSize = 16, 1, 32
	cfg.VocabSize, cfg.NumExperts, cfg.NumExpertsPerTok, cfg.NSharedExperts, cfg.RoutedScalingFactor = 8, 384, 6, 1, 1.5
	cfg.RopeScaling, cfg.LongRope, cfg.RopeFactor, cfg.RopeOrigContext, cfg.RopeTheta = "", nil, 0, 0, 10000
	cfg.Window = []int{-1}
	cfg.DeepSeekV41.Attention = DeepSeekV41AttentionGeometry{}
	cfg.DeepSeekV41.CompressRatios = []int{0}
	cfg.DeepSeekV41.KVSourceLayerIDs, cfg.DeepSeekV41.IndexSourceLayerIDs, cfg.DeepSeekV41.EngramLayerIDs = nil, nil, nil
	if full, err := v41ForwardGeometry(cfg); err != nil || !full {
		t.Fatalf("full geometry admission: %t %v", full, err)
	}
	var tensors []NamedTensorF32
	seed := uint64(0x9E3779B97F4A7C15)
	add := func(name string, shape ...int) {
		n := 1
		for _, dim := range shape {
			n *= dim
		}
		data := make([]float32, n)
		for i := range data {
			switch {
			case strings.HasSuffix(name, "attn.wq_a_norm.weight") || strings.HasSuffix(name, "attn.kv_norm.weight"):
				data[i] = .5 + float32(i)/float32(n)
			case name == "model.norm.weight" || strings.HasSuffix(name, "attn_norm.weight") || strings.HasSuffix(name, "ffn_norm.weight") || strings.HasSuffix(name, "mhc.scale"):
				data[i] = 1
			case strings.HasSuffix(name, "mhc.ffn_scale"):
				data[i] = .75
			case strings.HasSuffix(name, "mhc.base"):
			default:
				seed = seed*6364136223846793005 + 1442695040888963407
				x := float32(seed>>40)/float32(1<<24)*2 - 1
				scale := float32(.1)
				if strings.Contains(name, "embed_tokens") {
					scale = .2
				}
				if strings.HasSuffix(name, "attn.sink") {
					scale = .25
				}
				data[i] = scale * x
			}
		}
		tensors = append(tensors, NamedTensorF32{Name: name, Shape: shape, Data: data})
	}
	add("model.embed_tokens.weight", 8, 64)
	add("lm_head.weight", 8, 64)
	add("model.norm.weight", 64)
	layer := func(name string, shape ...int) { add(layerName(0, name), shape...) }
	layer("attn_norm.weight", 64)
	layer("ffn_norm.weight", 64)
	layer("mhc.mixes.weight", 256, 24)
	layer("mhc.ffn_mixes.weight", 256, 24)
	layer("mhc.base", 24)
	layer("mhc.scale", 3)
	layer("mhc.ffn_base", 24)
	layer("mhc.ffn_scale", 3)
	layer("attn.wq_a.weight", 32, 64)
	layer("attn.wq_a_norm.weight", 32)
	layer("attn.wq_b.weight", 512, 32)
	layer("attn.wkv.weight", 512, 64)
	layer("attn.kv_norm.weight", 512)
	layer("attn.wo_a.weight", 16, 512)
	layer("attn.wo_b.weight", 64, 16)
	layer("attn.sink", 1)
	layer("ffn.gate.weight", 384, 64)
	layer("ffn.gate.e_score_correction_bias", 384)
	layer("ffn.shared_experts.w1.weight", 32, 64)
	layer("ffn.shared_experts.w3.weight", 32, 64)
	layer("ffn.shared_experts.w2.weight", 64, 32)
	for expert := 0; expert < 384; expert++ {
		stem := fmt.Sprintf("ffn.experts.%d.", expert)
		layer(stem+"w1.weight", 32, 64)
		layer(stem+"w3.weight", 32, 64)
		layer(stem+"w2.weight", 64, 32)
	}
	m, err := NewFromF32Tensors(cfg, tensors)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	})
	return m
}

// fak-test:runtime medium est=3s lane=default
func TestV41AttentionInputPanelSessionPrefillScope(t *testing.T) {
	if compute.Default().Name() != "cpu-ref" {
		t.Skip("CPU metadata/parity witness; Vulkan selection requires its physical witness")
	}
	t.Cleanup(func() { computetrace.SetObserver(nil) })
	for _, mode := range []string{"unobserved", "serving metrics", "arbitrary observer", "artifact recorder", "selected failure"} {
		t.Run(mode, func(t *testing.T) {
			computetrace.SetObserver(nil)
			if mode == "serving metrics" || mode == "artifact recorder" {
				stepobs.Attach()
			}
			if mode == "arbitrary observer" {
				computetrace.SetObserver(func(computetrace.Event) {})
			}
			if mode == "artifact recorder" {
				_, disable := computetrace.Enable(8, "input-scope", "prefill")
				defer disable()
			}
			m := v41InputServingModel(t)
			b := &v41InputServingRecorder{Backend: compute.Default()}
			fault := errors.New("selected attention normalization failure")
			if mode == "selected failure" {
				b.fault = fault
			}
			s, err := m.NewBackendSessionChecked(b)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(s.Close)
			b.weight = func() compute.Tensor { return s.halW[layerName(0, "attn_norm.weight")] }
			source := s.v41State()
			if len(source.history) != 0 {
				t.Fatal("fixture must start with zero reused prefix tokens")
			}
			var captured *v41AttentionInputRun
			var observed *v41AttentionInputPanel
			var bound *v41ProjScratch
			b.inspect = func() {
				captured = source.attentionInputRun
				if captured == nil || captured.run == source || !captured.valid(s, captured.run) {
					t.Fatal("Session.Prefill did not bind its temporary cold-forward state")
				}
				run := captured.run
				bound = &v41ProjScratch{}
				bound.bindLayerSessionBindings(run)
				observed = &v41AttentionInputPanel{owner: s, model: m, backend: s.Backend, state: run, run: captured,
					norm: run.attentionNormBinding, route: run.projectionRoute, query: run.queryNorm, gather: run.gatherOps,
					publication: run.gatherOps.bf16Publish, scope: s.backendExecutionScope, epoch: s.cacheGeometryEpoch,
					observation: stepobs.CaptureHostInputObservation(), layer: 0, width: m.Cfg.HiddenSize, rows: 2}
				want := mode == "unobserved" || mode == "serving metrics" || mode == "selected failure"
				if got := observed.scopeValid(bound); got != want {
					t.Fatalf("actual cold-forward metadata admission=%t, want %t", got, want)
				}
				if observed.valid(bound) || run.attentionNormBinding.newInputPanel(m, run, bound, 0, m.Cfg.HiddenSize, 2) != nil {
					t.Fatal("CPU recorder bypassed the sealed backend gate")
				}
			}
			var got []float32
			var failure error
			func() {
				defer func() {
					if value := recover(); value != nil {
						var ok bool
						failure, ok = value.(error)
						if !ok {
							t.Fatalf("unexpected panic: %v", value)
						}
					}
				}()
				got = s.Prefill([]int{1, 2})
			}()
			if mode == "selected failure" {
				if !errors.Is(failure, fault) || !s.BackendSessionClosed() || len(source.history) != 0 {
					t.Fatalf("selected failure did not retire and roll back the owner: %v", failure)
				}
			} else if failure != nil {
				t.Fatalf("Session.Prefill failed: %v", failure)
			}
			if captured == nil || observed == nil || !reflect.DeepEqual(b.shapes, [][]int{{2, m.Cfg.HiddenSize}}) {
				t.Fatal("real cold normalization panel was not traversed")
			}
			if captured.live || captured.valid(s, captured.run) || source.attentionInputRun != nil || captured.run.attentionInputRun != nil || observed.scopeValid(bound) || s.backendExecutionScope != nil {
				t.Fatal("completed forward retained a live input-borrow capability")
			}
			if mode != "selected failure" {
				want := lastLogits(m.Forward([]int{1, 2}))
				if len(got) != len(want) || len(got) == 0 {
					t.Fatal("empty or mismatched logits")
				}
				var signal, delta float64
				for i, value := range got {
					if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.IsNaN(float64(want[i])) || math.IsInf(float64(want[i]), 0) {
						t.Fatal("nonfinite logits")
					}
					signal = max(signal, math.Abs(float64(want[i])))
					delta = max(delta, math.Abs(float64(value-want[i])))
				}
				if signal == 0 || delta > 1e-4*max(1, signal) {
					t.Fatalf("logits parity: signal=%g delta=%g", signal, delta)
				}
			}
		})
	}
}
