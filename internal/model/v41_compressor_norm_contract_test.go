package model

import (
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const v41CompressorNormTestLeaf = "attn.compressor.norm.weight"

type v41CompressorNormRecord struct {
	layer       int
	input, gain []float32
	eps         float32
	weight      compute.Buffer
}

// The unqualified recorder deliberately exposes generic device RMSNorm without
// the compressor promise. The qualified wrapper below is test-only and binds a
// fixed fixture width/epsilon; neither type establishes physical qualification.
type v41CompressorNormRecorder struct {
	*v41DenseTestBackend
	owner                                 *Session
	records                               []v41CompressorNormRecord
	outputs                               map[compute.Buffer]int
	weights                               map[compute.Buffer]bool
	frees                                 map[compute.Buffer]int
	readbacks, uploadBytes, readBytes, io int
	label                                 string
	faultLayer                            int
	faultSite                             string
	fault                                 any
	beforeFault                           func()
}

func newV41CompressorNormRecorder() *v41CompressorNormRecorder {
	return &v41CompressorNormRecorder{
		v41DenseTestBackend: newV41DenseTestBackend(),
		outputs:             map[compute.Buffer]int{}, weights: map[compute.Buffer]bool{}, frees: map[compute.Buffer]int{}, faultLayer: -1,
	}
}

func (b *v41CompressorNormRecorder) Name() string {
	if b.label != "" {
		return b.label
	}
	return b.Backend.Name()
}

func (b *v41CompressorNormRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, _ compute.MemoryClass, site string) compute.Tensor {
	b.io++
	if b.owner != nil && strings.HasSuffix(site, layerName(b.faultLayer, v41CompressorNormTestLeaf)) {
		if (b.faultSite == "weight upload" && strings.HasPrefix(site, "hal-weight ")) ||
			(b.faultSite == "activation upload" && strings.HasPrefix(site, "V4.1 RMSNorm activation ")) {
			panic(b.fault)
		}
	}
	out := b.v41DenseTestBackend.Upload(x, dt)
	if strings.HasPrefix(site, "hal-weight ") {
		b.weights[out.Buf()] = true
	}
	if strings.HasPrefix(site, "V4.1 RMSNorm activation ") && strings.HasSuffix(site, v41CompressorNormTestLeaf) {
		b.uploadBytes += b.uploads[out.Buf()]
	}
	return out
}

func (b *v41CompressorNormRecorder) RMSNorm(x, weight compute.Tensor, eps float32) compute.Tensor {
	b.io++
	layer := -1
	if b.owner != nil {
		for l := 0; l < b.owner.M.Cfg.NumLayers; l++ {
			if gain, ok := b.owner.halW[layerName(l, v41CompressorNormTestLeaf)]; ok && gain.Buf() == weight.Buf() {
				layer = l
				break
			}
		}
	}
	if layer >= 0 {
		b.records = append(b.records, v41CompressorNormRecord{layer: layer,
			input: append([]float32(nil), b.Backend.Read(x)...),
			gain:  append([]float32(nil), b.Backend.Read(weight)...), eps: eps, weight: weight.Buf()})
		if layer == b.faultLayer && b.faultSite == "rmsnorm" {
			if b.beforeFault != nil {
				b.beforeFault()
			}
			panic(b.fault)
		}
	}
	y := b.Backend.RMSNorm(x, weight, eps)
	b.live[y.Buf()] = true
	if layer >= 0 {
		b.outputs[y.Buf()] = layer
		if layer == b.faultLayer && b.faultSite == "malformed output" {
			y.Shape = []int{len(b.records[len(b.records)-1].input) + 1}
		}
	}
	return y
}

func (b *v41CompressorNormRecorder) Read(x compute.Tensor) []float32 {
	b.io++
	values := b.v41DenseTestBackend.Read(x)
	if layer, ok := b.outputs[x.Buf()]; ok {
		b.readbacks++
		b.readBytes += 4 * len(values)
		if layer == b.faultLayer {
			switch b.faultSite {
			case "readback":
				panic(b.fault)
			case "short readback":
				return values[:len(values)-1]
			case "nonfinite readback":
				values = append([]float32(nil), values...)
				values[0] = float32(math.NaN())
			case "BF16 output":
				values = append([]float32(nil), values...)
				values[0] = math.MaxFloat32 // Finite F32, infinite after BF16 RNE.
			}
		}
	}
	return values
}

func (b *v41CompressorNormRecorder) Free(x compute.Tensor) {
	b.io++
	b.frees[x.Buf()]++
	if _, ok := b.outputs[x.Buf()]; ok {
		values := b.Backend.Read(x)
		for i := range values {
			values[i] = float32(math.NaN())
		}
		delete(b.outputs, x.Buf())
	}
	b.v41DenseTestBackend.Free(x)
}

type v41CompressorNormQualifiedRecorder struct {
	*v41CompressorNormRecorder
	width   int
	epsBits uint32
	allow   bool
}

func (b *v41CompressorNormQualifiedRecorder) SupportsV41CompressorNorm(width int, eps float32) bool {
	return b.allow && width == b.width && math.Float32bits(eps) == b.epsBits
}

func v41CompressorNormTestSession(t *testing.T, m *Model) (*Session, *v41CompressorNormQualifiedRecorder) {
	t.Helper()
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("compressor normalization recorder requires cpu-ref")
	}
	b := &v41CompressorNormQualifiedRecorder{v41CompressorNormRecorder: newV41CompressorNormRecorder(),
		width: v41CompressorWidth(m.Cfg), epsBits: math.Float32bits(float32(m.Cfg.RMSNormEps)), allow: true}
	// Register model release first: cleanup is LIFO, so every session (including
	// a later restore target) closes before immutable device residents are freed.
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	})
	var s *Session
	if m.Cfg.DeepSeekV41 == nil {
		// Direct adapter controls need no model-forward/KV constructor: the
		// tiny manifest intentionally contains only one learned norm gain.
		s = &Session{M: m, Backend: b, halW: map[string]compute.Tensor{}}
		t.Cleanup(s.Close)
	} else {
		s = v41DenseTestSession(t, m, b)
	}
	b.owner = s
	return s, b
}

func v41CompressorNormTinyModel(t *testing.T, gain []float32, eps float32) *Model {
	t.Helper()
	name := layerName(0, v41CompressorNormTestLeaf)
	manifest, raw := synthBuildRaw([]synthTensor{{name, []int{len(gain)}}}, synthMatmulFill)
	m := &Model{Cfg: Config{NumLayers: 1, HeadDim: len(gain), RMSNormEps: float64(eps)}, manifest: manifest, raw: raw}
	v41WriteTensorF32(t, m, name, gain)
	return m
}

// This independently expresses ties-to-even by examining the discarded half,
// rather than using the production helper's additive rounding shortcut.
func v41CompressorNormRefCast(x float32) float32 {
	bits := math.Float32bits(x)
	hi, lo := bits>>16, bits&0xffff
	if lo > 0x8000 || (lo == 0x8000 && hi&1 != 0) {
		hi++
	}
	return math.Float32frombits(hi << 16)
}

func v41CompressorNormRefTail(input, gain []float32, eps float32, omit string) []float32 {
	row := append([]float32(nil), input...)
	var sum float32
	for i, value := range row {
		if omit != "input cast" {
			value = v41CompressorNormRefCast(value)
		}
		row[i] = value
		sum += value * value
	}
	inverse := float32(1 / math.Sqrt(float64(sum/float32(len(row))+eps)))
	for i := range row {
		row[i] *= inverse
		if omit != "gain" {
			row[i] *= gain[i]
		}
		if omit != "output cast" {
			row[i] = v41CompressorNormRefCast(row[i])
		}
	}
	return row
}

func v41CompressorNormEqualBits(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

// Exact software controls only. A finite corpus cannot qualify an arbitrary
// physical reduction, learned-gain domain, checkpoint, or Vulkan backend.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=200ms lane=default
func TestV41CompressorNormContract(t *testing.T) {
	t.Parallel()
	midpoints := []float32{math.Float32frombits(0x3f808000), math.Float32frombits(0x3f818000), math.Float32frombits(0xbf808000), math.Float32frombits(0xbf818000)}
	corpus := []struct {
		name        string
		input, gain []float32
		eps         float32
	}{
		{"input-midpoints", midpoints, []float32{0.5, -1.25, 1.75, 0.25}, 1e-5},
		{"output-midpoints", []float32{1, 1, 1, 1}, midpoints, 1e-20},
		{"cast-sensitive", []float32{1.003, 0.503, -0.126, 2.011}, []float32{0.7, -0.9, 1.3, 1.9}, 1e-5},
		{"small", []float32{1e-20, -3e-20, 2e-21, -4e-21}, []float32{-2, 0, 0.5, 1.25}, 1e-20},
		{"subnormal", []float32{math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32, 1e-40, -1e-40}, []float32{1, -1, 2, 0.5}, 1e-20},
		{"zero-row", []float32{0, math.Float32frombits(0x80000000), 0, 0}, []float32{1, 2, -1, 0}, 1e-5},
		{"zero-gain", []float32{1, -2, 3, -4}, []float32{0, 0, 0, 0}, 1e-5},
	}
	detected := map[string]bool{"input cast": false, "output cast": false, "gain": false}
	for _, tc := range corpus {
		t.Run(tc.name, func(t *testing.T) {
			m := v41CompressorNormTinyModel(t, tc.gain, tc.eps)
			s, b := v41CompressorNormTestSession(t, m)
			normalize := s.v41CompressorNormFunc()
			if normalize == nil {
				t.Fatal("explicit CPU fixture promise did not select the adapter")
			}
			input, gain := append([]float32(nil), tc.input...), append([]float32(nil), tc.gain...)
			got, err := normalize(0, input, gain, tc.eps)
			want := v41CompressorNormRefTail(tc.input, tc.gain, tc.eps, "")
			if err != nil || !v41CompressorNormEqualBits(got, want) {
				t.Fatalf("serial BF16/RMSNorm/BF16 contract mismatch: got=%v want=%v err=%v", got, want, err)
			}
			hostPool, err := NewV41CompressorPool(1, len(input))
			if err != nil {
				t.Fatal(err)
			}
			host, emitted, err := hostPool.PushNormalized(0, tc.input, nil, tc.gain, tc.eps)
			if err != nil || !emitted || !v41CompressorNormEqualBits(got, host) {
				t.Fatal("CPU adapter changed the original serial nil-callback tail")
			}
			if tc.name == "output-midpoints" {
				for i, bits := range []uint32{0x3f800000, 0x3f820000, 0xbf800000, 0xbf820000} {
					if math.Float32bits(got[i]) != bits {
						t.Fatal("final cast lost signed halfway-even/odd fixture")
					}
				}
			}
			if !v41CompressorNormEqualBits(input, tc.input) || !v41CompressorNormEqualBits(gain, tc.gain) {
				t.Fatal("normalization changed caller-owned input or learned gain")
			}
			if len(b.records) != 1 || b.readbacks != 1 || b.uploadBytes != 4*len(input) || b.readBytes != b.uploadBytes || len(b.outputs) != 0 {
				t.Fatal("one completed row did not produce one owned readback and one RMSNorm")
			}
			firstCast := make([]float32, len(input))
			for i, value := range input {
				firstCast[i] = v41CompressorNormRefCast(value)
				if !finite32(got[i]) || math.Float32bits(got[i])&0xffff != 0 {
					t.Fatal("published row is not finite widened BF16")
				}
			}
			if !v41CompressorNormEqualBits(b.records[0].input, firstCast) || !v41CompressorNormEqualBits(b.records[0].gain, tc.gain) || math.Float32bits(b.records[0].eps) != math.Float32bits(tc.eps) {
				t.Fatal("RMSNorm did not receive exact first-cast row and canonical gain/epsilon")
			}
			weight := b.records[0].weight
			if _, err := normalize(0, input, gain, tc.eps); err != nil || b.records[1].weight != weight || b.frees[weight] != 0 {
				t.Fatal("normalization restaged or freed the immutable borrowed gain")
			}
			s.Close()
			if b.frees[weight] != 0 {
				t.Fatal("session Close freed a borrowed norm gain")
			}
			if err := m.CloseWeights(); err != nil {
				t.Fatal(err)
			}
			if b.frees[weight] != 1 || len(b.live) != 0 {
				t.Fatal("model did not free its norm gain exactly once")
			}
			for omit := range detected {
				detected[omit] = detected[omit] || !v41CompressorNormEqualBits(want, v41CompressorNormRefTail(tc.input, tc.gain, tc.eps, omit))
			}
		})
	}
	for omit, caught := range detected {
		if !caught {
			t.Errorf("cast-sensitive corpus failed to detect omitted %s", omit)
		}
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=100ms lane=default
func TestV41CompressorNormAdmission(t *testing.T) {
	t.Parallel()
	m := v41CompressorNormTinyModel(t, []float32{1, -1, 0.5, 2}, 1e-5)
	var absent *Session
	if absent.v41CompressorNormFunc() != nil || (&Session{M: m}).v41CompressorNormFunc() != nil || (&Session{M: m, Backend: compute.Default()}).v41CompressorNormFunc() != nil {
		t.Fatal("absent or non-device backend selected compressor normalization")
	}
	ordinary := newV41CompressorNormRecorder()
	ordinary.label = "vulkan"
	if (&Session{M: m, Backend: ordinary}).v41RMSNormFunc() == nil || (&Session{M: m, Backend: ordinary}).v41CompressorNormFunc() != nil || len(ordinary.records) != 0 {
		t.Fatal("ordinary Vulkan-like generic RMSNorm must remain unselected")
	}
	for _, name := range []string{"capability-false", "wrong-width", "wrong-epsilon", "no-f32", "layernorm", "gain-plus-one", "zero-epsilon", "nan-epsilon"} {
		t.Run(name, func(t *testing.T) {
			model := v41CompressorNormTinyModel(t, []float32{1, 1, 1, 1}, 1e-5)
			b := &v41CompressorNormQualifiedRecorder{v41CompressorNormRecorder: newV41CompressorNormRecorder(), width: 4, epsBits: math.Float32bits(1e-5), allow: true}
			switch name {
			case "capability-false":
				b.allow = false
			case "wrong-width":
				b.width++
			case "wrong-epsilon":
				b.epsBits++
			case "no-f32":
				b.deny = true
			case "layernorm":
				model.Cfg.LayerNorm = true
			case "gain-plus-one":
				model.Cfg.NormGain1p = true
			case "zero-epsilon":
				model.Cfg.RMSNormEps = 0
			case "nan-epsilon":
				model.Cfg.RMSNormEps = math.NaN()
			}
			if (&Session{M: model, Backend: b}).v41CompressorNormFunc() != nil || b.io != 0 {
				t.Fatal("ineligible adapter selected or performed device work")
			}
		})
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=100ms lane=default
func TestV41CompressorNormPoolCompletion(t *testing.T) {
	t.Parallel()
	pool, err := NewV41CompressorPool(2, 2)
	if err != nil {
		t.Fatal(err)
	}
	calls := 0
	pool.normalize = func(pooled, gain []float32, eps float32) ([]float32, error) {
		calls++
		return v41CompressorNormRefTail(pooled, gain, eps, ""), nil
	}
	before := *pool
	for _, args := range []struct {
		gain []float32
		eps  float32
	}{
		{[]float32{1}, 1e-5}, {[]float32{1, float32(math.NaN())}, 1e-5}, {[]float32{1, 1}, 0},
	} {
		if _, emitted, err := pool.PushNormalized(0, []float32{1, 2}, []float32{0, 0}, args.gain, args.eps); err == nil || emitted || pool.nextPos != before.nextPos || pool.filled != before.filled || calls != 0 {
			t.Fatal("invalid gain/epsilon advanced an incomplete group or invoked callback")
		}
	}
	gain := []float32{1, -1}
	if row, emitted, err := pool.PushNormalized(0, []float32{1, 3}, []float32{0, 0}, gain, 1e-5); err != nil || emitted || row != nil || calls != 0 {
		t.Fatal("incomplete group invoked normalization")
	}
	row, emitted, err := pool.PushNormalized(1, []float32{3, 1}, []float32{0, 0}, gain, 1e-5)
	if err != nil || !emitted || calls != 1 || !reflect.DeepEqual(row, v41CompressorNormRefTail([]float32{2, 2}, gain, 1e-5, "")) {
		t.Fatal("complete group did not normalize exactly once")
	}
	// The exported standalone ratio-one pool enters its normalization tail.
	// Only ratio zero bypasses the production helper's compressor operations.
	one, err := NewV41CompressorPool(1, 2)
	if err != nil {
		t.Fatal(err)
	}
	one.normalize = pool.normalize
	if _, emitted, err := one.PushNormalized(0, []float32{1, 2}, nil, gain, 1e-5); err != nil || !emitted || calls != 2 {
		t.Fatal("standalone ratio-one tail semantics changed")
	}
	m := v41CompressorNormTinyModel(t, gain, 1e-5)
	input := [][]float32{{1, 2}}
	got, err := m.v41CompressedRowsWithOperations(0, 0, input, input, nil, func(int, []float32, []float32, float32) ([]float32, error) {
		t.Fatal("production bypass invoked compressor normalization")
		return nil, nil
	})
	if err != nil || !reflect.DeepEqual(got, input) {
		t.Fatal("production ratio bypass changed")
	}
}
