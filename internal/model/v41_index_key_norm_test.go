package model

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const v41IndexKeyNormTestLeaf = "indexer.k_norm.weight"

// Reuse the existing recording allocator, readback poisoning and release checks.
// Only the leaf being observed changes; this CPU recorder is no hardware witness.
type v41IndexKeyNormRecorder struct{ *v41CompressorNormRecorder }

func (b *v41IndexKeyNormRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	out := b.v41CompressorNormRecorder.UploadClass(x, dt, class, site)
	if strings.HasPrefix(site, "V4.1 RMSNorm activation ") && strings.HasSuffix(site, v41IndexKeyNormTestLeaf) {
		b.uploadBytes += b.uploads[out.Buf()]
	}
	return out
}

func (b *v41IndexKeyNormRecorder) RMSNorm(x, weight compute.Tensor, eps float32) compute.Tensor {
	b.io++
	layer := -1
	if b.owner != nil {
		for l := 0; l < b.owner.M.Cfg.NumLayers; l++ {
			if gain, ok := b.owner.halW[layerName(l, v41IndexKeyNormTestLeaf)]; ok && gain.Buf() == weight.Buf() {
				layer = l
				break
			}
		}
	}
	if layer >= 0 {
		b.records = append(b.records, v41CompressorNormRecord{layer: layer,
			input: append([]float32(nil), b.Backend.Read(x)...), gain: append([]float32(nil), b.Backend.Read(weight)...), eps: eps, weight: weight.Buf()})
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
	}
	return y
}

type v41IndexKeyNormQualifiedRecorder struct {
	*v41IndexKeyNormRecorder
	width   int
	epsBits uint32
	allow   bool
}

func (b *v41IndexKeyNormQualifiedRecorder) SupportsV41IndexKeyNorm(width int, eps float32) bool {
	return b.allow && width == b.width && math.Float32bits(eps) == b.epsBits
}

func v41IndexKeyNormTestSession(t *testing.T, m *Model) (*Session, *v41IndexKeyNormQualifiedRecorder) {
	t.Helper()
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("index normalization controls require cpu-ref")
	}
	b := &v41IndexKeyNormQualifiedRecorder{v41IndexKeyNormRecorder: &v41IndexKeyNormRecorder{newV41CompressorNormRecorder()},
		width: m.Cfg.IndexHeadDim, epsBits: math.Float32bits(float32(m.Cfg.RMSNormEps)), allow: true}
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	})
	var s *Session
	if m.Cfg.DeepSeekV41 == nil {
		s = &Session{M: m, Backend: b, halW: map[string]compute.Tensor{}}
		t.Cleanup(s.Close)
	} else {
		s = v41DenseTestSession(t, m, b)
	}
	b.owner = s
	return s, b
}

func v41IndexKeyNormTinyModel(t *testing.T, gain []float32, eps float32) *Model {
	t.Helper()
	name := layerName(0, v41IndexKeyNormTestLeaf)
	manifest, raw := synthBuildRaw([]synthTensor{{name, []int{len(gain)}}}, synthMatmulFill)
	m := &Model{Cfg: Config{NumLayers: 1, IndexHeadDim: len(gain), HeadDim: 512, QLoraRank: 8, RMSNormEps: float64(eps)}, manifest: manifest, raw: raw}
	v41WriteTensorF32(t, m, name, gain)
	return m
}

// Deliberately independent of rmsnormCfg and the production BF16 helper.
func v41IndexKeyNormReference(input, gain []float32, eps float32) []float32 {
	var squareSum float32
	for _, value := range input {
		squareSum += value * value
	}
	inverse := float32(1 / math.Sqrt(float64(squareSum/float32(len(input))+eps)))
	out := make([]float32, len(input))
	for i, value := range input {
		out[i] = value * inverse * gain[i]
	}
	return out
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=100ms lane=default
func TestV41IndexKeyNormF32Operands(t *testing.T) {
	t.Parallel()
	input, gain, eps := []float32{1.003, 0.503, -0.126, 2.011}, []float32{0.7, -0.9, 1.3, 1.9}, float32(1e-5)
	s, b := v41IndexKeyNormTestSession(t, v41IndexKeyNormTinyModel(t, gain, eps))
	normalize := s.v41IndexKeyNormFunc()
	if normalize == nil {
		t.Fatal("index-specific fixture qualification did not select actual IndexHeadDim")
	}
	inputBefore, gainBefore := append([]float32(nil), input...), append([]float32(nil), gain...)
	got, err := normalize(0, input, gain, eps)
	want := v41IndexKeyNormReference(inputBefore, gainBefore, eps)
	if err != nil || !v41CompressorNormEqualBits(got, want) {
		t.Fatalf("F32 index RMSNorm mismatch: got=%v want=%v err=%v", got, want, err)
	}
	castInput, castOutput := append([]float32(nil), input...), append([]float32(nil), want...)
	for i := range input {
		castInput[i] = v41CompressorNormRefCast(input[i])
		castOutput[i] = v41CompressorNormRefCast(want[i])
	}
	if v41CompressorNormEqualBits(want, v41IndexKeyNormReference(castInput, gain, eps)) || v41CompressorNormEqualBits(want, castOutput) {
		t.Fatal("control cannot detect an accidental BF16 input or output boundary")
	}
	if !v41CompressorNormEqualBits(input, inputBefore) || !v41CompressorNormEqualBits(gain, gainBefore) {
		t.Fatal("normalization mutated caller operands")
	}
	if len(b.records) != 1 || b.readbacks != 1 || b.uploadBytes != 4*len(input) || b.readBytes != b.uploadBytes || len(b.outputs) != 0 {
		t.Fatal("row did not produce one released, owned F32 readback")
	}
	r := b.records[0]
	if !v41CompressorNormEqualBits(r.input, input) || !v41CompressorNormEqualBits(r.gain, gain) || math.Float32bits(r.eps) != math.Float32bits(eps) {
		t.Fatal("backend received substituted index operands")
	}
	got[0] = 99
	again, err := normalize(0, input, gain, eps)
	if err != nil || !v41CompressorNormEqualBits(again, want) || b.records[1].weight != r.weight {
		t.Fatal("result alias or immutable learned-gain cache changed")
	}
	s.Close()
	if b.frees[r.weight] != 0 {
		t.Fatal("session freed model-owned index gain")
	}
	if err := s.M.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	if b.frees[r.weight] != 1 || len(b.live) != 0 {
		t.Fatal("model did not free index gain exactly once")
	}
}

func v41IndexKeyNormAssertFailure(t *testing.T, s *Session, err error, layer int, phase string) *BackendForwardOperationError {
	t.Helper()
	var selected *V41ProjectionOperationError
	var closed *BackendForwardOperationError
	if !errors.As(err, &selected) || selected.Leaf != v41IndexKeyNormTestLeaf || selected.Layer != layer || selected.Stage != string(v41StageIndexer) ||
		!errors.As(err, &closed) || closed.Path != "v41-index-key-norm" || closed.Layer != layer || closed.Stage != phase || closed.Forward != ForwardPathKind("deepseek41") ||
		!s.BackendSessionClosed() || s.halFailure != closed {
		t.Fatalf("selected index normalization lost attribution/close/latch: %v", err)
	}
	return closed
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=200ms lane=default
func TestV41IndexKeyNormAdmissionAndCanonicalOperands(t *testing.T) {
	t.Parallel()
	m := v41IndexKeyNormTinyModel(t, []float32{1, 1, 1, 1}, 1e-5)
	ordinary := &v41IndexKeyNormRecorder{newV41CompressorNormRecorder()}
	ordinary.label = "vulkan"
	generic := &Session{M: m, Backend: ordinary}
	if generic.v41RMSNormFunc() == nil || generic.v41IndexKeyNormFunc() != nil || ordinary.io != 0 {
		t.Fatal("ordinary Vulkan-like RMSNorm gained index qualification")
	}
	for _, reject := range []string{"capability", "width", "epsilon", "gain-signed-zero", "epsilon-bits", "payload-width"} {
		t.Run(reject, func(t *testing.T) {
			gain, input, eps := []float32{0, -1, 0.5, 2}, []float32{1, 2, 3, 4}, float32(1e-5)
			s, b := v41IndexKeyNormTestSession(t, v41IndexKeyNormTinyModel(t, gain, eps))
			switch reject {
			case "capability":
				b.allow = false
			case "width":
				b.width++
			case "epsilon":
				b.epsBits++
			}
			normalize := s.v41IndexKeyNormFunc()
			if reject == "capability" || reject == "width" || reject == "epsilon" {
				if normalize != nil || b.io != 0 {
					t.Fatal("unqualified index norm selected or performed I/O")
				}
				return
			}
			if normalize == nil {
				t.Fatal("fixture callback absent")
			}
			phase := "canonical operands"
			switch reject {
			case "gain-signed-zero":
				gain[0] = math.Float32frombits(0x80000000)
			case "epsilon-bits":
				eps = math.Float32frombits(math.Float32bits(eps) + 1)
			case "payload-width":
				input = input[:3]
				phase = "payload"
			}
			got, err := normalize(0, input, gain, eps)
			closed := v41IndexKeyNormAssertFailure(t, s, err, 0, phase)
			if got != nil || b.io != 0 {
				t.Fatal("selected invalid operands reached device work")
			}
			if recovered := v41CompressorTestRecover(func() { _, _ = normalize(0, input, gain, eps) }); recovered != closed || b.io != 0 {
				t.Fatal("closed operand failure retried or changed latch")
			}
		})
	}
}

// Same projection backend on both sides isolates normalization from known
// scalar-vs-HAL upstream reduction differences. Exact cache and top-k assertions
// stay in the existing publication helper; no tolerance is added to index keys.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=3s lane=default
func TestV41IndexKeyNormSourceReaderLifecycle(t *testing.T) {
	t.Parallel()
	for _, mode := range []string{"ratio-zero-reader", "ratio-two-reader", "own-index-source"} {
		t.Run(mode, func(t *testing.T) {
			fixture := func() *Model {
				m := v41IndexerTestFixture(t)
				if mode == "ratio-zero-reader" {
					m.Cfg.DeepSeekV41.CompressRatios[1] = 0
				}
				if mode == "own-index-source" {
					m.Cfg.DeepSeekV41.IndexSourceLayerIDs = []int{0, 1}
				}
				return m
			}
			m, reference := fixture(), fixture()
			s, b := v41IndexKeyNormTestSession(t, m)
			t.Cleanup(func() {
				if err := reference.CloseWeights(); err != nil {
					t.Error(err)
				}
			})
			host := v41DenseTestSession(t, reference, newV41DenseTestBackend())
			if s.v41State().indexKeyNorm == nil || host.v41State().indexKeyNorm != nil {
				t.Fatal("selected/control index callbacks not isolated")
			}
			producers := 1
			check := func(tokens int) {
				t.Helper()
				calls := producers * (tokens / 2)
				if len(b.records) != calls || b.readbacks != calls || b.uploadBytes != calls*4*m.Cfg.IndexHeadDim || b.readBytes != b.uploadBytes || len(b.outputs) != 0 {
					t.Fatalf("only complete own groups may normalize: calls=%d want=%d", len(b.records), calls)
				}
				counts := [2]int{}
				for _, record := range b.records {
					counts[record.layer]++
					if len(record.input) != m.Cfg.IndexHeadDim || !v41CompressorNormEqualBits(record.gain, m.tensor(layerName(record.layer, v41IndexKeyNormTestLeaf))) || math.Float32bits(record.eps) != math.Float32bits(float32(m.Cfg.RMSNormEps)) {
						t.Fatal("session lost actual index width or learned operands")
					}
				}
				if counts[0] != tokens/2 || counts[1] != (producers-1)*(tokens/2) {
					t.Fatal("reader renormalized borrowed keys or skipped own source keys")
				}
				v41CompressorNormAssertPublications(t, s.v41Forward, host.v41Forward)
				if keys, ok := s.v41Forward.attn.IndexKeys(1); ok || len(keys) != 0 {
					t.Fatal("index-only reader published private normalized keys")
				}
				for layer := 0; layer < producers; layer++ {
					keys, ok := s.v41Forward.attn.IndexKeys(layer)
					want, wantOK := host.v41Forward.attn.IndexKeys(layer)
					if !ok || !wantOK || len(keys) != tokens/2 || !reflect.DeepEqual(keys, want) {
						t.Fatal("own source index publication changed")
					}
				}
			}
			v41CompressorTestFiniteParity(t, s.Prefill([]int{1, 2, 3}), host.Prefill([]int{1, 2, 3}), "index normalization cold prefix")
			check(3)
			v41CompressorTestFiniteParity(t, s.Step(4), host.Step(4), "index normalization decode")
			check(4)
			v41CompressorTestFiniteParity(t, s.Prefill([]int{5, 6, 7}), host.Prefill([]int{5, 6, 7}), "index normalization suffix")
			check(7)
			before := captureV41ForwardSnapshot(s.v41Forward)
			dataOnly := before.clone().restore()
			if dataOnly.indexKeyNorm != nil || dataOnly.callbackOwner != nil || !reflect.DeepEqual(captureV41ForwardSnapshot(dataOnly), before) {
				t.Fatal("snapshot retained callback ownership or changed continuation")
			}
			snap, err := s.PrefixSnapshot()
			if err != nil {
				t.Fatal(err)
			}
			defer snap.Close()
			target := v41DenseTestSession(t, m, b)
			if err := snap.Restore(target); err != nil {
				t.Fatal(err)
			}
			if target.v41Forward.callbackOwner != target || target.v41Forward.indexKeyNorm == nil {
				t.Fatal("restore did not bind target's index callback")
			}
			cached := map[int]compute.Buffer{}
			for _, record := range b.records {
				cached[record.layer] = record.weight
			}
			s.Close()
			b.owner, s = target, target
			v41CompressorTestFiniteParity(t, s.Step(8), host.Step(8), "index normalization restored owner")
			check(8)
			for _, record := range b.records {
				if record.weight != cached[record.layer] || b.frees[record.weight] != 0 {
					t.Fatal("restore replaced or freed immutable index gain")
				}
			}
			// An incomplete next group still runs the real step while retaining no callback in scratch.
			scratch := &v41ProjScratch{}
			if _, stats, err := m.forwardV41Step(9, s.v41State(), scratch); err != nil || !stats.Committed || scratch.indexKeyNorm != nil || len(b.records) != producers*4 {
				t.Fatal("step retained callback or normalized an incomplete group")
			}
		})
	}
}

// A late own-source norm failure follows layer zero's newly published index row.
// The enclosing Session must roll it back without replaying projections on host.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=2s lane=default
func TestV41IndexKeyNormSelectedRollback(t *testing.T) {
	t.Parallel()
	for _, site := range []string{"rmsnorm", "short readback", "unknown panic"} {
		t.Run(site, func(t *testing.T) {
			m := v41IndexerTestFixture(t)
			m.Cfg.DeepSeekV41.IndexSourceLayerIDs = []int{0, 1}
			m.Cfg.DeepSeekV41.KVSourceLayerIDs = []int{0, 1}
			s, b := v41IndexKeyNormTestSession(t, m)
			s.Prefill([]int{1, 2, 3})
			before, norms := captureV41ForwardSnapshot(s.v41Forward), len(b.records)
			phaseBefore := v41IndexerTestPhase(t, m, "decode")
			b.faultLayer, b.faultSite = 1, site
			cause := &compute.BackendError{Backend: "index-norm-recorder", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
			b.fault = cause
			if site == "unknown panic" {
				b.faultSite = "rmsnorm"
				b.fault = &struct{ Marker string }{"index-norm-panic"}
			}
			b.beforeFault = func() {
				keys, ok := s.v41Forward.attn.IndexKeys(0)
				if !ok || len(keys) != 2 {
					t.Fatal("late fault lacked earlier index publication")
				}
			}
			value := v41CompressorTestRecover(func() { s.Step(4) })
			closed := s.halFailure
			if site == "unknown panic" {
				if value != b.fault || !s.BackendSessionClosed() || closed == nil {
					t.Fatal("unknown panic lost identity/close/latch")
				}
			} else {
				err, ok := value.(error)
				if !ok {
					t.Fatalf("selected fault did not return an error: %v", value)
				}
				phase := site
				if site == "short readback" {
					phase = "readback"
				}
				closed = v41IndexKeyNormAssertFailure(t, s, err, 1, phase)
				if site == "rmsnorm" && !errors.Is(err, cause) {
					t.Fatal("selected fault lost backend cause")
				}
			}
			if len(b.records) != norms+2 || b.records[norms].layer != 0 || b.records[norms+1].layer != 1 || !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
				t.Fatal("late failure replayed a group or committed continuation/publications/top-k")
			}
			v41CompressorNormAssertNoTransient(t, b.v41CompressorNormRecorder)
			delta := v41DenseTestDelta(v41IndexerTestPhase(t, m, "decode"), phaseBefore)
			if delta["indexer_projection_host_calls"] != 0 || delta["indexer_projection_host_rows"] != 0 || delta["indexer_projection_device_calls"] <= 0 {
				t.Fatalf("selected failure fell into historical host replay: %v", delta)
			}
			io, operations := b.io, len(b.ops)
			for _, entry := range []func(){func() { s.Step(4) }, func() { s.Prefill([]int{4, 5}) }} {
				if recovered := v41CompressorTestRecover(entry); recovered != closed || b.io != io || len(b.ops) != operations || !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
					t.Fatal("closed Session reused backend or changed its latch/continuation")
				}
			}
		})
	}
}
