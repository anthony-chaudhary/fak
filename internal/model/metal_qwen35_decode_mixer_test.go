//go:build darwin && arm64 && cgo

package model

import (
	"errors"
	"math"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

type recordingQwen35DecodeMixerBackend struct {
	*metalQwen35GDNSequenceBackend
	mu       sync.Mutex
	calls    int
	accepted int
	receipts []qwen35DecodeMixerReceipt
}

func (b *recordingQwen35DecodeMixerBackend) Qwen35MetalDecodeMixer(s *Session, layer int, xn []float32) ([]float32, qwen35DecodeMixerReceipt, bool, error) {
	out, receipt, accepted, err := b.metalQwen35GDNSequenceBackend.Qwen35MetalDecodeMixer(s, layer, xn)
	b.mu.Lock()
	b.calls++
	if accepted {
		b.accepted++
		b.receipts = append(b.receipts, receipt)
	}
	b.mu.Unlock()
	return out, receipt, accepted, err
}

func (b *recordingQwen35DecodeMixerBackend) decodeSnapshot() (calls, accepted int, receipts []qwen35DecodeMixerReceipt) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.calls, b.accepted, append([]qwen35DecodeMixerReceipt(nil), b.receipts...)
}

type qwen35DecodeMixerFixture struct {
	m       *Model
	weights metalgemm.Qwen35DecodeWeights
	handles []*metalgemm.Q8Weight
	panel   metalgemm.GDNPanel
	geom    metalgemm.GDNGeometry
}

func newQwen35DecodeMixerFixture(t *testing.T) qwen35DecodeMixerFixture {
	t.Helper()
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	p := func(suffix string) string { return layerName(0, suffix) }
	names := []string{
		p("linear_attn.in_proj_qkv.weight"), p("linear_attn.in_proj_z.weight"),
		p("linear_attn.in_proj_b.weight"), p("linear_attn.in_proj_a.weight"),
		p("linear_attn.out_proj.weight"),
	}
	handles := make([]*metalgemm.Q8Weight, len(names))
	for i, name := range names {
		qt := m.q8(name)
		handles[i] = metalgemm.UploadQ8(qt.q, qt.d, qt.out, qt.in)
		if handles[i] == nil {
			for _, handle := range handles[:i] {
				handle.Release()
			}
			t.Fatalf("UploadQ8(%s) declined", name)
		}
	}
	return qwen35DecodeMixerFixture{
		m: m,
		weights: metalgemm.Qwen35DecodeWeights{
			InQKV: handles[0], InZ: handles[1], InB: handles[2], InA: handles[3], Out: handles[4],
		},
		handles: handles,
		panel: metalgemm.GDNPanel{
			Tokens: 1, Conv1D: m.tensor(p("linear_attn.conv1d.weight")), ALog: m.tensor(p("linear_attn.A_log")),
			DTBias: m.tensor(p("linear_attn.dt_bias")), Norm: m.tensor(p("linear_attn.norm.weight")),
			RMSNormEpsilon: float32(cfg.RMSNormEps),
		},
		geom: metalgemm.GDNGeometry{
			NumKeyHeads: cfg.LinearNumKeyHeads, NumValueHeads: cfg.LinearNumValueHeads,
			KeyHeadDim: cfg.LinearKeyHeadDim, ValueHeadDim: cfg.LinearValueHeadDim, ConvKernel: cfg.LinearConvKernelDim,
		},
	}
}

func (f qwen35DecodeMixerFixture) close() {
	for i := len(f.handles) - 1; i >= 0; i-- {
		f.handles[i].Release()
	}
}

func TestQwen35MetalDecodeMixerFourSeededStepsParityAndAccounting(t *testing.T) {
	fixture := newQwen35DecodeMixerFixture(t)
	defer fixture.close()
	baseline := metalgemm.GDNLiveBufferCount()
	state, err := metalgemm.NewGDNState(fixture.geom)
	if err != nil {
		t.Fatal(err)
	}
	defer state.Close()

	cfg := fixture.m.Cfg
	_, nV, kHd, vHd, _, _, convDim := cfg.linearAttnDims()
	convSeed := randomVecF((cfg.LinearConvKernelDim-1)*convDim, 9486)
	recurrentSeed := randomVecF(nV*kHd*vHd, 9487)
	if err := state.Seed(convSeed, recurrentSeed); err != nil {
		t.Fatal(err)
	}
	cpu := fixture.m.NewSession()
	cpu.Cache.linear = newLinearAttnCache(cfg)
	cpuLayer := cpu.Cache.linear.layer(cfg, 0)
	cpuLayer.conv = make([][]float32, cfg.LinearConvKernelDim-1)
	for row := range cpuLayer.conv {
		cpuLayer.conv[row] = make([]float32, convDim)
		copy(cpuLayer.conv[row], convSeed[row*convDim:(row+1)*convDim])
	}
	for head := range cpuLayer.recurrent {
		copy(cpuLayer.recurrent[head], recurrentSeed[head*kHd*vHd:(head+1)*kHd*vHd])
	}

	for step := 0; step < 4; step++ {
		input := randomVecF(cfg.HiddenSize, int64(9500+step))
		want := cpu.linearAttnStep(0, input, q8Kernel{fixture.m})
		got, receipt, accepted, runErr := metalgemm.RunQwen35Decode(metalgemm.Qwen35DecodeRequest{
			Input: input, Weights: fixture.weights, State: state, Panel: fixture.panel,
		})
		if runErr != nil || !accepted {
			t.Fatalf("step %d accepted=%v err=%v", step, accepted, runErr)
		}
		assertQwen35DecodeMixerReceipt(t, receipt)
		assertQwen35DecodeMixerParity(t, "output", want, got)
		if decodeMixerArgmax(want) != decodeMixerArgmax(got) {
			t.Fatalf("step %d greedy token=%d want %d", step, decodeMixerArgmax(got), decodeMixerArgmax(want))
		}
		conv, recurrent, snapshotErr := state.Snapshot()
		if snapshotErr != nil {
			t.Fatal(snapshotErr)
		}
		assertQwen35DecodeMixerParity(t, "convolution state", flattenQwen35Conv(cpuLayer), conv)
		assertQwen35DecodeMixerParity(t, "recurrent state", flattenQwen35Recurrent(cpuLayer), recurrent)
	}
	state.Close()
	state.Close()
	if got := metalgemm.GDNLiveBufferCount(); got != baseline {
		t.Fatalf("exact-once Close left live buffers=%d, want %d", got, baseline)
	}
}

func TestQwen35MetalDecodeMixerIsolationDeclineAndPostSubmitFailure(t *testing.T) {
	fixture := newQwen35DecodeMixerFixture(t)
	defer fixture.close()
	first, err := metalgemm.NewGDNState(fixture.geom)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	second, err := metalgemm.NewGDNState(fixture.geom)
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	input := randomVecF(fixture.m.Cfg.HiddenSize, 9600)
	if _, _, accepted, runErr := metalgemm.RunQwen35Decode(metalgemm.Qwen35DecodeRequest{Input: input, Weights: fixture.weights, State: first, Panel: fixture.panel}); runErr != nil || !accepted {
		t.Fatalf("first operation accepted=%v err=%v", accepted, runErr)
	}
	firstConv, firstRecurrent, err := first.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if _, _, accepted, runErr := metalgemm.RunQwen35Decode(metalgemm.Qwen35DecodeRequest{Input: randomVecF(len(input), 9601), Weights: fixture.weights, State: second, Panel: fixture.panel}); runErr != nil || !accepted {
		t.Fatalf("isolated operation accepted=%v err=%v", accepted, runErr)
	}
	afterConv, afterRecurrent, err := first.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	assertQwen35DecodeMixerParity(t, "isolated convolution", firstConv, afterConv)
	assertQwen35DecodeMixerParity(t, "isolated recurrent", firstRecurrent, afterRecurrent)

	missing := fixture.weights
	missing.Out = nil
	beforeConv, beforeRecurrent, _ := second.Snapshot()
	if _, receipt, accepted, runErr := metalgemm.RunQwen35Decode(metalgemm.Qwen35DecodeRequest{Input: input, Weights: missing, State: second, Panel: fixture.panel}); accepted || runErr == nil || receipt.Committed {
		t.Fatalf("missing handle accepted=%v receipt=%+v err=%v", accepted, receipt, runErr)
	}
	declinedConv, declinedRecurrent, _ := second.Snapshot()
	assertQwen35DecodeMixerParity(t, "declined convolution", beforeConv, declinedConv)
	assertQwen35DecodeMixerParity(t, "declined recurrent", beforeRecurrent, declinedRecurrent)

	failing, err := metalgemm.NewGDNState(fixture.geom)
	if err != nil {
		t.Fatal(err)
	}
	_, receipt, accepted, runErr := metalgemm.RunQwen35Decode(metalgemm.Qwen35DecodeRequest{
		Input: input, Weights: fixture.weights, State: failing, Panel: fixture.panel, InjectPostSubmitFailureForTest: true,
	})
	var post *metalgemm.GraphPostSubmitError
	if !accepted || !errors.As(runErr, &post) || !receipt.Committed || !receipt.CompletedWait {
		t.Fatalf("post-submit accepted=%v receipt=%+v err=%v", accepted, receipt, runErr)
	}
	failing.Close()
	failing.Close()
}

func TestExactQwen38SessionSelectsWholeDecodeMixer(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	cfg := qwen35HybridTestCfg()
	cfg.ModelType = "qwen3_5_text"
	cfg.NumLayers = 64
	cfg.LayerTypes = make([]string, cfg.NumLayers)
	for layer := range cfg.LayerTypes {
		if layer%4 == 3 {
			cfg.LayerTypes[layer] = "full_attention"
		} else {
			cfg.LayerTypes[layer] = "linear_attention"
		}
	}
	m := NewSynthetic(cfg)
	m.Quantize()
	// tokenHiddenQ selects sessionQ4KKernel only when the raw-Q4_K store exists.
	// The mixer weights are the exact runtime-Q8 band; unrelated synthetic
	// projections deliberately retain sessionQ4KKernel's Q8 fallback.
	m.q4kw = make(map[string]*q4kTensor)
	s := m.NewSession()
	s.Q4K, s.MetalQ4K = true, true
	var handles []*metalgemm.Q8Weight
	defer func() {
		s.Close()
		m.releaseMetalQ8Residency()
		for i := len(handles) - 1; i >= 0; i-- {
			handles[i].Release()
		}
	}()
	backend := &recordingQwen35DecodeMixerBackend{metalQwen35GDNSequenceBackend: &metalQwen35GDNSequenceBackend{states: make(map[Qwen35GDNAuxState]*metalgemm.GDNState)}}
	accepted, err := s.initQwen35GDNPreprojectedSequence(backend)
	if err != nil || !accepted {
		t.Fatalf("initialize exact Qwen3.8 owners accepted=%v err=%v", accepted, err)
	}
	_, nV, kHd, vHd, _, _, convDim := cfg.linearAttnDims()
	snapshots := make([]qwen35GDNLayerSnapshot, 0, 48)
	for layer, state := range s.qwen35HAL.sequenceLayers {
		if state.valid() {
			snapshots = append(snapshots, qwen35GDNLayerSnapshot{
				layer: layer, conv: make([]float32, (cfg.LinearConvKernelDim-1)*convDim), recurrent: make([]float32, nV*kHd*vHd),
			})
		}
	}
	if selected, promoteErr := s.promoteQwen35MetalGDNDecode(snapshots); promoteErr != nil || !selected {
		t.Fatalf("promote exact Qwen3.8 decode selected=%v err=%v", selected, promoteErr)
	}
	p := func(suffix string) string { return layerName(0, suffix) }
	names := []string{
		p("linear_attn.in_proj_qkv.weight"), p("linear_attn.in_proj_z.weight"),
		p("linear_attn.in_proj_b.weight"), p("linear_attn.in_proj_a.weight"), p("linear_attn.out_proj.weight"),
	}
	handles = make([]*metalgemm.Q8Weight, len(names))
	table := make(map[string]*metalgemm.Q8Weight, len(names))
	for i, name := range names {
		qt := m.q8(name)
		handles[i] = metalgemm.UploadQ8(qt.q, qt.d, qt.out, qt.in)
		if handles[i] == nil {
			t.Fatalf("UploadQ8(%s) declined", name)
		}
		table[name] = handles[i]
	}
	metalQ4KMu.Lock()
	metalQ8KW[m] = table
	metalQ4KMu.Unlock()
	out, receipt, used, err := s.tryQwen35MetalDecodeMixer(0, randomVecF(cfg.HiddenSize, 9700))
	if err != nil || !used || len(out) != cfg.HiddenSize {
		t.Fatalf("exact Qwen3.8 decode mixer used=%v output=%d err=%v", used, len(out), err)
	}
	if receipt.CommandBuffers != 1 || receipt.ProjectionDispatches != 5 || receipt.Quantizers != 2 ||
		receipt.GDNEncoders != 1 || receipt.InputUploads != 1 || receipt.FinalReadbacks != 1 ||
		receipt.IntermediateReadbacks != 0 || receipt.StateH2DTransfers != 0 || receipt.StateD2HTransfers != 0 ||
		!receipt.Committed || !receipt.CompletedWait {
		t.Fatalf("exact Qwen3.8 model receipt=%+v", receipt)
	}
	baselineCalls, baselineAccepted, baselineReceipts := backend.decodeSnapshot()
	for _, state := range backend.states {
		if resetErr := state.Reset(); resetErr != nil {
			t.Fatal(resetErr)
		}
	}
	if promoteErr := m.promoteMetalQ8Residency(); promoteErr != nil {
		t.Fatalf("promote exact Qwen3.8 Q8 residency: %v", promoteErr)
	}
	reference := m.NewSession()
	reference.Q4K = true
	defer reference.Close()
	wantHidden := reference.tokenHiddenQ(3, 0)
	gotHidden := s.tokenHiddenQ(3, 0)
	afterCalls, afterAccepted, afterReceipts := backend.decodeSnapshot()
	if got, want := afterCalls-baselineCalls, 48; got != want {
		t.Fatalf("exact Qwen3.8 product path mixer calls=%d, want %d", got, want)
	}
	if got, want := afterAccepted-baselineAccepted, 48; got != want {
		t.Fatalf("exact Qwen3.8 product path accepted mixer calls=%d, want %d", got, want)
	}
	if got, want := len(afterReceipts)-len(baselineReceipts), 48; got != want {
		t.Fatalf("exact Qwen3.8 product path receipts=%d, want %d", got, want)
	}
	for i, routeReceipt := range afterReceipts[len(baselineReceipts):] {
		assertQwen35ModelDecodeMixerReceipt(t, i, routeReceipt)
	}
	assertQwen35DecodeMixerParity(t, "exact Qwen3.8 full-token hidden", wantHidden, gotHidden)
	head := m.q8(m.headName())
	wantLogits := qMatRows(head, quantizeVecQ8(wantHidden))
	gotLogits := qMatRows(head, quantizeVecQ8(gotHidden))
	if decodeMixerArgmax(wantLogits) != decodeMixerArgmax(gotLogits) {
		t.Fatalf("exact Qwen3.8 greedy token=%d want %d", decodeMixerArgmax(gotLogits), decodeMixerArgmax(wantLogits))
	}
}

func assertQwen35ModelDecodeMixerReceipt(t *testing.T, layer int, receipt qwen35DecodeMixerReceipt) {
	t.Helper()
	if receipt.CommandBuffers != 1 || receipt.Commits != 1 || receipt.CompletionWaits != 1 ||
		receipt.ProjectionDispatches != 5 || receipt.Quantizers != 2 || receipt.GDNEncoders != 1 || receipt.Encoders != 8 ||
		receipt.InputUploads != 1 || receipt.FinalReadbacks != 1 || receipt.IntermediateReadbacks != 0 ||
		receipt.StateH2DTransfers != 0 || receipt.StateD2HTransfers != 0 || !receipt.Committed || !receipt.CompletedWait {
		t.Fatalf("exact Qwen3.8 product path layer %d receipt=%+v", layer, receipt)
	}
}

func assertQwen35DecodeMixerReceipt(t *testing.T, receipt metalgemm.Qwen35DecodeReceipt) {
	t.Helper()
	if receipt.CommandBuffers != 1 || receipt.Commits != 1 || receipt.CompletionWaits != 1 ||
		receipt.ProjectionDispatches != 5 || receipt.Quantizers != 2 || receipt.GDNEncoders != 1 || receipt.Encoders != 8 ||
		receipt.InputUploads != 1 || receipt.FinalReadbacks != 1 || receipt.IntermediateReadbacks != 0 ||
		receipt.StateH2DTransfers != 0 || receipt.StateD2HTransfers != 0 || !receipt.Committed || !receipt.CompletedWait {
		t.Fatalf("decode mixer receipt=%+v", receipt)
	}
}

func assertQwen35DecodeMixerParity(t *testing.T, label string, want, got []float32) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s elements=%d want %d", label, len(got), len(want))
	}
	var dot, wn, gn, maxAbs float64
	for i := range want {
		w, g := float64(want[i]), float64(got[i])
		dot += w * g
		wn += w * w
		gn += g * g
		if d := math.Abs(w - g); d > maxAbs {
			maxAbs = d
		}
	}
	cosine := dot / math.Sqrt(wn*gn)
	if cosine < 0.999999 || maxAbs >= 1e-4 {
		t.Fatalf("%s cosine=%g maxAbs=%g", label, cosine, maxAbs)
	}
}

func flattenQwen35Conv(state *linearAttnLayerState) []float32 {
	var out []float32
	for _, row := range state.conv {
		out = append(out, row...)
	}
	return out
}

func flattenQwen35Recurrent(state *linearAttnLayerState) []float32 {
	var out []float32
	for _, head := range state.recurrent {
		out = append(out, head...)
	}
	return out
}

func decodeMixerArgmax(values []float32) int {
	best := 0
	for i := 1; i < len(values); i++ {
		if values[i] > values[best] {
			best = i
		}
	}
	return best
}
