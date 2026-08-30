//go:build darwin && arm64 && cgo

package model

import (
	"errors"
	"math"
	"path/filepath"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

type recordingQwen35DecodeBlockBackend struct {
	*metalQwen35GDNSequenceBackend
	mu       sync.Mutex
	calls    int
	mixers   int
	receipts []qwen35DecodeBlockReceipt
}

func (b *recordingQwen35DecodeBlockBackend) Qwen35MetalDecodeMixer(s *Session, layer int, x []float32) ([]float32, qwen35DecodeMixerReceipt, bool, error) {
	b.mu.Lock()
	b.mixers++
	b.mu.Unlock()
	return b.metalQwen35GDNSequenceBackend.Qwen35MetalDecodeMixer(s, layer, x)
}

func (b *recordingQwen35DecodeBlockBackend) Qwen35MetalDecodeBlock(s *Session, layer int, x []float32) ([]float32, qwen35DecodeBlockReceipt, bool, error) {
	out, receipt, accepted, err := b.metalQwen35GDNSequenceBackend.Qwen35MetalDecodeBlock(s, layer, x)
	b.mu.Lock()
	b.calls++
	if accepted {
		b.receipts = append(b.receipts, receipt)
	}
	b.mu.Unlock()
	return out, receipt, accepted, err
}

func (b *recordingQwen35DecodeBlockBackend) snapshot() (int, []qwen35DecodeBlockReceipt) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.calls, append([]qwen35DecodeBlockReceipt(nil), b.receipts...)
}

func (b *recordingQwen35DecodeBlockBackend) mixerCalls() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.mixers
}

type failingQwen35DecodeBlockBackend struct {
	*metalQwen35GDNSequenceBackend
	mu                     sync.Mutex
	blockCalls, mixerCalls int
	receipt                qwen35DecodeBlockReceipt
}

func (b *failingQwen35DecodeBlockBackend) Qwen35MetalDecodeBlock(s *Session, layer int, x []float32) ([]float32, qwen35DecodeBlockReceipt, bool, error) {
	out, receipt, accepted, err := b.metalQwen35GDNSequenceBackend.qwen35MetalDecodeBlock(s, layer, x, true)
	b.mu.Lock()
	b.blockCalls++
	b.receipt = receipt
	b.mu.Unlock()
	return out, receipt, accepted, err
}

func (b *failingQwen35DecodeBlockBackend) Qwen35MetalDecodeMixer(s *Session, layer int, x []float32) ([]float32, qwen35DecodeMixerReceipt, bool, error) {
	b.mu.Lock()
	b.mixerCalls++
	b.mu.Unlock()
	return b.metalQwen35GDNSequenceBackend.Qwen35MetalDecodeMixer(s, layer, x)
}

func (b *failingQwen35DecodeBlockBackend) snapshot() (int, int, qwen35DecodeBlockReceipt) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.blockCalls, b.mixerCalls, b.receipt
}

type qwen35DecodeBlockFixture struct {
	m       *Model
	weights metalgemm.Qwen35DecodeWeights
	q8      []*metalgemm.Q8Weight
	q4      []*metalgemm.Q4KWeight
	panel   metalgemm.GDNPanel
	block   metalgemm.Qwen35DecodeBlock
	geom    metalgemm.GDNGeometry
}

func newQwen35DecodeBlockFixture(t *testing.T) qwen35DecodeBlockFixture {
	t.Helper()
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	p := func(suffix string) string { return layerName(0, suffix) }
	mixerNames := []string{
		p("linear_attn.in_proj_qkv.weight"), p("linear_attn.in_proj_z.weight"),
		p("linear_attn.in_proj_b.weight"), p("linear_attn.in_proj_a.weight"), p("linear_attn.out_proj.weight"),
	}
	q8 := make([]*metalgemm.Q8Weight, len(mixerNames))
	for i, name := range mixerNames {
		qt := m.q8(name)
		q8[i] = metalgemm.UploadQ8(qt.q, qt.d, qt.out, qt.in)
		if q8[i] == nil {
			t.Fatalf("UploadQ8(%s) declined", name)
		}
	}
	mlpNames := []string{p("mlp.gate_proj.weight"), p("mlp.up_proj.weight"), p("mlp.down_proj.weight")}
	mlpShapes := [][2]int{{cfg.IntermediateSize, cfg.HiddenSize}, {cfg.IntermediateSize, cfg.HiddenSize}, {cfg.HiddenSize, cfg.IntermediateSize}}
	q4 := make([]*metalgemm.Q4KWeight, len(mlpNames))
	m.q4kw = make(map[string]*q4kTensor, len(mlpNames))
	for i, name := range mlpNames {
		qt := randomQ4KTensor(mlpShapes[i][0], mlpShapes[i][1], int64(94880+i))
		m.q4kw[name] = qt
		q4[i] = metalgemm.UploadQ4K(qt.raw, qt.out, qt.in)
		if q4[i] == nil {
			t.Fatalf("UploadQ4K(%s) declined", name)
		}
	}
	attnNorm, mlpNorm := m.attentionNorms(0), m.mlpNorms(0)
	return qwen35DecodeBlockFixture{
		m: m,
		weights: metalgemm.Qwen35DecodeWeights{
			InQKV: q8[0], InZ: q8[1], InB: q8[2], InA: q8[3], Out: q8[4],
			MLPActivation: q4[0], MLPUp: q4[1], MLPDownQ4: q4[2],
		},
		q8: q8, q4: q4,
		panel: metalgemm.GDNPanel{
			Tokens: 1, Conv1D: m.tensor(p("linear_attn.conv1d.weight")), ALog: m.tensor(p("linear_attn.A_log")),
			DTBias: m.tensor(p("linear_attn.dt_bias")), Norm: m.tensor(p("linear_attn.norm.weight")), RMSNormEpsilon: float32(cfg.RMSNormEps),
		},
		block: metalgemm.Qwen35DecodeBlock{
			InputNorm: attnNorm.pre, MLPNorm: mlpNorm.pre,
			RMSNormEpsilon: float32(cfg.RMSNormEps), NormGain1p: cfg.NormGain1p,
		},
		geom: metalgemm.GDNGeometry{
			NumKeyHeads: cfg.LinearNumKeyHeads, NumValueHeads: cfg.LinearNumValueHeads,
			KeyHeadDim: cfg.LinearKeyHeadDim, ValueHeadDim: cfg.LinearValueHeadDim, ConvKernel: cfg.LinearConvKernelDim,
		},
	}
}

func (f qwen35DecodeBlockFixture) close() {
	for i := len(f.q8) - 1; i >= 0; i-- {
		f.q8[i].Release()
	}
	for i := len(f.q4) - 1; i >= 0; i-- {
		f.q4[i].Release()
	}
}

func seedQwen35DecodeBlockStates(t *testing.T, f qwen35DecodeBlockFixture, state *metalgemm.GDNState, cpu *Session) {
	t.Helper()
	cfg := f.m.Cfg
	_, nV, kHd, vHd, _, _, convDim := cfg.linearAttnDims()
	convSeed := randomVecF((cfg.LinearConvKernelDim-1)*convDim, 94881)
	recurrentSeed := randomVecF(nV*kHd*vHd, 94882)
	if err := state.Seed(convSeed, recurrentSeed); err != nil {
		t.Fatal(err)
	}
	cpu.Cache.linear = newLinearAttnCache(cfg)
	layer := cpu.Cache.linear.layer(cfg, 0)
	layer.conv = make([][]float32, cfg.LinearConvKernelDim-1)
	for row := range layer.conv {
		layer.conv[row] = append([]float32(nil), convSeed[row*convDim:(row+1)*convDim]...)
	}
	for head := range layer.recurrent {
		copy(layer.recurrent[head], recurrentSeed[head*kHd*vHd:(head+1)*kHd*vHd])
	}
}

func qwen35DecodeBlockCPU(f qwen35DecodeBlockFixture, cpu *Session, input []float32) []float32 {
	cfg := f.m.Cfg
	x := append([]float32(nil), input...)
	attnNorm, mlpNorm := f.m.attentionNorms(0), f.m.mlpNorms(0)
	mat := sessionQ4KKernel{s: cpu}
	xn := normCfg(x, attnNorm.pre, attnNorm.preBias, float32(cfg.RMSNormEps), cfg)
	delta := cpu.linearAttnStep(0, xn, mat)
	for i := range x {
		x[i] += delta[i]
	}
	xn = normCfg(x, mlpNorm.pre, mlpNorm.preBias, float32(cfg.RMSNormEps), cfg)
	delta = f.m.ffnForLayer(0).apply(f.m, 0, mat.prep(xn), mat)
	for i := range x {
		x[i] += delta[i]
	}
	return x
}

func TestQwen35MetalDecodeBlockFourStepsParityAccountingAndGreedyToken(t *testing.T) {
	f := newQwen35DecodeBlockFixture(t)
	defer f.close()
	baseline := metalgemm.GDNLiveBufferCount()
	state, err := metalgemm.NewGDNState(f.geom)
	if err != nil {
		t.Fatal(err)
	}
	cpu := f.m.NewSession()
	cpu.Q4K = true
	seedQwen35DecodeBlockStates(t, f, state, cpu)
	defer cpu.Close()
	for step := 0; step < 4; step++ {
		input := randomVecF(f.m.Cfg.HiddenSize, int64(94900+step))
		want := qwen35DecodeBlockCPU(f, cpu, input)
		got, receipt, accepted, runErr := metalgemm.RunQwen35Decode(metalgemm.Qwen35DecodeRequest{
			Input: input, Weights: f.weights, State: state, Panel: f.panel, Block: &f.block,
		})
		if runErr != nil || !accepted {
			t.Fatalf("step %d accepted=%v err=%v", step, accepted, runErr)
		}
		assertQwen35DecodeBlockReceipt(t, receipt)
		assertQwen35DecodeBlockParity(t, "block hidden", want, got)
		cpuLayer := cpu.Cache.linear.layer(f.m.Cfg, 0)
		conv, recurrent, snapshotErr := state.Snapshot()
		if snapshotErr != nil {
			t.Fatal(snapshotErr)
		}
		assertQwen35DecodeMixerParity(t, "block convolution state", flattenQwen35Conv(cpuLayer), conv)
		assertQwen35DecodeMixerParity(t, "block recurrent state", flattenQwen35Recurrent(cpuLayer), recurrent)
		wantLogits := qMatRows(f.m.q8Head(), quantizeVecQ8(want))
		gotLogits := qMatRows(f.m.q8Head(), quantizeVecQ8(got))
		if decodeMixerArgmax(wantLogits) != decodeMixerArgmax(gotLogits) {
			t.Fatalf("step %d greedy token=%d want %d", step, decodeMixerArgmax(gotLogits), decodeMixerArgmax(wantLogits))
		}
	}
	state.Close()
	state.Close()
	if got := metalgemm.GDNLiveBufferCount(); got != baseline {
		t.Fatalf("exact-once Close left live buffers=%d, want %d", got, baseline)
	}
}

func TestQwen35MetalDecodeBlockIsolationDeclineAndPostSubmitFailure(t *testing.T) {
	f := newQwen35DecodeBlockFixture(t)
	defer f.close()
	first, err := metalgemm.NewGDNState(f.geom)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	second, err := metalgemm.NewGDNState(f.geom)
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	input := randomVecF(f.m.Cfg.HiddenSize, 95000)
	if _, _, accepted, runErr := metalgemm.RunQwen35Decode(metalgemm.Qwen35DecodeRequest{Input: input, Weights: f.weights, State: first, Panel: f.panel, Block: &f.block}); runErr != nil || !accepted {
		t.Fatalf("first block accepted=%v err=%v", accepted, runErr)
	}
	firstConv, firstRecurrent, _ := first.Snapshot()
	if _, _, accepted, runErr := metalgemm.RunQwen35Decode(metalgemm.Qwen35DecodeRequest{Input: randomVecF(len(input), 95001), Weights: f.weights, State: second, Panel: f.panel, Block: &f.block}); runErr != nil || !accepted {
		t.Fatalf("second block accepted=%v err=%v", accepted, runErr)
	}
	afterConv, afterRecurrent, _ := first.Snapshot()
	assertQwen35DecodeMixerParity(t, "isolated block convolution", firstConv, afterConv)
	assertQwen35DecodeMixerParity(t, "isolated block recurrent", firstRecurrent, afterRecurrent)

	missing := f.weights
	missing.MLPDownQ4 = nil
	beforeConv, beforeRecurrent, _ := second.Snapshot()
	if _, receipt, accepted, runErr := metalgemm.RunQwen35Decode(metalgemm.Qwen35DecodeRequest{Input: input, Weights: missing, State: second, Panel: f.panel, Block: &f.block}); accepted || runErr == nil || receipt.Committed {
		t.Fatalf("missing block handle accepted=%v receipt=%+v err=%v", accepted, receipt, runErr)
	}
	declinedConv, declinedRecurrent, _ := second.Snapshot()
	assertQwen35DecodeMixerParity(t, "declined block convolution", beforeConv, declinedConv)
	assertQwen35DecodeMixerParity(t, "declined block recurrent", beforeRecurrent, declinedRecurrent)

	failing, err := metalgemm.NewGDNState(f.geom)
	if err != nil {
		t.Fatal(err)
	}
	_, receipt, accepted, runErr := metalgemm.RunQwen35Decode(metalgemm.Qwen35DecodeRequest{
		Input: input, Weights: f.weights, State: failing, Panel: f.panel, Block: &f.block, InjectPostSubmitFailureForTest: true,
	})
	var post *metalgemm.GraphPostSubmitError
	if !accepted || !errors.As(runErr, &post) || !receipt.Committed || !receipt.CompletedWait {
		t.Fatalf("post-submit block accepted=%v receipt=%+v err=%v", accepted, receipt, runErr)
	}
	failing.Close()
	failing.Close()
}

func TestQwen35DecodeBlockDeclinesPerOperationTapAndPreservesLayerTap(t *testing.T) {
	backend := &recordingQwen35DecodeBlockBackend{metalQwen35GDNSequenceBackend: &metalQwen35GDNSequenceBackend{states: make(map[Qwen35GDNAuxState]*metalgemm.GDNState)}}
	s := &Session{M: &Model{}, qwen35HAL: &qwen35HALState{decodeAccepted: true, sequenceBackend: backend}}
	s.tapActive = &hiddenTap{ops: true}
	if _, _, accepted, err := s.tryQwen35MetalDecodeBlock(0, make([]float32, 256)); accepted || err != nil {
		t.Fatalf("per-operation tap accepted=%v err=%v", accepted, err)
	}
	if calls, _ := backend.snapshot(); calls != 0 {
		t.Fatalf("per-operation tap reached fused backend calls=%d, want 0", calls)
	}
	s.tapActive = &hiddenTap{ops: false}
	if _, _, accepted, err := s.tryQwen35MetalDecodeBlock(0, make([]float32, 256)); accepted || err != nil {
		t.Fatalf("layer-only tap preflight accepted=%v err=%v", accepted, err)
	}
	if calls, _ := backend.snapshot(); calls != 1 {
		t.Fatalf("layer-only tap reached fused backend calls=%d, want 1", calls)
	}
}

func TestQwen35DecodeBlockOperationTapUsesHistoricalTokenPath(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	m.q4kw = make(map[string]*q4kTensor)
	backend := &recordingQwen35DecodeBlockBackend{metalQwen35GDNSequenceBackend: &metalQwen35GDNSequenceBackend{states: make(map[Qwen35GDNAuxState]*metalgemm.GDNState)}}
	s := m.NewSession()
	defer s.Close()
	s.Q4K, s.MetalQ4K = true, true
	s.qwen35HAL = &qwen35HALState{decodeAccepted: true, sequenceBackend: backend, sequenceLayers: make([]Qwen35GDNAuxState, cfg.NumLayers)}
	dir := t.TempDir()
	s.tap = &hiddenTap{dir: dir, pos: 0, ops: true}
	_ = s.tokenHiddenQ(3, 0)
	if calls, _ := backend.snapshot(); calls != 0 {
		t.Fatalf("operation tap reached full-block backend calls=%d, want 0", calls)
	}
	if calls := backend.mixerCalls(); calls != 0 {
		t.Fatalf("operation tap reached mixer backend calls=%d, want 0", calls)
	}
	for _, op := range []string{"convOut", "qk_norm", "recurrent", "gated_norm", "out"} {
		if !tapFileExists(filepath.Join(dir, "layer_00_op_"+op+".f32")) {
			t.Fatalf("historical tokenHiddenQ path did not write operation tap %q", op)
		}
	}
}

func exactQwen38BlockTestCfg() Config {
	cfg := qwen35HybridQ4KTestCfg()
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
	return cfg
}

func TestQwen35DecodeBlockDeclinesLayerNormNearMiss(t *testing.T) {
	cfg := exactQwen38BlockTestCfg()
	cfg.LayerNorm = true
	b := &metalQwen35GDNSequenceBackend{states: make(map[Qwen35GDNAuxState]*metalgemm.GDNState)}
	s := &Session{M: &Model{Cfg: cfg}, Q4K: true, MetalQ4K: true, qwen35HAL: &qwen35HALState{
		decodeAccepted: true, sequenceLayers: make([]Qwen35GDNAuxState, cfg.NumLayers),
	}}
	if _, receipt, accepted, err := b.Qwen35MetalDecodeBlock(s, 0, make([]float32, cfg.HiddenSize)); accepted || err != nil || receipt.Committed {
		t.Fatalf("LayerNorm near-miss accepted=%v receipt=%+v err=%v", accepted, receipt, err)
	}
}

func TestExactQwen38SessionSelectsWholeDecodeBlockAndHooksOnce(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	cfg := exactQwen38BlockTestCfg()
	linearLayers := qwen35LinearLayerCount(cfg)
	m := NewSynthetic(cfg)
	m.Quantize()
	m.q4kw = make(map[string]*q4kTensor, cfg.NumLayers*3)
	m.kqw = make(map[string]*kQuantTensor, linearLayers)
	for layer := 0; layer < cfg.NumLayers; layer++ {
		p := func(suffix string) string { return layerName(layer, suffix) }
		for i, shape := range [][3]any{
			{p("mlp.gate_proj.weight"), cfg.IntermediateSize, cfg.HiddenSize},
			{p("mlp.up_proj.weight"), cfg.IntermediateSize, cfg.HiddenSize},
		} {
			m.q4kw[shape[0].(string)] = randomQ4KTensor(shape[1].(int), shape[2].(int), int64(95100+layer*3+i))
		}
		downName := p("mlp.down_proj.weight")
		if cfg.isLinearAttnLayer(layer) {
			m.kqw[downName] = randomQ6KTensor(cfg.HiddenSize, cfg.IntermediateSize, int64(95200+layer))
		} else {
			m.q4kw[downName] = randomQ4KTensor(cfg.HiddenSize, cfg.IntermediateSize, int64(95200+layer))
		}
	}
	defer func() {
		m.releaseMetalQ8Residency()
		releaseMetalQ4KResidency(m)
		metalQ4KMu.Lock()
		delete(metalQ6KW, m)
		metalQ4KMu.Unlock()
		metalgemm.ResetQ4K()
	}()
	if err := m.promoteMetalQ8Residency(); err != nil {
		t.Fatalf("promote exact Qwen3.8 Q8 residency: %v", err)
	}
	reference := m.NewSession()
	reference.Q4K = true
	wantHidden := reference.tokenHiddenQ(3, 0)
	reference.Close()

	candidate := m.NewSession()
	candidate.Q4K, candidate.MetalQ4K = true, true
	backend := &recordingQwen35DecodeBlockBackend{metalQwen35GDNSequenceBackend: &metalQwen35GDNSequenceBackend{states: make(map[Qwen35GDNAuxState]*metalgemm.GDNState)}}
	accepted, err := candidate.initQwen35GDNPreprojectedSequence(backend)
	if err != nil || !accepted {
		t.Fatalf("initialize exact Qwen3.8 owners accepted=%v err=%v", accepted, err)
	}
	_, nV, kHd, vHd, _, _, convDim := cfg.linearAttnDims()
	snapshots := make([]qwen35GDNLayerSnapshot, 0, linearLayers)
	for layer, state := range candidate.qwen35HAL.sequenceLayers {
		if state.valid() {
			snapshots = append(snapshots, qwen35GDNLayerSnapshot{
				layer: layer, conv: make([]float32, (cfg.LinearConvKernelDim-1)*convDim), recurrent: make([]float32, nV*kHd*vHd),
			})
		}
	}
	if selected, promoteErr := candidate.promoteQwen35MetalGDNDecode(snapshots); promoteErr != nil || !selected {
		t.Fatalf("promote exact Qwen3.8 decode selected=%v err=%v", selected, promoteErr)
	}
	hookCalls := 0
	m.Cfg.EnableResidualHook = true
	m.SetResidualHook(func(_ int, _ []float32) { hookCalls++ })
	gotHidden := candidate.tokenHiddenQ(3, 0)
	candidate.Close()
	calls, receipts := backend.snapshot()
	if calls != linearLayers || len(receipts) != linearLayers {
		t.Fatalf("exact Qwen3.8 product path block calls=%d receipts=%d, want one per %d linear layers", calls, len(receipts), linearLayers)
	}
	for i, receipt := range receipts {
		if receipt.Encoders != 16 || receipt.ProjectionDispatches != 8 || receipt.InputUploads != 1 || receipt.ConstantUploads != 6 ||
			receipt.FinalReadbacks != 1 || receipt.IntermediateReadbacks != 0 || receipt.StateH2DTransfers != 0 || receipt.StateD2HTransfers != 0 ||
			!receipt.Committed || !receipt.CompletedWait {
			t.Fatalf("exact Qwen3.8 product path block %d receipt=%+v", i, receipt)
		}
	}
	metalQ4KMu.Lock()
	q6Resolved := len(metalQ6KW[m])
	metalQ4KMu.Unlock()
	if q6Resolved != linearLayers {
		t.Fatalf("exact Qwen3.8 Q6_K block-down handles=%d, want one per %d linear layers", q6Resolved, linearLayers)
	}
	if hookCalls != cfg.NumLayers {
		t.Fatalf("residual hook calls=%d, want one per layer=%d", hookCalls, cfg.NumLayers)
	}
	assertQwen35DecodeBlockParity(t, "exact Qwen3.8 full-token hidden", wantHidden, gotHidden)
	wantLogits := qMatRows(m.q8Head(), quantizeVecQ8(wantHidden))
	gotLogits := qMatRows(m.q8Head(), quantizeVecQ8(gotHidden))
	if decodeMixerArgmax(wantLogits) != decodeMixerArgmax(gotLogits) {
		t.Fatalf("exact Qwen3.8 greedy token=%d want %d", decodeMixerArgmax(gotLogits), decodeMixerArgmax(wantLogits))
	}
}

func TestExactQwen38DecodeBlockPostSubmitFailureIsTerminal(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	cfg := exactQwen38BlockTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	m.q4kw = make(map[string]*q4kTensor, 3)
	p := func(suffix string) string { return layerName(0, suffix) }
	m.q4kw[p("mlp.gate_proj.weight")] = randomQ4KTensor(cfg.IntermediateSize, cfg.HiddenSize, 95301)
	m.q4kw[p("mlp.up_proj.weight")] = randomQ4KTensor(cfg.IntermediateSize, cfg.HiddenSize, 95302)
	m.q4kw[p("mlp.down_proj.weight")] = randomQ4KTensor(cfg.HiddenSize, cfg.IntermediateSize, 95303)
	defer func() {
		m.releaseMetalQ8Residency()
		releaseMetalQ4KResidency(m)
	}()
	if err := m.promoteMetalQ8Residency(); err != nil {
		t.Fatal(err)
	}
	baseline := metalgemm.GDNLiveBufferCount()
	base := &metalQwen35GDNSequenceBackend{states: make(map[Qwen35GDNAuxState]*metalgemm.GDNState)}
	backend := &failingQwen35DecodeBlockBackend{metalQwen35GDNSequenceBackend: base}
	s := m.NewSession()
	defer s.Close()
	s.Q4K, s.MetalQ4K = true, true
	accepted, err := s.initQwen35GDNPreprojectedSequence(backend)
	if err != nil || !accepted {
		t.Fatalf("initialize accepted=%v err=%v", accepted, err)
	}
	_, nV, kHd, vHd, _, _, convDim := cfg.linearAttnDims()
	snapshots := make([]qwen35GDNLayerSnapshot, 0, 48)
	for layer, state := range s.qwen35HAL.sequenceLayers {
		if state.valid() {
			snapshots = append(snapshots, qwen35GDNLayerSnapshot{layer: layer, conv: make([]float32, (cfg.LinearConvKernelDim-1)*convDim), recurrent: make([]float32, nV*kHd*vHd)})
		}
	}
	if selected, promoteErr := s.promoteQwen35MetalGDNDecode(snapshots); promoteErr != nil || !selected {
		t.Fatalf("promote selected=%v err=%v", selected, promoteErr)
	}
	hostLayer := s.Cache.linear.layer(cfg, 0)
	hostConvBefore := append([]float32(nil), flattenQwen35Conv(hostLayer)...)
	hostRecurrentBefore := append([]float32(nil), flattenQwen35Recurrent(hostLayer)...)
	panicErr := recoverError(func() { _ = s.tokenHiddenQ(3, 0) })
	if panicErr == nil {
		t.Fatal("accepted post-submit failure returned without fail-closed panic")
	}
	blockCalls, mixerCalls, receipt := backend.snapshot()
	if blockCalls != 1 || mixerCalls != 0 || receipt.Encoders != 16 || !receipt.Committed || !receipt.CompletedWait {
		t.Fatalf("post-submit calls block/mixer=%d/%d receipt=%+v", blockCalls, mixerCalls, receipt)
	}
	assertQwen35DecodeMixerParity(t, "post-submit host convolution", hostConvBefore, flattenQwen35Conv(hostLayer))
	assertQwen35DecodeMixerParity(t, "post-submit host recurrent", hostRecurrentBefore, flattenQwen35Recurrent(hostLayer))
	if s.qwen35HAL == nil || s.qwen35HAL.decodeAccepted || len(s.qwen35HAL.sequenceLayers) != 0 || s.qwen35HAL.sequenceBackend != nil {
		t.Fatalf("post-submit sequence ownership state=%#v", s.qwen35HAL)
	}
	if live := metalgemm.GDNLiveBufferCount(); live != baseline {
		t.Fatalf("post-submit failure retained GDN buffers=%d, baseline=%d", live, baseline)
	}
}

func assertQwen35DecodeBlockReceipt(t *testing.T, receipt metalgemm.Qwen35DecodeReceipt) {
	t.Helper()
	if receipt.CommandBuffers != 1 || receipt.Commits != 1 || receipt.CompletionWaits != 1 || receipt.Encoders != 16 ||
		receipt.ProjectionDispatches != 8 || receipt.MixerProjectionDispatches != 5 || receipt.MLPProjectionDispatches != 3 ||
		receipt.RMSNormEncoders != 2 || receipt.Quantizers != 2 || receipt.GDNEncoders != 1 || receipt.ResidualAddEncoders != 2 || receipt.SwiGLUEncoders != 1 ||
		receipt.InputUploads != 1 || receipt.ConstantUploads != 6 || receipt.FinalReadbacks != 1 || receipt.IntermediateReadbacks != 0 ||
		receipt.StateH2DTransfers != 0 || receipt.StateD2HTransfers != 0 || !receipt.Committed || !receipt.CompletedWait {
		t.Fatalf("decode block receipt=%+v", receipt)
	}
}

func assertQwen35DecodeBlockParity(t *testing.T, label string, want, got []float32) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s elements=%d want %d", label, len(got), len(want))
	}
	var dot, wn, gn, errorNorm float64
	for i := range want {
		w, g := float64(want[i]), float64(got[i])
		dot += w * g
		wn += w * w
		gn += g * g
		d := w - g
		errorNorm += d * d
	}
	cosine := dot / math.Sqrt(wn*gn)
	relativeL2 := math.Sqrt(errorNorm / math.Max(wn, 1))
	if cosine < 0.9999 || relativeL2 >= 0.02 {
		t.Fatalf("%s cosine=%g relativeL2=%g", label, cosine, relativeL2)
	}
}
