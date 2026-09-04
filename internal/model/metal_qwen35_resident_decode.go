//go:build darwin && arm64 && cgo

package model

import (
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// Qwen35ResidentStageProfile records the five staged latency attributions
// and synchronization accounting for one resident decode token forward.
type Qwen35ResidentStageProfile struct {
	HostCopy         time.Duration `json:"host_copy"`
	Encode           time.Duration `json:"encode"`
	CommitToComplete time.Duration `json:"commit_to_complete"`
	GPUCompute       time.Duration `json:"gpu_compute"`
	ResultReadback   time.Duration `json:"result_readback"`
	Total            time.Duration `json:"total"`

	HostCopyMS         float64 `json:"host_copy_ms"`
	EncodeMS           float64 `json:"encode_ms"`
	CommitToCompleteMS float64 `json:"commit_to_complete_ms"`
	GPUComputeMS       float64 `json:"gpu_compute_ms"`
	ResultReadbackMS   float64 `json:"result_readback_ms"`
	TotalMS            float64 `json:"total_ms"`

	Synchronizations      int     `json:"synchronizations"`
	AmortizedSyncPerLayer float64 `json:"amortized_sync_per_layer"`
	NumLayers             int     `json:"num_layers"`
	Fallback              bool    `json:"fallback"`
	FallbackReason        string  `json:"fallback_reason,omitempty"`
}

// Qwen35ResidentCPUCheck captures the CPU reference comparison for one decode token forward.
type Qwen35ResidentCPUCheck struct {
	CosineSimilarity float64 `json:"cosine_similarity"`
	MaxAbsDiff       float64 `json:"max_abs_diff"`
	GreedyMatch      bool    `json:"greedy_match"`
	ExpectedToken    int     `json:"expected_token"`
	ActualToken      int     `json:"actual_token"`
	Passed           bool    `json:"passed"`
}

type qwen35ResidentLinearLayer struct {
	layer       int
	weights     metalgemm.Qwen35DecodeWeights
	panel       metalgemm.GDNPanel
	block       metalgemm.Qwen35DecodeBlock
	geom        metalgemm.GDNGeometry
	state       *metalgemm.GDNState
	ownsState   bool
	ownsWeights []*metalgemm.Q8Weight
	ownsQ4K     []*metalgemm.Q4KWeight
}

type qwen35ResidentFullLayer struct {
	layer        int
	attnNorm     normWeights
	mlpNorm      normWeights
	qWeight      *metalgemm.Q8Weight
	kWeight      *metalgemm.Q8Weight
	vWeight      *metalgemm.Q4KWeight
	vWeightQ8    *metalgemm.Q8Weight
	oWeight      *metalgemm.Q4KWeight
	oWeightQ8    *metalgemm.Q8Weight
	gateWeight   *metalgemm.Q4KWeight
	upWeight     *metalgemm.Q4KWeight
	downWeight   *metalgemm.Q4KWeight
	downWeightQ6 *metalgemm.Q6KWeight
	qNorm        []float32
	kNorm        []float32
	ownsQ8       []*metalgemm.Q8Weight
	ownsQ4K      []*metalgemm.Q4KWeight
}

// Qwen35ResidentMetalDecoder routes dense Qwen3.5/Qwen3.8 hybrid models through a
// coarse resident Metal decode token forward, lifting the IsQwen35Hybrid() decline.
// It maintains activations, KV/recurrent states, and projections resident on device,
// amortizing synchronization once per token instead of per layer.
type Qwen35ResidentMetalDecoder struct {
	mu             sync.Mutex
	m              *Model
	s              *Session
	cfg            Config
	closed         bool
	fallbackActive bool
	fallbackReason string

	linearLayers map[int]*qwen35ResidentLinearLayer
	fullLayers   map[int]*qwen35ResidentFullLayer

	finalNormWeight []float32
	headQ8          *metalgemm.Q8Weight
	headQ4K         *metalgemm.Q4KWeight
	ownsHead        bool

	lastProfile Qwen35ResidentStageProfile
}

// NewQwen35ResidentMetalDecoder constructs a new resident Metal decoder for a session.
// If the model geometry, layer types, or hardware runtime are unsupported, it fails
// closed by activating the reference forward fallback.
func NewQwen35ResidentMetalDecoder(s *Session) (*Qwen35ResidentMetalDecoder, error) {
	if s == nil || s.M == nil {
		return nil, errors.New("model: nil session or model")
	}
	m := s.M
	cfg := m.Cfg

	decoder := &Qwen35ResidentMetalDecoder{
		m:            m,
		s:            s,
		cfg:          cfg,
		linearLayers: make(map[int]*qwen35ResidentLinearLayer),
		fullLayers:   make(map[int]*qwen35ResidentFullLayer),
	}

	if err := decoder.validateAndInit(); err != nil {
		decoder.fallbackActive = true
		decoder.fallbackReason = err.Error()
	}

	return decoder, nil
}

// NewQwen35ResidentMetalDecoderForModel constructs a decoder for a model by wrapping
// a fresh session.
func NewQwen35ResidentMetalDecoderForModel(m *Model) (*Qwen35ResidentMetalDecoder, error) {
	if m == nil {
		return nil, errors.New("model: nil model")
	}
	s := m.NewSession()
	s.Q4K = true
	s.MetalQ4K = true
	return NewQwen35ResidentMetalDecoder(s)
}

func (d *Qwen35ResidentMetalDecoder) validateAndInit() error {
	if err := validateResidentConfig(&d.cfg); err != nil {
		return err
	}

	if d.s.Cache == nil {
		d.s.Cache = NewKVCache(d.cfg)
	}
	if d.s.Cache.linear == nil {
		d.s.Cache.linear = newLinearAttnCache(d.cfg)
	}

	geom := metalgemm.GDNGeometry{
		NumKeyHeads:   d.cfg.LinearNumKeyHeads,
		NumValueHeads: d.cfg.LinearNumValueHeads,
		KeyHeadDim:    d.cfg.LinearKeyHeadDim,
		ValueHeadDim:  d.cfg.LinearValueHeadDim,
		ConvKernel:    d.cfg.LinearConvKernelDim,
	}

	for l := 0; l < d.cfg.NumLayers; l++ {
		if d.cfg.isLinearAttnLayer(l) {
			layer, err := d.initResidentLinearLayer(l, geom)
			if err != nil {
				return err
			}
			d.linearLayers[l] = layer
		} else {
			d.fullLayers[l] = d.initResidentFullLayer(l)
		}
	}

	d.finalNormWeight = d.m.tensor("model.norm.weight")
	return nil
}

func validateResidentConfig(cfg *Config) error {
	if !cfg.IsQwen35Hybrid() {
		return errors.New("unsupported geometry: not a Qwen3.5/Qwen3.8 hybrid model")
	}
	if cfg.IsMoE() {
		return errors.New("unsupported geometry: MoE architecture is not supported in resident decode")
	}
	if cfg.BlockTopology != PreNorm {
		return errors.New("unsupported geometry: only PreNorm topology is supported")
	}
	if cfg.LayerNorm {
		return errors.New("unsupported geometry: biased LayerNorm is not supported")
	}
	if cfg.ActGeluTanh || cfg.ActGeluErf {
		return errors.New("unsupported geometry: GELU activations are not supported")
	}
	if cfg.HiddenSize <= 0 || cfg.HiddenSize%32 != 0 {
		return fmt.Errorf("unsupported geometry: hidden size %d is not a multiple of 32", cfg.HiddenSize)
	}
	if cfg.NumLayers <= 0 {
		return errors.New("unsupported geometry: NumLayers must be positive")
	}
	if cfg.VocabSize <= 0 {
		return errors.New("unsupported geometry: VocabSize must be positive")
	}
	if cfg.LinearKeyHeadDim <= 0 || cfg.LinearValueHeadDim <= 0 || cfg.LinearConvKernelDim <= 0 ||
		cfg.LinearNumKeyHeads <= 0 || cfg.LinearNumValueHeads <= 0 {
		return errors.New("unsupported geometry: invalid linear attention dimension parameters")
	}
	if !metalgemm.Available() {
		return errors.New("runtime: Metal device is unavailable")
	}
	return nil
}

func (d *Qwen35ResidentMetalDecoder) initResidentLinearLayer(l int, geom metalgemm.GDNGeometry) (*qwen35ResidentLinearLayer, error) {
	p := func(suffix string) string { return layerName(l, suffix) }
	for _, reqName := range []string{
		p("linear_attn.conv1d.weight"),
		p("linear_attn.A_log"),
		p("linear_attn.dt_bias"),
		p("linear_attn.norm.weight"),
	} {
		if !d.m.has(reqName) {
			return nil, fmt.Errorf("unsupported geometry: missing linear attention tensor %s", reqName)
		}
	}

	names := []string{
		p("linear_attn.in_proj_qkv.weight"),
		p("linear_attn.in_proj_z.weight"),
		p("linear_attn.in_proj_b.weight"),
		p("linear_attn.in_proj_a.weight"),
		p("linear_attn.out_proj.weight"),
	}
	metalQ4KMu.Lock()
	table := metalQ8KW[d.m]
	handles := make([]*metalgemm.Q8Weight, len(names))
	if table != nil {
		for i, name := range names {
			handles[i] = table[name]
		}
	}
	metalQ4KMu.Unlock()

	var ownedQ8 []*metalgemm.Q8Weight
	for i, name := range names {
		if handles[i] == nil {
			qt := d.m.q8(name)
			if qt == nil {
				return nil, fmt.Errorf("missing Q8 tensor %s", name)
			}
			h := metalgemm.UploadQ8(qt.q, qt.d, qt.out, qt.in)
			if h == nil {
				return nil, fmt.Errorf("failed to upload Q8 weight %s", name)
			}
			handles[i] = h
			ownedQ8 = append(ownedQ8, h)
		}
	}

	gateName, upName, downName := p("mlp.gate_proj.weight"), p("mlp.up_proj.weight"), p("mlp.down_proj.weight")
	gateTensor, upTensor := d.m.q4kw[gateName], d.m.q4kw[upName]
	if gateTensor == nil || upTensor == nil {
		return nil, fmt.Errorf("missing MLP Q4_K tensors for layer %d", l)
	}
	gate := d.m.metalQ4KWeight(gateName, gateTensor)
	up := d.m.metalQ4KWeight(upName, upTensor)
	if gate == nil || up == nil {
		return nil, fmt.Errorf("failed to resolve MLP Q4_K weights for layer %d", l)
	}

	weights := metalgemm.Qwen35DecodeWeights{
		InQKV: handles[0], InZ: handles[1], InB: handles[2], InA: handles[3], Out: handles[4],
		MLPActivation: gate, MLPUp: up,
	}
	if downTensor := d.m.q4kw[downName]; downTensor != nil {
		weights.MLPDownQ4 = d.m.metalQ4KWeight(downName, downTensor)
	} else if downTensor := d.m.kqw[downName]; downTensor != nil && downTensor.kind == kindQ6K {
		weights.MLPDownQ6 = d.m.metalQ6KWeight(downName, downTensor)
	}
	if weights.MLPDownQ4 == nil && weights.MLPDownQ6 == nil {
		return nil, fmt.Errorf("missing MLP down projection for layer %d", l)
	}

	attnNorm, mlpNorm := d.m.attentionNorms(l), d.m.mlpNorms(l)
	if len(attnNorm.pre) != d.cfg.HiddenSize || len(mlpNorm.pre) != d.cfg.HiddenSize {
		return nil, fmt.Errorf("invalid normalization size for layer %d", l)
	}

	state, err := metalgemm.NewGDNState(geom)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate resident GDN state for layer %d: %w", l, err)
	}

	return &qwen35ResidentLinearLayer{
		layer:   l,
		weights: weights,
		panel: metalgemm.GDNPanel{
			Tokens:         1,
			Conv1D:         d.m.tensor(p("linear_attn.conv1d.weight")),
			ALog:           d.m.tensor(p("linear_attn.A_log")),
			DTBias:         d.m.tensor(p("linear_attn.dt_bias")),
			Norm:           d.m.tensor(p("linear_attn.norm.weight")),
			RMSNormEpsilon: float32(d.cfg.RMSNormEps),
		},
		block: metalgemm.Qwen35DecodeBlock{
			InputNorm:      attnNorm.pre,
			MLPNorm:        mlpNorm.pre,
			RMSNormEpsilon: float32(d.cfg.RMSNormEps),
			NormGain1p:     d.cfg.NormGain1p,
		},
		geom:        geom,
		state:       state,
		ownsState:   true,
		ownsWeights: ownedQ8,
	}, nil
}

func (d *Qwen35ResidentMetalDecoder) initResidentFullLayer(l int) *qwen35ResidentFullLayer {
	p := func(suffix string) string { return layerName(l, suffix) }
	attnNorm, mlpNorm := d.m.attentionNorms(l), d.m.mlpNorms(l)
	qName := p("self_attn.q_proj.weight")
	kName := p("self_attn.k_proj.weight")
	vName := p("self_attn.v_proj.weight")
	oName := p("self_attn.o_proj.weight")

	var qNorm, kNorm []float32
	if d.cfg.QKNorm || d.m.has(p("self_attn.q_norm.weight")) {
		qNorm = d.m.tensor(p("self_attn.q_norm.weight"))
		kNorm = d.m.tensor(p("self_attn.k_norm.weight"))
	}

	gateName, upName, downName := p("mlp.gate_proj.weight"), p("mlp.up_proj.weight"), p("mlp.down_proj.weight")
	gateTensor, upTensor := d.m.q4kw[gateName], d.m.q4kw[upName]
	var gate, up *metalgemm.Q4KWeight
	if gateTensor != nil && upTensor != nil {
		gate = d.m.metalQ4KWeight(gateName, gateTensor)
		up = d.m.metalQ4KWeight(upName, upTensor)
	}

	var downQ4 *metalgemm.Q4KWeight
	var downQ6 *metalgemm.Q6KWeight
	if downTensor := d.m.q4kw[downName]; downTensor != nil {
		downQ4 = d.m.metalQ4KWeight(downName, downTensor)
	} else if downTensor := d.m.kqw[downName]; downTensor != nil && downTensor.kind == kindQ6K {
		downQ6 = d.m.metalQ6KWeight(downName, downTensor)
	}

	return &qwen35ResidentFullLayer{
		layer:        l,
		attnNorm:     attnNorm,
		mlpNorm:      mlpNorm,
		qWeight:      d.m.metalQ8Weight(qName, d.m.q8(qName)),
		kWeight:      d.m.metalQ8Weight(kName, d.m.q8(kName)),
		vWeight:      d.m.metalQ4KWeight(vName, d.m.q4kw[vName]),
		oWeight:      d.m.metalQ4KWeight(oName, d.m.q4kw[oName]),
		gateWeight:   gate,
		upWeight:     up,
		downWeight:   downQ4,
		downWeightQ6: downQ6,
		qNorm:        qNorm,
		kNorm:        kNorm,
	}
}

// IsFallback reports whether fail-closed reference fallback is active.
func (d *Qwen35ResidentMetalDecoder) IsFallback() bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.fallbackActive
}

// FallbackReason returns the reason why fallback was activated.
func (d *Qwen35ResidentMetalDecoder) FallbackReason() string {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.fallbackReason
}

// LastProfile returns the profile recorded for the most recent decode forward.
func (d *Qwen35ResidentMetalDecoder) LastProfile() Qwen35ResidentStageProfile {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.lastProfile
}

// Close releases all resident Metal resources and states.
func (d *Qwen35ResidentMetalDecoder) Close() {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return
	}
	d.closed = true
	for _, l := range d.linearLayers {
		if l.ownsState && l.state != nil {
			l.state.Close()
		}
		for _, w := range l.ownsWeights {
			w.Release()
		}
		for _, w := range l.ownsQ4K {
			w.Release()
		}
	}
	for _, l := range d.fullLayers {
		for _, w := range l.ownsQ8 {
			w.Release()
		}
		for _, w := range l.ownsQ4K {
			w.Release()
		}
	}
}

// Step runs one decode step forward.
func (d *Qwen35ResidentMetalDecoder) Step(id, pos int) ([]float32, Qwen35ResidentStageProfile, error) {
	return d.DecodeToken(id, pos)
}

// ForwardToken is an alias for DecodeToken.
func (d *Qwen35ResidentMetalDecoder) ForwardToken(id, pos int) ([]float32, Qwen35ResidentStageProfile, error) {
	return d.DecodeToken(id, pos)
}

// DecodeToken executes one token decode forward through the coarse resident Metal path,
// maintaining intermediate activations and recurrent states on device, and paying
// synchronization once per token instead of per layer.
func (d *Qwen35ResidentMetalDecoder) DecodeToken(id, pos int) ([]float32, Qwen35ResidentStageProfile, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.closed {
		return nil, Qwen35ResidentStageProfile{}, errors.New("model: decoder is closed")
	}

	tTotal := time.Now()

	if d.fallbackActive {
		t0 := time.Now()
		out := d.s.token(id, pos)
		tot := time.Since(t0)
		prof := Qwen35ResidentStageProfile{
			HostCopy:              0,
			Encode:                0,
			CommitToComplete:      0,
			GPUCompute:            0,
			ResultReadback:        tot,
			Total:                 tot,
			ResultReadbackMS:      float64(tot.Nanoseconds()) / 1e6,
			TotalMS:               float64(tot.Nanoseconds()) / 1e6,
			Synchronizations:      1,
			AmortizedSyncPerLayer: 1.0 / float64(d.cfg.NumLayers),
			NumLayers:             d.cfg.NumLayers,
			Fallback:              true,
			FallbackReason:        d.fallbackReason,
		}
		d.lastProfile = prof
		return out, prof, nil
	}

	// 1. Host Copy
	tHost0 := time.Now()
	H := d.cfg.HiddenSize
	embed := d.m.embedRows()
	if id < 0 || (id+1)*H > len(embed) {
		return nil, Qwen35ResidentStageProfile{}, fmt.Errorf("model: invalid token ID %d", id)
	}
	x := make([]float32, H)
	copy(x, embed[id*H:(id+1)*H])
	scaleEmbedInPlace(x, d.cfg)
	hostCopyDur := time.Since(tHost0)

	// Sync linear state from session cache if populated
	_ = d.syncLinearFromHost()

	// 2. Encode Stage
	tEncode0 := time.Now()
	var commitDur, gpuDur time.Duration
	syncsPaid := 0

	// Walk layers. Consecutive GDN layers are batched into coarse resident projection graphs.
	l := 0
	for l < d.cfg.NumLayers {
		if d.cfg.isLinearAttnLayer(l) {
			// Find contiguous run of linear attention layers
			startL := l
			for l < d.cfg.NumLayers && d.cfg.isLinearAttnLayer(l) {
				l++
			}
			endL := l

			graph, err := metalgemm.BeginProjectionGraph(x, nil, nil, 1, H)
			if err != nil {
				return d.executeFallback(id, pos, fmt.Sprintf("graph begin failed: %v", err))
			}

			curX, err := graph.Input(H)
			if err != nil {
				graph.Free()
				return d.executeFallback(id, pos, fmt.Sprintf("graph input failed: %v", err))
			}

			for idx := startL; idx < endL; idx++ {
				lin := d.linearLayers[idx]
				req := metalgemm.Qwen35DecodeRequest{
					Weights: lin.weights,
					State:   lin.state,
					Panel:   lin.panel,
					Block:   &lin.block,
				}
				nextX, _, encErr := metalgemm.EncodeQwen35Decode(graph, curX, req)
				if encErr != nil {
					graph.Free()
					return d.executeFallback(id, pos, fmt.Sprintf("graph encode layer %d failed: %v", idx, encErr))
				}
				curX = nextX
			}

			// 3. Commit-to-Complete for this coarse resident segment
			tCommit0 := time.Now()
			outputs, receipt, finishErr := graph.FinishRead(curX)
			graph.Free()
			cDur := time.Since(tCommit0)
			commitDur += cDur
			syncsPaid++

			if finishErr != nil || len(outputs) != 1 || len(outputs[0]) != H {
				return d.executeFallback(id, pos, fmt.Sprintf("graph finish failed: %v", finishErr))
			}

			if receipt.TimingAvailable && receipt.GPUMilliseconds > 0 {
				gpuDur += time.Duration(receipt.GPUMilliseconds * float64(time.Millisecond))
			} else {
				gpuDur += cDur
			}

			x = outputs[0]
		} else {
			// Periodic Full Attention Layer
			fl := d.fullLayers[l]
			mat := sessionQ4KKernel{s: d.s}
			eps := float32(d.cfg.RMSNormEps)
			nH, nKV, hd := d.cfg.NumHeads, d.cfg.NumKVHeads, d.cfg.HeadDim
			w := nKV * hd
			qWidth := nH * hd
			scale := d.cfg.attnScale()
			p := func(str string) string { return layerName(l, str) }

			xn := normCfg(x, fl.attnNorm.pre, fl.attnNorm.preBias, eps, d.cfg)
			xp := mat.prep(xn)

			var q, gate, kk, vv []float32
			if d.cfg.AttnOutputGate {
				qf := mat.mul(p("self_attn.q_proj.weight"), xp, 2*qWidth, H)
				q, gate = splitPackedQueryGate(qf, nH, hd)
			} else {
				q = mat.mul(p("self_attn.q_proj.weight"), xp, qWidth, H)
			}
			kk = mat.mul(p("self_attn.k_proj.weight"), xp, w, H)
			vv = mat.mul(p("self_attn.v_proj.weight"), xp, w, H)

			d.m.applyProjBias(l, q, kk, vv)
			d.m.applyLayerQKNorm(l, q, kk)

			cos, sin := ropeRowForLayer(d.cfg, l, pos)
			if d.cfg.Alibi {
				d.s.Cache.Kraw[l] = append(d.s.Cache.Kraw[l], kk...)
			} else {
				d.s.ropeRowQK(l, q, kk, cos, sin)
			}

			d.s.Cache.K[l] = append(d.s.Cache.K[l], kk...)
			d.s.Cache.V[l] = append(d.s.Cache.V[l], vv...)

			nPos := len(d.s.Cache.K[l]) / w
			lo := windowLoStep(d.s.Cache.pos, nPos, pos, d.cfg.windowForLayer(l))
			attnOut := make([]float32, nH*hd)
			scores := make([]float32, nPos-lo)

			grp := d.cfg.GroupSize()
			attnCap := float32(d.cfg.AttnSoftcap)

			for h := 0; h < nH; h++ {
				kvh := h / grp
				qh := q[h*hd : (h+1)*hd]
				for j := lo; j < nPos; j++ {
					kh := d.s.Cache.K[l][j*w+kvh*hd : j*w+(kvh+1)*hd]
					scores[j-lo] = dot(qh, kh)*scale + d.cfg.alibiScoreBias(h, j, nPos)
				}
				softcapInPlace(scores, attnCap)
				d.m.softmaxAttentionScores(l, h, scores)
				outHead := attnOut[h*hd : (h+1)*hd]
				for j := lo; j < nPos; j++ {
					vh := d.s.Cache.V[l][j*w+kvh*hd : j*w+(kvh+1)*hd]
					saxpy(outHead, vh, scores[j-lo])
				}
			}

			if d.cfg.AttnOutputGate {
				for i := 0; i < qWidth; i++ {
					attnOut[i] *= sigmoidf(gate[i])
				}
			}

			ao := mat.prep(attnOut)
			outProj := mat.mul(p("self_attn.o_proj.weight"), ao, H, nH*hd)
			d.m.addBiasIfPresent(outProj, p("self_attn.o_proj.bias"))
			for i := 0; i < H; i++ {
				x[i] += outProj[i]
			}

			xn2 := normCfg(x, fl.mlpNorm.pre, fl.mlpNorm.preBias, eps, d.cfg)
			mlpOut := d.m.ffnForLayer(l).apply(d.m, l, mat.prep(xn2), mat)
			for i := 0; i < H; i++ {
				x[i] += mlpOut[i]
			}

			l++
		}
	}
	encodeDur := time.Since(tEncode0) - commitDur

	// 5. Result Readback: Final norm + LM head + state sync
	tRead0 := time.Now()
	xNorm := d.m.finalNorm(x)
	logits := d.s.headResident(xNorm)
	logitScaleInPlace(logits, d.cfg)

	d.s.Cache.appendPosition(pos, id)
	_ = d.syncLinearToHost()
	readbackDur := time.Since(tRead0)

	if syncsPaid == 0 {
		syncsPaid = 1
	}

	totalDur := time.Since(tTotal)
	profile := Qwen35ResidentStageProfile{
		HostCopy:              hostCopyDur,
		Encode:                encodeDur,
		CommitToComplete:      commitDur,
		GPUCompute:            gpuDur,
		ResultReadback:        readbackDur,
		Total:                 totalDur,
		HostCopyMS:            float64(hostCopyDur.Nanoseconds()) / 1e6,
		EncodeMS:              float64(encodeDur.Nanoseconds()) / 1e6,
		CommitToCompleteMS:    float64(commitDur.Nanoseconds()) / 1e6,
		GPUComputeMS:          float64(gpuDur.Nanoseconds()) / 1e6,
		ResultReadbackMS:      float64(readbackDur.Nanoseconds()) / 1e6,
		TotalMS:               float64(totalDur.Nanoseconds()) / 1e6,
		Synchronizations:      syncsPaid,
		AmortizedSyncPerLayer: float64(syncsPaid) / float64(d.cfg.NumLayers),
		NumLayers:             d.cfg.NumLayers,
		Fallback:              false,
	}
	d.lastProfile = profile

	return logits, profile, nil
}

func (d *Qwen35ResidentMetalDecoder) executeFallback(id, pos int, reason string) ([]float32, Qwen35ResidentStageProfile, error) {
	d.fallbackActive = true
	d.fallbackReason = reason
	t0 := time.Now()
	out := d.s.token(id, pos)
	tot := time.Since(t0)
	prof := Qwen35ResidentStageProfile{
		ResultReadback:        tot,
		Total:                 tot,
		ResultReadbackMS:      float64(tot.Nanoseconds()) / 1e6,
		TotalMS:               float64(tot.Nanoseconds()) / 1e6,
		Synchronizations:      1,
		AmortizedSyncPerLayer: 1.0 / float64(d.cfg.NumLayers),
		NumLayers:             d.cfg.NumLayers,
		Fallback:              true,
		FallbackReason:        reason,
	}
	d.lastProfile = prof
	return out, prof, nil
}

func (d *Qwen35ResidentMetalDecoder) syncLinearFromHost() error {
	if d.s == nil || d.s.Cache == nil || d.s.Cache.linear == nil {
		return nil
	}
	keep := d.cfg.LinearConvKernelDim - 1
	_, _, _, _, _, _, convDim := d.cfg.linearAttnDims()
	for l, lin := range d.linearLayers {
		cpuLayer := d.s.Cache.linear.layer(d.cfg, l)
		if len(cpuLayer.conv) > 0 {
			convFlat := FlattenLinearConvState(cpuLayer, keep, convDim)
			recFlat := FlattenLinearRecurrentState(cpuLayer)
			if len(convFlat) > 0 && len(recFlat) > 0 {
				_ = lin.state.Seed(convFlat, recFlat)
			}
		}
	}
	return nil
}

func (d *Qwen35ResidentMetalDecoder) syncLinearToHost() error {
	if d.s == nil || d.s.Cache == nil || d.s.Cache.linear == nil {
		return nil
	}
	keep := d.cfg.LinearConvKernelDim - 1
	_, nV, kHd, vHd, _, _, convDim := d.cfg.linearAttnDims()
	for l, lin := range d.linearLayers {
		conv, rec, err := lin.state.Snapshot()
		if err != nil {
			continue
		}
		cpuLayer := d.s.Cache.linear.layer(d.cfg, l)
		if len(cpuLayer.conv) != keep {
			cpuLayer.conv = make([][]float32, keep)
			for r := range cpuLayer.conv {
				cpuLayer.conv[r] = make([]float32, convDim)
			}
		}
		for r := 0; r < keep; r++ {
			copy(cpuLayer.conv[r], conv[r*convDim:(r+1)*convDim])
		}
		if len(cpuLayer.recurrent) != nV {
			cpuLayer.recurrent = make([][]float32, nV)
			for h := range cpuLayer.recurrent {
				cpuLayer.recurrent[h] = make([]float32, kHd*vHd)
			}
		}
		for h := 0; h < nV; h++ {
			copy(cpuLayer.recurrent[h], rec[h*kHd*vHd:(h+1)*kHd*vHd])
		}
	}
	return nil
}

// VerifyCPUReference compares this resident decoder's logits against the CPU reference
// forward for a single token, returning typed cosine similarity and greedy token match.
func (d *Qwen35ResidentMetalDecoder) VerifyCPUReference(id, pos int) (Qwen35ResidentCPUCheck, error) {
	// Create an isolated reference session to obtain unperturbed CPU reference logits
	refSession := d.m.NewSession()
	refSession.Q4K = true
	defer refSession.Close()

	// Clone current cache state into refSession
	if d.s.Cache != nil {
		refSession.Cache = d.s.Cache.Clone()
	}

	wantLogits := refSession.token(id, pos)
	gotLogits, _, err := d.DecodeToken(id, pos)
	if err != nil {
		return Qwen35ResidentCPUCheck{}, err
	}

	cos, maxDiff, err := CosineSimilarityAndMaxAbs(wantLogits, gotLogits)
	if err != nil {
		return Qwen35ResidentCPUCheck{}, err
	}

	wantGreedy := greedyArgmax(wantLogits)
	gotGreedy := greedyArgmax(gotLogits)
	greedyMatch := (wantGreedy == gotGreedy)
	passed := (cos >= 0.9999) && greedyMatch

	return Qwen35ResidentCPUCheck{
		CosineSimilarity: cos,
		MaxAbsDiff:       maxDiff,
		GreedyMatch:      greedyMatch,
		ExpectedToken:    wantGreedy,
		ActualToken:      gotGreedy,
		Passed:           passed,
	}, nil
}

// VerifyCPUReferenceSequence runs CPU reference comparison checks across a multi-token sequence.
func (d *Qwen35ResidentMetalDecoder) VerifyCPUReferenceSequence(tokens []int) ([]Qwen35ResidentCPUCheck, error) {
	verdicts := make([]Qwen35ResidentCPUCheck, len(tokens))
	for i, tok := range tokens {
		v, err := d.VerifyCPUReference(tok, d.s.Cache.Len())
		if err != nil {
			return nil, fmt.Errorf("probe step %d (token %d): %w", i, tok, err)
		}
		verdicts[i] = v
	}
	return verdicts, nil
}

func greedyArgmax(v []float32) int {
	if len(v) == 0 {
		return -1
	}
	best := 0
	maxVal := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > maxVal {
			maxVal = v[i]
			best = i
		}
	}
	return best
}
