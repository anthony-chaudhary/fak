//go:build darwin && arm64 && cgo

package model

// metal_prefill_hybrid.go — the Metal GPU twin of the Qwen3.6 hybrid (Gated-DeltaNet) prefill.
// Built by default on Apple Silicon with cgo. It is deliberately thin: the entire prefill body — both
// RMSNorms, the conv1d+SiLU mixer, the q/k L2-norm, the per-head delta-rule recurrent scan, the
// gated RMSNorm readout, the full-attention RoPE/GQA/output-gate, and every residual — lives in
// the backend-agnostic core prefillQwen35HybridViaMM (metal_prefill_hybrid_core.go), proven
// host-independently against the CPU template by TestQwen35HybridViaMMMatchesCPUTemplate. This
// file supplies only the one substitution that core abstracts: a GPU f16 GEMM for the projection
// /MLP matmuls. Keeping the GDN recurrence on the CPU and moving just the projections to the
// device is the measured lever (the projections are the prefill wall; the GDN scan is ~0.5%;
// #65, #977), and lifting requirePreNorm("Metal prefill") for the hybrid is what lets it use the
// Metal prefill at all (#71).
//
// Weights: like metalWeights(), the GPU holds an f16 copy of each projection, dequantized once
// from the Q8_0 store and cached per *Model. The hybrid's projection set is per-layer: every
// layer carries the three MLP matmuls, while the per-layer mixer is EITHER the five linear_attn
// projections (linear_attention layers) OR the four self_attn projections (full_attention
// layers), dispatched by isLinearAttnLayer — the same split the core's mm calls walk.

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

var (
	metalHybridMu           sync.Mutex
	metalHybridWt           = map[*Model]map[string]*metalgemm.Weight{} // per-Model name -> GPU f16 weight
	metalHybridMPSAvailable = metalgemm.MPSAvailable
)

type metalQwen35GDNSequenceBackend struct {
	mu                              sync.Mutex
	states                          map[Qwen35GDNAuxState]*metalgemm.GDNState
	forwardReceipt                  Qwen35MetalForwardSequenceReceipt
	forwardRan                      bool
	injectForwardPostSubmitFailure  bool
	injectMTPPanelPostSubmitFailure bool
}

const (
	Qwen35MetalMTPVerifyPanelSchema = "fak.qwen35-metal-mtp-p4-verify/1"
	Qwen35MetalMTPVerifyPanelPath   = "fak-native/metal/qwen3.8-mtp-p4-target-verify-v1"
	Qwen35MetalMTPStateDigestDomain = "fak.qwen35-metal-mtp-p4-state/v1\x00"
	qwen35MetalMTPVerifyPanelTokens = 4
)

// Qwen35MetalMTPKVPanel is one full-attention layer's P=4 cache suffix.
type Qwen35MetalMTPKVPanel struct {
	Layer          int
	KRaw, KPost, V []float32
}

// Qwen35MetalMTPVerifyPanelResult returns every terminal value needed to adopt
// the speculative panel plus the independent owner needed to undo it.
type Qwen35MetalMTPVerifyPanelResult struct {
	Logits     [][]float32
	RawHidden  [][]float32
	KV         []Qwen35MetalMTPKVPanel
	Checkpoint Qwen35MetalMTPCheckpoint
	Receipt    Qwen35MetalMTPVerifyPanelReceipt
}

type metalQwen35MTPCheckpoint struct {
	mu          sync.Mutex
	checkpoints []*metalgemm.GDNGraphCheckpoint
	backups     []*metalgemm.GDNState
	backupLayer map[int]*metalgemm.GDNState
	restored    bool
	closed      bool
	receipt     Qwen35MetalMTPVerifyPanelReceipt
}

func (c *metalQwen35MTPCheckpoint) add(checkpoint *metalgemm.GDNGraphCheckpoint) {
	c.checkpoints = append(c.checkpoints, checkpoint)
}

func (c *metalQwen35MTPCheckpoint) setReceipt(receipt Qwen35MetalMTPVerifyPanelReceipt) {
	c.mu.Lock()
	defer c.mu.Unlock()
	receipt.GDNCheckpointLayers = append([]int(nil), receipt.GDNCheckpointLayers...)
	receipt.GDNCheckpointLineageSHA256 = append([]string(nil), receipt.GDNCheckpointLineageSHA256...)
	c.receipt = receipt
}

func (c *metalQwen35MTPCheckpoint) Qwen35MetalMTPPanelReceipt() Qwen35MetalMTPVerifyPanelReceipt {
	c.mu.Lock()
	defer c.mu.Unlock()
	receipt := c.receipt
	receipt.GDNCheckpointLayers = append([]int(nil), receipt.GDNCheckpointLayers...)
	receipt.GDNCheckpointLineageSHA256 = append([]string(nil), receipt.GDNCheckpointLineageSHA256...)
	return receipt
}

func (c *metalQwen35MTPCheckpoint) Restore() error {
	if c == nil {
		return errors.New("model: missing Metal MTP checkpoint owner")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return errors.New("model: Metal MTP checkpoint owner is closed")
	}
	if c.restored {
		return errors.New("model: Metal MTP checkpoint owner already restored")
	}
	var restoreErr error
	for i := len(c.checkpoints) - 1; i >= 0; i-- {
		restoreErr = errors.Join(restoreErr, c.checkpoints[i].Restore())
	}
	if restoreErr == nil {
		c.restored = true
	}
	return restoreErr
}

func (c *metalQwen35MTPCheckpoint) Close() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return
	}
	for i := len(c.checkpoints) - 1; i >= 0; i-- {
		c.checkpoints[i].Close()
	}
	for i := len(c.backups) - 1; i >= 0; i-- {
		c.backups[i].Close()
	}
	c.checkpoints = nil
	c.backups = nil
	c.backupLayer = nil
	c.closed = true
}

type qwen35MetalGraphKVResult struct {
	layer          int
	kraw, kpost, v *metalgemm.GraphResult
}

type qwen35MetalMTPOperations struct {
	q6Down, q6Head int
}

func (b *metalQwen35GDNSequenceBackend) Qwen35MetalForwardSequenceReceipt() (Qwen35MetalForwardSequenceReceipt, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	receipt := b.forwardReceipt
	if receipt.StateIdentity != nil {
		identity := cloneQwen35MetalStateIdentityReceipt(*receipt.StateIdentity)
		receipt.StateIdentity = &identity
	}
	return receipt, b.forwardRan
}

func (b *metalQwen35GDNSequenceBackend) setForwardReceipt(r Qwen35MetalForwardSequenceReceipt) {
	b.mu.Lock()
	if r.StateIdentity != nil {
		identity := cloneQwen35MetalStateIdentityReceipt(*r.StateIdentity)
		r.StateIdentity = &identity
	}
	b.forwardReceipt = r
	b.forwardRan = true
	b.mu.Unlock()
}

func (b *metalQwen35GDNSequenceBackend) bindQwen35MetalStateIdentity(identity Qwen35MetalStateIdentityReceipt) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.forwardRan || b.forwardReceipt.StateIdentity != nil {
		return
	}
	cloned := cloneQwen35MetalStateIdentityReceipt(identity)
	b.forwardReceipt.StateIdentity = &cloned
	// FinishRead accounts the graph's one terminal pack. The GDN finalizer's
	// existing snapshot/seed transfers are outside GraphReceipt, so fold their
	// exact bytes into the model-level total only after every operation succeeds.
	b.forwardReceipt.HostReadbackBytes += identity.GDNStateD2HBytes
	b.forwardReceipt.HostUploadBytes += identity.GDNStateH2DBytes
}

func qwen35GraphProjection(g *metalgemm.ProjectionGraph, s *Session, name string, input *metalgemm.GraphResult, quantized map[*metalgemm.GraphResult]*metalgemm.QuantizedGraphResult) (*metalgemm.GraphResult, error) {
	m := s.M
	if qt := m.q4kw[name]; qt != nil {
		w := m.metalQ4KWeight(name, qt)
		if w == nil {
			return nil, fmt.Errorf("metalgemm: Q4_K graph weight unavailable: %s", name)
		}
		return g.EncodeQ4KFrom(w, input)
	}
	if qt := m.kqw[name]; qt != nil {
		w := m.metalQ6KWeight(name, qt)
		if w == nil {
			return nil, fmt.Errorf("metalgemm: Q6_K graph weight unavailable: %s", name)
		}
		return g.EncodeQ6KFrom(w, input)
	}
	qt := m.q8w[name]
	if qt == nil {
		return nil, fmt.Errorf("metalgemm: Q8 graph weight missing: %s", name)
	}
	w := m.metalQ8Weight(name, qt)
	if w == nil {
		return nil, fmt.Errorf("metalgemm: Q8 graph weight unavailable: %s", name)
	}
	q := quantized[input]
	if q == nil {
		var err error
		q, err = g.QuantizeQ8(input)
		if err != nil {
			return nil, err
		}
		quantized[input] = q
	}
	return g.EncodeQ8From(w, q)
}

func qwen35GraphProjections(g *metalgemm.ProjectionGraph, s *Session, names []string, input *metalgemm.GraphResult, quantized map[*metalgemm.GraphResult]*metalgemm.QuantizedGraphResult) ([]*metalgemm.GraphResult, error) {
	out := make([]*metalgemm.GraphResult, len(names))
	for i, name := range names {
		var err error
		out[i], err = qwen35GraphProjection(g, s, name, input, quantized)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

type qwen35MetalMTPHead struct {
	q4 *metalgemm.Q4KWeight
	q6 *metalgemm.Q6KWeight
	q8 *metalgemm.Q8Weight
}

func resolveQwen35MetalMTPHead(m *Model) qwen35MetalMTPHead {
	name := m.q4kHeadName()
	q4 := m.q4khead
	if q4 == nil {
		q4 = m.q4kw[name]
	}
	if q4 != nil {
		return qwen35MetalMTPHead{q4: m.metalQ4KWeight(name, q4)}
	}
	if name = m.kqHeadName(); name != "" {
		if q := m.kqw[name]; q != nil && q.kind == kindQ6K {
			return qwen35MetalMTPHead{q6: m.metalQ6KWeight(name, q)}
		}
		return qwen35MetalMTPHead{}
	}
	name = m.headName()
	q8 := m.q8head
	if q8 == nil {
		q8 = m.q8w[name]
	}
	if q8 != nil {
		return qwen35MetalMTPHead{q8: m.metalQ8Weight(name, q8)}
	}
	return qwen35MetalMTPHead{}
}

func (h qwen35MetalMTPHead) valid() bool { return h.q4 != nil || h.q6 != nil || h.q8 != nil }

func (h qwen35MetalMTPHead) encode(g *metalgemm.ProjectionGraph, input *metalgemm.GraphResult, quantized map[*metalgemm.GraphResult]*metalgemm.QuantizedGraphResult) (*metalgemm.GraphResult, error) {
	switch {
	case h.q4 != nil:
		return g.EncodeQ4KFrom(h.q4, input)
	case h.q6 != nil:
		return g.EncodeQ6KFrom(h.q6, input)
	case h.q8 != nil:
		q := quantized[input]
		if q == nil {
			var err error
			q, err = g.QuantizeQ8(input)
			if err != nil {
				return nil, err
			}
			quantized[input] = q
		}
		return g.EncodeQ8From(h.q8, q)
	default:
		return nil, errors.New("metalgemm: resident Qwen MTP LM head unavailable")
	}
}

func qwen35MetalMTPPanelGeometryError(cfg Config) error {
	if !cfg.IsQwen35Hybrid() || cfg.NumExperts != 0 || cfg.AttentionBias || cfg.LayerNorm || cfg.QKNormPerHeadWeight ||
		cfg.HiddenSize <= 0 || cfg.HiddenSize%32 != 0 || cfg.IntermediateSize <= 0 || cfg.IntermediateSize%32 != 0 ||
		cfg.HeadDim < 2 || cfg.HeadDim > 256 || cfg.HeadDim%2 != 0 || cfg.NumHeads <= 0 || cfg.NumKVHeads <= 0 ||
		cfg.NumHeads%cfg.NumKVHeads != 0 || cfg.BlockTopology != PreNorm || !cfg.AttnOutputGate || !cfg.NormGain1p {
		return fmt.Errorf("metalgemm: unsupported exact P4 Qwen3.8 MTP geometry")
	}
	return nil
}

func qwen35MetalMTPPanelProjectionNames(cfg Config) []string {
	names := make([]string, 0, 8*cfg.NumLayers)
	for l := 0; l < cfg.NumLayers; l++ {
		p := func(suffix string) string { return layerName(l, suffix) }
		if cfg.isLinearAttnLayer(l) {
			names = append(names, p("linear_attn.in_proj_qkv.weight"), p("linear_attn.in_proj_z.weight"), p("linear_attn.in_proj_b.weight"), p("linear_attn.in_proj_a.weight"), p("linear_attn.out_proj.weight"))
		} else {
			names = append(names, p("self_attn.q_proj.weight"), p("self_attn.k_proj.weight"), p("self_attn.v_proj.weight"), p("self_attn.o_proj.weight"))
		}
		names = append(names, p("mlp.gate_proj.weight"), p("mlp.up_proj.weight"), p("mlp.down_proj.weight"))
	}
	return names
}

func qwen35MetalMTPProjectionReady(m *Model, name string) bool {
	if q := m.q4kw[name]; q != nil {
		return m.metalQ4KWeight(name, q) != nil
	}
	if q := m.kqw[name]; q != nil {
		return q.kind == kindQ6K && m.metalQ6KWeight(name, q) != nil
	}
	if q := m.q8w[name]; q != nil {
		return m.metalQ8Weight(name, q) != nil
	}
	return false
}

func qwen35MetalMTPPanelAdmission(s *Session, ids []int) (*metalQwen35GDNSequenceBackend, qwen35MetalMTPHead, bool) {
	if s == nil || s.M == nil || s.Cache == nil || len(ids) != qwen35MetalMTPVerifyPanelTokens ||
		s.Backend != nil || !s.Q4K || !s.MetalQ4K || s.qwen35HAL == nil || !s.qwen35HAL.decodeAccepted ||
		s.qwen35HAL.decodePath != Qwen35MetalGDNDecodeForwardPath || s.activeTap() != nil {
		return nil, qwen35MetalMTPHead{}, false
	}
	if qwen35MetalMTPPanelGeometryError(s.M.Cfg) != nil || s.Cache.Len()+len(ids) > 4096 {
		return nil, qwen35MetalMTPHead{}, false
	}
	for _, id := range ids {
		if id < 0 || id >= s.M.Cfg.VocabSize {
			return nil, qwen35MetalMTPHead{}, false
		}
	}
	b, ok := s.qwen35HAL.sequenceBackend.(*metalQwen35GDNSequenceBackend)
	if !ok {
		return nil, qwen35MetalMTPHead{}, false
	}
	s.prefillQwen35HybridQ4KMetalUpload()
	for _, name := range qwen35MetalMTPPanelProjectionNames(s.M.Cfg) {
		if !qwen35MetalMTPProjectionReady(s.M, name) {
			return nil, qwen35MetalMTPHead{}, false
		}
	}
	head := resolveQwen35MetalMTPHead(s.M)
	if !head.valid() {
		return nil, qwen35MetalMTPHead{}, false
	}
	return b, head, true
}

func newMetalQwen35MTPCheckpoint(s *Session) (*metalQwen35MTPCheckpoint, error) {
	owner := &metalQwen35MTPCheckpoint{backupLayer: make(map[int]*metalgemm.GDNState)}
	geometry := metalQwen35GDNGeometry(s.qwen35GDNSequenceGeometry())
	for layer := 0; layer < s.M.Cfg.NumLayers; layer++ {
		if !s.M.Cfg.isLinearAttnLayer(layer) {
			continue
		}
		backup, err := metalgemm.NewGDNState(geometry)
		if err != nil {
			owner.Close()
			return nil, fmt.Errorf("metalgemm: allocate MTP GDN backup layer %d: %w", layer, err)
		}
		owner.backups = append(owner.backups, backup)
		owner.backupLayer[layer] = backup
	}
	return owner, nil
}

func qwen35MetalForwardGeometryError(cfg Config) error {
	if cfg.NumExperts != 0 || cfg.AttentionBias || cfg.LayerNorm || cfg.QKNormPerHeadWeight ||
		cfg.HiddenSize <= 0 || cfg.HiddenSize%32 != 0 || cfg.IntermediateSize <= 0 || cfg.IntermediateSize%32 != 0 ||
		cfg.HeadDim < 2 || cfg.HeadDim > 256 || cfg.HeadDim%2 != 0 || cfg.NumHeads <= 0 || cfg.NumKVHeads <= 0 ||
		cfg.NumHeads%cfg.NumKVHeads != 0 {
		return fmt.Errorf("metalgemm: unsupported exact P32 Qwen hybrid geometry")
	}
	return nil
}

func (b *metalQwen35GDNSequenceBackend) encodeQwen35MetalMTPP4(g *metalgemm.ProjectionGraph, s *Session, x *metalgemm.GraphResult, owner *metalQwen35MTPCheckpoint, base int, head qwen35MetalMTPHead, operations *qwen35MetalMTPOperations) (logits, rawHidden *metalgemm.GraphResult, kvResults []qwen35MetalGraphKVResult, err error) {
	m, cfg := s.M, s.M.Cfg
	const P = qwen35MetalMTPVerifyPanelTokens
	quantized := make(map[*metalgemm.GraphResult]*metalgemm.QuantizedGraphResult)
	eps := float32(cfg.RMSNormEps)
	for l := 0; l < cfg.NumLayers; l++ {
		p := func(suffix string) string { return layerName(l, suffix) }
		xn, runErr := g.RMSNorm(x, m.tensor(p("input_layernorm.weight")), eps, cfg.NormGain1p)
		if runErr != nil {
			return nil, nil, nil, runErr
		}
		var attnOut *metalgemm.GraphResult
		if cfg.isLinearAttnLayer(l) {
			in, runErr := qwen35GraphProjections(g, s, []string{
				p("linear_attn.in_proj_qkv.weight"), p("linear_attn.in_proj_z.weight"),
				p("linear_attn.in_proj_b.weight"), p("linear_attn.in_proj_a.weight"),
			}, xn, quantized)
			if runErr != nil {
				return nil, nil, nil, runErr
			}
			live := b.state(s.qwen35HAL.sequenceLayers[l])
			backup := owner.backupLayer[l]
			if live == nil || backup == nil {
				return nil, nil, nil, fmt.Errorf("metalgemm: missing MTP GDN checkpoint owner for layer %d", l)
			}
			checkpoint, runErr := g.CheckpointGDN(live, backup)
			if runErr != nil {
				return nil, nil, nil, runErr
			}
			owner.add(checkpoint)
			core, runErr := g.GDN(live, in[0], in[1], in[2], in[3], metalgemm.GDNPanel{
				Conv1D: m.tensor(p("linear_attn.conv1d.weight")), ALog: m.tensor(p("linear_attn.A_log")),
				DTBias: m.tensor(p("linear_attn.dt_bias")), Norm: m.tensor(p("linear_attn.norm.weight")), RMSNormEpsilon: eps,
			})
			if runErr != nil {
				return nil, nil, nil, runErr
			}
			attnOut, runErr = qwen35GraphProjection(g, s, p("linear_attn.out_proj.weight"), core, quantized)
			if runErr != nil {
				return nil, nil, nil, runErr
			}
		} else {
			qkv, runErr := qwen35GraphProjections(g, s, []string{p("self_attn.q_proj.weight"), p("self_attn.k_proj.weight"), p("self_attn.v_proj.weight")}, xn, quantized)
			if runErr != nil {
				return nil, nil, nil, runErr
			}
			q, gate, runErr := g.SplitGatedQ(qkv[0], cfg.NumHeads*cfg.HeadDim, cfg.HeadDim)
			if runErr != nil {
				return nil, nil, nil, runErr
			}
			qnorm, knorm := make([]float32, cfg.HeadDim), make([]float32, cfg.HeadDim)
			if cfg.QKNorm {
				qnorm = m.tensor(p("self_attn.q_norm.weight"))
				knorm = m.tensor(p("self_attn.k_norm.weight"))
				qwidth, kvwidth := cfg.NumHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim
				if (len(qnorm) != cfg.HeadDim && len(qnorm) != qwidth) || (len(knorm) != cfg.HeadDim && len(knorm) != kvwidth) {
					return nil, nil, nil, fmt.Errorf("metalgemm: P4 Qwen graph requires shared or projection-wide per-head Q/K norm weights")
				}
			}
			rotary := cfg.rotaryDim()
			cosv, sinv := make([]float32, 0, P*(rotary/2)), make([]float32, 0, P*(rotary/2))
			for pos := 0; pos < P; pos++ {
				c, si := ropeRowForLayer(cfg, l, base+pos)
				cosv, sinv = append(cosv, c...), append(sinv, si...)
			}
			attention, runErr := g.FullAttention(q, qkv[1], qkv[2], gate, qnorm, knorm, cosv, sinv,
				s.Cache.K[l], s.Cache.V[l], base, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, rotary, cfg.attnScale(), cfg.qkNormEps(), cfg.NormGain1p, cfg.QKNorm)
			if runErr != nil {
				return nil, nil, nil, runErr
			}
			attnOut, runErr = qwen35GraphProjection(g, s, p("self_attn.o_proj.weight"), attention.Output, quantized)
			if runErr != nil {
				return nil, nil, nil, runErr
			}
			kvResults = append(kvResults, qwen35MetalGraphKVResult{layer: l, kraw: attention.KRaw, kpost: attention.KPost, v: attention.V})
		}
		if err = g.AddInPlace(x, attnOut); err != nil {
			return nil, nil, nil, err
		}
		xn2, runErr := g.RMSNorm(x, m.tensor(p("post_attention_layernorm.weight")), eps, cfg.NormGain1p)
		if runErr != nil {
			return nil, nil, nil, runErr
		}
		gu, runErr := qwen35GraphProjections(g, s, []string{p("mlp.gate_proj.weight"), p("mlp.up_proj.weight")}, xn2, quantized)
		if runErr != nil {
			return nil, nil, nil, runErr
		}
		if err = g.SwiGLUInPlace(gu[0], gu[1]); err != nil {
			return nil, nil, nil, err
		}
		down, runErr := qwen35GraphProjection(g, s, p("mlp.down_proj.weight"), gu[0], quantized)
		if runErr != nil {
			return nil, nil, nil, runErr
		}
		if q := m.kqw[p("mlp.down_proj.weight")]; q != nil && q.kind == kindQ6K {
			operations.q6Down++
		}
		if err = g.AddInPlace(x, down); err != nil {
			return nil, nil, nil, err
		}
	}
	normalized, err := g.RMSNorm(x, m.tensor("model.norm.weight"), eps, cfg.NormGain1p)
	if err != nil {
		return nil, nil, nil, err
	}
	logits, err = head.encode(g, normalized, quantized)
	if err == nil && head.q6 != nil {
		operations.q6Head++
	}
	return logits, x, kvResults, err
}

// Qwen35MetalMTPVerifyPanel executes the exact production P=4 target panel.
// accepted becomes true before BeginProjectionGraph: after selection no error
// may fall through to the f32 verifier or ordinary target decode.
func (s *Session) Qwen35MetalMTPVerifyPanel(ids []int) (result Qwen35MetalMTPVerifyPanelResult, accepted bool, err error) {
	b, head, admitted := qwen35MetalMTPPanelAdmission(s, ids)
	if !admitted {
		return result, false, nil
	}
	accepted = true
	owner, err := newMetalQwen35MTPCheckpoint(s)
	if err != nil {
		return result, true, err
	}
	m, cfg := s.M, s.M.Cfg
	base, H := s.Cache.Len(), cfg.HiddenSize
	X := make([]float32, len(ids)*H)
	embed := m.embedRows()
	for i, id := range ids {
		copy(X[i*H:(i+1)*H], embed[id*H:(id+1)*H])
		scaleEmbedInPlace(X[i*H:(i+1)*H], cfg)
	}
	g, err := metalgemm.BeginProjectionGraph(X, nil, nil, len(ids), H)
	if err != nil {
		owner.Close()
		return result, true, err
	}
	if b.injectMTPPanelPostSubmitFailure {
		g.InjectPostSubmitFailureForTest()
	}
	x, err := g.Input(H)
	if err != nil {
		g.Free()
		owner.Close()
		return result, true, err
	}
	operations := &qwen35MetalMTPOperations{}
	logitsResult, hiddenResult, kvResults, err := b.encodeQwen35MetalMTPP4(g, s, x, owner, base, head, operations)
	if err != nil {
		g.Free()
		owner.Close()
		return result, true, err
	}
	terminal := []*metalgemm.GraphResult{logitsResult, hiddenResult}
	for _, kv := range kvResults {
		terminal = append(terminal, kv.kraw, kv.kpost, kv.v)
	}
	outputs, graphReceipt, finishErr := g.FinishRead(terminal...)
	g.Free()
	receipt := qwen35MetalMTPVerifyPanelReceipt(len(ids), base, graphReceipt, owner, operations)
	result.Receipt = receipt
	owner.setReceipt(receipt)
	for _, lineageIdentity := range receipt.GDNCheckpointLineageSHA256 {
		if len(lineageIdentity) != sha256.Size*2 {
			result.Checkpoint = owner
			return result, true, errors.New("metalgemm: completed MTP GDN checkpoint omitted lineage identity")
		}
	}
	if finishErr != nil || !graphReceipt.Committed || !graphReceipt.CompletedWait || graphReceipt.HostReadbacks != 1 {
		if !graphReceipt.Committed || !graphReceipt.CompletedWait {
			owner.Close()
		} else {
			result.Checkpoint = owner
		}
		if finishErr == nil {
			finishErr = fmt.Errorf("metalgemm: incomplete Qwen MTP P4 graph receipt: %+v", graphReceipt)
		}
		return result, true, finishErr
	}
	result.Logits = splitScaledLogits(nil, outputs[0], len(ids), cfg.VocabSize, cfg)
	result.RawHidden = splitQwen35MetalMTPRows(outputs[1], len(ids), H)
	outIndex := 2
	for _, kv := range kvResults {
		panel := Qwen35MetalMTPKVPanel{Layer: kv.layer, KRaw: outputs[outIndex], KPost: outputs[outIndex+1], V: outputs[outIndex+2]}
		result.KV = append(result.KV, panel)
		s.Cache.Kraw[kv.layer] = append(s.Cache.Kraw[kv.layer], panel.KRaw...)
		s.Cache.K[kv.layer] = append(s.Cache.K[kv.layer], panel.KPost...)
		s.Cache.V[kv.layer] = append(s.Cache.V[kv.layer], panel.V...)
		outIndex += 3
	}
	for row, id := range ids {
		s.Cache.appendPosition(base+row, id)
		s.rememberTargetHidden(base+row, id, result.RawHidden[row])
	}
	result.Receipt.StateSHA256 = qwen35MetalMTPStateDigest(base, ids, outputs[1], result.KV)
	result.Receipt.TransactionSHA256 = qwen35MetalMTPTransactionDigest(result.Receipt.StateSHA256, result.Receipt.GDNCheckpointBindingSHA256)
	owner.setReceipt(result.Receipt)
	result.Checkpoint = owner
	return result, true, nil
}

func qwen35MetalMTPVerifyPanelReceipt(tokens, base int, graphReceipt metalgemm.GraphReceipt, owner *metalQwen35MTPCheckpoint, operationReceipts ...*qwen35MetalMTPOperations) Qwen35MetalMTPVerifyPanelReceipt {
	operations := &qwen35MetalMTPOperations{}
	if len(operationReceipts) > 0 && operationReceipts[0] != nil {
		operations = operationReceipts[0]
	}
	layers, lineageIdentities, checkpointBinding := qwen35MetalMTPCheckpointBinding(base, owner)
	receipt := Qwen35MetalMTPVerifyPanelReceipt{
		Schema: Qwen35MetalMTPVerifyPanelSchema, Path: Qwen35MetalMTPVerifyPanelPath,
		StateDigestDomain: Qwen35MetalMTPStateDigestDomain, Tokens: tokens, Base: base,
		GDNCheckpointBindingSHA256: checkpointBinding, GDNCheckpointLayers: layers, GDNCheckpointLineageSHA256: lineageIdentities,
		CommandBuffers: 1, Encoders: graphReceipt.Encoders, CheckpointLayers: len(owner.checkpoints),
		IntermediateWaits: graphReceipt.IntermediateWaits, IntermediateReadbacks: graphReceipt.IntermediateReadbacks,
		TerminalWaits: qwen35MetalMTPObservedCount(graphReceipt.CompletedWait), TerminalReadbacks: graphReceipt.HostReadbacks,
		HostUploadBytes: graphReceipt.HostUploadBytes, HostReadbackBytes: graphReceipt.HostReadbackBytes,
		Committed: graphReceipt.Committed, CompletedWait: graphReceipt.CompletedWait, TimingAvailable: graphReceipt.TimingAvailable,
		GPUMilliseconds: graphReceipt.GPUMilliseconds, WaitMilliseconds: graphReceipt.WaitMilliseconds,
		Q6KDownProjectionOperations: operations.q6Down, Q6KHeadOperations: operations.q6Head,
	}
	if graphReceipt.Committed && graphReceipt.CompletedWait {
		for _, checkpoint := range owner.checkpoints {
			cr := checkpoint.Receipt()
			receipt.DeviceCheckpointCopies += cr.DeviceCopies
			receipt.CheckpointBufferSwaps += cr.BufferSwaps
			receipt.HostStateUploads += cr.HostStateUploads
			receipt.HostStateReadbacks += cr.HostStateReadbacks
		}
	}
	return receipt
}

// qwen35MetalMTPCheckpointBinding aggregates checkpoint lineage: native owner,
// mutation version, checkpoint generation, and covered layer/copy accounting.
// Recurrent and convolution buffer byte contents are intentionally not hashed.
func qwen35MetalMTPCheckpointBinding(base int, owner *metalQwen35MTPCheckpoint) ([]int, []string, string) {
	layers := make([]int, 0, len(owner.backupLayer))
	for layer := range owner.backupLayer {
		layers = append(layers, layer)
	}
	sort.Ints(layers)
	h := sha256.New()
	_, _ = h.Write([]byte("fak.qwen35-metal-mtp-p4-gdn-checkpoint/v1\x00"))
	qwen35MetalMTPDigestUint64(h, uint64(base))
	qwen35MetalMTPDigestUint64(h, uint64(len(layers)))
	lineageIdentities := make([]string, 0, len(layers))
	for i, layer := range layers {
		qwen35MetalMTPDigestUint64(h, uint64(layer))
		if i < len(owner.checkpoints) {
			receipt := owner.checkpoints[i].Receipt()
			lineageIdentity, identityErr := owner.checkpoints[i].StateIdentity()
			if identityErr != nil {
				lineageIdentity = ""
			}
			lineageIdentities = append(lineageIdentities, lineageIdentity)
			_, _ = h.Write([]byte(lineageIdentity))
			qwen35MetalMTPDigestUint64(h, uint64(receipt.DeviceCopies))
			qwen35MetalMTPDigestUint64(h, uint64(receipt.BufferSwaps))
			qwen35MetalMTPDigestUint64(h, uint64(receipt.HostStateUploads))
			qwen35MetalMTPDigestUint64(h, uint64(receipt.HostStateReadbacks))
		}
	}
	return layers, lineageIdentities, hex.EncodeToString(h.Sum(nil))
}

func qwen35MetalMTPTransactionDigest(state, checkpoint string) string {
	// state binds token, hidden, and KV content. checkpoint binds only GDN
	// checkpoint lineage; it makes no recurrent/conv byte-content claim.
	h := sha256.New()
	_, _ = h.Write([]byte("fak.qwen35-metal-mtp-p4-transaction/v1\x00"))
	_, _ = h.Write([]byte(state))
	_, _ = h.Write([]byte(checkpoint))
	return hex.EncodeToString(h.Sum(nil))
}

func qwen35MetalMTPObservedCount(observed bool) int {
	if observed {
		return 1
	}
	return 0
}

func splitQwen35MetalMTPRows(flat []float32, rows, width int) [][]float32 {
	out := make([][]float32, rows)
	for row := range out {
		out[row] = flat[row*width : (row+1)*width]
	}
	return out
}

func qwen35MetalMTPStateDigest(base int, ids []int, rawHidden []float32, kv []Qwen35MetalMTPKVPanel) string {
	h := sha256.New()
	_, _ = h.Write([]byte(Qwen35MetalMTPStateDigestDomain))
	qwen35MetalMTPDigestUint64(h, uint64(base))
	qwen35MetalMTPDigestUint64(h, uint64(len(ids)))
	for _, id := range ids {
		qwen35MetalMTPDigestUint64(h, uint64(id))
	}
	qwen35MetalMTPDigestFloats(h, rawHidden)
	for _, panel := range kv {
		qwen35MetalMTPDigestUint64(h, uint64(panel.Layer))
		qwen35MetalMTPDigestFloats(h, panel.KRaw)
		qwen35MetalMTPDigestFloats(h, panel.KPost)
		qwen35MetalMTPDigestFloats(h, panel.V)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func qwen35MetalMTPDigestUint64(h hash.Hash, value uint64) {
	var bits [8]byte
	binary.LittleEndian.PutUint64(bits[:], value)
	_, _ = h.Write(bits[:])
}

func qwen35MetalMTPDigestFloats(h hash.Hash, values []float32) {
	qwen35MetalMTPDigestUint64(h, uint64(len(values)))
	for _, value := range values {
		qwen35MetalMTPDigestUint64(h, uint64(math.Float32bits(value)))
	}
}

func (b *metalQwen35GDNSequenceBackend) Qwen35MetalForwardSequence(s *Session, ids []int) ([]float32, Qwen35MetalForwardSequenceReceipt, bool, error) {
	if s == nil || s.M == nil || s.Backend != nil || !s.Q4K || !s.MetalQ4K || len(ids) != 32 || s.qwen35HAL == nil || !s.qwen35HAL.sequenceAccepted {
		return nil, Qwen35MetalForwardSequenceReceipt{}, false, nil
	}
	m, cfg := s.M, s.M.Cfg
	if err := qwen35MetalForwardGeometryError(cfg); err != nil {
		return nil, Qwen35MetalForwardSequenceReceipt{}, true, err
	}
	base, H, P := s.Cache.Len(), cfg.HiddenSize, len(ids)
	if base+P > 4096 {
		return nil, Qwen35MetalForwardSequenceReceipt{}, true, fmt.Errorf("metalgemm: P32 graph attention context %d exceeds 4096", base+P)
	}
	// Resolve every resident handle before graph construction. Once Begin succeeds,
	// any failure remains accepted and cannot replay through the host forward.
	s.prefillQwen35HybridQ4KMetalUpload()
	embed := m.embedRows()
	X := make([]float32, P*H)
	for i, id := range ids {
		copy(X[i*H:(i+1)*H], embed[id*H:(id+1)*H])
		scaleEmbedInPlace(X[i*H:(i+1)*H], cfg)
	}
	g, err := metalgemm.BeginProjectionGraph(X, nil, nil, P, H)
	if err != nil {
		return nil, Qwen35MetalForwardSequenceReceipt{}, true, err
	}
	defer g.Free()
	if b.injectForwardPostSubmitFailure {
		g.InjectPostSubmitFailureForTest()
	}
	x, err := g.Input(H)
	if err != nil {
		return nil, Qwen35MetalForwardSequenceReceipt{}, true, err
	}
	quantized := make(map[*metalgemm.GraphResult]*metalgemm.QuantizedGraphResult)
	type kvResult struct {
		layer          int
		kraw, kpost, v *metalgemm.GraphResult
	}
	var kvResults []kvResult
	eps := float32(cfg.RMSNormEps)
	for l := 0; l < cfg.NumLayers; l++ {
		p := func(suffix string) string { return layerName(l, suffix) }
		xn, runErr := g.RMSNorm(x, m.tensor(p("input_layernorm.weight")), eps, cfg.NormGain1p)
		if runErr != nil {
			return nil, Qwen35MetalForwardSequenceReceipt{}, true, runErr
		}
		var attnOut *metalgemm.GraphResult
		if cfg.isLinearAttnLayer(l) {
			in, runErr := qwen35GraphProjections(g, s, []string{
				p("linear_attn.in_proj_qkv.weight"), p("linear_attn.in_proj_z.weight"),
				p("linear_attn.in_proj_b.weight"), p("linear_attn.in_proj_a.weight"),
			}, xn, quantized)
			if runErr != nil {
				return nil, Qwen35MetalForwardSequenceReceipt{}, true, runErr
			}
			state := b.state(s.qwen35HAL.sequenceLayers[l])
			if state == nil {
				return nil, Qwen35MetalForwardSequenceReceipt{}, true, fmt.Errorf("metalgemm: missing GDN graph owner for layer %d", l)
			}
			core, runErr := g.GDN(state, in[0], in[1], in[2], in[3], metalgemm.GDNPanel{
				Conv1D: m.tensor(p("linear_attn.conv1d.weight")), ALog: m.tensor(p("linear_attn.A_log")),
				DTBias: m.tensor(p("linear_attn.dt_bias")), Norm: m.tensor(p("linear_attn.norm.weight")), RMSNormEpsilon: eps,
			})
			if runErr != nil {
				return nil, Qwen35MetalForwardSequenceReceipt{}, true, runErr
			}
			attnOut, runErr = qwen35GraphProjection(g, s, p("linear_attn.out_proj.weight"), core, quantized)
			if runErr != nil {
				return nil, Qwen35MetalForwardSequenceReceipt{}, true, runErr
			}
		} else {
			qkv, runErr := qwen35GraphProjections(g, s, []string{p("self_attn.q_proj.weight"), p("self_attn.k_proj.weight"), p("self_attn.v_proj.weight")}, xn, quantized)
			if runErr != nil {
				return nil, Qwen35MetalForwardSequenceReceipt{}, true, runErr
			}
			q, gate, runErr := g.SplitGatedQ(qkv[0], cfg.NumHeads*cfg.HeadDim, cfg.HeadDim)
			if runErr != nil {
				return nil, Qwen35MetalForwardSequenceReceipt{}, true, runErr
			}
			qnorm, knorm := make([]float32, cfg.HeadDim), make([]float32, cfg.HeadDim)
			if cfg.QKNorm {
				qnorm = m.tensor(p("self_attn.q_norm.weight"))
				knorm = m.tensor(p("self_attn.k_norm.weight"))
				qwidth, kvwidth := cfg.NumHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim
				if len(qnorm) != cfg.HeadDim && len(qnorm) != qwidth || len(knorm) != cfg.HeadDim && len(knorm) != kvwidth {
					return nil, Qwen35MetalForwardSequenceReceipt{}, true, fmt.Errorf("metalgemm: P32 Qwen graph requires shared or projection-wide per-head Q/K norm weights")
				}
			}
			rotary := cfg.rotaryDim()
			cosv, sinv := make([]float32, 0, P*(rotary/2)), make([]float32, 0, P*(rotary/2))
			for pos := 0; pos < P; pos++ {
				c, si := ropeRowForLayer(cfg, l, base+pos)
				cosv, sinv = append(cosv, c...), append(sinv, si...)
			}
			attention, runErr := g.FullAttention(q, qkv[1], qkv[2], gate, qnorm, knorm, cosv, sinv,
				s.Cache.K[l], s.Cache.V[l], base, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, rotary, cfg.attnScale(), cfg.qkNormEps(), cfg.NormGain1p, cfg.QKNorm)
			if runErr != nil {
				return nil, Qwen35MetalForwardSequenceReceipt{}, true, runErr
			}
			attnOut, runErr = qwen35GraphProjection(g, s, p("self_attn.o_proj.weight"), attention.Output, quantized)
			if runErr != nil {
				return nil, Qwen35MetalForwardSequenceReceipt{}, true, runErr
			}
			kvResults = append(kvResults, kvResult{layer: l, kraw: attention.KRaw, kpost: attention.KPost, v: attention.V})
		}
		if err = g.AddInPlace(x, attnOut); err != nil {
			return nil, Qwen35MetalForwardSequenceReceipt{}, true, err
		}
		xn2, runErr := g.RMSNorm(x, m.tensor(p("post_attention_layernorm.weight")), eps, cfg.NormGain1p)
		if runErr != nil {
			return nil, Qwen35MetalForwardSequenceReceipt{}, true, runErr
		}
		gu, runErr := qwen35GraphProjections(g, s, []string{p("mlp.gate_proj.weight"), p("mlp.up_proj.weight")}, xn2, quantized)
		if runErr != nil {
			return nil, Qwen35MetalForwardSequenceReceipt{}, true, runErr
		}
		if err = g.SwiGLUInPlace(gu[0], gu[1]); err != nil {
			return nil, Qwen35MetalForwardSequenceReceipt{}, true, err
		}
		down, runErr := qwen35GraphProjection(g, s, p("mlp.down_proj.weight"), gu[0], quantized)
		if runErr != nil {
			return nil, Qwen35MetalForwardSequenceReceipt{}, true, runErr
		}
		if err = g.AddInPlace(x, down); err != nil {
			return nil, Qwen35MetalForwardSequenceReceipt{}, true, err
		}
	}
	hiddenResult, err := g.LastRMSNorm(x, m.tensor("model.norm.weight"), eps, cfg.NormGain1p)
	if err != nil {
		return nil, Qwen35MetalForwardSequenceReceipt{}, true, err
	}
	terminal := []*metalgemm.GraphResult{hiddenResult}
	for _, kv := range kvResults {
		terminal = append(terminal, kv.kraw, kv.kpost, kv.v)
	}
	outputs, graphReceipt, err := g.FinishRead(terminal...)
	receipt := Qwen35MetalForwardSequenceReceipt{
		Path: Qwen35MetalGDNSequenceForwardPath, Available: true,
		SelectorState: Qwen35MetalSequenceSelectorOn, EvidenceState: Qwen35MetalSequenceEvidenceExecuted, Tokens: P,
		CommandBuffers: 1, Encoders: graphReceipt.Encoders, TerminalWaits: 1, TerminalReadbacks: graphReceipt.HostReadbacks,
		IntermediateWaits: graphReceipt.IntermediateWaits, IntermediateReadbacks: graphReceipt.IntermediateReadbacks,
		HostUploadBytes: graphReceipt.HostUploadBytes, HostReadbackBytes: graphReceipt.HostReadbackBytes,
		Committed: graphReceipt.Committed, CompletedWait: graphReceipt.CompletedWait, TimingAvailable: graphReceipt.TimingAvailable,
		GPUMilliseconds: graphReceipt.GPUMilliseconds, WaitMilliseconds: graphReceipt.WaitMilliseconds,
	}
	b.setForwardReceipt(receipt)
	if err != nil || !graphReceipt.Committed || !graphReceipt.CompletedWait || graphReceipt.HostReadbacks != 1 {
		if err == nil {
			err = fmt.Errorf("metalgemm: incomplete Qwen graph receipt: %+v", graphReceipt)
		}
		return nil, receipt, true, err
	}
	hidden := outputs[0]
	outIndex := 1
	for _, kv := range kvResults {
		s.Cache.Kraw[kv.layer] = append(s.Cache.Kraw[kv.layer], outputs[outIndex]...)
		s.Cache.K[kv.layer] = append(s.Cache.K[kv.layer], outputs[outIndex+1]...)
		s.Cache.V[kv.layer] = append(s.Cache.V[kv.layer], outputs[outIndex+2]...)
		outIndex += 3
	}
	for i, id := range ids {
		s.Cache.appendPosition(base+i, id)
	}
	s.q4kHybridPrefillChunks++
	s.q4kHybridPrefillLastBase = base
	return hidden, receipt, true, nil
}

func init() {
	newQwen35MetalGDNSequenceBackend = func() Qwen35GDNPreprojectedSequenceBackend {
		return &metalQwen35GDNSequenceBackend{states: make(map[Qwen35GDNAuxState]*metalgemm.GDNState)}
	}
	qwen35MTPMetalP4Verify = func(s *Session, ids []int) ([][]float32, Qwen35MetalMTPCheckpoint, TargetVerificationReceipt, bool, error) {
		receipt := TargetVerificationReceipt{
			Schema: targetVerificationReceiptSchema, Engine: targetVerificationEngine,
			Path: targetVerificationDecodePath, DraftTokens: len(ids),
		}
		started := time.Now()
		result, accepted, err := s.Qwen35MetalMTPVerifyPanel(ids)
		if !accepted {
			return nil, nil, receipt, false, nil
		}
		receipt.Path = Qwen35MetalMTPVerifyPanelPath
		receipt.TargetVerificationOperations = 1
		receipt.OneOperation = err == nil
		receipt.Accounting.TargetVerification = measuredSpeculativeCost(started)
		return result.Logits, result.Checkpoint, receipt, true, err
	}
}

func (*metalQwen35GDNSequenceBackend) Qwen35GDNPreprojectedSequencePath() string {
	return Qwen35GDNPreprojectedSequencePath
}

func metalQwen35GDNGeometry(g Qwen35GDNSequenceGeometry) metalgemm.GDNGeometry {
	return metalgemm.GDNGeometry{
		NumKeyHeads: g.NumKeyHeads, NumValueHeads: g.NumValueHeads,
		KeyHeadDim: g.KeyHeadDim, ValueHeadDim: g.ValueHeadDim, ConvKernel: g.ConvKernel,
	}
}

func (b *metalQwen35GDNSequenceBackend) NewQwen35GDNAuxState(_ int, geometry Qwen35GDNSequenceGeometry) (Qwen35GDNAuxState, error) {
	state, err := metalgemm.NewGDNState(metalQwen35GDNGeometry(geometry))
	if err != nil {
		return Qwen35GDNAuxState{}, err
	}
	conv, recurrent := state.Handles()
	handles := Qwen35GDNAuxState{Convolution: Qwen35GDNAuxHandle(conv), Recurrent: Qwen35GDNAuxHandle(recurrent)}
	b.mu.Lock()
	b.states[handles] = state
	b.mu.Unlock()
	return handles, nil
}

func (b *metalQwen35GDNSequenceBackend) state(handles Qwen35GDNAuxState) *metalgemm.GDNState {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.states[handles]
}

func (b *metalQwen35GDNSequenceBackend) Qwen35GDNPreprojectedSequence(req Qwen35GDNPreprojectedSequenceRequest) (Qwen35GDNPreprojectedSequenceResult, error) {
	state := b.state(req.State)
	if state == nil {
		return Qwen35GDNPreprojectedSequenceResult{}, &metalgemm.GDNDeclinedError{Reason: "unknown auxiliary-state owner"}
	}
	g := metalQwen35GDNGeometry(req.Geometry)
	keyDim := g.NumKeyHeads * g.KeyHeadDim
	valueDim := g.NumValueHeads * g.ValueHeadDim
	convDim := 2*keyDim + valueDim
	core := make([]float32, 0, req.Tokens*valueDim)
	for start := 0; start < req.Tokens; start += metalgemm.GDNMaxPanelTokens {
		end := min(start+metalgemm.GDNMaxPanelTokens, req.Tokens)
		panel := metalgemm.GDNPanel{
			Tokens: end - start,
			Mixed:  req.Mixed[start*convDim : end*convDim], Z: req.Z[start*valueDim : end*valueDim],
			B: req.B[start*g.NumValueHeads : end*g.NumValueHeads], A: req.A[start*g.NumValueHeads : end*g.NumValueHeads],
			Conv1D: req.Conv1D, ALog: req.ALog, DTBias: req.DTBias, Norm: req.Norm,
			RMSNormEpsilon: req.RMSNormEpsilon,
		}
		panelCore, accounting, accepted, err := state.Run(panel)
		if err != nil {
			return Qwen35GDNPreprojectedSequenceResult{}, err
		}
		if !accepted || !accounting.Committed || !accounting.CompletedWait || accounting.Encoders != 1 ||
			accounting.StateH2DTransfers != 0 || accounting.StateD2HTransfers != 0 || accounting.HostRecurrenceSteps != 0 ||
			accounting.OwnedBuffers != 2 || accounting.PrivateStateBuffers != 2 {
			return Qwen35GDNPreprojectedSequenceResult{}, fmt.Errorf("metalgemm: incomplete resident GDN observation: %+v", accounting)
		}
		core = append(core, panelCore...)
	}
	return Qwen35GDNPreprojectedSequenceResult{Core: core, State: req.State}, nil
}

func (b *metalQwen35GDNSequenceBackend) SeedQwen35GDNAuxState(handles Qwen35GDNAuxState, conv, recurrent []float32) error {
	state := b.state(handles)
	if state == nil {
		return fmt.Errorf("metalgemm: unknown GDN auxiliary-state owner")
	}
	return state.Seed(conv, recurrent)
}

func (b *metalQwen35GDNSequenceBackend) SnapshotQwen35GDNAuxState(handles Qwen35GDNAuxState) ([]float32, []float32, error) {
	state := b.state(handles)
	if state == nil {
		return nil, nil, fmt.Errorf("metalgemm: unknown GDN auxiliary-state owner")
	}
	return state.Snapshot()
}

func (b *metalQwen35GDNSequenceBackend) FreeQwen35GDNAuxState(handles Qwen35GDNAuxState) error {
	b.mu.Lock()
	state := b.states[handles]
	delete(b.states, handles)
	b.mu.Unlock()
	if state != nil {
		state.Close()
	}
	return nil
}

// metalWeightsQwen35Hybrid returns this model's GPU projection table for the hybrid prefill,
// uploading it once. It mirrors metalWeights() (same dequantQ8 -> f16 Upload, big f32 buffer
// freed after each upload) but uploads the hybrid's per-layer projection set instead of the seven
// uniform standard-attention names.
func (m *Model) metalWeightsQwen35Hybrid() map[string]*metalgemm.Weight {
	metalHybridMu.Lock()
	defer metalHybridMu.Unlock()
	if w, ok := metalHybridWt[m]; ok {
		return w
	}
	cfg := m.Cfg
	w := make(map[string]*metalgemm.Weight, 8*cfg.NumLayers)
	upload := func(name string) {
		qt := m.q8(name)
		h := metalgemm.Upload(dequantQ8(qt), qt.out, qt.in)
		if h == nil {
			panic("model: metal hybrid weight upload failed for " + name)
		}
		w[name] = h
	}
	for l := 0; l < cfg.NumLayers; l++ {
		upload(layerName(l, "mlp.gate_proj.weight"))
		upload(layerName(l, "mlp.up_proj.weight"))
		upload(layerName(l, "mlp.down_proj.weight"))
		if cfg.isLinearAttnLayer(l) {
			upload(layerName(l, "linear_attn.in_proj_qkv.weight"))
			upload(layerName(l, "linear_attn.in_proj_z.weight"))
			upload(layerName(l, "linear_attn.in_proj_b.weight"))
			upload(layerName(l, "linear_attn.in_proj_a.weight"))
			upload(layerName(l, "linear_attn.out_proj.weight"))
		} else {
			upload(layerName(l, "self_attn.q_proj.weight"))
			upload(layerName(l, "self_attn.k_proj.weight"))
			upload(layerName(l, "self_attn.v_proj.weight"))
			upload(layerName(l, "self_attn.o_proj.weight"))
		}
	}
	metalHybridWt[m] = w
	return w
}

// prefillBatchedMetalQwen35Hybrid is the Metal hybrid prefill: it feeds the backend-agnostic core
// a GPU f16 GEMM (Y[P,out] = X[P,in] * W[name]^T) for each projection and lets the core run the
// recurrence/attention/norm body on the CPU. It fills the same f32 KV + linear-attn caches the
// CPU hybrid paths build (so decode stays valid) and returns the last token's post-final-norm
// hidden (caller applies the head). Reached only for a fresh prefill via metalQwen35HybridPrefillOK.
func (s *Session) prefillBatchedMetalQwen35Hybrid(ids []int) []float32 {
	// The hybrid f16 projections require MPS in addition to the shared Metal
	// device. Decline before uploading weights or mutating prompt state when that
	// capability is absent; the established Q8 implementation remains fak-native
	// and preserves the same hidden-state contract for the caller-owned head.
	if !metalHybridMPSAvailable() {
		return s.prefillQwen35HybridQHidden(ids)
	}
	m := s.M
	P := len(ids)
	gw := m.metalWeightsQwen35Hybrid()
	// mm runs Y[P,out] = X[P,in] * W[name]^T on the GPU into a fresh buffer; `in` is implicit in
	// the uploaded weight, so the core's hybridGemmFn signature drops it.
	mm := func(name string, X []float32, out int) []float32 {
		Y := make([]float32, P*out)
		gw[name].MatMul(X, P, Y)
		return Y
	}
	return s.prefillQwen35HybridViaMM(ids, mm)
}
