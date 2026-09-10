package model

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/mathx"
)

// Qwen35MTPForwardError reports a typed contract failure in the retained
// Qwen3.8 MTP path. It never permits substituting target-layer tensors or fake
// logits for a malformed or unsupported checkpoint.
type Qwen35MTPForwardError struct {
	Stage  string
	Tensor string
	Want   string
	Got    string
}

func (e *Qwen35MTPForwardError) Error() string {
	where := e.Stage
	if e.Tensor != "" {
		where += " tensor " + e.Tensor
	}
	return fmt.Sprintf("model: qwen3.8 MTP %s: got %s, want %s", where, e.Got, e.Want)
}

// Qwen35MTPForward is an isolated, stateful native draft head. Its decoder
// layer owns a separate KV cache; the target Session cache is never read or
// mutated. The retained checkpoint payload and shared target LM head remain
// immutable and are reused without copying.
type Qwen35MTPForward struct {
	target          *Model
	draft           *Session
	mat             matKernel
	tensorFormat    Qwen38MTPTensorFormat
	lastPos         int
	closed          bool
	closeOnce       sync.Once
	vocabFilter     *DraftVocabFilter
	resident        compute.Qwen35MTPDraftBackend
	transferCounter compute.Qwen35MTPTransferCounter
	receipt         Qwen35MTPForwardReceipt
}

// NewQwen35MTPForward binds the exact mtp.layers.0 namespace to the shared Qwen
// decoder-layer primitive and binds mtp.norm plus the target model's LM head.
// The exact Qwen3.8-27B-Q4_K_M Q8/Q4/Q6 inventory selects sessionQ4KKernel
// (and the existing per-format Metal dispatch on Apple Silicon); the original
// uniform-F32/BF16 layout remains unchanged. Other mixtures are refused.
func (m *Model) NewQwen35MTPForward() (*Qwen35MTPForward, error) {
	return m.newQwen35MTPForward(nil)
}

// NewQwen35MTPForwardWithBackend constructs the same retained draft head on an
// explicitly selected resident backend. The backend must advertise the exact
// MTP fusion path and physical transfer counters; unsupported implementations
// are refused before any draft position executes.
func (m *Model) NewQwen35MTPForwardWithBackend(be compute.Backend) (*Qwen35MTPForward, error) {
	if be == nil {
		return nil, &compute.UnsupportedQwen35MTPDraftError{Path: compute.Qwen35MTPDraftPath, Stage: "backend admission", Reason: "a non-nil backend is required"}
	}
	return m.newQwen35MTPForward(be)
}

func (m *Model) newQwen35MTPForward(be compute.Backend) (*Qwen35MTPForward, error) {
	if m == nil {
		return nil, qwen35MTPStateError("model", "non-nil model", "nil")
	}
	var resident compute.Qwen35MTPDraftBackend
	var counter compute.Qwen35MTPTransferCounter
	if be != nil {
		var capabilityErr error
		resident, counter, capabilityErr = qwen35MTPDraftCapability(be)
		if capabilityErr != nil {
			return nil, capabilityErr
		}
	}
	if !m.holdModelWeights() {
		return nil, qwen35MTPStateError("model weights", "open checkpoint weights", "closing or closed")
	}
	held := true
	defer func() {
		if held {
			m.releaseWeightSession()
		}
	}()

	layout, present, err := m.qwen38MTPTensorLayout()
	if err != nil {
		return nil, err
	}
	if !present {
		mode, admissionErr := qwen35MTPAdmission(m.Cfg, m.manifest, false)
		if admissionErr != nil {
			return nil, admissionErr
		}
		return nil, qwen35MTPStateError("model", "eligible one-layer shared-embedding Qwen3.8 MTP model", mode.Reason)
	}

	cfg := m.Cfg
	cfg.NumLayers = 1
	// Qwen3.8's retained MTP decoder is a full-attention layer. The target may
	// be hybrid, but its layer_types indices describe target layers, not this
	// separately named draft layer.
	if be == nil {
		cfg.LayerTypes = []string{"full_attention"}
	} else {
		// The retained decoder is unconditionally full attention. Leave the target
		// layer classification out of the resident draft config so the target's
		// long-context QSA host-gather heuristic cannot intercept this exact dense
		// device path.
		cfg.LayerTypes = nil
	}

	aliases := make(map[string]tensorMeta, len(qwen35MTPDecoderAliases)+2)
	for _, name := range []string{
		"mtp.fc.weight",
		"mtp.pre_fc_norm_hidden.weight",
		"mtp.pre_fc_norm_embedding.weight",
	} {
		if meta, ok := m.manifest[name]; ok {
			aliases[name] = meta
		}
	}
	for dst, src := range qwen35MTPDecoderAliases {
		if meta, ok := m.manifest[src]; ok {
			aliases[dst] = meta
		}
	}
	aliases["model.norm.weight"] = m.manifest["mtp.norm.weight"]

	q4Aliases := make(map[string]*q4kTensor, len(qwen35MTPDecoderAliases))
	q8Aliases := make(map[string]*q8Tensor, len(qwen35MTPDecoderAliases)+1)
	kqAliases := make(map[string]*kQuantTensor, len(qwen35MTPDecoderAliases)+1)
	var q4Head *q4kTensor
	var q8Head *q8Tensor
	if layout.Format == Qwen38MTPFormatQ4K {
		switch {
		case m.q4kw["mtp.fc.weight"] != nil:
			q4Aliases["mtp.fc.weight"] = m.q4kw["mtp.fc.weight"]
		case m.q8w["mtp.fc.weight"] != nil:
			q8Aliases["mtp.fc.weight"] = m.q8w["mtp.fc.weight"]
		case m.kqw["mtp.fc.weight"] != nil:
			kqAliases["mtp.fc.weight"] = m.kqw["mtp.fc.weight"]
		}
		for dst, src := range qwen35MTPDecoderAliases {
			if qt := m.q4kw[src]; qt != nil {
				q4Aliases[dst] = qt
			}
			if qt := m.q8w[src]; qt != nil {
				q8Aliases[dst] = qt
			}
			if qt := m.kqw[src]; qt != nil {
				kqAliases[dst] = qt
			}
		}
		head, err := m.qwen35MTPResidentHead()
		if err != nil {
			return nil, err
		}
		switch {
		case head.q4k != nil:
			q4Aliases["lm_head.weight"] = head.q4k
			q4Head = head.q4k
		case head.kq != nil:
			kqAliases["lm_head.weight"] = head.kq
		case head.q8 != nil:
			q8Aliases["lm_head.weight"] = head.q8
			q8Head = head.q8
		default:
			aliases["lm_head.weight"] = m.manifest[head.f32Name]
		}
	} else {
		// Preserve the original F32/BF16 path: resolve and alias the target head
		// from the packed manifest without introducing a resident quant store.
		headName, err := m.qwen35MTPHeadName()
		if err != nil {
			return nil, err
		}
		aliases["lm_head.weight"] = m.manifest[headName]
	}
	draftModel := &Model{
		Cfg:      cfg,
		manifest: aliases,
		raw:      m.raw,
		q4kw:     q4Aliases,
		q4khead:  q4Head,
		q8w:      q8Aliases,
		q8head:   q8Head,
		kqw:      kqAliases,
	}
	draft := &Session{M: draftModel, Cache: NewKVCache(cfg)}
	var mat matKernel = f32Kernel{draftModel}
	if layout.Format == Qwen38MTPFormatQ4K {
		// The admitted Q4_K_M inventory is intentionally mixed: mtp.fc is
		// Q8_0 while v/down/head are Q6_K. Enable every resident HAL resolver;
		// matWeightHAL still selects each tensor's exact retained store.
		if be != nil {
			draft.Quant = true
		}
		draft.Q4K = true
		// The non-Darwin implementation is an explicit CPU-native no-op. On Apple
		// Silicon this selects the existing resident Q4_K Metal dispatch.
		draft.MetalQ4K = true
		mat = sessionQ4KKernel{s: draft}
	}
	if be != nil {
		var halKV compute.KVStore
		func() {
			defer func() {
				if recovered := recover(); recovered != nil {
					halKV = nil
				}
			}()
			halKV = newHALKVStore(be, cfg)
		}()
		if halKV == nil {
			return nil, &compute.UnsupportedQwen35MTPDraftError{Backend: be.Name(), Path: compute.Qwen35MTPDraftPath, Stage: "KV allocation", Reason: "backend did not create a dedicated draft KV store"}
		}
		draft.Backend = be
		draft.halKV = halKV
		draft.halW = make(map[string]compute.Tensor)
		draft.borrowedHALW = make(map[string]struct{})
	}
	draft.initMixedQKV()
	forward := &Qwen35MTPForward{
		target: m, draft: draft, mat: mat, tensorFormat: layout.Format, lastPos: -1,
		resident: resident, transferCounter: counter,
	}
	if be != nil {
		forward.receipt.Path = compute.Qwen35MTPDraftPath
		forward.receipt.Backend = be.Name()
		forward.receipt.TransferCounterScope = "backend-global counter deltas; exclusive execution required for per-draft attribution"
		if err := forward.initResidentMTP(); err != nil {
			held = false
			forward.Close()
			return nil, err
		}
	}
	held = false
	return forward, nil
}

// Close releases the target checkpoint lifetime held by this draft head.
func (f *Qwen35MTPForward) Close() {
	if f == nil {
		return
	}
	f.closeOnce.Do(func() {
		f.closed = true
		if f.draft != nil {
			draftModel := f.draft.M
			if f.resident == nil && f.tensorFormat == Qwen38MTPFormatQ4K && draftModel != nil {
				releaseModelQ4KHandles(f.draft.M)
				f.draft.M.releaseMetalQ8Residency()
			}
			f.draft.Close()
			if f.resident != nil && draftModel != nil {
				// Resident weights are memoized on the short-lived aliased draft
				// model. Close that model-owned pool after the session drops its
				// borrowed handles so rebase/recreate cannot accumulate VRAM.
				_ = draftModel.CloseWeights()
			}
		}
		if f.target != nil {
			f.target.releaseWeightSession()
		}
	})
}

// Forward executes one native Qwen3.8 MTP draft position:
//
//  1. normalize current token embedding and prior target hidden separately;
//  2. concatenate [embedding, hidden] and apply mtp.fc;
//  3. execute the exact retained mtp.layers.0 decoder tensors;
//  4. apply mtp.norm and the shared target LM head.
func (f *Qwen35MTPForward) Forward(pos int, priorHidden, currentEmbedding []float32) ([]float32, error) {
	if f == nil || f.target == nil || f.draft == nil || f.draft.Cache == nil {
		return nil, qwen35MTPStateError("forward state", "initialized Qwen35MTPForward", "nil or incomplete")
	}
	if f.closed {
		return nil, qwen35MTPStateError("forward state", "open Qwen35MTPForward", "closed")
	}
	if pos < 0 {
		return nil, qwen35MTPStateError("position", "non-negative", fmt.Sprint(pos))
	}
	if pos <= f.lastPos {
		return nil, qwen35MTPStateError("position", fmt.Sprintf("greater than %d", f.lastPos), fmt.Sprint(pos))
	}
	if f.resident != nil {
		_, logits, err := f.residentForwardFeedback(pos, priorHidden, currentEmbedding)
		return logits, err
	}

	x, err := f.qwen38MTPFuse(priorHidden, currentEmbedding)
	if err != nil {
		return nil, err
	}
	cos, sin := ropeRowForLayer(f.draft.M.Cfg, 0, pos)
	x = f.draft.blockStep(0, pos, x, cos, sin, f.mat)
	f.draft.Cache.appendPosition(pos, -1)
	f.lastPos = pos
	return f.ProjectHead(f.draft.M.finalNorm(x)), nil
}

// SetDraftVocabFilter configures the empirical draft vocabulary filter on the forward head.
func (f *Qwen35MTPForward) SetDraftVocabFilter(filter *DraftVocabFilter) {
	if f != nil {
		f.vocabFilter = filter
	}
}

// DraftVocabFilter returns the configured draft vocabulary filter, if any.
func (f *Qwen35MTPForward) DraftVocabFilter() *DraftVocabFilter {
	if f == nil {
		return nil
	}
	return f.vocabFilter
}

// ProjectHead projects the normalized hidden vector to logits. If a DraftVocabFilter
// is configured on the forward head, it projects only across the truncated vocabulary
// subset. Otherwise, it projects across the full model vocabulary.
func (f *Qwen35MTPForward) ProjectHead(xf []float32) []float32 {
	if f == nil || f.draft == nil || f.draft.M == nil {
		return nil
	}
	if f.vocabFilter != nil && len(f.vocabFilter.Subset) > 0 {
		return f.ProjectFiltered(xf, f.vocabFilter.Subset)
	}
	return f.draft.headResident(xf)
}

// ProjectFiltered computes logits for an explicit subset of token IDs directly
// from the resident LM-head format. It never expands a quantized head or allocates
// a full-vocabulary logits buffer. The subset filter is opt-in; on Darwin it uses
// the resident CPU bytes because Metal currently has no row-subset GEMV contract.
func (f *Qwen35MTPForward) ProjectFiltered(xf []float32, subset []int) []float32 {
	if f == nil || f.draft == nil || f.draft.M == nil || len(subset) == 0 {
		return nil
	}
	m := f.draft.M
	var logits []float32
	switch {
	case m.q4khead != nil || m.q4kw[m.q4kHeadName()] != nil:
		head := m.q4khead
		if head == nil {
			head = m.q4kw[m.q4kHeadName()]
		}
		logits = q4kMatRowsSubset(head, xf, subset)
	case m.kqHeadName() != "":
		logits = kQuantMatRowsSubset(m.kqw[m.kqHeadName()], xf, subset)
	case m.q4head != nil:
		logits = q4MatRowsSubset(m.q4head, xf, subset)
	case m.q8head != nil || m.q8w[m.headName()] != nil:
		head := m.q8head
		if head == nil {
			head = m.q8w[m.headName()]
		}
		logits = q8MatRowsSubset(head, f.draft.quantizeVecQ8(xf), subset)
	default:
		logits = parMatRowsSubset(m.lmHead(), xf, subset, m.Cfg.HiddenSize)
	}
	scaleSubsetLogitsInPlace(logits, subset, f.draft.M.Cfg)
	return logits
}

func q4kMatRowsSubset(qt *q4kTensor, x []float32, subset []int) []float32 {
	qt.requireRawCPU("subset projection")
	y := newSubsetLogits(subset, qt.out)
	if q4kSDOTEnabled() {
		qv := quantizeVecQ8(x)
		parForRange(len(subset), len(subset)*qt.in, func(lo, hi int) {
			is := make([]int32, qt.nblk*8)
			ss := make([]int32, qt.nblk*8)
			rowBytes := qt.q4kRowBytes()
			for i := lo; i < hi; i++ {
				tok := subset[i]
				if tok < 0 || tok >= qt.out {
					continue
				}
				row := qt.raw[tok*rowBytes : (tok+1)*rowBytes]
				q4kReduceRow(row, qt.nblk, qv.q, is, ss)
				y[i] = q4kCombineRow(row, qt.nblk, qv.d, is, ss)
			}
		})
		return y
	}
	parForRange(len(subset), len(subset)*qt.in, func(lo, hi int) {
		buf := make([]float32, qkK)
		rowBytes := qt.q4kRowBytes()
		for i := lo; i < hi; i++ {
			tok := subset[i]
			if tok < 0 || tok >= qt.out {
				continue
			}
			row := qt.raw[tok*rowBytes : (tok+1)*rowBytes]
			var acc float32
			for b := 0; b < qt.nblk; b++ {
				q4kDequantSuperBlock(buf, row[b*q4kBlockBytes:(b+1)*q4kBlockBytes])
				acc += subsetDot4(buf, x[b*qkK:])
			}
			y[i] = acc
		}
	})
	return y
}

func kQuantMatRowsSubset(qt *kQuantTensor, x []float32, subset []int) []float32 {
	y := newSubsetLogits(subset, qt.out)
	if qt.kind == kindQ6K && kQuantSDOTEnabled(qt.kind) {
		qv := quantizeVecQ8(x)
		parForRange(len(subset), len(subset)*qt.in, func(lo, hi int) {
			is := make([]int32, qt.nblk*q6kGroupsPerBlock)
			ss := make([]int32, qt.nblk*q6kGroupsPerBlock)
			rowBytes := qt.rowBytes()
			for i := lo; i < hi; i++ {
				tok := subset[i]
				if tok < 0 || tok >= qt.out {
					continue
				}
				row := qt.raw[tok*rowBytes : (tok+1)*rowBytes]
				q6kReduceRow(row, qt.nblk, qv.q, is, ss)
				y[i] = q6kCombineRow(row, qt.nblk, qv.d, is, ss)
			}
		})
		return y
	}
	parForRange(len(subset), len(subset)*qt.in, func(lo, hi int) {
		blockWeights := qt.kind.blockWeights()
		buf := make([]float32, blockWeights)
		rowBytes := qt.rowBytes()
		blockBytes := qt.kind.blockBytes()
		for i := lo; i < hi; i++ {
			tok := subset[i]
			if tok < 0 || tok >= qt.out {
				continue
			}
			row := qt.raw[tok*rowBytes : (tok+1)*rowBytes]
			var acc float32
			for b := 0; b < qt.nblk; b++ {
				kQuantDequantSuperBlock(buf, row[b*blockBytes:(b+1)*blockBytes], qt.kind)
				acc += subsetDot4(buf, x[b*blockWeights:])
			}
			y[i] = acc
		}
	})
	return y
}

func q8MatRowsSubset(qt *q8Tensor, qv q8Vec, subset []int) []float32 {
	y := newSubsetLogits(subset, qt.out)
	parForRange(len(subset), len(subset)*qt.in, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			tok := subset[i]
			if tok < 0 || tok >= qt.out {
				continue
			}
			y[i] = qdot8GEMV(
				qt.q[tok*qt.in:(tok+1)*qt.in],
				qt.d[tok*qt.nblk:(tok+1)*qt.nblk],
				qv,
				qt.nblk,
			)
		}
	})
	return y
}

func q4MatRowsSubset(qt *q4Tensor, x []float32, subset []int) []float32 {
	y := newSubsetLogits(subset, qt.out)
	parForRange(len(subset), len(subset)*qt.in, func(lo, hi int) {
		buf := make([]float32, qBlk4)
		half := qBlk4 / 2
		for i := lo; i < hi; i++ {
			tok := subset[i]
			if tok < 0 || tok >= qt.out {
				continue
			}
			qrow := qt.q[tok*qt.nblk*half : (tok+1)*qt.nblk*half]
			drow := qt.d[tok*qt.nblk : (tok+1)*qt.nblk]
			var s0, s1, s2, s3 float32
			for b := 0; b < qt.nblk; b++ {
				dequantQ4Block(buf, drow[b], qrow[b*half:])
				xs := x[b*qBlk4:]
				s0 += buf[0]*xs[0] + buf[1]*xs[1] + buf[2]*xs[2] + buf[3]*xs[3]
				s1 += buf[4]*xs[4] + buf[5]*xs[5] + buf[6]*xs[6] + buf[7]*xs[7]
				s2 += buf[8]*xs[8] + buf[9]*xs[9] + buf[10]*xs[10] + buf[11]*xs[11]
				s3 += buf[12]*xs[12] + buf[13]*xs[13] + buf[14]*xs[14] + buf[15]*xs[15]
				s0 += buf[16]*xs[16] + buf[17]*xs[17] + buf[18]*xs[18] + buf[19]*xs[19]
				s1 += buf[20]*xs[20] + buf[21]*xs[21] + buf[22]*xs[22] + buf[23]*xs[23]
				s2 += buf[24]*xs[24] + buf[25]*xs[25] + buf[26]*xs[26] + buf[27]*xs[27]
				s3 += buf[28]*xs[28] + buf[29]*xs[29] + buf[30]*xs[30] + buf[31]*xs[31]
			}
			y[i] = (s0 + s1) + (s2 + s3)
		}
	})
	return y
}

func newSubsetLogits(subset []int, vocab int) []float32 {
	y := make([]float32, len(subset))
	negInf := float32(math.Inf(-1))
	for i, tok := range subset {
		if tok < 0 || tok >= vocab {
			y[i] = negInf
		}
	}
	return y
}

func subsetDot4(w, x []float32) float32 {
	var s0, s1, s2, s3 float32
	for i := 0; i < len(w); i += 4 {
		s0 += w[i] * x[i]
		s1 += w[i+1] * x[i+1]
		s2 += w[i+2] * x[i+2]
		s3 += w[i+3] * x[i+3]
	}
	return (s0 + s1) + (s2 + s3)
}

// Argmax resolves logits returned by Forward or ProjectHead to a full vocabulary token ID,
// respecting any configured DraftVocabFilter.
func (f *Qwen35MTPForward) Argmax(logits []float32) int {
	if f != nil && f.vocabFilter != nil && len(logits) == len(f.vocabFilter.Subset) {
		return f.vocabFilter.Argmax(logits)
	}
	if len(logits) == 0 {
		return -1
	}
	return argmaxF32(logits)
}

// Qwen35MTPFuse implements the checkpoint-defined pre-layer path exactly:
// normalize the current token embedding and prior target hidden state
// independently, concatenate [embedding, hidden] in that order, then apply mtp.fc.
func (m *Model) Qwen35MTPFuse(priorHidden, currentEmbedding []float32) ([]float32, error) {
	if m == nil {
		return nil, qwen35MTPStateError("model", "non-nil model", "nil")
	}
	if _, err := qwen35MTPAdmission(m.Cfg, m.manifest, false); err != nil {
		return nil, err
	}
	if !m.Cfg.isQwen35TextFamily() || m.Cfg.NumMTPLayers() != 1 || m.Cfg.MTPUseDedicatedEmbeddings {
		return nil, qwen35MTPStateError("model", "eligible one-layer shared-embedding Qwen3.8 MTP model", "ineligible config")
	}

	h := m.Cfg.HiddenSize
	if h <= 0 {
		return nil, qwen35MTPStateError("hidden size", "positive", fmt.Sprint(h))
	}
	if len(priorHidden) != h {
		return nil, qwen35MTPStateError("prior hidden shape", fmt.Sprintf("[%d]", h), fmt.Sprintf("[%d]", len(priorHidden)))
	}
	if len(currentEmbedding) != h {
		return nil, qwen35MTPStateError("current embedding shape", fmt.Sprintf("[%d]", h), fmt.Sprintf("[%d]", len(currentEmbedding)))
	}
	if m.Cfg.RMSNormEps < 0 {
		return nil, qwen35MTPStateError("RMS norm epsilon", "non-negative", fmt.Sprint(m.Cfg.RMSNormEps))
	}

	hiddenNorm, err := m.qwen35MTPF32Tensor("mtp.pre_fc_norm_hidden.weight", []int{h})
	if err != nil {
		return nil, err
	}
	embeddingNorm, err := m.qwen35MTPF32Tensor("mtp.pre_fc_norm_embedding.weight", []int{h})
	if err != nil {
		return nil, err
	}
	fc, err := m.qwen35MTPF32Tensor("mtp.fc.weight", []int{h, 2 * h})
	if err != nil {
		return nil, err
	}

	eps := float32(m.Cfg.RMSNormEps)
	fusedInput := make([]float32, 0, 2*h)
	normedEmbedding := rmsnormCfg(currentEmbedding, embeddingNorm, eps, m.Cfg)
	normedHidden := rmsnormCfg(priorHidden, hiddenNorm, eps, m.Cfg)
	fusedInput = append(fusedInput, normedEmbedding...)
	fusedInput = append(fusedInput, normedHidden...)
	return parMatRows(fc, fusedInput, h, 2*h), nil
}

var qwen35MTPDecoderAliases = map[string]string{
	"model.layers.0.input_layernorm.weight":          "mtp.layers.0.input_layernorm.weight",
	"model.layers.0.post_attention_layernorm.weight": "mtp.layers.0.post_attention_layernorm.weight",
	"model.layers.0.self_attn.q_norm.weight":         "mtp.layers.0.self_attn.q_norm.weight",
	"model.layers.0.self_attn.k_norm.weight":         "mtp.layers.0.self_attn.k_norm.weight",
	"model.layers.0.self_attn.q_proj.weight":         "mtp.layers.0.self_attn.q_proj.weight",
	"model.layers.0.self_attn.k_proj.weight":         "mtp.layers.0.self_attn.k_proj.weight",
	"model.layers.0.self_attn.v_proj.weight":         "mtp.layers.0.self_attn.v_proj.weight",
	"model.layers.0.self_attn.o_proj.weight":         "mtp.layers.0.self_attn.o_proj.weight",
	"model.layers.0.mlp.gate_proj.weight":            "mtp.layers.0.mlp.gate_proj.weight",
	"model.layers.0.mlp.up_proj.weight":              "mtp.layers.0.mlp.up_proj.weight",
	"model.layers.0.mlp.down_proj.weight":            "mtp.layers.0.mlp.down_proj.weight",
}

func (m *Model) validateQwen35MTPForwardTensors() error {
	cfg := m.Cfg
	h, hd := cfg.HiddenSize, cfg.HeadDim
	if h <= 0 || hd <= 0 || cfg.NumHeads <= 0 || cfg.NumKVHeads <= 0 || cfg.IntermediateSize <= 0 || cfg.VocabSize <= 0 {
		return qwen35MTPStateError("config dimensions", "positive hidden/head/attention/intermediate/vocab dimensions", fmt.Sprintf("hidden=%d head_dim=%d heads=%d kv_heads=%d intermediate=%d vocab=%d", h, hd, cfg.NumHeads, cfg.NumKVHeads, cfg.IntermediateSize, cfg.VocabSize))
	}
	qOut := cfg.NumHeads * hd
	if cfg.AttnOutputGate {
		qOut *= 2
	}
	shapes := map[string][]int{
		"mtp.fc.weight":                                {h, 2 * h},
		"mtp.pre_fc_norm_embedding.weight":             {h},
		"mtp.pre_fc_norm_hidden.weight":                {h},
		"mtp.norm.weight":                              {h},
		"mtp.layers.0.input_layernorm.weight":          {h},
		"mtp.layers.0.post_attention_layernorm.weight": {h},
		"mtp.layers.0.self_attn.q_norm.weight":         {hd},
		"mtp.layers.0.self_attn.k_norm.weight":         {hd},
		"mtp.layers.0.self_attn.q_proj.weight":         {qOut, h},
		"mtp.layers.0.self_attn.k_proj.weight":         {cfg.NumKVHeads * hd, h},
		"mtp.layers.0.self_attn.v_proj.weight":         {cfg.NumKVHeads * hd, h},
		"mtp.layers.0.self_attn.o_proj.weight":         {h, cfg.NumHeads * hd},
		"mtp.layers.0.mlp.gate_proj.weight":            {cfg.IntermediateSize, h},
		"mtp.layers.0.mlp.up_proj.weight":              {cfg.IntermediateSize, h},
		"mtp.layers.0.mlp.down_proj.weight":            {h, cfg.IntermediateSize},
	}
	for _, name := range qwen35MTPRequiredTensors {
		if _, err := m.qwen35MTPF32Tensor(name, shapes[name]); err != nil {
			return err
		}
	}
	_, err := m.qwen35MTPHeadName()
	return err
}

func (m *Model) qwen35MTPHeadName() (string, error) {
	name := "lm_head.weight"
	if !m.has(name) {
		name = "model.embed_tokens.weight"
	}
	if _, err := m.qwen35MTPF32Tensor(name, []int{m.Cfg.VocabSize, m.Cfg.HiddenSize}); err != nil {
		return "", err
	}
	return name, nil
}

type qwen35MTPHeadAlias struct {
	f32Name string
	q4k     *q4kTensor
	kq      *kQuantTensor
	q8      *q8Tensor
}

// qwen35MTPResidentHead resolves the target LM head from its retained store.
// The returned pointer aliases immutable target storage; the target model's
// weight-session hold keeps that storage alive until the draft is closed.
func (m *Model) qwen35MTPResidentHead() (qwen35MTPHeadAlias, error) {
	want := []int{m.Cfg.VocabSize, m.Cfg.HiddenSize}
	badShape := func(name string, out, in int) (qwen35MTPHeadAlias, error) {
		return qwen35MTPHeadAlias{}, &Qwen35MTPForwardError{
			Stage:  "weight shape",
			Tensor: name,
			Want:   fmt.Sprint(want),
			Got:    fmt.Sprint([]int{out, in}),
		}
	}

	q4Name := m.q4kHeadName()
	q4 := m.q4khead
	if q4 == nil {
		q4 = m.q4kw[q4Name]
	}
	if q4 != nil {
		if q4.out != want[0] || q4.in != want[1] {
			return badShape(q4Name, q4.out, q4.in)
		}
		return qwen35MTPHeadAlias{q4k: q4}, nil
	}

	for _, name := range [...]string{"lm_head.weight", "model.embed_tokens.weight"} {
		if q := m.kqw[name]; q != nil {
			if q.out != want[0] || q.in != want[1] {
				return badShape(name, q.out, q.in)
			}
			return qwen35MTPHeadAlias{kq: q}, nil
		}
	}

	q8Name := m.headName()
	q8 := m.q8head
	if q8 == nil {
		q8 = m.q8w[q8Name]
	}
	if q8 != nil {
		if q8.out != want[0] || q8.in != want[1] {
			return badShape(q8Name, q8.out, q8.in)
		}
		return qwen35MTPHeadAlias{q8: q8}, nil
	}

	name, err := m.qwen35MTPHeadName()
	if err != nil {
		return qwen35MTPHeadAlias{}, err
	}
	return qwen35MTPHeadAlias{f32Name: name}, nil
}

func (m *Model) qwen35MTPF32Tensor(name string, wantShape []int) ([]float32, error) {
	meta, ok := m.manifest[name]
	if !ok {
		return nil, &Qwen35MTPForwardError{Stage: "weight lookup", Tensor: name, Want: "present", Got: "missing"}
	}
	if !strings.EqualFold(meta.Dtype, "F32") && !strings.EqualFold(meta.Dtype, "BF16") {
		return nil, &Qwen35MTPForwardError{Stage: "weight dtype", Tensor: name, Want: "F32 or BF16", Got: meta.Dtype}
	}
	if !sameIntShape(meta.Shape, wantShape) {
		return nil, &Qwen35MTPForwardError{Stage: "weight shape", Tensor: name, Want: fmt.Sprint(wantShape), Got: fmt.Sprint(meta.Shape)}
	}
	elems, err := tensorShapeElems(name, wantShape)
	if err != nil {
		return nil, err
	}
	if strings.EqualFold(meta.Dtype, "BF16") && meta.Nbytes == elems*2 {
		if meta.Offset < 0 || meta.Offset+meta.Nbytes > len(m.raw) {
			return nil, &Qwen35MTPForwardError{Stage: "weight storage", Tensor: name, Want: fmt.Sprintf("%d bytes inside model payload", elems*2), Got: fmt.Sprintf("offset=%d nbytes=%d payload=%d", meta.Offset, meta.Nbytes, len(m.raw))}
		}
		rawSlice := m.raw[meta.Offset : meta.Offset+meta.Nbytes]
		out := make([]float32, elems)
		for i := 0; i < elems; i++ {
			u16 := binary.LittleEndian.Uint16(rawSlice[i*2 : i*2+2])
			out[i] = math.Float32frombits(uint32(u16) << 16)
		}
		return out, nil
	}
	wantBytes := 4 * elems
	if meta.Nbytes != wantBytes || meta.Offset < 0 || meta.Offset+meta.Nbytes > len(m.raw) {
		return nil, &Qwen35MTPForwardError{Stage: "weight storage", Tensor: name, Want: fmt.Sprintf("%d bytes inside model payload", wantBytes), Got: fmt.Sprintf("offset=%d nbytes=%d payload=%d", meta.Offset, meta.Nbytes, len(m.raw))}
	}
	return m.tensor(name), nil
}

func qwen35MTPStateError(stage, want, got string) *Qwen35MTPForwardError {
	return &Qwen35MTPForwardError{Stage: stage, Want: want, Got: got}
}

func sameIntShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// DraftVocabFilter defines an empirical high-frequency token subset for truncated
// vocabulary projection in MTP speculative draft heads.
type DraftVocabFilter struct {
	// Subset holds the empirical high-frequency token IDs (e.g., top 40k of 248k).
	Subset []int

	// CoverageThreshold is an optional minimum softmax probability threshold in [0, 1].
	// When positive, if the probability of the argmax token in the subset falls below
	// this threshold, candidate generation can fall back to standard decode or cleanly reject.
	CoverageThreshold float32

	subsetMap map[int]int
}

// NewDraftVocabFilter constructs a DraftVocabFilter for the given token subset.
func NewDraftVocabFilter(subset []int) *DraftVocabFilter {
	return NewDraftVocabFilterWithThreshold(subset, 0)
}

// NewDraftVocabFilterWithThreshold constructs a DraftVocabFilter with an optional coverage probability threshold.
func NewDraftVocabFilterWithThreshold(subset []int, threshold float32) *DraftVocabFilter {
	if len(subset) == 0 {
		return nil
	}
	s := append([]int(nil), subset...)
	m := make(map[int]int, len(s))
	for i, tok := range s {
		m[tok] = i
	}
	return &DraftVocabFilter{
		Subset:            s,
		CoverageThreshold: threshold,
		subsetMap:         m,
	}
}

// Len returns the count of token IDs in the truncated subset.
func (f *DraftVocabFilter) Len() int {
	if f == nil {
		return 0
	}
	return len(f.Subset)
}

// RemapIndex maps an index in the truncated subset back to the full vocabulary token ID.
// Returns -1 if subsetIdx is out of bounds.
func (f *DraftVocabFilter) RemapIndex(subsetIdx int) int {
	if f == nil || subsetIdx < 0 || subsetIdx >= len(f.Subset) {
		return -1
	}
	return f.Subset[subsetIdx]
}

// Contains reports whether tokenID is present in the truncated subset.
func (f *DraftVocabFilter) Contains(tokenID int) bool {
	if f == nil || f.subsetMap == nil {
		return false
	}
	_, ok := f.subsetMap[tokenID]
	return ok
}

// Index returns the subset index for tokenID, or (-1, false) if absent.
func (f *DraftVocabFilter) Index(tokenID int) (int, bool) {
	if f == nil || f.subsetMap == nil {
		return -1, false
	}
	idx, ok := f.subsetMap[tokenID]
	return idx, ok
}

// Argmax finds the index of the maximum logit in subsetLogits and remaps it to
// the corresponding full vocabulary token ID.
func (f *DraftVocabFilter) Argmax(subsetLogits []float32) int {
	tokID, _, _ := f.ArgmaxWithProb(subsetLogits)
	return tokID
}

// ArgmaxWithProb finds the argmax within subsetLogits, maps the subset index
// back to the full vocabulary token ID, and computes its softmax probability
// across the subset. It returns (tokenID, prob, ok). ok is false if subsetLogits
// is empty or if CoverageThreshold > 0 and prob < CoverageThreshold.
func (f *DraftVocabFilter) ArgmaxWithProb(subsetLogits []float32) (int, float32, bool) {
	if f == nil || len(subsetLogits) == 0 || len(f.Subset) == 0 {
		return -1, 0, false
	}
	n := len(subsetLogits)
	if n > len(f.Subset) {
		n = len(f.Subset)
	}
	maxIdx := 0
	maxVal := subsetLogits[0]
	for i := 1; i < n; i++ {
		if subsetLogits[i] > maxVal {
			maxVal = subsetLogits[i]
			maxIdx = i
		}
	}
	tokenID := f.Subset[maxIdx]
	var sumExp float64
	for i := 0; i < n; i++ {
		sumExp += math.Exp(float64(subsetLogits[i] - maxVal))
	}
	prob := float32(0)
	if sumExp > 0 {
		prob = float32(1.0 / sumExp)
	}
	if f.CoverageThreshold > 0 && prob < f.CoverageThreshold {
		return tokenID, prob, false
	}
	return tokenID, prob, true
}

// parMatRowsSubset parallelizes output row projections for an explicit subset
// of vocabulary token IDs. Output row y[i] is computed for token subset[i].
func parMatRowsSubset(w, x []float32, subset []int, in int) []float32 {
	if len(subset) == 0 || in <= 0 {
		return nil
	}
	y := make([]float32, len(subset))
	vocabSize := 0
	if in > 0 {
		vocabSize = len(w) / in
	}
	row := func(lo, hi int) {
		for i := lo; i < hi; i++ {
			tok := subset[i]
			if tok >= 0 && tok < vocabSize {
				y[i] = mathx.FDot(w[tok*in:tok*in+in], x)
			} else {
				y[i] = float32(math.Inf(-1))
			}
		}
	}
	if len(subset)*in < parThreshold {
		row(0, len(subset))
		return y
	}
	workers := currentWorkerCount()
	if maxW := len(subset) / 8192; maxW < workers {
		if maxW < 1 {
			maxW = 1
		}
		workers = maxW
	}
	parFor(len(subset), workers, row)
	return y
}

func scaleSubsetLogitsInPlace(logits []float32, subset []int, cfg Config) {
	s := float32(1)
	if cfg.LogitScale != 0 && cfg.LogitScale != 1 {
		s = float32(cfg.LogitScale)
	}
	if s != 1 {
		for i := range logits {
			logits[i] *= s
		}
	}
	softcapInPlace(logits, float32(cfg.LogitSoftcap))
	if len(cfg.SuppressTokens) > 0 {
		negInf := float32(math.Inf(-1))
		suppressed := make(map[int]bool, len(cfg.SuppressTokens))
		for _, id := range cfg.SuppressTokens {
			suppressed[id] = true
		}
		for i, id := range subset {
			if suppressed[id] {
				logits[i] = negInf
			}
		}
	}
}
