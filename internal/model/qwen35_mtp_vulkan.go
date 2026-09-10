package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Qwen35MTPForwardReceipt reports only transfers observed by the selected
// backend. Setup includes immutable weight placement. Boundary traffic is the
// two host inputs and the requested host outputs. Intermediate traffic covers
// the pre-layer fusion, retained decoder, final normalization, and LM head and
// must remain zero for a resident draft position.
type Qwen35MTPForwardReceipt struct {
	Path                 string `json:"path"`
	Backend              string `json:"backend"`
	TransferCounterScope string `json:"transfer_counter_scope"`
	Positions            uint64 `json:"positions"`
	SetupH2DBytes        uint64 `json:"setup_h2d_bytes"`
	SetupD2HBytes        uint64 `json:"setup_d2h_bytes"`
	BoundaryH2DBytes     uint64 `json:"boundary_h2d_bytes"`
	BoundaryD2HBytes     uint64 `json:"boundary_d2h_bytes"`
	IntermediateH2DBytes uint64 `json:"intermediate_h2d_bytes"`
	IntermediateD2HBytes uint64 `json:"intermediate_d2h_bytes"`
	DecoderOperations    uint64 `json:"decoder_operations"`
	HeadOperations       uint64 `json:"head_operations"`
}

// Receipt returns an immutable snapshot of resident MTP execution accounting.
func (f *Qwen35MTPForward) Receipt() Qwen35MTPForwardReceipt {
	if f == nil {
		return Qwen35MTPForwardReceipt{}
	}
	return f.receipt
}

type qwen35MTPTransferSnapshot struct{ h2d, d2h uint64 }

func qwen35MTPTransferNow(counter compute.Qwen35MTPTransferCounter) qwen35MTPTransferSnapshot {
	if counter == nil {
		return qwen35MTPTransferSnapshot{}
	}
	h2d, d2h := counter.Qwen35MTPTransferBytes()
	return qwen35MTPTransferSnapshot{h2d: h2d, d2h: d2h}
}

func qwen35MTPTransferDelta(before, after qwen35MTPTransferSnapshot) (h2d, d2h uint64, err error) {
	if after.h2d < before.h2d || after.d2h < before.d2h {
		return 0, 0, fmt.Errorf("transfer counter regressed from h2d=%d d2h=%d to h2d=%d d2h=%d", before.h2d, before.d2h, after.h2d, after.d2h)
	}
	return after.h2d - before.h2d, after.d2h - before.d2h, nil
}

func qwen35MTPDraftCapability(be compute.Backend) (compute.Qwen35MTPDraftBackend, compute.Qwen35MTPTransferCounter, error) {
	if be == nil {
		return nil, nil, &compute.UnsupportedQwen35MTPDraftError{Path: compute.Qwen35MTPDraftPath, Stage: "backend admission", Reason: "a non-nil backend is required"}
	}
	marker, advertised := be.(interface{ Qwen35MTPDraftPath() string })
	draft, implemented := be.(compute.Qwen35MTPDraftBackend)
	path := ""
	if advertised {
		path = marker.Qwen35MTPDraftPath()
	}
	if !advertised || !implemented || path != compute.Qwen35MTPDraftPath {
		reason := "backend does not implement the resident MTP operation"
		if advertised && !implemented {
			reason = "path marker has no operation implementation"
		} else if advertised && path != compute.Qwen35MTPDraftPath {
			reason = "wrong capability identity"
		}
		return nil, nil, &compute.UnsupportedQwen35MTPDraftError{Backend: be.Name(), Path: path, Stage: "backend admission", Reason: reason}
	}
	counter, ok := be.(compute.Qwen35MTPTransferCounter)
	if !ok {
		return nil, nil, &compute.UnsupportedQwen35MTPDraftError{Backend: be.Name(), Path: path, Stage: "transfer accounting", Reason: "backend does not expose physical transfer counters"}
	}
	return draft, counter, nil
}

func (f *Qwen35MTPForward) initResidentMTP() (err error) {
	if f == nil || f.draft == nil || f.draft.Backend == nil || f.resident == nil || f.transferCounter == nil {
		return qwen35MTPStateError("resident state", "initialized resident draft session", "nil or incomplete")
	}
	cfg := f.draft.M.Cfg
	if cfg.BlockTopology != PreNorm || cfg.LayerNorm {
		return &compute.UnsupportedQwen35MTPDraftError{Backend: f.draft.Backend.Name(), Path: compute.Qwen35MTPDraftPath, Stage: "decoder topology admission", Reason: "resident draft requires the checkpoint's serial pre-RMSNorm decoder topology"}
	}
	if cfg.DenseMLP || cfg.ActGeluTanh || cfg.ActGeluErf || cfg.NumExperts != 0 {
		return &compute.UnsupportedQwen35MTPDraftError{Backend: f.draft.Backend.Name(), Path: compute.Qwen35MTPDraftPath, Stage: "MLP admission", Reason: "resident draft requires the checkpoint's dense SwiGLU MLP"}
	}
	if cfg.AttentionBias || cfg.AttnSoftcap != 0 || cfg.Alibi {
		return &compute.UnsupportedQwen35MTPDraftError{Backend: f.draft.Backend.Name(), Path: compute.Qwen35MTPDraftPath, Stage: "attention admission", Reason: "resident draft requires unbiased rotary attention without score soft-capping"}
	}
	for _, name := range []string{
		"mtp.layers.0.self_attn.q_proj.bias",
		"mtp.layers.0.self_attn.k_proj.bias",
		"mtp.layers.0.self_attn.v_proj.bias",
		"mtp.layers.0.self_attn.o_proj.bias",
		"mtp.layers.0.mlp.gate_proj.bias",
		"mtp.layers.0.mlp.up_proj.bias",
		"mtp.layers.0.mlp.down_proj.bias",
	} {
		if f.target.hasWeight(name) {
			return &compute.UnsupportedQwen35MTPDraftError{Backend: f.draft.Backend.Name(), Path: compute.Qwen35MTPDraftPath, Stage: "bias admission", Reason: "resident draft does not admit retained decoder bias tensor " + name}
		}
	}
	if cfg.rotaryDim() != cfg.HeadDim && (cfg.RopeScaling != "" || cfg.LongRope != nil) {
		return &compute.UnsupportedQwen35MTPDraftError{Backend: f.draft.Backend.Name(), Path: compute.Qwen35MTPDraftPath, Stage: "rotary admission", Reason: "scaled partial RoPE would require host readback"}
	}
	if cfg.rotaryDim() != cfg.HeadDim {
		if _, ok := f.draft.Backend.(qwen35PartialRoPEBackend); !ok {
			return &compute.UnsupportedQwen35MTPDraftError{Backend: f.draft.Backend.Name(), Path: compute.Qwen35MTPDraftPath, Stage: "rotary admission", Reason: "backend lacks resident partial-RoPE Q/K"}
		}
	}
	if cfg.AttnOutputGate {
		if _, ok := f.draft.Backend.(qwen35SigmoidGateBackend); !ok {
			return &compute.UnsupportedQwen35MTPDraftError{Backend: f.draft.Backend.Name(), Path: compute.Qwen35MTPDraftPath, Stage: "attention gate admission", Reason: "backend lacks resident sigmoid output gating"}
		}
		if _, ok := f.draft.Backend.(qwen35QueryGateSplitBackend); !ok {
			return &compute.UnsupportedQwen35MTPDraftError{Backend: f.draft.Backend.Name(), Path: compute.Qwen35MTPDraftPath, Stage: "attention gate admission", Reason: "backend lacks resident query/gate split"}
		}
	}
	if cfg.QKNorm && (cfg.LayerNorm || cfg.QKNormPerHeadWeight) {
		return &compute.UnsupportedQwen35MTPDraftError{Backend: f.draft.Backend.Name(), Path: compute.Qwen35MTPDraftPath, Stage: "Q/K normalization admission", Reason: "resident draft supports shared-head RMS normalization only"}
	}
	needsQ6K := false
	for _, weight := range f.draft.M.kqw {
		if weight != nil && weight.kind == kindQ6K {
			needsQ6K = true
			break
		}
	}
	if needsQ6K {
		q6k, ok := f.draft.Backend.(compute.Qwen35MTPQ6KBackend)
		if !ok || !q6k.SupportsQ6KMatMul() {
			return &compute.UnsupportedQwen35MTPDraftError{Backend: f.draft.Backend.Name(), Path: compute.Qwen35MTPDraftPath, Stage: "Q6_K admission", Reason: "checkpoint requires native packed Q6_K MatMul"}
		}
	}

	before := qwen35MTPTransferNow(f.transferCounter)
	defer func() {
		if recovered := recover(); recovered != nil {
			err = &compute.UnsupportedQwen35MTPDraftError{Backend: f.draft.Backend.Name(), Path: compute.Qwen35MTPDraftPath, Stage: "weight placement", Reason: fmt.Sprint(recovered)}
		}
	}()
	for _, name := range []string{
		"mtp.fc.weight",
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.self_attn.k_proj.weight",
		"model.layers.0.self_attn.v_proj.weight",
		"model.layers.0.self_attn.o_proj.weight",
		"model.layers.0.mlp.gate_proj.weight",
		"model.layers.0.mlp.up_proj.weight",
		"model.layers.0.mlp.down_proj.weight",
	} {
		_ = f.draft.matWeightHAL(name)
	}
	for _, name := range []string{
		"mtp.pre_fc_norm_hidden.weight",
		"mtp.pre_fc_norm_embedding.weight",
		"model.layers.0.input_layernorm.weight",
		"model.layers.0.post_attention_layernorm.weight",
		"model.layers.0.self_attn.q_norm.weight",
		"model.layers.0.self_attn.k_norm.weight",
		"model.norm.weight",
	} {
		_ = f.draft.normWeightHAL(name)
	}
	_ = f.draft.lmHeadMatHAL()
	after := qwen35MTPTransferNow(f.transferCounter)
	f.receipt.SetupH2DBytes, f.receipt.SetupD2HBytes, err = qwen35MTPTransferDelta(before, after)
	return err
}

func (f *Qwen35MTPForward) residentForwardFeedback(pos int, priorHidden, currentEmbedding []float32) (feedback, logits []float32, err error) {
	be := f.draft.Backend
	h := f.draft.M.Cfg.HiddenSize
	if len(priorHidden) != h {
		return nil, nil, qwen35MTPStateError("prior hidden shape", fmt.Sprintf("[%d]", h), fmt.Sprintf("[%d]", len(priorHidden)))
	}
	if len(currentEmbedding) != h {
		return nil, nil, qwen35MTPStateError("current embedding shape", fmt.Sprintf("[%d]", h), fmt.Sprintf("[%d]", len(currentEmbedding)))
	}

	kvBase := f.draft.halKV.Len()
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("model: Qwen3.8 resident MTP failed closed: %v", recovered)
		}
		if err != nil && f.draft.halClosed {
			f.closed = true
		}
		if err != nil && f.draft.halKV != nil {
			if added := f.draft.halKV.Len() - kvBase; added > 0 {
				func() {
					defer func() {
						if recovered := recover(); recovered != nil {
							f.closed = true
							err = fmt.Errorf("%w; resident draft KV rollback failed: %v", err, recovered)
						}
					}()
					f.draft.halKV.Evict(kvBase, added)
				}()
			}
		}
		f.draft.recycleHALToken()
	}()

	boundaryStart := qwen35MTPTransferNow(f.transferCounter)
	prior := f.draft.uploadHostF32([]int{h}, append([]float32(nil), priorHidden...), compute.MemoryActivation, "qwen35-mtp-prior-hidden")
	embedding := f.draft.uploadHostF32([]int{h}, append([]float32(nil), currentEmbedding...), compute.MemoryActivation, "qwen35-mtp-current-embedding")
	boundaryUploaded := qwen35MTPTransferNow(f.transferCounter)
	h2d, d2h, deltaErr := qwen35MTPTransferDelta(boundaryStart, boundaryUploaded)
	if deltaErr != nil {
		return nil, nil, deltaErr
	}
	f.receipt.BoundaryH2DBytes += h2d
	f.receipt.BoundaryD2HBytes += d2h

	x, err := f.resident.Qwen35MTPFuse(compute.Qwen35MTPFuseRequest{
		PriorHidden: prior, CurrentEmbedding: embedding,
		HiddenNorm:    f.draft.normWeightHAL("mtp.pre_fc_norm_hidden.weight"),
		EmbeddingNorm: f.draft.normWeightHAL("mtp.pre_fc_norm_embedding.weight"),
		FC:            f.draft.matWeightHAL("mtp.fc.weight"), Epsilon: float32(f.draft.M.Cfg.RMSNormEps),
	})
	if err != nil {
		return nil, nil, err
	}
	cfg := f.draft.M.Cfg
	f.draft.qwen35FullAttentionHAL(0, pos, x, float32(cfg.RMSNormEps), cfg.attnScale(), cfg.GroupSize())
	postNorm := f.draft.normWeightHAL("model.layers.0.post_attention_layernorm.weight")
	xn := be.RMSNorm(x, postNorm, float32(cfg.RMSNormEps))
	gate := be.MatMul(f.draft.matWeightHAL("model.layers.0.mlp.gate_proj.weight"), xn)
	up := be.MatMul(f.draft.matWeightHAL("model.layers.0.mlp.up_proj.weight"), xn)
	ff := be.SwiGLU(gate, up)
	down := be.MatMul(f.draft.matWeightHAL("model.layers.0.mlp.down_proj.weight"), ff)
	be.AddInPlace(x, down)
	normalized := be.RMSNorm(x, f.draft.normWeightHAL("model.norm.weight"), float32(cfg.RMSNormEps))
	logitsTensor := be.MatMul(f.draft.lmHeadMatHAL(), normalized)

	residentDone := qwen35MTPTransferNow(f.transferCounter)
	h2d, d2h, deltaErr = qwen35MTPTransferDelta(boundaryUploaded, residentDone)
	if deltaErr != nil {
		return nil, nil, deltaErr
	}
	f.receipt.IntermediateH2DBytes += h2d
	f.receipt.IntermediateD2HBytes += d2h
	if h2d != 0 || d2h != 0 {
		return nil, nil, &compute.UnsupportedQwen35MTPDraftError{Backend: be.Name(), Path: compute.Qwen35MTPDraftPath, Stage: "resident execution", Reason: fmt.Sprintf("intermediate host transfer observed h2d=%d d2h=%d", h2d, d2h)}
	}

	feedback = append([]float32(nil), be.Read(normalized)...)
	logits = append([]float32(nil), be.Read(logitsTensor)...)
	boundaryRead := qwen35MTPTransferNow(f.transferCounter)
	h2d, d2h, deltaErr = qwen35MTPTransferDelta(residentDone, boundaryRead)
	if deltaErr != nil {
		return nil, nil, deltaErr
	}
	f.receipt.BoundaryH2DBytes += h2d
	f.receipt.BoundaryD2HBytes += d2h
	if len(feedback) != h || len(logits) != cfg.VocabSize {
		return nil, nil, fmt.Errorf("model: Qwen3.8 resident MTP returned feedback/logits shapes [%d]/[%d], want [%d]/[%d]", len(feedback), len(logits), h, cfg.VocabSize)
	}
	logitScaleInPlace(logits, cfg)
	f.draft.Cache.appendPosition(pos, -1)
	f.lastPos = pos
	f.receipt.Positions++
	f.receipt.DecoderOperations++
	f.receipt.HeadOperations++
	return feedback, logits, nil
}
