package model

// kimik31_kda.go — the Kimi-K3.1 KDA (Kimi Delta Attention) layer forward,
// host oracle (ticket 1611-kimi-k31 leaf L5).
//
// K3.1's 69 KDA layers are byte-for-byte the K3 KDA layer (llama.cpp master
// src/models/kimi-k31.cpp: "KDA layers, cross-layer residuals, latent MoE,
// situ and the MLA shapes are unchanged"); only the 24 full-attention layers
// gain the DSA indexer (leaf L6). This file implements ONE KDA layer's
// forward as a closed host oracle the L8 composite will call per layer,
// extending the existing KDA machinery (GLM5NextKDALayerState,
// GLM5NextKDAConvFilter, kdaMatVecMul, the qwen35 GDN primitives
// l2normInto/sigmoidf/softplus/silu) rather than inventing parallel state
// plumbing.
//
// REFERENCE (recurrence pinned against concordant sources, fetched 2026-11-07):
//   - llama.cpp master src/models/kimi-k3.cpp build_kda_layer +
//     build_kda_conv1d: the gate forms, L2-norm placement, the gated-RMSNorm
//     readout, the full-rank gate, and the per-stream conv wiring;
//   - llama.cpp master src/models/delta-net-base.cpp
//     build_delta_net_autoregressive: the per-token recurrence — the
//     per-value-channel decay, predict/delta/update/readout order, and the
//     1/sqrt(d) query scale applied AFTER L2 normalization;
//   - llama.cpp master src/models/models.h build_gdn_l2_norm:
//     x * rsqrt(sum(x^2) + eps) (FLA-style L2 norm, sum not mean);
//   - llama.cpp master conversion/kimi_k3.py: the stored ssm.a IS the
//     converter-folded -exp(A_log) (negative), and gate_lower_bound reaches
//     the GGUF only when the HF config carries it;
//   - SGLang kimi-k3.1-day-0 python/sglang/srt/models/kimi_linear.py
//     (KimiDeltaAttention): the same structure — q/k/v projections +
//     per-stream short conv, factored forget gate f_b(f_a(x))+dt_bias,
//     FusedRMSNormGated(head_dim, eps=rms_norm_eps, activation="sigmoid")
//     readout, safe_gate/lower_bound plumbing;
//   - moonshotai/Kimi-K3.1-Pro config.json: linear_attn_config
//     {gate_lower_bound -5.0, head_dim 128, num_heads 96,
//     short_conv_kernel_size 4, use_full_rank_gate true}, rms_norm_eps 1e-5.
//
// PER-TOKEN CONTRACT (x is the ALREADY input-layernorm'd layer input,
// [HiddenSize]; the composite owns attn_norm/res_mix, exactly as qwen35's
// linearAttnSeq takes normalized inputs and llama.cpp feeds build_kda_layer
// its post-attn_norm cur):
//  1. projections: qRaw=Wq x, kRaw=Wk x, vRaw=Wv x (each [dInner]);
//     fa = Wfa x ([d]); fRaw = Wfb fa + dtBias ([dInner]);
//     betaRaw = Wb x ([NH]); gateRaw = Wg x ([dInner], the full-rank output
//     gate — no conv on the gate stream).
//  2. causal depthwise conv1d + SiLU per stream (kernel K, left-padded, no
//     bias; the state keeps the last K-1 PRE-conv rows, GLM5NextKDAConvFilter
//     layout: tap 0 = oldest, tap K-1 = current).
//  3. decay log-gate, per VALUE channel c of each head h (raw = fRaw[h*d+c],
//     expAlog_h = -AFolded_h because the stored ssm.a IS -exp(A_log)):
//       GateLowerBound != 0 (K3/K3.1, -5.0): g = lb*sigmoid(expAlog_h*raw)
//         (the A multiply happens BEFORE the sigmoid, lb AFTER — the
//         "safe-gate" form; note lb < 0 so g in (lb, 0));
//       GateLowerBound == 0 (GGUF key absent, kimi-linear):
//         g = AFolded_h*softplus(raw)
//     alpha_c = exp(g), in (0, 1].
//  4. beta_h = sigmoid(betaRaw_h); L2-normalize q/k per head (FLA form
//     x*rsqrt(sum(x^2)+eps), eps = RMSEps) then query scale qhat = q/ sqrt(d).
//  5. delta-rule step per head, state S[h] row-major S[i*d+c] (key dim i x
//     value channel c — the llama.cpp state [S_v, S_v] orientation):
//     S[i,c] *= alpha_c;
//     kv_c = sum_i S[i,c]*k[i];
//     delta_c = beta_h*(v_c - kv_c);
//     S[i,c] += k[i]*delta_c;
//     o_c = sum_i S[i,c]*qhat[i].
//  6. readout: y_h = o_norm * rmsnorm(o_h, eps) * sigmoid(gateRaw_h) per head
//     (RMSNorm WITH weight, mean not sum), out = Wo * concat(y).
//
// DEFERRED (later leaves, no surface invented here): a chunked/carry prefill
// (this oracle scans token-by-token; L8 may add the carry form), quantized
// weight kernels (L8 wires matKernel), and the ClassifyForwardPath ->
// ForwardKimiK31 dispatch (L8 retires the refusal). Activation parity vs a
// runnable reference dump (PF-11) stays gated on a staged K3.1 artifact, as
// the study records; this leaf pins the recurrence contract itself.

import (
	"errors"
	"fmt"
	"math"
	"strconv"
)

// KimiK31LayerTypeKDA and KimiK31LayerTypeDSA are the layer_types tokens the
// L1 admission (ggufload.applyKimiK31Config) writes for K3.1 checkpoints;
// the model-side dispatch (L8) switches on them. They deliberately differ
// from the qwen35 "linear_attention"/"full_attention" vocabulary so a K3.1
// checkpoint never routes into the GDN forward.
const (
	KimiK31LayerTypeKDA = "kda" // Kimi Delta Attention (this file)
	KimiK31LayerTypeDSA = "dsa" // Gated DeepSeek Sparse Attention (leaf L6)
)

// IsKimiK31KDALayer reports whether decoder layer l is a KDA layer per
// config.layer_types.
func (c Config) IsKimiK31KDALayer(l int) bool {
	return l >= 0 && l < len(c.LayerTypes) && c.LayerTypes[l] == KimiK31LayerTypeKDA
}

// IsKimiK31DSALayer reports whether decoder layer l is a Gated-DSA layer per
// config.layer_types.
func (c Config) IsKimiK31DSALayer(l int) bool {
	return l >= 0 && l < len(c.LayerTypes) && c.LayerTypes[l] == KimiK31LayerTypeDSA
}

// KimiK31KDAWeightNames pins the canonical tensor names (the L2 map,
// internal/ggufload/kimik31_tensors.go) one KDA layer's forward consumes.
// L8 resolves these through Model.tensor; the oracle itself takes plain
// slices so the recurrence stays testable without a loaded checkpoint.
type KimiK31KDAWeightNames struct {
	QProj, KProj, VProj string
	QConv, KConv, VConv string
	FAProj, FBProj      string
	BetaProj            string
	// ALog carries the CONVERTER-FOLDED -exp(A_log) value and has no
	// ".weight" suffix (L2 A_LOG DOMAIN NOTE).
	ALog   string
	DTBias string
	// GateProj is the full-rank output gate (use_full_rank_gate: true).
	GateProj string
	ONorm    string
	OProj    string
}

// KimiK31KDAWeightNamesFor returns the canonical per-layer KDA tensor names
// for decoder layer l.
func KimiK31KDAWeightNamesFor(layer int) KimiK31KDAWeightNames {
	p := "model.layers." + strconv.Itoa(layer) + ".self_attn."
	return KimiK31KDAWeightNames{
		QProj:    p + "q_proj.weight",
		KProj:    p + "k_proj.weight",
		VProj:    p + "v_proj.weight",
		QConv:    p + "q_conv1d.weight",
		KConv:    p + "k_conv1d.weight",
		VConv:    p + "v_conv1d.weight",
		FAProj:   p + "f_a_proj.weight",
		FBProj:   p + "f_b_proj.weight",
		BetaProj: p + "b_proj.weight",
		ALog:     p + "A_log",
		DTBias:   p + "dt_bias",
		GateProj: p + "g_proj.weight",
		ONorm:    p + "o_norm",
		OProj:    p + "o_proj.weight",
	}
}

// KimiK31KDAParams holds the per-layer weights and scalars for one K3.1 KDA
// layer. Geometry fields mirror the state (NumHeads 96, HeadDim 128,
// ConvKernel 4, HiddenSize 7168 for K3.1); the oracle works at any
// consistent geometry so the recurrence stays testable on tiny tensors.
type KimiK31KDAParams struct {
	NumHeads   int
	HeadDim    int
	ConvKernel int
	HiddenSize int

	Wq     []float32 // [dInner * HiddenSize]
	Wk     []float32 // [dInner * HiddenSize]
	Wv     []float32 // [dInner * HiddenSize]
	ConvQ  *GLM5NextKDAConvFilter
	ConvK  *GLM5NextKDAConvFilter
	ConvV  *GLM5NextKDAConvFilter
	WFA    []float32 // [HeadDim * HiddenSize] (f_a_proj)
	WFB    []float32 // [dInner * HeadDim]     (f_b_proj)
	DTBias []float32 // [dInner]
	// AFolded[h] is the STORED ssm.a value: the converter-folded -exp(A_log),
	// strictly negative. Passing raw A_log (positive) trips validation.
	AFolded []float32 // [NumHeads]
	WBeta   []float32 // [NumHeads * HiddenSize] (b_proj)
	// GateLowerBound selects the decay-gate form: K3.1 ships -5.0 (safe-gate
	// sigmoid form); 0 means the GGUF key is absent (kimi-linear softplus
	// form). It is not a clamp.
	GateLowerBound float32
	WGate          []float32 // [dInner * HiddenSize] (g_proj, full-rank gate)
	ONorm          []float32 // [HeadDim]             (o_norm, RMSNorm weight)
	WO             []float32 // [HiddenSize * dInner] (o_proj)
	RMSEps         float32   // 1e-5 for K3.1
}

// ErrKimiK31KDAParamShape is the closed sentinel for fail-closed KDA param
// shape/domain validation failures.
var ErrKimiK31KDAParamShape = errors.New("model: kimi-k31 KDA param shape mismatch")

// ValidateKimiK31KDAParams fails closed on any malformed params, naming the
// offending field. Call it once when building params from tensors (L8); the
// forward assumes validated params and only guards impossible internal
// inconsistencies.
func ValidateKimiK31KDAParams(p *KimiK31KDAParams) error {
	if p == nil {
		return fmt.Errorf("%w: nil params", ErrKimiK31KDAParamShape)
	}
	if p.NumHeads <= 0 || p.HeadDim <= 0 || p.ConvKernel <= 0 || p.HiddenSize <= 0 {
		return fmt.Errorf("%w: bad geometry (heads=%d dim=%d conv=%d hidden=%d)",
			ErrKimiK31KDAParamShape, p.NumHeads, p.HeadDim, p.ConvKernel, p.HiddenSize)
	}
	if p.RMSEps <= 0 {
		return fmt.Errorf("%w: RMSEps %v must be > 0", ErrKimiK31KDAParamShape, p.RMSEps)
	}
	dInner := p.NumHeads * p.HeadDim
	if len(p.Wq) != dInner*p.HiddenSize {
		return fmt.Errorf("%w: Wq len %d != dInner*hidden %d", ErrKimiK31KDAParamShape, len(p.Wq), dInner*p.HiddenSize)
	}
	if len(p.Wk) != dInner*p.HiddenSize {
		return fmt.Errorf("%w: Wk len %d != dInner*hidden %d", ErrKimiK31KDAParamShape, len(p.Wk), dInner*p.HiddenSize)
	}
	if len(p.Wv) != dInner*p.HiddenSize {
		return fmt.Errorf("%w: Wv len %d != dInner*hidden %d", ErrKimiK31KDAParamShape, len(p.Wv), dInner*p.HiddenSize)
	}
	if len(p.WFA) != p.HeadDim*p.HiddenSize {
		return fmt.Errorf("%w: WFA len %d != headDim*hidden %d", ErrKimiK31KDAParamShape, len(p.WFA), p.HeadDim*p.HiddenSize)
	}
	if len(p.WFB) != dInner*p.HeadDim {
		return fmt.Errorf("%w: WFB len %d != dInner*headDim %d", ErrKimiK31KDAParamShape, len(p.WFB), dInner*p.HeadDim)
	}
	if len(p.DTBias) != dInner {
		return fmt.Errorf("%w: DTBias len %d != dInner %d", ErrKimiK31KDAParamShape, len(p.DTBias), dInner)
	}
	if len(p.AFolded) != p.NumHeads {
		return fmt.Errorf("%w: AFolded len %d != numHeads %d", ErrKimiK31KDAParamShape, len(p.AFolded), p.NumHeads)
	}
	for i, a := range p.AFolded {
		if a >= 0 {
			return fmt.Errorf("%w: AFolded[%d]=%v must be < 0 (stored ssm.a is the converter-folded -exp(A_log))",
				ErrKimiK31KDAParamShape, i, a)
		}
	}
	if len(p.WBeta) != p.NumHeads*p.HiddenSize {
		return fmt.Errorf("%w: WBeta len %d != numHeads*hidden %d", ErrKimiK31KDAParamShape, len(p.WBeta), p.NumHeads*p.HiddenSize)
	}
	if len(p.WGate) != dInner*p.HiddenSize {
		return fmt.Errorf("%w: WGate len %d != dInner*hidden %d", ErrKimiK31KDAParamShape, len(p.WGate), dInner*p.HiddenSize)
	}
	if len(p.ONorm) != p.HeadDim {
		return fmt.Errorf("%w: ONorm len %d != headDim %d", ErrKimiK31KDAParamShape, len(p.ONorm), p.HeadDim)
	}
	if len(p.WO) != p.HiddenSize*dInner {
		return fmt.Errorf("%w: WO len %d != hidden*dInner %d", ErrKimiK31KDAParamShape, len(p.WO), p.HiddenSize*dInner)
	}
	convCheck := func(name string, f *GLM5NextKDAConvFilter) error {
		if f == nil {
			return fmt.Errorf("%w: %s is nil", ErrKimiK31KDAParamShape, name)
		}
		if f.Dim != dInner || f.FilterWidth != p.ConvKernel || len(f.Weight) != dInner*p.ConvKernel {
			return fmt.Errorf("%w: %s dims (%d,%d,len %d) != (dInner %d, kernel %d)",
				ErrKimiK31KDAParamShape, name, f.Dim, f.FilterWidth, len(f.Weight), dInner, p.ConvKernel)
		}
		return nil
	}
	if err := convCheck("ConvQ", p.ConvQ); err != nil {
		return err
	}
	if err := convCheck("ConvK", p.ConvK); err != nil {
		return err
	}
	if err := convCheck("ConvV", p.ConvV); err != nil {
		return err
	}
	return nil
}

// ForwardKimiK31KDAStep runs one KDA layer over one token. x is the already
// input-layernorm'd layer input ([HiddenSize]); st carries the recurrent S
// matrices and conv histories (built with NewGLM5NextKDALayerState matching
// p's geometry); it returns the layer output ([HiddenSize], post o_proj) and
// updates st in place. Params must have passed ValidateKimiK31KDAParams.
func ForwardKimiK31KDAStep(st *GLM5NextKDALayerState, p *KimiK31KDAParams, x []float32) []float32 {
	if st == nil || p == nil {
		panic("model: kimik31 kda step: nil state or params")
	}
	if len(x) != p.HiddenSize {
		panic(fmt.Sprintf("model: kimik31 kda step: input length %d != HiddenSize %d", len(x), p.HiddenSize))
	}
	if st.NumHeads != p.NumHeads || st.HeadDim != p.HeadDim || st.ConvWindow != p.ConvKernel {
		panic(fmt.Sprintf("model: kimik31 kda step: state geometry (%d,%d,%d) != params (%d,%d,%d)",
			st.NumHeads, st.HeadDim, st.ConvWindow, p.NumHeads, p.HeadDim, p.ConvKernel))
	}

	d := p.HeadDim
	dInner := p.NumHeads * d
	hid := p.HiddenSize

	// 1. Projections.
	qRaw := kdaMatVecMul(p.Wq, x, dInner, hid)
	kRaw := kdaMatVecMul(p.Wk, x, dInner, hid)
	vRaw := kdaMatVecMul(p.Wv, x, dInner, hid)
	fa := kdaMatVecMul(p.WFA, x, d, hid)
	fRaw := kdaMatVecMul(p.WFB, fa, dInner, d)
	for i := range fRaw {
		fRaw[i] += p.DTBias[i]
	}
	betaRaw := kdaMatVecMul(p.WBeta, x, p.NumHeads, hid)
	gateRaw := kdaMatVecMul(p.WGate, x, dInner, hid)

	// 2. Causal depthwise conv1d + SiLU per stream (state keeps the last
	// K-1 pre-conv rows).
	qc := p.ConvQ.Step(qRaw, st.ConvQ)
	kc := p.ConvK.Step(kRaw, st.ConvK)
	vc := p.ConvV.Step(vRaw, st.ConvV)

	// 3-6. Per-head gates, delta-rule step, gated readout.
	out := make([]float32, dInner)
	od := make([]float32, d)
	alpha := make([]float32, d)
	qn := make([]float32, d)
	kn := make([]float32, d)
	qScale := float32(1.0 / math.Sqrt(float64(d)))

	for h := 0; h < p.NumHeads; h++ {
		base := h * d

		// 4. L2-normalize q/k per head, then the 1/sqrt(d) query scale.
		l2normInto(qn, qc[base:base+d], p.RMSEps)
		l2normInto(kn, kc[base:base+d], p.RMSEps)
		for i := range qn {
			qn[i] *= qScale
		}

		// 3. Decay log-gates -> alphas, per value channel.
		if p.GateLowerBound != 0 {
			// Safe-gate form (K3/K3.1): g = lb * sigmoid(exp(A_log)*raw),
			// with exp(A_log) = -AFolded (stored ssm.a is -exp(A_log)).
			lb := p.GateLowerBound
			expAlog := -p.AFolded[h]
			for c := 0; c < d; c++ {
				alpha[c] = float32(math.Exp(float64(lb * sigmoidf(expAlog*fRaw[base+c]))))
			}
		} else {
			// Key-absent form (kimi-linear): g = -exp(A_log) * softplus(raw).
			aFold := p.AFolded[h]
			for c := 0; c < d; c++ {
				alpha[c] = float32(math.Exp(float64(aFold * softplus(fRaw[base+c]))))
			}
		}

		beta := sigmoidf(betaRaw[h])

		// 5. Delta-rule step on S[h] (row-major S[i*d+c]).
		kimiK31KDAHeadStep(st.HeadMatrix(h), d, qn, kn, vc[base:base+d], alpha, beta, od)

		// 6. Gated RMSNorm readout: y = o_norm * rmsnorm(o, eps) * sigmoid(gateRaw).
		var ss float32
		for _, v := range od {
			ss += v * v
		}
		inv := float32(1.0 / math.Sqrt(float64(ss)/float64(d)+float64(p.RMSEps)))
		for c := 0; c < d; c++ {
			out[base+c] = od[c] * inv * p.ONorm[c] * sigmoidf(gateRaw[base+c])
		}
	}

	return kdaMatVecMul(p.WO, out, hid, dInner)
}

// ForwardKimiK31KDAPrefillSeq runs T tokens causally through the layer and
// returns the outputs for all T tokens ([T*HiddenSize]). The oracle scans
// token-by-token; a chunked/carry form is deferred to L8.
func ForwardKimiK31KDAPrefillSeq(st *GLM5NextKDALayerState, p *KimiK31KDAParams, xSeq []float32, T int) []float32 {
	if len(xSeq) != T*p.HiddenSize {
		panic(fmt.Sprintf("model: kimik31 kda prefill: xSeq length %d != T*HiddenSize %d", len(xSeq), T*p.HiddenSize))
	}
	hid := p.HiddenSize
	out := make([]float32, T*hid)
	for t := 0; t < T; t++ {
		o := ForwardKimiK31KDAStep(st, p, xSeq[t*hid:(t+1)*hid])
		copy(out[t*hid:(t+1)*hid], o)
	}
	return out
}

// kimiK31KDAHeadStep performs one delta-rule step for head h's state matrix
// S (row-major S[i*d+c], key dim i x value channel c — the llama.cpp state
// [S_v, S_v] orientation), updating S in place and writing the readout into
// od (q must already carry the 1/sqrt(d) scale):
//
//	S[i,c] *= alpha[c]                 (per-value-channel decay)
//	kv[c]  = sum_i S[i,c]*k[i]         (predict)
//	delta[c] = beta*(v[c]-kv[c])       (delta)
//	S[i,c] += k[i]*delta[c]            (rank-1 update)
//	od[c]  = sum_i S[i,c]*q[i]         (readout, on the UPDATED state)
func kimiK31KDAHeadStep(S []float32, d int, q, k, v, alpha []float32, beta float32, od []float32) {
	// 1. Decay.
	for i := 0; i < d; i++ {
		row := i * d
		for c := 0; c < d; c++ {
			S[row+c] *= alpha[c]
		}
	}
	// 2. Predict + 3. delta.
	kv := make([]float32, d)
	for i := 0; i < d; i++ {
		ki := k[i]
		if ki == 0 {
			continue
		}
		row := i * d
		for c := 0; c < d; c++ {
			kv[c] += S[row+c] * ki
		}
	}
	dl := make([]float32, d)
	for c := 0; c < d; c++ {
		dl[c] = beta * (v[c] - kv[c])
	}
	// 4. Rank-1 update.
	for i := 0; i < d; i++ {
		ki := k[i]
		if ki == 0 {
			continue
		}
		row := i * d
		for c := 0; c < d; c++ {
			S[row+c] += ki * dl[c]
		}
	}
	// 5. Readout.
	for c := range od {
		od[c] = 0
	}
	for i := 0; i < d; i++ {
		qi := q[i]
		if qi == 0 {
			continue
		}
		row := i * d
		for c := 0; c < d; c++ {
			od[c] += S[row+c] * qi
		}
	}
}
