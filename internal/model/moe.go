package model

import (
	"math"
	"sort"
)

// MoE (Mixture-of-Experts) FFN — the rank-7 structural axis of MODEL-ARCH-SEAM.
//
// The FFN sub-layer is the ONLY thing MoE changes. It is KV-orthogonal: the
// attention block, the KV cache (Cache.K/V/Kraw), Evict, and Clone are untouched,
// because every cache append lives in the attention section and the FFN writes
// only the residual delta. So kvmmu / R2 / R14 are entirely MoE-agnostic.
//
// Two FFN forms share one seam (ffnKind):
//
//   - DenseSwiGLU (Llama default): one gate/up/down SwiGLU. The math here is the
//     VERBATIM dense FFN — same matmul kernel, same loop order — so a dense config
//     (NumExperts==0) is Float32bits-identical to the inline FFN it replaces.
//   - MoE (Mixtral / Qwen3-MoE): a router (gate matmul -> softmax over ALL experts
//     -> top-k select -> optional renorm of the k gates) + per-expert SwiGLU over
//     the routed token + gate-weighted sum into the residual delta.
//
// The dispatch returns the residual DELTA (the value added to x), never mutating
// x — keeping the residual-add at the single FFN call site, exactly as the dense
// inline code did.

// ffnKind is the FFN sub-layer interface: given the post-attention-normed hidden
// xn for one position, return the residual delta to add to x. The mat kernel is
// the layer's f32 matmul (matRows / parMatRows) so the FFN shares the exact
// reduction order of the rest of the block.
type ffnKind interface {
	apply(m *Model, layer int, xn any, mat matKernel) []float32
}

// ffnFor selects the FFN form for the model. Dense is the default; an MoE config
// (NumExperts>0) selects the router path. Derived per call from Config so no extra
// state threads through Session.
func ffnFor(cfg Config) ffnKind {
	if cfg.IsMoE() {
		return moeFFN{}
	}
	if cfg.DenseMLP {
		return denseActivationMLP{}
	}
	return denseSwiGLU{}
}

// ffnForLayer keeps the old config-level default, but lets hybrid MoE checkpoints
// use dense SwiGLU on layers whose manifest carries dense MLP tensors instead of a
// router. Qwen3-MoE tiny fixtures use this pattern: dense layer 0, sparse layer 1.
func (m *Model) ffnForLayer(layer int) ffnKind {
	if !m.Cfg.IsMoE() {
		return ffnFor(m.Cfg)
	}
	if m.Cfg.isGLMMoeDsa() && m.has(routerName(layer)) {
		return glmMoeFFN{}
	}
	if m.Cfg.isMiniMaxSparseAttn() && m.has(routerName(layer)) {
		return minimaxMoeFFN{}
	}
	if _, ok := m.manifest[routerName(layer)]; ok {
		return moeFFN{}
	}
	if _, ok := m.manifest[layerName(layer, "mlp.gate_proj.weight")]; ok {
		if m.Cfg.DenseMLP {
			return denseActivationMLP{}
		}
		return denseSwiGLU{}
	}
	return moeFFN{}
}

// denseSwiGLU is the verbatim dense FFN: g=gate(xn); u=up(xn); g=silu(g)*u;
// delta=down(g). Identical kernel and loop order to the inline SwiGLU it replaces,
// so the dense path stays bit-identical (the load-bearing no-op gate).
type denseSwiGLU struct{}

func (denseSwiGLU) apply(m *Model, layer int, xn any, mat matKernel) []float32 {
	cfg := m.Cfg
	H, I := cfg.HiddenSize, cfg.IntermediateSize
	p := func(s string) string { return layerName(layer, s) }
	g := mat.mul(p("mlp.gate_proj.weight"), xn, I, H)
	u := mat.mul(p("mlp.up_proj.weight"), xn, I, H)
	m.addBiasIfPresent(g, p("mlp.gate_proj.bias"))
	m.addBiasIfPresent(u, p("mlp.up_proj.bias"))
	for i := 0; i < I; i++ {
		g[i] = act(g[i], cfg) * u[i]
	}
	out := mat.mul(p("mlp.down_proj.weight"), mat.prep(g), H, I)
	m.addBiasIfPresent(out, p("mlp.down_proj.bias"))
	return out
}

type denseActivationMLP struct{}

func (denseActivationMLP) apply(m *Model, layer int, xn any, mat matKernel) []float32 {
	cfg := m.Cfg
	H, I := cfg.HiddenSize, cfg.IntermediateSize
	p := func(s string) string { return layerName(layer, s) }
	h := mat.mul(p("mlp.gate_proj.weight"), xn, I, H)
	m.addBiasIfPresent(h, p("mlp.gate_proj.bias"))
	for i := 0; i < I; i++ {
		h[i] = act(h[i], cfg)
	}
	out := mat.mul(p("mlp.down_proj.weight"), mat.prep(h), H, I)
	m.addBiasIfPresent(out, p("mlp.down_proj.bias"))
	return out
}

// expertSwiGLU runs one expert's dense SwiGLU over xn and returns its [H] output.
// It is the per-expert primitive the MoE weighted sum reuses — the same SwiGLU
// arithmetic as the dense path, just over an expert-indexed weight set.
func expertSwiGLU(m *Model, layer, expert int, xn any, mat matKernel) []float32 {
	cfg := m.Cfg
	H, I := cfg.HiddenSize, cfg.IntermediateSize
	g := mat.mul(expertName(layer, expert, "gate_proj.weight"), xn, I, H)
	u := mat.mul(expertName(layer, expert, "up_proj.weight"), xn, I, H)
	m.addBiasIfPresent(g, expertName(layer, expert, "gate_proj.bias"))
	m.addBiasIfPresent(u, expertName(layer, expert, "up_proj.bias"))
	for i := 0; i < I; i++ {
		g[i] = act(g[i], cfg) * u[i]
	}
	out := mat.mul(expertName(layer, expert, "down_proj.weight"), mat.prep(g), H, I)
	m.addBiasIfPresent(out, expertName(layer, expert, "down_proj.bias"))
	return out
}

func expertGPTOSS(m *Model, layer, expert int, xn any, mat matKernel) []float32 {
	cfg := m.Cfg
	H, I := cfg.HiddenSize, cfg.IntermediateSize
	g := mat.mul(expertName(layer, expert, "gate_proj.weight"), xn, I, H)
	u := mat.mul(expertName(layer, expert, "up_proj.weight"), xn, I, H)
	m.addBiasIfPresent(g, expertName(layer, expert, "gate_proj.bias"))
	m.addBiasIfPresent(u, expertName(layer, expert, "up_proj.bias"))
	for i := 0; i < I; i++ {
		gate := g[i]
		if gate > 7 {
			gate = 7
		}
		up := u[i]
		if up > 7 {
			up = 7
		} else if up < -7 {
			up = -7
		}
		glu := gate * sigmoid(1.702*gate)
		g[i] = (up + 1) * glu
	}
	out := mat.mul(expertName(layer, expert, "down_proj.weight"), mat.prep(g), H, I)
	m.addBiasIfPresent(out, expertName(layer, expert, "down_proj.bias"))
	return out
}

// routerName / expertName resolve the MoE tensor names. Mixtral/Qwen3-MoE place
// the router at mlp.gate.weight and each expert's projections at
// mlp.experts.<e>.{gate,up,down}_proj.weight under the model.layers.<l>. prefix.
func routerName(layer int) string {
	return layerName(layer, "mlp.gate.weight")
}

func routerBiasName(layer int) string {
	return layerName(layer, "mlp.gate.bias")
}

func expertName(layer, expert int, suffix string) string {
	return layerName(layer, "mlp.experts."+itoa(expert)+"."+suffix)
}

// routePick is one selected (expert, gate-weight) pair from the router.
type routePick struct {
	expert int
	weight float32
}

// route runs the router for one position and returns the top-k (expert, weight)
// picks in HF order.
//
// HF Mixtral/Qwen3-MoE order, pinned exactly:
//  1. logits = router(xn)                         // [NumExperts]
//  2. probs  = softmax(logits) over ALL experts   // dim over all experts
//  3. top-k  = the k experts with the largest probs
//  4. if NormTopKProb: divide the k gate weights by their sum
//
// The top-k tie-break matches torch.topk: largest value first; on equal values,
// the lower expert index wins (stable). Selecting AFTER the full-width softmax (not
// before) is the load-bearing accumulation-order detail an HF oracle pins.
func route(m *Model, layer int, xn any, mat matKernel) []routePick {
	cfg := m.Cfg
	E, K := cfg.NumExperts, cfg.NumExpertsPerTok
	logits := mat.mul(routerName(layer), xn, E, cfg.HiddenSize)
	m.addBiasIfPresent(logits, routerBiasName(layer))
	if cfg.isGPTOSS() {
		return routeTopKSoftmax(logits, K)
	}
	probs := softmaxOf(logits)

	// Index list sorted by (prob desc, index asc) — torch.topk's stable order.
	idx := make([]int, E)
	for e := range idx {
		idx[e] = e
	}
	sort.SliceStable(idx, func(a, b int) bool {
		return probs[idx[a]] > probs[idx[b]]
	})

	picks := make([]routePick, K)
	var sum float32
	for i := 0; i < K; i++ {
		e := idx[i]
		picks[i] = routePick{expert: e, weight: probs[e]}
		sum += probs[e]
	}
	if cfg.NormTopKProb && sum != 0 {
		for i := range picks {
			picks[i].weight /= sum
		}
	}
	return picks
}

func routeTopKSoftmax(logits []float32, k int) []routePick {
	idx := make([]int, len(logits))
	for e := range idx {
		idx[e] = e
	}
	sort.SliceStable(idx, func(a, b int) bool {
		return logits[idx[a]] > logits[idx[b]]
	})
	if k > len(idx) {
		k = len(idx)
	}
	picks := make([]routePick, k)
	top := make([]float32, k)
	for i := 0; i < k; i++ {
		top[i] = logits[idx[i]]
	}
	probs := softmaxOf(top)
	for i := 0; i < k; i++ {
		picks[i] = routePick{expert: idx[i], weight: probs[i]}
	}
	return picks
}

// softmaxOf is the allocating softmax used by the router (the in-place
// softmaxInPlace is for attention scores). Max-subtracted for numerical stability,
// matching HF's F.softmax in f32.
func softmaxOf(z []float32) []float32 {
	out := make([]float32, len(z))
	mx := z[0]
	for _, v := range z {
		if v > mx {
			mx = v
		}
	}
	var sum float32
	for i, v := range z {
		e := float32(math.Exp(float64(v - mx)))
		out[i] = e
		sum += e
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func sigmoid(z float32) float32 {
	return 1 / (1 + float32(math.Exp(float64(-z))))
}

// moeFFN is the Mixture-of-Experts FFN: route to top-k experts, run each selected
// expert's SwiGLU over xn, and accumulate the gate-weighted outputs into the
// residual delta.
type moeFFN struct{}

func (moeFFN) apply(m *Model, layer int, xn any, mat matKernel) []float32 {
	H := m.Cfg.HiddenSize
	delta := make([]float32, H)
	// Accumulate in selection order (highest gate weight first) so the reduction
	// order is fixed and reproducible across runs.
	for _, pk := range route(m, layer, xn, mat) {
		var out []float32
		if m.Cfg.isGPTOSS() {
			out = expertGPTOSS(m, layer, pk.expert, xn, mat)
		} else {
			out = expertSwiGLU(m, layer, pk.expert, xn, mat)
		}
		for i := 0; i < H; i++ {
			delta[i] += pk.weight * out[i]
		}
	}
	return delta
}

type glmMoeFFN struct{}

func (glmMoeFFN) apply(m *Model, layer int, xn any, mat matKernel) []float32 {
	cfg := m.Cfg
	H := cfg.HiddenSize
	delta := make([]float32, H)
	for _, pk := range glmRoute(m, layer, xn, mat) {
		out := expertSwiGLU(m, layer, pk.expert, xn, mat)
		for i := 0; i < H; i++ {
			delta[i] += pk.weight * out[i]
		}
	}
	if cfg.NSharedExperts > 0 && m.has(layerName(layer, "mlp.shared_experts.gate_proj.weight")) {
		shared := glmSharedExperts(m, layer, xn, mat)
		for i := 0; i < H; i++ {
			delta[i] += shared[i]
		}
	}
	return delta
}

func glmRoute(m *Model, layer int, xn any, mat matKernel) []routePick {
	cfg := m.Cfg
	E, K := cfg.NumExperts, cfg.NumExpertsPerTok
	logits := mat.mul(routerName(layer), xn, E, cfg.HiddenSize)
	rawWeights := make([]float32, E)
	choice := make([]float32, E)
	correction := m.tensorOptional(layerName(layer, "mlp.gate.e_score_correction_bias"))
	for e, z := range logits {
		w := sigmoid(z)
		rawWeights[e] = w
		choice[e] = w
		if correction != nil {
			choice[e] += correction[e]
		}
	}

	nGroup := cfg.NGroup
	if nGroup <= 0 {
		nGroup = 1
	}
	if nGroup > E {
		nGroup = E
	}
	perGroup := E / nGroup
	topKGroup := cfg.TopKGroup
	if topKGroup <= 0 || topKGroup > nGroup {
		topKGroup = nGroup
	}
	groupScores := make([]float32, nGroup)
	for g := 0; g < nGroup; g++ {
		start := g * perGroup
		end := start + perGroup
		if g == nGroup-1 {
			end = E
		}
		groupScores[g] = sumTopK(choice[start:end], 2)
	}
	groups := make([]int, nGroup)
	for g := range groups {
		groups[g] = g
	}
	sort.SliceStable(groups, func(i, j int) bool {
		return groupScores[groups[i]] > groupScores[groups[j]]
	})
	allowedGroup := make([]bool, nGroup)
	for i := 0; i < topKGroup; i++ {
		allowedGroup[groups[i]] = true
	}

	candidates := make([]int, 0, E)
	for e := 0; e < E; e++ {
		g := e / perGroup
		if g >= nGroup {
			g = nGroup - 1
		}
		if allowedGroup[g] {
			candidates = append(candidates, e)
		}
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		return choice[candidates[i]] > choice[candidates[j]]
	})
	if K > len(candidates) {
		K = len(candidates)
	}
	picks := make([]routePick, K)
	var sum float32
	for i := 0; i < K; i++ {
		e := candidates[i]
		picks[i] = routePick{expert: e, weight: rawWeights[e]}
		sum += picks[i].weight
	}
	if cfg.NormTopKProb && sum != 0 {
		for i := range picks {
			picks[i].weight /= sum
		}
	}
	scale := float32(cfg.RoutedScalingFactor)
	if scale == 0 {
		scale = 1
	}
	for i := range picks {
		picks[i].weight *= scale
	}
	sort.SliceStable(picks, func(i, j int) bool {
		return picks[i].expert < picks[j].expert
	})
	return picks
}

func sumTopK(xs []float32, k int) float32 {
	if k > len(xs) {
		k = len(xs)
	}
	idx := make([]int, len(xs))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(i, j int) bool {
		return xs[idx[i]] > xs[idx[j]]
	})
	var sum float32
	for i := 0; i < k; i++ {
		sum += xs[idx[i]]
	}
	return sum
}

func glmSharedExperts(m *Model, layer int, xn any, mat matKernel) []float32 {
	cfg := m.Cfg
	H := cfg.HiddenSize
	I := cfg.MoEIntermediateSize * cfg.NSharedExperts
	if I == 0 {
		I = cfg.IntermediateSize
	}
	prefix := layerName(layer, "mlp.shared_experts.")
	g := mat.mul(prefix+"gate_proj.weight", xn, I, H)
	u := mat.mul(prefix+"up_proj.weight", xn, I, H)
	for i := 0; i < I; i++ {
		g[i] = act(g[i], cfg) * u[i]
	}
	return mat.mul(prefix+"down_proj.weight", mat.prep(g), H, I)
}
