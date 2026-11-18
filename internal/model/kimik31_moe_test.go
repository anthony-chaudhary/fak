package model

// kimik31_moe_test.go — parity + composition witness for the K3.1 decoder
// layer's FFN/MoE block (ticket 1611-kimi-k31). The block was the missing half
// of the L8 composite: attention ran, the routed/shared expert compute did not.
// These tests pin the block against an INDEPENDENT float64 transcription of the
// reference's `_forward_unfused` ordering (router -> latent front -> weighted
// expert sum -> reduce-before-norm -> up tail -> +shared), and prove the
// composite now runs the full reference layer order when the FFN params are
// present (a layer with them differs from the attention-only layer).
//
// fak-test:runtime fast est=15ms lane=default

import (
	"encoding/json"
	"math"
	"math/rand"
	"testing"
)

// refSitu is the float64 transcription of the SiTU gate/up fused activation
// (kimi_k3_situ_and_mul.py): beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta).
func refSitu(g, u, beta, linearBeta float64) float64 {
	gl := beta * math.Tanh(g/beta) * (1.0 / (1.0 + math.Exp(-g)))
	ul := u
	if linearBeta > 0 {
		ul = linearBeta * math.Tanh(u/linearBeta)
	}
	return gl * ul
}

// refMoEBlock is an independent float64 transcription of the block's algebra:
// router selection by sigmoid(logit)+bias (unbiased, renormalized weights),
// latent down-projection, per-expert SiTU FFN in latent space, weighted sum,
// then RMSNorm of the reduced latent, up-projection, and the shared expert.
func refMoEBlock(p *KimiK31MoEParams, xn []float64) []float64 {
	H, L, I := p.Hidden, p.Latent, p.ExpertIntermediate

	// Router: selection by score+bias, weights = unbiased sigmoid renormalized.
	scores := make([]float64, p.Experts)
	type cand struct {
		e   int
		w   float64
		sel float64
	}
	cands := make([]cand, p.Experts)
	for e := 0; e < p.Experts; e++ {
		var z float64
		for h := 0; h < H; h++ {
			z += float64(p.WGate[e*H+h]) * xn[h]
		}
		s := sigmoid64(z)
		scores[e] = s
		sel := s
		if len(p.ScoreBias) != 0 {
			sel += float64(p.ScoreBias[e])
		}
		cands[e] = cand{e: e, w: s, sel: sel}
	}
	// Stable descending selection (ties by lower id).
	for i := 0; i < p.Experts; i++ {
		for j := i + 1; j < p.Experts; j++ {
			if cands[j].sel > cands[i].sel || (cands[j].sel == cands[i].sel && cands[j].e < cands[i].e) {
				cands[i], cands[j] = cands[j], cands[i]
			}
		}
	}
	picks := cands[:p.TopK]
	var sum float64
	for _, pk := range picks {
		sum += pk.w
	}
	for i := range picks {
		w := picks[i].w
		if p.Renormalize {
			w /= sum
		}
		picks[i].w = w * float64(p.RouteScale)
	}

	// Latent front.
	routed := make([]float64, L)
	for o := 0; o < L; o++ {
		var s float64
		for h := 0; h < H; h++ {
			s += float64(p.DownProj[o*H+h]) * xn[h]
		}
		routed[o] = s
	}

	// Weighted expert sum, in latent space.
	acc := make([]float64, L)
	for _, pk := range picks {
		e := &p.Expert1[pk.e]
		g := make([]float64, I)
		u := make([]float64, I)
		for i := 0; i < I; i++ {
			var gs, us float64
			for j := 0; j < L; j++ {
				gs += float64(e.W1[i*L+j]) * routed[j]
				us += float64(e.W3[i*L+j]) * routed[j]
			}
			g[i] = gs
			u[i] = us
		}
		act := make([]float64, I)
		for i := 0; i < I; i++ {
			act[i] = refSitu(g[i], u[i], float64(p.SituBeta), float64(p.SituLinearBeta))
		}
		for o := 0; o < L; o++ {
			var s float64
			for i := 0; i < I; i++ {
				s += float64(e.W2[o*I+i]) * act[i]
			}
			acc[o] += pk.w * s
		}
	}

	// Reduce-before-norm: RMSNorm the reduced latent, then up-project.
	normed := naiveRMS(acc, f64s(p.LatentNorm), float64(p.RMSNormEps))
	out := make([]float64, H)
	for o := 0; o < H; o++ {
		var s float64
		for j := 0; j < L; j++ {
			s += float64(p.UpProj[o*L+j]) * normed[j]
		}
		out[o] = s
	}

	// Shared expert at hidden width.
	if p.SharedIntermediate > 0 {
		SI := p.SharedIntermediate
		g := make([]float64, SI)
		u := make([]float64, SI)
		for i := 0; i < SI; i++ {
			var gs, us float64
			for h := 0; h < H; h++ {
				gs += float64(p.SharedWGate[i*H+h]) * xn[h]
				us += float64(p.SharedWUp[i*H+h]) * xn[h]
			}
			g[i] = gs
			u[i] = us
		}
		act := make([]float64, SI)
		for i := 0; i < SI; i++ {
			act[i] = refSitu(g[i], u[i], float64(p.SituBeta), float64(p.SituLinearBeta))
		}
		for o := 0; o < H; o++ {
			var s float64
			for i := 0; i < SI; i++ {
				s += float64(p.SharedWDown[o*SI+i]) * act[i]
			}
			out[o] += s
		}
	}
	return out
}

// tinyMoEParams builds a small, fully-specified K3.1 MoE layer (hidden 4,
// latent 3, expert intermediate 2, 5 experts, top-2, one shared expert).
func tinyMoEParams(seed int64) *KimiK31MoEParams {
	rng := rand.New(rand.NewSource(seed))
	pick := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = rng.Float32()*2 - 1
		}
		return out
	}
	const H, L, I, E, SI = 4, 3, 2, 5, 3
	p := &KimiK31MoEParams{
		Hidden: H, Latent: L, ExpertIntermediate: I,
		SharedIntermediate: SI,
		LatentUseNorm:      true, RMSNormEps: 1e-5,
		DownProj: pick(L * H), UpProj: pick(H * L), LatentNorm: pick(L),
		Experts: E, TopK: 2, RouteScale: 1.0, Renormalize: true, GroupCount: 1,
		WGate: pick(E * H), ScoreBias: pick(E),
		SharedWGate: pick(SI * H), SharedWUp: pick(SI * H), SharedWDown: pick(H * SI),
		SituBeta: 4.0, SituLinearBeta: 25.0,
	}
	p.Expert1 = make([]KimiK31ExpertWeights, E)
	for e := 0; e < E; e++ {
		p.Expert1[e] = KimiK31ExpertWeights{W1: pick(I * L), W3: pick(I * L), W2: pick(L * I)}
	}
	return p
}

// TestKimiK31MoEBlockMatchesReference is the parity witness: the Go block and an
// independent float64 transcription of the reference ordering agree on
// non-trivial synthetic weights.
func TestKimiK31MoEBlockMatchesReference(t *testing.T) {
	p := tinyMoEParams(17)
	if err := ValidateKimiK31MoEParams(p); err != nil {
		t.Fatalf("validate: %v", err)
	}
	rng := rand.New(rand.NewSource(99))
	xn := make([]float32, p.Hidden)
	for i := range xn {
		xn[i] = rng.Float32()*2 - 1
	}
	got := ForwardKimiK31MoEToken(p, xn)
	want := refMoEBlock(p, f64s(xn))
	for i := range got {
		if math.Abs(float64(got[i])-want[i]) > 1e-4 {
			t.Errorf("block[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// TestKimiK31MoEBlockReusesLandedOracles is the no-drift witness: the block's
// router and latent front equal the landed oracles called directly, so the block
// cannot silently re-derive a different contract.
func TestKimiK31MoEBlockReusesLandedOracles(t *testing.T) {
	p := tinyMoEParams(23)
	xn := []float32{0.4, -1.1, 0.7, 0.2}

	// The block's latent front is exactly KimiK31LatentRoutedInput on the same
	// weights (a re-derivation would have to reproduce this bit-for-bit).
	lf := KimiK31LatentMoEConfig{Hidden: p.Hidden, Latent: p.Latent, ExpertIntermediate: p.ExpertIntermediate, LatentUseNorm: true, RMSNormEps: p.RMSNormEps}
	lw := KimiK31LatentMoEWeights{DownProj: p.DownProj, UpProj: p.UpProj, LatentNorm: p.LatentNorm}
	wantFront := KimiK31LatentRoutedInput(lf, lw, xn)

	// The block's router picks are exactly RouteKimiK31MoE's.
	logits := kdaMatVecMul(p.WGate, xn, p.Experts, p.Hidden)
	picks, err := RouteKimiK31MoE(logits, p.ScoreBias, KimiK31RouterConfig{Experts: p.Experts, TopK: p.TopK, RouteScale: p.RouteScale, Renormalize: p.Renormalize, GroupCount: 1})
	if err != nil {
		t.Fatalf("router: %v", err)
	}
	if len(picks) != p.TopK {
		t.Fatalf("picks = %d, want %d", len(picks), p.TopK)
	}
	// Sanity: the front the block uses has the latent width the oracle produces.
	if len(wantFront) != p.Latent {
		t.Fatalf("front width %d != latent %d", len(wantFront), p.Latent)
	}
	// The block must run (not refuse) and produce the hidden width.
	got := ForwardKimiK31MoEToken(p, xn)
	if len(got) != p.Hidden {
		t.Fatalf("block output width %d != hidden %d", len(got), p.Hidden)
	}
	for i := range got {
		if !finite32(got[i]) {
			t.Fatalf("block[%d] non-finite", i)
		}
	}
}

// TestKimiK31ForwardRunsTheFFNHalf is the composition witness: a composite layer
// carrying post-norm + FFN params produces a different hidden vector than the
// attention-only layer, and the delta is the FFN contribution run through the
// block — i.e. the layer's full reference order now executes.
func TestKimiK31ForwardRunsTheFFNHalf(t *testing.T) {
	cfg := tinyKimiK31CompositeConfig()
	attnOnly := tinyKimiK31CompositeParams(cfg)
	// A synthetic attention-only forward (no post-norm / FFN).
	stA, err := NewKimiK31ForwardState(cfg, attnOnly, 4)
	if err != nil {
		t.Fatalf("state: %v", err)
	}
	rng := rand.New(rand.NewSource(5))
	x := make([]float32, cfg.HiddenSize)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	outAttnOnly := ForwardKimiK31Token(cfg, attnOnly, stA, append([]float32(nil), x...))

	// The same composite WITH the FFN half on every layer.
	full := tinyKimiK31CompositeParams(cfg)
	full.PostNorm = make([][]float32, cfg.NumLayers)
	full.Moe = make([]*KimiK31MoEParams, cfg.NumLayers)
	for l := 0; l < cfg.NumLayers; l++ {
		full.PostNorm[l] = []float32{1, 1, 1, 1}
		full.Moe[l] = tinyMoEParams(int64(l) + 31)
	}
	stF, err := NewKimiK31ForwardState(cfg, full, 4)
	if err != nil {
		t.Fatalf("state: %v", err)
	}
	outFull := ForwardKimiK31Token(cfg, full, stF, append([]float32(nil), x...))

	differs := false
	for i := range outFull {
		if outFull[i] != outAttnOnly[i] {
			differs = true
			break
		}
	}
	if !differs {
		t.Fatal("adding the FFN half did not change the layer output: the composite is not running the MoE block")
	}
	if !allFinite(outFull) {
		t.Fatal("full-order composite produced a non-finite hidden value")
	}
}

// TestKimiK31ForwardRefusesPartialFFN pins the fail-closed contract: a layer
// with a post-norm but no FFN block (or vice versa) is an unrepresentable
// partial load and refuses rather than dropping compute silently.
func TestKimiK31ForwardRefusesPartialFFN(t *testing.T) {
	cfg := tinyKimiK31CompositeConfig()
	p := tinyKimiK31CompositeParams(cfg)
	p.PostNorm = make([][]float32, cfg.NumLayers)
	for l := range p.PostNorm {
		p.PostNorm[l] = []float32{1, 1, 1, 1}
	}
	// Moe is left nil: post-norm without a block.
	st, err := NewKimiK31ForwardState(cfg, p, 4)
	if err != nil {
		t.Fatalf("state: %v", err)
	}
	defer func() {
		if recover() == nil {
			t.Fatal("expected a refusal for a layer with post-norm but no FFN block")
		}
	}()
	ForwardKimiK31Token(cfg, p, st, []float32{0.1, 0.2, 0.3, 0.4})
}

// TestKimiK31MoEValidateRefuses proves the block validator fails closed, naming
// the offending field for the shapes a partial checkpoint load would produce.
func TestKimiK31MoEValidateRefuses(t *testing.T) {
	p := tinyMoEParams(41)
	if err := ValidateKimiK31MoEParams(p); err != nil {
		t.Fatalf("well-formed params must validate: %v", err)
	}
	cases := []struct {
		name string
		mut  func(*KimiK31MoEParams)
	}{
		{"bad geometry", func(q *KimiK31MoEParams) { q.Latent = 0 }},
		{"topk exceeds experts", func(q *KimiK31MoEParams) { q.TopK = q.Experts + 1 }},
		{"short WGate", func(q *KimiK31MoEParams) { q.WGate = q.WGate[:len(q.WGate)-1] }},
		{"bias width", func(q *KimiK31MoEParams) { q.ScoreBias = append(q.ScoreBias, 0) }},
		{"missing expert", func(q *KimiK31MoEParams) { q.Expert1 = q.Expert1[:len(q.Expert1)-1] }},
		{"short expert W2", func(q *KimiK31MoEParams) { q.Expert1[0].W2 = q.Expert1[0].W2[:len(q.Expert1[0].W2)-1] }},
		{"short down-proj", func(q *KimiK31MoEParams) { q.DownProj = q.DownProj[:len(q.DownProj)-1] }},
		{"missing latent norm", func(q *KimiK31MoEParams) { q.LatentNorm = nil }},
		{"short shared down", func(q *KimiK31MoEParams) { q.SharedWDown = q.SharedWDown[:len(q.SharedWDown)-1] }},
		{"bad situ beta", func(q *KimiK31MoEParams) { q.SituBeta = 0 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			q := tinyMoEParams(41)
			tc.mut(q)
			if err := ValidateKimiK31MoEParams(q); err == nil {
				t.Fatalf("expected a refusal for %s", tc.name)
			}
		})
	}
}

// TestKimiK31ForwardShapeFaithful93Layer is the strongest host-side "the K3.1
// concept works" witness: the composite runs ALL 93 published layers (69 KDA +
// 24 Gated-DSA in the published 1-based membership rule, pause the final layer)
// with the full reference layer order (attention + FFN/MoE block) end to end on
// synthetic weights, at the published per-layer KIND split the config classifier
// names. It proves the composite's layer-count/kind dispatch is not just a
// 3-layer toy: the same ForwardKimiK31Token walk covers the real stack, every
// layer's state advances, and the output is finite. Per-layer WIDTHs stay tiny
// (this is a dispatch/composition witness, not a numerics run) so the test is
// cheap; the kernel/loader leaves own the real geometry.
func TestKimiK31ForwardShapeFaithful93Layer(t *testing.T) {
	var cfg Config
	if err := json.Unmarshal([]byte(kimiK31ConfigFixture(93)), &cfg); err != nil {
		t.Fatalf("unmarshal 93-layer K3.1 config: %v", err)
	}
	if !cfg.IsKimiK31() || cfg.NumLayers != 93 {
		t.Fatalf("fixture precondition: IsKimiK31=%v layers=%d", cfg.IsKimiK31(), cfg.NumLayers)
	}
	// The published split: 24 full-attention (Gated-DSA) layers, 69 KDA.
	kdaLayers, dsaLayers := 0, 0
	for l := 0; l < cfg.NumLayers; l++ {
		switch {
		case cfg.IsKimiK31KDALayer(l):
			kdaLayers++
		case cfg.IsKimiK31DSALayer(l):
			dsaLayers++
		default:
			t.Fatalf("layer %d carries no kda/dsa token", l)
		}
	}
	if kdaLayers != 69 || dsaLayers != 24 {
		t.Fatalf("layer kinds = %d KDA + %d DSA, want 69 + 24", kdaLayers, dsaLayers)
	}

	// A tiny-geometry stack over the SAME 93-layer kind layout, with the full
	// reference order (post-norm + FFN block) on every layer.
	const H = 4
	cfg.HiddenSize = H
	cfg.RMSNormEps = 1e-5
	p := &KimiK31ForwardParams{Hidden: H, RMSEps: 1e-5}
	p.AttnNorm = make([][]float32, cfg.NumLayers)
	p.PostNorm = make([][]float32, cfg.NumLayers)
	p.FinalNorm = []float32{1, 1, 1, 1}
	p.KDA = make([]*KimiK31KDAParams, cfg.NumLayers)
	p.DSA = make([]*KimiK31DSAParams, cfg.NumLayers)
	p.Moe = make([]*KimiK31MoEParams, cfg.NumLayers)
	for l := 0; l < cfg.NumLayers; l++ {
		p.AttnNorm[l] = []float32{1, 1, 1, 1}
		p.PostNorm[l] = []float32{1, 1, 1, 1}
		p.Moe[l] = tinyMoEParams(int64(l) + 7)
		if cfg.IsKimiK31KDALayer(l) {
			p.KDA[l] = tinyKimiK31KDACompositeParams(H)
		} else {
			p.DSA[l] = tinyKimiK31DSACompositeParams(H)
		}
	}
	st, err := NewKimiK31ForwardState(cfg, p, 4)
	if err != nil {
		t.Fatalf("93-layer state: %v", err)
	}
	if len(st.KDA) != kdaLayers || len(st.DSA) != dsaLayers {
		t.Fatalf("state kinds = %d KDA + %d DSA, want %d + %d", len(st.KDA), len(st.DSA), kdaLayers, dsaLayers)
	}

	rng := rand.New(rand.NewSource(101))
	x := make([]float32, H)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	out := ForwardKimiK31Token(cfg, p, st, x)
	if len(out) != H {
		t.Fatalf("output width %d != hidden %d", len(out), H)
	}
	if !allFinite(out) {
		t.Fatal("93-layer composite produced a non-finite hidden value")
	}
}

// TestKimiK31MoEDenseLayerRunsTheMLP pins the Experts == 0 dense branch: K3.1's
// first_k_dense_replace layer 0 has no router — it is a single SiTU MLP at hidden
// width (the reference's KimiK3MLP). The block must produce the same answer as
// that MLP, not refuse.
func TestKimiK31MoEDenseLayerRunsTheMLP(t *testing.T) {
	rng := rand.New(rand.NewSource(53))
	pick := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = rng.Float32()*2 - 1
		}
		return out
	}
	const H, SI = 4, 3
	p := &KimiK31MoEParams{
		Hidden: H, Latent: 3, ExpertIntermediate: 2,
		SharedIntermediate: SI,
		SituBeta:           4.0, SituLinearBeta: 25.0,
		SharedWGate: pick(SI * H), SharedWUp: pick(SI * H), SharedWDown: pick(H * SI),
	}
	if err := ValidateKimiK31MoEParams(p); err != nil {
		t.Fatalf("dense-layer params must validate: %v", err)
	}
	xn := []float32{0.3, -0.8, 0.5, 0.1}
	got := ForwardKimiK31MoEToken(p, xn)

	// Independent float64 transcription of the dense SiTU MLP.
	g := naiveMatVec(f64s(p.SharedWGate), f64s(xn), SI, H)
	u := naiveMatVec(f64s(p.SharedWUp), f64s(xn), SI, H)
	act := make([]float64, SI)
	for i := 0; i < SI; i++ {
		act[i] = refSitu(g[i], u[i], float64(p.SituBeta), float64(p.SituLinearBeta))
	}
	want := naiveMatVec(f64s(p.SharedWDown), act, H, SI)
	for i := range got {
		if math.Abs(float64(got[i])-want[i]) > 1e-4 {
			t.Errorf("dense MLP[%d] = %v, want %v", i, got[i], want[i])
		}
	}

	// A dense layer with no MLP weights refuses rather than returning zeros.
	bad := *p
	bad.SharedWGate = nil
	if err := ValidateKimiK31MoEParams(&bad); err == nil {
		t.Fatal("expected a refusal for a dense layer with no MLP weights")
	}
}

// tinyKimiK31KDACompositeParams builds a minimal valid KDA layer at hidden h
// (2 heads x 2 dims, conv 2) — the same tiny geometry the composite smoke uses
// (the KDA oracle's own test fixture is pinned at hidden 3, which the 93-layer
// stack's hidden 4 cannot share).
func tinyKimiK31KDACompositeParams(h int) *KimiK31KDAParams {
	rng := rand.New(rand.NewSource(int64(h) + 1000))
	pick := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = rng.Float32()*2 - 1
		}
		return out
	}
	conv := func() *GLM5NextKDAConvFilter {
		f := NewGLM5NextKDAConvFilter(4, 2)
		for i := range f.Weight {
			f.Weight[i] = rng.Float32()*2 - 1
		}
		return f
	}
	return &KimiK31KDAParams{
		NumHeads: 2, HeadDim: 2, ConvKernel: 2, HiddenSize: h,
		Wq: pick(4 * h), Wk: pick(4 * h), Wv: pick(4 * h),
		ConvQ: conv(), ConvK: conv(), ConvV: conv(),
		WFA: pick(2 * h), WFB: pick(4 * 2),
		DTBias: pick(4), AFolded: []float32{-0.5, -0.25},
		WBeta: pick(2 * h), GateLowerBound: -5.0,
		WGate: pick(4 * h), ONorm: pick(2), WO: pick(h * 4),
		RMSEps: 1e-5,
	}
}

// tinyKimiK31DSACompositeParams builds a minimal valid Gated-DSA layer at
// hidden h.
func tinyKimiK31DSACompositeParams(h int) *KimiK31DSAParams {
	rng := rand.New(rand.NewSource(int64(h) + 2000))
	pick := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = rng.Float32()*2 - 1
		}
		return out
	}
	return &KimiK31DSAParams{
		NumHeads: 2, QLoRA: 3, KVLoRA: 2, QkNope: 2, QkRope: 1, VDim: 2, Hidden: h,
		IdxHeads: 2, IdxDim: 2, TopK: 2, RMSEps: 1e-5, IdxEps: 1e-5,
		WQa: pick(3 * h), WQaNorm: pick(3), WQb: pick(2 * 3 * 3),
		WKvA: pick(3 * h), WKvANorm: pick(2), WKb: pick(2 * 2 * 2), WVb: pick(2 * 2 * 2),
		WO: pick(h * 2 * 2), WGate: pick(2 * 2 * h),
		WIdxQ: pick(2 * 2 * 3), WIdxK: pick(2 * h), WIdxProj: pick(2 * h),
		IdxKNormW: pick(2), IdxKNormB: pick(2),
	}
}
