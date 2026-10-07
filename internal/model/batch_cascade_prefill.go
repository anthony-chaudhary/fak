package model

// batch_cascade_prefill.go — ONE-PASS shared-prefix ("cascade") prefill for a cold
// prompt family: N requests that share a prefix P and diverge in equal-length
// suffixes S_1..S_N.
//
// The pattern (Hydragen, arXiv:2402.05099; FlashInfer's cascade/shared-prefix
// batch): instead of prefilling the prefix first and then the N suffixes in a second
// forward (the leader-then-followers shape every prefix cache — SGLang's radix cache,
// fak's PrefixFlightGroup — falls into on a cold family), stack the P prefix rows and
// all N*S suffix rows into ONE [P+N*S, H] panel and run every per-layer weight GEMM
// (q/k/v, o, gate/up/down) once over it. Each weight row is streamed once for the
// whole family, and the prefix's K/V for a layer is computed exactly once.
//
// What is shared and what is per-lane:
//   - Shared: every projection GEMM, the prefix rows' norm/RoPE/attention, and the
//     prefix K/V (computed once per layer, then appended into each lane's own cache so
//     each lane leaves holding a complete, independently-owned prefix+suffix cache —
//     the same per-lane copy NewBatchFromPrefix's clone pays).
//   - Per lane: suffix row t of lane b attends causally to the prefix K/V plus lane b's
//     own suffix rows 0..t — never another lane's suffix.
//
// Bit-identity contract (f32). matMulBatch row r is bit-for-bit parMatRows(weight,
// row r) independent of the panel's other rows; norm, RoPE and bias are row-local; and
// attention reuses attnPrefillMultiInto, the same kernel the rectangular PrefillEach
// lane uses, over a cache whose row prefix is byte-identical to a fresh prefill's. So
// every lane's last-token logits AND every lane's KV rows are bit-identical to an
// independent Session.Prefill(prefix ++ suffix_b) — the same contract PrefillEach's
// rectangular f32 lane already clears (TestCascadePrefillMatchesIndependentPrefill).
//
// The honest cost. Because the whole family completes in one pass, NO lane has logits
// before the full T(P + N*S) forward finishes: the "leader" a two-phase scheme would
// finish at T(P+S) waits for everyone. Cascade trades the first request's TTFT for the
// family's mean/tail TTFT and makespan; dedupbench reports both sides.
//
// Gating. The lane is exactly the rectangular f32 PrefillEach lane's model class
// (batchRectFastPathOK: PreNorm, dense, non-MLA, non-Qwen35-hybrid, uniform RoPE),
// f32 KV only, KV-prefix-reuse-capable, and the prefix and each suffix are capped at
// batchRectPrefillMaxTokens like a rectangular prompt. Anything else is REFUSED with
// an error naming the reason; there is no silent fallback that would make a
// "cascade" number measure something else.

import "fmt"

// CascadePrefill prefills one shared prefix and n equal-length suffixes in a single
// shared-weight pass and returns an n-lane BatchSession (lane b's cache holds
// prefix ++ suffixes[b]) plus each lane's last-token logits. The model must not be on
// the Q8 weight lane for this call (f32 only).
func (m *Model) CascadePrefill(prefix []int, suffixes [][]int) (*BatchSession, [][]float32, error) {
	if err := m.cascadePrefillRefusal(prefix, suffixes); err != nil {
		return nil, nil, err
	}
	bs := m.NewBatchSession(len(suffixes))
	for b, s := range bs.Seqs {
		if s.Cache.quantized() || s.Cache.linear != nil || s.Cache.Len() != 0 {
			return nil, nil, fmt.Errorf("model: cascade prefill refused: lane %d cache is not a fresh f32 KV cache", b)
		}
	}
	logits := bs.cascadePrefillF32(prefix, suffixes)
	return bs, logits, nil
}

// cascadePrefillRefusal is the gate: nil iff the one-pass lane is proven for this
// model + family shape.
func (m *Model) cascadePrefillRefusal(prefix []int, suffixes [][]int) error {
	cfg := m.Cfg
	if !batchRectFastPathOK(cfg, false) {
		return fmt.Errorf("model: cascade prefill refused for architecture %s: only the PreNorm dense non-MoE non-MLA non-Qwen35-hybrid batch lane is supported", cfg.archFamilyKey())
	}
	if !cfg.KVPrefixReuseSupported() {
		return fmt.Errorf("model: cascade prefill refused for architecture %s: its session state is not the KV cache", cfg.archFamilyKey())
	}
	if len(prefix) == 0 || len(prefix) > batchRectPrefillMaxTokens {
		return fmt.Errorf("model: cascade prefill refused: prefix length %d outside [1,%d]", len(prefix), batchRectPrefillMaxTokens)
	}
	if len(suffixes) < 1 {
		return fmt.Errorf("model: cascade prefill refused: need at least one suffix")
	}
	S := len(suffixes[0])
	if S == 0 || S > batchRectPrefillMaxTokens {
		return fmt.Errorf("model: cascade prefill refused: suffix length %d outside [1,%d]", S, batchRectPrefillMaxTokens)
	}
	for b, s := range suffixes {
		if len(s) != S {
			return fmt.Errorf("model: cascade prefill refused: suffix %d length %d != %d (suffixes must be equal length)", b, len(s), S)
		}
	}
	return nil
}

func (bs *BatchSession) cascadePrefillF32(prefix []int, suffixes [][]int) [][]float32 {
	dispatchWorkers := currentWorkerCount()
	m, cfg := bs.M, bs.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	w := nKV * hd
	qw := nH * hd
	scale := cfg.attnScale()
	P, S, B := len(prefix), len(suffixes[0]), len(suffixes)
	R := P + B*S // total panel rows: the prefix ONCE, then every lane's suffix

	caches := make([]*KVCache, B)
	for b, s := range bs.Seqs {
		caches[b] = s.Cache
	}
	// Row r < P is prefix position r; row P + b*S + t is lane b's position P+t.
	cosR := make([][]float32, R)
	sinR := make([][]float32, R)
	for r := 0; r < P; r++ {
		cosR[r], sinR[r] = ropeRow(cfg, r)
	}
	for t := 0; t < S; t++ {
		c, s := ropeRow(cfg, P+t)
		for b := 0; b < B; b++ {
			cosR[P+b*S+t], sinR[P+b*S+t] = c, s
		}
	}

	X := make([]float32, R*H)
	m.embedRowsInto(X, prefix, H, cfg)
	m.embedRectRowsInto(X[P*H:], suffixes, S, H, cfg)

	prefixCache := []*KVCache{caches[0]}
	prefixBase := []int{0}
	suffixBase := make([]int, B)
	for b := range suffixBase {
		suffixBase[b] = P
	}

	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(s string) string { return layerName(l, s) }

		Xn := make([]float32, R*H)
		wIn := m.tensor(lp("input_layernorm.weight"))
		bIn := m.tensorOptional(lp("input_layernorm.bias"))
		parFor(R, dispatchWorkers, func(lo, hi int) {
			for row := lo; row < hi; row++ {
				copy(Xn[row*H:(row+1)*H], normCfg(X[row*H:(row+1)*H], wIn, bIn, eps, cfg))
			}
		})

		// ONE shared-weight GEMM per projection over prefix + all suffix rows.
		Q := matMulBatch(m.tensor(lp("self_attn.q_proj.weight")), Xn, qw, H, R)
		K := matMulBatch(m.tensor(lp("self_attn.k_proj.weight")), Xn, w, H, R)
		V := matMulBatch(m.tensor(lp("self_attn.v_proj.weight")), Xn, w, H, R)
		for row := 0; row < R; row++ {
			m.applyProjBias(l, Q[row*qw:(row+1)*qw], K[row*w:(row+1)*w], V[row*w:(row+1)*w])
			m.applyLayerQKNorm(l, Q[row*qw:(row+1)*qw], K[row*w:(row+1)*w])
		}

		// Pre-RoPE K: the prefix rows (computed once) then the lane's own suffix rows.
		for b, c := range caches {
			c.Kraw[l] = append(c.Kraw[l], K[:P*w]...)
			c.Kraw[l] = append(c.Kraw[l], K[(P+b*S)*w:(P+(b+1)*S)*w]...)
		}
		parFor(R, dispatchWorkers, func(lo, hi int) {
			for row := lo; row < hi; row++ {
				ropeRowQKInto(Q[row*qw:(row+1)*qw], K[row*w:(row+1)*w], cosR[row], sinR[row], hd, nH, nKV)
			}
		})
		for b, c := range caches {
			c.appendBatchedKV(l, K[:P*w], V[:P*w], P, w)
			c.appendBatchedKV(l, K[(P+b*S)*w:(P+(b+1)*S)*w], V[(P+b*S)*w:(P+(b+1)*S)*w], S, w)
		}

		// Attention. The prefix rows attend once (causally, positions 0..t) — lane 0's
		// cache holds the shared prefix rows at [0,P) and the causal bound never reads its
		// suffix rows. Each suffix row then attends to prefix + its own lane's suffix.
		attnOut := make([]float32, R*qw)
		attnPrefillMultiInto(attnOut[:P*qw], Q[:P*qw], prefixCache, prefixBase, l, P, nH, hd, w, grp, cfg.windowForLayer(l), scale, dot, nil)
		attnPrefillMultiInto(attnOut[P*qw:], Q[P*qw:], caches, suffixBase, l, S, nH, hd, w, grp, cfg.windowForLayer(l), scale, dot, nil)

		O := matMulBatch(m.tensor(lp("self_attn.o_proj.weight")), attnOut, H, qw, R)
		for row := 0; row < R; row++ {
			m.addBiasIfPresent(O[row*H:(row+1)*H], lp("self_attn.o_proj.bias"))
		}
		for i := range X {
			X[i] += O[i]
		}

		Xn2 := make([]float32, R*H)
		wPost := m.tensor(lp("post_attention_layernorm.weight"))
		bPost := m.tensorOptional(lp("post_attention_layernorm.bias"))
		parFor(R, dispatchWorkers, func(lo, hi int) {
			for row := lo; row < hi; row++ {
				copy(Xn2[row*H:(row+1)*H], normCfg(X[row*H:(row+1)*H], wPost, bPost, eps, cfg))
			}
		})
		Down := m.batchedGatedMLP(lp, Xn2, R, H, cfg.IntermediateSize, cfg)
		for i := range X {
			X[i] += Down[i]
		}
	}

	for b, c := range caches {
		for t := 0; t < P; t++ {
			c.appendPosition(t, prefix[t])
		}
		for t := 0; t < S; t++ {
			c.appendPosition(P+t, suffixes[b][t])
		}
	}

	// LM head only on each lane's last suffix row.
	last := make([]int, B)
	for b := range last {
		last[b] = P + b*S + S - 1
	}
	Xnorm, err := selectLMHeadProjectionRows(make([]float32, B*H), X, H, last, lmHeadProjectSampledRows)
	if err != nil {
		panic(err) // geometry validated by cascadePrefillRefusal
	}
	rows := lmHeadProjectedRows(Xnorm, H)
	for b := 0; b < rows; b++ {
		copy(Xnorm[b*H:(b+1)*H], m.finalNorm(Xnorm[b*H:(b+1)*H]))
	}
	Logits := matMulBatch(m.lmHead(), Xnorm, cfg.VocabSize, H, rows)
	return splitScaledLogits(nil, Logits, rows, cfg.VocabSize, cfg)
}
