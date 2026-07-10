package model

// rescore.go — issue #2626: the cheap query re-score over Kraw (Tier 2a of
// docs/notes/CONTEXT-VIEWS-AT-MARGINAL-COST-2026-07-04.md).
//
// "Which of my stored spans are relevant to THIS new query" cannot be answered from
// the attention mass the observer already witnessed — that mass is about the OLD
// queries (attention is query-dependent). Answering it fresh requires re-attending.
// The position-independent-caching result (Prompt Cache / PIC) is that the cheap way
// re-uses the cached keys and pays only the new query's Q projection plus one
// QK^T·softmax read, skipping the K/V projection of the whole candidate subset. The
// enabling trick — unrotated keys, RoPE applied at attention time — is exactly the
// Kraw stash the exact-eviction primitive already keeps (kvcache.go), so this file
// adds exposure, not numerics: it re-rotates candidate Kraw rows into SCRATCH (the
// same single-rotation op Evict uses in place) and runs the same dot/softcap/softmax
// score math blockStep runs.
//
// HONEST MODE LABEL (the CacheBlend caveat): this is a LAYER-0 relevance signal. The
// probe's Q is computed only through the first decoder block's pre-attention norm and
// q_proj — no deeper layers, no MLP mixing — because a deeper Q would need the full
// forward pass this path exists to skip. The top-k oracle in
// internal/kvmmu/rescore_test.go adjudicated it against a full re-attend on the
// deterministic synthetic fixture: the layer-0 score recovers the full re-attend's
// top-k SET, but the order of near-tied leaders flips between tiers. Treat the result
// as a NARROWING signal (which spans to keep / re-attend), not a total order; exact
// ranking within the selected subset needs the full re-attend (Tier 2b).
//
// INVARIANT: the cached Kraw/K/V are never written — re-rotation targets a scratch
// copy (asserted bit-identical before/after by the kvmmu oracle test).

import (
	"errors"
	"fmt"
)

// ErrReScoreUnsupported is the typed refusal ReScoreSpans returns for a cache/model
// layout whose layer-0 keys cannot be re-rotated for scoring: an MLA/DSA layout (no
// dense per-position Kraw rows), an ALiBi model (relevance would need the position
// bias this path deliberately drops), a hybrid whose layer 0 is a recurrent
// linear-attention layer (no Kraw at layer 0), or a resident-quantized model with no
// f32 q_proj to project the probe through. Fail-closed: no partial score is returned.
var ErrReScoreUnsupported = errors.New("model: ReScoreSpans unsupported for this model/cache layout")

// ReScoreSpans scores each candidate span's cached keys against a NEW probe query —
// position-independent relevance over the kernel-owned cache, without a forward pass
// and without mutating it.
//
// probe is the new query's token ids. spans are candidate [from, n) position ranges
// into this session's cache (the kvmmu ledger's From/Len pairs). The candidates are
// laid out contiguously at scoring positions [0, T) in argument order (T = Σn), each
// key re-derived from its pre-RoPE Kraw row in ONE rotation at its scoring position —
// bit-exact to a prefill that saw only the candidates, by the same argument Evict's
// survivor re-rotation is. The probe's layer-0 queries sit at positions T.. so every
// candidate key is causally visible.
//
// The returned relevance, parallel to spans, is the post-softmax attention mass the
// probe's layer-0 queries place on each span's keys, averaged over probe rows and
// heads; the softmax runs over the candidate keys ONLY, so the scores sum to ~1.0
// across spans (a share-of-candidate-attention, comparative by construction).
func (s *Session) ReScoreSpans(probe []int, spans [][2]int) ([]float64, error) {
	if s == nil || s.M == nil || s.Cache == nil {
		return nil, errors.New("model: ReScoreSpans: nil session or cache")
	}
	m, cfg := s.M, s.M.Cfg
	if len(probe) == 0 {
		return nil, errors.New("model: ReScoreSpans: empty probe query")
	}
	if len(spans) == 0 {
		return nil, errors.New("model: ReScoreSpans: no candidate spans")
	}
	const l = 0 // the one layer whose Q is computable without a forward pass
	switch {
	case cfg.usesMLAMoELayout():
		return nil, fmt.Errorf("%w: MLA/DSA cache keeps no dense Kraw rows", ErrReScoreUnsupported)
	case cfg.Alibi:
		return nil, fmt.Errorf("%w: ALiBi scoring needs the position bias this path drops", ErrReScoreUnsupported)
	case cfg.isLinearAttnLayer(l):
		return nil, fmt.Errorf("%w: layer 0 is recurrent (no Kraw rows)", ErrReScoreUnsupported)
	case !m.has(layerName(l, "self_attn.q_proj.weight")):
		return nil, fmt.Errorf("%w: no f32 q_proj resident to project the probe", ErrReScoreUnsupported)
	}

	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	w := nKV * hd
	cached := s.Cache.Len()

	// Lay the candidates' keys into scratch at scoring positions [0, T): copy the
	// pre-RoPE Kraw row, rotate ONCE at the scoring position. The cached rows are
	// only read.
	T := 0
	for i, sp := range spans {
		from, n := sp[0], sp[1]
		if from < 0 || n <= 0 || from+n > cached {
			return nil, fmt.Errorf("model: ReScoreSpans: span %d [%d,+%d) outside cache of %d positions", i, from, n, cached)
		}
		T += n
	}
	scratchK := make([]float32, T*w)
	spanOf := make([]int, T) // scoring position -> index into spans
	t0 := 0
	for si, sp := range spans {
		from, n := sp[0], sp[1]
		for i := 0; i < n; i++ {
			row := scratchK[(t0+i)*w : (t0+i+1)*w]
			copy(row, s.Cache.Kraw[l][(from+i)*w:(from+i+1)*w])
			cos, sin := ropeRowForLayer(cfg, l, t0+i)
			for h := 0; h < nKV; h++ {
				applyRopeRow(row[h*hd:(h+1)*hd], cos, sin)
			}
			spanOf[t0+i] = si
		}
		t0 += n
	}

	// Probe Q at layer 0: embed → pre-attention norm → q_proj (+bias, +qk-norm) →
	// RoPE at position T+j. This mirrors blockStep's attnBody prologue exactly, so
	// the scores below are the same math the real cached decode runs at layer 0.
	embed := m.embedRows()
	eps := float32(cfg.RMSNormEps)
	attnNorm := m.attentionNorms(l)
	mat := f32Kernel{m}
	pn := func(str string) string { return layerName(l, str) }
	scale := cfg.attnScale()
	attnCap := float32(cfg.AttnSoftcap)
	qWidth := nH * hd

	rel := make([]float64, len(spans))
	scores := make([]float32, T)
	for j, id := range probe {
		if id < 0 || (id+1)*H > len(embed) {
			return nil, fmt.Errorf("model: ReScoreSpans: probe token %d out of vocabulary", id)
		}
		x := append([]float32(nil), embed[id*H:(id+1)*H]...)
		scaleEmbedInPlace(x, cfg)
		xn := normCfg(x, attnNorm.pre, attnNorm.preBias, eps, cfg)
		xp := mat.prep(xn)
		var q []float32
		if cfg.AttnOutputGate {
			// doubled q_proj: [query|gate] interleaved per head; scoring needs only the query half.
			qf := mat.mul(pn("self_attn.q_proj.weight"), xp, 2*qWidth, H)
			q = make([]float32, qWidth)
			for h := 0; h < nH; h++ {
				copy(q[h*hd:(h+1)*hd], qf[h*2*hd:h*2*hd+hd])
			}
		} else {
			q = mat.mul(pn("self_attn.q_proj.weight"), xp, qWidth, H)
		}
		// Bias/qk-norm expect k/v operands; the probe appends nothing, so hand them
		// discarded scratch of the right shape.
		kd, vd := make([]float32, w), make([]float32, w)
		m.applyProjBias(l, q, kd, vd)
		m.applyLayerQKNorm(l, q, kd)
		cos, sin := ropeRowForLayer(cfg, l, T+j)
		for h := 0; h < nH; h++ {
			applyRopeRow(q[h*hd:(h+1)*hd], cos, sin)
		}

		// One QK^T·softmax read per head over the candidate keys only — the same
		// dot/softcap/softmax chain blockStep runs, minus the value accumulation.
		for h := 0; h < nH; h++ {
			kvh := h / grp
			qh := q[h*hd : (h+1)*hd]
			for t := 0; t < T; t++ {
				kh := scratchK[t*w+kvh*hd : t*w+(kvh+1)*hd]
				scores[t] = dot(qh, kh) * scale
			}
			softcapInPlace(scores, attnCap)
			m.softmaxAttentionScores(l, h, scores)
			for t := 0; t < T; t++ {
				rel[spanOf[t]] += float64(scores[t])
			}
		}
	}
	// Each (probe row, head) softmax contributed mass 1; average so Σ rel ≈ 1.0.
	den := float64(len(probe) * nH)
	for i := range rel {
		rel[i] /= den
	}
	return rel, nil
}

// ReScoreSpans forwards the cheap re-score through the in-process abi.KVBackend
// adapter so kvmmu can reach it by type assertion — additive to the frozen seam,
// exactly like CanEvict: a backend that does not implement it simply reports
// "no cheap re-score" by absence.
func (b kvBackend) ReScoreSpans(probe []int, spans [][2]int) ([]float64, error) {
	return b.s.ReScoreSpans(probe, spans)
}
