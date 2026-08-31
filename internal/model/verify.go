package model

import (
	"errors"
	"fmt"
	"math"
	"time"
)

// verify.go — the single-pass batched + tree-attention VERIFY execution (rung #533 of the
// poly-model serving epic #529; docs/serving/polymodel-prefill-share-plan.md §5/§7). It is
// the throughput half that turns the already-shipped accept DECISION (internal/polymodel:
// AcceptGreedy / AcceptTree) and bit-exact rollback (internal/spec: the ProvisionalSink +
// model.KVCache.Evict) into a real single forward pass over the candidate tokens — instead
// of one bandwidth-bound decode step per candidate.
//
// VerifyForward runs the P candidate tokens in ids through the model in ONE pass and
// returns each panel position's next-token logits (post-head), so the caller takes argmax
// per position and feeds the result to AcceptGreedy (chain) or AcceptTree (tree). Two
// attention shapes share the one forward:
//
//   - CHAIN (the linear / greedy-speculation case): pos == nil and allow == nil. Each
//     panel token attends causally to the committed prefix plus every earlier panel token
//     (honoring the layer's sliding window), exactly like prefillBatched. The per-position
//     logits AND the appended KV are bit-identical to P sequential Session.Step calls
//     (VerifyForward shares prefillBatched's math, proven bit-exact to the per-token loop
//     by TestPrefillBatchedMatchesSerial, and re-asserted directly by
//     TestVerifyForwardMatchesSerial), so it is a lossless drop-in for the sequential verify.
//   - TREE (the Medusa / EAGLE-2 / SpecInfer case): pos gives each panel token's absolute
//     RoPE position and allow(q,k) reports whether panel query q may attend to panel key k.
//     Siblings share a position (base+depth-1) and allow is the ancestor relation, so each
//     node attends only to its ancestor chain + the committed prefix — tree-attention masks.
//     The accepted path is then token-identical to plain greedy decode (its context at every
//     depth IS the greedy context), witnessed in internal/spec.
//
// The committed prefix [0, base) is ALWAYS attended in both shapes (it is real context,
// never masked). The supported regime is the plain f32 PreNorm path (standard GQA + RoPE,
// no backend / quant / MoE / Alibi / SWA-hybrid / Qwen-hybrid / per-layer-RoPE) — the
// regime internal/spec and cmd/polymodelbench use; for the chain, any other regime falls
// back to P sequential Steps (still correct, still returns per-position logits, just not
// single-pass). A tree (allow != nil) needs the masked attention and so requires the
// supported regime (returns nil otherwise).
func (s *Session) VerifyForward(ids []int, pos []int, allow func(q, k int) bool) [][]float32 {
	P := len(ids)
	if P == 0 {
		return nil
	}
	if !verifyForwardBatchedOK(s) {
		if allow != nil {
			return nil // tree verify needs the masked batched attention; unsupported regime
		}
		return s.verifyForwardSequential(ids) // chain fallback: correct, universal, not single-pass
	}
	return s.verifyForwardBatched(ids, pos, allow)
}

const (
	targetVerificationReceiptSchema = "fak-target-verification/1"
	targetVerificationEngine        = "fak-native"
	targetVerificationBatchedPath   = "fak-native/batched-target-verify-v1"
	targetVerificationQwen38Path    = "fak-native/f32/qwen3.8-whole-sequence-target-verify-v1"
	targetVerificationDecodePath    = "fak-native/ordinary-target-decode-v1"
	qwen38VerifyBoundaryTolerance   = float32(2e-5)
)

// ErrTargetVerificationDowngrade marks a request that cannot be represented by
// one proven target operation. The caller may explicitly retain ordinary
// fak-native target decode; it must not count that N-step path as one verify.
var ErrTargetVerificationDowngrade = errors.New("model: one-operation target verification downgraded")

// TargetVerificationDowngradeError gives the typed reason for retaining
// ordinary fak-native target decode. It never authorizes another runtime.
type TargetVerificationDowngradeError struct {
	Reason string
}

func (e *TargetVerificationDowngradeError) Error() string {
	if e == nil || e.Reason == "" {
		return ErrTargetVerificationDowngrade.Error()
	}
	return fmt.Sprintf("%v: %s; use ordinary fak-native target decode", ErrTargetVerificationDowngrade, e.Reason)
}

func (e *TargetVerificationDowngradeError) Unwrap() error {
	return ErrTargetVerificationDowngrade
}

type qwen38TargetForward func(*Model, []int) *Activations

// VerifyForwardOneOperation evaluates a greedy draft chain only when the
// complete panel is represented by one genuine fak-native target operation.
// Unsupported shapes return a typed downgrade without silently executing N
// Steps. The Qwen3.8 hybrid lane recomputes the exact token-lineage prefix and
// draft through one cacheless native Forward; the transaction later restores
// and replays only the accepted prefix, preserving live recurrent/KV state.
func (s *Session) VerifyForwardOneOperation(ids []int, boundaryLogits []float32) ([][]float32, TargetVerificationReceipt, error) {
	return s.verifyForwardOneOperation(ids, boundaryLogits, nil)
}

func (s *Session) verifyForwardOneOperation(ids []int, boundaryLogits []float32, forward qwen38TargetForward) (rows [][]float32, receipt TargetVerificationReceipt, err error) {
	receipt = TargetVerificationReceipt{
		Schema:      targetVerificationReceiptSchema,
		Engine:      targetVerificationEngine,
		Path:        targetVerificationDecodePath,
		DraftTokens: len(ids),
	}
	setupStart := time.Now()
	setupMeasured := false
	finishSetup := func() {
		if setupMeasured {
			return
		}
		receipt.Accounting.Setup.Nanoseconds += time.Since(setupStart).Nanoseconds()
		receipt.Accounting.Setup.Measured = true
		setupMeasured = true
	}
	defer func() {
		finishSetup()
	}()
	if len(ids) == 0 {
		receipt.OneOperation = true
		return nil, receipt, nil
	}
	if s == nil || s.M == nil || s.Cache == nil {
		return nil, receipt, targetVerificationDowngrade("target session or host cache is unavailable")
	}
	if verifyForwardBatchedOK(s) {
		finishSetup()
		started := time.Now()
		rows = s.verifyForwardBatched(ids, nil, nil)
		receipt.Accounting.TargetVerification = measuredSpeculativeCost(started)
		receipt.Path = targetVerificationBatchedPath
		receipt.TargetVerificationOperations = 1
		receipt.OneOperation = true
		return rows, receipt, nil
	}

	prefix, reason := qwen38OneOperationPrefix(s)
	if reason != "" {
		return nil, receipt, targetVerificationDowngrade(reason)
	}
	if len(boundaryLogits) != s.M.Cfg.VocabSize {
		return nil, receipt, targetVerificationDowngrade(fmt.Sprintf(
			"boundary logits width %d differs from Qwen3.8 vocabulary %d",
			len(boundaryLogits), s.M.Cfg.VocabSize,
		))
	}
	if forward == nil {
		forward = func(m *Model, tokens []int) *Activations { return m.Forward(tokens) }
	}
	tokens := append(prefix, ids...)
	finishSetup()
	started := time.Now()
	receipt.TargetVerificationOperations = 1
	defer func() {
		if recovered := recover(); recovered != nil {
			rows = nil
			err = fmt.Errorf("model: Qwen3.8 one-operation target verification: %v", recovered)
			receipt.Accounting.Recovery = measuredSpeculativeCost(started)
		}
	}()
	act := forward(s.M, tokens)
	receipt.Accounting.TargetVerification = measuredSpeculativeCost(started)
	if act == nil || len(act.Logits) != len(tokens) {
		return nil, receipt, fmt.Errorf("model: Qwen3.8 one-operation target verification returned %d rows for %d tokens", verificationRows(act), len(tokens))
	}
	boundary := act.Logits[len(prefix)-1]
	if len(boundary) != len(boundaryLogits) || argmaxF32(boundary) != argmaxF32(boundaryLogits) ||
		maxAbsF32Delta(boundary, boundaryLogits) > qwen38VerifyBoundaryTolerance {
		receipt.DowngradeReason = "cacheless Qwen3.8 boundary calibration diverged from the live target"
		return nil, receipt, targetVerificationDowngrade(receipt.DowngradeReason)
	}
	rows = make([][]float32, len(ids))
	for i := range rows {
		rows[i] = append([]float32(nil), act.Logits[len(prefix)+i]...)
	}
	receipt.Path = targetVerificationQwen38Path
	receipt.OneOperation = true
	return rows, receipt, nil
}

func qwen38OneOperationPrefix(s *Session) ([]int, string) {
	if !s.M.Cfg.IsQwen35Hybrid() {
		return nil, "target architecture is not the witnessed Qwen3.8 hybrid"
	}
	if s.Backend != nil || s.Quant || s.Q4 || s.Q4K || s.F16 || s.GPTQ ||
		s.Metal || s.MetalQ4K || s.PrecisionPolicy != nil {
		return nil, "only the native f32 Qwen3.8 target is admitted"
	}
	cfg := s.M.Cfg
	if cfg.IsMoE() || cfg.DenseMLP || cfg.Alibi || cfg.BlockTopology != PreNorm ||
		!cfg.AttnOutputGate || !cfg.NormGain1p {
		return nil, "Qwen3.8 target topology is outside the witnessed dense PreNorm hybrid envelope"
	}
	if s.Cache.lineage.fault != "" {
		return nil, "target token lineage is invalid: " + s.Cache.lineage.fault
	}
	if len(s.Cache.pos) == 0 || len(s.Cache.pos) != len(s.Cache.lineage.ids) {
		return nil, fmt.Sprintf("target token lineage positions=%d ids=%d", len(s.Cache.pos), len(s.Cache.lineage.ids))
	}
	prefix := make([]int, len(s.Cache.lineage.ids))
	for i, pos := range s.Cache.pos {
		if pos != i {
			return nil, fmt.Sprintf("target prefix position %d carries absolute position %d", i, pos)
		}
		token := uint64(s.Cache.lineage.ids[i])
		if token >= uint64(s.M.Cfg.VocabSize) {
			return nil, fmt.Sprintf("target prefix token %d is outside vocabulary %d", token, s.M.Cfg.VocabSize)
		}
		prefix[i] = int(token)
	}
	return prefix, ""
}

func targetVerificationDowngrade(reason string) *TargetVerificationDowngradeError {
	return &TargetVerificationDowngradeError{Reason: reason}
}

func measuredSpeculativeCost(started time.Time) SpeculativeCostComponent {
	return SpeculativeCostComponent{Nanoseconds: time.Since(started).Nanoseconds(), Measured: true}
}

func verificationRows(act *Activations) int {
	if act == nil {
		return 0
	}
	return len(act.Logits)
}

func maxAbsF32Delta(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.Inf(1))
	}
	var max float32
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > max {
			max = d
		}
	}
	return max
}

// verifyForwardSequential is the always-correct chain fallback: P sequential Steps, each
// returning that position's logits. It is the pre-#533 verify path (one step per
// candidate), kept so VerifyForward never regresses a model the batched path does not cover.
func (s *Session) verifyForwardSequential(ids []int) [][]float32 {
	out := make([][]float32, len(ids))
	for i, id := range ids {
		out[i] = s.Step(id)
	}
	return out
}

// verifyForwardBatchedOK reports whether the batched f32 PreNorm verify path supports this
// session. It mirrors the dispatch in Prefill (kv.go): the plain PreNorm standard path with
// no backend / quant / MoE / Alibi / Qwen-hybrid / non-PreNorm / per-layer-RoPE.
func verifyForwardBatchedOK(s *Session) bool {
	if s.Backend != nil || s.Quant || s.Q4 || s.Q4K || s.GPTQ || s.Metal || s.PrecisionPolicy != nil {
		return false
	}
	cfg := s.M.Cfg
	if cfg.usesMLAMoELayout() || q8PrefillNeedsTokenLoop(cfg) {
		return false
	}
	return true
}

// verifyForwardBatched is the single-pass forward, structurally a generalization of
// prefillBatched: the embed / projection / RoPE / MLP cores are identical, and the
// attention is (a) the exact contiguous-causal loop from prefillBatched when allow == nil
// (bit-identical, the chain) or (b) an explicit per-query allowed-key set when allow != nil
// (the tree-attention mask). pos (nil ⇒ base..base+P-1) gives each panel token's absolute
// RoPE position; the cache appends P positions with those absolute positions. The final
// norm + head mirror Prefill's head(prefillBatched(...)) per panel token so the chain
// logits are bit-identical to serial head(finalNorm(tokenHidden(...))).
func (s *Session) verifyForwardBatched(ids []int, pos []int, allow func(q, k int) bool) [][]float32 {
	dispatchWorkers := currentWorkerCount()
	m, cfg := s.M, s.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	w := nKV * hd
	scale := cfg.attnScale()
	attnCap := float32(cfg.AttnSoftcap)
	P := len(ids)
	base := s.Cache.Len()
	if pos == nil {
		pos = make([]int, P)
		for q := 0; q < P; q++ {
			pos[q] = base + q
		}
	}

	embed := m.embedRows()
	X := make([]float32, P*H)
	for q, id := range ids {
		copy(X[q*H:(q+1)*H], embed[id*H:(id+1)*H])
		scaleEmbedInPlace(X[q*H:(q+1)*H], cfg)
	}

	cosP := make([][]float32, P)
	sinP := make([][]float32, P)
	for q := 0; q < P; q++ {
		cosP[q], sinP[q] = ropeRow(cfg, pos[q])
	}

	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(str string) string { return layerName(l, str) }

		// verifyForwardBatchedOK gates only on backend/quant/topology (q8PrefillNeedsTokenLoop),
		// never on the norm kind, so a biased-LayerNorm family verifies through this lane. The
		// documented contract above is bit-identity with serial head(finalNorm(tokenHidden(...))),
		// and tokenHidden's blockStep passes n.preBias — so this call must too. rmsnormCfg's
		// hard-coded nil would break exactly that contract.
		Xn := make([]float32, P*H)
		parFor(P, dispatchWorkers, func(lo, hi int) {
			wIn := m.tensor(lp("input_layernorm.weight"))
			bIn := m.tensorOptional(lp("input_layernorm.bias"))
			for q := lo; q < hi; q++ {
				copy(Xn[q*H:(q+1)*H], normCfg(X[q*H:(q+1)*H], wIn, bIn, eps, cfg))
			}
		})

		Q := matMulBatch(m.tensor(lp("self_attn.q_proj.weight")), Xn, nH*hd, H, P)
		K := matMulBatch(m.tensor(lp("self_attn.k_proj.weight")), Xn, w, H, P)
		V := matMulBatch(m.tensor(lp("self_attn.v_proj.weight")), Xn, w, H, P)
		for q := 0; q < P; q++ {
			m.applyProjBias(l, Q[q*nH*hd:(q+1)*nH*hd], K[q*w:(q+1)*w], V[q*w:(q+1)*w])
			m.applyLayerQKNorm(l, Q[q*nH*hd:(q+1)*nH*hd], K[q*w:(q+1)*w])
		}

		// Kraw (pre-RoPE, post-qk-norm K) stashed before roping K, exactly like the
		// per-token path, so a later Evict can reposition a survivor in a single rotation.
		Kraw := append([]float32(nil), K...)
		parFor(P, dispatchWorkers, func(lo, hi int) {
			for q := lo; q < hi; q++ {
				ropeRowQKInto(Q[q*nH*hd:(q+1)*nH*hd], K[q*w:(q+1)*w], cosP[q], sinP[q], hd, nH, nKV)
			}
		})

		Kl, Vl, Wl, attnOut := preparePrefillAttention(s.Cache, l, Kraw, K, V, cfg, P, nH, hd)
		if allow == nil {
			// CHAIN: byte-identical to prefillBatched's contiguous-causal attention — query
			// q (absolute base+q) attends to cached keys [j0, base+q] inclusive.
			parFor(P, dispatchWorkers, func(lo, hi int) {
				for q := lo; q < hi; q++ {
					nPos := base + q + 1
					j0 := windowLoContig(nPos, base+q, Wl)
					for h := 0; h < nH; h++ {
						kvh := h / grp
						qh := packedHead(Q, q, nH*hd, h, hd)
						scores := make([]float32, nPos-j0)
						fillAttentionScores(scores, qh, Kl, j0, nPos, w, kvh, hd, scale, dot)
						softcapInPlace(scores, attnCap)
						softmaxInPlace(scores)
						out := packedHead(attnOut, q, nH*hd, h, hd)
						accumulateAttentionValues(out, Vl, scores, j0, nPos, w, kvh, hd)
					}
				}
			})
		} else {
			// TREE: query q attends to the committed prefix [0,base) plus the panel keys
			// allow admits (its ancestor chain). The set is sparse and per-query; iterate
			// it in index order for a deterministic reduction. Siblings are NOT ancestors
			// of each other, so two candidate continuations never attend to one another —
			// the structural difference from the chain.
			parFor(P, dispatchWorkers, func(lo, hi int) {
				for q := lo; q < hi; q++ {
					for h := 0; h < nH; h++ {
						kvh := h / grp
						qh := Q[q*nH*hd+h*hd : q*nH*hd+(h+1)*hd]
						keys := make([]int, 0, base+P)
						for j := 0; j < base; j++ {
							keys = append(keys, j)
						}
						for k := 0; k < P; k++ {
							if allow(q, k) {
								keys = append(keys, base+k)
							}
						}
						scores := make([]float32, len(keys))
						for idx, j := range keys {
							kh := Kl[j*w+kvh*hd : j*w+(kvh+1)*hd]
							scores[idx] = dot(qh, kh) * scale
						}
						softcapInPlace(scores, attnCap)
						softmaxInPlace(scores)
						out := attnOut[q*nH*hd+h*hd : q*nH*hd+(h+1)*hd]
						for idx, j := range keys {
							vh := Vl[j*w+kvh*hd : j*w+(kvh+1)*hd]
							wj := scores[idx]
							saxpy(out, vh, wj)
						}
					}
				}
			})
		}

		// The optional self_attn.o_proj.bias goes on before the residual, exactly as the
		// per-token Step this lane is contracted to reproduce bit-for-bit does.
		O := matMulBatch(m.tensor(lp("self_attn.o_proj.weight")), attnOut, H, nH*hd, P)
		for q := 0; q < P; q++ {
			m.addBiasIfPresent(O[q*H:(q+1)*H], lp("self_attn.o_proj.bias"))
		}
		for i := range X {
			X[i] += O[i]
		}

		Xn2 := make([]float32, P*H)
		parFor(P, dispatchWorkers, func(lo, hi int) {
			wPost := m.tensor(lp("post_attention_layernorm.weight"))
			bPost := m.tensorOptional(lp("post_attention_layernorm.bias"))
			for q := lo; q < hi; q++ {
				copy(Xn2[q*H:(q+1)*H], normCfg(X[q*H:(q+1)*H], wPost, bPost, eps, cfg))
			}
		})
		Down := m.batchedGatedMLP(lp, Xn2, P, H, cfg.IntermediateSize, cfg)
		for i := range X {
			X[i] += Down[i]
		}
	}

	for q := 0; q < P; q++ {
		s.Cache.appendPosition(pos[q], ids[q])
	}
	out := make([][]float32, P)
	for q := 0; q < P; q++ {
		out[q] = s.head(m.finalNorm(X[q*H : (q+1)*H]))
	}
	return out
}
