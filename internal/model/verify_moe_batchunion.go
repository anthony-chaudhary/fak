package model

import "sort"

// verify_moe_batchunion.go — a GATED PROTOTYPE (gen/second-next, #4355) of colibri's
// batch-union MoE for the draft/verify window. It borrows one property from colibri
// (study-repo pass, epic #4352; verdict `inspire`, Apache-2.0<->Apache-2.0, clean-room,
// no bytes vendored): fold the N speculative verifications of a draft window into one MoE
// pass whose expert reads are amortized across the run-length — each UNIQUE routed expert
// of the panel is read ONCE and applied to every position that routes to it, instead of
// re-streaming the expert once per position (colibri c/glm.c:1619 spec-verify batch-union;
// the same trick at c/glm.c:1211 prefill). colibri streams experts from SSD, so amortizing
// the draft window's expert UNION is what makes verification cheaper than the tokens it checks.
//
// SCOPE / GATE (allowed risk = architectural exploration, NEVER default exposure):
// This is the FFN half of the axis only, and it is NOT wired into VerifyForward's default
// dispatch. verifyForwardBatchedOK still excludes usesMLAMoELayout(), so a GLM-MoE / DeepSeek
// verify keeps its proven per-position sequential fallback (verifyForwardSequential). The
// prototype is exercised solely by TestVerifyMoEBatchUnionMatchesPerPosition, which is the
// compatibility proof (proof bar = a token-exact simulation): the panel delta this produces
// is Float32bits-identical to applying glmMoeFFN per panel position — the sequential path's
// MoE FFN sublayer. Promotion to the live batched verify path is a SEPARATE, larger rung
// (see the lifecycle note at the bottom): the batched verify attention is standard GQA+RoPE,
// and usesMLAMoELayout() is excluded ALSO because of the MLA latent-KV attention this file
// does not touch — so swapping the dense-MLP block for this op is necessary but NOT sufficient
// to admit a GLM-MoE model onto verifyForwardBatched.
//
// verifyMoEPanelDelta computes, for each of the P panel positions whose post-attention-normed
// hidden state is Xn[q], the GLM-MoE routed-expert residual delta, organized as the batch-union:
// route every position (glmRoute), UNION the picked experts, and for each unique expert apply it
// ONCE — reading its gate/up/down weights a single time and running one batched GEMM (matMulBatch)
// over the gathered sub-batch of positions that route to it — then scatter the gate-weighted rows
// back into the per-position deltas.
//
// Token-exact to the per-position path because:
//   - matMulBatch(W, Xg, out, in, n)[r*out+o] is bit-identical to parMatRows(W, Xg[r], out, in)[o]
//     (both reduce over `in` in the SAME fdot order; batching only reuses each weight row across the
//     sub-batch), and the f32 per-position expertSwiGLU runs exactly parMatRows for gate/up/down.
//   - Experts are visited in ASCENDING index order, so each position accumulates its routed experts
//     in the same order glmRoute emits them (glmRoute sorts picks ascending by expert index), making
//     the per-position float reduction order identical.
//   - The always-on GLM shared expert is added per position AFTER the routed sum via the identical
//     glmSharedExperts call, matching glmMoeFFN's routed-then-shared order.
//
// ASSUMPTIONS (the guard is the same one hostBatchedGLMExperts uses): the routed experts carry NO
// per-expert bias and the model has NO active LoRA. Under those, the SwiGLU math here is the plain
// expertSwiGLU. A bias/LoRA config would diverge, so a promotion of this op MUST re-assert the guard
// (decline to the loop) exactly as hostBatchedGLMExperts does — it is not modeled here.
func (m *Model) verifyMoEPanelDelta(layer int, Xn [][]float32) [][]float32 {
	P := len(Xn)
	cfg := m.Cfg
	H := cfg.HiddenSize
	I := cfg.expertIntermediate()
	mat := f32Kernel{m}

	deltas := make([][]float32, P)
	for q := range deltas {
		deltas[q] = make([]float32, H)
	}

	// Route every panel position and gather, per expert, the sub-batch of positions that
	// route to it (with each position's gate weight). This IS the batch UNION: gathered's
	// key set is exactly the union of experts any panel position picked.
	type routedPos struct {
		pos    int
		weight float32
	}
	gathered := map[int][]routedPos{}
	for q := 0; q < P; q++ {
		for _, pk := range glmRoute(m, layer, Xn[q], mat) {
			gathered[pk.expert] = append(gathered[pk.expert], routedPos{q, pk.weight})
		}
	}
	experts := make([]int, 0, len(gathered))
	for e := range gathered {
		experts = append(experts, e)
	}
	sort.Ints(experts) // ascending — matches glmRoute's per-position accumulation order

	for _, e := range experts {
		grp := gathered[e]
		n := len(grp)
		// Stream expert e's weights ONCE; apply to all n gathered positions in one batched
		// GEMM per projection. matMulBatch reads each weight row once and sweeps the whole
		// sub-batch, so a hot expert routed by every position pays a single weight read.
		Xg := make([]float32, n*H)
		for r, g := range grp {
			copy(Xg[r*H:(r+1)*H], Xn[g.pos])
		}
		G := matMulBatch(m.tensor(expertName(layer, e, "gate_proj.weight")), Xg, I, H, n)
		U := matMulBatch(m.tensor(expertName(layer, e, "up_proj.weight")), Xg, I, H, n)
		for i := range G {
			G[i] = act(G[i], cfg) * U[i]
		}
		D := matMulBatch(m.tensor(expertName(layer, e, "down_proj.weight")), G, H, I, n)
		// Scatter: accumulate each gathered position's gate-weighted expert output back into
		// its delta. Because experts are ascending, dst[i] receives its terms in glmRoute order.
		for r, g := range grp {
			drow := D[r*H : (r+1)*H]
			dst := deltas[g.pos]
			w := g.weight
			for i := 0; i < H; i++ {
				dst[i] += w * drow[i]
			}
		}
	}

	// GLM's always-on, un-gated shared expert — added AFTER the routed sum, per position, via
	// the identical glmSharedExperts call glmMoeFFN uses (no union win: it fires for every
	// position, so batching it buys nothing and would only risk a reduction-order drift).
	if cfg.NSharedExperts > 0 && m.hasWeight(layerName(layer, "mlp.shared_experts.gate_proj.weight")) {
		for q := 0; q < P; q++ {
			shared := glmSharedExperts(m, layer, Xn[q], mat)
			for i := 0; i < H; i++ {
				deltas[q][i] += shared[i]
			}
		}
	}
	return deltas
}

// Lifecycle (gen/second-next closure evidence for #4355):
//   - PROMOTION evidence: TestVerifyMoEBatchUnionMatchesPerPosition proves the batch-union
//     delta is token-exact (Float32bits) to the per-position glmMoeFFN across a panel with
//     both shared-and-solo routed experts and a shared-expert layer. That is the correctness
//     precondition for wiring it into verifyForwardBatched behind a runtime gate.
//   - DEMOTION / RETIREMENT criteria: retire this op if (a) the parity test ever fails
//     (a reduction-order or router-order drift makes it non-token-exact), or (b) native
//     expert streaming / residency (#2726, #3174) never lands, so there is no bandwidth to
//     amortize and the per-position sequential verify stays cheaper than the gather bookkeeping.
//   - INVALIDATING assumption: the issue frames this as "replacing the dense-MLP path" in
//     verifyForwardBatched, implying the FFN swap alone admits GLM-MoE onto the batched verify.
//     It does NOT: verifyForwardBatchedOK excludes usesMLAMoELayout() ALSO for the MLA latent-KV
//     attention, which the batched verify path (standard GQA+RoPE) does not implement. So this
//     FFN op is necessary but not sufficient; the batched MLA verify attention is the real
//     promotion blocker, and default exposure stays off until it lands.
