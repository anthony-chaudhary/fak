package model

// kimik31_situ.go — the Kimi-K3.1 SiTU (SoftCap-GLU) activation (ticket
// 1611-kimi-k31, port leaf sglang-kimi-k31-1; goal Wave 1.1). This small
// elementwise op blocks ALL expert execution: K3.1's dense MLP layer 0, its 2
// shared experts, and its 896 routed experts all read through SiTU, so no
// expert can run until this exists.
//
// SOURCE (direct port, pinned): SGLang
// python/sglang/kernels/ops/activation/kimi_k3_situ_and_mul.py at
// the authoritative formula:
//
//	gate_out = beta * tanh(gate / beta) * sigmoid(gate)
//	up_out   = linear_beta * tanh(up / linear_beta)   [if linear_beta is set]
//	output   = gate_out * up_out
//
// The gate branch is a sigmoid multiply softened by a tanh soft-cap of width
// beta; the up branch is a symmetric tanh soft-cap of width linear_beta (the
// "clamp" the study note names). K3.1 ships beta = 4.0 and linear_beta = 25.0
// (moonshotai/Kimi-K3.1-Pro config.json activation_situ_beta /
// activation_situ_linear_beta; the port study also pins them via a MegaMoE
// assert). Intermediates are F32; the output is stored in the input's dtype by
// the reference (the Go host form is F32 throughout).
//
// CONTRACT: the caller owns gate and must ensure it does not overlap up; the
// fused form writes gate_out * up_out into gate in increasing index order
// (so it is bit-reproducible), reading up read-only. A linear_beta <= 0 means
// "not set" — the reference's linear_beta=None — and leaves up unmodified
// (up_out = up), matching the Python `has_linear_beta` branch.

import "math"

// situGateOut applies the SiTU gate branch: beta * tanh(gate/beta) * sigmoid(gate).
// beta must be > 0. It reports the F32 result of the pinned formula.
func situGateOut(gate, beta float32) float32 {
	g64 := float64(gate)
	b64 := float64(beta)
	softcap := b64 * math.Tanh(g64/b64)
	sig := 1.0 / (1.0 + math.Exp(-g64))
	return float32(softcap * sig)
}

// situUpOut applies the SiTU up branch: linear_beta * tanh(up/linear_beta) when
// linear_beta > 0, else the identity (linear_beta=None in the reference).
func situUpOut(up, linearBeta float32) float32 {
	if linearBeta <= 0 {
		return up
	}
	return float32(float64(linearBeta) * math.Tanh(float64(up)/float64(linearBeta)))
}

// SituAndMul is the fused SiTU-GLU elementwise op: it writes
// gate_out(gate) * up_out(up) into gate in increasing index order and returns
// gate. It mirrors the reference's fused kernel (one pass, gate consumed in
// place, up read-only). gate and up must be nonempty and of equal length; a
// refused call leaves both unchanged.
//
// beta is the gate soft-cap width (K3.1: 4.0); linearBeta is the up soft-cap
// width (K3.1: 25.0) or <= 0 to skip the up soft-cap (reference
// linear_beta=None).
func SituAndMul(gate, up []float32, beta, linearBeta float32) []float32 {
	if len(gate) == 0 || len(gate) != len(up) {
		panic("model: kimi-k31 situ: gate and up must be nonempty and of equal length")
	}
	if beta <= 0 {
		panic("model: kimi-k31 situ: beta must be > 0")
	}
	for i := range gate {
		gate[i] = situGateOut(gate[i], beta) * situUpOut(up[i], linearBeta)
	}
	return gate
}

// usesSituActivation reports whether this config's MLP gate activation is SiTU
// (K3.1 declares hidden_act: "situ"). Exported intent via Config.isKimiK31 gates
// the caller; the predicate itself is pure so a non-K3.1 config that merely
// names the activation still routes correctly.
func (c Config) usesSituActivation() bool {
	return c.HiddenAct == "situ"
}
