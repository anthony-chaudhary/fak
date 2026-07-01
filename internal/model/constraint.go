package model

// Native structured-output decoding for the in-kernel reference engine (#929,
// the native follow-on to #907's ride-mode passthrough).
//
// #907 wired the RIDE half: the gateway forwards a client's OpenAI
// `response_format` / `logit_bias` to a vLLM/SGLang upstream, which enforces the
// JSON-schema/grammar constraint during generation. That covers every backend fak
// FRONTS. It does NOT cover this package — the in-kernel decode is plain greedy
// argmax over post-head logits (kv.go:Generate -> argmaxF32) with no constraint
// hook. There is no upstream to forward to, so fak must own the sampler hook.
//
// This file is that sink, smallest-first per the issue:
//
//  1. a LOGIT-BIAS mask — apply the per-request logit_bias map (token id -> bias,
//     the OpenAI -100..100 convention) to the post-head logits before argmax.
//  2. a JSON-SCHEMA / GRAMMAR logit mask behind a feature flag — a per-step token
//     mask that keeps the decode on a valid path.
//
// LAYERING. internal/model is tier-1 (foundation); internal/grammar is tier-2 and
// the tokenizer-aware compiler that turns a `oneOf`-of-tools JSON-Schema into a
// per-step token mask lives ABOVE this package. So model does NOT import grammar /
// tokenizer: it defines the LogitMask SEAM and a higher layer (which can see both)
// injects the compiled mask — the same dependency-inversion the gateway uses for
// session state. The concrete masks here (AllowedSetMask / StepMask) are the
// minimal, real building blocks such a compiler emits.
//
// BIT-EXACT-OFF (the load-bearing criterion). With no logit_bias and no active
// mask, sampleConstrained returns argmaxF32(logits) VERBATIM — the same function
// the unconstrained greedy path calls — so a constraint that is off is a proven
// no-op, token-identical to Session.Generate. The default Generate path never
// constructs a DecodeConstraint, so unconstrained decode is structurally unchanged.

import (
	"math"
	"os"
)

// LogitBias is the OpenAI per-token logit-bias map: token id -> an additive bias
// on that token's post-head logit, clamped to the standard [-100, 100] range. A
// +100 bias forces a token where it is otherwise reachable; -100 effectively
// removes it from the argmax. The nil / empty map is the unconstrained identity.
// It mirrors agent.SampleParams.LogitBias (map[int]float64), the carrier #907
// already threads to the gateway; this package is the in-kernel sink.
type LogitBias map[int]float64

// LogitBiasClamp is the OpenAI logit_bias magnitude bound: a bias is clamped to
// [-LogitBiasClamp, +LogitBiasClamp] before it is added to a logit.
const LogitBiasClamp = 100.0

// LogitMask is the injected schema/grammar constraint seam. A higher layer compiles
// a JSON-Schema (the canonical `oneOf`-of-tools shape, sourced from internal/grammar)
// to a per-step token predicate and passes an implementation in here; internal/model
// never imports grammar or the tokenizer. MaskLogits is called on the post-head
// logits BEFORE argmax and must set the logit of every token that would leave the
// valid path to negative infinity, given the tokens emitted so far THIS TURN
// (history is the generated ids, not the prompt). A masked token can never be
// selected, regardless of any logit_bias. A nil mask is the identity.
type LogitMask interface {
	MaskLogits(history []int, logits []float32)
}

// DecodeConstraint bundles the two native sampler hooks #929 adds at the decode
// boundary. The zero value (and a nil *DecodeConstraint) is inert: the constrained
// sampler then reduces to the exact greedy argmax. Bias applies whenever it is
// non-empty; Mask is the FLAGGED half (item 2) and is applied only when the native
// guided-decode feature flag is on — see GuidedDecodeEnabled.
type DecodeConstraint struct {
	Bias LogitBias
	Mask LogitMask
}

// GuidedDecodeEnabled reports whether the native JSON-schema/grammar logit mask is
// turned on. It is gated by FAK_NATIVE_GUIDED_DECODE=1 and DEFAULTS OFF, so a
// schema mask is never applied unless an operator opts in — the logit-bias half
// (item 1) needs no flag because an empty bias map is already inert. This is the
// "flag defaults OFF" half of #929's acceptance: a non-nil Mask on a constraint is
// dormant until the flag is set.
func GuidedDecodeEnabled() bool {
	return os.Getenv("FAK_NATIVE_GUIDED_DECODE") == "1"
}

// maskActive reports whether the schema/grammar mask should run: a mask is present
// AND the feature flag is on. With the flag off, the mask is dormant.
func (c *DecodeConstraint) maskActive() bool {
	return c != nil && c.Mask != nil && GuidedDecodeEnabled()
}

// Active reports whether the constraint changes the decode at all. When false the
// sampler is a proven no-op (returns argmaxF32(logits) verbatim) — the bit-exact-off
// path. A non-nil Mask alone does NOT make a constraint active unless the feature
// flag is on, so a flag-off decode with a compiled-in mask stays bit-exact.
func (c *DecodeConstraint) Active() bool {
	if c == nil {
		return false
	}
	return len(c.Bias) > 0 || c.maskActive()
}

// sampleConstrained selects the next token from post-head logits, applying the
// schema/grammar mask and then the logit-bias map before argmax. With an inert
// constraint it is byte-for-byte the greedy argmax path (bit-exact-off).
//
// The mask runs FIRST and sets disallowed tokens to -inf, so it wins over any bias
// (-inf + 100 is still -inf): a mask is a hard structural constraint, a bias is a
// soft preference. Logits are copied into a scratch buffer so the caller's logits
// slice (a quantized Step may reuse it) is never mutated.
func sampleConstrained(history []int, logits []float32, c *DecodeConstraint) int {
	if !c.Active() {
		return argmaxF32(logits) // bit-exact-off: identical to Session.Generate
	}
	eff := append([]float32(nil), logits...)
	if c.maskActive() {
		c.Mask.MaskLogits(history, eff)
	}
	for tok, bias := range c.Bias {
		if tok < 0 || tok >= len(eff) {
			continue // out-of-vocab bias entries are ignored, not an error
		}
		if bias > LogitBiasClamp {
			bias = LogitBiasClamp
		} else if bias < -LogitBiasClamp {
			bias = -LogitBiasClamp
		}
		eff[tok] += float32(bias)
	}
	return argmaxF32(eff)
}

// GenerateConstrained greedily decodes up to n tokens after the prompt, applying the
// decode constraint (logit-bias + optional flagged schema/grammar mask) at each step
// before argmax. With an inert constraint (nil, or empty bias and a dormant mask) it
// is token-identical to Generate — the bit-exact-off contract #929 pins. The schema
// mask is opt-in by construction: the default Generate path builds no constraint, and
// even a compiled-in mask stays dormant until FAK_NATIVE_GUIDED_DECODE=1.
func (s *Session) GenerateConstrained(prompt []int, n int, c *DecodeConstraint) []int {
	logits := s.Prefill(prompt)
	out := make([]int, 0, n)
	for i := 0; i < n; i++ {
		next := sampleConstrained(out, logits, c)
		out = append(out, next)
		if s.M.Cfg.IsEOS(next) {
			break
		}
		logits = s.Step(next)
	}
	return out
}

// GenerateBatchConstrained is GenerateBatch with an optional per-lane native decode
// constraint. It applies each lane's logit-bias / flagged schema mask at the same
// token-selection boundary that consumes StepBatchActive's logits. A nil constraints
// slice, a missing lane entry, or an inert constraint is the identity: selection falls
// through sampleConstrained's bit-exact-off path and matches GenerateBatch token-for-token.
func (bs *BatchSession) GenerateBatchConstrained(prompts [][]int, n int, constraints []*DecodeConstraint) [][]int {
	logits := bs.PrefillEach(prompts)
	return bs.generateBatchDecode(logits, n, func(b int, prior []int, laneLogits []float32) int {
		return sampleConstrained(prior, laneLogits, constraintForLane(constraints, b))
	})
}

func constraintForLane(constraints []*DecodeConstraint, lane int) *DecodeConstraint {
	if lane < len(constraints) {
		return constraints[lane]
	}
	return nil
}

// AllowedSetMask is the minimal concrete LogitMask: it permits only the token ids in
// the set at EVERY step (a static mask). It is the simplest correct schema constraint
// — e.g. "the next token must be one of the tool-name `const` tokens" — and the
// building block a per-step grammar/tokenizer compiler emits. Tokens not in the set
// have their logit set to -inf, so a constrained decode can never emit them. The
// compiler that maps a JSON-Schema `oneOf`-of-tools to these token sets lives above
// this package (the LogitMask seam); this is the sink it targets.
type AllowedSetMask map[int]bool

// MaskLogits sets every token not in the allowed set to -inf. An empty set masks
// everything (a degenerate over-constraint the compiler must avoid); the history is
// unused because the allowed set is the same at every step.
func (m AllowedSetMask) MaskLogits(_ []int, logits []float32) {
	neg := float32(math.Inf(-1))
	for i := range logits {
		if !m[i] {
			logits[i] = neg
		}
	}
}

// StepMask permits a different allowed-token set at each decode step, indexed by the
// number of tokens emitted so far this turn (len(history)). It is the shape a per-step
// grammar/FSM mask takes: step k consults PerStep[k]. A nil entry, or a step past the
// end of PerStep, allows all tokens — the constraint has terminated and the rest of
// the turn is unconstrained. This is the per-step generalization of AllowedSetMask.
type StepMask struct {
	PerStep []map[int]bool
}

// MaskLogits applies the allowed set for the current step (len(history)). Steps with
// no entry are unconstrained.
func (m *StepMask) MaskLogits(history []int, logits []float32) {
	step := len(history)
	if step >= len(m.PerStep) {
		return // constraint terminated: no mask
	}
	allowed := m.PerStep[step]
	if allowed == nil {
		return // this step is unconstrained
	}
	neg := float32(math.Inf(-1))
	for i := range logits {
		if !allowed[i] {
			logits[i] = neg
		}
	}
}
