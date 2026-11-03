package model

// Native structured-output decoding for the in-kernel reference engine (#929). The gateway
// already rides client constraints onto upstream backends; the in-kernel decode has no upstream
// to forward to, so this file owns the sampler hook:
//
//  1. a LOGIT-BIAS mask — apply the per-request logit_bias map to the post-head logits before argmax.
//  2. a JSON-SCHEMA / GRAMMAR logit mask behind a feature flag — a per-step token mask that keeps
//     the decode on a valid path.
//
// LAYERING: model does not import grammar or the tokenizer. It defines the LogitMask seam and a
// higher layer injects the compiled mask (the same dependency-inversion the gateway uses);
// AllowedSetMask / StepMask are the minimal building blocks such a compiler emits.
//
// BIT-EXACT-OFF: with no logit_bias and no active mask, sampleConstrained returns
// argmaxF32(logits) verbatim — a constraint that is off is a proven no-op, token-identical to
// Session.Generate.

import (
	"math"
	"os"

	"github.com/anthony-chaudhary/fak/internal/guideddecode"
)

// LogitBias is the OpenAI per-token logit-bias map: token id -> an additive bias on that
// token's post-head logit, clamped to [-100, 100]. The nil/empty map is the identity.
type LogitBias map[int]float64

// LogitBiasClamp is the OpenAI logit_bias magnitude bound: a bias is clamped to
// [-LogitBiasClamp, +LogitBiasClamp] before it is added to a logit.
const LogitBiasClamp = 100.0

// LogitMask is the injected schema/grammar constraint seam. MaskLogits is called on the
// post-head logits before argmax and must set every token that would leave the valid path to
// -inf, given the tokens emitted so far this turn (history is the generated ids, not the
// prompt). A masked token can never be selected, regardless of any logit_bias. Nil is the
// identity.
type LogitMask interface {
	MaskLogits(history []int, logits []float32)
}

// DecodeConstraint bundles the two native sampler hooks at the decode boundary. The zero
// value is inert: the constrained sampler reduces to the exact greedy argmax. Bias applies
// whenever non-empty; Mask applies only when the guided-decode flag is on.
type DecodeConstraint struct {
	Bias LogitBias
	Mask LogitMask
}

// GuidedDecodeEnabled reports whether the schema/grammar logit mask is on:
// FAK_NATIVE_GUIDED_DECODE=1, defaulting OFF, so a compiled-in Mask stays dormant until opted in.
func GuidedDecodeEnabled() bool {
	return os.Getenv("FAK_NATIVE_GUIDED_DECODE") == "1"
}

// maskActive reports whether the schema/grammar mask should run: a mask is present
// AND the feature flag is on. With the flag off, the mask is dormant.
func (c *DecodeConstraint) maskActive() bool {
	return c != nil && c.Mask != nil && GuidedDecodeEnabled()
}

// Active reports whether the constraint changes the decode at all; when false the sampler is
// the bit-exact-off path. A non-nil Mask alone is not active unless the flag is on.
func (c *DecodeConstraint) Active() bool {
	if c == nil {
		return false
	}
	return len(c.Bias) > 0 || c.maskActive()
}

// sampleConstrained selects the next token, applying the mask and then the logit-bias map
// before argmax; with an inert constraint it is byte-for-byte the greedy argmax path. The mask
// runs first so it wins over any bias (-inf + 100 is still -inf). Logits are copied so the
// caller's slice (a quantized Step may reuse it) is never mutated.
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

// GenerateConstrained greedily decodes up to n tokens after the prompt, applying the decode
// constraint at each step before argmax. With an inert constraint it is token-identical to
// Generate.
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

// GenerateBatchConstrained is GenerateBatch with an optional per-lane decode constraint. A nil
// slice, a missing lane entry, or an inert constraint is the identity matching GenerateBatch
// token-for-token.
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

// AllowedSetMask is the minimal concrete LogitMask: it permits only the token ids in the set at
// every step (a static mask), the building block a per-step grammar compiler emits.
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

// StepMask permits a different allowed-token set at each decode step, indexed by len(history):
// step k consults PerStep[k]. A nil entry, or a step past the end of PerStep, allows all
// tokens — the constraint has terminated.
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

// GuidedByteMask composes the byte-level guideddecode FSM into a per-step TOKEN mask:
// guideddecode.AllowedNextBytes decides which bytes may legally extend a valid tool-call
// envelope, and this adapter admits a candidate token iff its decoded bytes keep the prefix
// emitted so far this turn on an FSM-legal path.
//
// internal/model must not import internal/tokenizer, so the id->bytes decode is injected as
// TokenBytes (pass tokenizer.(*Tokenizer).TokenBytes as a func); guideddecode is a stdlib-only
// leaf and is imported directly.
//
// NOT WIRED LIVE — a tested adapter only: no default path constructs a DecodeConstraint with a
// Mask, and a compiled-in mask is dormant until FAK_NATIVE_GUIDED_DECODE=1. Live wiring must
// apply the mask inside the agent's stochastic sampler to preserve temperature/top-p; routing
// through the greedy GenerateConstrained would silently drop sampling.
type GuidedByteMask struct {
	// Schema is the declared tool-name set the envelope FSM constrains to.
	Schema guideddecode.ToolSchema
	// TokenBytes maps a token id to the exact bytes it decodes to, or nil for an id
	// that is not a decodable token. Inject tokenizer.(*Tokenizer).TokenBytes.
	TokenBytes func(id int) []byte
}

// MaskLogits sets to -inf every token whose decoded bytes cannot legally extend the current
// tool-call-envelope prefix (reconstructed by decoding history through TokenBytes).
//
// Admission is sound by construction: a candidate is admitted iff, replaying its decoded bytes
// from the current prefix, every byte is in guideddecode.AllowedNextBytes at the point it is
// consumed, or the FSM turns unconstrained partway through. It can only prevent an invalid
// tool name or broken envelope, never steer the decode off every valid envelope.
//
// Two degenerate prefixes are deliberate no-ops: an unconstrained prefix (skeleton consumed)
// and a dead-end prefix (empty non-nil allowed set). Masking every token on a dead end would
// zero the whole distribution, so the adapter declines and leaves recovery to the live-wiring
// slice. A nil receiver or nil TokenBytes is likewise the identity.
func (m *GuidedByteMask) MaskLogits(history []int, logits []float32) {
	if m == nil || m.TokenBytes == nil {
		return
	}
	prefix := m.decodePrefix(history)
	allowed := guideddecode.AllowedNextBytes(prefix, m.Schema)
	if allowed == nil {
		return // UNCONSTRAINED: skeleton consumed, no token constraint this step
	}
	if len(allowed) == 0 {
		return // DEAD END: decline rather than emit an all -inf vector (unreachable under correct masking)
	}
	neg := float32(math.Inf(-1))
	for id := range logits {
		if !m.tokenAdmissible(prefix, allowed, m.TokenBytes(id)) {
			logits[id] = neg
		}
	}
}

// decodePrefix reconstructs the byte prefix emitted so far this turn by decoding each history
// token through TokenBytes and concatenating.
func (m *GuidedByteMask) decodePrefix(history []int) []byte {
	var prefix []byte
	for _, id := range history {
		prefix = append(prefix, m.TokenBytes(id)...)
	}
	return prefix
}

// tokenAdmissible reports whether appending tb to prefix keeps the decode on an FSM-legal
// path. first == AllowedNextBytes(prefix), guaranteed non-nil and non-empty. A zero-byte
// candidate makes no progress and is rejected; prefix is never mutated.
func (m *GuidedByteMask) tokenAdmissible(prefix []byte, first map[byte]bool, tb []byte) bool {
	if len(tb) == 0 {
		return false
	}
	buf := make([]byte, 0, len(prefix)+len(tb))
	buf = append(buf, prefix...)
	allowed := first
	for i, b := range tb {
		if allowed == nil {
			return true // unconstrained from here: remaining bytes are free
		}
		if !allowed[b] {
			return false // diverges (also covers a dead-end empty set: no byte matches)
		}
		buf = append(buf, b)
		if i < len(tb)-1 {
			allowed = guideddecode.AllowedNextBytes(buf, m.Schema)
		}
	}
	return true
}
