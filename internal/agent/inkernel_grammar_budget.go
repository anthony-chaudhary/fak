package agent

// Grammar/completion-token budget adapted from TensorFold's grammar_budget
// feature flag (Apache-2.0): when armed, every grammar mask keeps the reply
// completable within the request's token ceiling by clearing the bits of the
// tokens after which no reply completes within the remaining draws. Source:
// ashhart/TensorFold zig/src/server/server.zig:339 (the grammar_budget arm and
// refusal ladder) and zig/src/core/lanes/grammar.zig:234 (Constraint.Budget)
// at 9e8c368b0b4c2c7892a0c9f0f302d7c164016209; see THIRD_PARTY_NOTICES.md. No
// upstream Zig code blocks are copied.

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// nativeGuidedBudgetKey is the guided-decode side-band member that arms the
// completion-token budget on whichever grammar the request carries. It is an
// option, not a constraint carrier: it binds no pattern of its own.
const nativeGuidedBudgetKey = "grammar_budget"

var (
	// ErrNativeGrammarBudget is the parent of every grammar-budget refusal.
	ErrNativeGrammarBudget = errors.New("native grammar budget refused")
	// ErrNativeGrammarBudgetNotBoolean is a grammar_budget value that is not a
	// boolean.
	ErrNativeGrammarBudgetNotBoolean = fmt.Errorf("%w: value must be a boolean", ErrNativeGrammarBudget)
	// ErrNativeGrammarBudgetUncompiled is a budget armed with no compiled
	// grammar mask to bound.
	ErrNativeGrammarBudgetUncompiled = fmt.Errorf("%w: no compiled grammar mask", ErrNativeGrammarBudget)
	// ErrNativeGrammarBudgetABI is a runtime without a usable probing ABI for
	// the mask's completions (missing vocabulary or out-of-range EOS).
	ErrNativeGrammarBudgetABI = fmt.Errorf("%w: no usable completion-bound ABI", ErrNativeGrammarBudget)
	// ErrNativeGrammarBudgetRules is a grammar whose rules are not coverable by
	// a completion bound within the exploration caps.
	ErrNativeGrammarBudgetRules = fmt.Errorf("%w: rules not coverable by a completion bound", ErrNativeGrammarBudget)
)

// nativeGuidedBudgetError carries a grammar-budget sentinel across the guided
// boundary. With a carrier it is also an ErrNativeStructuredUnsupported
// refusal, so a budget refusal refuses before sampling like every other
// guided refusal.
type nativeGuidedBudgetError struct {
	Carrier string
	Err     error
}

func (e *nativeGuidedBudgetError) Error() string {
	if e.Carrier == "" {
		return e.Err.Error()
	}
	return fmt.Sprintf("%v: %v (guided carrier %q)", ErrNativeStructuredUnsupported, e.Err, e.Carrier)
}

func (e *nativeGuidedBudgetError) Unwrap() []error {
	if e.Carrier == "" {
		return []error{e.Err}
	}
	return []error{e.Err, ErrNativeStructuredUnsupported}
}

// parseNativeGrammarBudget reads one present, non-null guided-decode raw
// message as the grammar_budget flag.
func parseNativeGrammarBudget(raw json.RawMessage) (bool, error) {
	var flag bool
	if err := json.Unmarshal(raw, &flag); err != nil {
		return false, ErrNativeGrammarBudgetNotBoolean
	}
	return flag, nil
}

// nativeGrammarBudgetCeiling is the request's token ceiling: sp.MaxTokens when
// positive, otherwise the planner's configured default, matching the decode
// loop's own maxNew derivation.
func nativeGrammarBudgetCeiling(sp SampleParams, plannerDefault int) int {
	if sp.MaxTokens != nil && *sp.MaxTokens > 0 {
		return *sp.MaxTokens
	}
	return plannerDefault
}

const (
	// nativeGrammarBudgetNodeCap bounds the frontier expansions one budget
	// context performs; past it the budget degrades to fail-open rather than
	// block a completable path.
	nativeGrammarBudgetNodeCap = 4096
	// nativeGrammarBudgetReadyDepth is the draw depth arm-time exploration
	// accepts as proof that the grammar's rules are coverable.
	nativeGrammarBudgetReadyDepth = 2048
	// nativeGrammarBudgetMemoCap bounds the shared memo; lookups keep working
	// once it is full.
	nativeGrammarBudgetMemoCap = 8192
)

// nativeGrammarBudgetContext is the mutable exploration state shared by the
// armed budget and its speculative forks: the expansion counter, the degraded
// flag, and the memo of proven completion distances.
type nativeGrammarBudgetContext struct {
	nodes    int
	degraded bool
	memo     map[string]int
}

// nativeGrammarBudgetProbe bounds a compiled grammar mask's completions
// through its public logit-mask surface. It walks the mask's allowed frontier
// depth-first from each queried prefix and records the fewest draws to a
// complete reply. The walk is node-capped: a frontier that outgrows the cap
// degrades the shared context to fail-open instead of blocking a completable
// path. Compiled masks answer from their full history alone, so sibling
// prefixes may be queried in any order.
type nativeGrammarBudgetProbe struct {
	mask    model.LogitMask
	eosID   int
	scratch []float32
	ctx     *nativeGrammarBudgetContext
}

// newNativeGrammarBudgetProbe returns the generic probe for mask over a
// vocabulary of vocab ids with stop id eosID, or nil when the runtime has no
// usable probing ABI.
func newNativeGrammarBudgetProbe(mask model.LogitMask, eosID, vocab int) *nativeGrammarBudgetProbe {
	if mask == nil || vocab <= 0 || eosID < 0 || eosID >= vocab {
		return nil
	}
	return &nativeGrammarBudgetProbe{mask: mask, eosID: eosID, scratch: make([]float32, vocab), ctx: &nativeGrammarBudgetContext{memo: make(map[string]int)}}
}

// nativeBudgetReady reports whether the mask's rules are covered by a
// completion bound and, when they are, the fewest draws from the start of the
// reply to a complete reply, counting the closing EOS draw.
func (b *nativeGrammarBudgetProbe) nativeBudgetReady() (int, bool) {
	minDraws := b.minDrawsFrom(nil, nativeGrammarBudgetReadyDepth)
	if minDraws > nativeGrammarBudgetReadyDepth {
		return 0, false
	}
	return minDraws, true
}

// nativeBudgetMinDraws reports the fewest further draws to a complete reply
// from the state after history, counting the closing EOS draw, capped at
// limit+1 (limit+1: more than limit, or not covered).
func (b *nativeGrammarBudgetProbe) nativeBudgetMinDraws(history []int, limit int) int {
	return b.minDrawsFrom(append([]int(nil), history...), limit)
}

// minDrawsFrom explores the mask's allowed frontier from prefix and reports
// the fewest draws to a complete reply, counting the closing EOS draw, or
// limit+1 when no completion fits within limit draws or the exploration
// degrades. Results are memoized per (prefix, limit) in the shared context.
func (b *nativeGrammarBudgetProbe) minDrawsFrom(prefix []int, limit int) int {
	if limit < 1 || b.ctx.degraded {
		return limit + 1
	}
	key := nativeGrammarBudgetKey(prefix, limit)
	if value, ok := b.ctx.memo[key]; ok {
		return value
	}
	value := b.explorePrefix(prefix, limit)
	if len(b.ctx.memo) < nativeGrammarBudgetMemoCap {
		b.ctx.memo[key] = value
	}
	return value
}

// explorePrefix is one uncached frontier expansion from prefix. A successor
// distance equal to its own cap is a "does not fit" sentinel, never a real
// distance, so only sub <= limit-1 advances the bound.
func (b *nativeGrammarBudgetProbe) explorePrefix(prefix []int, limit int) int {
	if b.ctx.nodes >= nativeGrammarBudgetNodeCap {
		b.ctx.degraded = true
		return limit + 1
	}
	b.ctx.nodes++
	allowed, eosOK := b.allowedAt(prefix)
	if eosOK {
		return 1
	}
	best := limit + 1
	for _, token := range allowed {
		if token == b.eosID {
			continue
		}
		sub := b.minDrawsFrom(append(prefix, token), limit-1)
		if sub <= limit-1 && sub+1 < best {
			best = sub + 1
			if best == 2 {
				break // a token then EOS: nothing shorter remains
			}
		}
	}
	return best
}

// allowedAt applies the mask to a zero logit vector at prefix and reports the
// surviving token ids in ascending order, and whether the EOS survives.
func (b *nativeGrammarBudgetProbe) allowedAt(prefix []int) (allowed []int, eosOK bool) {
	for i := range b.scratch {
		b.scratch[i] = 0
	}
	b.mask.MaskLogits(prefix, b.scratch)
	for id, logit := range b.scratch {
		if !nativeStructuredLogitFinite(logit) {
			continue
		}
		allowed = append(allowed, id)
		if id == b.eosID {
			eosOK = true
		}
	}
	return allowed, eosOK
}

// nativeGrammarBudgetKey encodes a (prefix, limit) memo key.
func nativeGrammarBudgetKey(prefix []int, limit int) string {
	buf := make([]byte, 4*len(prefix)+8)
	for i, id := range prefix {
		binary.LittleEndian.PutUint32(buf[4*i:], uint32(id))
	}
	// Limits are native ints, not vocabulary IDs. Retain all their bits on
	// 64-bit hosts so budgets separated by 2^32 never reuse a memo result.
	binary.LittleEndian.PutUint64(buf[4*len(prefix):], uint64(int64(limit)))
	return string(buf)
}

// nativeGrammarBudget is the armed completion budget: it keeps every masked
// step on a path that completes the reply within maxTokens draws, reasoning
// tokens included. emitted counts accepted tokens for the attempt; reset zeros
// it because an OOM retry restarts the attempt, while the arming itself is
// per-request state the retry keeps.
type nativeGrammarBudget struct {
	probe     *nativeGrammarBudgetProbe
	maxTokens int
	emitted   int
	minDraws  int // fewest draws from the start of the reply to a complete reply
}

// clone shares the probe (and its memo) with a speculative fork while copying
// the fork's own emission count.
func (b *nativeGrammarBudget) clone() *nativeGrammarBudget {
	return &nativeGrammarBudget{probe: b.probe, maxTokens: b.maxTokens, emitted: b.emitted, minDraws: b.minDraws}
}

// armGrammarBudget arms the completion-token budget on the constraint's
// compiled mask with ceiling as the request's token limit. It refuses a
// missing mask (uncompiled grammar), a runtime without a usable probing ABI,
// and rules whose completions cannot be bounded within the exploration caps.
func (c *nativeStructuredConstraint) armGrammarBudget(p *InKernelPlanner, ceiling int) error {
	refuse := func(sentinel error) error {
		return &nativeGuidedBudgetError{Carrier: nativeGuidedBudgetKey, Err: sentinel}
	}
	if c.mask == nil {
		return refuse(ErrNativeGrammarBudgetUncompiled)
	}
	vocab := 0
	if p != nil && p.m != nil {
		vocab = p.m.Cfg.VocabSize
	}
	probe := newNativeGrammarBudgetProbe(c.mask, c.stopID, vocab)
	if probe == nil {
		return refuse(ErrNativeGrammarBudgetABI)
	}
	minDraws, ready := probe.nativeBudgetReady()
	if !ready {
		return refuse(ErrNativeGrammarBudgetRules)
	}
	c.budget = &nativeGrammarBudget{probe: probe, maxTokens: ceiling, minDraws: minDraws}
	return nil
}

// applyBudget keeps every surviving non-stop token on a path that completes
// the reply within the request's remaining draws, mirroring TensorFold's
// budget mask: a token is cleared only when no completion provably fits after
// it, and when nothing fits the grammar mask is left untouched rather than
// dead-ending the reply.
func (c *nativeStructuredConstraint) applyBudget() {
	b := c.budget
	left := b.maxTokens - b.emitted
	if left < 1 || b.probe.ctx.degraded {
		return
	}
	if cap(c.budgetSnapshot) < len(c.maskBuffer) {
		c.budgetSnapshot = make([]float32, len(c.maskBuffer))
	} else {
		c.budgetSnapshot = c.budgetSnapshot[:len(c.maskBuffer)]
	}
	copy(c.budgetSnapshot, c.maskBuffer)
	kept := 0
	for id, logit := range c.maskBuffer {
		if !nativeStructuredLogitFinite(logit) || id == c.stopID {
			continue
		}
		if b.probe.nativeBudgetMinDraws(append(c.history, id), left-1) <= left-1 {
			kept++
		} else {
			c.maskBuffer[id] = float32(math.Inf(-1))
		}
	}
	if kept == 0 {
		copy(c.maskBuffer, c.budgetSnapshot)
	}
}
