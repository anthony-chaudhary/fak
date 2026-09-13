package mtptune

import (
	"errors"
	"fmt"
)

// ErrSpecDecodeAnchorCorruption is the typed refusal for the cited corruption
// mode: a speculative draft that anchors the FIRST emitted token. Under the
// default 1+N bonus layout the first emitted token comes from the target's own
// position (the anchor's already-computed logits), NOT from a draft token.
// Reusing the anchor as the draft anchor makes a rejected draft commit a stale
// self-KV entry, so the committed stream diverges from base greedy decoding.
// Callers match it with errors.Is.
var ErrSpecDecodeAnchorCorruption = errors.New("mtptune: speculative draft anchored the first token; this commits stale self-KV on rejection")

// SpecTokenLayout names how a speculative verify panel lays out emitted tokens
// relative to the target's own position. The layout is load-bearing for
// losslessness: it decides which token is committed when a draft is rejected.
type SpecTokenLayout string

const (
	// LayoutBonus1N is the default 1+N bonus layout. The panel emits one
	// target/bonus token (the target's own argmax at the anchor, index 0)
	// followed by N accepted draft tokens. The first emitted token therefore
	// comes from the target's own logits, so a rejection cannot commit a stale
	// self-KV entry and the stream stays bit-exact with base greedy decoding.
	LayoutBonus1N SpecTokenLayout = "bonus_1n"

	// LayoutAnchorFirst is the corruption mode: the draft reuses the anchor as
	// the first emitted token. On rejection the anchor's stale self-KV is
	// committed, flipping a later greedy decision. Engaging this layout is
	// refused with ErrSpecDecodeAnchorCorruption.
	LayoutAnchorFirst SpecTokenLayout = "anchor_first"
)

// DefaultSpecTokenLayout returns the only layout a lossless decoder may engage.
func DefaultSpecTokenLayout() SpecTokenLayout { return LayoutBonus1N }

// SpecStep is one speculative draft/verify step: the draft token ids proposed
// after the anchor and the target's own greedy token at each verify position.
// Under the 1+N bonus layout, TargetTokens[0] is the target's token at the
// anchor (the bonus token) and TargetTokens[i] is the target's token after
// accepting Draft[:i].
type SpecStep struct {
	Anchor       int
	Draft        []int
	TargetTokens []int
}

// AcceptResult is the losslessness-accounted outcome of a speculative step.
//
// Emitted is the committed token stream: the target's bonus token followed by
// the accepted draft prefix. Accepted is the length of the accepted draft
// prefix; Rejected is len(Draft)-Accepted. Next is the target's correction
// token at the rejection point (TargetTokens[Accepted]), i.e. the next token
// the decoder continues from.
//
// RolledBack holds the KV positions a full-panel append would have used for the
// REJECTED DRAFT SUFFIX: the notional positions base+1+Accepted .. base+1+len(Draft)-1
// that a naive "append all drafts, then truncate" would have created and then
// undone. These are hypothetical append positions, NOT committed positions that
// must be truncated — the committed stream already excludes them. The real undo
// contract is the pre-step length (KV.Rollback(base)); the committed length
// after this step is base + len(Emitted).
type AcceptResult struct {
	Emitted    []int
	Accepted   int
	Rejected   int
	RolledBack []int
	Next       int
}

// KVCache is a minimal committed-position store. Committed holds the token ids
// that have actually been committed to KV; Len is len(Committed). A speculative
// step may read the cache but must only append through Commit, and must undo any
// appended positions through Rollback when a draft is rejected. The invariant is
// that after a step Len equals the pre-step length plus the target's own bonus
// token plus any accepted drafts; rejected draft tokens never remain committed.
type KVCache struct {
	Committed []int
}

// NewKVCache returns a cache seeded with the given committed token ids.
func NewKVCache(committed ...int) *KVCache {
	return &KVCache{Committed: append([]int(nil), committed...)}
}

// Len reports the number of committed positions.
func (c *KVCache) Len() int { return len(c.Committed) }

// Commit appends accepted token ids to the committed KV stream.
func (c *KVCache) Commit(tokens ...int) {
	c.Committed = append(c.Committed, tokens...)
}

// Rollback truncates the committed stream to n positions. It is a no-op if n is
// not a valid position, so a caller can always roll back to the pre-step length.
func (c *KVCache) Rollback(n int) {
	if n < 0 || n > len(c.Committed) {
		return
	}
	c.Committed = c.Committed[:n]
}

// Snapshot returns a copy of the committed token ids.
func (c *KVCache) Snapshot() []int {
	return append([]int(nil), c.Committed...)
}

// BaseGreedyToken returns the base decoder's greedy token for one logits row:
// the argmax, ties broken by the lowest index to stay deterministic.
func BaseGreedyToken(logitsRow []float64) int {
	best := -1
	bestVal := 0.0
	for i, v := range logitsRow {
		if best == -1 || v > bestVal {
			bestVal = v
			best = i
		}
	}
	return best
}

// BaseGreedyDecode is the pure, deterministic base greedy oracle. It walks a
// sequence of logits rows and emits one argmax token per row. Speculative output
// must be bit-exact with this stream. It takes no model and no state beyond its
// arguments, so a test can compare against it directly.
func BaseGreedyDecode(logitsRows [][]float64) []int {
	out := make([]int, 0, len(logitsRows))
	for _, row := range logitsRows {
		out = append(out, BaseGreedyToken(row))
	}
	return out
}

// VerifySpecStep runs one draft/verify/accept step under the losslessness
// contract and mutates the KV cache accordingly:
//
//   - The layout must be LayoutBonus1N. Any other layout (notably
//     LayoutAnchorFirst) is refused with ErrSpecDecodeAnchorCorruption.
//   - Draft and TargetTokens must be aligned: TargetTokens[i+1] is the target's
//     own token for Draft[i], and TargetTokens has len(Draft)+1 entries (index 0
//     is the bonus/anchor token).
//   - The target's own bonus token (TargetTokens[0]) is ALWAYS committed,
//     whether or not any draft is accepted. The draft's accepted prefix is
//     committed after it, so committed length grows by 1 + accepted.
//   - On a partial rejection the step commits the bonus token plus the accepted
//     draft prefix; only the rejected draft suffix is never committed. The step
//     is NOT all-or-nothing and does not roll the cache back to the pre-step
//     length unless the first draft token itself is rejected (accepted == 0),
//     in which case the committed length grows by exactly 1 (the bonus token).
//
// The returned AcceptResult reports the accepted/rejected counts and the KV
// positions a full-panel append would have used for the rejected suffix
// (RolledBack), so the rollback contract is checkable against KV.Rollback(base).
// Emitted is bit-exact with BaseGreedyDecode over the same verify positions
// under a perfect draft.
func VerifySpecStep(layout SpecTokenLayout, kv *KVCache, step SpecStep) (AcceptResult, error) {
	if layout != LayoutBonus1N {
		return AcceptResult{}, ErrSpecDecodeAnchorCorruption
	}
	if len(step.TargetTokens) != len(step.Draft)+1 {
		return AcceptResult{}, fmt.Errorf("mtptune: 1+N bonus layout needs %d target tokens for %d draft tokens, got %d",
			len(step.Draft)+1, len(step.Draft), len(step.TargetTokens))
	}
	if kv == nil {
		return AcceptResult{}, errors.New("mtptune: nil KV cache")
	}

	base := len(kv.Committed)

	accepted := 0
	for accepted < len(step.Draft) && step.Draft[accepted] == step.TargetTokens[accepted+1] {
		accepted++
	}
	next := step.TargetTokens[accepted]
	rejected := len(step.Draft) - accepted

	// The committed stream under 1+N bonus is ALWAYS the target's own bonus
	// token at the anchor followed by the accepted draft prefix — a partial
	// rejection still commits every token the target itself produced, so the
	// stream stays bit-exact with base greedy decoding. Only the rejected draft
	// suffix is rolled back; rejected tokens are never passed to Commit.
	emitted := make([]int, 0, accepted+1)
	emitted = append(emitted, step.TargetTokens[0])
	emitted = append(emitted, step.Draft[:accepted]...)
	kv.Commit(emitted...)

	// RolledBack reports the notional KV positions a full-panel append would
	// have used for the rejected draft suffix. len(emitted) is accepted+1, so
	// these are base+1+accepted .. base+1+len(Draft)-1: positions the rollback
	// elides, not committed positions. A caller that instead appends the whole
	// panel can undo it with kv.Rollback(base+len(emitted)).
	rolledBack := make([]int, 0, rejected)
	for i := accepted; i < len(step.Draft); i++ {
		rolledBack = append(rolledBack, base+len(emitted)+i-accepted)
	}

	return AcceptResult{
		Emitted:    emitted,
		Accepted:   accepted,
		Rejected:   rejected,
		RolledBack: rolledBack,
		Next:       next,
	}, nil
}

// SpecDecodeConfig gates speculative decode. Spec-decode is OFF by default: the
// zero value (Enabled:false) never engages. Even when Enabled, Engage refuses
// unless the losslessness guard holds, so an operator cannot turn on a
// corrupting path by flipping one field.
type SpecDecodeConfig struct {
	Enabled  bool
	DraftLen int
	Layout   SpecTokenLayout
}

// DefaultSpecDecodeConfig returns the off-by-default configuration.
func DefaultSpecDecodeConfig() SpecDecodeConfig {
	return SpecDecodeConfig{
		Enabled:  false,
		DraftLen: 1,
		Layout:   DefaultSpecTokenLayout(),
	}
}

// Validate checks the config without engaging spec-decode. It refuses an unknown
// layout or a draft length below 1, and refuses LayoutAnchorFirst with the typed
// error. It does not require Enabled.
func (cfg SpecDecodeConfig) Validate() error {
	if cfg.Layout != LayoutBonus1N {
		return ErrSpecDecodeAnchorCorruption
	}
	if cfg.DraftLen < 1 {
		return fmt.Errorf("mtptune: draft length must be >= 1, got %d", cfg.DraftLen)
	}
	return nil
}

// Engage returns the effective draft length only when speculative decode is
// enabled AND the losslessness guard passes. The default config returns
// ErrSpecDecodeDisabled. A layout other than the default 1+N bonus layout is
// refused with ErrSpecDecodeAnchorCorruption, so losslessness is a precondition
// of engagement rather than a runtime hope.
func (cfg SpecDecodeConfig) Engage() (int, error) {
	if !cfg.Enabled {
		return 0, ErrSpecDecodeDisabled
	}
	if err := cfg.Validate(); err != nil {
		return 0, err
	}
	return cfg.DraftLen, nil
}

// ErrSpecDecodeDisabled is returned when Engage is called on a config that has
// not enabled speculative decode. The zero SpecDecodeConfig is disabled.
var ErrSpecDecodeDisabled = errors.New("mtptune: speculative decode is disabled by default")
