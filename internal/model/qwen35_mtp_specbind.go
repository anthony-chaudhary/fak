package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// Qwen35MTPTargetHidden returns the target model's final hidden state at the
// last token in prefix. prefix is non-empty and is copied before the callback,
// so the callback cannot corrupt the drafter's committed-prefix accounting.
// The callback must return a real target hidden state; the MTP adapter never
// substitutes an embedding or synthesizes one.
type Qwen35MTPTargetHidden func(prefix []int) ([]float32, error)

// Qwen35MTPTokenEmbedding returns the target model's input embedding for token.
// Acquisition is deliberately caller-owned because Qwen35MTPForward requires
// the exact current-token embedding, which is distinct from target hidden state.
type Qwen35MTPTokenEmbedding func(token int) ([]float32, error)

// Qwen35MTPDrafter setup and lifecycle errors that callers may match with
// errors.Is. Runtime callback and forward failures retain their original error
// as the innermost cause of Qwen35MTPDrafterError.
var (
	ErrQwen35MTPInvalidDraftLength = errors.New("model: Qwen3.8 MTP draft length must be positive")
	ErrQwen35MTPMissingHidden      = errors.New("model: Qwen3.8 MTP target-hidden callback is required")
	ErrQwen35MTPMissingEmbedding   = errors.New("model: Qwen3.8 MTP token-embedding callback is required")
	ErrQwen35MTPEmptyPrefix        = errors.New("model: Qwen3.8 MTP drafting needs a non-empty committed prefix")
	ErrQwen35MTPEmptyLogits        = errors.New("model: Qwen3.8 MTP forward returned empty logits")
	ErrQwen35MTPNilForward         = errors.New("model: Qwen3.8 MTP forward factory returned nil")
	ErrQwen35MTPDrafterClosed      = errors.New("model: Qwen3.8 MTP drafter is closed")
)

// Qwen35MTPDrafterError identifies the failed adapter stage and absolute token
// position. Position is -1 for setup or lifecycle failures not tied to a token.
type Qwen35MTPDrafterError struct {
	Stage    string
	Position int
	Err      error
}

func (e *Qwen35MTPDrafterError) Error() string {
	if e == nil {
		return "<nil>"
	}
	if e.Position >= 0 {
		return fmt.Sprintf("model: Qwen3.8 MTP drafter %s at position %d: %v", e.Stage, e.Position, e.Err)
	}
	return fmt.Sprintf("model: Qwen3.8 MTP drafter %s: %v", e.Stage, e.Err)
}

func (e *Qwen35MTPDrafterError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

type qwen35MTPDraftForward interface {
	Forward(pos int, priorHidden, currentEmbedding []float32) ([]float32, error)
	Close()
}

type qwen35MTPDraftForwardFactory func() (qwen35MTPDraftForward, error)

// Qwen35MTPDrafter owns a Qwen35MTPForward and adapts it to polymodel.Drafter.
// It is intentionally single-threaded, matching polymodel.SpecDecode's call
// pattern. Call Close when the speculative run is done; Close is idempotent.
// Reset closes and recreates the owned forward, and is the only supported way
// to clear a latched runtime error and reuse the adapter.
//
// Because polymodel.Drafter cannot return an error, Propose latches the first
// callback or forward error and returns no proposal. The caller must check Err
// after SpecDecodeGreedyWithDrafter returns and surface it. Returning no proposal
// never substitutes a target/session drafter: polymodel merely verifies and
// commits its own greedy token for that round.
type Qwen35MTPDrafter struct {
	k              int
	targetHidden   Qwen35MTPTargetHidden
	tokenEmbedding Qwen35MTPTokenEmbedding
	newForward     qwen35MTPDraftForwardFactory
	forward        qwen35MTPDraftForward
	processed      []int
	lastLogits     []float32
	runtimeErr     error
	closed         bool
}

// NewQwen35MTPDrafter constructs a polymodel drafter backed by the model's real
// Qwen35MTPForward. draftLength is the maximum greedy MTP tokens returned per
// call. The callbacks explicitly supply the target hidden state and target
// current-token embedding required at every forwarded position.
func NewQwen35MTPDrafter(m *Model, draftLength int, targetHidden Qwen35MTPTargetHidden, tokenEmbedding Qwen35MTPTokenEmbedding) (*Qwen35MTPDrafter, error) {
	if m == nil {
		return nil, &Qwen35MTPDrafterError{Stage: "setup", Position: -1, Err: qwen35MTPStateError("model", "non-nil model", "nil")}
	}
	return newQwen35MTPDrafter(draftLength, targetHidden, tokenEmbedding, func() (qwen35MTPDraftForward, error) {
		return m.NewQwen35MTPForward()
	})
}

func newQwen35MTPDrafter(draftLength int, targetHidden Qwen35MTPTargetHidden, tokenEmbedding Qwen35MTPTokenEmbedding, factory qwen35MTPDraftForwardFactory) (*Qwen35MTPDrafter, error) {
	if draftLength <= 0 {
		return nil, &Qwen35MTPDrafterError{Stage: "setup", Position: -1, Err: ErrQwen35MTPInvalidDraftLength}
	}
	if targetHidden == nil {
		return nil, &Qwen35MTPDrafterError{Stage: "setup", Position: -1, Err: ErrQwen35MTPMissingHidden}
	}
	if tokenEmbedding == nil {
		return nil, &Qwen35MTPDrafterError{Stage: "setup", Position: -1, Err: ErrQwen35MTPMissingEmbedding}
	}
	if factory == nil {
		return nil, &Qwen35MTPDrafterError{Stage: "setup", Position: -1, Err: ErrQwen35MTPNilForward}
	}

	forward, err := factory()
	if err != nil {
		return nil, &Qwen35MTPDrafterError{Stage: "forward setup", Position: -1, Err: err}
	}
	if forward == nil {
		return nil, &Qwen35MTPDrafterError{Stage: "forward setup", Position: -1, Err: ErrQwen35MTPNilForward}
	}
	return &Qwen35MTPDrafter{
		k:              draftLength,
		targetHidden:   targetHidden,
		tokenEmbedding: tokenEmbedding,
		newForward:     factory,
		forward:        forward,
	}, nil
}

// Drafter returns the polymodel seam consumed by SpecDecodeGreedyWithDrafter.
// The returned closure shares this adapter's lifecycle and error latch.
func (d *Qwen35MTPDrafter) Drafter() polymodel.Drafter {
	if d == nil {
		return nilQwen35MTPDraft
	}
	return d.Propose
}

// Propose returns up to the configured number of greedy MTP tokens. It is
// exported mainly for direct lifecycle use; SpecDecodeGreedyWithDrafter callers
// normally pass Drafter().
func (d *Qwen35MTPDrafter) Propose(committed []int) []int {
	if d == nil {
		return nil
	}
	if d.runtimeErr != nil {
		return nil
	}
	if d.closed {
		d.latch("lifecycle", -1, ErrQwen35MTPDrafterClosed)
		return nil
	}
	if len(committed) == 0 {
		d.latch("committed prefix", -1, ErrQwen35MTPEmptyPrefix)
		return nil
	}

	path := append([]int(nil), committed...)
	if !tokenPrefix(d.processed, path) {
		if err := d.recreateForward(false); err != nil {
			return nil
		}
	}
	if !d.forwardThrough(path) {
		return nil
	}

	draft := make([]int, 0, d.k)
	for len(draft) < d.k {
		next := argmaxF32(d.lastLogits)
		draft = append(draft, next)
		if len(draft) == d.k {
			break
		}
		path = append(path, next)
		if !d.forwardThrough(path) {
			return nil
		}
	}
	return draft
}

// Err returns the first runtime callback, forward, or misuse error. It is stable
// until a successful Reset clears it; Close does not erase diagnostic state.
func (d *Qwen35MTPDrafter) Err() error {
	if d == nil {
		return ErrQwen35MTPDrafterClosed
	}
	return d.runtimeErr
}

// Reset discards all MTP KV state by closing and recreating the forward. Close
// is not treated as a reset: a closed adapter cannot be reopened. A successful
// Reset clears the prior runtime error and committed-prefix history.
func (d *Qwen35MTPDrafter) Reset() error {
	if d == nil || d.closed {
		return &Qwen35MTPDrafterError{Stage: "reset", Position: -1, Err: ErrQwen35MTPDrafterClosed}
	}
	return d.recreateForward(true)
}

// Close releases the currently owned Qwen35MTPForward. It is idempotent. Any
// forwards replaced by Reset or prefix divergence were closed at replacement.
func (d *Qwen35MTPDrafter) Close() {
	if d == nil || d.closed {
		return
	}
	d.closed = true
	if d.forward != nil {
		d.forward.Close()
		d.forward = nil
	}
	d.processed = nil
	d.lastLogits = nil
}

func (d *Qwen35MTPDrafter) forwardThrough(path []int) bool {
	for len(d.processed) < len(path) {
		pos := len(d.processed)
		prefix := append([]int(nil), path[:pos+1]...)
		hidden, err := d.targetHidden(prefix)
		if err != nil {
			d.latch("target hidden", pos, err)
			return false
		}
		embedding, err := d.tokenEmbedding(path[pos])
		if err != nil {
			d.latch("token embedding", pos, err)
			return false
		}
		logits, err := d.forward.Forward(pos, hidden, embedding)
		if err != nil {
			d.latch("forward", pos, err)
			return false
		}
		if len(logits) == 0 {
			d.latch("forward", pos, ErrQwen35MTPEmptyLogits)
			return false
		}
		d.processed = append(d.processed, path[pos])
		d.lastLogits = append(d.lastLogits[:0], logits...)
	}
	return true
}

func (d *Qwen35MTPDrafter) recreateForward(clearRuntimeErr bool) error {
	if d.forward != nil {
		d.forward.Close()
		d.forward = nil
	}
	d.processed = nil
	d.lastLogits = nil

	forward, err := d.newForward()
	if err == nil && forward == nil {
		err = ErrQwen35MTPNilForward
	}
	if err != nil {
		wrapped := &Qwen35MTPDrafterError{Stage: "forward reset", Position: -1, Err: err}
		d.runtimeErr = wrapped
		return wrapped
	}
	d.forward = forward
	if clearRuntimeErr {
		d.runtimeErr = nil
	}
	return nil
}

func (d *Qwen35MTPDrafter) latch(stage string, pos int, err error) {
	if d.runtimeErr != nil {
		return
	}
	d.runtimeErr = &Qwen35MTPDrafterError{Stage: stage, Position: pos, Err: err}
	// A callback or forward failure leaves no proven reusable KV state. Close it
	// now; Reset must create a fresh forward before another proposal can run.
	if d.forward != nil {
		d.forward.Close()
		d.forward = nil
	}
}

func tokenPrefix(prefix, tokens []int) bool {
	if len(prefix) > len(tokens) {
		return false
	}
	for i := range prefix {
		if prefix[i] != tokens[i] {
			return false
		}
	}
	return true
}
