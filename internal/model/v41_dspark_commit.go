package model

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
)

// These internal host-reference operations bind token lineage, not checkpoint
// authentication. The shard reader validates schema only. A verified payload,
// tokenizer/template identity and qualified quantized shared embedding/head are
// still required before any serving admission. No caller-supplied digest string
// is treated as evidence of those properties.
//
// The caller exclusively owns the Session through preparation, target drain and
// publication. f.mu serializes reference calls; it does not lock the Session.
func (f *v41DSparkReference) captureBindingLocked(c *v41DSparkCapture, round *v41StepRound) bool {
	if f.closed || f.invalid || f.target == nil || f.model == nil || f.state == nil ||
		!f.nativeHostLocked() || f.target.M != f.model || !f.target.modelWeightsHeld ||
		f.target.v41Forward != f.state || f.state.callbackOwner != f.target ||
		c == nil || c.closed || c.owner != f.target || c.model != f.model || c.state != f.state ||
		c.round != round || f.state.dsparkCapture != c || f.state.activeStepRound != round ||
		c.width != f.cfg.HiddenSize || c.layers != f.cfg.NumLayers {
		return false
	}
	taps := f.cfg.DeepSeekV41.DSpark.TargetLayerIDs
	if len(taps) != 3 || c.taps != [3]int{taps[0], taps[1], taps[2]} ||
		c.end != len(f.state.history) || c.finalHash != v41DSparkPrefix(f.state.history) {
		return false
	}
	return true
}

// BootstrapCapturedTail consumes a successful committed capture into a fresh
// empty reference. Absolute positions and the entire token-prefix hash survive
// tail eviction. Empty history already has an empty draft and needs no bootstrap.
//
// This does not execute or own cold prefill. In particular, it cannot restore a
// target published before a late cold cleanup failure. The enclosing cold owner
// must restore a complete snapshot or discard that Session. A failed projection
// leaves this draft empty; a consumed capture is not reusable.
func (f *v41DSparkReference) BootstrapCapturedTail(ctx context.Context, c *v41DSparkCapture) error {
	if f == nil {
		return errV41DSparkReference
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.ready(ctx); err != nil {
		return err
	}
	if f.next != 0 || f.prefix != sha256.Sum256(nil) || !f.captureBindingLocked(c, nil) ||
		c.end < 1 || c.start != max(0, c.end-v41WindowSize) || len(c.rows) != c.end-c.start {
		return fmt.Errorf("%w: fresh complete captured tail required", errV41DSparkReference)
	}
	for _, ring := range f.ring {
		for _, row := range ring {
			if row != nil {
				return fmt.Errorf("%w: bootstrap ring is not empty", errV41DSparkReference)
			}
		}
	}
	defer c.close()
	next, prefix, start := c.end, c.finalHash, c.start
	rows, err := c.committed()
	if err != nil {
		return err
	}
	staged, err := f.stageRowsLocked(ctx, rows, start)
	if err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	f.ring, f.next, f.prefix = staged, next, prefix
	return nil
}

// FinishCapturedRound consumes only an identically bound complete retained
// round. keep is 1..6: the already-selected anchor plus accepted draft tokens.
// The caller supplies keep from exact target verification; this helper does not
// choose an acceptance count. Zero remains an explicit round.abort operation.
// Foreign or mismatched inputs
// remain caller-owned and untouched, including their abort/close obligation.
//
// All fallible draft work is completed while the target undo journal is live.
// After finish returns cleanly, only field assignments publish the draft.
// A clean preparation-error abort preserves the old draft. An incomplete drain
// permanently invalidates it, but does not mark it closed or release its hold.
func (f *v41DSparkReference) FinishCapturedRound(ctx context.Context, c *v41DSparkCapture, round *v41StepRound, keep int) (err error) {
	if f == nil {
		return errV41DSparkReference
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if ctx == nil || !f.captureBindingLocked(c, round) || round == nil || !round.ownsState() ||
		round.model != f.model || round.state != f.state || keep < 1 || keep > 6 || keep > len(c.rows) ||
		f.next != round.baseHistoryLen || f.next != c.base || f.prefix != c.baseHash ||
		f.next < 0 || f.next > len(f.state.history) || v41DSparkPrefix(f.state.history[:f.next]) != f.prefix {
		return fmt.Errorf("%w: retained round ownership or prefix", errV41DSparkReference)
	}
	normal, published, finishStarted := false, false, false
	// Registered before the inner direct abort: observe its final drain state,
	// including panic(nil)/Goexit, while retaining the reference mutex.
	defer func() {
		if !normal || !round.closed || round.poisoned || (finishStarted && !published) {
			f.invalid = true
			if normal {
				err = errors.Join(err, fmt.Errorf("%w: incomplete target drain", errV41DSparkReference))
			}
		}
	}()
	err = func() error {
		defer c.close()
		// Reuse the target's bounded reverse undo and its documented preservation
		// of a non-nil primary panic. No stronger compound-unwind promise is made.
		defer round.abort()
		rows, err := c.staged(round, keep)
		if err != nil {
			return err
		}
		staged, err := f.stageRowsLocked(ctx, rows, f.next)
		if err != nil {
			return err
		}
		next := f.next + keep
		prefix := v41DSparkPrefix(f.state.history[:next])
		if !f.captureBindingLocked(c, round) {
			return fmt.Errorf("%w: target changed during draft preparation", errV41DSparkReference)
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		finishStarted = true
		if err := round.finish(keep); err != nil {
			return err
		}
		f.ring, f.next, f.prefix = staged, next, prefix
		published = true
		return nil
	}()
	normal = true
	return err
}
