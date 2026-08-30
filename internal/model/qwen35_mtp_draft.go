package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// Qwen35MTPMaxDraftDepth is the first production admission envelope for the
// single retained Qwen3.8 MTP head. Larger depths remain an explicit ordinary
// fak-native target-decode downgrade until they have their own bounded witness.
const Qwen35MTPMaxDraftDepth = 4

type qwen35MTPDraftStep func(*Qwen35MTPForward, int, []float32, []float32) ([]float32, []float32, error)

type qwen35MTPDraftCheckpoint struct {
	snapshot *PrefixSnapshot
	lastPos  int
}

// Qwen35MTPDraftSession owns one native MTP forward/cache and generates an
// ordered greedy draft block. Committed positions are always caught up from
// target hidden history. Only the unevaluated suffix feeds each MTP output
// hidden into the next MTP step.
type Qwen35MTPDraftSession struct {
	target     *Session
	depth      int
	forward    *Qwen35MTPForward
	step       qwen35MTPDraftStep
	processed  []int
	lastLogits []float32
	pending    *qwen35MTPDraftCheckpoint
	runtimeErr error
	closed     bool
}

// NewQwen35MTPDraftSession binds a fresh native Qwen3.8 MTP cache to an already
// evaluated target session. Unsupported depths, target paths, and checkpoint
// shapes fail before either session is mutated.
func NewQwen35MTPDraftSession(target *Session, depth int) (*Qwen35MTPDraftSession, error) {
	if err := validateQwen35MTPDepthNTarget(target, depth, false); err != nil {
		return nil, err
	}
	forward, err := target.M.NewQwen35MTPForward()
	if err != nil {
		return nil, err
	}
	return &Qwen35MTPDraftSession{
		target:  target,
		depth:   depth,
		forward: forward,
		step:    qwen35MTPForwardFeedback,
	}, nil
}

// Propose returns exactly the admitted depth unless a runtime failure is
// latched. The returned slice never aliases session-owned state.
func (d *Qwen35MTPDraftSession) Propose(committed []int) (draft []int) {
	if d == nil || d.runtimeErr != nil {
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

	defer func() {
		if recovered := recover(); recovered != nil {
			draft = nil
			d.failProposal("forward", -1, fmt.Errorf("%v", recovered))
		}
	}()

	if err := d.restoreProposal(); err != nil {
		d.latch("rollback", -1, err)
		return nil
	}
	if err := d.syncCommitted(committed); err != nil {
		d.latch("committed catch-up", len(d.processed), err)
		return nil
	}

	draft = make([]int, 0, d.depth)
	draft = append(draft, argmaxF32(d.lastLogits))
	if d.depth == 1 {
		return draft
	}
	if err := d.checkpointProposal(); err != nil {
		d.latch("checkpoint", len(committed), err)
		return nil
	}

	priorHidden, err := d.target.TargetHiddenAt(len(committed) - 1)
	if err != nil {
		d.failProposal("target hidden", len(committed)-1, err)
		return nil
	}
	current := draft[0]
	for len(draft) < d.depth {
		pos := len(committed) + len(draft) - 1
		embedding, err := d.target.TokenEmbedding(current)
		if err != nil {
			d.failProposal("token embedding", pos, err)
			return nil
		}
		feedback, logits, err := d.step(d.forward, pos, priorHidden, embedding)
		if err != nil {
			d.failProposal("forward", pos, err)
			return nil
		}
		if len(feedback) != d.target.M.Cfg.HiddenSize {
			d.failProposal("forward feedback", pos, qwen35MTPStateError(
				"draft feedback shape",
				fmt.Sprintf("[%d]", d.target.M.Cfg.HiddenSize),
				fmt.Sprintf("[%d]", len(feedback)),
			))
			return nil
		}
		if len(logits) == 0 {
			d.failProposal("forward", pos, ErrQwen35MTPEmptyLogits)
			return nil
		}
		current = argmaxF32(logits)
		draft = append(draft, current)
		priorHidden = feedback
	}
	return draft
}

// Drafter returns the polymodel seam used by the native speculative loop.
func (d *Qwen35MTPDraftSession) Drafter() polymodel.Drafter {
	if d == nil {
		return func([]int) []int { return nil }
	}
	return d.Propose
}

// Err returns the first draft-session runtime failure.
func (d *Qwen35MTPDraftSession) Err() error {
	if d == nil {
		return ErrQwen35MTPDrafterClosed
	}
	return d.runtimeErr
}

// Close releases the proposal checkpoint and native MTP cache. It is
// idempotent.
func (d *Qwen35MTPDraftSession) Close() {
	if d == nil || d.closed {
		return
	}
	d.closed = true
	if d.pending != nil {
		d.pending.snapshot.Close()
		d.pending = nil
	}
	if d.forward != nil {
		d.forward.Close()
		d.forward = nil
	}
	d.processed = nil
	d.lastLogits = nil
}

func (d *Qwen35MTPDraftSession) syncCommitted(committed []int) (err error) {
	if !qwen35MTPTargetHasEvaluatedPrefix(d.target, committed) {
		return fmt.Errorf(
			"model: target hidden for unevaluated committed prefix length %d is unavailable (cache=%d hidden=%d)",
			len(committed), d.target.Cache.Len(), len(d.target.targetHiddenTokens),
		)
	}
	if !tokenPrefix(d.processed, committed) {
		if err := d.recreateForward(); err != nil {
			return err
		}
	}
	if len(d.processed) == len(committed) {
		if len(d.lastLogits) == 0 {
			return ErrQwen35MTPEmptyLogits
		}
		return nil
	}

	checkpoint, err := d.forward.draft.PrefixSnapshot()
	if err != nil {
		return fmt.Errorf("model: snapshot Qwen3.8 MTP committed catch-up: %w", err)
	}
	baseProcessed := append([]int(nil), d.processed...)
	baseLogits := append([]float32(nil), d.lastLogits...)
	baseLastPos := d.forward.lastPos
	checkpointActive := true
	restore := func(cause error) error {
		if !checkpointActive {
			return cause
		}
		checkpointActive = false
		restoreErr := checkpoint.Restore(d.forward.draft)
		checkpoint.Close()
		d.forward.lastPos = baseLastPos
		d.processed = baseProcessed
		d.lastLogits = baseLogits
		if restoreErr != nil {
			return fmt.Errorf("model: Qwen3.8 MTP committed catch-up failed: %v; rollback failed: %w", cause, restoreErr)
		}
		return cause
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			err = restore(fmt.Errorf("model: Qwen3.8 MTP committed catch-up panic: %v", recovered))
		}
	}()

	for pos := len(d.processed); pos < len(committed); pos++ {
		priorHidden := make([]float32, d.target.M.Cfg.HiddenSize)
		if pos > 0 {
			priorHidden, err = d.target.TargetHiddenAt(pos - 1)
			if err != nil {
				return restore(err)
			}
		}
		embedding, err := d.target.TokenEmbedding(committed[pos])
		if err != nil {
			return restore(err)
		}
		_, logits, err := d.step(d.forward, pos, priorHidden, embedding)
		if err != nil {
			return restore(err)
		}
		if len(logits) == 0 {
			return restore(ErrQwen35MTPEmptyLogits)
		}
		d.processed = append(d.processed, committed[pos])
		d.lastLogits = append(d.lastLogits[:0], logits...)
	}
	checkpointActive = false
	checkpoint.Close()
	return nil
}

func (d *Qwen35MTPDraftSession) checkpointProposal() error {
	if d.pending != nil {
		return errors.New("model: Qwen3.8 MTP proposal checkpoint is already active")
	}
	snapshot, err := d.forward.draft.PrefixSnapshot()
	if err != nil {
		return fmt.Errorf("model: snapshot Qwen3.8 MTP proposal: %w", err)
	}
	d.pending = &qwen35MTPDraftCheckpoint{snapshot: snapshot, lastPos: d.forward.lastPos}
	return nil
}

func (d *Qwen35MTPDraftSession) restoreProposal() error {
	if d.pending == nil {
		return nil
	}
	pending := d.pending
	d.pending = nil
	err := pending.snapshot.Restore(d.forward.draft)
	pending.snapshot.Close()
	d.forward.lastPos = pending.lastPos
	if err != nil {
		return fmt.Errorf("model: restore Qwen3.8 MTP proposal: %w", err)
	}
	return nil
}

func (d *Qwen35MTPDraftSession) recreateForward() error {
	if d.forward != nil {
		d.forward.Close()
	}
	forward, err := d.target.M.NewQwen35MTPForward()
	if err != nil {
		d.forward = nil
		return err
	}
	d.forward = forward
	d.processed = nil
	d.lastLogits = nil
	return nil
}

func (d *Qwen35MTPDraftSession) failProposal(stage string, pos int, cause error) {
	if rollbackErr := d.restoreProposal(); rollbackErr != nil {
		cause = errors.Join(cause, rollbackErr)
	}
	d.latch(stage, pos, cause)
}

func (d *Qwen35MTPDraftSession) latch(stage string, pos int, cause error) {
	if d.runtimeErr == nil {
		d.runtimeErr = &Qwen35MTPDrafterError{Stage: stage, Position: pos, Err: cause}
	}
}

// qwen35MTPForwardFeedback returns the normalized MTP output hidden as well as
// logits. The hidden is the next speculative step's recurrent input; the draft
// layer's KV advances at the same position.
//
// Mechanism provenance (semantic adaptation; no source copied): vLLM
// b2dc864bb668da328aee8a8b0b72a0ad13c82252 returns the Qwen3.5 MTP normalized
// hidden and feeds it through AutoRegressiveSpeculator; SGLang
// 8a87079dbbf0f5b1543ec25d914dfd988eba42de carries logits_output.hidden_states
// step to step; llama.cpp 9723942adc518b43c4b95dc4dce6906903eb5e09 is
// reference-only and likewise feeds each Qwen3.5 draft hidden to the next step.
func qwen35MTPForwardFeedback(f *Qwen35MTPForward, pos int, priorHidden, currentEmbedding []float32) ([]float32, []float32, error) {
	if f == nil || f.target == nil || f.draft == nil || f.draft.Cache == nil {
		return nil, nil, qwen35MTPStateError("forward state", "initialized Qwen35MTPForward", "nil or incomplete")
	}
	if f.closed {
		return nil, nil, qwen35MTPStateError("forward state", "open Qwen35MTPForward", "closed")
	}
	if pos < 0 {
		return nil, nil, qwen35MTPStateError("position", "non-negative", fmt.Sprint(pos))
	}
	if pos <= f.lastPos {
		return nil, nil, qwen35MTPStateError("position", fmt.Sprintf("greater than %d", f.lastPos), fmt.Sprint(pos))
	}

	x, err := f.target.Qwen35MTPFuse(priorHidden, currentEmbedding)
	if err != nil {
		return nil, nil, err
	}
	cos, sin := ropeRowForLayer(f.draft.M.Cfg, 0, pos)
	x = f.draft.blockStep(0, pos, x, cos, sin, f32Kernel{f.draft.M})
	f.draft.Cache.appendPosition(pos, -1)
	f.lastPos = pos
	feedback := f.draft.M.finalNorm(x)
	return append([]float32(nil), feedback...), f.draft.head(feedback), nil
}

// SpecDecodeGreedyQwen35MTPDepthN is the bounded fak-native Qwen3.8 depth-N
// entry. It never selects, links, launches, or recovers through an external
// inference runtime.
func SpecDecodeGreedyQwen35MTPDepthN(target *Session, prompt []int, n, depth int) (polymodel.SpecDecodeRun, error) {
	return specDecodeGreedyQwen35MTPDepthN(target, prompt, n, depth, NewQwen35MTPDraftSession)
}

type qwen35MTPDepthNDraftBuilder func(*Session, int) (*Qwen35MTPDraftSession, error)

func specDecodeGreedyQwen35MTPDepthN(target *Session, prompt []int, n, depth int, build qwen35MTPDepthNDraftBuilder) (polymodel.SpecDecodeRun, error) {
	if len(prompt) == 0 {
		return polymodel.SpecDecodeRun{}, ErrQwen35MTPEmptyPrefix
	}
	if err := validateQwen35MTPDepthNTarget(target, depth, true); err != nil {
		return polymodel.SpecDecodeRun{}, err
	}
	if build == nil {
		return polymodel.SpecDecodeRun{}, errors.New("model: Qwen3.8 MTP depth-N drafter builder is required")
	}
	for _, token := range prompt {
		if _, err := target.TokenEmbedding(token); err != nil {
			return polymodel.SpecDecodeRun{}, err
		}
	}

	target.captureTargetHidden = true
	targetLogits := target.Prefill(prompt)
	if len(targetLogits) == 0 {
		return polymodel.SpecDecodeRun{}, errors.New("model: Qwen3.8 MTP target returned empty prompt logits")
	}
	advanceTarget := func(committed []int) error {
		if !qwen35MTPTargetMatchesCommitted(target, committed) {
			return errors.New("model: Qwen3.8 MTP live target diverged from committed prefix")
		}
		for _, token := range committed[target.Cache.Len():] {
			targetLogits = target.Step(token)
			if len(targetLogits) == 0 {
				return errors.New("model: Qwen3.8 MTP target returned empty decode logits")
			}
		}
		return nil
	}

	draftSession, err := build(target, depth)
	if err != nil {
		return polymodel.SpecDecodeRun{}, err
	}
	defer draftSession.Close()

	var runtimeErr error
	var pending *qwen35MTPTargetTransaction
	var pendingCommitted int
	propose := func(committed []int) []int {
		if runtimeErr != nil || draftSession.Err() != nil {
			return nil
		}
		if pending != nil {
			targetLogits, runtimeErr = pending.Commit(len(pending.draft))
			pending = nil
			pendingCommitted = 0
		}
		if runtimeErr == nil {
			runtimeErr = advanceTarget(committed)
		}
		if runtimeErr != nil {
			return nil
		}
		draft := draftSession.Propose(committed)
		if err := draftSession.Err(); err != nil {
			runtimeErr = err
		}
		return draft
	}
	verify := func(committed, draft []int) []int {
		if runtimeErr != nil || draftSession.Err() != nil {
			return nil
		}
		if err := advanceTarget(committed); err != nil {
			runtimeErr = err
			return nil
		}
		pending, err = beginQwen35MTPTargetTransaction(target, targetLogits)
		if err != nil {
			runtimeErr = err
			return nil
		}
		pendingCommitted = len(committed)
		rows, err := pending.Verify(draft)
		if err != nil {
			runtimeErr = err
			_ = pending.Abort()
			pending = nil
			pendingCommitted = 0
			return nil
		}
		argmax := make([]int, 0, len(rows)+1)
		argmax = append(argmax, argmaxF32(targetLogits))
		for _, row := range rows {
			argmax = append(argmax, argmaxF32(row))
		}
		return argmax
	}

	run, decodeErr := polymodel.SpecDecode(prompt, propose, verify, polymodel.SpecDecodeConfig{
		MaxNewTokens: n,
		MaxDraft:     depth,
		Rollback: func(evictKV int) {
			if pending == nil || runtimeErr != nil {
				return
			}
			accepted := len(pending.draft) - evictKV
			targetLogits, runtimeErr = pending.Commit(accepted)
			pending = nil
			pendingCommitted = 0
		},
	})
	if pending != nil {
		if decodeErr == nil && runtimeErr == nil {
			emitted := len(prompt) + len(run.Output) - pendingCommitted
			accepted := min(len(pending.draft), max(0, emitted))
			targetLogits, runtimeErr = pending.Commit(accepted)
		} else {
			_ = pending.Abort()
		}
		pending = nil
		pendingCommitted = 0
	}
	if draftErr := draftSession.Err(); draftErr != nil {
		return run, draftErr
	}
	if runtimeErr != nil {
		return run, runtimeErr
	}
	if decodeErr != nil {
		return run, decodeErr
	}
	committed := append(append([]int(nil), prompt...), run.Output...)
	if err := advanceTarget(committed); err != nil {
		return run, err
	}
	return run, nil
}

func validateQwen35MTPDepthNTarget(target *Session, depth int, requireFresh bool) error {
	if target == nil || target.M == nil {
		return errors.New("model: Qwen3.8 MTP target session is required")
	}
	if depth <= 0 {
		return ErrQwen35MTPInvalidDraftLength
	}
	if depth > Qwen35MTPMaxDraftDepth {
		return &Qwen35MTPSpecDecodeUnsupportedError{Reason: fmt.Sprintf(
			"draft depth %d exceeds the witnessed native bound %d",
			depth, Qwen35MTPMaxDraftDepth,
		)}
	}
	if target.Cache == nil || requireFresh && target.Cache.Len() != 0 {
		return &Qwen35MTPSpecDecodeUnsupportedError{Reason: "target session must be fresh; pass the complete prompt explicitly"}
	}
	if target.Backend != nil || target.Quant || target.Q4 || target.Q4K || target.F16 || target.GPTQ ||
		target.Metal || target.MetalQ4K || target.PrecisionPolicy != nil {
		return &Qwen35MTPSpecDecodeUnsupportedError{Reason: "only the native f32 target path has exact pre-final-norm hidden capture; production quant formats remain #9985"}
	}
	mode, err := target.M.Qwen35MTPMode(false)
	if err != nil {
		return err
	}
	if !mode.Enabled {
		return &Qwen35MTPSpecDecodeUnsupportedError{Reason: mode.Reason}
	}
	if err := target.M.validateQwen35MTPForwardTensors(); err != nil {
		return &Qwen35MTPSpecDecodeUnsupportedError{Reason: err.Error()}
	}
	return nil
}
