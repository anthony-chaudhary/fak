package ggufload

import (
	"context"
	"fmt"
	"runtime"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Inkling2LoopMaxBatch is the widest cohort one row-batched forward serves.
const Inkling2LoopMaxBatch = 16

// inkling2BatchCtxAlign rounds a cohort's KV capacity so cohorts of similar length
// share arena sizes.
const inkling2BatchCtxAlign = 64

// Inkling2LoopTurn is the per-token contract of one greedy turn, shared by the
// single-sequence and the row-batched decode so both count the same tokens: a stop
// token ends the turn unemitted and uncounted, every other token is counted and
// handed to Emit, and the turn ends when Emit refuses it or MaxTokens are counted.
type Inkling2LoopTurn struct {
	MaxTokens int
	Stop      func(int) bool
	Emit      func(int) bool
	Generated int
	// Stopped is set when a stop token or a refusing Emit ended the turn.
	Stopped bool
}

// Accept applies the contract to the next greedy token and reports whether the
// turn wants another.
func (t *Inkling2LoopTurn) Accept(token int) bool {
	if t.Generated >= t.MaxTokens {
		return false
	}
	if t.Stop != nil && t.Stop(token) {
		t.Stopped = true
		return false
	}
	t.Generated++
	if !t.Emit(token) {
		t.Stopped = true
		return false
	}
	return t.Generated < t.MaxTokens
}

// Inkling2LoopSessionDecode runs turn on a fresh session: prefill prompt, then
// step greedily until the turn ends. ctx is checked before every step.
func Inkling2LoopSessionDecode(ctx context.Context, s Inkling2LoopSession, prompt []int, turn *Inkling2LoopTurn) (st Inkling2LoopGreedyStats, err error) {
	start := time.Now()
	next, err := s.Prefill(prompt)
	st.Prefill = time.Since(start)
	if err != nil {
		return st, err
	}
	start = time.Now()
	defer func() { st.Decode = time.Since(start) }()
	for turn.Accept(next) {
		if err := ctx.Err(); err != nil {
			return st, err
		}
		if next, err = s.Step(next); err != nil {
			return st, err
		}
		st.DecodeSteps++
	}
	return st, nil
}

// Inkling2LoopBatchRequest is one row of a cohort. Emit follows Inkling2LoopTurn.
// Done, when set, receives the row's result as soon as the row ends, while the
// cohort may still be stepping; neither Emit nor Done is called for the row after.
type Inkling2LoopBatchRequest struct {
	Context   context.Context
	Prompt    []int
	MaxTokens int
	Stop      func(int) bool
	Emit      func(int) bool
	Done      func(Inkling2LoopBatchResult)
}

// Inkling2LoopBatchResult is what one row of a cohort did.
type Inkling2LoopBatchResult struct {
	Stats     Inkling2LoopGreedyStats
	Generated int
	Stopped   bool
	Err       error
}

// Inkling2LoopBatchEligible reports whether the row-batched forward can serve m on
// be: it needs every expert resident and the backend's row-batched ops.
func Inkling2LoopBatchEligible(m *Inkling2LoopModel, be compute.Backend) bool {
	if m == nil || be == nil || !inkling2ResidentEligible(m, be) {
		return false
	}
	_, ok := be.(compute.Inkling2PrefillDevice)
	return ok
}

// inkling2BatchCohortCtx sizes every row's global-slot KV to the cohort's longest
// row instead of the resident context: fp32 KV at the resident context times 16
// rows does not fit a 16 GB card. Every row steps until the cohort drains, so the
// longest prompt+budget bounds every row's position.
func inkling2BatchCohortCtx(reqs []Inkling2LoopBatchRequest, resident int) (int, error) {
	need := 0
	for i, r := range reqs {
		n := len(r.Prompt) + r.MaxTokens
		if n-1 > resident {
			return 0, fmt.Errorf("inkling2 batch stream: lane %d needs %d positions, past the %d-token context", i, n-1, resident)
		}
		need = max(need, n)
	}
	ctx := (need + inkling2BatchCtxAlign - 1) / inkling2BatchCtxAlign * inkling2BatchCtxAlign
	return min(resident, ctx), nil
}

type inkling2BatchLane struct {
	req         Inkling2LoopBatchRequest
	turn        Inkling2LoopTurn
	result      Inkling2LoopBatchResult
	last        int
	done        bool
	started     time.Time
	decodeStart time.Time
	sinkTime    time.Duration
	// skip is the prompt positions prefilled before the first step
	skip int
}

func newInkling2BatchLane(req Inkling2LoopBatchRequest, started time.Time) inkling2BatchLane {
	if req.Context == nil {
		req.Context = context.Background()
	}
	return inkling2BatchLane{req: req, started: started,
		turn: Inkling2LoopTurn{MaxTokens: req.MaxTokens, Stop: req.Stop, Emit: req.Emit}}
}

// finish seals the row's result and hands it to Done; the row is inert after.
func (lane *inkling2BatchLane) finish(now time.Time, err error) {
	lane.done = true
	r := &lane.result
	r.Err, r.Generated, r.Stopped = err, lane.turn.Generated, lane.turn.Stopped
	r.Stats.DecodeSteps = max(0, r.Generated-1)
	if lane.decodeStart.IsZero() {
		r.Stats.Prefill = now.Sub(lane.started)
	} else {
		r.Stats.Decode = now.Sub(lane.decodeStart) - lane.sinkTime
	}
	if lane.req.Done != nil {
		lane.req.Done(*r)
	}
}

func (lane *inkling2BatchLane) accept(token int, now time.Time) {
	if lane.decodeStart.IsZero() {
		lane.result.Stats.Prefill = now.Sub(lane.started)
		lane.decodeStart = now
	}
	if err := lane.req.Context.Err(); err != nil {
		lane.finish(now, err)
		return
	}
	t0 := time.Now()
	more := lane.turn.Accept(token)
	lane.sinkTime += time.Since(t0)
	lane.last = token
	if !more {
		lane.finish(time.Now(), nil)
	}
}

// Inkling2LoopCohorts runs cohorts on one cached state: the weight binding is made
// once, and the per-sequence arenas grow to the widest and longest cohort seen and
// are reused by every cohort that fits, so back-to-back cohorts neither rebind the
// weights nor churn the device allocator. It is not safe for concurrent use.
type Inkling2LoopCohorts struct {
	m  *Inkling2LoopModel
	be compute.Backend
	s  *inkling2Resident
	b  *inkling2Batch
}

func NewInkling2LoopCohorts(m *Inkling2LoopModel, be compute.Backend) *Inkling2LoopCohorts {
	return &Inkling2LoopCohorts{m: m, be: be}
}

// Close frees the cached state; a later Generate builds it again.
func (bt *Inkling2LoopCohorts) Close() {
	if bt.b != nil {
		bt.b.release()
		bt.b = nil
	}
	if bt.s != nil {
		bt.s.close()
		bt.s = nil
	}
}

// ReleaseArenas frees the per-sequence arenas and keeps the weight binding, so a
// single-sequence session can allocate beside the binding instead of beside the
// widest cohort's KV; the next cohort allocates arenas again.
func (bt *Inkling2LoopCohorts) ReleaseArenas() {
	if bt.b == nil {
		return
	}
	// a captured step graph addresses the arenas being freed
	if bt.s != nil && bt.s.dev != nil {
		bt.s.dev.Ink2GraphReset()
	}
	bt.b.release()
	bt.b = nil
}

// acquire returns the cached state readied for a C x ctx cohort, regrowing it to
// cover both the old and the new shape when the cohort does not fit.
func (bt *Inkling2LoopCohorts) acquire(C, ctx int) (*inkling2Batch, error) {
	if bt.s == nil {
		s, err := newInkling2BatchResident(bt.m, bt.be)
		if err != nil {
			return nil, err
		}
		bt.s = s
	}
	if bt.b != nil && C <= bt.b.capC && ctx <= bt.b.capCtx {
		return bt.b, bt.b.begin(C, ctx)
	}
	growC, growCtx := C, ctx
	if bt.b != nil {
		growC, growCtx = max(C, bt.b.capC), max(ctx, bt.b.capCtx)
		// free first so the old and the grown state never coexist on the device
		bt.b.release()
		bt.b = nil
	}
	b, err := newInkling2BatchState(bt.s, growC, growCtx)
	if err != nil {
		return nil, err
	}
	bt.b = b
	return b, b.begin(C, ctx)
}

// Inkling2LoopBatchGenerate runs one cohort on a state built for it and freed after.
func Inkling2LoopBatchGenerate(m *Inkling2LoopModel, be compute.Backend, reqs []Inkling2LoopBatchRequest) ([]Inkling2LoopBatchResult, error) {
	bt := NewInkling2LoopCohorts(m, be)
	defer bt.Close()
	return bt.Generate(reqs)
}

// Generate runs one row-batched greedy forward over a cohort of
// 2..Inkling2LoopMaxBatch rows. Each row owns its prompt, cache, stop/cancel state
// and token budget, and its context is checked before every step; a finished row
// feeds an ignored token until the cohort drains and its callbacks and accounting
// never advance again.
func (bt *Inkling2LoopCohorts) Generate(reqs []Inkling2LoopBatchRequest) (_ []Inkling2LoopBatchResult, fatal error) {
	if len(reqs) < 2 || len(reqs) > Inkling2LoopMaxBatch {
		return nil, fmt.Errorf("inkling2 batch stream: cohort size %d outside 2..%d", len(reqs), Inkling2LoopMaxBatch)
	}
	for i, req := range reqs {
		if len(req.Prompt) == 0 || req.MaxTokens <= 0 || req.Emit == nil {
			return nil, fmt.Errorf("inkling2 batch stream: invalid lane %d", i)
		}
	}
	ctx, err := inkling2BatchCohortCtx(reqs, Inkling2LoopResidentContext)
	if err != nil {
		return nil, err
	}
	// a captured step graph is bound to the capturing thread
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	// a failed or panicking step leaves the binding's sticky error or a half-built
	// graph behind, so only a clean cohort keeps the state
	clean := false
	defer func() {
		if !clean {
			bt.Close()
		}
	}()
	b, err := bt.acquire(len(reqs), ctx)
	if err != nil {
		return nil, err
	}

	started := time.Now()
	lanes := make([]inkling2BatchLane, len(reqs))
	for i, req := range reqs {
		lanes[i] = newInkling2BatchLane(req, started)
	}
	if b.prefillT > 0 {
		fatal = prefillLanes(b, lanes)
	}
	tokens := make([]int, len(lanes))
	for step := 0; fatal == nil; step++ {
		active := 0
		for i := range lanes {
			lane := &lanes[i]
			if !lane.done {
				if err := lane.req.Context.Err(); err != nil {
					lane.finish(time.Now(), err)
				} else {
					active++
				}
			}
			if lane.done && lane.skip > 0 {
				b.park(i)
			}
			switch fed := step + lane.skip; {
			case fed < len(lane.req.Prompt):
				tokens[i] = lane.req.Prompt[fed]
			case lane.turn.Generated > 0:
				tokens[i] = lane.last
			default:
				tokens[i] = lane.req.Prompt[len(lane.req.Prompt)-1]
			}
		}
		if active == 0 {
			break
		}
		next, err := b.step(tokens)
		if err != nil {
			fatal = err
			now := time.Now()
			for i := range lanes {
				if !lanes[i].done {
					lanes[i].finish(now, err)
				}
			}
			break
		}
		now := time.Now()
		for i := range lanes {
			lane := &lanes[i]
			if lane.done || step+lane.skip < len(lane.req.Prompt)-1 {
				continue
			}
			lane.accept(next[i], now)
		}
	}
	results := make([]Inkling2LoopBatchResult, len(lanes))
	for i := range lanes {
		results[i] = lanes[i].result
	}
	clean = fatal == nil
	return results, fatal
}

// prefillLanes prefills every row's prompt and hands each live row its first greedy
// token, leaving the row where the lockstep loop would be once it had fed the whole
// prompt; a failed prefill finishes every row with the error, as a failed step does.
func prefillLanes(b *inkling2Batch, lanes []inkling2BatchLane) error {
	prompts := make([][]int, len(lanes))
	for i := range lanes {
		prompts[i] = lanes[i].req.Prompt
	}
	first, err := b.prefill(prompts)
	now := time.Now()
	for i := range lanes {
		lane := &lanes[i]
		switch {
		case err != nil:
			lane.finish(now, err)
		default:
			lane.skip = len(lane.req.Prompt)
			lane.accept(first[i], now)
		}
	}
	return err
}
