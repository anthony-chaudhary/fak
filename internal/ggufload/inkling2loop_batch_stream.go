package ggufload

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"sort"
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
	next, err := s.Prefill(ctx, prompt)
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
	// origin is the request index a cohort lane answers, or -1 for an admitted one
	origin int
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
// once, and the per-sequence arenas are reused by every later cohort they cover
// without holding much more than that cohort needs, so back-to-back cohorts of a
// similar shape neither rebind the weights nor churn the device allocator, while a
// narrower or shorter cohort after a wide one gives the surplus back. It is not safe
// for concurrent use.
type Inkling2LoopCohorts struct {
	m  *Inkling2LoopModel
	be compute.Backend
	s  *inkling2Resident
	b  *inkling2Batch
	// wideCooldown is how many more wide-wanting cohorts run at the base chunk before
	// a refused wide chunk is tried again
	wideCooldown int
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

func (bt *Inkling2LoopCohorts) bind() error {
	if bt.s != nil {
		return nil
	}
	s, err := newInkling2BatchResident(bt.m, bt.be)
	if err != nil {
		return err
	}
	bt.s = s
	return nil
}

// inkling2BatchReuseSlack bounds the surplus a cached state may hold over a cohort it
// covers and still serve it: a state with more than 1/inkling2BatchReuseSlack more
// sequence positions (rows x ctx, which the KV and conv state scale with) than the
// cohort is freed and rebuilt at the cohort's shape.
const inkling2BatchReuseSlack = 4

// reuses reports whether the cached state serves a C x ctx cohort.
func (bt *Inkling2LoopCohorts) reuses(C, ctx int) bool {
	b := bt.b
	if b == nil || C > b.capC || ctx > b.capCtx {
		return false
	}
	cells := C * ctx
	return b.capC*b.capCtx <= cells+cells/inkling2BatchReuseSlack
}

// acquire returns a state readied for a C x ctx cohort: the cached one when it covers
// the cohort without much surplus, else one built at exactly the cohort's shape. The
// old state is freed first, so the two never coexist on the device.
func (bt *Inkling2LoopCohorts) acquire(C, ctx int) (*inkling2Batch, error) {
	return bt.acquireFor(C, ctx, 0)
}

// acquireFor is acquire for a cohort whose longest prompt is longest tokens: only a
// cohort with a prompt past the base chunk asks for the wide one, and a cached state
// built narrower than that is rebuilt. After a refusal the wide chunk is not probed
// again for a few cohorts, so a full device does not rebuild the state on every cohort.
func (bt *Inkling2LoopCohorts) acquireFor(C, ctx, longest int) (*inkling2Batch, error) {
	if err := bt.bind(); err != nil {
		return nil, err
	}
	want := inkling2CohortChunk(bt.s.chunkWanted, longest)
	if want > inkling2PrefillChunkBase && bt.wideCooldown > 0 {
		bt.wideCooldown--
		want = inkling2PrefillChunkBase
	}
	if bt.reuses(C, ctx) && want <= bt.b.capT {
		bt.b.usePrefill(want)
		return bt.b, bt.b.begin(C, ctx)
	}
	if bt.b != nil {
		bt.ReleaseArenas()
	}
	b, err := newInkling2BatchState(bt.s, C, ctx, want)
	if err != nil {
		return nil, err
	}
	if b.capT < want {
		bt.wideCooldown = inkling2PrefillReprobeEvery
	}
	bt.b = b
	return b, nil
}

// Split partitions a cohort into sub-cohorts whose states each fit free device memory.
// positions[i] is row i's len(prompt)+MaxTokens. A cohort that fits, or a device that
// cannot report free memory, stays one group. Otherwise rows are taken shortest first
// and each group grows while its state, sized at its own longest row, still fits, so
// a long row never sizes the short rows' KV. A group of one row is for the serialized
// engine: a cohort needs two. The cached state counts as free, since acquire frees it
// before building one it does not reuse.
func (bt *Inkling2LoopCohorts) Split(positions []int) ([][]int, error) {
	all := make([]int, len(positions))
	for i := range all {
		all[i] = i
	}
	if len(positions) < 2 || !inkling2BatchAdmitOn(os.LookupEnv) {
		return [][]int{all}, nil
	}
	if err := bt.bind(); err != nil {
		return nil, err
	}
	inkling2TrimIdle(bt.be)
	_, free, known := inkling2DeviceFree(bt.be)
	if !known || free < 0 {
		return [][]int{all}, nil
	}
	if bt.b != nil {
		free += bt.b.need
	}
	ctxOf := func(n int) int {
		ctx := (n + inkling2BatchCtxAlign - 1) / inkling2BatchCtxAlign * inkling2BatchCtxAlign
		return max(1, min(Inkling2LoopResidentContext, ctx))
	}
	// groups run in plan order, so the cached state is there to reuse only until a
	// planned group that does not reuse it has its acquire replace it
	cached := true
	fits := func(C, longest int) bool {
		ctx := ctxOf(longest)
		sh, err := bt.s.batchShape(C, ctx)
		if err != nil {
			return false
		}
		return (cached && bt.reuses(C, ctx)) || sh.need+inkling2BatchHeadroom <= free
	}
	longest := 0
	for _, n := range positions {
		longest = max(longest, n)
	}
	if len(positions) <= Inkling2LoopMaxBatch && fits(len(positions), longest) {
		return [][]int{all}, nil
	}
	order := append([]int(nil), all...)
	sort.SliceStable(order, func(a, b int) bool { return positions[order[a]] < positions[order[b]] })
	var groups [][]int
	group := []int{order[0]}
	for _, i := range order[1:] {
		if len(group) < Inkling2LoopMaxBatch && fits(len(group)+1, positions[i]) {
			group = append(group, i)
			continue
		}
		cached = cached && bt.reuses(len(group), ctxOf(positions[group[len(group)-1]]))
		groups = append(groups, group)
		group = []int{i}
	}
	return append(groups, group), nil
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
func (bt *Inkling2LoopCohorts) Generate(reqs []Inkling2LoopBatchRequest) ([]Inkling2LoopBatchResult, error) {
	return bt.GenerateAdmitting(reqs, nil)
}

// GenerateAdmitting is Generate with continuous admission: when a row finishes, its
// slot takes the next request admit returns, at the next chunk or step boundary.
// Admission rides the mixed driver, so admit is called only when cohort prefill and the
// long-prefill cap are both on. The cohort width and the per-row KV capacity b.ctx are
// fixed at begin, so a request joins only when len(prompt)+MaxTokens <= b.ctx; that bound
// is checked here as well, and a request past it is answered through its Done with an
// error rather than admitted. A joined row restarts at position 0 on a cleared history,
// and attention reads no position its row has not written.
func (bt *Inkling2LoopCohorts) GenerateAdmitting(reqs []Inkling2LoopBatchRequest, admit func(maxPositions int) (Inkling2LoopBatchRequest, bool)) (_ []Inkling2LoopBatchResult, fatal error) {
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
	longest := 0
	for _, req := range reqs {
		longest = max(longest, len(req.Prompt))
	}
	b, err := bt.acquireFor(len(reqs), ctx, longest)
	if err != nil {
		// a refused or failed arena allocation ran nothing on the binding, so it is kept
		// for the narrower cohorts the caller retries with
		var refused *Inkling2BatchMemoryError
		clean = errors.As(err, &refused)
		return nil, err
	}

	started := time.Now()
	lanes := make([]inkling2BatchLane, len(reqs))
	results := make([]Inkling2LoopBatchResult, len(reqs))
	for i, req := range reqs {
		lanes[i] = newInkling2BatchLane(req, started)
		lanes[i].origin = i
	}
	longCap := inkling2CohortLongPrefillCap(os.LookupEnv)
	mixed := b.prefillT > 0 && longCap > 0
	if b.prefillT > 0 && !mixed {
		fatal = prefillLanes(b, lanes)
	}
	keep, first := prefillHooks(lanes)
	if mixed && b.sharedMin > 0 {
		fatal = prefillSharedLanes(b, lanes, keep)
	}
	tokens := make([]int, len(lanes))
	for step := 0; fatal == nil; {
		if mixed && admit != nil {
			if fatal = admitLanes(b, lanes, results, admit); fatal != nil {
				break
			}
		}
		active, prefilling := 0, false
		for i := range lanes {
			lane := &lanes[i]
			if !lane.done {
				if err := lane.req.Context.Err(); err != nil {
					lane.finish(time.Now(), err)
				} else {
					active++
					prefilling = prefilling || (mixed && b.pos[i] < len(lane.req.Prompt))
				}
			}
			// a slot joined above is live here, so it is never parked
			if lane.done {
				b.park(i)
			}
		}
		if active == 0 {
			break
		}
		if prefilling {
			if fatal = prefillMixedStep(b, lanes, longCap, keep, first); fatal != nil {
				break
			}
			continue
		}
		for i := range lanes {
			lane := &lanes[i]
			switch fed := step + lane.skip; {
			case fed < len(lane.req.Prompt):
				tokens[i] = lane.req.Prompt[fed]
			case lane.turn.Generated > 0:
				tokens[i] = lane.last
			default:
				tokens[i] = lane.req.Prompt[len(lane.req.Prompt)-1]
			}
		}
		next, err := b.step(tokens)
		if err != nil {
			fatal = err
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
		step++
	}
	if fatal != nil {
		finishLive(lanes, fatal)
	}
	for i := range lanes {
		if lanes[i].origin >= 0 {
			results[lanes[i].origin] = lanes[i].result
		}
	}
	clean = fatal == nil
	return results, fatal
}

// prefillLanes prefills every lane's prompt and hands each live lane its first greedy
// token at the chunk that ends its prompt, leaving the row where the lockstep loop
// would be once it had fed the whole prompt. A lane cancelled before a chunk is
// finished at that chunk, so its remaining prompt costs no compute and its caller
// hears the cancellation as soon as the cohort reaches it. A failed prefill finishes
// every unfinished lane with the error, as a failed step does.
func prefillLanes(b *inkling2Batch, lanes []inkling2BatchLane) error {
	prompts := make([][]int, len(lanes))
	for i := range lanes {
		prompts[i] = lanes[i].req.Prompt
	}
	keep, first := prefillHooks(lanes)
	err := b.prefill(prompts, keep, first)
	finishPrefilledLanes(lanes, err)
	return err
}

// prefillSharedLanes prefills the prompt prefix the cohort's lanes share once before
// the mixed driver takes over, leaving each sharing row past the prefix; the mixed
// chunks then start every row at its own position.
func prefillSharedLanes(b *inkling2Batch, lanes []inkling2BatchLane, keep func(int) bool) error {
	prompts := make([][]int, len(lanes))
	end := make([]int, len(lanes))
	for i := range lanes {
		prompts[i] = lanes[i].req.Prompt
		end[i] = len(prompts[i])
	}
	cursor := make([]int, len(lanes))
	if err := b.prefillShared(prompts, end, cursor, keep); err != nil {
		return err
	}
	for _, c := range cursor {
		if c > 0 {
			b.endPrefill(cursor)
			break
		}
	}
	return nil
}

// prefillHooks are the prefill callbacks for lanes: keep drops a cancelled lane at the
// chunk boundary, and first hands a lane its first greedy token.
func prefillHooks(lanes []inkling2BatchLane) (keep func(int) bool, first func(int, int)) {
	keep = func(i int) bool {
		lane := &lanes[i]
		if lane.done {
			return false
		}
		if err := lane.req.Context.Err(); err != nil {
			lane.skip = len(lane.req.Prompt)
			lane.finish(time.Now(), err)
			return false
		}
		return true
	}
	first = func(i, token int) {
		lane := &lanes[i]
		if lane.done {
			return
		}
		lane.skip = len(lane.req.Prompt)
		if err := lane.req.Context.Err(); err != nil {
			lane.finish(time.Now(), err)
			return
		}
		lane.accept(token, time.Now())
	}
	return keep, first
}

// finishPrefilledLanes ends the lanes a prefill left unfinished: with its error when
// it failed, and with a plain error for a live lane that never got a first token.
func finishPrefilledLanes(lanes []inkling2BatchLane, err error) {
	now := time.Now()
	for i := range lanes {
		lane := &lanes[i]
		switch {
		case lane.done:
		case err != nil:
			lane.finish(now, err)
		case lane.skip != len(lane.req.Prompt):
			lane.finish(now, errors.New("inkling2 batch: prefill ended without a first token for a live row"))
		}
	}
}

// prefillMixedStep runs one mixed chunk over the live rows: a prefilling row takes its
// planned share of prompt tokens and a decoding row feeds its last token. Each row's
// position is its cursor, so the chunk and the lockstep step agree on where a row writes.
// It runs nothing when no live row is left to prefill once keep has dropped cancelled ones.
func prefillMixedStep(b *inkling2Batch, lanes []inkling2BatchLane, longCap int,
	keep func(int) bool, first func(int, int)) error {
	cursor := append([]int(nil), b.pos...)
	lens := make([]int, len(lanes))
	var decoding []int
	more := false
	for i := range lanes {
		lane := &lanes[i]
		keep(i)
		lens[i] = len(lane.req.Prompt)
		switch {
		case lane.done:
			lens[i] = cursor[i]
		case cursor[i] < lens[i]:
			more = true
		default:
			decoding = append(decoding, i)
		}
	}
	if !more {
		return nil
	}
	if b.prefillT-len(decoding) < 1 {
		return errors.New("inkling2 batch: no room in a prefill chunk for a prompt token")
	}
	// a mixed chunk runs prefill kernels between captured steps, which may grow scratch
	// a captured step still addresses, so the graph is captured again after it
	if b.graphReady {
		b.s.dev.Ink2GraphReset()
		b.graphReady = false
	}
	segs := inkling2MixedPrefillPlan(lens, cursor, decoding, b.prefillT, longCap)
	toks := make([][]int, len(segs))
	for k, sg := range segs {
		prompt := lanes[sg.row].req.Prompt
		if sg.pos0 < len(prompt) {
			toks[k] = prompt[sg.pos0 : sg.pos0+sg.n]
		} else {
			toks[k] = []int{lanes[sg.row].last}
		}
	}
	if err := b.prefillChunk(segs, toks, first); err != nil {
		return err
	}
	for _, sg := range segs {
		cursor[sg.row] += sg.n
	}
	b.endPrefill(cursor)
	return nil
}

// finishLive ends every row still unfinished with err.
func finishLive(lanes []inkling2BatchLane, err error) {
	now := time.Now()
	for i := range lanes {
		if !lanes[i].done {
			lanes[i].finish(now, err)
		}
	}
}

// admitLanes offers every finished slot to admit in slot order and stops at the first
// refusal. A request that does not fit is answered with its error and never joins. A
// request with no Done is dropped, since nothing could hear its result.
func admitLanes(b *inkling2Batch, lanes []inkling2BatchLane, results []Inkling2LoopBatchResult,
	admit func(int) (Inkling2LoopBatchRequest, bool)) error {
	for i := range lanes {
		if !lanes[i].done {
			continue
		}
		req, ok := admit(b.ctx)
		if !ok {
			return nil
		}
		if req.Done == nil {
			continue
		}
		if err := inkling2AdmitCheck(req, b.ctx); err != nil {
			req.Done(Inkling2LoopBatchResult{Err: err})
			continue
		}
		if old := &lanes[i]; old.origin >= 0 {
			results[old.origin] = old.result
		}
		lanes[i] = newInkling2BatchLane(req, time.Now())
		lanes[i].origin = -1
		if err := b.resetRow(i); err != nil {
			return err
		}
	}
	return nil
}

// inkling2AdmitCheck validates an admitted request as Generate validates its cohort.
func inkling2AdmitCheck(req Inkling2LoopBatchRequest, maxPositions int) error {
	switch {
	case len(req.Prompt) == 0 || req.MaxTokens <= 0 || req.Emit == nil:
		return errors.New("inkling2 batch stream: invalid admitted request")
	case len(req.Prompt)+req.MaxTokens > maxPositions:
		return fmt.Errorf("inkling2 batch stream: admitted request needs %d positions, past the %d-token cohort", len(req.Prompt)+req.MaxTokens, maxPositions)
	}
	return nil
}
