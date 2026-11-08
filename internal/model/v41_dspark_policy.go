// v41_dspark_policy.go — V4.1 DSpark confidence-scheduled variable-length
// verification slot allocation (borrow-port oss-port-deepinfer-dspark-varlen-policy).
//
// Clean-room ADAPT of the pure policy core of DeepInfer's MTP variable-length
// scheduler (deepseek-ai/DeepInfer, csrc/engine/mtp_varlen.hpp, schedule() at
// lines 150-267 plus its two runtime estimators, pin
// 774ce244b87cc0eb8d4b94ab9f5de2ab22049f01, Apache-2.0). The contract is
// reimplemented from that reading; no upstream source text is copied.
//
// The policy decides how many draft positions each scheduled request gets
// verified in one decode step:
//   - Total verification work is chosen from AGE-SEPARATED N-2 confidence
//     estimates only, so a future round's confidences never influence an
//     earlier decision. A paired request (ProcessCnt == ScheduleCnt) reports
//     its newest round as N-1, so its N-2 rates sit in PrevAcceptRates; an
//     unpaired request (ScheduleCnt == ProcessCnt+1) still carries N-2 in
//     AcceptRates. Both cases feed the total-work queue with N-2 rates.
//   - Distribution of the chosen extra slots uses the newest available
//     per-position rates (N-1 when paired, N-2 otherwise), sorted descending
//     with ties broken by request index.
//   - Requests missing rate evidence pad conservatively: 1.0 per position when
//     drafts are known to exist, 0.0 otherwise.
//   - Each candidate total batch size maps to the smallest profiled CUDA graph
//     that holds it and scores expected accepted tokens over
//     baseline + learnedForwardSlope*graphSize + learnedSamplerTime.
//
// Scope stops at allocation: no draft generation, target forward, or rollback
// lives here; those remain the hardware-gated P1 successor. This complements,
// and does not replace, the existing spec-depth confidence gate in
// internal/polymodel.

package model

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

var (
	// ErrV41DSparkVarlenNoProfile marks scheduling attempted without any
	// recorded decode performance observations to bound the learned terms.
	ErrV41DSparkVarlenNoProfile = errors.New("model: dspark varlen policy has no decode performance profile")
	// ErrV41DSparkVarlenInvalidRequest marks a request violating the
	// schedule/process counter invariant or its rate lengths.
	ErrV41DSparkVarlenInvalidRequest = errors.New("model: dspark varlen request violates the counter or rate-length invariant")
	// ErrV41DSparkVarlenBatchExceedsProfile marks a batch no profiled graph
	// can hold.
	ErrV41DSparkVarlenBatchExceedsProfile = errors.New("model: dspark varlen batch exceeds every profiled graph size")
)

// V41DSparkPerfPoint is one profiled decode-step cost sample: the largest total
// verification batch the named CUDA graph size holds, and its measured forward
// time in milliseconds before any learned correction.
type V41DSparkPerfPoint struct {
	GraphSize  int
	BaselineMS float64
}

// V41DSparkVarlenRequest is the per-request evidence the varlen policy consumes
// for one scheduled decode step. Counters mirror the upstream pairing
// invariant ProcessCnt <= ScheduleCnt <= ProcessCnt+1.
type V41DSparkVarlenRequest struct {
	// DraftCount is the number of draft tokens the request produced this step.
	DraftCount int
	// HasDraftInfo is independent knowledge that the request carries drafts
	// even when its counters do not yet prove it.
	HasDraftInfo bool
	// ScheduleCnt counts engine schedulings; ProcessCnt counts forwards that
	// produced an accept-rate record. Equal means the newest record is the
	// current round (N-1); ScheduleCnt == ProcessCnt+1 means the newest record
	// is one round old (N-2).
	ScheduleCnt int
	ProcessCnt  int
	// AcceptRates holds the newest completed round's per-position acceptance
	// probabilities, one entry per draft position. Nil means unknown.
	AcceptRates []float64
	// PrevAcceptRates holds the round immediately older than AcceptRates.
	PrevAcceptRates []float64
}

const (
	v41DSparkSamplerWindowSize = 9
	v41DSparkSamplerMinSamples = 5
	v41DSparkSlopeWindowSize   = 5000
	v41DSparkSlopeMinPerGraph  = 10
)

// V41DSparkSamplerTimeEstimator keeps a sliding window of observed sampler
// times and reports their median once enough samples exist.
type V41DSparkSamplerTimeEstimator struct {
	samples [v41DSparkSamplerWindowSize]float64
	next    int
	size    int
}

// Update folds one observed sampler time into the window.
func (e *V41DSparkSamplerTimeEstimator) Update(valueMS float64) {
	e.samples[e.next] = valueMS
	e.next = (e.next + 1) % len(e.samples)
	if e.size < len(e.samples) {
		e.size++
	}
}

// Get reports the window median and whether it rests on enough samples.
func (e *V41DSparkSamplerTimeEstimator) Get() (float64, bool) {
	if e.size < v41DSparkSamplerMinSamples {
		return 0, false
	}
	return v41DSparkMedian(e.samples[:e.size]), true
}

// V41DSparkForwardSlopeEstimator learns the per-token forward-time slope above
// the profiled baseline. It keeps the most recent observations grouped by
// graph size, admits a group's median once the group holds enough samples, and
// reports the median of the admitted group medians. Recomputing from the
// retained window on read is observationally identical to the upstream
// incremental multiset maintenance and stays deterministic without ordering
// state.
type V41DSparkForwardSlopeEstimator struct {
	window []v41DSparkSlopeSample
}

type v41DSparkSlopeSample struct {
	graphSize int
	residual  float64 // (forward_ms - baseline_ms) / graph_size
}

// Update records one observed forward-step residual for its graph size.
func (e *V41DSparkForwardSlopeEstimator) Update(graphSize int, residualMS float64) {
	e.window = append(e.window, v41DSparkSlopeSample{graphSize: graphSize, residual: residualMS})
	if overflow := len(e.window) - v41DSparkSlopeWindowSize; overflow > 0 {
		e.window = e.window[overflow:]
	}
}

// Get reports the learned slope in ms per token, zero until some graph size
// holds enough samples.
func (e *V41DSparkForwardSlopeEstimator) Get() float64 {
	if len(e.window) == 0 {
		return 0
	}
	byGraph := make(map[int][]float64, len(e.window))
	for _, s := range e.window {
		byGraph[s.graphSize] = append(byGraph[s.graphSize], s.residual)
	}
	medians := make([]float64, 0, len(byGraph))
	for _, values := range byGraph {
		if len(values) >= v41DSparkSlopeMinPerGraph {
			medians = append(medians, v41DSparkMedian(values))
		}
	}
	if len(medians) == 0 {
		return 0
	}
	return v41DSparkMedian(medians)
}

func v41DSparkMedian(values []float64) float64 {
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	mid := len(sorted) / 2
	if len(sorted)%2 == 0 {
		return (sorted[mid-1] + sorted[mid]) * 0.5
	}
	return sorted[mid]
}

// V41DSparkVarlenPolicy allocates variable-length verification slots across
// scheduled requests from age-separated confidence queues plus learned runtime
// terms. All inputs arrive as arguments or recorded observations; the policy
// holds no clock and no randomness.
type V41DSparkVarlenPolicy struct {
	curve           []V41DSparkPerfPoint
	slope           V41DSparkForwardSlopeEstimator
	samplerBySeqs   map[int]*V41DSparkSamplerTimeEstimator
	samplerFallback *V41DSparkSamplerTimeEstimator
}

// NewV41DSparkVarlenPolicy builds a policy over a profiled decode performance
// curve whose graph sizes are strictly increasing. An empty curve is admitted:
// scheduling then verifies every draft position.
func NewV41DSparkVarlenPolicy(curve []V41DSparkPerfPoint) (*V41DSparkVarlenPolicy, error) {
	p := &V41DSparkVarlenPolicy{
		samplerBySeqs:   map[int]*V41DSparkSamplerTimeEstimator{},
		samplerFallback: &V41DSparkSamplerTimeEstimator{},
	}
	if err := p.SetDecodePerfCurve(curve); err != nil {
		return nil, err
	}
	return p, nil
}

// SetDecodePerfCurve replaces the profiled curve. Graph sizes must be positive
// and strictly increasing; baselines must be finite.
func (p *V41DSparkVarlenPolicy) SetDecodePerfCurve(curve []V41DSparkPerfPoint) error {
	clean := make([]V41DSparkPerfPoint, len(curve))
	copy(clean, curve)
	for i, pt := range clean {
		if pt.GraphSize <= 0 || math.IsNaN(pt.BaselineMS) || math.IsInf(pt.BaselineMS, 0) {
			return fmt.Errorf("%w: point %d has graph_size=%d baseline_ms=%v", ErrV41DSparkVarlenInvalidRequest, i, pt.GraphSize, pt.BaselineMS)
		}
		if i > 0 && clean[i].GraphSize <= clean[i-1].GraphSize {
			return fmt.Errorf("%w: point %d graph_size=%d not strictly increasing", ErrV41DSparkVarlenInvalidRequest, i, clean[i].GraphSize)
		}
	}
	p.curve = clean
	return nil
}

// UpdateRuntime folds one observed decode step into the learned terms. The
// forward residual is normalized per token against the baseline of the
// smallest profiled graph that holds graphSize; the sampler time updates both
// the num-seqs-specific estimator and the global fallback.
func (p *V41DSparkVarlenPolicy) UpdateRuntime(numSeqs, graphSize int, forwardMS, samplerMS float64) error {
	if numSeqs < 0 || graphSize <= 0 || math.IsNaN(forwardMS) || math.IsInf(forwardMS, 0) || math.IsNaN(samplerMS) || math.IsInf(samplerMS, 0) {
		return fmt.Errorf("%w: num_seqs=%d graph_size=%d forward_ms=%v sampler_ms=%v", ErrV41DSparkVarlenInvalidRequest, numSeqs, graphSize, forwardMS, samplerMS)
	}
	idx := sort.Search(len(p.curve), func(i int) bool { return p.curve[i].GraphSize >= graphSize })
	if idx == len(p.curve) {
		return fmt.Errorf("%w: graph_size=%d beyond profile", ErrV41DSparkVarlenBatchExceedsProfile, graphSize)
	}
	baseline := p.curve[idx].BaselineMS
	residual := (forwardMS - baseline) / float64(graphSize)
	if math.IsNaN(residual) || math.IsInf(residual, 0) {
		return fmt.Errorf("%w: non-finite derived forward residual", ErrV41DSparkVarlenInvalidRequest)
	}
	p.slope.Update(graphSize, residual)
	est := p.samplerBySeqs[numSeqs]
	if est == nil {
		est = &V41DSparkSamplerTimeEstimator{}
		p.samplerBySeqs[numSeqs] = est
	}
	est.Update(samplerMS)
	p.samplerFallback.Update(samplerMS)
	return nil
}

// Schedule assigns each request its verification length: one golden token plus
// the draft positions the chosen total batch verifies. It returns one entry per
// request in input order.
func (p *V41DSparkVarlenPolicy) Schedule(reqs []V41DSparkVarlenRequest) ([]int, error) {
	numSeqs := len(reqs)
	totalDrafts := 0
	for i, r := range reqs {
		if err := v41DSparkValidateRequest(r); err != nil {
			return nil, fmt.Errorf("request %d: %w", i, err)
		}
		// Every sequence also needs one golden token. Bound the combined
		// count before addition, allocation, or an individual 1+DraftCount.
		if r.DraftCount > int(^uint(0)>>1)-numSeqs-totalDrafts {
			return nil, fmt.Errorf("request %d: %w: total verification tokens overflow int", i, ErrV41DSparkVarlenInvalidRequest)
		}
		totalDrafts += r.DraftCount
	}
	if numSeqs == 0 {
		return nil, nil
	}
	if len(p.curve) == 0 {
		seqlens := make([]int, numSeqs)
		for i, r := range reqs {
			seqlens[i] = 1 + r.DraftCount
		}
		return seqlens, nil
	}

	type slot struct {
		rate float64
		seq  int
	}
	push := func(queue *[]slot, rates []float64, nd, seq int, hasDraft bool) {
		if rates == nil {
			pad := 0.0
			if hasDraft {
				pad = 1.0
			}
			for j := 0; j < nd; j++ {
				*queue = append(*queue, slot{rate: pad, seq: seq})
			}
			return
		}
		for j := 0; j < nd; j++ {
			*queue = append(*queue, slot{rate: rates[j], seq: seq})
		}
	}

	bszQueue := make([]slot, 0, totalDrafts)
	seqsQueue := make([]slot, 0, totalDrafts)
	for i, r := range reqs {
		paired := r.ProcessCnt == r.ScheduleCnt
		hasDraft := r.HasDraftInfo || !paired
		// Total work always reads the N-2 round: prev when paired, newest otherwise.
		bszRates := r.PrevAcceptRates
		if !paired {
			bszRates = r.AcceptRates
		}
		push(&bszQueue, bszRates, r.DraftCount, i, hasDraft)
		// Distribution always reads the newest available round.
		push(&seqsQueue, r.AcceptRates, r.DraftCount, i, hasDraft)
	}
	descRate := func(q []slot) {
		sort.SliceStable(q, func(a, b int) bool {
			if q[a].rate != q[b].rate {
				return q[a].rate > q[b].rate
			}
			return q[a].seq < q[b].seq
		})
	}
	descRate(bszQueue)
	descRate(seqsQueue)

	samplerMS := 0.0
	used := false
	if est, ok := p.samplerBySeqs[numSeqs]; ok {
		if v, ready := est.Get(); ready {
			samplerMS, used = v, true
		}
	}
	if !used {
		if v, ready := p.samplerFallback.Get(); ready {
			samplerMS = v
		}
	}
	slopeMS := p.slope.Get()
	if slopeMS < 0 {
		slopeMS = 0
	}

	bestThroughput := 0.0
	bestBsz := 0
	curExpected := float64(numSeqs)
	curBsz := numSeqs
	curIdx := 0
	updateBest := func() {
		for curIdx < len(p.curve) && curBsz > p.curve[curIdx].GraphSize {
			curIdx++
		}
		if curIdx == len(p.curve) {
			return
		}
		pt := p.curve[curIdx]
		timeMS := pt.BaselineMS + slopeMS*float64(pt.GraphSize) + samplerMS
		throughput := curExpected / timeMS
		if throughput > bestThroughput {
			bestThroughput = throughput
			bestBsz = curBsz
		}
	}
	updateBest() // all golden tokens
	for _, s := range bszQueue {
		curExpected += s.rate
		curBsz++
		updateBest()
	}
	if bestBsz < numSeqs {
		return nil, fmt.Errorf("%w: num_seqs=%d", ErrV41DSparkVarlenBatchExceedsProfile, numSeqs)
	}

	seqlens := make([]int, numSeqs)
	for i := range seqlens {
		seqlens[i] = 1
	}
	for i := 0; i < bestBsz-numSeqs; i++ {
		seqlens[seqsQueue[i].seq]++
	}
	return seqlens, nil
}

func v41DSparkValidateRequest(r V41DSparkVarlenRequest) error {
	if r.DraftCount < 0 || r.ScheduleCnt < 0 || r.ProcessCnt < 0 ||
		r.ProcessCnt > r.ScheduleCnt || r.ScheduleCnt-r.ProcessCnt > 1 {
		return fmt.Errorf("%w: draft=%d schedule=%d process=%d", ErrV41DSparkVarlenInvalidRequest, r.DraftCount, r.ScheduleCnt, r.ProcessCnt)
	}
	checkRates := func(name string, rates []float64) error {
		if rates == nil {
			return nil
		}
		if len(rates) != r.DraftCount {
			return fmt.Errorf("%w: %s length %d != draft count %d", ErrV41DSparkVarlenInvalidRequest, name, len(rates), r.DraftCount)
		}
		for _, v := range rates {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return fmt.Errorf("%w: %s holds a non-finite rate", ErrV41DSparkVarlenInvalidRequest, name)
			}
		}
		return nil
	}
	if err := checkRates("accept_rates", r.AcceptRates); err != nil {
		return err
	}
	return checkRates("prev_accept_rates", r.PrevAcceptRates)
}
