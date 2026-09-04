package servingsim

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// EventQueue is a min-heap priority queue of SimEvents ordered by TimeMS and Seq.
type EventQueue []*SimEvent

func (eq EventQueue) Len() int { return len(eq) }

func (eq EventQueue) Less(i, j int) bool {
	if eq[i].TimeMS == eq[j].TimeMS {
		return eq[i].Seq < eq[j].Seq
	}
	return eq[i].TimeMS < eq[j].TimeMS
}

func (eq EventQueue) Swap(i, j int) {
	eq[i], eq[j] = eq[j], eq[i]
	eq[i].index = i
	eq[j].index = j
}

func (eq *EventQueue) Push(x any) {
	n := len(*eq)
	item := x.(*SimEvent)
	item.index = n
	*eq = append(*eq, item)
}

func (eq *EventQueue) Pop() any {
	old := *eq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	item.index = -1
	*eq = old[0 : n-1]
	return item
}

type prefillTask struct {
	req    *RequestState
	tokens int
}

type activeStep struct {
	startTimeMS   float64
	durationMS    float64
	prefillChunks []prefillTask
	decodeReqs    []*RequestState
}

// Engine implements the discrete-event continuous-batching simulation kernel.
type Engine struct {
	config SimulatorConfig
	hw     HardwareLatencyTable

	currentTimeMS float64
	startMS       float64
	eventQueue    EventQueue
	seqCounter    int64
	rng           *rand.Rand

	// Queues and execution pools
	waitingQueue  []*RequestState
	activePrefill []*RequestState
	activeDecode  []*RequestState

	// KV block accounting
	usedKVBlocks        int
	peakKVBlocksUsed    int
	kvBlockTimeIntegral float64
	lastKVUpdateTimeMS  float64

	// Hardware step tracking
	stepInFlight bool
	currentStep  *activeStep

	// Output & stats
	completedRequests []*RequestState
	totalSpecProposed int
	totalSpecAccepted int

	traceCollector *TraceCollector
}

// NewServingSimulator initializes a simulation engine with validated configuration and latency model.
func NewServingSimulator(config SimulatorConfig, hw HardwareLatencyTable) (*Engine, error) {
	if config.KVBlockTokens <= 0 {
		config.KVBlockTokens = 16
	}
	if config.TotalKVBlocks <= 0 {
		config.TotalKVBlocks = 4096
	}
	if config.MaxBatchSize <= 0 {
		config.MaxBatchSize = 32
	}
	if config.AcceptanceRate < 0 || config.AcceptanceRate > 1.0 {
		return nil, fmt.Errorf("servingsim: AcceptanceRate must be in [0.0, 1.0], got %f", config.AcceptanceRate)
	}
	if config.SpeculativeHorizon < 0 {
		return nil, fmt.Errorf("servingsim: SpeculativeHorizon cannot be negative, got %d", config.SpeculativeHorizon)
	}

	seed := config.Seed
	if seed == 0 {
		seed = 42
	}

	var tc *TraceCollector
	if config.EnableTrace {
		tc = NewTraceCollector()
	}

	return &Engine{
		config:         config,
		hw:             hw,
		rng:            rand.New(rand.NewSource(seed)),
		eventQueue:     make(EventQueue, 0, 128),
		waitingQueue:   make([]*RequestState, 0, 64),
		activePrefill:  make([]*RequestState, 0, 32),
		activeDecode:   make([]*RequestState, 0, 32),
		traceCollector: tc,
	}, nil
}

// Run executes the discrete-event simulation over the given requests workload.
func (e *Engine) Run(requests []RequestState) (*SimulationMetrics, error) {
	if len(requests) == 0 {
		return &SimulationMetrics{}, nil
	}

	heap.Init(&e.eventQueue)
	e.seqCounter = 0
	e.currentTimeMS = 0
	e.startMS = 0
	e.usedKVBlocks = 0
	e.peakKVBlocksUsed = 0
	e.kvBlockTimeIntegral = 0
	e.lastKVUpdateTimeMS = 0
	e.stepInFlight = false
	e.currentStep = nil
	e.completedRequests = make([]*RequestState, 0, len(requests))
	e.totalSpecProposed = 0
	e.totalSpecAccepted = 0

	// Validate and enqueue arrival events
	sortedRequests := make([]*RequestState, len(requests))
	for i := range requests {
		req := requests[i]
		if req.PromptTokens <= 0 {
			return nil, fmt.Errorf("servingsim: request %q has non-positive prompt tokens %d", req.ID, req.PromptTokens)
		}
		if req.OutputTarget <= 0 {
			return nil, fmt.Errorf("servingsim: request %q has non-positive output target %d", req.ID, req.OutputTarget)
		}
		if req.ArrivalTimeMS < 0 {
			return nil, fmt.Errorf("servingsim: request %q has negative arrival time %f", req.ID, req.ArrivalTimeMS)
		}
		reqCopy := req
		reqCopy.PrefillComputed = 0
		reqCopy.TokensGenerated = 0
		reqCopy.FirstTokenTimeMS = 0
		reqCopy.CompletionTimeMS = 0
		reqCopy.SpecAccepted = 0
		reqCopy.SpecProposed = 0
		reqCopy.QueueTimeMS = 0
		reqCopy.AllocatedKVBlocks = 0
		reqCopy.Admitted = false
		sortedRequests[i] = &reqCopy
	}

	sort.Slice(sortedRequests, func(i, j int) bool {
		return sortedRequests[i].ArrivalTimeMS < sortedRequests[j].ArrivalTimeMS
	})

	e.startMS = sortedRequests[0].ArrivalTimeMS
	e.currentTimeMS = e.startMS
	e.lastKVUpdateTimeMS = e.startMS

	for _, req := range sortedRequests {
		e.pushEvent(&SimEvent{
			TimeMS:  req.ArrivalTimeMS,
			Type:    EventRequestArrival,
			Request: req,
		})
	}

	// Main discrete-event loop
	for e.eventQueue.Len() > 0 {
		event := heap.Pop(&e.eventQueue).(*SimEvent)
		e.advanceTimeTo(event.TimeMS)

		switch event.Type {
		case EventRequestArrival:
			e.handleRequestArrival(event.Request)
		case EventStepComplete:
			e.handleStepComplete()
		}

		// If there are simultaneous events at the exact same virtual time (e.g. concurrent arrivals),
		// process them before scheduling the next batch step so they can be grouped together.
		if e.eventQueue.Len() > 0 && e.eventQueue[0].TimeMS <= e.currentTimeMS {
			continue
		}

		if !e.stepInFlight {
			e.tryScheduleStep()
		}
	}

	return e.finalizeMetrics(), nil
}

func (e *Engine) pushEvent(ev *SimEvent) {
	e.seqCounter++
	ev.Seq = e.seqCounter
	heap.Push(&e.eventQueue, ev)
}

func (e *Engine) advanceTimeTo(targetTimeMS float64) {
	if targetTimeMS <= e.currentTimeMS {
		return
	}
	delta := targetTimeMS - e.currentTimeMS
	e.kvBlockTimeIntegral += float64(e.usedKVBlocks) * delta
	e.currentTimeMS = targetTimeMS
	e.lastKVUpdateTimeMS = targetTimeMS
}

func (e *Engine) handleRequestArrival(req *RequestState) {
	e.waitingQueue = append(e.waitingQueue, req)
	if e.traceCollector != nil {
		e.traceCollector.RecordInstant("arrival:"+req.ID, "request", e.currentTimeMS, 1, 2, map[string]any{
			"id":            req.ID,
			"prompt_tokens": req.PromptTokens,
			"output_target": req.OutputTarget,
		})
	}
}

func (e *Engine) handleStepComplete() {
	e.stepInFlight = false
	step := e.currentStep
	e.currentStep = nil
	if step == nil {
		return
	}

	// 1. Process completed prefill chunks
	for _, chunk := range step.prefillChunks {
		req := chunk.req
		req.PrefillComputed += chunk.tokens
		if req.PrefillComputed >= req.PromptTokens {
			req.PrefillComputed = req.PromptTokens
			// Prefill completion yields the first generated output token
			req.TokensGenerated = 1
			req.FirstTokenTimeMS = e.currentTimeMS

			e.removeActivePrefill(req)

			if req.OutputTarget <= 1 {
				// Single-token generation completed right after prefill
				req.CompletionTimeMS = e.currentTimeMS
				e.finishRequest(req)
			} else {
				// Move to active decode pool
				e.activeDecode = append(e.activeDecode, req)
				e.ensureBlocks(req, req.PromptTokens+req.TokensGenerated)
			}
		}
	}

	// 2. Process decode requests
	for _, req := range step.decodeReqs {
		needed := req.OutputTarget - req.TokensGenerated
		if needed <= 0 {
			e.removeActiveDecode(req)
			req.CompletionTimeMS = e.currentTimeMS
			e.finishRequest(req)
			continue
		}

		if e.config.SpeculativeHorizon > 0 && needed > 1 {
			draftK := e.config.SpeculativeHorizon
			if draftK > needed {
				draftK = needed
			}
			accepted := e.sampleSpeculativeAcceptance(draftK)
			req.SpecProposed += draftK
			req.SpecAccepted += accepted
			e.totalSpecProposed += draftK
			e.totalSpecAccepted += accepted

			tokensProduced := 1
			if accepted < draftK {
				tokensProduced = accepted + 1
			} else {
				if accepted < needed {
					tokensProduced = accepted + 1
				} else {
					tokensProduced = accepted
				}
			}

			req.TokensGenerated += tokensProduced
		} else {
			// Non-speculative autoregressive decode
			req.TokensGenerated++
		}

		if req.TokensGenerated >= req.OutputTarget {
			req.TokensGenerated = req.OutputTarget
			req.CompletionTimeMS = e.currentTimeMS
			e.removeActiveDecode(req)
			e.finishRequest(req)
		} else {
			e.ensureBlocks(req, req.PromptTokens+req.TokensGenerated)
		}
	}

	if e.traceCollector != nil {
		e.traceCollector.RecordCounter("KVBlocks", "memory", e.currentTimeMS, 1, 1, map[string]any{
			"used": e.usedKVBlocks,
			"free": e.config.TotalKVBlocks - e.usedKVBlocks,
		})
	}
}

func (e *Engine) tryScheduleStep() {
	if e.stepInFlight {
		return
	}

	freeBlocks := e.config.TotalKVBlocks - e.usedKVBlocks
	if freeBlocks < 0 {
		freeBlocks = 0
	}

	maxBatch := e.config.MaxBatchSize
	tokenBudget := e.config.MaxTokensPerStep
	useTokenBudget := tokenBudget > 0

	// Phase 1: Schedule active decodes (highest priority in continuous batching)
	scheduledDecodes, freeBlocks := e.scheduleActiveDecodes(freeBlocks, maxBatch)

	freeBatchCapacity := maxBatch - len(scheduledDecodes)
	remainingTokens := tokenBudget
	if useTokenBudget {
		// Decodes consume compute budget (1 token or draft length)
		decodeTokenConsumption := len(scheduledDecodes) * (1 + e.config.SpeculativeHorizon)
		remainingTokens -= decodeTokenConsumption
		if remainingTokens < 0 {
			remainingTokens = 0
		}
	}

	// Phase 2: Schedule in-progress chunked prefills
	scheduledPrefills, freeBlocks, freeBatchCapacity, remainingTokens := e.scheduleActivePrefills(
		freeBlocks, freeBatchCapacity, remainingTokens, useTokenBudget,
	)

	// Phase 3: Admit waiting requests from arrival queue
	admittedPrefills := e.admitWaitingRequests(freeBlocks, freeBatchCapacity, remainingTokens, useTokenBudget)
	scheduledPrefills = append(scheduledPrefills, admittedPrefills...)

	// If no work scheduled, remain idle until next event
	if len(scheduledDecodes) == 0 && len(scheduledPrefills) == 0 {
		return
	}

	e.executeScheduledStep(scheduledDecodes, scheduledPrefills)
}

func (e *Engine) tryAllocateDecodes(freeBlocks, maxBatch int) ([]*RequestState, int) {
	var scheduled []*RequestState
	for _, req := range e.activeDecode {
		if len(scheduled) >= maxBatch {
			break
		}
		draftK := e.config.SpeculativeHorizon
		nextTokens := req.PromptTokens + req.TokensGenerated + (draftK + 1)
		targetBlocks := e.blocksForTokens(nextTokens)
		additionalBlocks := targetBlocks - req.AllocatedKVBlocks
		if additionalBlocks < 0 {
			additionalBlocks = 0
		}

		if freeBlocks >= additionalBlocks {
			freeBlocks -= additionalBlocks
			if additionalBlocks > 0 {
				e.allocateBlocks(req, targetBlocks)
			}
			scheduled = append(scheduled, req)
		}
	}
	return scheduled, freeBlocks
}

func (e *Engine) scheduleActiveDecodes(freeBlocks, maxBatch int) ([]*RequestState, int) {
	var scheduled []*RequestState
	scheduled, freeBlocks = e.tryAllocateDecodes(freeBlocks, maxBatch)

	// If no decode could get a block due to memory exhaustion, preempt the newest decode
	if len(scheduled) == 0 && len(e.activeDecode) > 1 {
		preemptIdx := len(e.activeDecode) - 1
		preempted := e.activeDecode[preemptIdx]
		e.activeDecode = e.activeDecode[:preemptIdx]
		e.freeRequestBlocks(preempted)
		preempted.TokensGenerated = 0
		preempted.PrefillComputed = 0
		preempted.Admitted = false
		e.waitingQueue = append([]*RequestState{preempted}, e.waitingQueue...)
		freeBlocks = e.config.TotalKVBlocks - e.usedKVBlocks

		scheduled, freeBlocks = e.tryAllocateDecodes(freeBlocks, maxBatch)
	}
	return scheduled, freeBlocks
}

func (e *Engine) scheduleActivePrefills(freeBlocks, freeBatchCapacity, remainingTokens int, useTokenBudget bool) ([]prefillTask, int, int, int) {
	var scheduledPrefills []prefillTask
	for _, req := range e.activePrefill {
		if freeBatchCapacity <= 0 || (useTokenBudget && remainingTokens <= 0) {
			break
		}

		needed := req.PromptTokens - req.PrefillComputed
		chunk := needed
		if useTokenBudget && chunk > remainingTokens {
			chunk = remainingTokens
		}
		if chunk <= 0 {
			continue
		}

		targetBlocks := e.blocksForTokens(req.PrefillComputed + chunk)
		additionalBlocks := targetBlocks - req.AllocatedKVBlocks
		if additionalBlocks < 0 {
			additionalBlocks = 0
		}

		if freeBlocks >= additionalBlocks {
			freeBlocks -= additionalBlocks
			e.allocateBlocks(req, targetBlocks)
			scheduledPrefills = append(scheduledPrefills, prefillTask{req: req, tokens: chunk})
			freeBatchCapacity--
			if useTokenBudget {
				remainingTokens -= chunk
			}
		}
	}
	return scheduledPrefills, freeBlocks, freeBatchCapacity, remainingTokens
}

func (e *Engine) admitWaitingRequests(freeBlocks, freeBatchCapacity, remainingTokens int, useTokenBudget bool) []prefillTask {
	runningCommitted := 0
	for _, r := range e.activeDecode {
		runningCommitted += e.blocksForTokens(r.PromptTokens + r.OutputTarget)
	}
	for _, r := range e.activePrefill {
		runningCommitted += e.blocksForTokens(r.PromptTokens + r.OutputTarget)
	}

	var scheduledPrefills []prefillTask
	var admittedIndices []int
	for i, req := range e.waitingQueue {
		if freeBatchCapacity <= 0 || (useTokenBudget && remainingTokens <= 0) {
			break
		}

		// Admission control: do not admit if it risks starvating running requests
		reqTotalNeed := e.blocksForTokens(req.PromptTokens + req.OutputTarget)
		if len(e.activeDecode) > 0 || len(e.activePrefill) > 0 {
			if runningCommitted+reqTotalNeed > e.config.TotalKVBlocks {
				break
			}
		}

		needed := req.PromptTokens
		chunk := needed
		if useTokenBudget && chunk > remainingTokens {
			chunk = remainingTokens
		}
		if chunk <= 0 {
			break
		}

		neededBlocks := e.blocksForTokens(chunk)
		if freeBlocks >= neededBlocks {
			freeBlocks -= neededBlocks
			req.Admitted = true
			req.QueueTimeMS = e.currentTimeMS - req.ArrivalTimeMS
			e.allocateBlocks(req, neededBlocks)
			e.activePrefill = append(e.activePrefill, req)
			scheduledPrefills = append(scheduledPrefills, prefillTask{req: req, tokens: chunk})
			admittedIndices = append(admittedIndices, i)
			runningCommitted += reqTotalNeed
			freeBatchCapacity--
			if useTokenBudget {
				remainingTokens -= chunk
			}
		}
	}

	// Remove admitted requests from waiting queue (in reverse order to preserve indices)
	for i := len(admittedIndices) - 1; i >= 0; i-- {
		idx := admittedIndices[i]
		e.waitingQueue = append(e.waitingQueue[:idx], e.waitingQueue[idx+1:]...)
	}
	return scheduledPrefills
}

func (e *Engine) executeScheduledStep(scheduledDecodes []*RequestState, scheduledPrefills []prefillTask) {
	totalPrefillTokens := 0
	for _, p := range scheduledPrefills {
		totalPrefillTokens += p.tokens
	}

	decodeBatch := len(scheduledDecodes)
	prefillBatch := len(scheduledPrefills)
	draftK := 0
	if decodeBatch > 0 {
		draftK = e.config.SpeculativeHorizon
	}

	stepLatencyMS := e.hw.StepLatency(totalPrefillTokens, prefillBatch, decodeBatch, draftK)
	if stepLatencyMS <= 0 {
		stepLatencyMS = 0.001
	}

	e.stepInFlight = true
	e.currentStep = &activeStep{
		startTimeMS:   e.currentTimeMS,
		durationMS:    stepLatencyMS,
		prefillChunks: scheduledPrefills,
		decodeReqs:    scheduledDecodes,
	}

	if e.traceCollector != nil {
		stepName := "step_decode"
		if prefillBatch > 0 && decodeBatch > 0 {
			stepName = "step_mixed"
		} else if prefillBatch > 0 {
			stepName = "step_prefill"
		}
		e.traceCollector.RecordStep(stepName, "gpu_step", e.currentTimeMS, stepLatencyMS, 1, 1, map[string]any{
			"prefill_tokens": totalPrefillTokens,
			"prefill_batch":  prefillBatch,
			"decode_batch":   decodeBatch,
			"draft_k":        draftK,
		})
	}

	e.pushEvent(&SimEvent{
		TimeMS: e.currentTimeMS + stepLatencyMS,
		Type:   EventStepComplete,
	})
}

func (e *Engine) blocksForTokens(tokens int) int {
	if tokens <= 0 {
		return 0
	}
	return (tokens + e.config.KVBlockTokens - 1) / e.config.KVBlockTokens
}

func (e *Engine) allocateBlocks(req *RequestState, targetBlocks int) {
	diff := targetBlocks - req.AllocatedKVBlocks
	if diff != 0 {
		e.usedKVBlocks += diff
		req.AllocatedKVBlocks = targetBlocks
		if e.usedKVBlocks > e.peakKVBlocksUsed {
			e.peakKVBlocksUsed = e.usedKVBlocks
		}
	}
}

func (e *Engine) ensureBlocks(req *RequestState, tokens int) {
	target := e.blocksForTokens(tokens)
	if target > req.AllocatedKVBlocks {
		e.allocateBlocks(req, target)
	}
}

func (e *Engine) freeRequestBlocks(req *RequestState) {
	if req.AllocatedKVBlocks > 0 {
		e.usedKVBlocks -= req.AllocatedKVBlocks
		req.AllocatedKVBlocks = 0
	}
}

func (e *Engine) finishRequest(req *RequestState) {
	e.freeRequestBlocks(req)
	e.completedRequests = append(e.completedRequests, req)

	if e.traceCollector != nil {
		admitTime := req.ArrivalTimeMS + req.QueueTimeMS
		execDuration := req.CompletionTimeMS - admitTime
		e.traceCollector.RecordStep("req:"+req.ID, "request", req.ArrivalTimeMS, req.QueueTimeMS, 1, 2, map[string]any{
			"phase":         "queue",
			"id":            req.ID,
			"prompt_tokens": req.PromptTokens,
		})
		e.traceCollector.RecordStep("exec:"+req.ID, "request", admitTime, execDuration, 1, 2, map[string]any{
			"phase":            "execution",
			"id":               req.ID,
			"tokens_generated": req.TokensGenerated,
			"spec_accepted":    req.SpecAccepted,
		})
	}
}

func (e *Engine) removeActivePrefill(req *RequestState) {
	for i, r := range e.activePrefill {
		if r == req {
			e.activePrefill = append(e.activePrefill[:i], e.activePrefill[i+1:]...)
			return
		}
	}
}

func (e *Engine) removeActiveDecode(req *RequestState) {
	for i, r := range e.activeDecode {
		if r == req {
			e.activeDecode = append(e.activeDecode[:i], e.activeDecode[i+1:]...)
			return
		}
	}
}

func (e *Engine) sampleSpeculativeAcceptance(draftK int) int {
	if draftK <= 0 {
		return 0
	}

	mode := e.config.SpeculativeMode
	if mode == "" {
		if e.config.Deterministic {
			mode = SpeculativeModeDeterministic
		} else {
			mode = SpeculativeModePrefix
		}
	}

	alpha := e.config.AcceptanceRate
	if alpha < 0 {
		alpha = 0
	} else if alpha > 1.0 {
		alpha = 1.0
	}

	switch mode {
	case SpeculativeModeDeterministic:
		accepted := int(math.Round(float64(draftK) * alpha))
		if accepted > draftK {
			accepted = draftK
		}
		if accepted < 0 {
			accepted = 0
		}
		return accepted

	case SpeculativeModeBinomial:
		accepted := 0
		for i := 0; i < draftK; i++ {
			p := alpha
			if e.config.PositionalAlphaFunc != nil {
				p = e.config.PositionalAlphaFunc(i)
			}
			if e.rng.Float64() < p {
				accepted++
			}
		}
		return accepted

	case SpeculativeModePoisson:
		lambda := float64(draftK) * alpha
		if lambda <= 0 {
			return 0
		}
		L := math.Exp(-lambda)
		k := 0
		p := 1.0
		for p > L {
			k++
			p *= e.rng.Float64()
		}
		accepted := k - 1
		if accepted > draftK {
			accepted = draftK
		}
		if accepted < 0 {
			accepted = 0
		}
		return accepted

	case SpeculativeModePrefix:
		fallthrough
	default:
		accepted := 0
		for i := 0; i < draftK; i++ {
			p := alpha
			if e.config.PositionalAlphaFunc != nil {
				p = e.config.PositionalAlphaFunc(i)
			}
			if e.rng.Float64() < p {
				accepted++
			} else {
				break
			}
		}
		return accepted
	}
}

func (e *Engine) finalizeMetrics() *SimulationMetrics {
	totalReqs := len(e.completedRequests)
	simDurationMS := e.currentTimeMS - e.startMS
	if simDurationMS <= 0 {
		simDurationMS = 0.001
	}

	var reqThroughput float64
	var tokenThroughput float64
	totalOutputTokens := 0

	ttftList := make([]float64, 0, totalReqs)
	tpotList := make([]float64, 0, totalReqs)
	completedCopies := make([]RequestState, totalReqs)

	for i, req := range e.completedRequests {
		completedCopies[i] = *req
		totalOutputTokens += req.TokensGenerated

		ttft := req.FirstTokenTimeMS - req.ArrivalTimeMS
		if ttft >= 0 {
			ttftList = append(ttftList, ttft)
		}

		if req.TokensGenerated > 1 {
			decodeDuration := req.CompletionTimeMS - req.FirstTokenTimeMS
			tpot := decodeDuration / float64(req.TokensGenerated-1)
			if tpot >= 0 {
				tpotList = append(tpotList, tpot)
			}
		}
	}

	durationSec := simDurationMS / 1000.0
	if durationSec > 0 {
		reqThroughput = float64(totalReqs) / durationSec
		tokenThroughput = float64(totalOutputTokens) / durationSec
	}

	var kvUtil float64
	if e.config.TotalKVBlocks > 0 && simDurationMS > 0 {
		kvUtil = e.kvBlockTimeIntegral / (float64(e.config.TotalKVBlocks) * simDurationMS)
		if kvUtil > 1.0 {
			kvUtil = 1.0
		}
	}

	var specYield, specWaste float64
	if e.totalSpecProposed > 0 {
		specYield = float64(e.totalSpecAccepted) / float64(e.totalSpecProposed)
		specWaste = float64(e.totalSpecProposed-e.totalSpecAccepted) / float64(e.totalSpecProposed)
	}

	var traceEvents []TraceEvent
	if e.traceCollector != nil {
		traceEvents = e.traceCollector.Events()
	}

	return &SimulationMetrics{
		TotalRequests:       totalReqs,
		SimulatedDurationMS: simDurationMS,
		RequestThroughput:   reqThroughput,
		TokenThroughput:     tokenThroughput,
		TTFT:                ComputePercentiles(ttftList),
		TPOT:                ComputePercentiles(tpotList),
		PeakKVBlocksUsed:    e.peakKVBlocksUsed,
		KVBlockUtilization:  kvUtil,
		SpeculativeYield:    specYield,
		SpeculativeWaste:    specWaste,
		CompletedRequests:   completedCopies,
		TraceEvents:         traceEvents,
	}
}

// Run executes a trace simulation with the provided configuration, requests, and latency model.
func Run(config SimulatorConfig, requests []RequestState, hw HardwareLatencyTable) (*SimulationMetrics, error) {
	engine, err := NewServingSimulator(config, hw)
	if err != nil {
		return nil, err
	}
	return engine.Run(requests)
}
