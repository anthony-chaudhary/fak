package agent

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func v41CoordinatorRequest(ctx context.Context, run func(context.Context) (inKernelGenerateResult, error)) *inKernelCoalesceRequest {
	return &inKernelCoalesceRequest{
		kind: inKernelCoalesceV41, ctx: ctx, run: run,
		prepared: make(chan *decodeLane, 1), proceed: make(chan error, 1),
		result: make(chan inKernelCoalesceResult, 1), done: make(chan struct{}),
		drain: make(chan struct{}, 1), receiptReady: make(chan struct{}),
	}
}

type v41CoordinatorTrimProbe struct {
	compute.Backend
	trims atomic.Int32
}

func (b *v41CoordinatorTrimProbe) Trim() { b.trims.Add(1) }

func v41LateFillRequest(maxNew int) (*inKernelCoalesceRequest, *decodeLane, *batchLaneSink, context.CancelFunc) {
	lanes, sinks := v41DriverLanes(1, maxNew)
	requestContext, cancel := context.WithCancel(context.Background())
	lane := lanes[0]
	request := v41CoordinatorRequest(requestContext, func(runContext context.Context) (inKernelGenerateResult, error) {
		defer lane.s.Close()
		coordinated, err := coalescedDecode(runContext, lane)
		if !coordinated && err == nil {
			err = errors.New("request did not enter coordinated decode")
		}
		return inKernelGenerateResult{gen: lane.gen}, err
	})
	return request, lane, sinks[0], cancel
}

func v41LateFillSelected(input []model.V41BatchLane) model.V41BatchResult {
	result := model.V41BatchResult{Selected: true, SharedHeadPanels: 1, SharedHeadRows: len(input), Lanes: make([]model.V41BatchLaneResult, len(input))}
	for index, lane := range input {
		result.Lanes[index] = model.V41BatchLaneResult{ID: lane.ID, Committed: true, Logits: []float32{0, 0, 10}}
	}
	return result
}

func v41LateFillQueue(planner *InKernelPlanner, requests ...*inKernelCoalesceRequest) {
	planner.coalesceMu.Lock()
	for _, request := range requests {
		request.enqueued = time.Now()
		planner.coalesceReady = append(planner.coalesceReady, request)
	}
	planner.coalesceRunning = true
	planner.coalesceMu.Unlock()
}

func v41LateFillQueued(planner *InKernelPlanner) []*inKernelCoalesceRequest {
	planner.coalesceMu.Lock()
	defer planner.coalesceMu.Unlock()
	return append([]*inKernelCoalesceRequest(nil), planner.coalesceReady...)
}

func v41LateFillClearQueue(planner *InKernelPlanner) {
	planner.coalesceMu.Lock()
	for index := range planner.coalesceReady {
		planner.coalesceReady[index] = nil
	}
	planner.coalesceReady = nil
	planner.coalesceRunning = false
	planner.coalesceMu.Unlock()
}

func v41LateFillIDs(input []model.V41BatchLane) []uint64 {
	ids := make([]uint64, len(input))
	for index, lane := range input {
		ids[index] = lane.ID
	}
	return ids
}

// fak-test:runtime fast est=2s lane=default
func TestV41CohortContinuousRetirement(t *testing.T) {
	modelInstance := model.NewSynthetic(tinyCfg())
	planner := &InKernelPlanner{m: modelInstance}
	requestA, _, _, cancelA := v41LateFillRequest(4)
	requestB, _, _, cancelB := v41LateFillRequest(2)
	prepareErr := errors.New("continuous replacement preparation failed")
	requestC := v41CoordinatorRequest(context.Background(), func(context.Context) (inKernelGenerateResult, error) {
		return inKernelGenerateResult{}, prepareErr
	})
	requestD, _, _, cancelD := v41LateFillRequest(2)
	requestE, _, _, cancelE := v41LateFillRequest(2)
	requestF, _, _, cancelF := v41LateFillRequest(2)
	defer func() {
		cancelA()
		cancelB()
		cancelD()
		cancelE()
		cancelF()
		v41LateFillClearQueue(planner)
	}()

	var batches [][]uint64
	var busyInput []model.V41BatchLane
	busyStable := false
	busyQueueStable := false
	retiredBeforeReplacement := false
	var retiredReceipt InKernelBatchReceipt
	calls := 0
	operations := inKernelV41DecodeOps{
		batch: func(input []model.V41BatchLane) (model.V41BatchResult, error) {
			calls++
			batches = append(batches, v41LateFillIDs(input))
			switch calls {
			case 1:
				busyInput = append([]model.V41BatchLane(nil), input...)
				v41LateFillQueue(planner, requestC, requestD, requestE, requestF)
				return v41DriverRefusal(input), model.ErrV41BatchBusy
			case 2:
				busyStable = reflect.DeepEqual(busyInput, input)
			case 3:
				select {
				case <-requestB.receiptReady:
					retiredBeforeReplacement = true
					retiredReceipt = requestB.receipt
				default:
				}
			}
			return v41LateFillSelected(input), nil
		},
		serial: func(*model.Session, int) ([]float32, error) {
			return nil, errors.New("unexpected serial fallback")
		},
		wait: func(context.Context) error {
			busyQueueStable = reflect.DeepEqual(v41LateFillQueued(planner), []*inKernelCoalesceRequest{requestC, requestD, requestE, requestF})
			return nil
		},
	}
	before := enginestep.Default.Snapshot(0, enginestep.KindCohort).Cohorts.Steps
	finished := make(chan struct{})
	go func() {
		planner.runV41DecodeCohort([]*inKernelCoalesceRequest{requestA, requestB}, func(ctx context.Context, lanes []*decodeLane, _ *model.Model) (inKernelV41DecodeReceipt, error) {
			return inKernelDecodeV41LanesWithOps(ctx, lanes, operations)
		})
		close(finished)
	}()
	select {
	case <-finished:
	case <-time.After(time.Second):
		cancelA()
		cancelB()
		cancelD()
		cancelE()
		cancelF()
		select {
		case <-finished:
		case <-time.After(time.Second):
			t.Fatal("continuous cohort failed to join after cancellation")
		}
		t.Fatal("continuous cohort exceeded deadline")
	}
	after := enginestep.Default.Snapshot(1, enginestep.KindCohort)
	if after.Cohorts.Steps != before+1 || len(after.Recent) != 1 || after.Recent[0].Lanes != 4 {
		t.Errorf("continuous cohort telemetry before=%d after=%d recent=%+v", before, after.Cohorts.Steps, after.Recent)
	}

	if !busyStable || !busyQueueStable {
		t.Errorf("busy retry stable=%t queue=%t", busyStable, busyQueueStable)
	}
	if !retiredBeforeReplacement {
		t.Errorf("finished request receipt was not published before replacement batch")
	} else if retiredReceipt.CohortSize != 2 || retiredReceipt.SharedPanels != 1 || retiredReceipt.SharedRows != 2 || retiredReceipt.SessionCloses != 1 {
		t.Errorf("retired receipt=%+v", retiredReceipt)
	}
	wantBatches := [][]uint64{{1, 2}, {1, 2}, {1, 3, 4, 5}, {1}}
	if !reflect.DeepEqual(batches, wantBatches) {
		t.Errorf("continuous batch IDs=%v want=%v", batches, wantBatches)
	}
	if queued := v41LateFillQueued(planner); len(queued) != 0 {
		t.Errorf("continuous queue retained %d requests", len(queued))
	}
	if result := <-requestC.result; result.err != prepareErr {
		t.Errorf("preparation error=%v", result.err)
	}
	for _, request := range []*inKernelCoalesceRequest{requestA, requestB, requestC, requestD, requestE, requestF} {
		select {
		case <-request.receiptReady:
		default:
			t.Errorf("request lacks terminal receipt")
		}
	}
	if requestA.closes.Load() != 1 || requestB.closes.Load() != 1 || requestC.closes.Load() != 0 || requestD.closes.Load() != 1 || requestE.closes.Load() != 1 || requestF.closes.Load() != 1 {
		t.Errorf("closes A/B/C/D/E/F=%d/%d/%d/%d/%d/%d", requestA.closes.Load(), requestB.closes.Load(), requestC.closes.Load(), requestD.closes.Load(), requestE.closes.Load(), requestF.closes.Load())
	}
	if requestC.receipt.CohortID == 0 || requestC.receipt.CohortID != requestA.receipt.CohortID || requestC.receipt.CohortSize != 0 || requestC.receipt.SharedPanels != 0 || requestC.receipt.SharedRows != 0 || requestC.receipt.SharedMACs != 0 || requestC.receipt.SessionCloses != 1 {
		t.Errorf("failed preparation receipt=%+v episode=%d", requestC.receipt, requestA.receipt.CohortID)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41CohortContinuousAllTerminalRefill(t *testing.T) {
	planner := &InKernelPlanner{m: model.NewSynthetic(tinyCfg())}
	requests := make([]*inKernelCoalesceRequest, 5)
	cancels := make([]context.CancelFunc, len(requests))
	for index := range requests {
		requests[index], _, _, cancels[index] = v41LateFillRequest(2)
	}
	defer func() {
		for _, cancel := range cancels {
			cancel()
		}
		v41LateFillClearQueue(planner)
	}()
	var batches [][]uint64
	planner.runV41DecodeCohort(requests[:4], func(ctx context.Context, lanes []*decodeLane, _ *model.Model) (inKernelV41DecodeReceipt, error) {
		return inKernelDecodeV41LanesWithOps(ctx, lanes, inKernelV41DecodeOps{
			batch: func(input []model.V41BatchLane) (model.V41BatchResult, error) {
				batches = append(batches, v41LateFillIDs(input))
				if len(batches) == 1 {
					v41LateFillQueue(planner, requests[4])
				}
				return v41LateFillSelected(input), nil
			},
			serial: func(*model.Session, int) ([]float32, error) { return nil, errors.New("unexpected serial fallback") },
			wait:   func(context.Context) error { return errors.New("unexpected busy wait") },
		})
	})
	if !reflect.DeepEqual(batches, [][]uint64{{1, 2, 3, 4}, {5}}) {
		t.Errorf("all-terminal refill batches=%v", batches)
	}
	for index, request := range requests {
		if request.receipt.CohortID == 0 || request.receipt.CohortID != requests[0].receipt.CohortID || request.closes.Load() != 1 {
			t.Errorf("request %d receipt=%+v closes=%d", index, request.receipt, request.closes.Load())
		}
	}
	if queued := v41LateFillQueued(planner); len(queued) != 0 {
		t.Errorf("all-terminal refill retained %d requests", len(queued))
	}
}

type v41ContinuousCompleteBackend struct {
	*v41CompleteBackend
	headCalls atomic.Int32
	entered   chan struct{}
	release   <-chan struct{}
	enterOnce sync.Once
}

func (backend *v41ContinuousCompleteBackend) BatchedMatMul(weight, input compute.Tensor, rows int) compute.Tensor {
	if len(weight.Shape) == 2 && weight.Shape[0] == 320 && weight.Shape[1] == 64 && backend.headCalls.Add(1) == 2 {
		backend.enterOnce.Do(func() { close(backend.entered) })
		<-backend.release
	}
	return backend.v41CompleteBackend.BatchedMatMul(weight, input, rows)
}

// fak-test:runtime fast est=3s lane=default
func TestV41CohortContinuousComplete(t *testing.T) {
	t.Setenv("FAK_V41_STEP_BATCH", "1")
	modelInstance := v41CompleteModel(t)
	defer modelInstance.CloseWeights()
	baseBackend := &v41CompleteBackend{Backend: compute.Default(), available: true}
	stepRelease := make(chan struct{})
	var stepReleaseOnce sync.Once
	releaseStep := func() { stepReleaseOnce.Do(func() { close(stepRelease) }) }
	defer releaseStep()
	backend := &v41ContinuousCompleteBackend{v41CompleteBackend: baseBackend, entered: make(chan struct{}), release: stepRelease}
	planner := NewInKernelPlannerWithConfig(modelInstance, loadProbeTok(t), "v41-cpu-contract", false, backend, false, InKernelPlannerConfig{BatchDecode: true})
	planner.quant, planner.maxNew, planner.tree, planner.scopedTree = false, 3, nil, nil
	queueRelease := make(chan struct{})
	planner.coalesceReadyHook = func() { <-queueRelease }
	messages := []Message{{Role: RoleUser, Content: "a"}}
	tokenLimits := []int{2, 3, 2, 2, 2, 2, 2, 2}
	type answer struct {
		completion *Completion
		err        error
	}
	done := make([]chan answer, len(tokenLimits))
	for index, tokenLimit := range tokenLimits {
		done[index] = make(chan answer, 1)
		go func(index, tokenLimit int) {
			completion, err := planner.Complete(context.Background(), messages, nil, WithRawPrompt(), WithMaxTokens(tokenLimit))
			done[index] <- answer{completion: completion, err: err}
		}(index, tokenLimit)
		awaitCoalescedQueue(t, planner, index+1)
	}
	close(queueRelease)
	select {
	case <-backend.entered:
	case <-time.After(time.Second):
		releaseStep()
		t.Fatal("long follower did not reach the held step")
	}
	var leader answer
	leaderReturned := false
	select {
	case leader = <-done[0]:
		leaderReturned = true
	case <-time.After(time.Second):
	}
	if baseBackend.kvCloses.Load() < 1 {
		t.Errorf("short leader session remained open while long follower was held")
	}
	releaseStep()
	receipts := make([]*InKernelBatchReceipt, len(tokenLimits))
	summaries := make([]*NativeDecodeSummary, len(tokenLimits))
	for index := range tokenLimits {
		result := leader
		if index != 0 || !leaderReturned {
			select {
			case result = <-done[index]:
			case <-time.After(5 * time.Second):
				t.Fatal("continuous Complete request did not finish")
			}
		}
		if result.err != nil || result.completion == nil || result.completion.Usage.CompletionTokens != tokenLimits[index] {
			t.Errorf("Complete request %d: completion=%+v err=%v", index, result.completion, result.err)
			continue
		}
		receipts[index] = result.completion.InKernelBatch
		summaries[index] = result.completion.NativeDecode
	}
	if !leaderReturned {
		t.Errorf("short initial leader did not return before long follower finished")
	}
	if receipts[0] == nil {
		t.Fatal("continuous Complete leader lacks receipt")
	}
	for index, receipt := range receipts[1:] {
		if receipt == nil || receipt.CohortID != receipts[0].CohortID {
			t.Errorf("Complete request %d receipt=%+v leader=%+v", index+1, receipt, receipts[0])
		}
	}
	isolated := receipts[len(receipts)-1]
	isolatedSummary := summaries[len(summaries)-1]
	if isolated == nil || isolated.CohortSize != 1 || isolated.SharedPanels != 0 || isolated.SharedRows != 0 || isolated.SessionCloses != 8 {
		t.Errorf("late isolated receipt=%+v", isolated)
	}
	if isolatedSummary == nil || isolatedSummary.Path != NativeDecodePathSerial || isolatedSummary.CohortSize != 1 {
		t.Errorf("late isolated summary=%+v", isolatedSummary)
	}
	if baseBackend.kvCreates.Load() != int32(len(tokenLimits)) || baseBackend.kvCloses.Load() != int32(len(tokenLimits)) {
		t.Errorf("Complete KV=%d/%d", baseBackend.kvCreates.Load(), baseBackend.kvCloses.Load())
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41CohortContinuousTerminalHandoff(t *testing.T) {
	planner := &InKernelPlanner{m: model.NewSynthetic(tinyCfg())}
	lanes, _ := v41DriverLanes(2, 3)
	laneA, laneB := lanes[0], lanes[1]
	laneB.maxNew = 2
	cleanupStarted, releaseCleanup := make(chan struct{}), make(chan struct{})
	requestA := v41CoordinatorRequest(context.Background(), func(runContext context.Context) (inKernelGenerateResult, error) {
		defer func() {
			close(cleanupStarted)
			<-releaseCleanup
			laneA.s.Close()
		}()
		_, err := coalescedDecode(runContext, laneA)
		return inKernelGenerateResult{gen: laneA.gen}, err
	})
	requestB := v41CoordinatorRequest(context.Background(), func(runContext context.Context) (inKernelGenerateResult, error) {
		defer laneB.s.Close()
		_, err := coalescedDecode(runContext, laneB)
		return inKernelGenerateResult{gen: laneB.gen}, err
	})
	primary := &struct{ stage string }{"continuous replacement preparation"}
	retiredBeforePanic := false
	var retiredReceipt InKernelBatchReceipt
	requestPanic := v41CoordinatorRequest(context.Background(), func(context.Context) (inKernelGenerateResult, error) {
		select {
		case <-requestB.receiptReady:
			retiredBeforePanic = true
			retiredReceipt = requestB.receipt
		default:
		}
		panic(primary)
	})
	requestSelected, _, _, cancelSelected := v41LateFillRequest(2)
	requestQwen, _, _, cancelQwen := v41LateFillRequest(2)
	requestQwen.kind = inKernelCoalesceQwen
	requestQueued, _, _, cancelQueued := v41LateFillRequest(2)
	defer func() {
		cancelSelected()
		cancelQwen()
		cancelQueued()
		v41LateFillClearQueue(planner)
	}()

	calls := 0
	finished := make(chan any, 1)
	go func() {
		defer func() { finished <- recover() }()
		planner.runV41DecodeCohort([]*inKernelCoalesceRequest{requestA, requestB}, func(ctx context.Context, got []*decodeLane, _ *model.Model) (inKernelV41DecodeReceipt, error) {
			return inKernelDecodeV41LanesWithOps(ctx, got, inKernelV41DecodeOps{
				batch: func(input []model.V41BatchLane) (model.V41BatchResult, error) {
					calls++
					if calls == 1 {
						v41LateFillQueue(planner, requestPanic, requestSelected, requestQwen, requestQueued)
					}
					return v41LateFillSelected(input), nil
				},
				serial: func(*model.Session, int) ([]float32, error) { return nil, errors.New("unexpected serial fallback") },
				wait:   func(context.Context) error { return errors.New("unexpected busy wait") },
			})
		})
	}()
	select {
	case <-cleanupStarted:
	case <-time.After(time.Second):
		close(releaseCleanup)
		t.Fatal("fatal continuous cleanup never began")
	}
	select {
	case got := <-finished:
		close(releaseCleanup)
		t.Fatalf("panic escaped before selected cleanup: %v", got)
	case <-time.After(10 * time.Millisecond):
	}
	close(releaseCleanup)
	select {
	case got := <-finished:
		if got != primary {
			t.Errorf("panic identity=%v", got)
		}
	case <-time.After(time.Second):
		t.Fatal("fatal continuous cohort failed to join")
	}
	if !retiredBeforePanic || retiredReceipt.CohortSize != 2 || retiredReceipt.SharedPanels != 1 || retiredReceipt.SharedRows != 2 || retiredReceipt.SessionCloses != 1 {
		t.Errorf("retired before panic=%t receipt=%+v", retiredBeforePanic, retiredReceipt)
	}
	if requestB.receipt != retiredReceipt || requestB.closes.Load() != 1 {
		t.Errorf("retired request changed receipt=%+v closes=%d", requestB.receipt, requestB.closes.Load())
	}
	if _, err := finishCoalescedRequest(requestB, <-requestB.result); err != nil {
		t.Errorf("retired request inherited fatal error=%v", err)
	}
	for _, request := range []*inKernelCoalesceRequest{requestA, requestPanic, requestSelected} {
		select {
		case <-request.receiptReady:
		default:
			t.Errorf("selected request lacks terminal receipt")
		}
	}
	if !errors.Is(requestA.terminalErr, errInKernelV41CohortAborted) || !errors.Is(requestSelected.terminalErr, errInKernelV41CohortAborted) || requestPanic.panicValue != primary {
		t.Errorf("fatal terminal state A=%v selected=%v panic=%v", requestA.terminalErr, requestSelected.terminalErr, requestPanic.panicValue)
	}
	if queued := v41LateFillQueued(planner); !reflect.DeepEqual(queued, []*inKernelCoalesceRequest{requestQwen, requestQueued}) {
		t.Errorf("fatal handoff queue=%p", queued)
	}
	gateContext, gateCancel := context.WithTimeout(context.Background(), time.Second)
	defer gateCancel()
	if err := planner.devMu.LockContext(gateContext); err != nil {
		t.Errorf("fatal continuous cohort retained planner gate")
	} else {
		planner.devMu.Unlock()
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41CohortContinuousIdleArrivalHandoff(t *testing.T) {
	modelInstance := model.NewSynthetic(tinyCfg())
	queueRelease := make(chan struct{})
	planner := &InKernelPlanner{m: modelInstance, coalesceReadyHook: func() { <-queueRelease }}
	cleanupStarted, releaseCleanup := make(chan struct{}), make(chan struct{})
	var batchMu sync.Mutex
	var batchSizes []int
	planner.coalesceBatchHook = func(size int) {
		batchMu.Lock()
		batchSizes = append(batchSizes, size)
		batchMu.Unlock()
	}
	type answer struct {
		result inKernelGenerateResult
		err    error
	}
	done := make([]chan answer, 3)
	for index := range done {
		done[index] = make(chan answer, 1)
	}
	run := func(index int, blockCleanup bool) {
		result, err := planner.runCoalescedGenerateKind(context.Background(), func(runContext context.Context) (inKernelGenerateResult, error) {
			lane, _ := newDecodeLaneOver(modelInstance, []int{1, 2}, 1, int64(index))
			defer lane.s.Close()
			_, decodeErr := coalescedDecode(runContext, lane)
			if blockCleanup {
				close(cleanupStarted)
				<-releaseCleanup
			}
			return inKernelGenerateResult{gen: lane.gen}, decodeErr
		}, inKernelCoalesceV41)
		done[index] <- answer{result: result, err: err}
	}
	go run(0, true)
	awaitCoalescedQueue(t, planner, 1)
	close(queueRelease)
	select {
	case <-cleanupStarted:
	case <-time.After(time.Second):
		close(releaseCleanup)
		t.Fatal("idle-boundary cleanup did not start")
	}
	go run(1, false)
	awaitCoalescedQueue(t, planner, 1)
	go run(2, false)
	awaitCoalescedQueue(t, planner, 2)
	close(releaseCleanup)
	answers := make([]answer, len(done))
	for index := range done {
		select {
		case answers[index] = <-done[index]:
		case <-time.After(time.Second):
			t.Fatal("active-to-idle arrival stranded")
		}
	}
	for index, got := range answers {
		if got.err != nil || got.result.gen != 1 {
			t.Errorf("request %d result=%+v err=%v", index, got.result, got.err)
		}
	}
	batchMu.Lock()
	sizes := append([]int(nil), batchSizes...)
	batchMu.Unlock()
	if !reflect.DeepEqual(sizes, []int{1, 2}) || answers[0].result.batchReceipt.CohortID == answers[1].result.batchReceipt.CohortID || answers[1].result.batchReceipt.CohortID != answers[2].result.batchReceipt.CohortID {
		t.Errorf("idle handoff sizes=%v receipts=%+v/%+v/%+v", sizes, answers[0].result.batchReceipt, answers[1].result.batchReceipt, answers[2].result.batchReceipt)
	}
	planner.coalesceMu.Lock()
	running, queued := planner.coalesceRunning, len(planner.coalesceReady)
	planner.coalesceMu.Unlock()
	if running || queued != 0 {
		t.Errorf("idle handoff running=%t queued=%d", running, queued)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41CohortLateFillBounded(t *testing.T) {
	t.Run("fills initial headroom once and counts failed preparation", func(t *testing.T) {
		modelInstance := model.NewSynthetic(tinyCfg())
		planner := &InKernelPlanner{m: modelInstance}
		requestA, laneA, _, cancelA := v41LateFillRequest(3)
		requestB, _, _, cancelB := v41LateFillRequest(3)
		requestD, _, _, cancelD := v41LateFillRequest(3)
		requestE, _, _, cancelE := v41LateFillRequest(3)
		prepareErr := errors.New("late preparation failed")
		requestC := v41CoordinatorRequest(context.Background(), func(context.Context) (inKernelGenerateResult, error) {
			return inKernelGenerateResult{}, prepareErr
		})
		defer func() {
			cancelA()
			cancelB()
			cancelD()
			cancelE()
			v41LateFillClearQueue(planner)
		}()

		var batches [][]uint64
		calls := 0
		operations := inKernelV41DecodeOps{
			batch: func(input []model.V41BatchLane) (model.V41BatchResult, error) {
				calls++
				batches = append(batches, v41LateFillIDs(input))
				if calls == 1 {
					v41LateFillQueue(planner, requestB, requestC, requestD, requestE)
				}
				return v41LateFillSelected(input), nil
			},
			serial: func(*model.Session, int) ([]float32, error) {
				return nil, errors.New("unexpected serial fallback")
			},
			wait: func(context.Context) error { return errors.New("unexpected busy wait") },
		}
		finished := make(chan struct{})
		go func() {
			planner.runV41DecodeCohort([]*inKernelCoalesceRequest{requestA}, func(ctx context.Context, lanes []*decodeLane, _ *model.Model) (inKernelV41DecodeReceipt, error) {
				return inKernelDecodeV41LanesWithOps(ctx, lanes, operations)
			})
			close(finished)
		}()
		select {
		case <-finished:
		case <-time.After(time.Second):
			cancelA()
			cancelB()
			cancelD()
			cancelE()
			select {
			case <-finished:
			case <-time.After(time.Second):
				t.Fatal("late-fill cohort failed to join after cancellation")
			}
			t.Fatal("late-fill cohort exceeded deadline")
		}
		wantBatches := [][]uint64{{1}, {1, 2, 3, 4}, {2, 3, 4}}
		if !reflect.DeepEqual(batches, wantBatches) {
			t.Errorf("batch IDs=%v want=%v", batches, wantBatches)
		}
		queued := v41LateFillQueued(planner)
		if len(queued) != 0 {
			t.Errorf("queue after replacement fill=%p", queued)
		}
		if requestA.decodePass.Load() != 1 || requestB.decodePass.Load() != 1 || requestC.decodePass.Load() != 0 || requestD.decodePass.Load() != 1 || requestE.decodePass.Load() != 1 {
			t.Errorf("decode passes A/B/C/D/E=%d/%d/%d/%d/%d", requestA.decodePass.Load(), requestB.decodePass.Load(), requestC.decodePass.Load(), requestD.decodePass.Load(), requestE.decodePass.Load())
		}
		if requestA.closes.Load() != 1 || requestB.closes.Load() != 1 || requestC.closes.Load() != 0 || requestD.closes.Load() != 1 || requestE.closes.Load() != 1 {
			t.Errorf("closes A/B/C/D/E=%d/%d/%d/%d/%d", requestA.closes.Load(), requestB.closes.Load(), requestC.closes.Load(), requestD.closes.Load(), requestE.closes.Load())
		}
		for _, request := range []*inKernelCoalesceRequest{requestA, requestB, requestC, requestD, requestE} {
			select {
			case <-request.receiptReady:
			default:
				t.Errorf("selected request lacks receipt")
			}
		}
		if requestA.receipt.CohortSize != 4 || requestA.receipt.SharedPanels != 2 || requestA.receipt.SharedRows != 5 || requestA.receipt.SessionCloses != 1 {
			t.Errorf("A receipt=%+v", requestA.receipt)
		}
		for _, request := range []*inKernelCoalesceRequest{requestB, requestD, requestE} {
			if request.receipt.CohortSize != 4 || request.receipt.SharedPanels != 2 || request.receipt.SharedRows != 7 || request.receipt.SessionCloses != 4 {
				t.Errorf("replacement receipt=%+v", request.receipt)
			}
		}
		if requestC.receipt.CohortSize != 0 || requestC.receipt.SharedPanels != 0 || requestC.receipt.SharedRows != 0 || requestC.receipt.SessionCloses != 0 {
			t.Errorf("failed preparation receipt=%+v", requestC.receipt)
		}
		select {
		case result := <-requestC.result:
			if result.err != prepareErr {
				t.Errorf("preparation error=%v", result.err)
			}
		default:
			t.Errorf("failed preparation did not complete")
		}
		if laneA.gen != 3 {
			t.Errorf("A generated=%d", laneA.gen)
		}
	})

	t.Run("busy retry blocks admission and FIFO kinds cannot be bypassed", func(t *testing.T) {
		modelInstance := model.NewSynthetic(tinyCfg())
		planner := &InKernelPlanner{m: modelInstance}
		requestA, laneA, sinkA, cancelA := v41LateFillRequest(3)
		requestB, _, _, cancelB := v41LateFillRequest(3)
		requestQwen, _, _, cancelQwen := v41LateFillRequest(2)
		requestQwen.kind = inKernelCoalesceQwen
		requestC, _, _, cancelC := v41LateFillRequest(2)
		defer func() {
			cancelA()
			cancelB()
			cancelQwen()
			cancelC()
			v41LateFillClearQueue(planner)
		}()

		var batches [][]uint64
		var first []model.V41BatchLane
		busyStable := false
		busyUnchanged := false
		fifoHeld := false
		calls := 0
		operations := inKernelV41DecodeOps{
			batch: func(input []model.V41BatchLane) (model.V41BatchResult, error) {
				calls++
				batches = append(batches, v41LateFillIDs(input))
				switch calls {
				case 1:
					first = append([]model.V41BatchLane(nil), input...)
					v41LateFillQueue(planner, requestB, requestQwen, requestC)
					return v41DriverRefusal(input), model.ErrV41BatchBusy
				case 2:
					busyStable = reflect.DeepEqual(first, input)
				case 3, 4:
					fifoHeld = reflect.DeepEqual(v41LateFillQueued(planner), []*inKernelCoalesceRequest{requestQwen, requestC})
				}
				return v41LateFillSelected(input), nil
			},
			serial: func(*model.Session, int) ([]float32, error) {
				return nil, errors.New("unexpected serial fallback")
			},
			wait: func(context.Context) error {
				busyUnchanged = laneA.gen == 1 && len(laneA.forwarded) == 0 && reflect.DeepEqual(sinkA.toks, []int{1}) && reflect.DeepEqual(v41LateFillQueued(planner), []*inKernelCoalesceRequest{requestB, requestQwen, requestC})
				return nil
			},
		}
		finished := make(chan struct{})
		go func() {
			planner.runV41DecodeCohort([]*inKernelCoalesceRequest{requestA}, func(ctx context.Context, lanes []*decodeLane, _ *model.Model) (inKernelV41DecodeReceipt, error) {
				return inKernelDecodeV41LanesWithOps(ctx, lanes, operations)
			})
			close(finished)
		}()
		select {
		case <-finished:
		case <-time.After(time.Second):
			cancelA()
			cancelB()
			cancelQwen()
			cancelC()
			select {
			case <-finished:
			case <-time.After(time.Second):
				t.Fatal("busy late-fill cohort failed to join after cancellation")
			}
			t.Fatal("busy late-fill cohort exceeded deadline")
		}

		wantBatches := [][]uint64{{1}, {1}, {1, 2}, {2}}
		if !reflect.DeepEqual(batches, wantBatches) || !busyStable || !busyUnchanged || !fifoHeld {
			t.Errorf("batches=%v stable=%t unchanged=%t fifo=%t", batches, busyStable, busyUnchanged, fifoHeld)
		}
		if requestA.decodePass.Load() != 1 || requestB.decodePass.Load() != 1 || requestQwen.decodePass.Load() != 0 || requestC.decodePass.Load() != 0 {
			t.Errorf("decode passes A/B/Q/C=%d/%d/%d/%d", requestA.decodePass.Load(), requestB.decodePass.Load(), requestQwen.decodePass.Load(), requestC.decodePass.Load())
		}
		if !reflect.DeepEqual(v41LateFillQueued(planner), []*inKernelCoalesceRequest{requestQwen, requestC}) {
			t.Errorf("FIFO queue changed after cohort")
		}
	})
}

// fak-test:runtime fast est=100ms lane=default
func TestV41CohortLateFillTerminalCleanup(t *testing.T) {
	t.Run("late preparation panic joins selected teardown", func(t *testing.T) {
		planner := &InKernelPlanner{m: model.NewSynthetic(tinyCfg())}
		lanes, sinks := v41DriverLanes(1, 3)
		laneA := lanes[0]
		cleanupStarted, releaseCleanup := make(chan struct{}), make(chan struct{})
		requestA := v41CoordinatorRequest(context.Background(), func(runContext context.Context) (inKernelGenerateResult, error) {
			defer func() {
				close(cleanupStarted)
				<-releaseCleanup
				laneA.s.Close()
			}()
			_, err := coalescedDecode(runContext, laneA)
			return inKernelGenerateResult{gen: laneA.gen}, err
		})
		primary := &struct{ stage string }{"late preparation"}
		requestPanic := v41CoordinatorRequest(context.Background(), func(context.Context) (inKernelGenerateResult, error) { panic(primary) })
		requestC, _, _, cancelC := v41LateFillRequest(2)
		requestD, _, _, cancelD := v41LateFillRequest(2)
		requestE, _, _, cancelE := v41LateFillRequest(2)
		defer func() {
			cancelC()
			cancelD()
			cancelE()
			v41LateFillClearQueue(planner)
		}()

		calls := 0
		finished := make(chan any, 1)
		go func() {
			defer func() { finished <- recover() }()
			planner.runV41DecodeCohort([]*inKernelCoalesceRequest{requestA}, func(ctx context.Context, got []*decodeLane, _ *model.Model) (inKernelV41DecodeReceipt, error) {
				return inKernelDecodeV41LanesWithOps(ctx, got, inKernelV41DecodeOps{
					batch: func(input []model.V41BatchLane) (model.V41BatchResult, error) {
						calls++
						if calls == 1 {
							v41LateFillQueue(planner, requestPanic, requestC, requestD, requestE)
						}
						return v41LateFillSelected(input), nil
					},
					serial: func(*model.Session, int) ([]float32, error) { return nil, errors.New("unexpected serial fallback") },
					wait:   func(context.Context) error { return errors.New("unexpected busy wait") },
				})
			})
		}()
		select {
		case <-cleanupStarted:
		case <-time.After(time.Second):
			close(releaseCleanup)
			t.Fatal("fatal late-fill cleanup never began")
		}
		select {
		case got := <-finished:
			close(releaseCleanup)
			t.Fatalf("panic escaped before selected cleanup: %v", got)
		case <-time.After(10 * time.Millisecond):
		}
		close(releaseCleanup)
		select {
		case got := <-finished:
			if got != primary {
				t.Errorf("panic identity=%v", got)
			}
		case <-time.After(time.Second):
			t.Fatal("fatal late-fill cohort failed to join")
		}
		if calls != 1 || laneA.gen != 2 || !reflect.DeepEqual(laneA.forwarded, []int{1}) || !reflect.DeepEqual(sinks[0].toks, []int{1, 2}) {
			t.Errorf("panic replayed A: calls=%d gen=%d forwarded=%v emitted=%v", calls, laneA.gen, laneA.forwarded, sinks[0].toks)
		}
		for _, request := range []*inKernelCoalesceRequest{requestA, requestPanic, requestC, requestD} {
			select {
			case <-request.done:
			default:
				t.Errorf("selected request did not finish")
			}
			select {
			case <-request.receiptReady:
			default:
				t.Errorf("selected request lacks terminal receipt")
			}
		}
		if requestA.receipt.SharedPanels != 1 || requestA.receipt.SharedRows != 1 || requestA.receipt.SharedMACs != 0 {
			t.Errorf("fatal late-fill erased committed receipt: %+v", requestA.receipt)
		}
		if requestA.closes.Load() != 1 || !reflect.DeepEqual(v41LateFillQueued(planner), []*inKernelCoalesceRequest{requestE}) {
			t.Errorf("fatal cleanup closes=%d queue=%p", requestA.closes.Load(), v41LateFillQueued(planner))
		}
		gateContext, gateCancel := context.WithTimeout(context.Background(), time.Second)
		defer gateCancel()
		if err := planner.devMu.LockContext(gateContext); err != nil {
			t.Error("fatal late-fill retained planner gate")
		} else {
			planner.devMu.Unlock()
		}
	})

	t.Run("selected cancellation is local and refills retired capacity", func(t *testing.T) {
		planner := &InKernelPlanner{m: model.NewSynthetic(tinyCfg())}
		requestA, _, _, cancelA := v41LateFillRequest(3)
		lanesB, _ := v41DriverLanes(1, 3)
		laneB := lanesB[0]
		contextB, cancelB := context.WithCancel(context.Background())
		startedB, releaseB := make(chan struct{}), make(chan struct{})
		requestB := v41CoordinatorRequest(contextB, func(runContext context.Context) (inKernelGenerateResult, error) {
			close(startedB)
			<-releaseB
			defer laneB.s.Close()
			_, err := coalescedDecode(runContext, laneB)
			return inKernelGenerateResult{gen: laneB.gen}, err
		})
		requestC, _, _, cancelC := v41LateFillRequest(3)
		requestD, _, _, cancelD := v41LateFillRequest(3)
		requestE, _, _, cancelE := v41LateFillRequest(2)
		var releaseOnce sync.Once
		release := func() { releaseOnce.Do(func() { close(releaseB) }) }
		defer func() {
			cancelA()
			cancelB()
			cancelC()
			cancelD()
			cancelE()
			release()
			v41LateFillClearQueue(planner)
		}()

		var batches [][]uint64
		finished := make(chan struct{})
		go func() {
			planner.runV41DecodeCohort([]*inKernelCoalesceRequest{requestA}, func(ctx context.Context, got []*decodeLane, _ *model.Model) (inKernelV41DecodeReceipt, error) {
				return inKernelDecodeV41LanesWithOps(ctx, got, inKernelV41DecodeOps{
					batch: func(input []model.V41BatchLane) (model.V41BatchResult, error) {
						batches = append(batches, v41LateFillIDs(input))
						if len(batches) == 1 {
							v41LateFillQueue(planner, requestB, requestC, requestD, requestE)
						}
						return v41LateFillSelected(input), nil
					},
					serial: func(*model.Session, int) ([]float32, error) { return nil, errors.New("unexpected serial fallback") },
					wait:   func(context.Context) error { return errors.New("unexpected busy wait") },
				})
			})
			close(finished)
		}()
		selected := false
		select {
		case <-startedB:
			selected = true
			cancelB()
			release()
		case <-finished:
		case <-time.After(time.Second):
			t.Fatal("selected cancellation never reached admission")
		}
		if selected {
			select {
			case <-finished:
			case <-time.After(time.Second):
				t.Fatal("selected cancellation cohort failed to join")
			}
		}
		if !selected || !reflect.DeepEqual(batches, [][]uint64{{1}, {1, 3, 4}, {3, 4, 5}}) {
			t.Errorf("selected=%t batches=%v", selected, batches)
		}
		select {
		case result := <-requestB.result:
			if !errors.Is(result.err, context.Canceled) {
				t.Errorf("selected cancellation result=%v", result.err)
			}
		default:
			t.Errorf("selected cancellation has no result")
		}
		if laneB.gen != 0 || len(laneB.forwarded) != 0 || requestB.closes.Load() != 1 || requestE.decodePass.Load() != 1 || len(v41LateFillQueued(planner)) != 0 {
			t.Errorf("canceled lane gen=%d forwarded=%v closes=%d Epasses=%d queue=%p", laneB.gen, laneB.forwarded, requestB.closes.Load(), requestE.decodePass.Load(), v41LateFillQueued(planner))
		}
	})
}

// Scripted positive results below qualify request/driver state transitions only.
// No Flash head arithmetic or actual Complete admission is fabricated.
// fak-test:runtime fast est=30ms lane=default
func TestInKernelV41CoordinatorLaneOutcomes(t *testing.T) {
	lanes, sinks := v41DriverLanes(4, 3)
	m := model.NewSynthetic(tinyCfg())
	backend := &v41CoordinatorTrimProbe{Backend: compute.Default()}
	p := &InKernelPlanner{m: m, backend: backend}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	fault := &InKernelOOMError{Bytes: 17, Site: "scripted-cohort-fault"}
	var closed atomic.Int32
	requests := make([]*inKernelCoalesceRequest, len(lanes))
	for i, lane := range lanes {
		lane.s = m.NewSession()
		laneCtx := context.Background()
		if i == 1 {
			laneCtx = ctx
		}
		requests[i] = v41CoordinatorRequest(laneCtx, func(runCtx context.Context) (inKernelGenerateResult, error) {
			defer func() { lane.s.Close(); closed.Add(1) }()
			coordinated, err := coalescedDecode(runCtx, lane)
			if !coordinated {
				t.Error("request lost coalesced context")
			}
			if lane.err != nil {
				err = lane.err
			}
			if err != nil {
				if retry, _ := p.admitDeviceOOMRetry(runCtx, fault, 1, 3); retry {
					t.Error("accepted/emitted request authorized whole-request retry")
				}
			}
			return inKernelGenerateResult{gen: lane.gen}, err
		})
	}
	calls, waits, scalar := 0, 0, 0
	var first []model.V41BatchLane
	ops := inKernelV41DecodeOps{
		batch: func(input []model.V41BatchLane) (model.V41BatchResult, error) {
			calls++
			switch calls {
			case 1:
				first = append([]model.V41BatchLane(nil), input...)
				return v41DriverRefusal(input), model.ErrV41BatchBusy
			case 2:
				if !reflect.DeepEqual(first, input) {
					t.Fatal("owner-busy retry changed chosen tokens or stable IDs")
				}
				cancel()
				return model.V41BatchResult{Selected: true, SharedHeadPanels: 1, SharedHeadRows: 4, Lanes: []model.V41BatchLaneResult{
					{ID: input[3].ID, Committed: true, Err: fault},
					{ID: input[1].ID, Err: context.Canceled},
					{ID: input[0].ID, Committed: true, Logits: []float32{0, 0, 10}},
					{ID: input[2].ID, Err: fault},
				}}, fault
			case 3:
				if len(input) != 1 || input[0].ID != 1 || input[0].Token != 2 {
					t.Fatal("failed or committed-error lane was replayed")
				}
				return v41DriverRefusal(input), model.ErrV41BatchUnavailable
			default:
				t.Fatal("unexpected batch replay")
				return model.V41BatchResult{}, nil
			}
		},
		serial: func(s *model.Session, token int) ([]float32, error) {
			scalar++
			if s != lanes[0].s || token != 2 {
				t.Fatal("scalar fallback changed surviving request")
			}
			return []float32{0, 0, 10}, nil
		},
		wait: func(context.Context) error {
			waits++
			for i, lane := range lanes {
				if lane.gen != 1 || lane.counts[1] != 1 || !reflect.DeepEqual(sinks[i].toks, []int{1}) {
					t.Fatal("busy wait resampled or emitted twice")
				}
			}
			return nil
		},
	}
	p.runV41DecodeCohort(requests, func(ctx context.Context, got []*decodeLane, _ *model.Model) (inKernelV41DecodeReceipt, error) {
		return inKernelDecodeV41LanesWithOps(ctx, got, ops)
	})
	if calls != 3 || waits != 1 || scalar != 1 || closed.Load() != 4 || backend.trims.Load() != 0 {
		t.Fatalf("calls/waits/scalar/closed/trims=%d/%d/%d/%d/%d", calls, waits, scalar, closed.Load(), backend.trims.Load())
	}
	for i, req := range requests {
		out := <-req.result
		<-req.receiptReady
		want := error(fault)
		if i == 0 {
			want = nil
		} else if i == 1 {
			want = context.Canceled
		}
		wantCloses := uint32(3)
		if i == 0 {
			wantCloses = 4
		}
		if out.err != want || req.decodePass.Load() != 1 || req.receipt.SharedPanels != 1 || req.receipt.SharedRows != 4 || req.receipt.SharedMACs != 0 || req.receipt.SessionCloses != wantCloses {
			t.Fatalf("lane %d outcome=%v receipt=%+v passes=%d", i, out.err, req.receipt, req.decodePass.Load())
		}
	}
	if !reflect.DeepEqual(sinks[0].toks, []int{1, 2, 2}) || !reflect.DeepEqual(lanes[0].forwarded, []int{1, 2}) || len(lanes[1].forwarded) != 0 || len(lanes[2].forwarded) != 0 || !reflect.DeepEqual(lanes[3].forwarded, []int{1}) {
		t.Fatal("per-lane scatter lost healthy output or actual commit accounting")
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestInKernelV41CoordinatorPanicJoinsTeardown(t *testing.T) {
	lanes, sinks := v41DriverLanes(2, 3)
	m := model.NewSynthetic(tinyCfg())
	p := &InKernelPlanner{m: m}
	primary := &struct{ name string }{"decode panic"}
	secondary := &struct{ name string }{"cleanup panic"}
	cleanupStarted, releaseCleanup := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(releaseCleanup) }) }
	defer release()
	var closed atomic.Int32
	requests := make([]*inKernelCoalesceRequest, len(lanes))
	for i, lane := range lanes {
		lane.s = m.NewSession()
		requests[i] = v41CoordinatorRequest(context.Background(), func(ctx context.Context) (inKernelGenerateResult, error) {
			defer func() {
				if i == 0 {
					close(cleanupStarted)
					<-releaseCleanup
				}
				lane.s.Close()
				closed.Add(1)
				if i == 1 {
					panic(secondary)
				}
			}()
			_, err := coalescedDecode(ctx, lane)
			return inKernelGenerateResult{}, err
		})
	}
	finished := make(chan any, 1)
	go func() {
		defer func() { finished <- recover() }()
		p.runV41DecodeCohort(requests, func(ctx context.Context, got []*decodeLane, _ *model.Model) (inKernelV41DecodeReceipt, error) {
			return inKernelDecodeV41LanesWithOps(ctx, got, inKernelV41DecodeOps{
				batch:  func([]model.V41BatchLane) (model.V41BatchResult, error) { panic(primary) },
				serial: func(*model.Session, int) ([]float32, error) { panic("unexpected replay") },
				wait:   func(context.Context) error { panic("unexpected retry") },
			})
		})
	}()
	select {
	case <-cleanupStarted:
	case <-time.After(time.Second):
		t.Fatal("request cleanup never began")
	}
	select {
	case got := <-finished:
		t.Fatalf("panic escaped while cleanup blocked: %v", got)
	case <-time.After(10 * time.Millisecond):
	}
	for _, req := range requests {
		select {
		case <-req.receiptReady:
			t.Fatal("receipt published before all teardown joined")
		default:
		}
	}
	release()
	select {
	case got := <-finished:
		if got != primary {
			t.Fatalf("panic identity changed: %v", got)
		}
	case <-time.After(time.Second):
		t.Fatal("fatal cohort failed to join")
	}
	for i, req := range requests {
		<-req.done
		<-req.receiptReady
		if out := <-req.result; !errors.Is(out.err, errInKernelV41CohortAborted) || req.decodePass.Load() != 1 || !reflect.DeepEqual(sinks[i].toks, []int{1}) {
			t.Fatal("fatal request replayed or lost terminal cause")
		}
	}
	if closed.Load() != 2 {
		t.Fatal("panic escaped before every Session.Close")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := p.devMu.LockContext(ctx); err != nil {
		t.Fatal("fatal cohort retained planner gate")
	}
	p.devMu.Unlock()
}

// This executes real queue/handoff and dense CPU sessions, not Flash admission.
// fak-test:runtime fast est=100ms lane=default
func TestInKernelV41CoordinatorFIFOAndCap(t *testing.T) {
	const v41CohortLimit = 4
	m := model.NewSynthetic(tinyCfg())
	release := make(chan struct{})
	var releaseOnce sync.Once
	open := func() { releaseOnce.Do(func() { close(release) }) }
	p := &InKernelPlanner{m: m, coalesceReadyHook: func() { <-release }}
	kinds := []inKernelCoalesceKind{inKernelCoalesceV41, inKernelCoalesceV41, inKernelCoalesceV41, inKernelCoalesceV41, inKernelCoalesceV41, inKernelCoalesceQwen, inKernelCoalesceQwen, inKernelCoalesceV41, inKernelCoalesceQwen}
	wantID := make([]uint64, len(kinds))
	var wantSizes []int
	var cohortID uint64
	for first := 0; first < len(kinds); {
		limit := inKernelDecodeCohortMax
		if kinds[first] == inKernelCoalesceV41 {
			limit = v41CohortLimit
		}
		last := first + 1
		for last < len(kinds) && last-first < limit && kinds[last] == kinds[first] {
			last++
		}
		cohortID++
		wantSizes = append(wantSizes, last-first)
		for i := first; i < last; i++ {
			wantID[i] = cohortID
		}
		first = last
	}
	var mu sync.Mutex
	var order, sizes []int
	p.coalesceBatchHook = func(n int) { mu.Lock(); sizes = append(sizes, n); mu.Unlock() }
	answers := make([]chan inKernelCoalesceResult, len(kinds))
	defer open()
	for i, kind := range kinds {
		answers[i] = make(chan inKernelCoalesceResult, 1)
		go func(i int, kind inKernelCoalesceKind) {
			res, err := p.runCoalescedGenerateKind(context.Background(), func(ctx context.Context) (inKernelGenerateResult, error) {
				mu.Lock()
				order = append(order, i)
				mu.Unlock()
				lane, _ := newDecodeLaneOver(m, []int{1, 2}, 1, 0)
				defer lane.s.Close()
				_, err := coalescedDecode(ctx, lane)
				return inKernelGenerateResult{gen: lane.gen}, err
			}, kind)
			answers[i] <- inKernelCoalesceResult{result: res, err: err}
		}(i, kind)
		awaitCoalescedQueue(t, p, i+1)
	}
	open()
	for i, done := range answers {
		select {
		case out := <-done:
			if out.err != nil || out.result.gen != 1 || out.result.batchReceipt.CohortID != wantID[i] || out.result.batchReceipt.SharedPanels != 0 || out.result.batchReceipt.SharedRows != 0 {
				t.Fatalf("request %d: %+v %v", i, out.result, out.err)
			}
		case <-time.After(time.Second):
			t.Fatal("FIFO request stranded")
		}
	}
	wantOrder := make([]int, len(kinds))
	for i := range wantOrder {
		wantOrder[i] = i
	}
	if !reflect.DeepEqual(order, wantOrder) || !reflect.DeepEqual(sizes, wantSizes) || p.coalesceRunning || len(p.coalesceReady) != 0 {
		t.Fatalf("FIFO order=%v sizes=%v idle=%t", order, sizes, !p.coalesceRunning)
	}
}

// fak-test:runtime fast est=30ms lane=default
func TestInKernelV41CoordinatorFatalHandsOffQueue(t *testing.T) {
	m := model.NewSynthetic(tinyCfg())
	release := make(chan struct{})
	var once sync.Once
	open := func() { once.Do(func() { close(release) }) }
	defer open()
	p := &InKernelPlanner{m: m, coalesceReadyHook: func() { <-release }}
	primary := &struct{ value int }{17}
	calls := 0
	p.coalesceBatchHook = func(int) {
		calls++
		if calls == 1 {
			panic(primary)
		}
	}
	var closed atomic.Int32
	run := func(ctx context.Context) (inKernelGenerateResult, error) {
		lane, _ := newDecodeLaneOver(m, []int{1, 2}, 1, 0)
		defer func() { lane.s.Close(); closed.Add(1) }()
		_, err := coalescedDecode(ctx, lane)
		return inKernelGenerateResult{gen: lane.gen}, err
	}
	type recovered struct {
		value    any
		gateFree bool
	}
	leader := make(chan recovered, 1)
	go func() {
		defer func() {
			got := recover()
			p.devMu.init()
			gateFree := false
			select {
			case p.devMu.slot <- struct{}{}:
				gateFree = true
				p.devMu.Unlock()
			default:
			}
			leader <- recovered{value: got, gateFree: gateFree}
		}()
		_, _ = p.runCoalescedGenerateKind(context.Background(), run, inKernelCoalesceV41)
	}()
	awaitCoalescedQueue(t, p, 1)
	follower := v41CoordinatorRequest(context.Background(), run)
	follower.kind = inKernelCoalesceQwen
	follower.enqueued = time.Now()
	p.coalesceMu.Lock()
	p.coalesceReady = append(p.coalesceReady, follower)
	p.coalesceMu.Unlock()
	open()
	var got recovered
	select {
	case got = <-leader:
		if got.value != primary || !got.gateFree || closed.Load() < 1 {
			t.Fatal("leader escaped before cleanup or changed fatal identity")
		}
	case <-time.After(time.Second):
		t.Fatal("fatal leader stranded")
	}
	select {
	case <-follower.drain:
	case <-time.After(time.Second):
		t.Fatal("fatal coordinator did not hand off queued Qwen baton")
	}
	p.drainCoalescedGenerates()
	if _, err := finishCoalescedRequest(follower, <-follower.result); err != nil {
		t.Fatal(err)
	}
	if closed.Load() != 2 || calls != 2 || p.coalesceRunning || len(p.coalesceReady) != 0 {
		t.Fatal("fatal Flash cohort failed to hand off the queued Qwen cohort")
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestInKernelV41CoordinatorPreparationPanic(t *testing.T) {
	m := model.NewSynthetic(tinyCfg())
	p := &InKernelPlanner{m: m}
	lane, _ := newDecodeLaneOver(m, []int{1, 2}, 2, 0)
	primary := &struct{ stage string }{"prepare"}
	var closed, unstarted atomic.Int32
	requests := []*inKernelCoalesceRequest{
		v41CoordinatorRequest(context.Background(), func(ctx context.Context) (inKernelGenerateResult, error) {
			defer func() { lane.s.Close(); closed.Add(1) }()
			_, err := coalescedDecode(ctx, lane)
			return inKernelGenerateResult{}, err
		}),
		v41CoordinatorRequest(context.Background(), func(context.Context) (inKernelGenerateResult, error) {
			s := m.NewSession()
			defer func() { s.Close(); closed.Add(1) }()
			panic(primary)
		}),
		v41CoordinatorRequest(context.Background(), func(context.Context) (inKernelGenerateResult, error) {
			unstarted.Add(1)
			return inKernelGenerateResult{}, nil
		}),
	}
	var got any
	func() {
		defer func() { got = recover() }()
		p.runV41DecodeCohort(requests, func(context.Context, []*decodeLane, *model.Model) (inKernelV41DecodeReceipt, error) {
			t.Fatal("fatal preparation reached decode")
			return inKernelV41DecodeReceipt{}, nil
		})
	}()
	if got != primary || closed.Load() != 2 || unstarted.Load() != 0 || lane.gen != 0 {
		t.Fatal("preparation panic changed identity, skipped cleanup or started later work")
	}
	for _, req := range requests {
		<-req.done
		<-req.receiptReady
		if out := <-req.result; !errors.Is(out.err, errInKernelV41CohortAborted) {
			t.Fatalf("fatal preparation did not terminate every request: %v", out.err)
		}
	}
}

type v41CoordinatorOwnerProbe struct {
	compute.Backend
	owner *sync.Mutex
}

func (b *v41CoordinatorOwnerProbe) ForwardExecutionOwner() *sync.Mutex { return b.owner }

// No positive capability is advertised or execution initialized by these cases.
// fak-test:runtime fast est=5ms lane=default
func TestInKernelV41CoordinatorAdvisoryRefusal(t *testing.T) {
	m := &model.Model{Cfg: model.Config{ModelType: "deepseek_v41"}}
	for _, p := range []*InKernelPlanner{
		nil,
		{m: m, backend: compute.Default()},
		{m: m, batchDecode: true, backend: compute.Default()},
		{m: m, batchDecode: true, backend: &v41CoordinatorOwnerProbe{Backend: compute.Default()}},
		{m: m, batchDecode: true, backend: &v41CoordinatorOwnerProbe{Backend: compute.Default(), owner: &sync.Mutex{}}},
	} {
		_, err := p.runV41CoalescedGenerate(context.Background(), func(context.Context) (inKernelGenerateResult, error) {
			t.Fatal("missing prerequisites started a request")
			return inKernelGenerateResult{}, nil
		})
		if !errors.Is(err, model.ErrV41BatchUnavailable) {
			t.Fatalf("advisory refusal = %v", err)
		}
	}
}

// A selected lane error is not proof that a later panic came from cleanup.
// fak-test:runtime fast est=20ms lane=default
func TestInKernelV41CoordinatorPostDecodePanic(t *testing.T) {
	lanes, _ := v41DriverLanes(2, 2)
	m := model.NewSynthetic(tinyCfg())
	p := &InKernelPlanner{m: m}
	fault := errors.New("selected lane fault")
	primary := &struct{ stage string }{"unrelated post-decode work"}
	var closed atomic.Int32
	requests := make([]*inKernelCoalesceRequest, len(lanes))
	for i, lane := range lanes {
		lane.s = m.NewSession()
		requests[i] = v41CoordinatorRequest(context.Background(), func(ctx context.Context) (inKernelGenerateResult, error) {
			defer func() { lane.s.Close(); closed.Add(1) }()
			_, err := coalescedDecode(ctx, lane)
			if i == 0 {
				panic(primary)
			}
			return inKernelGenerateResult{}, err
		})
	}
	var got any
	func() {
		defer func() { got = recover() }()
		p.runV41DecodeCohort(requests, func(ctx context.Context, lanes []*decodeLane, _ *model.Model) (inKernelV41DecodeReceipt, error) {
			return inKernelDecodeV41LanesWithOps(ctx, lanes, inKernelV41DecodeOps{
				batch: func(input []model.V41BatchLane) (model.V41BatchResult, error) {
					out := v41DriverRefusal(input)
					out.Selected = true
					for i := range out.Lanes {
						out.Lanes[i].Err = fault
					}
					return out, fault
				},
				serial: func(*model.Session, int) ([]float32, error) { panic("selected replay") },
				wait:   func(context.Context) error { panic("selected retry") },
			})
		})
	}()
	if got != primary || closed.Load() != 2 {
		t.Fatalf("lane error suppressed unknown panic or escaped cleanup: %v closed=%d", got, closed.Load())
	}
	for _, req := range requests {
		<-req.done
		<-req.receiptReady
		if out := <-req.result; out.err != fault {
			t.Fatalf("unknown panic erased per-lane result: %v", out.err)
		}
	}
}
