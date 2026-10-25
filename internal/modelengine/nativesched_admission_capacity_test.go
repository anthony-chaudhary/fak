package modelengine

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
)

// fak-test:runtime medium est=8s lane=default
func TestEngineWaitingAdmissionDoesNotMaterializeKV(test *testing.T) {
	test.Run("impossible_prompt_is_rejected_before_session_allocation", func(test *testing.T) {
		engine := New()
		scheduler := engine.nativeScheduler()
		scheduler.SetMaxRunning(1)
		scheduler.SetKVPreemptionPolicy(NativePreemptionPolicy{
			Mode:        NativePreemptRecompute,
			MaxBlocks:   1,
			BlockTokens: 4,
		})
		scheduler.prepare = nativeSchedulerPrefillPrepare(map[string][]int{
			"capacity-impossible": {1, 2, 3, 4, 5},
			"capacity-exact":      {1, 2, 3, 4},
			"capacity-empty":      {},
		})
		var freshSessions atomic.Int64
		scheduler.SetSessionProfilerFactory(func(lifecycle NativeSessionLifecycle) *model.PhaseProfiler {
			if lifecycle == NativeSessionFresh {
				freshSessions.Add(1)
			}
			return model.NewPhaseProfiler()
		})
		test.Cleanup(func() {
			closeContext, cancelClose := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancelClose()
			if err := scheduler.CloseAndWait(closeContext); err != nil {
				test.Errorf("scheduler teardown: %v", err)
			}
		})

		request, err := engine.Admit(context.Background(), inlineCall("capacity-impossible", "{}"))
		if request != nil {
			test.Fatalf("impossible admission request = %T, want nil", request)
		}
		if !errors.Is(err, ErrNativePromptExceedsKVCapacity) {
			test.Fatalf("impossible admission error = %v, want ErrNativePromptExceedsKVCapacity", err)
		}
		if got := freshSessions.Load(); got != 0 {
			test.Fatalf("fresh sessions after impossible admission = %d, want 0", got)
		}

		for _, tool := range []string{"capacity-exact", "capacity-empty"} {
			requestContext, cancelRequest := context.WithCancel(context.Background())
			validRequest := admitCapacityRequest(test, engine, requestContext, tool)
			awaitCapacityToken(test, validRequest)
			cancelRequest()
			drainCapacityRequest(test, validRequest, context.Canceled)
		}
		if got := freshSessions.Load(); got != 2 {
			test.Fatalf("fresh sessions after exact and normalized-empty admissions = %d, want 2", got)
		}
	})

	test.Run("live_budget_shrink_rechecks_before_publish", func(test *testing.T) {
		engine := New()
		scheduler := engine.nativeScheduler()
		scheduler.SetMaxRunning(1)
		scheduler.SetKVPreemptionPolicy(NativePreemptionPolicy{Mode: NativePreemptRecompute, MaxBlocks: 2, BlockTokens: 4})
		scheduler.prepare = nativeSchedulerPrefillPrepare(map[string][]int{
			"capacity-shrink": {1, 2, 3, 4, 5},
		})
		materializing := make(chan struct{})
		release := make(chan struct{})
		scheduler.SetSessionProfilerFactory(func(NativeSessionLifecycle) *model.PhaseProfiler {
			close(materializing)
			<-release
			return model.NewPhaseProfiler()
		})
		var closedSessions atomic.Int64
		scheduler.closeSession = func(session *model.Session) {
			closedSessions.Add(1)
			session.Close()
		}

		type admissionResult struct {
			request abi.EngineRequest
			err     error
		}
		admitted := make(chan admissionResult, 1)
		go func() {
			request, err := engine.Admit(context.Background(), inlineCall("capacity-shrink", "{}"))
			admitted <- admissionResult{request: request, err: err}
		}()
		<-materializing
		scheduler.SetKVPreemptionPolicy(NativePreemptionPolicy{Mode: NativePreemptRecompute, MaxBlocks: 1, BlockTokens: 4})
		close(release)
		result := <-admitted
		if result.request != nil || !errors.Is(result.err, ErrNativePromptExceedsKVCapacity) {
			test.Fatalf("shrink admission request=%T error=%v, want nil request and capacity sentinel", result.request, result.err)
		}
		if got := closedSessions.Load(); got != 1 {
			test.Fatalf("closed sessions after shrink = %d, want 1", got)
		}
	})

	test.Run("impossible_prompt_is_rejected_while_state_slot_is_occupied", func(test *testing.T) {
		engine := New()
		scheduler := engine.nativeScheduler()
		scheduler.SetMaxRunning(1)
		scheduler.SetKVPreemptionPolicy(NativePreemptionPolicy{Mode: NativePreemptRecompute, MaxBlocks: 1, BlockTokens: 4})
		scheduler.prepare = nativeSchedulerPrefillPrepare(map[string][]int{
			"capacity-slot-active":     {1},
			"capacity-slot-impossible": {1, 2, 3, 4, 5},
		})
		var freshSessions atomic.Int64
		scheduler.SetSessionProfilerFactory(func(lifecycle NativeSessionLifecycle) *model.PhaseProfiler {
			if lifecycle == NativeSessionFresh {
				freshSessions.Add(1)
			}
			return model.NewPhaseProfiler()
		})
		test.Cleanup(func() {
			closeContext, cancelClose := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancelClose()
			if err := scheduler.CloseAndWait(closeContext); err != nil {
				test.Errorf("scheduler teardown: %v", err)
			}
		})

		activeContext, cancelActive := context.WithCancel(context.Background())
		activeRequest := admitCapacityRequest(test, engine, activeContext, "capacity-slot-active")
		awaitCapacityToken(test, activeRequest)
		impossibleRequest, err := engine.Admit(context.Background(), inlineCall("capacity-slot-impossible", "{}"))
		if impossibleRequest != nil || !errors.Is(err, ErrNativePromptExceedsKVCapacity) {
			test.Fatalf("occupied-slot impossible request=%T error=%v, want nil request and capacity sentinel", impossibleRequest, err)
		}
		if got := freshSessions.Load(); got != 1 {
			test.Fatalf("fresh sessions after occupied-slot rejection = %d, want 1", got)
		}
		cancelActive()
		drainCapacityRequest(test, activeRequest, context.Canceled)
	})

	test.Run("transient_kv_pressure_defers_with_spare_bounded_running_slot", func(test *testing.T) {
		fixture := newEngineAdmissionCapacityFixture(test, true)
		fixture.scheduler.SetMaxRunning(2)
		runContext, cancelRun := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancelRun()

		activeRequest := admitCapacityRequest(test, fixture.engine, runContext, "capacity-kv-active")
		awaitCapacityToken(test, activeRequest)
		waitingRequest := admitCapacityRequest(test, fixture.engine, runContext, "capacity-kv-waiting")
		assertCapacityMaterialization(test, fixture, 1, 1)
		assertCapacityNoToken(test, waitingRequest)

		drainCapacityRequest(test, activeRequest, nil)
		awaitCapacityToken(test, waitingRequest)
		assertCapacityMaterialization(test, fixture, 2, 2)
		drainCapacityRequest(test, waitingRequest, nil)
	})

	test.Run("bounded_swap_preserves_deliberate_preemption_overlap", func(test *testing.T) {
		fixture := newEngineAdmissionCapacityFixture(test, true)
		fixture.scheduler.SetMaxRunning(2)
		fixture.scheduler.SetKVPreemptionPolicy(NativePreemptionPolicy{
			Mode:        NativePreemptSwap,
			MaxBlocks:   1,
			BlockTokens: 128,
		})
		runContext, cancelRun := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancelRun()

		requests := []abi.EngineRequest{
			admitCapacityRequest(test, fixture.engine, runContext, "capacity-swap-first"),
			admitCapacityRequest(test, fixture.engine, runContext, "capacity-swap-second"),
		}
		assertCapacityMaterialization(test, fixture, 2, 2)
		var drains sync.WaitGroup
		for _, request := range requests {
			drains.Add(1)
			go func() {
				defer drains.Done()
				for range request.Tokens() {
				}
			}()
		}
		drains.Wait()
		for _, request := range requests {
			if _, err := request.Result(); err != nil {
				test.Fatalf("swap request result: %v", err)
			}
		}
		stats := fixture.scheduler.KVPreemptionStats()
		if stats.Preemptions == 0 || stats.SwapPreemptions == 0 || stats.Readmitted == 0 {
			test.Fatalf("swap compatibility stats preemptions=%d swaps=%d readmitted=%d, want each >0", stats.Preemptions, stats.SwapPreemptions, stats.Readmitted)
		}
	})

	test.Run("promotion_fills_multi_lane_kv_capacity_once_per_reservation", func(test *testing.T) {
		fixture := newEngineAdmissionCapacityFixture(test, true)
		fixture.scheduler.SetMaxRunning(3)
		runContext, cancelRun := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancelRun()

		activeContext, cancelActive := context.WithCancel(runContext)
		activeRequest := admitCapacityRequest(test, fixture.engine, activeContext, "capacity-fill-active")
		awaitCapacityToken(test, activeRequest)
		waitingRequests := make([]abi.EngineRequest, 0, 3)
		waitingCancels := make([]context.CancelFunc, 0, 3)
		for index := 0; index < 3; index++ {
			waitingContext, cancelWaiting := context.WithCancel(runContext)
			waitingCancels = append(waitingCancels, cancelWaiting)
			waitingRequests = append(waitingRequests, admitCapacityRequest(test, fixture.engine, waitingContext, "capacity-fill-waiting"))
		}
		assertCapacityMaterialization(test, fixture, 1, 1)

		fixture.scheduler.SetKVPreemptionPolicy(NativePreemptionPolicy{
			Mode:        NativePreemptRecompute,
			MaxBlocks:   3,
			BlockTokens: 128,
		})
		cancelActive()
		drainCapacityRequest(test, activeRequest, context.Canceled)
		deadline := time.Now().Add(5 * time.Second)
		for (fixture.freshSessions.Load() != 4 || capacityPrefillObservations(fixture.scheduler) != 4) && time.Now().Before(deadline) {
			time.Sleep(5 * time.Millisecond)
		}
		assertCapacityMaterialization(test, fixture, 4, 4)

		for _, cancelWaiting := range waitingCancels {
			cancelWaiting()
		}
		for _, waitingRequest := range waitingRequests {
			drainCapacityRequest(test, waitingRequest, context.Canceled)
		}
	})

	test.Run("canceled_waiter_allocates_nothing", func(test *testing.T) {
		fixture := newEngineAdmissionCapacityFixture(test, true)
		runContext, cancelRun := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancelRun()

		activeRequest := admitCapacityRequest(test, fixture.engine, runContext, "capacity-active")
		awaitCapacityToken(test, activeRequest)
		assertCapacityMaterialization(test, fixture, 1, 1)

		waitContext, cancelWait := context.WithCancel(runContext)
		waitingRequest := admitCapacityRequest(test, fixture.engine, waitContext, "capacity-cancel")
		assertCapacityMaterialization(test, fixture, 1, 1)
		assertCapacityNoToken(test, waitingRequest)

		cancelWait()
		drainCapacityRequest(test, activeRequest, nil)
		drainCapacityRequest(test, waitingRequest, context.Canceled)
		assertCapacityMaterialization(test, fixture, 1, 1)
		assertCapacityPressure(test, fixture.scheduler)
	})

	test.Run("promotion_materializes_once", func(test *testing.T) {
		fixture := newEngineAdmissionCapacityFixture(test, true)
		runContext, cancelRun := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancelRun()

		activeRequest := admitCapacityRequest(test, fixture.engine, runContext, "capacity-first")
		awaitCapacityToken(test, activeRequest)
		waitingRequest := admitCapacityRequest(test, fixture.engine, runContext, "capacity-promote")
		assertCapacityMaterialization(test, fixture, 1, 1)
		assertCapacityNoToken(test, waitingRequest)

		drainCapacityRequest(test, activeRequest, nil)
		awaitCapacityToken(test, waitingRequest)
		assertCapacityMaterialization(test, fixture, 2, 2)
		drainCapacityRequest(test, waitingRequest, nil)
		assertCapacityMaterialization(test, fixture, 2, 2)
		assertCapacityPressure(test, fixture.scheduler)
	})

	test.Run("materialization_failure_is_deferred_to_result", func(test *testing.T) {
		fixture := newEngineAdmissionCapacityFixture(test, true)
		runContext, cancelRun := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancelRun()

		activeRequest := admitCapacityRequest(test, fixture.engine, runContext, "capacity-before-error")
		awaitCapacityToken(test, activeRequest)
		fixture.scheduler.EnableQwen35MetalStateIdentityCapture()

		waitingRequest, err := fixture.engine.Admit(runContext, inlineCall("capacity-error", "{}"))
		if err != nil {
			test.Fatalf("waiting Admit returned materialization error early: %v", err)
		}
		if waitingRequest == nil {
			test.Fatal("waiting Admit returned a nil request")
		}
		assertCapacityMaterialization(test, fixture, 1, 1)

		drainCapacityRequest(test, activeRequest, nil)
		drainCapacityRequest(test, waitingRequest, &model.Qwen35MetalStateIdentityUnavailableError{})
		if got := fixture.freshSessions.Load(); got != 2 {
			test.Fatalf("fresh sessions after failed promotion = %d, want 2", got)
		}
		if got := capacityPrefillObservations(fixture.scheduler); got != 1 {
			test.Fatalf("prefill observations after failed promotion = %d, want 1", got)
		}
		assertCapacityPressure(test, fixture.scheduler)
	})

	test.Run("failed_materializer_releases_orphaned_waiter", func(test *testing.T) {
		fixture := newEngineAdmissionCapacityFixture(test, true)
		runContext, cancelRun := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancelRun()

		firstFreshEntered := make(chan struct{})
		releaseFirstFresh := make(chan struct{})
		var firstFresh atomic.Bool
		var releaseOnce sync.Once
		defer releaseOnce.Do(func() { close(releaseFirstFresh) })
		fixture.scheduler.SetSessionProfilerFactory(func(lifecycle NativeSessionLifecycle) *model.PhaseProfiler {
			if lifecycle == NativeSessionFresh {
				fixture.freshSessions.Add(1)
				if firstFresh.CompareAndSwap(false, true) {
					close(firstFreshEntered)
					<-releaseFirstFresh
				}
			}
			return model.NewPhaseProfiler()
		})
		fixture.scheduler.EnableQwen35MetalStateIdentityCapture()

		type admissionResult struct {
			request abi.EngineRequest
			err     error
		}
		firstAdmitted := make(chan admissionResult, 1)
		go func() {
			request, err := fixture.engine.Admit(runContext, inlineCall("capacity-failing-materializer", "{}"))
			firstAdmitted <- admissionResult{request: request, err: err}
		}()
		select {
		case <-firstFreshEntered:
		case <-time.After(2 * time.Second):
			test.Fatal("first fresh session did not enter materialization")
		}

		waitingAdmitted := make(chan admissionResult, 1)
		go func() {
			request, err := fixture.engine.Admit(runContext, inlineCall("capacity-orphaned-waiter", "{}"))
			waitingAdmitted <- admissionResult{request: request, err: err}
		}()
		var waitingRequest abi.EngineRequest
		select {
		case result := <-waitingAdmitted:
			if result.err != nil {
				test.Fatalf("waiting Admit: %v", result.err)
			}
			if result.request == nil {
				test.Fatal("waiting Admit returned a nil request")
			}
			waitingRequest = result.request
		case <-time.After(2 * time.Second):
			test.Fatal("waiting Admit blocked behind materialization")
		}

		releaseOnce.Do(func() { close(releaseFirstFresh) })
		select {
		case result := <-firstAdmitted:
			var unavailable *model.Qwen35MetalStateIdentityUnavailableError
			if !errors.As(result.err, &unavailable) {
				test.Fatalf("first Admit error = %v, want Qwen35MetalStateIdentityUnavailableError", result.err)
			}
		case <-time.After(5 * time.Second):
			test.Fatal("first Admit did not finish after materialization release")
		}

		streamDone := make(chan int, 1)
		go func() {
			tokenCount := 0
			for range waitingRequest.Tokens() {
				tokenCount++
			}
			streamDone <- tokenCount
		}()
		select {
		case tokenCount := <-streamDone:
			if tokenCount != 0 {
				test.Fatalf("waiting request emitted %d tokens, want 0", tokenCount)
			}
		case <-time.After(5 * time.Second):
			test.Fatal("waiting request token stream did not close")
		}

		type resultOutcome struct {
			result *abi.Result
			err    error
		}
		resultDone := make(chan resultOutcome, 1)
		go func() {
			result, err := waitingRequest.Result()
			resultDone <- resultOutcome{result: result, err: err}
		}()
		select {
		case outcome := <-resultDone:
			var unavailable *model.Qwen35MetalStateIdentityUnavailableError
			if !errors.As(outcome.err, &unavailable) {
				test.Fatalf("waiting Result error = %v, want Qwen35MetalStateIdentityUnavailableError", outcome.err)
			}
			if outcome.result != nil {
				test.Fatalf("waiting Result = %+v, want nil", outcome.result)
			}
		case <-time.After(5 * time.Second):
			test.Fatal("waiting Result did not finish")
		}
	})

	test.Run("policy_disabled_keeps_synchronous_admission", func(test *testing.T) {
		fixture := newEngineAdmissionCapacityFixture(test, false)
		runContext, cancelRun := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancelRun()

		activeRequest := admitCapacityRequest(test, fixture.engine, runContext, "capacity-legacy-active")
		awaitCapacityToken(test, activeRequest)
		waitContext, cancelWait := context.WithCancel(runContext)
		waitingRequest := admitCapacityRequest(test, fixture.engine, waitContext, "capacity-legacy-waiter")
		assertCapacityMaterialization(test, fixture, 2, 2)

		cancelWait()
		drainCapacityRequest(test, activeRequest, nil)
		drainCapacityRequest(test, waitingRequest, context.Canceled)
	})

	test.Run("spare_capacity_admission_stays_live_under_backpressure", func(test *testing.T) {
		fixture := newEngineAdmissionCapacityFixture(test, true)
		fixture.scheduler.SetMaxRunning(2)
		fixture.scheduler.SetKVPreemptionPolicy(NativePreemptionPolicy{
			Mode:        NativePreemptRecompute,
			MaxBlocks:   2,
			BlockTokens: 128,
		})
		runContext, cancelRun := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancelRun()

		activeContext, cancelActive := context.WithCancel(runContext)
		activeRequest := admitCapacityRequest(test, fixture.engine, activeContext, "capacity-live-active")
		awaitCapacityToken(test, activeRequest)

		secondContext, cancelSecond := context.WithCancel(runContext)
		type admissionResult struct {
			request abi.EngineRequest
			err     error
		}
		admitted := make(chan admissionResult, 1)
		go func() {
			request, err := fixture.engine.Admit(secondContext, inlineCall("capacity-live-second", "{}"))
			admitted <- admissionResult{request: request, err: err}
		}()

		var secondRequest abi.EngineRequest
		select {
		case result := <-admitted:
			if result.err != nil {
				test.Fatalf("second Admit: %v", result.err)
			}
			secondRequest = result.request
		case <-time.After(2 * time.Second):
			cancelActive()
			cancelSecond()
			drainCapacityRequest(test, activeRequest, context.Canceled)
			test.Fatal("second Admit blocked despite spare capacity")
		}
		assertCapacityMaterialization(test, fixture, 2, 2)

		cancelActive()
		cancelSecond()
		drainCapacityRequest(test, activeRequest, context.Canceled)
		drainCapacityRequest(test, secondRequest, context.Canceled)
		if peak := fixture.scheduler.MaxObservedKVBlocks(); peak > 2 {
			test.Fatalf("peak resident KV blocks = %d, want at most 2", peak)
		}
	})

	test.Run("oversized_prefill_waits_for_empty_token_budget", func(test *testing.T) {
		prefillModel := nativeSchedulerPrefillModel(test)
		engine := New()
		engine.PreloadQ4K(prefillModel)
		scheduler := engine.nativeScheduler()
		scheduler.prepare = nativeSchedulerPrefillPrepare(map[string][]int{
			"capacity-budget-a":    nativeSchedulerQwenPrompt(8),
			"capacity-budget-b":    nativeSchedulerQwenPrompt(8),
			"capacity-budget-long": nativeSchedulerQwenPrompt(33),
		})
		scheduler.SetMaxRunning(2)
		scheduler.SetMaxTokensPerStep(2)
		if err := scheduler.SetQwenPrefillMaxTokensPerIteration(16); err != nil {
			test.Fatalf("SetQwenPrefillMaxTokensPerIteration: %v", err)
		}
		scheduler.SetKVPreemptionPolicy(NativePreemptionPolicy{
			Mode:        NativePreemptRecompute,
			MaxBlocks:   32,
			BlockTokens: 16,
		})
		var freshSessions atomic.Int64
		scheduler.SetSessionProfilerFactory(func(lifecycle NativeSessionLifecycle) *model.PhaseProfiler {
			if lifecycle == NativeSessionFresh {
				freshSessions.Add(1)
			}
			return model.NewPhaseProfiler()
		})
		test.Cleanup(func() {
			closeContext, cancelClose := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancelClose()
			if err := scheduler.CloseAndWait(closeContext); err != nil {
				test.Errorf("scheduler teardown: %v", err)
			}
		})

		runContext, cancelRun := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancelRun()
		firstContext, cancelFirst := context.WithCancel(runContext)
		secondContext, cancelSecond := context.WithCancel(runContext)
		longContext, cancelLong := context.WithCancel(runContext)
		defer cancelFirst()
		defer cancelSecond()
		defer cancelLong()

		firstRequest := admitCapacityRequest(test, engine, firstContext, "capacity-budget-a")
		awaitCapacityToken(test, firstRequest)
		secondRequest := admitCapacityRequest(test, engine, secondContext, "capacity-budget-b")
		longRequest := admitCapacityRequest(test, engine, longContext, "capacity-budget-long")
		if got := freshSessions.Load(); got != 2 {
			test.Fatalf("fresh sessions before capacity release = %d, want 2", got)
		}
		assertCapacityNoToken(test, longRequest)

		cancelFirst()
		drainCapacityRequest(test, firstRequest, context.Canceled)
		awaitCapacityToken(test, secondRequest)
		longLane, ok := longRequest.(*schedLane)
		if !ok {
			test.Fatalf("long request type = %T, want *schedLane", longRequest)
		}
		secondLane, ok := secondRequest.(*schedLane)
		if !ok {
			test.Fatalf("second request type = %T, want *schedLane", secondRequest)
		}
		scheduler.mu.Lock()
		longWaiting := false
		for _, waitingLane := range scheduler.waiting {
			if waitingLane == longLane {
				longWaiting = true
				break
			}
		}
		longSessionless := longLane.sess == nil
		secondDecodeActive := !secondLane.terminal && secondLane.state == schedLaneDecode
		scheduler.mu.Unlock()
		if !longWaiting || !longSessionless || !secondDecodeActive {
			test.Fatalf("budget state waiting=%t sessionless=%t decode_active=%t, want true/true/true", longWaiting, longSessionless, secondDecodeActive)
		}
		if got := freshSessions.Load(); got != 2 {
			test.Fatalf("fresh sessions beside active decode = %d, want 2", got)
		}

		cancelSecond()
		drainCapacityRequest(test, secondRequest, context.Canceled)
		awaitCapacityToken(test, longRequest)
		if got := freshSessions.Load(); got != 3 {
			test.Fatalf("fresh sessions after empty-budget promotion = %d, want 3", got)
		}
		cancelLong()
		drainCapacityRequest(test, longRequest, context.Canceled)
	})

	test.Run("concurrent_waiters_share_the_capacity_gate", func(test *testing.T) {
		fixture := newEngineAdmissionCapacityFixture(test, true)
		runContext, cancelRun := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancelRun()

		activeRequest := admitCapacityRequest(test, fixture.engine, runContext, "capacity-concurrent-active")
		awaitCapacityToken(test, activeRequest)

		type admissionResult struct {
			request abi.EngineRequest
			err     error
		}
		results := make(chan admissionResult, 2)
		waitCancels := make([]context.CancelFunc, 0, 2)
		var admitGroup sync.WaitGroup
		for index := 0; index < 2; index++ {
			waitContext, cancelWait := context.WithCancel(runContext)
			waitCancels = append(waitCancels, cancelWait)
			admitGroup.Add(1)
			go func(tool string, requestContext context.Context) {
				defer admitGroup.Done()
				request, err := fixture.engine.Admit(requestContext, inlineCall(tool, "{}"))
				results <- admissionResult{request: request, err: err}
			}("capacity-concurrent-"+string(rune('a'+index)), waitContext)
		}
		admitGroup.Wait()
		close(results)

		waitingRequests := make([]abi.EngineRequest, 0, 2)
		for result := range results {
			if result.err != nil {
				test.Fatalf("concurrent waiting Admit: %v", result.err)
			}
			waitingRequests = append(waitingRequests, result.request)
		}
		assertCapacityMaterialization(test, fixture, 1, 1)
		for _, waitingRequest := range waitingRequests {
			assertCapacityNoToken(test, waitingRequest)
		}

		for _, cancelWait := range waitCancels {
			cancelWait()
		}
		drainCapacityRequest(test, activeRequest, nil)
		for _, waitingRequest := range waitingRequests {
			drainCapacityRequest(test, waitingRequest, context.Canceled)
		}
		assertCapacityMaterialization(test, fixture, 1, 1)
		assertCapacityPressure(test, fixture.scheduler)
	})
}

type engineAdmissionCapacityFixture struct {
	engine        *Engine
	scheduler     *NativeScheduler
	freshSessions atomic.Int64
}

func newEngineAdmissionCapacityFixture(test *testing.T, enablePolicy bool) *engineAdmissionCapacityFixture {
	test.Helper()
	engine := New()
	scheduler := engine.nativeScheduler()
	fixture := &engineAdmissionCapacityFixture{engine: engine, scheduler: scheduler}
	scheduler.SetMaxRunning(1)
	scheduler.SetSessionProfilerFactory(func(lifecycle NativeSessionLifecycle) *model.PhaseProfiler {
		if lifecycle == NativeSessionFresh {
			fixture.freshSessions.Add(1)
		}
		return model.NewPhaseProfiler()
	})
	if enablePolicy {
		scheduler.SetKVPreemptionPolicy(NativePreemptionPolicy{
			Mode:        NativePreemptRecompute,
			MaxBlocks:   1,
			BlockTokens: 128,
		})
	}
	test.Cleanup(func() {
		closeContext, cancelClose := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancelClose()
		if err := scheduler.CloseAndWait(closeContext); err != nil {
			test.Errorf("scheduler teardown: %v", err)
		}
	})
	return fixture
}

func admitCapacityRequest(test *testing.T, engine *Engine, requestContext context.Context, tool string) abi.EngineRequest {
	test.Helper()
	request, err := engine.Admit(requestContext, inlineCall(tool, "{}"))
	if err != nil {
		test.Fatalf("Admit %q: %v", tool, err)
	}
	return request
}

func awaitCapacityToken(test *testing.T, request abi.EngineRequest) abi.EngineToken {
	test.Helper()
	select {
	case token, open := <-request.Tokens():
		if !open {
			test.Fatal("token stream closed before first token")
		}
		return token
	case <-time.After(5 * time.Second):
		test.Fatal("timed out waiting for first token")
		return abi.EngineToken{}
	}
}

func assertCapacityNoToken(test *testing.T, request abi.EngineRequest) {
	test.Helper()
	select {
	case token, open := <-request.Tokens():
		test.Fatalf("waiting request token stream changed before capacity release: token=%+v open=%t", token, open)
	default:
	}
}

func drainCapacityRequest(test *testing.T, request abi.EngineRequest, wantErr error) {
	test.Helper()
	drained := make(chan struct{})
	go func() {
		for range request.Tokens() {
		}
		close(drained)
	}()
	select {
	case <-drained:
	case <-time.After(5 * time.Second):
		test.Fatal("timed out draining request tokens")
	}
	result, err := request.Result()
	switch {
	case wantErr == nil && err != nil:
		test.Fatalf("Result error = %v, want nil", err)
	case wantErr != nil:
		if _, typed := wantErr.(*model.Qwen35MetalStateIdentityUnavailableError); typed {
			var unavailable *model.Qwen35MetalStateIdentityUnavailableError
			if !errors.As(err, &unavailable) {
				test.Fatalf("Result error = %v, want Qwen35MetalStateIdentityUnavailableError", err)
			}
		} else if !errors.Is(err, wantErr) {
			test.Fatalf("Result error = %v, want %v", err, wantErr)
		}
	}
	if wantErr == nil {
		if result == nil || result.Status != abi.StatusOK {
			test.Fatalf("Result = %+v, want StatusOK", result)
		}
	} else if result != nil {
		test.Fatalf("failed request Result = %+v, want nil", result)
	}
}

func assertCapacityMaterialization(test *testing.T, fixture *engineAdmissionCapacityFixture, wantSessions, wantPrefills int64) {
	test.Helper()
	if got := fixture.freshSessions.Load(); got != wantSessions {
		test.Fatalf("fresh sessions = %d, want %d", got, wantSessions)
	}
	if got := capacityPrefillObservations(fixture.scheduler); got != uint64(wantPrefills) {
		test.Fatalf("prefill observations = %d, want %d", got, wantPrefills)
	}
}

func capacityPrefillObservations(scheduler *NativeScheduler) uint64 {
	for _, phase := range scheduler.CachePhaseLatencyReceipt().Phases {
		if phase.Phase == modelperfobs.CachePipelinePhasePrefill {
			return phase.Observations
		}
	}
	return 0
}

func assertCapacityPressure(test *testing.T, scheduler *NativeScheduler) {
	test.Helper()
	if peak := scheduler.MaxObservedKVBlocks(); peak > 1 {
		test.Fatalf("peak resident KV blocks = %d, want at most 1", peak)
	}
	if stats := scheduler.KVPreemptionStats(); stats.Preemptions != 0 {
		test.Fatalf("preemption stats = %+v, want no preemption for capacity-gated waiters", stats)
	}
}
