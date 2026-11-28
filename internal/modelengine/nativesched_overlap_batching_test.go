package modelengine

import (
	"context"
	"encoding/json"
	"reflect"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/refutil"
)

// fak-test:runtime fast est=2s lane=default
func TestNativeOverlapBatchingServeKnobs(t *testing.T) {
	clearServeKnobEnv(t)
	t.Setenv(envNativeOverlapScheduling, "")

	m := model.NewSynthetic(SyntheticConfig())
	e := New()
	e.Preload(m)
	sched := e.nativeScheduler()
	if !sched.OverlapScheduling() {
		t.Fatal("default Engine.nativeScheduler().OverlapScheduling() = false, want true")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	t.Cleanup(func() {
		sched.Close()
		wait, stop := context.WithTimeout(context.Background(), 2*time.Second)
		defer stop()
		if err := sched.CloseAndWait(wait); err != nil {
			t.Errorf("scheduler did not stop: %v", err)
		}
	})

	// Imported lane with GenBudget > genTokens sizes its outbox to the full budget.
	prompt := []int{2, 4, 6}
	sess := m.NewSession()
	logits := sess.Prefill(prompt)
	const customBudget = genTokens + 8
	req, err := sched.AdmitImported(ctx, inlineCall("overlap-budget", "{}"), ImportedSequence{
		Session:   sess,
		Logits:    logits,
		Prompt:    prompt,
		PromptLen: len(prompt),
		Tool:      "overlap-budget",
		GenBudget: customBudget,
	})
	if err != nil {
		sess.Close()
		t.Fatalf("AdmitImported: %v", err)
	}
	ln := req.(*schedLane)
	if got := cap(ln.tokens); got != customBudget {
		t.Fatalf("AdmitImported outbox capacity = %d, want %d", got, customBudget)
	}
	for range req.Tokens() {
	}
	if _, err := req.Result(); err != nil {
		t.Fatalf("AdmitImported Result: %v", err)
	}

	// Explicit opt-out via env var disables overlap scheduling and reverts outbox capacity to 1.
	t.Setenv(envNativeOverlapScheduling, "0")
	eOff := New()
	eOff.Preload(m)
	schedOff := eOff.nativeScheduler()
	t.Cleanup(func() {
		schedOff.Close()
		wait, stop := context.WithTimeout(context.Background(), 2*time.Second)
		defer stop()
		_ = schedOff.CloseAndWait(wait)
	})
	if schedOff.OverlapScheduling() {
		t.Fatalf("%s=0 OverlapScheduling() = true, want false", envNativeOverlapScheduling)
	}
	reqOff, err := eOff.Admit(ctx, inlineCall("overlap-off", "{}"))
	if err != nil {
		t.Fatalf("Admit(off): %v", err)
	}
	if got := cap(reqOff.(*schedLane).tokens); got != 1 {
		t.Fatalf("opt-out outbox capacity = %d, want 1", got)
	}
	for range reqOff.Tokens() {
	}
	if _, err := reqOff.Result(); err != nil {
		t.Fatalf("Result(off): %v", err)
	}
}

// fak-test:runtime fast est=2s lane=default
func TestNativeOverlapConcurrentBatchingC1To64(t *testing.T) {
	clearServeKnobEnv(t)
	t.Setenv(envNativeOverlapScheduling, "")

	for _, c := range []int{1, 16, 64} {
		t.Run("C="+strconv.Itoa(c), func(t *testing.T) {
			cfg := SyntheticConfig()
			cfg.HiddenSize = 8
			cfg.NumLayers = 1
			cfg.NumHeads = 2
			cfg.NumKVHeads = 1
			cfg.HeadDim = 4
			cfg.IntermediateSize = 16
			cfg.VocabSize = 32
			m := model.NewSynthetic(cfg)
			e := New()
			e.Preload(m)
			sched := e.nativeScheduler()
			sched.SetKVPreemptionPolicy(NativePreemptionPolicy{})
			sched.SetInBatchPrefixDedup(false)

			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			t.Cleanup(func() {
				sched.Close()
				wait, stop := context.WithTimeout(context.Background(), 5*time.Second)
				defer stop()
				if err := sched.CloseAndWait(wait); err != nil {
					t.Errorf("scheduler did not stop: %v", err)
				}
			})

			// Hold executor while admitting all C requests so they enter together
			// and exercise concurrent batching across the full C=1..64 envelope.
			sched.executor.Lock()
			var unlockOnce sync.Once
			release := func() { unlockOnce.Do(sched.executor.Unlock) }
			t.Cleanup(release)

			reqs := make([]abi.EngineRequest, c)
			wants := make([][]int, c)
			for i := 0; i < c; i++ {
				call := inlineCall("l", strconv.Itoa(i))
				req, err := e.Admit(ctx, call)
				if err != nil {
					t.Fatalf("Admit(%d): %v", i, err)
				}
				reqs[i] = req
				wants[i] = nativeOutputOracle(m, req.(*schedLane).prompt)
			}
			release()

			// Even-indexed lanes deliberately await Result() BEFORE draining Tokens()
			// to prove delayed consumers do not stall concurrent batching.
			var wg sync.WaitGroup
			errs := make(chan error, c)
			for i := 0; i < c; i++ {
				wg.Add(1)
				go func(idx int, req abi.EngineRequest) {
					defer wg.Done()
					if idx%2 == 1 {
						for range req.Tokens() {
						}
					}
					res, err := req.Result()
					if err != nil {
						errs <- err
						return
					}
					if idx%2 == 0 {
						var streamed []int
						for tok := range req.Tokens() {
							streamed = append(streamed, tok.ID)
						}
						if !reflect.DeepEqual(streamed, wants[idx]) {
							errs <- context.DeadlineExceeded
							return
						}
					}
					var payload struct {
						Tokens []int `json:"generated_tokens"`
					}
					if err := json.Unmarshal(refutil.Bytes(ctx, res.Payload), &payload); err != nil {
						errs <- err
						return
					}
					if !reflect.DeepEqual(payload.Tokens, wants[idx]) {
						errs <- errNativeSchedulerLaneNotDecodeReady
					}
				}(i, reqs[i])
			}
			wg.Wait()
			close(errs)
			for err := range errs {
				t.Fatalf("lane error: %v", err)
			}

			if got := sched.MaxObservedRunning(); got < c {
				t.Fatalf("MaxObservedRunning() = %d, want >= %d", got, c)
			}
			if c > 1 {
				receipt := sched.SharedWorkReceipt()
				if receipt.Steps == 0 || receipt.Panels == 0 || receipt.MACs == 0 {
					t.Fatalf("C=%d SharedWorkReceipt = %+v, want nonzero Steps/Panels/MACs", c, receipt)
				}
			}
		})
	}
}
