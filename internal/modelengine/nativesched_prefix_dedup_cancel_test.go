package modelengine

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// Estimate only; this test has not been timed.
// fak-test:runtime medium est=2s lane=default
func TestNativeSchedulerCanceledPrefixFollowerDoesNotPrefillOrPublish(t *testing.T) {
	s := NewNativeScheduler(model.NewSynthetic(SyntheticConfig()))
	s.SetInBatchPrefixDedup(true)
	nativeSchedulerBeginManualDrain(t, s)
	defer nativeSchedulerEndManualDrain(s)
	prompt := nativeSchedulerQwenPrompt(40)
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	defer unblock()
	s.prefillFlightHook = func() { close(entered); <-release }
	type result struct {
		req abi.EngineRequest
		err error
	}
	leader, follower := make(chan result, 1), make(chan result, 1)
	go func() { r, err := s.AdmitTokenIDs(context.Background(), "leader", prompt); leader <- result{r, err} }()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("leader did not enter flight")
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() { r, err := s.AdmitTokenIDs(ctx, "follower", prompt); follower <- result{r, err} }()
	prefixDedupAwait(t, "follower joined", func() bool { return s.prefixFlightGroup().Coalesced() == 1 })
	cancel()
	select {
	case got := <-follower:
		if got.req != nil || got.err != context.Canceled {
			t.Fatalf("canceled admission = (%v, %v)", got.req, got.err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("canceled follower did not return while leader blocked")
	}
	if got := s.CachePhaseLatencyReceipt().Observations; got != 0 {
		t.Fatalf("canceled follower executed %d prefill observations", got)
	}
	s.mu.Lock()
	queued, sequence := len(s.waiting), s.seqNo
	s.mu.Unlock()
	if queued != 0 || sequence != 0 {
		t.Fatalf("canceled follower published work: queued=%d sequence=%d", queued, sequence)
	}
	unblock()
	select {
	case got := <-leader:
		if got.err != nil || got.req == nil {
			t.Fatalf("uncanceled leader = (%v, %v)", got.req, got.err)
		}
		lane := got.req.(*schedLane)
		if lane.sess.Cache.Len() != len(prompt) {
			t.Fatal("leader cache incomplete")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("leader did not finish")
	}
	if got := s.CachePhaseLatencyReceipt().Observations; got != 1 {
		t.Fatalf("prefill observations = %d, want only leader", got)
	}
}

// Estimate only; this test has not been timed.
// fak-test:runtime medium est=2s lane=default
func TestNativeSchedulerPrefixLeaderFailureStillAllowsLiveFollower(t *testing.T) {
	for _, failed := range []error{errors.New("injected upstream prefill failure"), context.Canceled} {
		t.Run(failed.Error(), func(t *testing.T) {
			s := NewNativeScheduler(model.NewSynthetic(SyntheticConfig()))
			s.SetInBatchPrefixDedup(true)
			nativeSchedulerBeginManualDrain(t, s)
			defer nativeSchedulerEndManualDrain(s)
			prompt := nativeSchedulerQwenPrompt(40)
			entered, release := make(chan struct{}), make(chan struct{})
			var once sync.Once
			unblock := func() { once.Do(func() { close(release) }) }
			defer unblock()
			leaderDone := make(chan error, 1)
			go func() {
				_, _, _, _, err := s.prefixFlightGroup().CoalesceSharedPrefixNS(context.Background(), "", prompt, nativeInBatchPrefixDedupMinShared,
					func(context.Context) (*model.KVCache, []float32, error) {
						close(entered)
						<-release
						return nil, nil, failed
					})
				leaderDone <- err
			}()
			select {
			case <-entered:
			case <-time.After(5 * time.Second):
				t.Fatal("upstream flight did not start")
			}
			type result struct {
				req abi.EngineRequest
				err error
			}
			admitted := make(chan result, 1)
			go func() {
				r, err := s.AdmitTokenIDs(context.Background(), "live-follower", prompt)
				admitted <- result{r, err}
			}()
			prefixDedupAwait(t, "live follower joined", func() bool { return s.prefixFlightGroup().Coalesced() == 1 })
			unblock()
			select {
			case err := <-leaderDone:
				if err != failed {
					t.Fatalf("leader error = %v", err)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("upstream leader stuck")
			}
			select {
			case got := <-admitted:
				if got.err != nil || got.req == nil {
					t.Fatalf("live follower = (%v, %v)", got.req, got.err)
				}
				if got.req.(*schedLane).sess.Cache.Len() != len(prompt) {
					t.Fatal("fallback cold prefill incomplete")
				}
			case <-time.After(5 * time.Second):
				t.Fatal("live follower did not recover")
			}
			if got := s.CachePhaseLatencyReceipt().Observations; got != 1 {
				t.Fatalf("cold prefill observations = %d, want 1", got)
			}
		})
	}
}

// Estimate only; this test has not been timed.
// fak-test:runtime fast est=200ms lane=default
func TestNativeSchedulerCanceledAdmissionPublicationBoundaries(t *testing.T) {
	for _, during := range []bool{false, true} {
		name := "before-session"
		if during {
			name = "during-synchronous-prefill"
		}
		t.Run(name, func(t *testing.T) {
			s := NewNativeScheduler(model.NewSynthetic(SyntheticConfig()))
			defer s.Close()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var wantPrefills uint64
			if during {
				// coldPrefillSync calls now immediately before the synchronous forward.
				// Cancellation cannot interrupt that work, but must prevent publication.
				s.now = func() time.Time { cancel(); return time.Now() }
				wantPrefills = 1
			} else {
				cancel()
			}
			req, err := s.AdmitTokenIDs(ctx, "canceled", []int{1})
			if req != nil || err != context.Canceled {
				t.Fatalf("admission = (%v, %v)", req, err)
			}
			if got := s.CachePhaseLatencyReceipt().Observations; got != wantPrefills {
				t.Fatalf("prefills = %d, want %d", got, wantPrefills)
			}
			s.mu.Lock()
			queued, sequence, started := len(s.waiting), s.seqNo, s.runStarted
			s.mu.Unlock()
			if queued != 0 || sequence != 0 || started {
				t.Fatalf("published canceled work: queued=%d sequence=%d started=%v", queued, sequence, started)
			}
		})
	}
}
