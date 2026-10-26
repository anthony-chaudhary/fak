package modelengine

import (
	"context"
	"errors"
	"reflect"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=2s lane=default
func TestNativeSchedulerDeferredTokenLifecycle(t *testing.T) {
	t.Run("forward_precedes_prior_token_publication", func(t *testing.T) {
		s, ln := deferredLifecycleFixture(t, []int{1, 2, 3})
		initialToken := argmax(append([]float32(nil), ln.logits...))
		var events []string
		s.beforeModelExecute = func(kind nativeSchedulerEventKind, lane *schedLane) {
			if kind == nativeSchedulerEventDecode && lane == ln {
				events = append(events, "forward")
			}
		}
		s.observeNativeEvent = func(event nativeSchedulerEvent) {
			if event.Kind == nativeSchedulerEventDecode && event.Lane == ln {
				events = append(events, "publish")
			}
		}

		if !ln.stageDeferredToken() {
			t.Fatal("bootstrap sampling did not stage a token")
		}
		if ln.deferredToken == nil || ln.deferredToken.id != initialToken {
			t.Fatalf("bootstrap pending token = %+v, want id %d", ln.deferredToken, initialToken)
		}

		s.runIteration(true)
		if !reflect.DeepEqual(events, []string{"forward", "publish"}) {
			t.Fatalf("events = %v, want forward then publish", events)
		}
		select {
		case token := <-ln.tokens:
			if token.ID != initialToken {
				t.Fatalf("published token = %d, want initial-logits argmax %d", token.ID, initialToken)
			}
		default:
			t.Fatal("decode boundary did not publish the prior sampled token")
		}
		if ln.deferredToken == nil || ln.deferredToken.id != argmax(ln.logits) {
			t.Fatalf("successor pending token = %+v, want argmax of forward-produced logits", ln.deferredToken)
		}
	})

	t.Run("cancellation_discards_pending_and_reclaims_once", func(t *testing.T) {
		s, ln := deferredLifecycleFixture(t, []int{4, 5, 6})
		var closes atomic.Int64
		s.closeSession = func(session *model.Session) {
			closes.Add(1)
			session.Close()
		}
		if !ln.stageDeferredToken() {
			t.Fatal("bootstrap sampling did not stage a token")
		}
		ln.Cancel()
		s.runIteration(true)
		if _, err := ln.Result(); !errors.Is(err, context.Canceled) {
			t.Fatalf("Result error = %v, want context.Canceled", err)
		}
		if !ln.Reclaimed() || closes.Load() != 1 {
			t.Fatalf("reclaimed=%t closes=%d, want true/1", ln.reclaimed, closes.Load())
		}
		for token := range ln.tokens {
			t.Fatalf("cancelled pending token was published: %+v", token)
		}
	})

	t.Run("invalid_logits_fail_without_publication_and_reclaim_once", func(t *testing.T) {
		s, ln := deferredLifecycleFixture(t, []int{7, 8, 9})
		var closes atomic.Int64
		s.closeSession = func(session *model.Session) {
			closes.Add(1)
			session.Close()
		}
		ln.logits = nil
		s.runIteration(true)
		if _, err := ln.Result(); !errors.Is(err, errNativeSchedulerLaneNotDecodeReady) {
			t.Fatalf("Result error = %v, want decode-ready sentinel", err)
		}
		if !ln.Reclaimed() || closes.Load() != 1 {
			t.Fatalf("reclaimed=%t closes=%d, want true/1", ln.reclaimed, closes.Load())
		}
		for token := range ln.tokens {
			t.Fatalf("failed lane published token: %+v", token)
		}
	})
}

func deferredLifecycleFixture(t *testing.T, prompt []int) (*NativeScheduler, *schedLane) {
	t.Helper()
	m := model.NewSynthetic(SyntheticConfig())
	s := NewNativeScheduler(m)
	s.started.Do(func() {})
	ctx, cancel := context.WithCancel(context.Background())
	sess := m.NewSession()
	ln := &schedLane{
		sched: s, ctx: ctx, cancel: cancel,
		sess: sess, logits: sess.Prefill(prompt),
		prompt: append([]int(nil), prompt...), promptLen: len(prompt),
		state: schedLaneDecode, seqNo: 1,
		tokens: make(chan abi.EngineToken, 4), done: make(chan struct{}),
	}
	s.lanes = []*schedLane{ln}
	t.Cleanup(func() {
		if !ln.terminal {
			ln.Cancel()
			s.runIteration(true)
		}
		s.Close()
	})
	return s, ln
}
