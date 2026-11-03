//go:build fak_vllm_packed_prefill

package modelengine

import (
	"context"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func packedServingModel() *model.Model {
	return model.NewSynthetic(model.Config{HiddenSize: 64, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 128, VocabSize: 257, RMSNormEps: 1e-5, RopeTheta: 10000, TieWordEmbeddings: true, EOSTokenID: -1})
}

func packedServingPrompt(n, offset int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = (offset + i*7) % 257
	}
	return out
}

// fak-test:runtime fast est=1s lane=default
func TestNativePackedPrefillServingIngress(t *testing.T) {
	m := packedServingModel()
	e := New()
	e.Preload(m)
	s := e.nativeScheduler()
	if !s.packedPrefillEnabled {
		t.Fatal("qualification did not arm ordinary Engine scheduler")
	}
	prompts := map[string][]int{"long": packedServingPrompt(37, 3), "short": packedServingPrompt(5, 9), "medium": packedServingPrompt(11, 21)}
	s.prepare = func(_ context.Context, c *abi.ToolCall, _ *model.Model) schedPrepare {
		return schedPrepare{prompt: prompts[c.Tool]}
	}
	nativeSchedulerBeginManualDrain(t, s)
	defer nativeSchedulerEndManualDrain(s)
	var lanes []*schedLane
	var want [][]int
	for _, name := range []string{"long", "short", "medium"} {
		req, err := e.Admit(context.Background(), inlineCall(name, "{}"))
		if err != nil {
			t.Fatal(err)
		}
		ln := req.(*schedLane)
		if !ln.packedPrefill || ln.state != schedLanePrefilling || ln.sess.Cache.Len() != 0 || len(ln.logits) != 0 {
			t.Fatal("ordinary Engine.Admit silently prefilled or fell back")
		}
		lanes = append(lanes, ln)
		want = append(want, nativeSchedulerSynchronousTokens(m, prompts[name]))
	}
	got := make([][]int, len(lanes))
	for step := 0; step < genTokens*4+16; step++ {
		all := true
		for _, ln := range lanes {
			select {
			case <-ln.done:
			default:
				all = false
			}
		}
		if all {
			break
		}
		nativeSchedulerDriveIteration(t, s)
		for i, ln := range lanes {
			got[i] = append(got[i], nativeSchedulerDrainAvailable(ln)...)
		}
	}
	for i, ln := range lanes {
		select {
		case <-ln.done:
		default:
			t.Fatal("lane never retired")
		}
		if _, err := ln.Result(); err != nil {
			t.Fatal(err)
		}
		if !ln.Reclaimed() || !reflect.DeepEqual(got[i], want[i]) {
			t.Fatalf("lane %d ownership/output parity", i)
		}
	}
	stats := s.PackedPrefillReceipt()
	if stats.Calls < 2 || stats.Sequences <= stats.Calls || stats.TokenRows != 53 ||
		stats.MaxTokenRows > model.PackedPrefillMaxTokens || stats.MaxTokenRows == 0 {
		t.Fatalf("no actual heterogeneous bounded packing: %+v", stats)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestNativePackedPrefillCancellationAndCloseJoin(t *testing.T) {
	for _, closeScheduler := range []bool{false, true} {
		t.Run(map[bool]string{false: "cancel", true: "close"}[closeScheduler], func(t *testing.T) {
			m := packedServingModel()
			e := New()
			e.Preload(m)
			s := e.nativeScheduler()
			s.prepare = func(context.Context, *abi.ToolCall, *model.Model) schedPrepare {
				return schedPrepare{prompt: packedServingPrompt(9, 4)}
			}
			nativeSchedulerBeginManualDrain(t, s)
			req, err := e.Admit(context.Background(), inlineCall("paused", "{}"))
			if err != nil {
				t.Fatal(err)
			}
			ln := req.(*schedLane)
			entered, release := make(chan struct{}), make(chan struct{})
			var unblock sync.Once
			defer func() {
				unblock.Do(func() { close(release) })
				s.endBlockedDrain()
				ctx, cancel := context.WithTimeout(context.Background(), time.Second)
				defer cancel()
				if err := s.CloseAndWait(ctx); err != nil {
					t.Error(err)
				}
			}()
			s.beforeModelExecute = func(kind nativeSchedulerEventKind, _ *schedLane) {
				if kind == nativeSchedulerEventPrefill {
					close(entered)
					<-release
				}
			}
			driven := make(chan struct{})
			go func() { nativeSchedulerDriveIteration(t, s); close(driven) }()
			select {
			case <-entered:
			case <-time.After(time.Second):
				t.Fatal("prefill did not reach owned boundary")
			}
			if closeScheduler {
				s.Close()
			} else {
				ln.Cancel()
			}
			// While execution owns the session, cancel/Close cannot close done or
			// reclaim its backing. Stats never expose the mutable cache.
			s.mu.Lock()
			if ln.inflightSess == nil || ln.sess == nil || ln.sess.Cache != nil {
				s.mu.Unlock()
				t.Fatal("missing borrowed owner/statistics shell")
			}
			s.mu.Unlock()
			select {
			case <-ln.done:
				t.Fatal("retired in-flight session early")
			default:
			}
			unblock.Do(func() { close(release) })
			select {
			case <-driven:
			case <-time.After(time.Second):
				t.Fatal("execution did not retire")
			}
			select {
			case <-ln.done:
			default:
				t.Fatal("canceled owner not retired at boundary")
			}
			if _, err := ln.Result(); err == nil || !ln.Reclaimed() {
				t.Fatal("cancellation lost")
			}
			if len(nativeSchedulerDrainAvailable(ln)) != 0 {
				t.Fatal("published token after cancellation")
			}
			s.beforeModelExecute = nil
		})
	}
}

// fak-test:runtime fast est=1s lane=default
func TestNativePackedPrefillReservesUnfinishedPromptCapacity(t *testing.T) {
	m := packedServingModel()
	e := New()
	e.Preload(m)
	s := e.nativeScheduler()
	s.SetMaxRunning(2)
	s.SetKVPreemptionPolicy(NativePreemptionPolicy{Mode: NativePreemptRecompute, MaxBlocks: 2, BlockTokens: 16})
	s.prepare = func(context.Context, *abi.ToolCall, *model.Model) schedPrepare {
		return schedPrepare{prompt: packedServingPrompt(32, 4)}
	}
	nativeSchedulerBeginManualDrain(t, s)
	defer nativeSchedulerEndManualDrain(s)
	first := nativeSchedulerAdmitLane(t, s, "first")
	second := nativeSchedulerAdmitLane(t, s, "second")
	// Admission must reserve the first empty PREFILLING owner's complete prompt.
	// A second same-sized owner must not materialize behind that zero-length KV.
	s.mu.Lock()
	initialOK := first.packedPrefill && first.state == schedLanePrefilling &&
		first.sess != nil && first.sess.Cache.Len() == 0 &&
		second.sess == nil && !second.materializing &&
		s.canMaterializeLaneLocked(second) == nativeAllocationLater
	s.promoteWaitingLocked(2)
	promotedOK := len(s.lanes) == 1 && s.lanes[0] == first && second.sess == nil
	s.mu.Unlock()
	if !initialOK || !promotedOK {
		t.Fatal("unfinished packed prompt did not reserve its full capacity")
	}
	// Drive the real packed boundary under the existing executor, stopping before
	// legacy decode: decode growth is not a new hard-resident-cap promise.
	s.executor.Lock()
	ready := s.advancePackedPrefill(1)
	s.executor.Unlock()
	s.mu.Lock()
	completedOK := len(ready) == 1 && ready[0] == first &&
		first.state == schedLaneDecode && first.sess.Cache.Len() == 32 &&
		s.usedKVBlocksLocked() == 2 && second.sess == nil &&
		s.canMaterializeLaneLocked(second) == nativeAllocationLater
	s.mu.Unlock()
	if !completedOK {
		t.Fatal("packed boundary overcommitted or lost reservation")
	}
	first.Cancel()
	second.Cancel()
	for i := 0; i < 4; i++ {
		nativeSchedulerDriveIteration(t, s)
	}
	for _, ln := range []*schedLane{first, second} {
		select {
		case <-ln.done:
		default:
			t.Fatal("reserved owner did not retire")
		}
		if _, err := ln.Result(); err == nil || !ln.Reclaimed() {
			t.Fatal("reserved owner cancellation/reclamation failed")
		}
	}
}
