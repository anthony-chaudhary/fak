package modelengine

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// Estimate only; this test has not been timed.
// fak-test:runtime medium est=2s lane=default
func TestNativeSchedulerRejectedAdmissionReleasesPrefixLease(t *testing.T) {
	for _, closed := range []bool{false, true} {
		name := "request-canceled"
		if closed {
			name = "scheduler-closed"
		}
		t.Run(name, func(t *testing.T) {
			m := nativeSchedulerPrefillModel(t)
			prompt := nativeSchedulerQwenPrompt(48)
			tree := radixkv.New(0)
			// A real radix lease needs only token accounting. No cache is installed
			// and no model forward should occur before this admission is rejected.
			boundary, matched := tree.Lookup(prompt)
			tree.Done(tree.Insert(boundary, prompt[matched:], nil))
			s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(map[string][]int{"reject": prompt}))
			defer s.Close()
			s.SetRadixKV(tree)
			if err := s.SetQwenPrefillMaxTokensPerIteration(nativeQwenPrefillMinChunkTokens); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			s.SetSessionProfilerFactory(func(NativeSessionLifecycle) *model.PhaseProfiler {
				if closed {
					s.Close()
				} else {
					cancel()
				}
				return nil
			})
			closes := 0
			s.closeSession = func(sess *model.Session) {
				closes++
				// Cleanup must release the lease before the native session close hook.
				if got := tree.Stats().ProtectedTokens; got != 0 {
					t.Errorf("session closed with %d prefix tokens still leased", got)
				}
				sess.Close()
			}
			req, err := s.Admit(ctx, inlineCall("reject", `{}`))
			wantErr := context.Canceled
			if closed {
				wantErr = errSchedClosed
			}
			if req != nil || err != wantErr {
				t.Fatalf("admission = (%v, %v), want nil/%v", req, err, wantErr)
			}
			if closes != 1 {
				t.Fatalf("session closes = %d, want 1", closes)
			}
			if got := s.PrefixStats().Hits; got != 1 {
				t.Fatalf("prefix hits = %d, want real lookup", got)
			}
			st := s.getPrefixState()
			st.mu.RLock()
			holders, lanes := len(st.holderLookups), len(st.laneLookups)
			st.mu.RUnlock()
			if holders != 0 || lanes != 0 {
				t.Fatalf("rejected admission retained prefix ownership: holders=%d lanes=%d", holders, lanes)
			}
			s.mu.Lock()
			queued, seq, started := len(s.waiting), s.seqNo, s.runStarted
			s.mu.Unlock()
			if queued != 0 || seq != 0 || started {
				t.Fatalf("rejected admission published: queued=%d seq=%d started=%v", queued, seq, started)
			}
			if got := s.CachePhaseLatencyReceipt().Observations; got != 0 {
				t.Fatalf("rejected async admission ran %d forwards", got)
			}
			tree.SetRetention(0)
			if got := tree.Stats().Tokens; got != 0 {
				t.Fatalf("unreleased radix lease prevented eviction of %d tokens", got)
			}
		})
	}
}
