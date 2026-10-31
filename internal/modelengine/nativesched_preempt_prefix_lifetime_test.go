package modelengine

import (
	"bytes"
	"context"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// Real scheduler/radix ownership with a synthetic host Qwen model. This does not
// exercise a GPU, physical direct storage, or measured memory/throughput gains.
// Estimate only; this test has not been timed.
// fak-test:runtime medium est=4s lane=default
func TestNativePreemptionRetiresPrefixOwnerAfterSnapshot(t *testing.T) {
	cfg := nativeSchedulerPrefillConfig()
	cfg.EOSTokenID, cfg.EOSTokenIDs = -1, nil
	m := nativeSchedulerPrefillModel(t, cfg)
	for _, tc := range []struct {
		name string
		mode NativePreemptionMode
	}{{"recompute", NativePreemptRecompute}, {"host-swap", NativePreemptSwap}} {
		t.Run(tc.name, func(t *testing.T) {
			prompt := nativeSchedulerQwenPrompt(32)
			seed := m.NewSession()
			seed.Quant, seed.Q4K = false, true
			seed.Prefill(prompt[:16])
			tree := radixkv.New(0)
			boundary, matched := tree.Lookup(prompt[:16])
			tree.Done(tree.Insert(boundary, prompt[matched:16], seed.Cache))
			seed.Close()
			s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(map[string][]int{"victim": prompt, "peer": prompt}))
			s.coupler = nil // Isolate ownership from host-dependent worker budgets.
			s.recorder = enginestep.New(16)
			s.SetRadixKV(tree)
			if err := s.SetQwenPrefillMaxTokensPerIteration(nativeQwenPrefillMinChunkTokens); err != nil {
				t.Fatal(err)
			}
			nativeSchedulerBeginManualDrain(t, s)
			defer nativeSchedulerEndManualDrain(s)
			victim := nativeSchedulerAdmitLane(t, s, "victim")
			peer := nativeSchedulerAdmitLane(t, s, "peer")
			for i := 0; i < 4 && (victim.state != schedLaneDecode || peer.state != schedLaneDecode); i++ {
				nativeSchedulerDriveIteration(t, s)
				nativeSchedulerDrainAvailable(victim)
				nativeSchedulerDrainAvailable(peer)
			}
			if victim.terminal || peer.terminal || victim.state != schedLaneDecode || peer.state != schedLaneDecode {
				t.Fatal("fixture did not produce two live decode-ready prefix owners")
			}
			old, peerSession := victim.sess, peer.sess
			st := s.getPrefixState()
			st.mu.RLock()
			oldInfo, peerInfo := st.holderLookups[old], st.holderLookups[peerSession]
			owned := oldInfo != nil && peerInfo != nil && oldInfo != peerInfo &&
				oldInfo.applied && peerInfo.applied && oldInfo.ownerTree == tree && peerInfo.ownerTree == tree &&
				st.laneLookups[victim] == oldInfo && st.laneLookups[peer] == peerInfo
			st.mu.RUnlock()
			if !owned || tree.Stats().ProtectedTokens != 16 {
				t.Fatal("fixture did not acquire and apply both real prefix leases")
			}
			// Retirement must use the captured owner, not this replacement provider.
			s.SetRadixKV(radixkv.New(0))
			closes := make(map[*model.Session]int)
			s.closeSession = func(sess *model.Session) {
				closes[sess]++
				if sess == old {
					st.mu.RLock()
					_, holder := st.holderLookups[old]
					_, lane := st.laneLookups[victim]
					st.mu.RUnlock()
					if holder || lane {
						t.Error("native close ran before old prefix ownership retired")
					}
				}
				sess.Close()
			}
			history := append(append([]int(nil), victim.prompt...), victim.gen...)
			if _, err := old.VerifyTokenLineage(history); err != nil {
				t.Fatalf("fixture history: %v", err)
			}
			wantLogits, deferred := copyF32(victim.logits), victim.deferredToken
			before, err := model.QwenHybridKVCacheToHost(old.Cache, 16)
			if err != nil {
				t.Fatal(err)
			}
			// The test owns the actual executor; production calls these methods
			// under mu at a completed model-execution boundary too.
			s.executor.Lock()
			defer s.executor.Unlock()
			s.mu.Lock()
			defer s.mu.Unlock()
			s.preemption = NativePreemptionPolicy{Mode: NativePreemptGPUDirectSwap, MaxBlocks: 128, BlockTokens: 16}
			if err := s.preemptLaneLocked(victim); err == nil {
				t.Fatal("missing swapper unexpectedly succeeded")
			}
			st.mu.RLock()
			retained := st.holderLookups[old] == oldInfo && st.laneLookups[victim] == oldInfo &&
				st.holderLookups[peerSession] == peerInfo && st.laneLookups[peer] == peerInfo
			st.mu.RUnlock()
			if !retained || victim.sess != old || closes[old] != 0 || len(s.preempted) != 0 || tree.Stats().ProtectedTokens != 16 {
				t.Fatal("refused snapshot retired a live owner")
			}
			s.preemption.Mode = tc.mode
			if err := s.preemptLaneLocked(victim); err != nil {
				t.Fatal(err)
			}
			// Match the pressure loop's removal after preemptLaneLocked succeeds.
			s.lanes = []*schedLane{peer}
			st.mu.RLock()
			_, oldHolder := st.holderLookups[old]
			_, oldLane := st.laneLookups[victim]
			peerKept := st.holderLookups[peerSession] == peerInfo && st.laneLookups[peer] == peerInfo
			st.mu.RUnlock()
			if oldHolder || oldLane || !peerKept || closes[old] != 1 || closes[peerSession] != 0 ||
				victim.sess != nil || len(s.preempted) != 1 || s.preempted[0] != victim || victim.terminal {
				t.Fatal("successful preemption did not retire only its old holder")
			}
			if tree.Stats().ProtectedTokens != 16 {
				t.Fatal("preemption released the peer's prefix lease")
			}
			peer.finish(nil, context.Canceled)
			s.lanes = nil
			tree.SetRetention(0)
			if tree.Stats().Tokens != 0 || closes[peerSession] != 1 {
				t.Fatal("retired prefix remained pinned after its last live owner")
			}
			// The old tree is now evicted: restoration cannot borrow its lease/KV.
			s.readmitPreemptedLocked()
			restored := victim.sess
			if restored == nil || restored == old || len(s.preempted) != 0 || len(s.lanes) != 1 || s.lanes[0] != victim {
				t.Fatal("preempted lane was not restored into a new holder")
			}
			if _, err := restored.VerifyTokenLineage(history); err != nil {
				t.Fatalf("restored history: %v", err)
			}
			if victim.deferredToken != deferred || !reflect.DeepEqual(append(append([]int(nil), victim.prompt...), victim.gen...), history) {
				t.Fatal("retirement changed queued output or token history")
			}
			if tc.mode == NativePreemptSwap {
				after, err := model.QwenHybridKVCacheToHost(restored.Cache, 16)
				if err != nil || !bytes.Equal(after, before) || !reflect.DeepEqual(victim.logits, wantLogits) {
					t.Fatalf("host swap changed saved KV/logits after lease retirement: %v", err)
				}
			} else {
				control := m.NewSession()
				control.Quant, control.Q4K = false, true
				defer control.Close()
				if !reflect.DeepEqual(victim.logits, control.Prefill(history)) {
					t.Fatal("recompute differs from an independent full-history prefill")
				}
			}
			if s.getPrefixHitInfo(victim) != nil || closes[old] != 1 {
				t.Fatal("readmission reused the retired prefix holder")
			}
			victim.finish(nil, context.Canceled)
			s.lanes = nil
			st.mu.RLock()
			holders, lanes := len(st.holderLookups), len(st.laneLookups)
			st.mu.RUnlock()
			if holders != 0 || lanes != 0 || closes[old] != 1 || closes[restored] != 1 {
				t.Fatal("terminal readmission retained metadata or closed the old owner twice")
			}
		})
	}
}
