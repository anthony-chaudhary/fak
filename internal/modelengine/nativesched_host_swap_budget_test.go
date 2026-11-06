package modelengine

import (
	"bytes"
	"context"
	"errors"
	"math"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime medium est=1s lane=default
func TestNativeHostSwapBudget(t *testing.T) {
	t.Run("retained-backing-capacity", func(t *testing.T) {
		first := &schedLane{hostKV: make([]byte, 2, 64), savedLogits: make([]float32, 1, 8)}
		second := &schedLane{hostKV: make([]byte, 3, 128), savedLogits: make([]float32, 2, 16)}
		first.hostKV[0], second.hostKV[0] = 17, 29
		s := &NativeScheduler{preempted: []*schedLane{first, second}}
		st := s.KVPreemptionStats()
		if st.HostSwapBytes != 192 || st.PreemptedLogitsBytes != 96 || st.SwappedOut != 2 {
			t.Fatalf("backing capacity across owners=%+v", st)
		}
		// Saved logits are reported separately, not charged to the serialized-KV limit.
		if err := s.SetKVHostSwapByteLimit(192); err != nil {
			t.Fatal(err)
		}
		if err := s.SetKVHostSwapByteLimit(191); !errors.Is(err, ErrNativeHostSwapCapacity) {
			t.Fatalf("shrink below retained backing error=%v", err)
		}
		if st = s.KVPreemptionStats(); st.HostSwapLimitBytes != 192 || st.HostSwapBytes != 192 ||
			st.PreemptedLogitsBytes != 96 || first.hostKV[0] != 17 || second.hostKV[0] != 29 {
			t.Fatalf("refused shrink changed ownership or configuration: %+v", st)
		}
		// Accounting follows the retained owners, without a duplicate byte ledger.
		first.hostKV, first.savedLogits = nil, nil
		if st = s.KVPreemptionStats(); st.HostSwapBytes != 128 || st.PreemptedLogitsBytes != 64 {
			t.Fatalf("cleared owner still charged: %+v", st)
		}
		if err := s.SetKVHostSwapByteLimit(128); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("saturating-byte-sum", func(t *testing.T) {
		for _, tc := range []struct {
			total, count, want int64
		}{
			{0, 0, 0}, {64, 128, 192},
			{math.MaxInt64 - 1, 1, math.MaxInt64},
			{math.MaxInt64 - 1, 2, math.MaxInt64},
			{math.MaxInt64, 1, math.MaxInt64},
			{0, -1, math.MaxInt64},
		} {
			if got := nativeHostByteSum(tc.total, tc.count); got != tc.want {
				t.Errorf("sum(%d, %d)=%d, want %d", tc.total, tc.count, got, tc.want)
			}
		}
	})

	t.Run("concurrent-limit-and-stats", func(t *testing.T) {
		// Owners stay immutable while public configuration and stats contend.
		s := &NativeScheduler{preempted: []*schedLane{
			{hostKV: make([]byte, 2, 64), savedLogits: make([]float32, 1, 8)},
			{hostKV: make([]byte, 3, 128), savedLogits: make([]float32, 2, 16)},
		}}
		limits := []int64{0, 192, 384}
		start := make(chan struct{})
		var wg sync.WaitGroup
		for worker := range 4 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				<-start
				for i := range 256 {
					if worker < 2 {
						if err := s.SetKVHostSwapByteLimit(limits[(i+worker)%len(limits)]); err != nil {
							t.Errorf("concurrent limit: %v", err)
							return
						}
					} else {
						st := s.KVPreemptionStats()
						if st.HostSwapBytes != 192 || st.PreemptedLogitsBytes != 96 || st.SwappedOut != 2 ||
							(st.HostSwapLimitBytes != 0 && st.HostSwapLimitBytes != 192 && st.HostSwapLimitBytes != 384) {
							t.Errorf("concurrent snapshot=%+v", st)
							return
						}
					}
				}
			}()
		}
		close(start)
		wg.Wait()
	})

	for _, hybrid := range []bool{false, true} {
		name := "dense"
		m := model.NewSynthetic(SyntheticConfig())
		if hybrid {
			name, m = "qwen-hybrid", nativeSchedulerQwenSwapModel()
		}
		t.Run(name, func(t *testing.T) {
			makeScheduler := func() (*NativeScheduler, *schedLane, *schedLane, *schedLane, int64) {
				t.Helper()
				s := NewNativeScheduler(m)
				s.SetKVPreemptionPolicy(NativePreemptionPolicy{Mode: NativePreemptSwap, MaxBlocks: 1, BlockTokens: 128})
				lanes := make([]*schedLane, 3)
				for i := range lanes {
					lanes[i] = nativeSchedulerQwenReadmitLane(t, s, []int{3, 7, 11}, 0)
					lanes[i].seqNo = int64(i + 1)
					lanes[i].state = schedLaneDecode
					t.Cleanup(func() { lanes[i].finish(nil, context.Canceled) })
				}
				size, err := model.KVCacheHostSwapBytes(lanes[0].sess.Cache, 128)
				if err != nil {
					t.Fatal(err)
				}
				s.lanes = lanes
				return s, lanes[0], lanes[1], lanes[2], int64(size)
			}

			t.Run("capacity-ownership-readmit", func(t *testing.T) {
				s, survivor, refused, held, size := makeScheduler()
				wantLogits := copyF32(held.logits)
				if err := s.SetKVHostSwapByteLimit(size); err != nil {
					t.Fatal(err)
				}
				s.enforcePreemptionLocked()
				if len(s.lanes) != 1 || s.lanes[0] != survivor || len(s.preempted) != 1 || s.preempted[0] != held {
					t.Fatal("pressure route lost its survivor or stored owner")
				}
				if !errors.Is(refused.err, ErrNativeHostSwapCapacity) || !refused.terminal || refused.sess != nil || refused.hostKV != nil || refused.savedLogits != nil {
					t.Fatal("capacity failure did not use existing request-error retirement")
				}
				st := s.KVPreemptionStats()
				if st.HostSwapBytes != size || st.HostSwapLimitBytes != size || st.HostSwapRefusals != 1 ||
					st.SwapPreemptions != 1 || st.RecomputeCount != 0 || st.PreemptedLogitsBytes == 0 {
					t.Fatalf("retention/denominators=%+v", st)
				}
				before := append([]byte(nil), held.hostKV...)
				if err := s.SetKVHostSwapByteLimit(size - 1); !errors.Is(err, ErrNativeHostSwapCapacity) ||
					s.KVPreemptionStats().HostSwapLimitBytes != size || !bytes.Equal(before, held.hostKV) {
					t.Fatal("live shrink changed the old limit or owner")
				}
				survivor.finish(nil, context.Canceled)
				s.lanes = nil
				s.readmitPreemptedLocked()
				if held.sess == nil || held.hostKV != nil || held.savedLogits != nil || !reflect.DeepEqual(wantLogits, held.logits) {
					t.Fatal("readmit did not transfer/release the exact stored owner")
				}
				st = s.KVPreemptionStats()
				if st.HostSwapBytes != 0 || st.PreemptedLogitsBytes != 0 || st.Readmitted != 1 || st.SwapRestoredBytes != size {
					t.Fatalf("readmit release=%+v", st)
				}
				var metrics strings.Builder
				writeNativePreemptionMetrics(&metrics, st)
				for _, row := range []string{"fak_sched_preempt_host_swap_bytes 0\n", "fak_sched_preempt_host_swap_refused_total 1\n", "fak_sched_preempt_saved_logits_bytes 0\n"} {
					if !strings.Contains(metrics.String(), row) {
						t.Fatalf("missing metric %q", row)
					}
				}
			})

			for _, action := range []string{"cancel", "restore-error", "close"} {
				t.Run(action, func(t *testing.T) {
					s, first, second, held, size := makeScheduler()
					if err := s.SetKVHostSwapByteLimit(size); err != nil {
						t.Fatal(err)
					}
					if err := s.preemptLaneLocked(held); err != nil {
						t.Fatal(err)
					}
					// The normal pressure caller removes the stored lane from running.
					s.lanes = []*schedLane{first, second}
					switch action {
					case "cancel":
						held.cancel()
						s.dropCanceledPreemptedLocked()
					case "restore-error":
						held.hostKV[0] ^= 0xff
						s.lanes = nil
						s.readmitPreemptedLocked()
					case "close":
						s.Close()
						// Close cancels; the serialized executor owns actual retirement.
						s.dropCanceledPreemptedLocked()
					}
					st := s.KVPreemptionStats()
					if !held.terminal || held.hostKV != nil || held.savedLogits != nil || st.HostSwapBytes != 0 || st.PreemptedLogitsBytes != 0 {
						t.Fatalf("terminal owner retained after %s: %+v", action, st)
					}
				})
			}

			t.Run("preallocation-refusal-and-legacy", func(t *testing.T) {
				s, _, _, lane, size := makeScheduler()
				original, logits := lane.sess, copyF32(lane.logits)
				if err := s.SetKVHostSwapByteLimit(size - 1); err != nil {
					t.Fatal(err)
				}
				if err := s.preemptLaneLocked(lane); !errors.Is(err, ErrNativeHostSwapCapacity) {
					t.Fatalf("oversized store error=%v", err)
				}
				if lane.sess != original || lane.hostKV != nil || lane.savedLogits != nil ||
					len(s.preempted) != 0 || !reflect.DeepEqual(lane.logits, logits) || s.preemptStats.Preemptions != 0 {
					t.Fatal("pre-allocation refusal changed resident ownership or committed a swap")
				}
				if err := s.SetKVHostSwapByteLimit(-1); err == nil || s.hostSwapLimitBytes != size-1 {
					t.Fatal("negative limit changed configuration")
				}
				if err := s.SetKVHostSwapByteLimit(0); err != nil {
					t.Fatal(err)
				}
				if err := s.preemptLaneLocked(lane); err != nil || int64(cap(lane.hostKV)) != size {
					t.Fatalf("legacy unbounded store failed: %v", err)
				}
			})

			t.Run("ordinary-admit-output", func(t *testing.T) {
				control := NewNativeScheduler(m)
				defer control.Close()
				want := hostSwapDrainOrdinaryRequests(t, control)
				s := NewNativeScheduler(m)
				defer s.Close()
				s.SetKVPreemptionPolicy(NativePreemptionPolicy{Mode: NativePreemptSwap, MaxBlocks: 1, BlockTokens: 128})
				if err := s.SetKVHostSwapByteLimit(64 << 20); err != nil {
					t.Fatal(err)
				}
				got := hostSwapDrainOrdinaryRequests(t, s)
				if !reflect.DeepEqual(want, got) {
					t.Fatal("bounded ordinary Admit changed complete token streams")
				}
				if st := s.KVPreemptionStats(); st.SwapPreemptions == 0 || st.HostSwapBytes != 0 || st.HostSwapRefusals != 0 {
					t.Fatalf("ordinary pressure route was not exercised: %+v", st)
				}
			})
		})
	}
}

func hostSwapDrainOrdinaryRequests(t *testing.T, s *NativeScheduler) [][]int {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	// Hold the real execution mutex, not the state/admission mutex, until
	// both synchronous Admit calls have installed their waiting owners.
	// This makes the pressure witness independent of token-channel timing.
	reqs := func() []abi.EngineRequest {
		s.executor.Lock()
		defer s.executor.Unlock()
		var out []abi.EngineRequest
		for _, call := range issue31Calls() {
			req, err := s.Admit(ctx, call)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, req)
		}
		return out
	}()
	out := make([][]int, len(reqs))
	var wg sync.WaitGroup
	for i, req := range reqs {
		wg.Add(1)
		go func(i int, req abi.EngineRequest) {
			defer wg.Done()
			for token := range req.Tokens() {
				out[i] = append(out[i], token.ID)
			}
		}(i, req)
	}
	wg.Wait()
	for _, req := range reqs {
		result, err := req.Result()
		if err != nil || result == nil || result.Status != abi.StatusOK {
			t.Fatalf("ordinary result=%+v err=%v", result, err)
		}
	}
	if err := s.CloseAndWait(ctx); err != nil {
		t.Fatal(err)
	}
	return out
}
