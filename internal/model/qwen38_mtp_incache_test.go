package model

import (
	"encoding/json"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Read the wire counters so the regression can also be applied to the previous
// implementation: absent accounting must fail, not silently mean zero work.
func qwen38InCacheWork(t *testing.T, r *Qwen38MTPBatchedReceipt) (string, int, int) {
	t.Helper()
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	var work struct {
		Path   string `json:"target_verification_path"`
		Replay *int   `json:"full_target_replay_steps"`
		Repair *int   `json:"recurrent_repair_tokens"`
	}
	if err := json.Unmarshal(data, &work); err != nil || work.Replay == nil || work.Repair == nil {
		t.Fatalf("missing actual commit-work accounting: %s; %v", data, err)
	}
	return work.Path, *work.Replay, *work.Repair
}

type qwen38InCacheCleanupCheckpoint struct {
	restore, close func()
	closeCalls     int
}

func (c *qwen38InCacheCleanupCheckpoint) Restore() error {
	c.restore()
	return nil
}

func (c *qwen38InCacheCleanupCheckpoint) Close() {
	c.closeCalls++
	if c.close != nil {
		c.close()
	}
}

// fak-test:runtime fast est=2s lane=default
func TestQwen38MTP_InCacheAbortRetiresInterruptedRestore(t *testing.T) {
	for _, owned := range []bool{false, true} {
		for _, mode := range []string{"normal", "restore_panic", "restore_goexit", "close_panic"} {
			t.Run(mode+"/owned_"+itoa(map[bool]int{false: 0, true: 1}[owned]), func(t *testing.T) {
				m := qwen38HybridMTPEnabledSyntheticModel(t)
				var target *Session
				var owner *sync.Mutex
				if owned {
					be := &ownedQwen35CloseBackend{recordingQwen35Backend: newRecordingQwen35Backend(m)}
					defer be.reference.Close()
					be.Backend = compute.Pick("cpu-ref")
					if be.Backend == nil {
						t.Fatal("CPU reference backend unavailable")
					}
					var err error
					target, err = m.NewBackendSessionChecked(be)
					if err != nil {
						t.Fatal(err)
					}
					owner = &be.owner
				} else {
					target = m.NewSession()
				}
				defer target.Close()
				boundary := target.Prefill([]int{3, 7, 11})
				tx, err := beginQwen35MTPTargetTransaction(target, boundary)
				if err != nil {
					t.Fatal(err)
				}
				snapshot := tx.snapshot
				checkpoint := &qwen38InCacheCleanupCheckpoint{restore: func() {
					requirePrefixSnapshotCleanupOwner(owner)
					switch mode {
					case "restore_panic":
						panic("restore interrupted")
					case "restore_goexit":
						runtime.Goexit()
					}
				}}
				checkpoint.close = func() {
					requirePrefixSnapshotCleanupOwner(owner)
					if mode != "normal" {
						panic("secondary close interrupted")
					}
				}
				tx.checkpoint = checkpoint
				type outcome struct {
					returned bool
					panic    any
					err      error
				}
				done := make(chan outcome, 1)
				go func() {
					var got outcome
					defer func() { got.panic = recover(); done <- got }()
					got.err = abortQwen38BatchedTransaction(tx)
					got.returned = true
				}()
				select {
				case got := <-done:
					if mode == "normal" && (!got.returned || got.err != nil || got.panic != nil) {
						t.Fatalf("normal abort failed: %+v", got)
					}
					if mode == "restore_panic" && (got.returned || got.panic != "restore interrupted") {
						t.Fatalf("cleanup replaced the original restore panic: %+v", got)
					}
					if mode == "restore_goexit" && (got.returned || got.panic != nil) {
						t.Fatalf("cleanup replaced Goexit: %+v", got)
					}
					if mode == "close_panic" && (got.returned || got.panic != "secondary close interrupted") {
						t.Fatalf("cleanup panic was reported as successful abort: %+v", got)
					}
				case <-time.After(5 * time.Second):
					t.Fatal("interrupted abort stranded its caller")
				}
				if !tx.closed || tx.closeCount != 1 || tx.snapshot != nil || tx.checkpoint != nil || tx.prefixReplay != nil ||
					!snapshot.closed || snapshot.Cache != nil || snapshot.halKV != nil || snapshot.qwen35 != nil || checkpoint.closeCalls != 1 {
					t.Fatal("interrupted restore stranded or duplicated transaction ownership")
				}
				if err := abortQwen38BatchedTransaction(tx); err != nil || checkpoint.closeCalls != 1 {
					t.Fatal("terminal abort retried checkpoint cleanup")
				}
				if owner != nil {
					if target.backendExecutionScope != nil || !owner.TryLock() {
						t.Fatal("abort stranded its execution owner")
					}
					owner.Unlock()
				}
			})
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestQwen38MTP_InCacheCommitRetirementFailure(t *testing.T) {
	for _, breakRestore := range []bool{false, true} {
		t.Run(itoa(map[bool]int{false: 0, true: 1}[breakRestore]), func(t *testing.T) {
			m := qwen38HybridMTPEnabledSyntheticModel(t)
			prefix := []int{3, 7, 11}
			draft := qwen38InCacheDraft(t, m, prefix, 3, false)
			target, serial := m.NewSession(), m.NewSession()
			defer target.Close()
			defer serial.Close()
			target.captureTargetHidden, serial.captureTargetHidden = true, true
			boundary := target.Prefill(prefix)
			serial.Prefill(prefix)
			normalizeSnapshotForTest(t, target)
			normalizeSnapshotForTest(t, serial)
			checkpoint := &qwen38InCacheCleanupCheckpoint{restore: func() {}, close: func() {
				if breakRestore {
					target.cacheGeometryEpoch++
				}
				panic("commit retirement failure")
			}}
			old := qwen35MTPMetalP4Verify
			t.Cleanup(func() { qwen35MTPMetalP4Verify = old })
			qwen35MTPMetalP4Verify = func(s *Session, ids []int) ([][]float32, Qwen35MetalMTPCheckpoint, TargetVerificationReceipt, bool, error) {
				rows, receipt, err := s.verifyQwen35MTPPanel(ids, boundary)
				return rows, checkpoint, receipt, true, err
			}
			var receipt *Qwen38MTPBatchedReceipt
			var err error
			returned := false
			var caught any
			func() {
				defer func() { caught = recover() }()
				receipt, err = VerifyMTPBlockBatched(target, draft, boundary)
				returned = true
			}()
			if checkpoint.closeCalls != 1 {
				t.Fatal("commit checkpoint was not retired exactly once")
			}
			if breakRestore {
				cause, ok := caught.(error)
				if returned || !ok || !strings.Contains(cause.Error(), "commit retirement failure") || !strings.Contains(cause.Error(), "stale prefix snapshot") {
					t.Fatalf("unsafe restore became retryable or lost its primary cause: returned=%v panic=%v", returned, caught)
				}
				return
			}
			if !returned || caught != nil || receipt != nil || err == nil || !strings.Contains(err.Error(), "commit retirement failure") {
				t.Fatalf("retirement failure lost: returned=%v receipt=%+v err=%v panic=%v", returned, receipt, err, caught)
			}
			assertQwen35MTPTargetStateEqual(t, target, serial)
			assertTargetHiddenHistoryMatches(t, target, serial, prefix)
			// This is the persistent caller's scalar retry after an ordinary error.
			next := argmaxF32(boundary)
			assertFloat32BitsEqual(t, "safe scalar retry", target.Step(next), serial.Step(next))
		})
	}
}

func qwen38InCacheDraft(t *testing.T, m *Model, prefix []int, accepted int, scalarPrefix bool) []int {
	t.Helper()
	s := m.NewSession()
	defer s.Close()
	var logits []float32
	if scalarPrefix {
		for _, token := range prefix {
			logits = s.Step(token)
		}
	} else {
		logits = s.Prefill(prefix)
	}
	draft := make([]int, 3)
	for i := range draft {
		draft[i] = argmaxF32(logits)
		logits = s.Step(draft[i])
	}
	if accepted < len(draft) {
		draft[accepted] = (draft[accepted] + 1) % m.Cfg.VocabSize
	}
	return draft
}

// fak-test:runtime fast est=3s lane=default
func TestQwen38MTP_InCacheVerification(t *testing.T) {
	prefix := []int{3, 7, 11, 5, 17, 19}
	for _, tc := range []struct {
		name           string
		hybrid, hidden bool
	}{{"hybrid", true, true}, {"generic", false, false}, {"generic_hidden", false, true}} {
		for _, accepted := range []int{3, 2, 0} {
			t.Run(tc.name+"/accept_"+itoa(accepted), func(t *testing.T) {
				m := NewSynthetic(cfgV(32, 2, 4, 2, 8, 64))
				if tc.hybrid {
					m = qwen38HybridMTPEnabledSyntheticModel(t)
				}
				scalarPrefix := tc.name == "generic_hidden"
				draft := qwen38InCacheDraft(t, m, prefix, accepted, scalarPrefix)
				target, serial := m.NewSession(), m.NewSession()
				defer target.Close()
				defer serial.Close()
				target.captureTargetHidden, serial.captureTargetHidden = tc.hidden, tc.hidden
				var boundary, want []float32
				if scalarPrefix {
					// Generic batched Prefill does not capture raw hidden history.
					// Seed an actual captured prefix without changing that API here.
					for _, token := range prefix {
						boundary, want = target.Step(token), serial.Step(token)
					}
				} else {
					boundary, want = target.Prefill(prefix), serial.Prefill(prefix)
				}
				normalizeSnapshotForTest(t, target)
				normalizeSnapshotForTest(t, serial)
				for _, token := range draft[:accepted] {
					want = serial.Step(token)
				}
				target.PhaseProfiler = NewPhaseProfiler()
				r, err := VerifyMTPBlockBatched(target, draft, boundary)
				if err != nil {
					t.Fatal(err)
				}
				wantOneOperation := tc.name != "generic_hidden"
				if err := r.Validate(); err != nil || r.AcceptedCount != accepted || r.OneOperation != wantOneOperation {
					t.Fatalf("acceptance/receipt=%+v; %v", r, err)
				}
				path, replay, repair := qwen38InCacheWork(t, r)
				wantPath := targetVerificationBatchedPath
				if tc.hybrid {
					wantPath = targetVerificationQwen38PanelPath
				}
				if !wantOneOperation {
					wantPath = targetVerificationDecodePath
					if r.TargetDecodeSteps != len(draft) || r.FallbackEngine != Qwen38EngineTargetDecode {
						t.Fatal("hidden-history fallback did not report actual scalar verification")
					}
				}
				wantReplay := accepted
				if accepted == len(draft) {
					wantReplay = 0
					if wantOneOperation && (target.PhaseProfiler.stat["mlp_decode"] != nil || target.PhaseProfiler.stat["qwen35_linear_step_in_proj"] != nil) {
						t.Fatal("full acceptance replayed ordinary target steps")
					}
				}
				if path != wantPath || replay != wantReplay || repair != 0 {
					t.Fatalf("path/replay/repair=%q/%d/%d, want %q/%d/0", path, replay, repair, wantPath, wantReplay)
				}
				if target.Cache.Len() != len(prefix)+accepted || !reflect.DeepEqual(target.Cache.Clone(), serial.Cache.Clone()) {
					t.Fatal("committed target cache differs from exact accepted prefix")
				}
				assertFloat32BitsEqual(t, "accepted boundary", r.AcceptedLogits, want)
				if tc.hybrid {
					assertQwen35MTPTargetStateEqual(t, target, serial)
				}
				if tc.hidden {
					committed := append(append([]int(nil), prefix...), draft[:accepted]...)
					assertTargetHiddenHistoryMatches(t, target, serial, committed)
				}
				assertFloat32BitsEqual(t, "continuation", target.Step(23), serial.Step(23))
			})
		}
	}

	// The injected provider uses a host numerical oracle, not physical GPU
	// execution. Its independent call counters make serial replay observable.
	for _, accepted := range []int{3, 2, 0} {
		t.Run("device_contract/accept_"+itoa(accepted), func(t *testing.T) {
			m := qwen38HybridMTPEnabledSyntheticModel(t)
			draft := qwen38InCacheDraft(t, m, prefix, accepted, false)
			target, backend := newMTPRawHiddenPartialSession(t, m)
			boundary := target.Prefill(prefix)
			backend.calls, backend.gdnCalls = 0, 0
			r, err := VerifyMTPBlockBatched(target, draft, boundary)
			if err != nil {
				t.Fatal(err)
			}
			if backend.calls != 1 || backend.gdnCalls != 0 || r.AcceptedCount != accepted {
				t.Fatalf("sequence/scalar/accepted=%d/%d/%d, want 1/0/%d", backend.calls, backend.gdnCalls, r.AcceptedCount, accepted)
			}
			path, replay, repair := qwen38InCacheWork(t, r)
			wantRepair := 0
			if accepted > 0 && accepted < len(draft) {
				wantRepair = accepted
			}
			if path != targetVerificationQwen38DevicePath || replay != 0 || repair != wantRepair {
				t.Fatalf("device work=%q/%d/%d, want incremental/0/%d", path, replay, repair, wantRepair)
			}
			if len(backend.checkpoints) != 1 || backend.checkpoints[0].closeCalls != 1 {
				t.Fatal("prefix checkpoint was not retired exactly once")
			}
			serial := m.NewSession()
			defer serial.Close()
			serial.captureTargetHidden = true
			want := serial.Prefill(prefix)
			for _, token := range draft[:accepted] {
				want = serial.Step(token)
			}
			assertFloat32BitsEqual(t, "device accepted boundary", r.AcceptedLogits, want)
			committed := append(append([]int(nil), prefix...), draft[:accepted]...)
			assertDeviceStateMatchesSerial(t, target, serial, committed)
			assertTargetHiddenHistoryMatches(t, target, serial, committed)
			assertFloat32BitsEqual(t, "device continuation", target.Step(23), serial.Step(23))
		})
	}

	t.Run("missing_raw_hidden_keeps_scalar_fallback", func(t *testing.T) {
		m := qwen38HybridMTPEnabledSyntheticModel(t)
		draft := qwen38InCacheDraft(t, m, prefix, 3, false)
		target, backend := newDevicePanelTestSession(t, m)
		target.captureTargetHidden = true
		boundary := target.Prefill(prefix)
		backend.calls, backend.gdnCalls = 0, 0
		r, err := VerifyMTPBlockBatched(target, draft, boundary)
		if err != nil {
			t.Fatal(err)
		}
		path, replay, repair := qwen38InCacheWork(t, r)
		if r.OneOperation || r.FallbackEngine != Qwen38EngineTargetDecode || r.TargetDecodeSteps != len(draft) ||
			path != targetVerificationDecodePath || replay != 0 || repair != 0 || backend.calls != 0 || backend.gdnCalls == 0 {
			t.Fatalf("missing raw-hidden support mislabeled: %+v; path/replay/repair=%q/%d/%d", r, path, replay, repair)
		}
		serial := m.NewSession()
		defer serial.Close()
		serial.captureTargetHidden = true
		serial.Prefill(prefix)
		for _, token := range draft {
			serial.Step(token)
		}
		for pos := range len(prefix) + len(draft) {
			got, err := target.TargetHiddenAt(pos)
			if err != nil {
				t.Fatal(err)
			}
			want, err := serial.TargetHiddenAt(pos)
			if err != nil {
				t.Fatal(err)
			}
			assertFloat32BitsEqual(t, "scalar raw hidden", got, want)
		}
		assertFloat32BitsEqual(t, "scalar hidden continuation", target.Step(23), serial.Step(23))
		got, err := target.TargetHiddenAt(len(prefix) + len(draft))
		if err != nil {
			t.Fatal(err)
		}
		want, err := serial.TargetHiddenAt(len(prefix) + len(draft))
		if err != nil {
			t.Fatal(err)
		}
		assertFloat32BitsEqual(t, "continued raw hidden", got, want)
	})
}
