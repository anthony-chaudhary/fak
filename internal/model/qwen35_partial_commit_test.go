package model

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type partialCommitTestBackend struct {
	*devicePanelTestBackend
	checkpoints            []*partialCommitTestReplay
	failCommit             bool
	panicReadAfterSequence bool
	sequenceReturned       bool
}

type partialCommitTestReplay struct {
	backend     *partialCommitTestBackend
	prefix      []int
	draft       []int
	states      []compute.Qwen35SequenceState
	beforeKV    compute.KVStore
	owned       compute.Tensor
	failCommit  bool
	commitCalls int
	closeCalls  int
	closed      bool
	beforeKept  bool
}

func newPartialCommitTestSession(t *testing.T, m *Model) (*Session, *partialCommitTestBackend) {
	t.Helper()
	backend := &partialCommitTestBackend{devicePanelTestBackend: newDevicePanelTestBackend(m)}
	target, err := m.NewBackendSessionChecked(backend)
	if err != nil {
		t.Fatal(err)
	}
	backend.target = target
	backend.devicePanelTestBackend.target = target
	target.captureTargetHidden = false
	t.Cleanup(target.Close)
	return target, backend
}

func (*partialCommitTestBackend) Qwen35SequencePrefixReplayPath() string {
	return compute.Qwen35SequencePrefixReplayPath
}

func (b *partialCommitTestBackend) Qwen35SequencePrefill(req compute.Qwen35SequencePrefillRequest) (compute.Qwen35SequencePrefillResult, error) {
	prefix := make([]int, len(b.target.halLineage.ids))
	for i, id := range b.target.halLineage.ids {
		prefix[i] = int(id)
	}
	var beforeKV compute.KVStore
	if req.CapturePrefixReplay {
		beforeKV = b.target.halKV.Clone()
	}
	result, err := b.devicePanelTestBackend.Qwen35SequencePrefill(req)
	if err != nil || !req.CapturePrefixReplay {
		if beforeKV != nil {
			beforeKV.Free()
		}
		return result, err
	}
	replay := &partialCommitTestReplay{
		backend:    b,
		prefix:     prefix,
		draft:      append([]int(nil), req.TokenIDs...),
		states:     append([]compute.Qwen35SequenceState(nil), req.States...),
		beforeKV:   beforeKV,
		owned:      compute.NewF32(b, []int{1}, []float32{float32(len(req.TokenIDs))}),
		failCommit: b.failCommit,
	}
	b.checkpoints = append(b.checkpoints, replay)
	result.PrefixReplay = replay
	b.sequenceReturned = true
	return result, nil
}

func (b *partialCommitTestBackend) Read(t compute.Tensor) []float32 {
	if b.panicReadAfterSequence && b.sequenceReturned {
		panic("injected logits read failure")
	}
	return b.devicePanelTestBackend.Read(t)
}

func (r *partialCommitTestReplay) CommitPrefix(accepted int, before []compute.Qwen35SequenceState) error {
	r.commitCalls++
	if r.closed {
		return errors.New("test prefix replay is closed")
	}
	if accepted <= 0 || accepted >= len(r.draft) {
		return errors.New("test prefix replay received a non-partial cut")
	}
	if len(before) != len(r.states) {
		return errors.New("test prefix replay received malformed before-state set")
	}
	if r.failCommit {
		return errors.New("injected recurrent-only commit failure")
	}
	beforeConv := make([][]float32, len(before))
	beforeRecurrent := make([][]float32, len(before))
	for layer := range before {
		if before[layer].Conv.Buf() == nil {
			continue
		}
		beforeConv[layer] = append([]float32(nil), r.backend.Read(before[layer].Conv)...)
		beforeRecurrent[layer] = append([]float32(nil), r.backend.Read(before[layer].Recurrent)...)
	}

	// This injected capability uses the ordinary CPU model only as an independent
	// oracle for the expected linear states. It writes those states directly and
	// trims the already-produced attention KV tail, so Session.Step is never part
	// of the commit implementation being observed by the test.
	reference := r.backend.model.NewSession()
	reference.captureTargetHidden = true
	reference.Prefill(append(append([]int(nil), r.prefix...), r.draft[:accepted]...))
	defer reference.Close()
	for layer := range r.states {
		if !r.backend.model.Cfg.isLinearAttnLayer(layer) {
			continue
		}
		want := reference.Cache.linear.layers[layer]
		copy(r.backend.Read(r.states[layer].Conv), flattenRows(want.conv))
		copy(r.backend.Read(r.states[layer].Recurrent), flattenRows(want.recurrent))
	}
	oldKV := r.backend.target.halKV
	r.backend.target.halKV = r.beforeKV
	r.beforeKV = nil
	oldKV.Free()
	width := r.backend.model.Cfg.NumKVHeads * r.backend.model.Cfg.HeadDim
	for token := 0; token < accepted; token++ {
		pos := len(r.prefix) + token
		lo, hi := pos*width, (pos+1)*width
		for layer := 0; layer < r.backend.model.Cfg.NumLayers; layer++ {
			if r.backend.model.Cfg.isLinearAttnLayer(layer) {
				continue
			}
			compact := qwen35HALKVLayer(r.backend.model.Cfg, layer)
			raw := compute.NewF32(r.backend, []int{width}, reference.Cache.Kraw[layer][lo:hi])
			key := compute.NewF32(r.backend, []int{width}, reference.Cache.K[layer][lo:hi])
			value := compute.NewF32(r.backend, []int{width}, reference.Cache.V[layer][lo:hi])
			r.backend.target.halKV.AppendKV(compact, raw, key, value, pos)
			r.backend.Free(raw)
			r.backend.Free(key)
			r.backend.Free(value)
		}
	}
	r.beforeKept = true
	for layer := range before {
		if before[layer].Conv.Buf() == nil {
			continue
		}
		if !reflect.DeepEqual(beforeConv[layer], r.backend.Read(before[layer].Conv)) || !reflect.DeepEqual(beforeRecurrent[layer], r.backend.Read(before[layer].Recurrent)) {
			r.beforeKept = false
			return errors.New("test prefix replay mutated borrowed before-state")
		}
	}
	return nil
}

func (r *partialCommitTestReplay) Close() {
	if r.closed {
		return
	}
	r.closed = true
	r.closeCalls++
	if r.beforeKV != nil {
		r.beforeKV.Free()
		r.beforeKV = nil
	}
	r.backend.Free(r.owned)
}

// TestQwen35DevicePartialCommitAvoidsFullTargetReplay is the behavioral
// regression for partial device-panel adoption. The injected backend evaluates
// the same sequence operation as the production path; the independent serial
// session remains the state and continuation oracle. Counting tx.step makes a
// full target replay observable without trusting receipt bookkeeping.
func TestQwen35DevicePartialCommitAvoidsFullTargetReplay(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	prefix := []int{3, 7, 11, 5, 17, 19}
	draft := []int{23, 2, 29, 31}

	for accepted := 1; accepted < len(draft); accepted++ {
		t.Run(itoa(accepted)+"-accepted", func(t *testing.T) {
			target, backend := newPartialCommitTestSession(t, m)
			boundary := target.Prefill(prefix)

			tx, err := beginQwen35MTPTargetTransaction(target, boundary)
			if err != nil {
				t.Fatal(err)
			}
			rows, err := tx.Verify(draft)
			if err != nil {
				t.Fatal(err)
			}

			verified := m.NewSession()
			verified.Prefill(prefix)
			for row, token := range draft {
				assertFloat32BitsEqual(t, "verified row", rows[row], verified.Step(token))
			}
			verified.Close()

			steps := 0
			tx.step = func(token int) []float32 {
				steps++
				return target.Step(token)
			}
			gotBoundary, err := tx.Commit(accepted)
			if err != nil {
				t.Fatal(err)
			}
			if steps != 0 {
				t.Fatalf("accepted=%d full target replay steps=%d, want 0", accepted, steps)
			}
			receipt := tx.VerificationReceipt()
			if receipt.FullTargetReplaySteps != 0 || receipt.RecurrentRepairTokens != accepted {
				t.Fatalf("accepted=%d replay/repair receipt=%d/%d, want 0/%d", accepted, receipt.FullTargetReplaySteps, receipt.RecurrentRepairTokens, accepted)
			}
			if len(backend.checkpoints) != 1 {
				t.Fatalf("retained prefix checkpoints=%d, want 1", len(backend.checkpoints))
			}
			replay := backend.checkpoints[0]
			if replay.commitCalls != 1 || replay.closeCalls != 1 || !replay.closed || !replay.beforeKept || backend.freeCalls[replay.owned.Buf()] != 1 {
				t.Fatalf("checkpoint lifecycle commit=%d close=%d closed=%t frees=%d, want 1/1/true/1", replay.commitCalls, replay.closeCalls, replay.closed, backend.freeCalls[replay.owned.Buf()])
			}
			replay.Close()
			if replay.closeCalls != 1 || backend.freeCalls[replay.owned.Buf()] != 1 {
				t.Fatalf("idempotent close count/frees=%d/%d, want 1/1", replay.closeCalls, backend.freeCalls[replay.owned.Buf()])
			}

			serial := m.NewSession()
			defer serial.Close()
			serial.Prefill(prefix)
			var wantBoundary []float32
			for _, token := range draft[:accepted] {
				wantBoundary = serial.Step(token)
			}
			assertFloat32BitsEqual(t, "committed boundary", gotBoundary, wantBoundary)
			committed := append(append([]int(nil), prefix...), draft[:accepted]...)
			assertDeviceStateMatchesSerial(t, target, serial, committed)

			continuation := (accepted*7 + 3) % m.Cfg.VocabSize
			assertFloat32BitsEqual(t, "continuation", target.Step(continuation), serial.Step(continuation))
		})
	}
}

func TestQwen35PartialCommitCheckpointLifecycleAtNonPartialExits(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	prefix := []int{3, 7, 11, 5, 17, 19}
	draft := []int{23, 2, 29, 31}

	for _, accepted := range []int{0, len(draft)} {
		t.Run(itoa(accepted)+"-accepted", func(t *testing.T) {
			target, backend := newPartialCommitTestSession(t, m)
			boundary := target.Prefill(prefix)
			before, err := target.PrefixSnapshot()
			if err != nil {
				t.Fatal(err)
			}
			defer before.Close()
			tx, err := beginQwen35MTPTargetTransaction(target, boundary)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := tx.Verify(draft); err != nil {
				t.Fatal(err)
			}
			steps := 0
			tx.step = func(token int) []float32 { steps++; return target.Step(token) }
			if _, err := tx.Commit(accepted); err != nil {
				t.Fatal(err)
			}
			if steps != 0 || len(backend.checkpoints) != 1 {
				t.Fatalf("accepted=%d steps/checkpoints=%d/%d, want 0/1", accepted, steps, len(backend.checkpoints))
			}
			receipt := tx.VerificationReceipt()
			if receipt.FullTargetReplaySteps != 0 || receipt.RecurrentRepairTokens != 0 {
				t.Fatalf("accepted=%d replay/repair receipt=%d/%d, want 0/0", accepted, receipt.FullTargetReplaySteps, receipt.RecurrentRepairTokens)
			}
			replay := backend.checkpoints[0]
			if replay.commitCalls != 0 || replay.closeCalls != 1 || backend.freeCalls[replay.owned.Buf()] != 1 {
				t.Fatalf("accepted=%d checkpoint commit/close/free=%d/%d/%d, want 0/1/1", accepted, replay.commitCalls, replay.closeCalls, backend.freeCalls[replay.owned.Buf()])
			}
			if accepted == 0 {
				after, err := target.PrefixSnapshot()
				if err != nil {
					t.Fatal(err)
				}
				defer after.Close()
				assertPrefixSnapshotsEqual(t, after, before)
			}
		})
	}

	t.Run("abort", func(t *testing.T) {
		target, backend := newPartialCommitTestSession(t, m)
		boundary := target.Prefill(prefix)
		before, err := target.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		defer before.Close()
		tx, err := beginQwen35MTPTargetTransaction(target, boundary)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := tx.Verify(draft); err != nil {
			t.Fatal(err)
		}
		if err := tx.Abort(); err != nil {
			t.Fatal(err)
		}
		after, err := target.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		defer after.Close()
		assertPrefixSnapshotsEqual(t, after, before)
		if len(backend.checkpoints) != 1 {
			t.Fatalf("abort checkpoints=%d, want 1", len(backend.checkpoints))
		}
		replay := backend.checkpoints[0]
		if replay.commitCalls != 0 || replay.closeCalls != 1 || backend.freeCalls[replay.owned.Buf()] != 1 {
			t.Fatalf("abort checkpoint commit/close/free=%d/%d/%d, want 0/1/1", replay.commitCalls, replay.closeCalls, backend.freeCalls[replay.owned.Buf()])
		}
	})

	t.Run("cancellation-after-panel", func(t *testing.T) {
		target, backend := newPartialCommitTestSession(t, m)
		boundary := target.Prefill(prefix)
		before, err := target.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		defer before.Close()
		first := argmaxF32(boundary)
		serial := m.NewSession()
		serial.Prefill(prefix)
		second := argmaxF32(serial.Step(first))
		serial.Close()
		ctx, cancel := context.WithCancel(context.Background())
		backend.afterSequence = cancel
		if _, err := target.VerifyGreedyDeviceDraft(ctx, []int{first, second}, boundary); !errors.Is(err, context.Canceled) {
			t.Fatalf("verification error=%v, want context cancellation", err)
		}
		after, err := target.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		defer after.Close()
		assertPrefixSnapshotsEqual(t, after, before)
		if len(backend.checkpoints) != 1 {
			t.Fatalf("cancellation checkpoints=%d, want 1", len(backend.checkpoints))
		}
		replay := backend.checkpoints[0]
		if replay.commitCalls != 0 || replay.closeCalls != 1 || backend.freeCalls[replay.owned.Buf()] != 1 {
			t.Fatalf("cancellation checkpoint commit/close/free=%d/%d/%d, want 0/1/1", replay.commitCalls, replay.closeCalls, backend.freeCalls[replay.owned.Buf()])
		}
	})
}

func TestQwen35PartialCommitFallbackAndFailureStayAtomic(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	prefix := []int{3, 7, 11, 5, 17, 19}
	draft := []int{23, 2, 29, 31}

	t.Run("unsupported-backend-retains-full-replay", func(t *testing.T) {
		target, backend := newDevicePanelTestSession(t, m)
		target.captureTargetHidden = false
		boundary := target.Prefill(prefix)
		tx, err := beginQwen35MTPTargetTransaction(target, boundary)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := tx.Verify(draft); err != nil {
			t.Fatal(err)
		}
		steps := 0
		tx.step = func(token int) []float32 { steps++; return target.Step(token) }
		if _, err := tx.Commit(2); err != nil {
			t.Fatal(err)
		}
		if steps != 2 || backend.requests[len(backend.requests)-1].CapturePrefixReplay {
			t.Fatalf("unsupported fallback steps/capture=%d/%t, want 2/false", steps, backend.requests[len(backend.requests)-1].CapturePrefixReplay)
		}
		receipt := tx.VerificationReceipt()
		if receipt.FullTargetReplaySteps != 2 || receipt.RecurrentRepairTokens != 0 {
			t.Fatalf("unsupported fallback replay/repair receipt=%d/%d, want 2/0", receipt.FullTargetReplaySteps, receipt.RecurrentRepairTokens)
		}
	})

	t.Run("raw-hidden-capture-declines-before-device-mutation", func(t *testing.T) {
		target, backend := newPartialCommitTestSession(t, m)
		target.captureTargetHidden = true
		boundary := target.Prefill(prefix)
		calls := backend.calls
		before, err := target.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		defer before.Close()
		if rows, _, err := target.verifyQwen35DevicePanel(draft, boundary); err == nil || rows != nil {
			t.Fatalf("direct hidden verification rows/error=%d/%v, want typed decline", len(rows), err)
		} else {
			var downgrade *TargetVerificationDowngradeError
			if !errors.As(err, &downgrade) || downgrade.Reason != "device target verification cannot preserve raw target-hidden history" {
				t.Fatalf("direct hidden verification error=%v", err)
			}
		}
		if backend.calls != calls {
			t.Fatalf("direct hidden verification sequence calls=%d, want %d", backend.calls, calls)
		}
		tx, err := beginQwen35MTPTargetTransaction(target, boundary)
		if err != nil {
			t.Fatal(err)
		}
		rows, err := tx.Verify(draft)
		var downgrade *TargetVerificationDowngradeError
		if !errors.As(err, &downgrade) || downgrade.Reason != "device target verification cannot preserve raw target-hidden history" || rows != nil {
			t.Fatalf("transaction hidden verification rows/error=%v/%v, want typed decline", rows, err)
		}
		if backend.calls != calls || len(backend.checkpoints) != 0 || !tx.closed || tx.snapshot != nil {
			t.Fatalf("hidden decline calls/checkpoints/closed/snapshot=%d/%d/%t/%p, want %d/0/true/nil", backend.calls, len(backend.checkpoints), tx.closed, tx.snapshot, calls)
		}
		after, err := target.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		defer after.Close()
		assertPrefixSnapshotsEqual(t, after, before)
	})

	t.Run("recurrent-commit-failure-restores-and-releases", func(t *testing.T) {
		target, backend := newPartialCommitTestSession(t, m)
		backend.failCommit = true
		boundary := target.Prefill(prefix)
		before, err := target.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		defer before.Close()
		tx, err := beginQwen35MTPTargetTransaction(target, boundary)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := tx.Verify(draft); err != nil {
			t.Fatal(err)
		}
		if _, err := tx.Commit(2); err == nil || err.Error() == "" {
			t.Fatalf("commit error=%v, want injected recurrent-only failure", err)
		}
		after, err := target.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		defer after.Close()
		assertPrefixSnapshotsEqual(t, after, before)
		if len(backend.checkpoints) != 1 {
			t.Fatalf("retained prefix checkpoints=%d, want 1", len(backend.checkpoints))
		}
		replay := backend.checkpoints[0]
		if replay.commitCalls != 1 || replay.closeCalls != 1 || backend.freeCalls[replay.owned.Buf()] != 1 {
			t.Fatalf("failed checkpoint lifecycle commit=%d close=%d frees=%d, want 1/1/1", replay.commitCalls, replay.closeCalls, backend.freeCalls[replay.owned.Buf()])
		}
	})

	t.Run("logits-read-panic-restores-and-releases-detached-checkpoint", func(t *testing.T) {
		target, backend := newPartialCommitTestSession(t, m)
		boundary := target.Prefill(prefix)
		before, err := target.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		defer before.Close()
		backend.panicReadAfterSequence = true
		tx, err := beginQwen35MTPTargetTransaction(target, boundary)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := tx.Verify(draft); err == nil || !strings.Contains(err.Error(), "injected logits read failure") {
			t.Fatalf("verification error=%v, want injected logits read panic", err)
		}
		backend.panicReadAfterSequence = false
		after, err := target.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		defer after.Close()
		assertPrefixSnapshotsEqual(t, after, before)
		if len(backend.checkpoints) != 1 {
			t.Fatalf("read panic checkpoints=%d, want 1", len(backend.checkpoints))
		}
		replay := backend.checkpoints[0]
		if replay.commitCalls != 0 || replay.closeCalls != 1 || backend.freeCalls[replay.owned.Buf()] != 1 {
			t.Fatalf("read panic checkpoint commit/close/free=%d/%d/%d, want 0/1/1", replay.commitCalls, replay.closeCalls, backend.freeCalls[replay.owned.Buf()])
		}
	})
}
