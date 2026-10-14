package localbench

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime integration est=30ms lane=default
func TestV41EPOwnerFollowerLoopback(t *testing.T) {
	for _, mode := range []string{"healthy-shutdown", "close-panic-after-shutdown", "cancel-after-both-claims"} {
		t.Run(mode, func(t *testing.T) {
			f := newV41EPGroupTCPFixture(t, 3)
			primary := errors.New("loopback follower Close interrupted")
			var followers [3]*v41EPTCPFollower
			for rank := 1; rank < 3; rank++ {
				var closePanic error
				if mode == "close-panic-after-shutdown" && rank == 1 {
					closePanic = primary
				}
				followers[rank] = newV41EPTCPFollower(t, f, rank, closePanic)
			}
			rootJoin := f.startJoin(0, f.req.Budgets.JoinNS)
			for rank := 1; rank < 3; rank++ {
				followers[rank].start(f, rank)
			}
			if joined := f.await(t, rootJoin); joined.err() != nil || joined.panic != nil {
				f.fail(t, "root Join failed: %v panic=%v", joined.err(), joined.panic)
			}
			rootWork := f.startPhase(0, f.req.Budgets.ForwardNS, func(ctx context.Context) error {
				group := f.groups[0].Value.(*model.DistComm)
				coordinator, err := model.NewEPDecodeCoordinator(group)
				if err != nil {
					return err
				}
				// Empty Session.Prefill returns before announcing. This is an
				// independent nine-byte PREFILL control, not a model forward or
				// generated-token observation. It makes followers own a Session.
				frame := [9]byte{1}
				if _, err := group.BroadcastFromRoot(frame[:]); err != nil {
					return err
				}
				for rank := 1; rank < 3; rank++ {
					select {
					case <-followers[rank].claimed:
					case <-ctx.Done():
						return ctx.Err()
					}
				}
				if mode == "cancel-after-both-claims" {
					f.cancel()
					return ctx.Err()
				}
				return coordinator.Shutdown()
			})
			root := f.await(t, rootWork)
			var outcomes [3]v41EPOwnerLifecycleOutcome
			var lifecycleCloses [3]int
			for rank := 1; rank < 3; rank++ {
				outcomes[rank] = v41EPTCPAwaitFollower(t, f, followers[rank].done)
				lifecycleCloses[rank] = followers[rank].groupCloses
			}
			// Every full lifecycle and root operation has joined. The fixture
			// retires root raw sockets last, before any final counters/assertions.
			f.retire(t)
			if mode == "cancel-after-both-claims" {
				if !errors.Is(root.err(), context.Canceled) || root.panic != nil {
					t.Fatalf("root cancellation changed: %v panic=%v", root.err(), root.panic)
				}
			} else if root.err() != nil || root.panic != nil {
				t.Fatalf("root Shutdown failed: %v panic=%v", root.err(), root.panic)
			}
			preamble := uint64(len(v41EPTestPreamble(t, f.req, 0)))
			for rank := 1; rank < 3; rank++ {
				peer, out := followers[rank], outcomes[rank]
				if peer.factories != 1 || peer.claims != 1 || peer.observations != 1 || peer.runnerCloses != 0 ||
					peer.backend.ownerCalls != 4 || peer.backend.flushes != 1 || lifecycleCloses[rank] != 1 ||
					peer.active.Load() || peer.fake.count("model") != 1 || peer.fake.count("backend") != 1 ||
					out.Receipt.ExecutionJoined == nil || !*out.Receipt.ExecutionJoined || out.Receipt.CleanupCompleted == nil ||
					out.Receipt.RootWork.PrefillTokens != nil || out.Receipt.RootWork.GeneratedTokenIDs != nil {
					t.Fatalf("rank%d ownership/join/receipt changed: peer=%+v outcome=%+v", rank, peer, out)
				}
				if !peer.backend.owner.TryLock() {
					t.Fatal("follower Session execution owner remained locked")
				}
				peer.backend.owner.Unlock()
				closeFailed := mode == "close-panic-after-shutdown" && rank == 1
				switch {
				case closeFailed:
					var carrier *model.EPFollowerPanicError
					if !errors.Is(out.Err, primary) || !errors.As(out.Err, &carrier) || carrier.PanicValue() != primary ||
						out.PrimaryPanic != nil || out.Receipt.Status != "incomplete" || *out.Receipt.CleanupCompleted ||
						len(out.Receipt.CleanupFailures) != 1 || peer.backend.retirementComplete || peer.backend.recycles != 0 {
						t.Fatalf("rank%d interrupted Close was credited: %+v", rank, out)
					}
				case mode == "cancel-after-both-claims":
					if !errors.Is(out.Err, context.Canceled) || out.Receipt.Status != "incomplete" || !*out.Receipt.CleanupCompleted ||
						len(out.Receipt.CleanupFailures) != 0 || !peer.backend.retirementComplete || peer.backend.recycles != 1 {
						t.Fatalf("rank%d canceled execution lost proven cleanup: %+v", rank, out)
					}
				default:
					if out.Err != nil || out.Receipt.Status != "complete" || !*out.Receipt.CleanupCompleted ||
						len(out.Receipt.CleanupFailures) != 0 || !peer.backend.retirementComplete || peer.backend.recycles != 1 {
						t.Fatalf("rank%d healthy follower failed: %+v", rank, out)
					}
				}
				counts, ok := f.groups[rank].SocketCounts()
				if !ok || out.Receipt.SocketIO == nil || *out.Receipt.SocketIO != counts {
					t.Fatalf("rank%d receipt differs from joined exact raw snapshot: %+v %+v", rank, out.Receipt.SocketIO, counts)
				}
				if mode == "cancel-after-both-claims" {
					// The next five-byte request may be interrupted. Preserve its
					// actual n/error observation, without inventing frame completion.
					if counts.ReadBytes != preamble+14 || counts.WrittenBytes < preamble+4+5 || counts.WrittenBytes > preamble+4+10 || counts.ReadErrors+counts.WriteErrors > 1 {
						t.Fatalf("rank%d partial traffic outside bounded control: %+v", rank, counts)
					}
				} else if counts != (V41EPOwnerSocketCounters{ReadBytes: preamble + 28, WrittenBytes: preamble + 4 + 10}) {
					t.Fatalf("rank%d complete wire totals changed: %+v", rank, counts)
				}
			}
			rootCounts, ok := f.groups[0].SocketCounts()
			wantRoot := V41EPOwnerSocketCounters{ReadBytes: 2 * (preamble + 4 + 10), WrittenBytes: 2 * (preamble + 28)}
			if mode == "cancel-after-both-claims" {
				wantRoot = V41EPOwnerSocketCounters{ReadBytes: 2 * (preamble + 4 + 5), WrittenBytes: 2 * (preamble + 14)}
			}
			if !ok || rootCounts != wantRoot {
				t.Fatalf("root control traffic changed: %+v available=%v want=%+v", rootCounts, ok, wantRoot)
			}
			f.assertRetired(t, 2)
		})
	}
}

// Reuse the qualified fake admission/load hooks only as test dependencies. This
// is real socket/protocol/Session lifecycle composition, not native artifact,
// supervisor or model-forward qualification. No fixture outcome is published as
// a benchmark receipt, and there is no new production entry point.
type v41EPTCPFollower struct {
	fake                                                       *v41EPLifecycleFake
	backend                                                    *v41EPFollowerRetirementBackend
	session                                                    *model.Session
	result                                                     model.EPFollowerResult
	claimed                                                    chan struct{}
	done                                                       chan v41EPOwnerLifecycleOutcome
	active                                                     atomic.Bool
	factories, claims, observations, runnerCloses, groupCloses int
}

func newV41EPTCPFollower(t *testing.T, fixture *v41EPGroupTCPFixture, rank int, closePanic error) *v41EPTCPFollower {
	t.Helper()
	peer := &v41EPTCPFollower{fake: newV41EPLifecycleFake(), backend: &v41EPFollowerRetirementBackend{closePanic: closePanic}, claimed: make(chan struct{}), done: make(chan v41EPOwnerLifecycleOutcome, 1)}
	f := peer.fake
	f.req, f.group = fixture.req, fixture.groups[rank]
	f.base, f.deps.Now = time.Now(), time.Now
	f.backend.Identity.Device = f.req.Owners[rank].ExpectedDevice
	groupClose := f.group.Close
	f.group.Close = func() error {
		if peer.active.Load() {
			return errors.New("group teardown before follower executor joined")
		}
		peer.groupCloses++
		return groupClose()
	}
	f.deps.Load = func(context.Context, V41EPOwnerRequest, *v41EPOwnerLifecycleBackend, *v41EPOwnerLifecycleGroup) (*v41EPOwnerLifecycleLoaded, error) {
		f.event("load")
		peer.session = &model.Session{Backend: peer.backend}
		f.loaded.Value = peer.session
		return f.loaded, nil
	}
	f.loaded.CloseSession = func() error {
		peer.runnerCloses++
		peer.session.Close()
		return nil
	}
	f.loaded.FollowerRetirement = func() (model.EPFollowerRetirement, error) {
		peer.observations++
		if peer.active.Load() {
			return model.EPFollowerRetirement{}, errors.New("retirement observed before follower joined")
		}
		return peer.result.Retirement, nil
	}
	f.deps.Follower = func(_ context.Context, loaded *v41EPOwnerLifecycleLoaded, claim func() error) error {
		peer.active.Store(true)
		defer peer.active.Store(false)
		peer.result = model.RunEPFollowerWithResult(f.group.Value.(*model.DistComm), func() *model.Session {
			peer.factories++
			if err := claim(); err != nil {
				panic(err)
			}
			peer.claims++
			close(peer.claimed)
			return loaded.Value.(*model.Session)
		})
		return peer.result.Err
	}
	t.Cleanup(func() {
		fixture.close() // Joins every user before examining emergency ownership.
		if peer.session != nil && peer.backend.flushes == 0 {
			if err, _ := v41EPOwnerInvoke(func() error { peer.session.Close(); return nil }); err != nil {
				t.Errorf("emergency follower Session retirement: %v", err)
			}
		}
	})
	return peer
}

func (p *v41EPTCPFollower) start(f *v41EPGroupTCPFixture, rank int) {
	f.users.Add(1)
	go func() {
		defer f.users.Done()
		p.done <- runV41EPOwnerLifecycle(f.ctx, f.req, rank, p.fake.deps)
	}()
}

func v41EPTCPAwaitFollower(t *testing.T, f *v41EPGroupTCPFixture, done <-chan v41EPOwnerLifecycleOutcome) v41EPOwnerLifecycleOutcome {
	t.Helper()
	timer := time.NewTimer(5 * time.Second)
	defer timer.Stop()
	select {
	case outcome := <-done:
		return outcome
	case <-timer.C:
		f.fail(t, "full follower lifecycle did not return within bounded wait")
		return v41EPOwnerLifecycleOutcome{}
	}
}
