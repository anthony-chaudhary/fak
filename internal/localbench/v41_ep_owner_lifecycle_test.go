package localbench

import (
	"context"
	"encoding/binary"
	"errors"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=30ms lane=default
func TestV41EPOwnerLifecycleContract(t *testing.T) {
	t.Run("successful-root-order-and-observations", func(t *testing.T) {
		f := newV41EPLifecycleFake()
		out := runV41EPOwnerLifecycle(context.Background(), f.req, 0, f.deps)
		want := []string{"nonce", "process", "verify", "group-factory", "join", "backend-factory", "load", "session", "root", "rounds", "shutdown", "model", "stop-io", "group", "socket-counts", "backend"}
		if out.Err != nil || out.PrimaryPanic != nil || out.Receipt.Status != "complete" || !reflect.DeepEqual(f.events(), want) {
			t.Fatalf("successful lifecycle err=%v status=%s events=%v", out.Err, out.Receipt.Status, f.events())
		}
		r := out.Receipt
		if r.ExecutableSHA256 == nil || *r.ExecutableSHA256 != f.process.ExecutableSHA256 || r.SupervisorEvidenceSHA256 == nil ||
			r.ArtifactAuthenticated == nil || !*r.ArtifactAuthenticated || r.ExecutionJoined == nil || !*r.ExecutionJoined ||
			r.ShutdownCompleted == nil || !*r.ShutdownCompleted || r.CleanupCompleted == nil || !*r.CleanupCompleted ||
			r.SocketIO == nil || r.OwnerRounds == nil || r.RootWork.CompletedStepCalls == nil || *r.RootWork.CompletedStepCalls != 1 {
			t.Fatalf("missing observed success fields: %+v", r)
		}
		for _, elapsed := range []*int64{r.Phases.JoinNS, r.Phases.LoadNS, r.Phases.ForwardNS, r.Phases.CleanupNS, r.Phases.TotalNS} {
			if elapsed == nil || *elapsed < 0 {
				t.Fatal("missing measured phase")
			}
		}
		*f.work.PrefillTokens = 99
		f.work.GeneratedTokenIDs[0] = 99
		if *r.RootWork.PrefillTokens != len(f.req.PromptTokenIDs) || r.RootWork.GeneratedTokenIDs[0] == 99 {
			t.Fatal("receipt retained mutable executor aliases")
		}
	})
	t.Run("supervisor-refusal-before-resources", func(t *testing.T) {
		for _, mode := range []string{"missing-verifier", "verifier-error", "zero-challenge", "wrong-nonce", "wrong-request", "wrong-rank", "wrong-process", "wrong-contract", "stale", "late-terminate", "late-kill", "bad-evidence", "binary-mismatch"} {
			t.Run(mode, func(t *testing.T) {
				f := newV41EPLifecycleFake()
				base := f.deps.VerifySupervisor
				f.deps.VerifySupervisor = func(ctx context.Context, challenge v41EPOwnerSupervisorChallenge) (v41EPOwnerSupervisorHandoff, error) {
					h, err := base(ctx, challenge)
					switch mode {
					case "verifier-error":
						return h, errors.New("live verifier refused")
					case "wrong-nonce":
						h.Challenge.Nonce[0] ^= 1
					case "wrong-request":
						h.Challenge.RequestSHA256 = strings.Repeat("f", 64)
					case "wrong-rank":
						h.Challenge.Rank = 1
					case "wrong-process":
						h.Challenge.Process.PID++
					case "wrong-contract":
						h.Challenge.ContractID = "cached"
					case "stale":
						h.TerminateAt = f.base
					case "late-terminate":
						h.TerminateAt = challenge.LatestTerminateAt.Add(1)
					case "late-kill":
						h.KillAt = h.TerminateAt.Add(challenge.KillGrace + 1)
					case "bad-evidence":
						h.EvidenceSHA256 = strings.Repeat("a", 40)
					}
					return h, err
				}
				if mode == "missing-verifier" {
					f.deps.VerifySupervisor = nil
				}
				if mode == "zero-challenge" {
					f.deps.NewNonce = func() ([32]byte, error) { return [32]byte{}, nil }
				}
				if mode == "binary-mismatch" {
					f.req.Owners[0].ExpectedExecutableSHA256 = strings.Repeat("f", 64)
				}
				out := runV41EPOwnerLifecycle(context.Background(), f.req, 0, f.deps)
				if out.Err == nil || out.Receipt.Status != "refused" || f.has("group-factory") || f.has("backend-factory") || f.has("load") || out.Receipt.SupervisorEvidenceSHA256 != nil || out.Receipt.Phases.JoinNS != nil {
					t.Fatalf("supervisor declaration admitted resources: %+v %v", out, f.events())
				}
			})
		}
	})
	t.Run("central-phase-caps-replace-initial-join-cap", func(t *testing.T) {
		f := newV41EPLifecycleFake()
		budgets := []time.Duration{f.req.Budgets.JoinNS, f.req.Budgets.LoadNS, f.req.Budgets.ForwardNS, f.req.Budgets.CleanupNS}
		var caps []time.Time
		f.group.SetPhaseDeadline = func(deadline time.Time) error {
			if len(caps) >= len(budgets) {
				return errors.New("extra phase cap")
			}
			want := f.base.Add(time.Duration(f.ticks.Load())).Add(budgets[len(caps)])
			if !deadline.Equal(want) {
				return errors.New("phase deadline not advanced from actual phase context")
			}
			caps = append(caps, deadline)
			return nil
		}
		out := runV41EPOwnerLifecycle(context.Background(), f.req, 0, f.deps)
		if out.Err != nil || len(caps) != 4 {
			t.Fatalf("phase caps: %v %v", caps, out.Err)
		}
		for _, cap := range caps[1:] {
			if cap.Equal(caps[0]) {
				t.Fatal("old join cap survived")
			}
		}
	})
	t.Run("setter-is-cancelable-and-local-error-survives-normal-stop", func(t *testing.T) {
		for _, mode := range []string{"blocked-setter", "normal-stop"} {
			t.Run(mode, func(t *testing.T) {
				f := newV41EPLifecycleFake()
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				stopped := make(chan struct{})
				var once sync.Once
				f.group.StopIO = func(error) error { once.Do(func() { close(stopped) }); return nil }
				failure := errors.New("phase cap installation")
				calls := 0
				f.group.SetPhaseDeadline = func(time.Time) error {
					calls++
					if calls != 3 {
						return nil
					}
					if mode == "normal-stop" {
						f.group.StopIO(nil)
					} else {
						cancel()
					}
					select {
					case <-stopped:
					case <-time.After(2 * time.Second):
						return errors.New("watcher did not stop raw I/O during setter")
					}
					return failure
				}
				out := runV41EPOwnerLifecycle(ctx, f.req, 0, f.deps)
				if !errors.Is(out.Err, failure) || f.has("root") || f.count("session") != 1 || f.count("model") != 1 || f.count("group") != 1 || f.count("backend") != 1 {
					t.Fatalf("phase failure masked or leaked: %v %v", out.Err, f.events())
				}
			})
		}
	})
	t.Run("cached-handoff-cannot-admit-next-invocation", func(t *testing.T) {
		f := newV41EPLifecycleFake()
		base := f.deps.VerifySupervisor
		var saved *v41EPOwnerSupervisorHandoff
		f.deps.VerifySupervisor = func(ctx context.Context, ch v41EPOwnerSupervisorChallenge) (v41EPOwnerSupervisorHandoff, error) {
			if saved != nil {
				return *saved, nil
			}
			h, err := base(ctx, ch)
			saved = &h
			return h, err
		}
		if first := runV41EPOwnerLifecycle(context.Background(), f.req, 0, f.deps); first.Err != nil {
			t.Fatal(first.Err)
		}
		before := len(f.events())
		second := runV41EPOwnerLifecycle(context.Background(), f.req, 0, f.deps)
		if second.Err == nil || second.Receipt.Status != "refused" || len(f.events()) != before+2 {
			t.Fatalf("cached handoff reused: %v %v", second.Err, f.events()[before:])
		}
	})
	t.Run("partial-factory-transfers-and-panic-local-cleanup", func(t *testing.T) {
		for _, stage := range []string{"group", "backend", "load", "factory-panic"} {
			t.Run(stage, func(t *testing.T) {
				f := newV41EPLifecycleFake()
				failure := errors.New("partial factory")
				switch stage {
				case "group":
					f.deps.Group = func(context.Context, V41EPOwnerRequest, int, time.Time) (*v41EPOwnerLifecycleGroup, error) {
						return f.group, failure
					}
				case "backend":
					f.deps.Backend = func(context.Context, V41EPOwnerSelection) (*v41EPOwnerLifecycleBackend, error) {
						return f.backend, failure
					}
				case "load":
					f.deps.Load = func(context.Context, V41EPOwnerRequest, *v41EPOwnerLifecycleBackend, *v41EPOwnerLifecycleGroup) (*v41EPOwnerLifecycleLoaded, error) {
						return f.loaded, failure
					}
				case "factory-panic":
					f.deps.Backend = func(context.Context, V41EPOwnerSelection) (*v41EPOwnerLifecycleBackend, error) {
						defer f.event("factory-local-release")
						panic(failure)
					}
				}
				out := runV41EPOwnerLifecycle(context.Background(), f.req, 0, f.deps)
				if !errors.Is(out.Err, failure) || out.Receipt.Status != "incomplete" || f.has("root") || out.Receipt.ExecutionJoined != nil || out.Receipt.ArtifactAuthenticated != nil || out.Receipt.RootWork.PrefillTokens != nil {
					t.Fatalf("partial factory became execution: %+v", out)
				}
				if f.count("group") != 1 || stage != "group" && stage != "factory-panic" && f.count("backend") != 1 || stage == "load" && (f.count("session") != 1 || f.count("model") != 1) {
					t.Fatalf("partial bundle lost/doubled ownership: %v", f.events())
				}
				if stage == "factory-panic" && (out.PrimaryPanic != failure || f.count("factory-local-release") != 1 || f.has("backend")) {
					t.Fatal("factory panic/local ownership lost")
				}
			})
		}
	})
	t.Run("follower-claim-is-atomic-and-one-shot", func(t *testing.T) {
		for _, mode := range []string{"once", "never", "twice"} {
			t.Run(mode, func(t *testing.T) {
				f := newV41EPLifecycleFake()
				f.group.Rank = 1
				f.backend.Identity.Device = f.req.Owners[1].ExpectedDevice
				var wins atomic.Int32
				f.deps.Follower = func(ctx context.Context, loaded *v41EPOwnerLifecycleLoaded, claim func() error) error {
					f.event("follower")
					if mode == "never" {
						return nil
					}
					attempts := 1
					if mode == "twice" {
						attempts = 2
					}
					var joined sync.WaitGroup
					for i := 0; i < attempts; i++ {
						joined.Add(1)
						go func() {
							defer joined.Done()
							if claim() == nil {
								wins.Add(1)
							}
						}()
					}
					joined.Wait()
					return loaded.CloseSession()
				}
				out := runV41EPOwnerLifecycle(context.Background(), f.req, 1, f.deps)
				if (mode == "once") != (out.Err == nil) || f.count("session") != 1 || mode != "never" && wins.Load() != 1 || f.has("root") || f.has("shutdown") || out.Receipt.RootWork.PrefillTokens != nil {
					t.Fatalf("bad follower transfer mode=%s err=%v wins=%d events=%v", mode, out.Err, wins.Load(), f.events())
				}
			})
		}
	})
	t.Run("primary-errors-and-panics-survive-all-cleanup", func(t *testing.T) {
		for _, mode := range []string{"root-error", "root-panic", "initial-session-panic", "cleanup-only-panic"} {
			t.Run(mode, func(t *testing.T) {
				f := newV41EPLifecycleFake()
				primary, cleanupPanic := errors.New("primary"), errors.New("cleanup panic")
				f.loaded.CloseModel = func() error { f.event("model"); panic(cleanupPanic) }
				switch mode {
				case "root-error":
					f.deps.Root = func(context.Context, V41EPOwnerRequest, *v41EPOwnerLifecycleLoaded) (*V41EPOwnerRootWork, error) {
						f.event("root")
						return f.work, primary
					}
				case "root-panic":
					f.deps.Root = func(context.Context, V41EPOwnerRequest, *v41EPOwnerLifecycleLoaded) (*V41EPOwnerRootWork, error) {
						f.event("root")
						panic(primary)
					}
				case "initial-session-panic":
					f.loaded.CloseSession = func() error { f.event("session"); panic(primary) }
				}
				out := runV41EPOwnerLifecycle(context.Background(), f.req, 0, f.deps)
				if out.Err == nil || !errors.Is(out.Err, cleanupPanic) || out.Receipt.Status != "incomplete" || out.Receipt.CleanupCompleted == nil || *out.Receipt.CleanupCompleted || f.count("session") != 1 || f.count("model") != 1 || f.count("group") != 1 || f.count("backend") != 1 {
					t.Fatalf("cleanup masked failure or skipped release: %+v %v", out, f.events())
				}
				if mode != "cleanup-only-panic" && (!errors.Is(out.Err, primary) || f.has("shutdown") || out.Receipt.RootWork.PrefillTokens != nil) {
					t.Fatal("primary failed execution became successful")
				}
				if mode == "root-error" && out.PrimaryPanic != nil {
					t.Fatal("cleanup panic replaced primary error")
				}
				if (mode == "root-panic" || mode == "initial-session-panic") && out.PrimaryPanic != primary {
					t.Fatal("primary panic identity changed")
				}
				if mode == "cleanup-only-panic" && out.PrimaryPanic != cleanupPanic {
					t.Fatal("cleanup primary panic identity changed")
				}
			})
		}
	})
	t.Run("cancel-waits-for-executor-and-watcher-before-teardown", func(t *testing.T) {
		f := newV41EPLifecycleFake()
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		entered, release := make(chan struct{}), make(chan struct{})
		var released sync.Once
		t.Cleanup(func() { released.Do(func() { close(release) }) })
		var stopOnce sync.Once
		f.group.StopIO = func(error) error {
			stopOnce.Do(func() { f.event("stop-enter"); close(entered); <-release; f.event("stop-exit") })
			return nil
		}
		f.deps.Root = func(context.Context, V41EPOwnerRequest, *v41EPOwnerLifecycleLoaded) (*V41EPOwnerRootWork, error) {
			f.event("root")
			cancel()
			select {
			case <-entered:
			case <-time.After(2 * time.Second):
				return nil, errors.New("fake raw stop did not start")
			}
			f.event("executor-joined")
			return f.work, nil
		}
		done := make(chan v41EPOwnerLifecycleOutcome, 1)
		go func() { done <- runV41EPOwnerLifecycle(ctx, f.req, 0, f.deps) }()
		v41EPAwait(t, entered)
		if f.has("model") || f.has("group") || f.has("backend") {
			t.Fatal("teardown raced live watcher")
		}
		released.Do(func() { close(release) })
		out := v41EPAwait(t, done)
		if !errors.Is(out.Err, context.Canceled) || f.has("shutdown") || out.Receipt.RootWork.PrefillTokens != nil || out.Receipt.ExecutionJoined == nil || !*out.Receipt.ExecutionJoined {
			t.Fatalf("cancellation accepted output: %+v", out)
		}
		if !f.before("executor-joined", "model") || !f.before("stop-exit", "group") || !f.before("group", "backend") {
			t.Fatalf("join ordering changed: %v", f.events())
		}
		if out.Receipt.Status != "incomplete" || out.Receipt.CleanupCompleted == nil || !*out.Receipt.CleanupCompleted || len(out.Receipt.CleanupFailures) != 0 {
			t.Fatalf("cancellation mislabeled successful releases: %+v", out)
		}
		for _, kind := range []string{"session", "model", "group", "backend"} {
			if f.count(kind) != 1 {
				t.Fatalf("release count changed for %s: %v", kind, f.events())
			}
		}
	})
	t.Run("late-cancellation-keeps-cleanup-bounded-and-shutdown-refused", func(t *testing.T) {
		for _, mode := range []string{"cancel-after-root", "earlier-parent-deadline", "cleanup-error", "cleanup-expiry", "cancel-during-shutdown"} {
			t.Run(mode, func(t *testing.T) {
				f := newV41EPLifecycleFake()
				parent := context.Background()
				var parentDeadline time.Time
				if mode == "earlier-parent-deadline" {
					parentDeadline = f.base.Add(time.Millisecond)
					var cancelDeadline context.CancelFunc
					parent, cancelDeadline = context.WithDeadline(parent, parentDeadline)
					defer cancelDeadline()
				}
				ctx, cancel := context.WithCancel(parent)
				defer cancel()
				var caps []time.Time
				f.group.SetPhaseDeadline = func(deadline time.Time) error { caps = append(caps, deadline); return nil }
				f.loaded.OwnerCounts = func() (V41EPOwnerRoundCounters, bool) {
					f.event("rounds")
					if mode != "cancel-during-shutdown" {
						cancel()
					}
					return V41EPOwnerRoundCounters{}, true
				}
				if mode == "cancel-during-shutdown" {
					stopped := make(chan struct{})
					var once sync.Once
					f.group.StopIO = func(error) error {
						once.Do(func() { f.event("stop-io"); close(stopped) })
						return nil
					}
					f.deps.Shutdown = func(shutdownCtx context.Context, _ *v41EPOwnerLifecycleLoaded) error {
						f.event("shutdown")
						cancel()
						if shutdownCtx.Err() != context.Canceled {
							t.Error("Shutdown lost execution cancellation")
						}
						// A synchronous protocol call may wait on raw I/O rather than
						// selecting ctx.Done; the watcher must close its owned handle.
						select {
						case <-stopped:
						case <-time.After(2 * time.Second):
							t.Error("Shutdown cancellation did not interrupt raw I/O")
						}
						f.event("shutdown-joined")
						return nil
					}
				}
				cleanupErr := errors.New("release failed after cancellation")
				if mode == "cleanup-error" {
					f.loaded.CloseModel = func() error { f.event("model"); return cleanupErr }
				}
				if mode == "cleanup-expiry" {
					f.backend.Close = func() error { f.event("backend"); f.advance(f.req.Budgets.CleanupNS + 1); return nil }
				}
				out := runV41EPOwnerLifecycle(ctx, f.req, 0, f.deps)
				if !errors.Is(out.Err, context.Canceled) || out.Receipt.Status != "incomplete" || out.Receipt.ExecutionJoined == nil || !*out.Receipt.ExecutionJoined {
					t.Fatalf("late cancellation lost its verdict: %+v", out)
				}
				if mode == "cancel-during-shutdown" {
					if f.count("shutdown") != 1 || out.Receipt.ShutdownCompleted == nil || *out.Receipt.ShutdownCompleted || !f.before("stop-io", "shutdown-joined") || !f.before("shutdown-joined", "model") {
						t.Fatalf("canceled Shutdown did not join before retirement: %+v events=%v", out, f.events())
					}
				} else if f.has("shutdown") || out.Receipt.ShutdownCompleted != nil {
					t.Fatalf("Shutdown ran after cancellation: %+v", out)
				}
				wantClean := mode != "cleanup-error" && mode != "cleanup-expiry" && mode != "cancel-during-shutdown"
				if out.Receipt.CleanupCompleted == nil || *out.Receipt.CleanupCompleted != wantClean || (len(out.Receipt.CleanupFailures) == 0) != wantClean {
					t.Fatalf("cleanup outcome disagrees with releases: %+v", out)
				}
				if mode == "cleanup-error" && !errors.Is(out.Err, cleanupErr) || mode == "cleanup-expiry" && !errors.Is(out.Err, context.DeadlineExceeded) {
					t.Fatalf("cleanup failure masked: %+v", out)
				}
				if len(caps) != 4 || mode == "earlier-parent-deadline" && !caps[3].Equal(parentDeadline) {
					t.Fatalf("cleanup extended an earlier caller deadline: %v", caps)
				}
				for _, kind := range []string{"session", "model", "group", "backend"} {
					if f.count(kind) != 1 {
						t.Fatalf("release count changed for %s: %v", kind, f.events())
					}
				}
			})
		}
	})
	t.Run("deadline-expiry-stays-failed-through-cleanup", func(t *testing.T) {
		for _, mode := range []string{"forward", "last-backend-close"} {
			t.Run(mode, func(t *testing.T) {
				f := newV41EPLifecycleFake()
				if mode == "forward" {
					f.deps.Root = func(context.Context, V41EPOwnerRequest, *v41EPOwnerLifecycleLoaded) (*V41EPOwnerRootWork, error) {
						f.event("root")
						f.advance(f.req.Budgets.ForwardNS + 1)
						return f.work, nil
					}
				} else {
					f.backend.Close = func() error { f.event("backend"); f.advance(f.req.Budgets.CleanupNS + 1); return nil }
				}
				out := runV41EPOwnerLifecycle(context.Background(), f.req, 0, f.deps)
				if !errors.Is(out.Err, context.DeadlineExceeded) || out.Receipt.Status != "incomplete" || f.count("group") != 1 || f.count("backend") != 1 {
					t.Fatalf("deadline lost: %+v", out)
				}
				if mode == "forward" && (f.has("shutdown") || out.Receipt.RootWork.PrefillTokens != nil) {
					t.Fatal("late success acquired root work or shutdown")
				}
				if mode == "last-backend-close" && (out.Receipt.CleanupCompleted == nil || *out.Receipt.CleanupCompleted) {
					t.Fatal("late final cleanup claimed bounded success")
				}
			})
		}
	})
	t.Run("explicit-backend-and-unknown-observations", func(t *testing.T) {
		for _, mode := range []string{"fallback", "wrong-device", "unavailable-sockets", "partial-root-work", "root-time-overflow", "shutdown-error", "cleanup-cap-error"} {
			t.Run(mode, func(t *testing.T) {
				f := newV41EPLifecycleFake()
				switch mode {
				case "fallback":
					f.backend.Identity.Backend = "cpu-ref"
				case "wrong-device":
					f.backend.Identity.Device = "other"
				case "unavailable-sockets":
					f.group.SocketCounts = func() (V41EPOwnerSocketCounters, bool) { return V41EPOwnerSocketCounters{}, false }
				case "partial-root-work":
					f.work.PrefillNS = nil
				case "root-time-overflow":
					value := int64(^uint64(0) >> 1)
					f.work.PrefillNS = &value
				case "shutdown-error":
					f.deps.Shutdown = func(context.Context, *v41EPOwnerLifecycleLoaded) error {
						f.event("shutdown")
						return errors.New("shutdown failed")
					}
				case "cleanup-cap-error":
					caps := 0
					f.group.SetPhaseDeadline = func(time.Time) error {
						caps++
						if caps == 4 {
							return errors.New("cleanup cap failed")
						}
						return nil
					}
				}
				out := runV41EPOwnerLifecycle(context.Background(), f.req, 0, f.deps)
				if out.Err == nil || out.Receipt.Status != "incomplete" {
					t.Fatalf("unavailable observation accepted: %+v", out)
				}
				if (mode == "fallback" || mode == "wrong-device") && (f.has("load") || f.has("root") || out.Receipt.ArtifactAuthenticated != nil) {
					t.Fatal("implicit backend reached loader")
				}
				if mode == "unavailable-sockets" && out.Receipt.SocketIO != nil {
					t.Fatal("unavailable counter became zero")
				}
				if mode == "unavailable-sockets" && (out.Receipt.CleanupCompleted == nil || !*out.Receipt.CleanupCompleted || len(out.Receipt.CleanupFailures) != 0 || f.count("session") != 1 || f.count("model") != 1 || f.count("group") != 1 || f.count("backend") != 1) {
					t.Fatal("observation failure mislabeled successful resource cleanup")
				}
				if (mode == "partial-root-work" || mode == "root-time-overflow") && out.Receipt.RootWork.PrefillTokens != nil {
					t.Fatal("partial phase observation became work")
				}
				if mode == "cleanup-cap-error" && f.has("shutdown") {
					t.Fatal("Shutdown started without installed cleanup cap")
				}
			})
		}
	})
}

// fak-test:runtime fast est=10ms lane=default
func TestV41EPOwnerLifecycleRawRetirement(t *testing.T) {
	for _, mode := range []string{"successful-open-shutdown", "canceled-clean-release", "canceled-close-error", "canceled-counter-unavailable", "missing-cleanup-hook"} {
		t.Run(mode, func(t *testing.T) {
			f := newV41EPLifecycleFake()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			releaseErr := errors.New("actual fake raw release failure")
			raw := &v41EPFakeConn{closeFn: func() error {
				if mode == "canceled-close-error" {
					return errors.Join(releaseErr, context.Canceled)
				}
				return nil
			}}
			var transport *v41EPOwnerTransport
			var cleanupCap time.Time
			var normalCaps int
			baseGroup, baseStop, baseClose, baseCounts := f.deps.Group, f.group.StopIO, f.group.Close, f.group.SocketCounts
			f.deps.Group = func(phaseCtx context.Context, req V41EPOwnerRequest, rank int, total time.Time) (*v41EPOwnerLifecycleGroup, error) {
				deadline, _ := phaseCtx.Deadline()
				var err error
				transport, err = newV41EPOwnerTransport(f.base, total, deadline)
				if err != nil {
					return nil, err
				}
				conn, err := transport.ownConn(raw)
				if err != nil {
					return nil, err
				}
				if mode == "canceled-counter-unavailable" {
					conn.(*v41EPOwnerConn).countersAvailable = false
				}
				f.group.SetPhaseDeadline = func(deadline time.Time) error {
					normalCaps++
					return transport.setPhaseDeadline(deadline)
				}
				f.group.SetCleanupDeadline = func(deadline time.Time) error {
					cleanupCap = deadline
					return transport.setCleanupDeadline(deadline)
				}
				if mode == "missing-cleanup-hook" {
					f.group.SetCleanupDeadline = nil
				}
				f.group.StopIO = func(cause error) error {
					return errors.Join(baseStop(cause), transport.stopIO(cause))
				}
				f.group.Close = func() error {
					retired, available := transport.retirementAfterJoin()
					if !available {
						return errV41EPOwnerRetirementUnsettled
					}
					return errors.Join(retired, baseClose())
				}
				f.group.SocketCounts = func() (V41EPOwnerSocketCounters, bool) {
					baseCounts()
					return transport.snapshotAfterJoin()
				}
				return baseGroup(phaseCtx, req, rank, total)
			}
			if strings.HasPrefix(mode, "canceled-") {
				f.deps.Root = func(phaseCtx context.Context, _ V41EPOwnerRequest, _ *v41EPOwnerLifecycleLoaded) (*V41EPOwnerRootWork, error) {
					f.event("root")
					cancel()
					return nil, phaseCtx.Err()
				}
			}
			f.deps.Shutdown = func(context.Context, *v41EPOwnerLifecycleLoaded) error {
				f.event("shutdown")
				if cleanupCap.IsZero() {
					return errors.New("Shutdown lacked cleanup deadline")
				}
				raw.wantDeadline(t, "read", cleanupCap)
				raw.wantDeadline(t, "write", cleanupCap)
				return nil
			}
			out := runV41EPOwnerLifecycle(ctx, f.req, 0, f.deps)
			if transport == nil || raw.closes.Load() != 1 || f.count("group") != 1 {
				t.Fatalf("raw/group ownership not retired once: %+v events=%v closes=%d", out, f.events(), raw.closes.Load())
			}
			if _, available := transport.retirementAfterJoin(); !available {
				t.Fatal("lifecycle returned before raw retirement settled")
			}
			switch mode {
			case "successful-open-shutdown":
				if out.Err != nil || out.Receipt.Status != "complete" || f.count("shutdown") != 1 || normalCaps != 3 {
					t.Fatalf("open cleanup did not preserve ordinary Shutdown: %+v events=%v caps=%d", out, f.events(), normalCaps)
				}
			case "missing-cleanup-hook":
				if out.Err == nil || f.has("join") || f.has("backend-factory") || f.has("root") || out.Receipt.CleanupCompleted == nil || *out.Receipt.CleanupCompleted {
					t.Fatalf("missing cleanup hook admitted work: %+v events=%v", out, f.events())
				}
				return
			default:
				if !errors.Is(out.Err, context.Canceled) || out.Receipt.Status != "incomplete" || f.has("shutdown") ||
					out.Receipt.RootWork.PrefillTokens != nil || out.Receipt.ExecutionJoined == nil || !*out.Receipt.ExecutionJoined || normalCaps != 3 {
					t.Fatalf("canceled execution became successful: %+v events=%v caps=%d", out, f.events(), normalCaps)
				}
				if !errors.Is(transport.stop(nil), context.Canceled) {
					t.Fatal("retirement-only adapter erased the primary stop cause")
				}
			}
			wantClean := mode != "canceled-close-error"
			if out.Receipt.CleanupCompleted == nil || *out.Receipt.CleanupCompleted != wantClean || (len(out.Receipt.CleanupFailures) == 0) != wantClean {
				t.Fatalf("cleanup result disagrees with actual releases: %+v", out)
			}
			if !wantClean && !errors.Is(out.Err, releaseErr) {
				t.Fatal("actual Close error was lost")
			}
			if mode == "canceled-counter-unavailable" && out.Receipt.SocketIO != nil {
				t.Fatal("unavailable counters became observed zero")
			}
		})
	}
}

// Every identity, observation, and resource below is synthetic. These hooks
// establish software control flow only; they do not authenticate real artifacts,
// observe a process/backend, verify a supervisor, or invoke a model or network.
type v41EPLifecycleFake struct {
	req        V41EPOwnerRequest
	deps       v41EPOwnerLifecycleDeps
	process    v41EPOwnerProcessIdentity
	group      *v41EPOwnerLifecycleGroup
	backend    *v41EPOwnerLifecycleBackend
	loaded     *v41EPOwnerLifecycleLoaded
	work       *V41EPOwnerRootWork
	retirement model.EPFollowerRetirement
	base       time.Time
	ticks      atomic.Int64
	nonce      atomic.Uint64
	mu         sync.Mutex
	log        []string
}

func newV41EPLifecycleFake() *v41EPLifecycleFake {
	f := &v41EPLifecycleFake{req: v41EPOwnerRequestFixture(), base: time.Date(2100, 1, 1, 0, 0, 0, 0, time.UTC), process: v41EPOwnerProcessIdentity{PID: 123, ExecutableSHA256: strings.Repeat("e", 64)}}
	prompt, steps, one, eos := len(f.req.PromptTokenIDs), 1, int64(1), false
	f.work = &V41EPOwnerRootWork{PrefillTokens: &prompt, CompletedStepCalls: &steps, PrefillNS: &one, DecodeLoopNS: &one, SessionSetupNS: &one, FirstSampleNS: &one, SessionTeardownNS: &one, GeneratedTokenIDs: []int{3, 4}, EOSStopped: &eos}
	var stop sync.Once
	f.group = &v41EPOwnerLifecycleGroup{Rank: 0, Size: 3,
		SetPhaseDeadline:   func(time.Time) error { return nil },
		SetCleanupDeadline: func(deadline time.Time) error { return f.group.SetPhaseDeadline(deadline) },
		Join:               func(context.Context) error { f.event("join"); return nil },
		StopIO:             func(error) error { stop.Do(func() { f.event("stop-io") }); return nil },
		Close:              func() error { f.event("group"); return nil },
		SocketCounts: func() (V41EPOwnerSocketCounters, bool) {
			f.event("socket-counts")
			return V41EPOwnerSocketCounters{}, true
		},
	}
	f.backend = &v41EPOwnerLifecycleBackend{Identity: compute.BackendRuntimeIdentity{Backend: "vulkan", Device: f.req.Owners[0].ExpectedDevice, Driver: "fake-driver", Runtime: "fake-runtime"}, Close: func() error { f.event("backend"); return nil }}
	f.retirement = model.EPFollowerRetirement{AcceptedSessions: 1, Available: true}
	f.loaded = &v41EPOwnerLifecycleLoaded{
		CloseSession: func() error {
			f.retirement.CloseAttempts++
			f.event("session")
			f.retirement.CloseCompletions++
			return nil
		},
		CloseModel:  func() error { f.event("model"); return nil },
		OwnerCounts: func() (V41EPOwnerRoundCounters, bool) { f.event("rounds"); return V41EPOwnerRoundCounters{}, true },
		// The fake Root allocates no intermediate resources. Tests invoking a
		// real executor must replace this with that invocation's observation.
		RootRetirement: func() (v41EPOwnerRootRetirement, error) {
			return v41EPOwnerRootRetirement{Available: true, Completed: true}, nil
		},
		FollowerRetirement: func() (model.EPFollowerRetirement, error) {
			f.event("follower-retirement")
			return f.retirement, nil
		},
	}
	f.deps = v41EPOwnerLifecycleDeps{
		Now: func() time.Time { return f.base.Add(time.Duration(f.ticks.Add(10))) },
		NewNonce: func() ([32]byte, error) {
			f.event("nonce")
			var nonce [32]byte
			binary.LittleEndian.PutUint64(nonce[:8], f.nonce.Add(1))
			return nonce, nil
		},
		ObserveProcess: func(context.Context) (v41EPOwnerProcessIdentity, error) { f.event("process"); return f.process, nil },
		VerifySupervisor: func(_ context.Context, ch v41EPOwnerSupervisorChallenge) (v41EPOwnerSupervisorHandoff, error) {
			f.event("verify")
			return v41EPOwnerSupervisorHandoff{Challenge: ch, TerminateAt: ch.LatestTerminateAt, KillAt: ch.LatestTerminateAt.Add(ch.KillGrace), EvidenceSHA256: strings.Repeat("a", 64)}, nil
		},
		Group: func(context.Context, V41EPOwnerRequest, int, time.Time) (*v41EPOwnerLifecycleGroup, error) {
			f.event("group-factory")
			return f.group, nil
		},
		Backend: func(context.Context, V41EPOwnerSelection) (*v41EPOwnerLifecycleBackend, error) {
			f.event("backend-factory")
			return f.backend, nil
		},
		Load: func(context.Context, V41EPOwnerRequest, *v41EPOwnerLifecycleBackend, *v41EPOwnerLifecycleGroup) (*v41EPOwnerLifecycleLoaded, error) {
			f.event("load")
			return f.loaded, nil
		},
		Root: func(context.Context, V41EPOwnerRequest, *v41EPOwnerLifecycleLoaded) (*V41EPOwnerRootWork, error) {
			f.event("root")
			return f.work, nil
		},
		Follower: func(_ context.Context, loaded *v41EPOwnerLifecycleLoaded, claim func() error) error {
			f.event("follower")
			if err := claim(); err != nil {
				return err
			}
			return loaded.CloseSession()
		},
		Shutdown: func(context.Context, *v41EPOwnerLifecycleLoaded) error { f.event("shutdown"); return nil },
	}
	return f
}

func (f *v41EPLifecycleFake) event(value string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.log = append(f.log, value)
}
func (f *v41EPLifecycleFake) events() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.log...)
}
func (f *v41EPLifecycleFake) count(value string) int {
	n := 0
	for _, event := range f.events() {
		if event == value {
			n++
		}
	}
	return n
}
func (f *v41EPLifecycleFake) has(value string) bool { return f.count(value) != 0 }
func (f *v41EPLifecycleFake) before(first, second string) bool {
	a, b := -1, -1
	for i, event := range f.events() {
		if event == first {
			a = i
		}
		if event == second {
			b = i
		}
	}
	return a >= 0 && b > a
}
func (f *v41EPLifecycleFake) advance(duration time.Duration) { f.ticks.Add(int64(duration)) }

// fak-test:runtime fast est=10ms lane=default
func TestV41EPOwnerLifecycleFollowerEvidence(t *testing.T) {
	complete := model.EPFollowerRetirement{AcceptedSessions: 1, CloseAttempts: 1, CloseCompletions: 1, Available: true}
	for _, tc := range []struct {
		name, want string
		counts     model.EPFollowerRetirement
	}{
		{"complete", "true", complete},
		{"missing", "unknown", complete},
		{"unavailable", "unknown", model.EPFollowerRetirement{}},
		{"observer-error", "unknown", complete},
		{"observer-panic", "unknown", complete},
		{"completion-exceeds-attempt", "unknown", model.EPFollowerRetirement{AcceptedSessions: 1, CloseCompletions: 1, Available: true}},
		{"attempt-exceeds-accepted", "unknown", model.EPFollowerRetirement{AcceptedSessions: 1, CloseAttempts: 2, Available: true}},
		{"no-accepted-session", "unknown", model.EPFollowerRetirement{Available: true}},
		{"multiple-accepted-sessions", "unknown", model.EPFollowerRetirement{AcceptedSessions: 2, CloseAttempts: 2, CloseCompletions: 2, Available: true}},
		{"close-not-returned", "false", model.EPFollowerRetirement{AcceptedSessions: 1, CloseAttempts: 1, Available: true}},
		{"close-not-attempted", "false", model.EPFollowerRetirement{AcceptedSessions: 1, Available: true}},
		{"unknown-plus-release-error", "false", model.EPFollowerRetirement{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newV41EPLifecycleFake()
			f.group.Rank = 1
			f.backend.Identity.Device = f.req.Owners[1].ExpectedDevice
			primary, observation, release := errors.New("primary execution error"), errors.New("observer error"), errors.New("actual model release error")
			base := f.deps.Follower
			joined, observations := false, 0
			f.deps.Follower = func(ctx context.Context, loaded *v41EPOwnerLifecycleLoaded, claim func() error) error {
				defer func() { joined = true }()
				return errors.Join(base(ctx, loaded, claim), primary)
			}
			f.loaded.FollowerRetirement = func() (model.EPFollowerRetirement, error) {
				observations++
				f.event("observe-follower")
				if !joined {
					return model.EPFollowerRetirement{}, errors.New("observed before executor joined")
				}
				switch tc.name {
				case "observer-error":
					return tc.counts, observation
				case "observer-panic":
					panic(observation)
				}
				return tc.counts, nil
			}
			if tc.name == "missing" {
				f.loaded.FollowerRetirement = nil
			}
			if tc.name == "unknown-plus-release-error" {
				f.loaded.CloseModel = func() error { f.event("model"); return release }
			}
			out := runV41EPOwnerLifecycle(context.Background(), f.req, 1, f.deps)
			if !errors.Is(out.Err, primary) || out.PrimaryPanic != nil || out.Receipt.Status != "incomplete" ||
				out.Receipt.ExecutionJoined == nil || !*out.Receipt.ExecutionJoined || !joined ||
				f.count("session") != 1 || f.count("model") != 1 || f.count("group") != 1 || f.count("backend") != 1 {
				t.Fatalf("primary/ownership/join changed: %+v events=%v", out, f.events())
			}
			if tc.name == "missing" && observations != 0 || tc.name != "missing" && (observations != 1 || !f.before("observe-follower", "model")) {
				t.Fatal("observer repeated or crossed cleanup boundary")
			}
			if (tc.name == "observer-error" || tc.name == "observer-panic") && !errors.Is(out.Err, observation) || tc.name == "unknown-plus-release-error" && !errors.Is(out.Err, release) {
				t.Fatal("later observation/release error was masked")
			}
			got := "unknown"
			if out.Receipt.CleanupCompleted != nil {
				got = "false"
				if *out.Receipt.CleanupCompleted {
					got = "true"
				}
			}
			if got != tc.want || (len(out.Receipt.CleanupFailures) > 0) != (tc.want == "false") {
				t.Fatalf("retirement evidence got=%s want=%s cleanupFailures=%v", got, tc.want, out.Receipt.CleanupFailures)
			}
		})
	}
	for _, mode := range []string{"cancellation-after-close", "panic-after-close", "unclaimed-session-stays-runner-owned"} {
		t.Run(mode, func(t *testing.T) {
			f := newV41EPLifecycleFake()
			f.group.Rank = 1
			f.backend.Identity.Device = f.req.Owners[1].ExpectedDevice
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			primary := errors.New("panic after completed Close")
			base := f.deps.Follower
			f.deps.Follower = func(ctx context.Context, loaded *v41EPOwnerLifecycleLoaded, claim func() error) error {
				if mode == "unclaimed-session-stays-runner-owned" {
					return context.Canceled
				}
				if err := base(ctx, loaded, claim); err != nil {
					return err
				}
				if mode == "panic-after-close" {
					panic(primary)
				}
				cancel()
				return ctx.Err()
			}
			out := runV41EPOwnerLifecycle(ctx, f.req, 1, f.deps)
			if out.Err == nil || out.Receipt.Status != "incomplete" || out.Receipt.CleanupCompleted == nil || !*out.Receipt.CleanupCompleted ||
				len(out.Receipt.CleanupFailures) != 0 || out.Receipt.ExecutionJoined == nil || !*out.Receipt.ExecutionJoined || f.count("session") != 1 {
				t.Fatalf("completed retirement was erased by execution failure: %+v events=%v", out, f.events())
			}
			if mode == "panic-after-close" && (out.PrimaryPanic != primary || !errors.Is(out.Err, primary)) || mode != "panic-after-close" && !errors.Is(out.Err, context.Canceled) {
				t.Fatal("primary failure identity changed")
			}
			wantObservations := 1
			if mode == "unclaimed-session-stays-runner-owned" {
				wantObservations = 0
			}
			if f.count("follower-retirement") != wantObservations {
				t.Fatal("untransferred ownership was observed as follower retirement")
			}
		})
	}
}
