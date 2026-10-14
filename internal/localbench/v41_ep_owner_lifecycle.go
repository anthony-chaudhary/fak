package localbench

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type v41EPOwnerProcessIdentity struct {
	PID              int
	ExecutableSHA256 string
}

type v41EPOwnerSupervisorChallenge struct {
	Nonce                     [32]byte
	ContractID, RequestSHA256 string
	Rank                      int
	Process                   v41EPOwnerProcessIdentity
	LatestTerminateAt         time.Time
	KillGrace                 time.Duration
}

// Only a trusted, invocation-local live verifier may return this handoff.
// Matching these fields is structural binding, not proof that a verifier exists.
// No concrete verifier or request-to-handoff conversion is provided here.
type v41EPOwnerSupervisorHandoff struct {
	Challenge      v41EPOwnerSupervisorChallenge
	TerminateAt    time.Time
	KillAt         time.Time
	EvidenceSHA256 string
}

// Non-nil bundles transfer ownership at return, even alongside an error.
// Factories own all allocations they do not return, including before a panic.
// Value is an opaque native-adapter handle, never receipt evidence by itself.
type v41EPOwnerLifecycleBackend struct {
	Value    any
	Identity compute.BackendRuntimeIdentity
	Close    func() error
}

type v41EPOwnerLifecycleGroup struct {
	Value              any
	Rank, Size         int
	Join               func(context.Context) error
	SetPhaseDeadline   func(time.Time) error
	SetCleanupDeadline func(time.Time) error // May accept an already stopped, settled registry.
	StopIO             func(error) error     // Stable raw handles; return release failures, not the execution cause.
	Close              func() error          // Require joined/settled raw retirement before group-table teardown.
	SocketCounts       func() (V41EPOwnerSocketCounters, bool)
}

type v41EPOwnerLifecycleLoaded struct {
	Value        any
	CloseSession func() error
	CloseModel   func() error // Includes retained authenticated readers.
	OwnerCounts  func() (V41EPOwnerRoundCounters, bool)
	// A nonblocking snapshot from this loaded bundle's joined follower call.
	// It must describe actual accepted/Close events, not infer them from Err.
	FollowerRetirement func() (model.EPFollowerRetirement, error)
}

// v41EPOwnerLoadRetirement is an invocation-local observation of construction-
// owned resources that a failed Load did not transfer in its returned bundle.
// A complete preflight refusal means no owned construction work began. Successful
// transfer is not a factory-retirement observation; its actual releases are used.
// Unavailable is false/false/nil. Available failure requires a release error.
// Native adapters explicitly map the authenticated loader's result into this
// value, without inferring completion from nil ownership or an error string.
type v41EPOwnerLoadRetirement struct {
	Available bool
	Completed bool
	Err       error
}

// These hooks are trusted internal dependencies, not user-supplied declarations.
// Native bindings must use actual identity observation, live supervisor admission,
// the authenticated loader, and genuine successful rawdecode observations.
// Executors must join every owned user on every exit, including panic. A follower
// calls claim exactly once at its factory handoff and then owns Session cleanup.
type v41EPOwnerLifecycleDeps struct {
	// Now must be monotonic and non-panicking (the native binding uses time.Now).
	// Cleanup-safe panic capture covers the other hooks, not a broken clock.
	Now              func() time.Time
	NewNonce         func() ([32]byte, error)
	ObserveProcess   func(context.Context) (v41EPOwnerProcessIdentity, error)
	VerifySupervisor func(context.Context, v41EPOwnerSupervisorChallenge) (v41EPOwnerSupervisorHandoff, error)
	Group            func(context.Context, V41EPOwnerRequest, int, time.Time) (*v41EPOwnerLifecycleGroup, error)
	Backend          func(context.Context, V41EPOwnerSelection) (*v41EPOwnerLifecycleBackend, error)
	Load             func(context.Context, V41EPOwnerRequest, *v41EPOwnerLifecycleBackend, *v41EPOwnerLifecycleGroup) (*v41EPOwnerLifecycleLoaded, error)
	// LoadRetirement is a nonblocking snapshot of this invocation's failed
	// factory. Called only after Load and its phase watcher have joined. All
	// factory-owned users must already be joined, including panic unwind.
	LoadRetirement func() (v41EPOwnerLoadRetirement, error)
	Root           func(context.Context, V41EPOwnerRequest, *v41EPOwnerLifecycleLoaded) (*V41EPOwnerRootWork, error)
	Follower       func(context.Context, *v41EPOwnerLifecycleLoaded, func() error) error
	Shutdown       func(context.Context, *v41EPOwnerLifecycleLoaded) error
}

// A captured panic is an explicitly failed outcome with its original identity.
// A later command boundary may serialize Receipt before surfacing that failure.
type v41EPOwnerLifecycleOutcome struct {
	Receipt      V41EPOwnerReceipt
	Err          error
	PrimaryPanic any
}

func runV41EPOwnerLifecycle(ctx context.Context, req V41EPOwnerRequest, rank int, deps v41EPOwnerLifecycleDeps) (out v41EPOwnerLifecycleOutcome) {
	req.Shards, req.Owners, req.PromptTokenIDs = slices.Clone(req.Shards), slices.Clone(req.Owners), slices.Clone(req.PromptTokenIDs)
	out.Receipt = V41EPOwnerReceipt{Schema: V41EPOwnerReceiptSchema, RunID: req.RunID, Rank: rank, WorldSize: req.WorldSize,
		Status: "refused", LastPhase: "admission", Shards: slices.Clone(req.Shards), ExpectedShardSetSHA256: req.ExpectedShardSetSHA256, SourceSHA256: req.SourceSHA256}
	r := &out.Receipt
	var effectiveDeadline time.Time
	if deps.Now == nil {
		out.add(errors.New("v41 EP owner lifecycle: clock is required"), nil)
		return out
	}
	r.StartedAt = deps.Now()
	defer func() {
		finished := deps.Now()
		r.FinishedAt = &finished
		if !effectiveDeadline.IsZero() && !finished.Before(effectiveDeadline) {
			out.add(context.DeadlineExceeded, nil)
		}
		if ctx != nil && ctx.Err() != nil {
			out.add(ctx.Err(), nil)
		}
		if elapsed, ok := v41EPOwnerElapsed(r.StartedAt, finished); ok {
			r.Phases.TotalNS = &elapsed
		} else {
			out.add(errors.New("v41 EP owner lifecycle: invalid total clock interval"), nil)
		}
		if out.Err != nil {
			r.Failure = out.Err.Error()
			if r.Status != "refused" {
				r.Status = "incomplete"
			}
		} else {
			r.Status = "complete"
		}
	}()
	if ctx == nil || r.StartedAt.IsZero() || rank < 0 || rank > 2 {
		out.add(errors.New("v41 EP owner lifecycle: context, clock, and local rank are required"), nil)
		return out
	}
	digest, err := V41EPOwnerRequestDigest(req)
	if err != nil {
		out.add(err, nil)
		return out
	}
	r.RequestSHA256 = digest
	if deps.NewNonce == nil || deps.ObserveProcess == nil || deps.VerifySupervisor == nil || deps.Group == nil || deps.Backend == nil || deps.Load == nil ||
		rank == 0 && (deps.Root == nil || deps.Shutdown == nil) || rank != 0 && deps.Follower == nil {
		out.add(errors.New("v41 EP owner lifecycle: live supervisor verifier and complete trusted dependencies are required"), nil)
		return out
	}
	totalDeadline := r.StartedAt.Add(req.Budgets.TotalNS)
	admissionCtx, admissionCancel := context.WithDeadline(ctx, totalDeadline)
	defer admissionCancel()
	var process v41EPOwnerProcessIdentity
	var handoff v41EPOwnerSupervisorHandoff
	admissionErr, admissionPanic := v41EPOwnerInvoke(func() error {
		if err := admissionCtx.Err(); err != nil {
			return err
		}
		nonce, err := deps.NewNonce()
		if err != nil {
			return err
		}
		if nonce == ([32]byte{}) {
			return errors.New("v41 EP owner lifecycle: empty invocation challenge")
		}
		process, err = deps.ObserveProcess(admissionCtx)
		if err != nil {
			return err
		}
		if process.PID <= 0 || !v41EPOwnerSHA256(process.ExecutableSHA256, true) {
			return errors.New("v41 EP owner lifecycle: invalid observed process identity")
		}
		actual := process.ExecutableSHA256
		r.ExecutableSHA256 = &actual
		if expected := req.Owners[rank].ExpectedExecutableSHA256; expected != "" && expected != actual {
			return errors.New("v41 EP owner lifecycle: executable identity disagrees")
		}
		if err := admissionCtx.Err(); err != nil {
			return err
		}
		challenge := v41EPOwnerSupervisorChallenge{Nonce: nonce, ContractID: req.Supervisor.ContractID, RequestSHA256: digest,
			Rank: rank, Process: process, LatestTerminateAt: r.StartedAt.Add(req.Supervisor.TerminateNS), KillGrace: req.Supervisor.KillGraceNS}
		handoff, err = deps.VerifySupervisor(admissionCtx, challenge)
		if err != nil {
			return err
		}
		if handoff.Challenge != challenge || !v41EPOwnerSHA256(handoff.EvidenceSHA256, true) ||
			!handoff.TerminateAt.After(deps.Now()) || handoff.TerminateAt.After(challenge.LatestTerminateAt) ||
			!handoff.KillAt.After(handoff.TerminateAt) || handoff.KillAt.After(handoff.TerminateAt.Add(challenge.KillGrace)) {
			return errors.New("v41 EP owner lifecycle: supervisor handoff is stale, mismatched, or unbounded")
		}
		return admissionCtx.Err()
	})
	if admissionErr != nil {
		out.add(admissionErr, admissionPanic)
		return out
	}
	if !deps.Now().Before(totalDeadline) {
		out.add(context.DeadlineExceeded, nil)
		return out
	}
	evidence := handoff.EvidenceSHA256
	r.SupervisorEvidenceSHA256 = &evidence
	if handoff.TerminateAt.Before(totalDeadline) {
		totalDeadline = handoff.TerminateAt
	}
	effectiveDeadline = totalDeadline
	runCtx, runCancel := context.WithDeadline(ctx, totalDeadline)
	defer runCancel()
	r.Status = "incomplete"
	var group *v41EPOwnerLifecycleGroup
	var backend *v41EPOwnerLifecycleBackend
	var loaded *v41EPOwnerLifecycleLoaded
	var sessionState atomic.Uint32 // 0 runner-owned, 1 follower-owned, 2 close attempted.
	var claimViolation atomic.Bool
	executionEntered, rootSucceeded := false, false
	loadAttempted, loadComplete := false, false
	cleanupOK, cleanupKnown := true, true
	cleanup := func(label string, fn func() error) {
		err, p := v41EPOwnerInvoke(func() error {
			if fn == nil {
				return errors.New("missing release operation")
			}
			return fn()
		})
		if err != nil {
			cleanupOK = false
			wrapped := fmt.Errorf("%s: %w", label, err)
			r.CleanupFailures = append(r.CleanupFailures, wrapped.Error())
			out.add(wrapped, p)
		}
	}
	closeSession := func() error {
		if loaded == nil || !sessionState.CompareAndSwap(0, 2) {
			return nil
		}
		if loaded.CloseSession == nil {
			return errors.New("missing Session release operation")
		}
		return loaded.CloseSession()
	}
	join := v41EPOwnerRunPhase(runCtx, totalDeadline, req.Budgets.JoinNS, deps.Now, false, func(phaseCtx context.Context, watch func(func(error) error) *v41EPOwnerPhaseWatcher) error {
		r.LastPhase = "join"
		var err error
		group, err = deps.Group(phaseCtx, req, rank, totalDeadline)
		if err != nil {
			return err
		}
		if group == nil || group.Join == nil || group.SetPhaseDeadline == nil || group.SetCleanupDeadline == nil || group.StopIO == nil || group.Close == nil || group.SocketCounts == nil {
			return errors.New("v41 EP owner lifecycle: incomplete group ownership bundle")
		}
		watch(group.StopIO)
		if err := v41EPOwnerSetPhase(group, phaseCtx); err != nil {
			return err
		}
		if err := phaseCtx.Err(); err != nil {
			return err
		}
		if err := group.Join(phaseCtx); err != nil {
			return err
		}
		if group.Rank != rank || group.Size != 3 {
			return errors.New("v41 EP owner lifecycle: actual group rank or size disagrees")
		}
		return nil
	})
	r.Phases.JoinNS = join.elapsed
	out.add(join.err(), join.panic)
	if out.Err == nil {
		load := v41EPOwnerRunPhase(runCtx, totalDeadline, req.Budgets.LoadNS, deps.Now, false, func(phaseCtx context.Context, watch func(func(error) error) *v41EPOwnerPhaseWatcher) error {
			r.LastPhase = "load"
			watch(group.StopIO)
			if err := v41EPOwnerSetPhase(group, phaseCtx); err != nil {
				return err
			}
			var err error
			backend, err = deps.Backend(phaseCtx, req.Owners[rank])
			if err != nil {
				return err
			}
			if backend == nil || backend.Close == nil {
				return errors.New("v41 EP owner lifecycle: incomplete backend ownership bundle")
			}
			identity := backend.Identity
			r.BackendIdentity = &identity
			if identity.Backend != req.Owners[rank].Backend || identity.Device != req.Owners[rank].ExpectedDevice ||
				!v41EPOwnerText(identity.Driver, V41EPOwnerMaxDeviceBytes) || !v41EPOwnerText(identity.Runtime, V41EPOwnerMaxDeviceBytes) {
				return errors.New("v41 EP owner lifecycle: actual explicit backend/device identity disagrees")
			}
			if err := phaseCtx.Err(); err != nil {
				return err
			}
			loadAttempted = true
			loaded, err = deps.Load(phaseCtx, req, backend, group)
			if err != nil {
				return err
			}
			if loaded == nil || loaded.CloseSession == nil || loaded.CloseModel == nil {
				return errors.New("v41 EP owner lifecycle: incomplete authenticated loader ownership bundle")
			}
			loadComplete = true
			return nil
		})
		r.Phases.LoadNS = load.elapsed
		out.add(load.err(), load.panic)
		// A complete bundle can still be followed by phase cancellation. Its
		// transferred resources are judged by their actual releases below; do
		// not confuse that case with failed opaque construction. runPhase has
		// joined the callback and raw-stop watcher before this observation.
		if loadAttempted && !loadComplete {
			var retired v41EPOwnerLoadRetirement
			observeErr, observePanic := v41EPOwnerInvoke(func() error {
				if deps.LoadRetirement == nil {
					return errors.New("loader construction retirement observer unavailable")
				}
				var err error
				retired, err = deps.LoadRetirement()
				if err != nil {
					return err
				}
				if !retired.Available {
					if retired.Completed || retired.Err != nil {
						return errors.New("loader construction retirement observation incoherent")
					}
					return errors.New("loader construction retirement unavailable")
				}
				if retired.Completed != (retired.Err == nil) {
					return errors.New("loader construction retirement observation incoherent")
				}
				return nil
			})
			if observeErr != nil {
				cleanupKnown = false
				out.add(fmt.Errorf("loader construction retirement observation: %w", observeErr), observePanic)
			} else if !retired.Completed {
				cleanupOK = false
				failure := fmt.Errorf("loader construction retirement: %w", retired.Err)
				r.CleanupFailures = append(r.CleanupFailures, failure.Error())
				out.add(failure, nil)
			}
		}
		if out.Err == nil {
			authenticated := true
			r.ArtifactAuthenticated = &authenticated
		}
	}
	if out.Err == nil {
		var work *V41EPOwnerRootWork
		forward := v41EPOwnerRunPhase(runCtx, totalDeadline, req.Budgets.ForwardNS, deps.Now, false, func(phaseCtx context.Context, watch func(func(error) error) *v41EPOwnerPhaseWatcher) error {
			r.LastPhase = "forward"
			watch(group.StopIO)
			if err := v41EPOwnerSetPhase(group, phaseCtx); err != nil {
				return err
			}
			if rank == 0 {
				closeErr, closePanic := v41EPOwnerInvoke(closeSession)
				if closeErr != nil {
					cleanupOK = false
					r.CleanupFailures = append(r.CleanupFailures, "initial Session: "+closeErr.Error())
					if closePanic != nil {
						panic(closePanic)
					}
					return closeErr
				}
				if err := phaseCtx.Err(); err != nil {
					return err
				}
			}
			executionEntered = true
			defer func() { joined := true; r.ExecutionJoined = &joined }()
			if rank == 0 {
				var err error
				work, err = deps.Root(phaseCtx, req, loaded)
				return err
			}
			claim := func() error {
				if err := phaseCtx.Err(); err != nil {
					return err
				}
				if !sessionState.CompareAndSwap(0, 1) {
					claimViolation.Store(true)
					return errors.New("v41 EP owner lifecycle: factory Session already transferred or closed")
				}
				return nil
			}
			err := deps.Follower(phaseCtx, loaded, claim)
			if sessionState.Load() != 1 || claimViolation.Load() {
				err = errors.Join(err, errors.New("v41 EP owner lifecycle: follower must claim exactly one factory Session"))
			}
			return err
		})
		r.Phases.ForwardNS = forward.elapsed
		out.add(forward.err(), forward.panic)
		if out.Err == nil && rank == 0 {
			if err := v41EPOwnerCheckRootWork(work, req); err != nil {
				out.add(err, nil)
			} else {
				r.RootWork = v41EPOwnerCopyRootWork(*work)
				rootSucceeded = true
			}
		}
	}
	// Ownership transferred to the follower cannot be credited merely because
	// the executor joined or returned an ordinary error. Observe its exact local
	// result only after all executor users have joined, including panic unwind.
	if executionEntered && sessionState.Load() == 1 {
		var retired model.EPFollowerRetirement
		observeErr, observePanic := v41EPOwnerInvoke(func() error {
			if loaded.FollowerRetirement == nil {
				return errors.New("follower Session retirement observer unavailable")
			}
			var err error
			retired, err = loaded.FollowerRetirement()
			if err != nil {
				return err
			}
			if !retired.Available {
				return errors.New("follower Session retirement counts unavailable")
			}
			if retired.CloseCompletions > retired.CloseAttempts || retired.CloseAttempts > retired.AcceptedSessions || retired.AcceptedSessions != 1 {
				return errors.New("follower Session retirement counts incoherent or outside the one-session scope")
			}
			return nil
		})
		if observeErr != nil {
			cleanupKnown = false
			out.add(fmt.Errorf("follower Session retirement observation: %w", observeErr), observePanic)
		} else if retired.CloseAttempts != 1 || retired.CloseCompletions != 1 {
			cleanupOK = false
			failure := errors.New("follower Session retirement did not complete")
			r.CleanupFailures = append(r.CleanupFailures, failure.Error())
			out.add(failure, nil)
		}
	}
	if executionEntered && loaded != nil && loaded.OwnerCounts != nil {
		observeErr, observePanic := v41EPOwnerInvoke(func() error {
			if counts, available := loaded.OwnerCounts(); available {
				r.OwnerRounds = &counts
			}
			return nil
		})
		out.add(observeErr, observePanic)
	}
	// Cleanup remains accounted against its absolute deadline through the final
	// backend release. A blocked synchronous release still needs the supervisor.
	// Cancellation stops execution, but does not itself mean releases failed.
	// Detach it only for cleanup, retaining any earlier caller deadline as well
	// as the immutable supervisor/total cap and the cleanup phase budget.
	cleanupDeadline := totalDeadline
	if deadline, ok := runCtx.Deadline(); ok && deadline.Before(cleanupDeadline) {
		cleanupDeadline = deadline
	}
	cleaned := v41EPOwnerRunPhase(context.WithoutCancel(runCtx), cleanupDeadline, req.Budgets.CleanupNS, deps.Now, true, func(cleanCtx context.Context, watch func(func(error) error) *v41EPOwnerPhaseWatcher) error {
		r.LastPhase = "cleanup"
		var watcher *v41EPOwnerPhaseWatcher
		if group != nil && group.StopIO != nil {
			watcher = watch(group.StopIO)
		}
		phaseCapOK := group == nil
		if group != nil {
			capErr, capPanic := v41EPOwnerInvoke(func() error { return v41EPOwnerSetCleanupPhase(group, cleanCtx) })
			phaseCapOK = capErr == nil
			if capErr != nil {
				cleanupOK = false
				r.CleanupFailures = append(r.CleanupFailures, "cleanup phase deadline: "+capErr.Error())
				out.add(capErr, capPanic)
				if watcher != nil {
					watcher.stop(capErr)
				}
			}
		}
		if rootSucceeded && phaseCapOK {
			if err := errors.Join(runCtx.Err(), cleanCtx.Err()); err != nil {
				out.add(err, nil)
			} else {
				// Shutdown is successful execution protocol work, not unconditional
				// retirement; cancellation during it must still reach the hook.
				deadline, _ := cleanCtx.Deadline()
				shutdownCtx, cancelShutdown := context.WithDeadline(runCtx, deadline)
				shutdownWatch := newV41EPOwnerPhaseWatcher(shutdownCtx, group.StopIO)
				shutdownErr, shutdownPanic := v41EPOwnerInvoke(func() error { return deps.Shutdown(shutdownCtx, loaded) })
				shutdownErr = errors.Join(shutdownErr, shutdownCtx.Err())
				if shutdownErr != nil {
					shutdownWatch.stop(shutdownErr)
				}
				shutdownWatch.join()
				shutdownErr = errors.Join(shutdownErr, shutdownCtx.Err(), shutdownWatch.stopErr)
				cancelShutdown()
				ok := shutdownErr == nil
				r.ShutdownCompleted = &ok
				if !ok {
					cleanupOK = false
					r.CleanupFailures = append(r.CleanupFailures, "Shutdown: "+shutdownErr.Error())
				}
				out.add(shutdownErr, shutdownPanic)
			}
		}
		cleanup("Session", closeSession)
		if loaded != nil {
			cleanup("model/readers", loaded.CloseModel)
		}
		if watcher != nil {
			watcher.join()
		}
		if group != nil {
			if group.StopIO != nil {
				cleanup("raw transport", func() error { return group.StopIO(nil) })
			}
			cleanup("group", group.Close)
			if group.SocketCounts != nil {
				observeErr, observePanic := v41EPOwnerInvoke(func() error {
					counts, available := group.SocketCounts()
					if !available {
						return errors.New("settled exact socket counters unavailable")
					}
					r.SocketIO = &counts
					return nil
				})
				out.add(observeErr, observePanic)
			}
		}
		if backend != nil {
			cleanup("backend", backend.Close)
		}
		return nil
	})
	r.Phases.CleanupNS = cleaned.elapsed
	if cleaned.err() != nil {
		cleanupOK = false
		r.CleanupFailures = append(r.CleanupFailures, cleaned.err().Error())
		out.add(cleaned.err(), cleaned.panic)
	}
	// Unknown factory/follower evidence cannot override a known release
	// failure. Otherwise preserve unknown as null rather than observed success.
	if cleanupKnown || !cleanupOK {
		r.CleanupCompleted = &cleanupOK
	}
	return out
}

func v41EPOwnerSetPhase(group *v41EPOwnerLifecycleGroup, ctx context.Context) error {
	deadline, ok := ctx.Deadline()
	if !ok || group.SetPhaseDeadline == nil {
		return errors.New("v41 EP owner lifecycle: group phase deadline operation is required")
	}
	return group.SetPhaseDeadline(deadline)
}

func v41EPOwnerSetCleanupPhase(group *v41EPOwnerLifecycleGroup, ctx context.Context) error {
	deadline, ok := ctx.Deadline()
	if !ok || group.SetCleanupDeadline == nil {
		return errors.New("v41 EP owner lifecycle: group cleanup deadline operation is required")
	}
	return group.SetCleanupDeadline(deadline)
}

func (out *v41EPOwnerLifecycleOutcome) add(err error, primaryPanic any) {
	if err == nil {
		return
	}
	if out.Err == nil {
		out.PrimaryPanic = primaryPanic
	}
	out.Err = errors.Join(out.Err, err)
	out.Receipt.Failure = out.Err.Error()
}

func v41EPOwnerInvoke(fn func() error) (err error, originalPanic any) {
	defer func() {
		if originalPanic = recover(); originalPanic != nil {
			if cause, ok := originalPanic.(error); ok {
				err = fmt.Errorf("panic: %w", cause)
			} else {
				err = fmt.Errorf("panic: %v", originalPanic)
			}
		}
	}()
	err = fn()
	return
}

type v41EPOwnerPhaseResult struct {
	workErr, contextErr, watchErr error
	panic                         any
	elapsed                       *int64
}

func (r v41EPOwnerPhaseResult) err() error { return errors.Join(r.workErr, r.contextErr, r.watchErr) }

func v41EPOwnerRunPhase(parent context.Context, total time.Time, budget time.Duration, now func() time.Time, allowCancelled bool,
	work func(context.Context, func(func(error) error) *v41EPOwnerPhaseWatcher) error) (result v41EPOwnerPhaseResult) {
	start := now()
	deadline := start.Add(budget)
	if total.Before(deadline) {
		deadline = total
	}
	ctx, cancel := context.WithDeadline(parent, deadline)
	defer cancel()
	if !allowCancelled && (ctx.Err() != nil || !start.Before(deadline)) {
		result.contextErr = ctx.Err()
		if result.contextErr == nil {
			result.contextErr = context.DeadlineExceeded
		}
		return result
	}
	var watcher *v41EPOwnerPhaseWatcher
	startWatch := func(stop func(error) error) *v41EPOwnerPhaseWatcher {
		if watcher != nil {
			panic("v41 EP owner lifecycle: duplicate phase watcher")
		}
		watcher = newV41EPOwnerPhaseWatcher(ctx, stop)
		return watcher
	}
	result.workErr, result.panic = v41EPOwnerInvoke(func() error { return work(ctx, startWatch) })
	end := now()
	result.contextErr = ctx.Err()
	if !end.Before(deadline) && result.contextErr == nil {
		result.contextErr = context.DeadlineExceeded
	}
	if watcher != nil {
		if result.workErr != nil {
			watcher.stop(result.workErr)
		}
		if result.contextErr != nil {
			watcher.stop(result.contextErr)
		}
		watcher.join()
		result.watchErr = watcher.stopErr
	}
	// A deadline may expire while a raw Close is settling during watcher join.
	if err := ctx.Err(); err != nil {
		result.contextErr = err
	}
	end = now()
	if !end.Before(deadline) && result.contextErr == nil {
		result.contextErr = context.DeadlineExceeded
	}
	if watcher != nil && result.contextErr != nil {
		watcher.stop(result.contextErr)
		result.watchErr = watcher.stopErr
		end = now()
	}
	if elapsed, ok := v41EPOwnerElapsed(start, end); ok {
		result.elapsed = &elapsed
	} else {
		result.contextErr = errors.New("v41 EP owner lifecycle: invalid phase clock interval")
	}
	return result
}

type v41EPOwnerPhaseWatcher struct {
	done, finished     chan struct{}
	joinOnce, stopOnce sync.Once
	stopFn             func(error) error
	stopErr            error
}

func newV41EPOwnerPhaseWatcher(ctx context.Context, stop func(error) error) *v41EPOwnerPhaseWatcher {
	w := &v41EPOwnerPhaseWatcher{done: make(chan struct{}), finished: make(chan struct{}), stopFn: stop}
	go func() {
		defer close(w.finished)
		select {
		case <-ctx.Done():
			w.stop(ctx.Err())
		case <-w.done:
		}
	}()
	return w
}

func (w *v41EPOwnerPhaseWatcher) stop(cause error) {
	w.stopOnce.Do(func() { w.stopErr, _ = v41EPOwnerInvoke(func() error { return w.stopFn(cause) }) })
}

func (w *v41EPOwnerPhaseWatcher) join() { w.joinOnce.Do(func() { close(w.done); <-w.finished }) }

func v41EPOwnerElapsed(start, end time.Time) (int64, bool) {
	if start.IsZero() || end.Before(start) {
		return 0, false
	}
	return int64(end.Sub(start)), true
}

func v41EPOwnerCheckRootWork(work *V41EPOwnerRootWork, req V41EPOwnerRequest) error {
	if work == nil || work.PrefillTokens == nil || work.CompletedStepCalls == nil || work.EOSStopped == nil ||
		*work.PrefillTokens != len(req.PromptTokenIDs) || len(work.GeneratedTokenIDs) == 0 || len(work.GeneratedTokenIDs) > req.GeneratedTokenLimit ||
		*work.CompletedStepCalls != len(work.GeneratedTokenIDs)-1 || req.IgnoreEOS && *work.EOSStopped ||
		!*work.EOSStopped && len(work.GeneratedTokenIDs) != req.GeneratedTokenLimit {
		return errors.New("v41 EP owner lifecycle: complete root work observation is required")
	}
	remaining := int64(req.Budgets.ForwardNS)
	for _, duration := range []*int64{work.PrefillNS, work.DecodeLoopNS, work.SessionSetupNS, work.FirstSampleNS, work.SessionTeardownNS} {
		if duration == nil || *duration < 0 || *duration > remaining {
			return errors.New("v41 EP owner lifecycle: root phase observation unavailable or outside budget")
		}
		remaining -= *duration
	}
	for _, token := range work.GeneratedTokenIDs {
		if token < 0 || uint64(token) > uint64(^uint32(0)>>1) {
			return errors.New("v41 EP owner lifecycle: invalid generated token")
		}
	}
	return nil
}

func v41EPOwnerCopyRootWork(work V41EPOwnerRootWork) V41EPOwnerRootWork {
	copyInt := func(p *int) *int {
		if p == nil {
			return nil
		}
		value := *p
		return &value
	}
	copyNS := func(p *int64) *int64 {
		if p == nil {
			return nil
		}
		value := *p
		return &value
	}
	work.PrefillTokens, work.CompletedStepCalls = copyInt(work.PrefillTokens), copyInt(work.CompletedStepCalls)
	work.PrefillNS, work.DecodeLoopNS = copyNS(work.PrefillNS), copyNS(work.DecodeLoopNS)
	work.SessionSetupNS, work.FirstSampleNS, work.SessionTeardownNS = copyNS(work.SessionSetupNS), copyNS(work.FirstSampleNS), copyNS(work.SessionTeardownNS)
	if work.EOSStopped != nil {
		value := *work.EOSStopped
		work.EOSStopped = &value
	}
	work.GeneratedTokenIDs = slices.Clone(work.GeneratedTokenIDs)
	return work
}
