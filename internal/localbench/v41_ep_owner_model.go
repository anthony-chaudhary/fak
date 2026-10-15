package localbench

import (
	"context"
	"errors"
	"fmt"
	"go/version"
	"runtime"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/rawdecode"
)

// This private binding supplies model operations only. It neither admits an
// owner nor chooses/owns its backend or group. Each instance serves one Load
// and one execution. The lifecycle joins operations before reading snapshots.
// No native command installs this binding until its separate admission exists.
type v41EPOwnerModelAdapter struct {
	panicMode    v41EPOwnerPanicModeInput
	loadCalled   bool
	loadReturned bool
	loadResult   ggufload.V41ExpertParallelSessionResult
	loaded       *v41EPOwnerLifecycleLoaded
	state        *v41EPOwnerModelState
}

type v41EPOwnerModelState struct {
	requestSHA256 string
	model         *model.Model
	session       *model.Session
	backend       compute.Backend
	group         *model.DistComm
	coordinator   *model.EPDecodeCoordinator

	initialAttempted, initialCompleted bool
	modelAttempted                     bool
	executionCalled, executionReturned bool
	rootComplete, shutdownCalled       bool
	factoryCalled                      bool
	rootRetirement                     v41EPOwnerRootRetirement
	followerResult                     model.EPFollowerResult
}

// v41EPOwnerPanicModeInput is a trusted caller's observed raw GODEBUG value,
// including an observed empty value. It is copied at bind time. The caller must
// keep the process setting stable from capture through root completion. This
// private input is not authenticated configuration or native admission; current
// callers are tests. Future runtime activation needs an authoritative capture.
type v41EPOwnerPanicModeInput struct {
	environment string
	observed    bool
}

func bindV41EPOwnerModelDeps(base v41EPOwnerLifecycleDeps, panicMode v41EPOwnerPanicModeInput) (v41EPOwnerLifecycleDeps, *v41EPOwnerModelAdapter) {
	a := &v41EPOwnerModelAdapter{panicMode: panicMode}
	base.Load, base.LoadRetirement = a.load, a.loadRetirement
	base.Root, base.Follower, base.Shutdown = a.root, a.follower, a.shutdown
	return base, a
}

func (a *v41EPOwnerModelAdapter) load(ctx context.Context, req V41EPOwnerRequest, backend *v41EPOwnerLifecycleBackend, group *v41EPOwnerLifecycleGroup) (*v41EPOwnerLifecycleLoaded, error) {
	if a.loadCalled || a.loaded != nil {
		return nil, errors.New("v41 EP model: Load is one-shot")
	}
	a.loadCalled = true
	if backend == nil || group == nil {
		return nil, errors.New("v41 EP model: missing borrowed backend/group")
	}
	be, backendOK := backend.Value.(compute.Backend)
	g, groupOK := group.Value.(*model.DistComm)
	if !backendOK || be == nil || !groupOK || g == nil || g.Rank() != group.Rank || g.Size() != group.Size {
		return nil, errors.New("v41 EP model: invalid borrowed backend/group binding")
	}
	// Cache the entire actual result before any error or panic inspection.
	// Return its typed Err, retaining its carrier/PanicValue and all cleanup
	// errors. Panicked is independent of a nil payload. This result contract
	// deliberately does not re-panic or populate lifecycle.PrimaryPanic.
	a.loadResult = ggufload.LoadV41ExpertParallelSessionWithResult(ctx, ggufload.V41ExpertParallelSessionRequest{
		Shards: slices.Clone(req.Shards), ExpectedShardSetSHA256: req.ExpectedShardSetSHA256,
		SourceSHA256: req.SourceSHA256, Group: g, Backend: be,
	})
	a.loadReturned = true
	result := a.loadResult
	if !result.OwnershipTransferred {
		if result.Err == nil || result.Session != nil || result.Coordinator != nil {
			return nil, errors.Join(result.Err, errors.New("v41 EP model: incoherent nontransfer result"))
		}
		return nil, result.Err
	}
	// Adopt immediately, even if a later invariant or cancellation check fails.
	// The nonnil bundle transfers all visible ownership back to the lifecycle.
	loaded, bindErr := a.bindLoaded(req, be, g, result.Session, result.Coordinator)
	if loaded != nil {
		loaded.OwnerCounts = func() (V41EPOwnerRoundCounters, bool) {
			if a.state.model == nil {
				return V41EPOwnerRoundCounters{}, false
			}
			stats := a.state.model.V41ExpertParallelTransportStats()
			return V41EPOwnerRoundCounters{CompletedGathers: stats.CompletedGathers,
				SentFrameBytes: stats.SentFrameBytes, ReceivedFrameBytes: stats.ReceivedFrameBytes,
				GatherNanoseconds: stats.GatherNanoseconds}, true
		}
	}
	if result.Panicked || result.Retirement.Available || result.Retirement.Completed || result.Retirement.Err != nil {
		bindErr = errors.Join(bindErr, errors.New("v41 EP model: incoherent transferred result"))
	}
	return loaded, errors.Join(result.Err, bindErr, ctx.Err())
}

func (a *v41EPOwnerModelAdapter) loadRetirement() (v41EPOwnerLoadRetirement, error) {
	if !a.loadReturned || a.loadResult.OwnershipTransferred {
		return v41EPOwnerLoadRetirement{}, nil
	}
	r := a.loadResult.Retirement
	return v41EPOwnerLoadRetirement{Available: r.Available, Completed: r.Completed, Err: r.Err}, nil
}

// bindLoaded owns the supplied Session/model after its nonnil return, including
// an error return. It does not authenticate them. Only load above can install
// authenticated-loader owner statistics. Tiny CPU tests use this internal
// binding directly, never a synthetic successful lifecycle Load receipt.
func (a *v41EPOwnerModelAdapter) bindLoaded(req V41EPOwnerRequest, be compute.Backend, g *model.DistComm, session *model.Session, coordinator *model.EPDecodeCoordinator) (*v41EPOwnerLifecycleLoaded, error) {
	if a.loaded != nil || session == nil {
		return nil, errors.New("v41 EP model: missing or already-bound Session")
	}
	s := &v41EPOwnerModelState{model: session.M, session: session, backend: be, group: g, coordinator: coordinator}
	loaded := &v41EPOwnerLifecycleLoaded{Value: s}
	a.state, a.loaded = s, loaded
	loaded.CloseSession = func() error {
		if s.initialAttempted || s.factoryCalled {
			return errors.New("v41 EP model: initial Session already closed or handed off")
		}
		s.initialAttempted = true
		s.session.Close()
		s.initialCompleted = true
		return nil
	}
	loaded.CloseModel = func() error {
		if s.modelAttempted || s.model == nil {
			return errors.New("v41 EP model: model missing or already released")
		}
		s.modelAttempted = true
		return s.model.CloseWeights()
	}
	loaded.RootRetirement = func() (v41EPOwnerRootRetirement, error) {
		if !s.executionReturned {
			return v41EPOwnerRootRetirement{}, errors.New("v41 EP model: root has not joined")
		}
		return s.rootRetirement, nil
	}
	loaded.FollowerRetirement = func() (model.EPFollowerRetirement, error) {
		if !s.executionReturned {
			return model.EPFollowerRetirement{}, errors.New("v41 EP model: follower has not joined")
		}
		return s.followerResult.Retirement, nil
	}
	var err error
	s.requestSHA256, err = V41EPOwnerRequestDigest(req)
	if err != nil {
		return loaded, err
	}
	if s.model == nil || be == nil || g == nil || g.Size() != 3 || g.Rank() < 0 || g.Rank() > 2 ||
		(g.Rank() == 0) != (coordinator != nil) {
		return loaded, errors.New("v41 EP model: incomplete loaded model binding")
	}
	return loaded, nil
}

func (a *v41EPOwnerModelAdapter) bound(loaded *v41EPOwnerLifecycleLoaded) (*v41EPOwnerModelState, error) {
	if loaded == nil || loaded != a.loaded || a.state == nil || loaded.Value != a.state {
		return nil, errors.New("v41 EP model: bundle belongs to another invocation")
	}
	return a.state, nil
}

func (a *v41EPOwnerModelAdapter) root(ctx context.Context, req V41EPOwnerRequest, loaded *v41EPOwnerLifecycleLoaded) (*V41EPOwnerRootWork, error) {
	s, err := a.bound(loaded)
	if err != nil {
		return nil, err
	}
	digest, err := V41EPOwnerRequestDigest(req)
	if err != nil || digest != s.requestSHA256 {
		return nil, errors.Join(err, errors.New("v41 EP model: execution request differs from loaded binding"))
	}
	if s.group == nil || s.group.Rank() != 0 || s.executionCalled || !s.initialCompleted || s.modelAttempted {
		return nil, errors.New("v41 EP model: root requires its closed initial Session and one fresh execution")
	}
	s.executionCalled = true
	defer func() { s.executionReturned = true }()
	if err := v41EPOwnerRootPanicMode(a.panicMode); err != nil {
		return nil, err
	}
	// This exact invocation is the only source of Execution/Run below. It owns
	// the fresh Session, while this binding retains model/backend ownership.
	execution, err := rawdecode.ExecuteModelContext(ctx, rawdecode.Request{
		BackendName: s.backend.Name(), PromptTokenIDs: slices.Clone(req.PromptTokenIDs),
		ContextLimit: req.ContextLimit, GeneratedTokenLimit: req.GeneratedTokenLimit,
		Repetitions: 1, IgnoreEOS: req.IgnoreEOS,
	}, s.model, s.backend)
	if err != nil {
		return nil, err
	}
	// A successful direct return follows the actual fresh Session.Close return.
	// Failed/panicking opaque calls remain unknown. Record this before reporting
	// checks or late cancellation; model-pin/release checks remain independent.
	s.rootRetirement = v41EPOwnerRootRetirement{Available: true, Completed: true}
	generation, ok := execution.GenerationObservation(0)
	if !ok || len(execution.Runs) != 1 || !execution.FiniteLogits ||
		!slices.Equal(execution.PromptTokenIDs, req.PromptTokenIDs) || execution.ContextLimit != req.ContextLimit ||
		execution.GeneratedLimit != req.GeneratedTokenLimit || execution.IgnoreEOS != req.IgnoreEOS {
		return nil, errors.New("v41 EP model: complete bound generation observation unavailable")
	}
	tokens, tokensOK := generation.OutputTokenIDs()
	count, countOK := generation.ActualGeneratedTokens()
	ignoreEOS, ignoreOK := generation.IgnoreEOS()
	eos, eosOK := generation.EOSStopped()
	if !tokensOK || !countOK || !ignoreOK || !eosOK || count != len(tokens) || ignoreEOS != req.IgnoreEOS {
		return nil, errors.New("v41 EP model: generation policy observation disagrees")
	}
	run := execution.Runs[0]
	// These software durations come directly from this successful internal call.
	// GenerationObservation does not seal them, and CPU execution does not
	// acquire TimingObservation or backend-resource evidence through this mapping.
	prompt, steps := len(req.PromptTokenIDs), len(tokens)-1
	setup, prefill, sample := int64(run.SessionSetupDuration), int64(run.PrefillDuration), int64(run.FirstSampleDuration)
	decode, teardown := int64(run.DecodeDuration), int64(run.TeardownDuration)
	work := &V41EPOwnerRootWork{PrefillTokens: &prompt, CompletedStepCalls: &steps,
		SessionSetupNS: &setup, PrefillNS: &prefill, FirstSampleNS: &sample,
		DecodeLoopNS: &decode, SessionTeardownNS: &teardown, GeneratedTokenIDs: slices.Clone(tokens), EOSStopped: &eos}
	if err := v41EPOwnerCheckRootWork(work, req); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.rootComplete = true
	return work, nil
}

func (a *v41EPOwnerModelAdapter) follower(ctx context.Context, loaded *v41EPOwnerLifecycleLoaded, claim func() error) error {
	s, err := a.bound(loaded)
	if err != nil {
		return err
	}
	if s.group == nil || s.group.Rank() == 0 || s.executionCalled || s.initialAttempted || s.modelAttempted || claim == nil {
		return errors.New("v41 EP model: follower requires one unclaimed initial Session")
	}
	s.executionCalled = true
	defer func() { s.executionReturned = true }()
	s.followerResult = model.RunEPFollowerWithResult(s.group, func() *model.Session {
		if s.factoryCalled {
			panic(errors.New("v41 EP model: follower factory is one-shot"))
		}
		if err := ctx.Err(); err != nil {
			panic(err)
		}
		if err := claim(); err != nil {
			panic(err)
		}
		// Nothing that can fail may intervene between claim and return.
		s.factoryCalled = true
		return s.session
	})
	return s.followerResult.Err
}

func (a *v41EPOwnerModelAdapter) shutdown(ctx context.Context, loaded *v41EPOwnerLifecycleLoaded) error {
	s, err := a.bound(loaded)
	if err != nil {
		return err
	}
	if s.coordinator == nil || !s.rootComplete || !s.executionReturned || s.shutdownCalled {
		return errors.New("v41 EP model: Shutdown requires joined successful root work")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.shutdownCalled = true
	if err := s.coordinator.Shutdown(); err != nil {
		return fmt.Errorf("v41 EP model: Shutdown: %w", err)
	}
	return ctx.Err()
}

// A normal successful rawdecode return proves Close returned only under the
// modern panic(nil) semantics. Its opaque cleanup helpers cannot distinguish a
// recovered nil payload in compatibility mode. Require the caller's observed
// settings and probe the actual runtime as well: stale or forged modern input
// cannot admit a live compatibility mode. The probe is point-in-time only, not
// retirement evidence or a replacement for the caller's stability contract.
func v41EPOwnerRootPanicMode(input v41EPOwnerPanicModeInput) error {
	if !input.observed {
		return errors.New("v41 EP model: observed runtime panic settings unavailable")
	}
	build, ok := debug.ReadBuildInfo()
	if !ok || !version.IsValid(build.GoVersion) || version.Compare(build.GoVersion, "go1.21") < 0 {
		return errors.New("v41 EP model: runtime panic semantics unavailable")
	}
	defaults := ""
	for _, setting := range build.Settings {
		if setting.Key == "DefaultGODEBUG" {
			defaults = setting.Value
		}
	}
	if err := v41EPOwnerCheckPanicnil(defaults, input.environment); err != nil {
		return err
	}
	if !v41EPOwnerModernPanicnil() {
		return errors.New("v41 EP model: live panicnil compatibility refuses root execution")
	}
	return nil
}

func v41EPOwnerModernPanicnil() (modern bool) {
	defer func() { _, modern = recover().(*runtime.PanicNilError) }()
	panic(nil)
}

func v41EPOwnerCheckPanicnil(defaults, environment string) error {
	last := func(settings string) (string, bool) {
		value, found := "", false
		for _, field := range strings.Split(settings, ",") {
			key, arg, ok := strings.Cut(field, "=")
			if ok && key == "panicnil" {
				value, found = arg, true
			}
		}
		return value, found
	}
	value, found := last(environment)
	if !found {
		value, found = last(defaults)
	}
	if !found {
		return nil // Go >=1.21 defaults to modern semantics absent compatibility.
	}
	// runtime.parsegodebug accepts decimal int32, including +1 and 01. An
	// invalid dynamic value can leave the prior atomic setting unchanged,
	// so ambiguity is refused rather than interpreted as the default.
	n, err := strconv.ParseInt(value, 10, 32)
	if err != nil || n == 1 {
		return errors.New("v41 EP model: panicnil compatibility or ambiguous runtime setting refuses root execution")
	}
	return nil
}
