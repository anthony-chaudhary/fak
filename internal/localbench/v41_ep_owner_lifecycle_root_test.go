package localbench

import (
	"context"
	"errors"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/rawdecode"
)

// fak-test:runtime fast est=30ms lane=default
func TestV41EPOwnerLifecycleRootRetirement(t *testing.T) {
	t.Setenv("FAK_PAGED_KV", "off")
	healthy := v41EPRunRootRetirement(t, false)
	if healthy.out.Err != nil || healthy.out.Receipt.Status != "complete" ||
		healthy.out.Receipt.CleanupCompleted == nil || !*healthy.out.Receipt.CleanupCompleted ||
		healthy.backend.kv[1].frees != 1 || healthy.backend.recycles < 2 ||
		len(healthy.execution.Runs) != 1 || len(healthy.execution.Runs[0].GeneratedTokens) != 2 {
		t.Fatalf("healthy actual root did not retire its fresh Session: outcome=%+v backend=%+v", healthy.out, healthy.backend)
	}
	t.Log("ROOT_RAWDECODE_HEALTHY_RETIREMENT_CONFIRMED")

	failed := v41EPRunRootRetirement(t, true)
	if failed.out.PrimaryPanic != failed.backend.primary || !errors.Is(failed.out.Err, failed.backend.primary) ||
		errors.Is(failed.out.Err, failed.backend.secondary) || failed.out.Receipt.Status != "incomplete" ||
		len(failed.execution.Runs) != 0 || failed.out.Receipt.RootWork.PrefillTokens != nil {
		t.Fatalf("original opaque-constructor panic or failed output changed: %+v", failed.out)
	}
	if failed.backend.cleanupAttempts != 1 || failed.backend.cleanupCompletions != 0 ||
		failed.backend.kv[1].frees != 0 || failed.backend.recycles != 1 {
		t.Fatalf("constructor cleanup did not stop before actual later release hooks: %+v", failed.backend)
	}
	// The constructor separately unwinds its model pin. Successful model close
	// therefore cannot establish completion of its hidden Session retirement.
	if failed.modelCloseErr != nil || failed.modelCloses != 1 || len(failed.out.Receipt.CleanupFailures) != 0 {
		t.Fatalf("unrelated model cleanup must succeed in this discriminator: %+v", failed)
	}
	t.Log("ROOT_RAWDECODE_CONSTRUCTOR_RETIREMENT_UNOBSERVED")
	if failed.out.Receipt.CleanupCompleted != nil && *failed.out.Receipt.CleanupCompleted {
		t.Fatal("ROOT_RAWDECODE_RETIREMENT_UNPROVEN: failed opaque root constructor interrupted Session cleanup but lifecycle reported CleanupCompleted=true")
	}
	if failed.out.Receipt.CleanupCompleted != nil {
		t.Fatal("unobserved root retirement must remain unknown without a reported release failure")
	}
}

type v41EPRootRetirementResult struct {
	out           v41EPOwnerLifecycleOutcome
	execution     rawdecode.Execution
	backend       *v41EPRootRetirementBackend
	modelCloses   int
	modelCloseErr error
}

func v41EPRunRootRetirement(t *testing.T, interrupted bool) (result v41EPRootRetirementResult) {
	t.Helper()
	f := newV41EPLifecycleFake()
	f.base, f.deps.Now = time.Now(), time.Now
	f.req.IgnoreEOS = true
	// Only the public model/rawdecode path is real. Admission, artifact labels,
	// supervisor, group and outer backend hooks remain synthetic control flow.
	// No checkpoint, native backend, resource window or network is exercised.
	m := model.NewSynthetic(model.Config{
		ModelType: "llama", HiddenSize: 8, IntermediateSize: 16, VocabSize: 8,
		NumLayers: 1, NumHeads: 2, NumKVHeads: 1, HeadDim: 4,
		RMSNormEps: 1e-5, RopeTheta: 10000, EOSTokenID: -1,
	})
	b := &v41EPRootRetirementBackend{Backend: compute.Default(),
		primary: errors.New("actual root Session constructor panic"), secondary: errors.New("actual constructor Session cleanup panic")}
	result.backend = b
	m.SetWeightCloser(v41EPRootRetirementCloser(func() error { result.modelCloses++; return nil }))
	initial, err := m.NewBackendSessionChecked(b)
	if err != nil {
		_ = m.CloseWeights()
		t.Fatal(err)
	}
	initialAttempted, initialCompleted, modelAttempted, rootReturned := false, false, false, false
	var rootRetirement v41EPOwnerRootRetirement
	observations := 0
	// Explicitly replace the nonallocating fake's complete observation. This
	// actual executor stays unknown until its own successful return proves Close
	// returned; the failed opaque constructor supplies no retirement result.
	f.loaded.RootRetirement = func() (v41EPOwnerRootRetirement, error) {
		observations++
		if !rootReturned {
			return v41EPOwnerRootRetirement{}, errors.New("root retirement observed before return")
		}
		return rootRetirement, nil
	}
	t.Cleanup(func() {
		if !initialAttempted {
			initial.Close()
		}
		if !modelAttempted {
			_ = m.CloseWeights()
		}
		// The fixture retains its CPU-only KV wrappers, allowing final host
		// retirement after evidence collection without retrying Session.Close.
		for _, kv := range b.kv {
			if kv.frees == 0 {
				kv.Free()
			}
		}
	})
	f.loaded.Value = initial
	f.loaded.CloseSession = func() error {
		f.event("session")
		initialAttempted = true
		initial.Close()
		initialCompleted = true
		return nil
	}
	f.loaded.CloseModel = func() error {
		f.event("model")
		modelAttempted = true
		result.modelCloseErr = m.CloseWeights()
		return result.modelCloseErr
	}
	f.deps.Root = func(ctx context.Context, req V41EPOwnerRequest, _ *v41EPOwnerLifecycleLoaded) (*V41EPOwnerRootWork, error) {
		f.event("root")
		defer func() { rootReturned = true }()
		if !initialCompleted || len(b.kv) != 1 || b.kv[0].frees != 1 {
			return nil, errors.New("initial Session was not retired before fresh root construction")
		}
		b.interrupted = interrupted // Never fault the lifecycle-owned initial Session.
		var err error
		result.execution, err = rawdecode.ExecuteModelContext(ctx, rawdecode.Request{
			BackendName: b.Name(), PromptTokenIDs: slices.Clone(req.PromptTokenIDs),
			ContextLimit: req.ContextLimit, GeneratedTokenLimit: req.GeneratedTokenLimit,
			Repetitions: 1, IgnoreEOS: req.IgnoreEOS,
		}, m, b)
		if err != nil {
			return nil, err
		}
		rootRetirement = v41EPOwnerRootRetirement{Available: true, Completed: true}
		generation, ok := result.execution.GenerationObservation(0)
		if !ok || len(result.execution.Runs) != 1 || !slices.Equal(result.execution.PromptTokenIDs, req.PromptTokenIDs) {
			return nil, errors.New("actual root generation observation unavailable")
		}
		if _, available := result.execution.TimingObservation(0); available || result.execution.Runs[0].BackendExecution != nil {
			return nil, errors.New("CPU fixture gained resource-window timing evidence")
		}
		tokens, ok := generation.OutputTokenIDs()
		eos, eosOK := generation.EOSStopped()
		ignore, ignoreOK := generation.IgnoreEOS()
		if !ok || !eosOK || !ignoreOK || ignore != req.IgnoreEOS || len(tokens) != req.GeneratedTokenLimit || eos {
			return nil, errors.New("actual root generation policy changed")
		}
		prompt, steps := len(req.PromptTokenIDs), len(tokens)-1
		run := result.execution.Runs[0]
		// These are software durations from this immediate successful call,
		// not sealed TimingObservation or device-resource evidence.
		setup, prefill, sample := int64(run.SessionSetupDuration), int64(run.PrefillDuration), int64(run.FirstSampleDuration)
		decode, teardown := int64(run.DecodeDuration), int64(run.TeardownDuration)
		return &V41EPOwnerRootWork{PrefillTokens: &prompt, CompletedStepCalls: &steps,
			SessionSetupNS: &setup, PrefillNS: &prefill, FirstSampleNS: &sample,
			DecodeLoopNS: &decode, SessionTeardownNS: &teardown, GeneratedTokenIDs: tokens, EOSStopped: &eos}, nil
	}
	result.out = runV41EPOwnerLifecycle(context.Background(), f.req, 0, f.deps)
	if !initialAttempted || !initialCompleted || !rootReturned || !modelAttempted || observations != 1 ||
		b.constructors != 2 || len(b.kv) != 2 || b.kv[0].frees != 1 ||
		f.count("session") != 1 || f.count("model") != 1 || f.count("group") != 1 || f.count("backend") != 1 ||
		!f.before("session", "root") || !f.before("root", "model") ||
		result.out.Receipt.ExecutionJoined == nil || !*result.out.Receipt.ExecutionJoined {
		t.Fatalf("actual root ownership or joined order changed: outcome=%+v backend=%+v events=%v", result.out, b, f.events())
	}
	return result
}

type v41EPRootRetirementBackend struct {
	compute.Backend
	primary, secondary                  error
	interrupted, cleanupArmed           bool
	constructors, recycles              int
	cleanupAttempts, cleanupCompletions int
	kv                                  []*v41EPRootRetirementKV
}

func (b *v41EPRootRetirementBackend) NewKV(cfg compute.KVConfig) compute.KVStore {
	kv := &v41EPRootRetirementKV{KVStore: b.Backend.NewKV(cfg)}
	b.kv = append(b.kv, kv)
	return kv
}

func (b *v41EPRootRetirementBackend) GraphReset() {
	b.constructors++
	if b.interrupted {
		b.cleanupArmed = true
		panic(b.primary)
	}
}

func (*v41EPRootRetirementBackend) BeginBatch() {}
func (b *v41EPRootRetirementBackend) FlushBatch() {
	if b.cleanupArmed {
		b.cleanupAttempts++
		panic(b.secondary)
	}
}
func (b *v41EPRootRetirementBackend) Recycle() {
	b.recycles++
	if b.cleanupArmed {
		b.cleanupCompletions++
	}
}

type v41EPRootRetirementKV struct {
	compute.KVStore
	frees int
}

func (kv *v41EPRootRetirementKV) KVConfig() compute.KVConfig {
	return kv.KVStore.(compute.KVGeometry).KVConfig()
}

func (kv *v41EPRootRetirementKV) Free() {
	kv.frees++
	if free, ok := kv.KVStore.(interface{ Free() }); ok {
		free.Free()
	}
}

type v41EPRootRetirementCloser func() error

func (close v41EPRootRetirementCloser) Close() error { return close() }

// fak-test:runtime fast est=20ms lane=default
func TestV41EPOwnerLifecycleRootEvidence(t *testing.T) {
	for _, mode := range []string{"complete", "failed-release", "unavailable", "missing", "unavailable-complete", "complete-error", "failed-without-error", "observer-error", "observer-panic", "unknown-plus-release-error"} {
		t.Run(mode, func(t *testing.T) {
			f := newV41EPLifecycleFake()
			primary, release, observation := errors.New("root execution"), errors.New("owned root release"), errors.New("root observer")
			returned, observations := false, 0
			f.deps.Root = func(context.Context, V41EPOwnerRequest, *v41EPOwnerLifecycleLoaded) (*V41EPOwnerRootWork, error) {
				f.event("root")
				returned = true
				return nil, primary
			}
			f.loaded.RootRetirement = func() (v41EPOwnerRootRetirement, error) {
				observations++
				f.event("root-retirement")
				if !returned {
					return v41EPOwnerRootRetirement{}, errors.New("root observer preceded return")
				}
				switch mode {
				case "complete":
					return v41EPOwnerRootRetirement{Available: true, Completed: true}, nil
				case "failed-release":
					return v41EPOwnerRootRetirement{Available: true, Err: release}, nil
				case "unavailable-complete":
					return v41EPOwnerRootRetirement{Completed: true}, nil
				case "complete-error":
					return v41EPOwnerRootRetirement{Available: true, Completed: true, Err: release}, nil
				case "failed-without-error":
					return v41EPOwnerRootRetirement{Available: true}, nil
				case "observer-error":
					return v41EPOwnerRootRetirement{}, observation
				case "observer-panic":
					panic(observation)
				default:
					return v41EPOwnerRootRetirement{}, nil
				}
			}
			if mode == "missing" {
				f.loaded.RootRetirement = nil
			}
			if mode == "unknown-plus-release-error" {
				f.loaded.CloseModel = func() error { f.event("model"); return release }
			}
			out := runV41EPOwnerLifecycle(context.Background(), f.req, 0, f.deps)
			wantObservations := 1
			if mode == "missing" {
				wantObservations = 0
			}
			if !returned || observations != wantObservations || !errors.Is(out.Err, primary) || out.PrimaryPanic != nil ||
				out.Receipt.Status != "incomplete" || out.Receipt.ExecutionJoined == nil || !*out.Receipt.ExecutionJoined ||
				out.Receipt.RootWork.PrefillTokens != nil || f.has("shutdown") || f.count("session") != 1 ||
				f.count("model") != 1 || f.count("group") != 1 || f.count("backend") != 1 ||
				mode != "missing" && !f.before("root-retirement", "model") {
				t.Fatalf("root observation or ownership changed: mode=%s out=%+v events=%v", mode, out, f.events())
			}
			switch mode {
			case "complete":
				if out.Receipt.CleanupCompleted == nil || !*out.Receipt.CleanupCompleted || len(out.Receipt.CleanupFailures) != 0 {
					t.Fatal("observed root retirement lost")
				}
			case "failed-release", "unknown-plus-release-error":
				if out.Receipt.CleanupCompleted == nil || *out.Receipt.CleanupCompleted || len(out.Receipt.CleanupFailures) != 1 || !errors.Is(out.Err, release) {
					t.Fatal("known release failure did not dominate unknown")
				}
			default:
				if out.Receipt.CleanupCompleted != nil || len(out.Receipt.CleanupFailures) != 0 {
					t.Fatal("unknown or malformed observation fabricated release evidence")
				}
			}
			if (mode == "observer-error" || mode == "observer-panic") && !errors.Is(out.Err, observation) {
				t.Fatal("observer error identity lost")
			}
		})
	}
	t.Run("observer-skipped-before-root-and-on-follower", func(t *testing.T) {
		for _, rank := range []int{0, 1} {
			f := newV41EPLifecycleFake()
			f.group.Rank = rank
			f.backend.Identity.Device = f.req.Owners[rank].ExpectedDevice
			if rank == 0 {
				f.loaded.CloseSession = func() error { f.event("session"); return errors.New("initial release") }
			}
			observations := 0
			f.loaded.RootRetirement = func() (v41EPOwnerRootRetirement, error) { observations++; return v41EPOwnerRootRetirement{}, nil }
			out := runV41EPOwnerLifecycle(context.Background(), f.req, rank, f.deps)
			if observations != 0 || f.has("root") || rank == 0 && out.Err == nil || rank == 1 && out.Err != nil {
				t.Fatalf("root observer ran outside its invocation: rank=%d out=%+v", rank, out)
			}
		}
	})
	t.Run("observer-follows-watcher-join-and-late-cancel-keeps-known-close", func(t *testing.T) {
		f := newV41EPLifecycleFake()
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		entered, release := make(chan struct{}), make(chan struct{})
		var stop sync.Once
		var observations atomic.Int32
		f.group.StopIO = func(error) error {
			stop.Do(func() { close(entered); <-release; f.event("stop-returned") })
			return nil
		}
		f.deps.Root = func(context.Context, V41EPOwnerRequest, *v41EPOwnerLifecycleLoaded) (*V41EPOwnerRootWork, error) {
			f.event("root")
			cancel()
			return f.work, nil
		}
		f.loaded.RootRetirement = func() (v41EPOwnerRootRetirement, error) {
			observations.Add(1)
			if !f.has("stop-returned") {
				return v41EPOwnerRootRetirement{}, errors.New("root observer preceded watcher join")
			}
			return v41EPOwnerRootRetirement{Available: true, Completed: true}, nil
		}
		done := make(chan v41EPOwnerLifecycleOutcome, 1)
		go func() { done <- runV41EPOwnerLifecycle(ctx, f.req, 0, f.deps) }()
		enteredOnTime := false
		select {
		case <-entered:
			enteredOnTime = true
		case <-time.After(2 * time.Second):
			cancel()
		}
		beforeJoin := observations.Load()
		close(release)
		// Always join before assertions or fixture retirement.
		out := <-done
		if !enteredOnTime || beforeJoin != 0 || observations.Load() != 1 || !errors.Is(out.Err, context.Canceled) ||
			out.Receipt.CleanupCompleted == nil || !*out.Receipt.CleanupCompleted || len(out.Receipt.CleanupFailures) != 0 ||
			out.Receipt.RootWork.PrefillTokens != nil || f.has("shutdown") {
			t.Fatalf("joined root retirement or late cancellation changed: before=%d after=%d out=%+v", beforeJoin, observations.Load(), out)
		}
	})
}
