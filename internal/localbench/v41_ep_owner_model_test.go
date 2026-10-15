package localbench

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/rawdecode"
)

// fak-test:runtime fast est=10ms lane=default
func TestV41EPOwnerModelAdapterLoad(t *testing.T) {
	for _, mode := range []string{"preflight-cancel", "missing-file", "panic-error", "panic-nil"} {
		t.Run(mode, func(t *testing.T) {
			t.Setenv("FAK_GGUF_MMAP", "off")
			if mode == "panic-nil" {
				// Exercise the actual result API's independent panic-presence bit.
				// t.Setenv restores compatibility mode before any CPU execution test.
				t.Setenv("GODEBUG", "panicnil=1")
			}
			f := newV41EPLifecycleFake()
			f.base, f.deps.Now = time.Now(), time.Now
			missing := filepath.Join(t.TempDir(), "missing.gguf")
			f.req.Shards = []ggufload.ShardIdentity{{Path: missing, Size: 1, RawFileSHA256: strings.Repeat("a", 64)}}
			manifest, err := json.Marshal(f.req.Shards)
			if err != nil {
				t.Fatal(err)
			}
			sum := sha256.Sum256(manifest)
			f.req.ExpectedShardSetSHA256 = hex.EncodeToString(sum[:])
			conn := &v41EPFollowerRetirementConn{reader: bytes.NewReader(nil)}
			group, err := model.Join(conn, 1, 3)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = conn.Close(); _ = group.Close() })
			primary := errors.New("actual loader Caps panic")
			backend := &v41EPModelLoadBackend{v41EPLoadRetirementBackend: &v41EPLoadRetirementBackend{},
				panics: strings.HasPrefix(mode, "panic"), payload: primary}
			if mode == "panic-nil" {
				backend.payload = nil
			}
			f.backend.Value, f.group.Value, f.group.Rank = backend, group, 1
			f.backend.Identity.Device = f.req.Owners[1].ExpectedDevice
			f.group.StopIO = func(error) error { return conn.Close() }
			f.group.Close = group.Close
			deps, adapter := bindV41EPOwnerModelDeps(f.deps, v41EPOwnerPanicModeInput{environment: os.Getenv("GODEBUG"), observed: true})
			load := deps.Load
			if mode == "preflight-cancel" {
				deps.Load = func(ctx context.Context, req V41EPOwnerRequest, be *v41EPOwnerLifecycleBackend, g *v41EPOwnerLifecycleGroup) (*v41EPOwnerLifecycleLoaded, error) {
					cancelled, cancel := context.WithCancel(ctx)
					cancel()
					return load(cancelled, req, be, g)
				}
			}
			out := runV41EPOwnerLifecycle(t.Context(), f.req, 1, deps)
			r := adapter.loadResult
			if !adapter.loadCalled || !adapter.loadReturned || r.Err == nil || r.OwnershipTransferred || r.Session != nil ||
				adapter.loaded != nil || !errors.Is(out.Err, r.Err) || out.PrimaryPanic != nil ||
				out.Receipt.ArtifactAuthenticated != nil || out.Receipt.ExecutionJoined != nil || len(out.Receipt.CleanupFailures) != 0 {
				t.Fatalf("actual refusal ownership/error changed: result=%+v outcome=%+v", r, out)
			}
			if mode == "missing-file" {
				if !errors.Is(r.Err, os.ErrNotExist) || r.Retirement.Available || out.Receipt.CleanupCompleted != nil {
					t.Fatalf("opaque failed open gained retirement credit: %+v %+v", r, out)
				}
			} else if !r.Retirement.Available || !r.Retirement.Completed || r.Retirement.Err != nil ||
				out.Receipt.CleanupCompleted == nil || !*out.Receipt.CleanupCompleted {
				t.Fatalf("observed no-owned-work refusal lost retirement: %+v %+v", r, out)
			}
			if strings.HasPrefix(mode, "panic") {
				var original, carried *ggufload.V41ExpertParallelLoadPanicError
				if !r.Panicked || r.PrimaryPanic != backend.payload || !errors.As(r.Err, &original) ||
					!errors.As(out.Err, &carried) || carried != original || original.PanicValue() != backend.payload {
					t.Fatalf("panic presence/payload/carrier identity lost: result=%+v outcome=%+v", r, out)
				}
			} else if r.Panicked || mode == "preflight-cancel" && !errors.Is(r.Err, context.Canceled) {
				t.Fatalf("ordinary refusal changed: %+v", r)
			}
			if again, err := load(t.Context(), f.req, f.backend, f.group); err == nil || again != nil || adapter.loadResult.Err != r.Err {
				t.Fatal("second Load changed the invocation result")
			}
		})
	}
}

type v41EPModelLoadBackend struct {
	*v41EPLoadRetirementBackend
	panics  bool
	payload any
}

func (b *v41EPModelLoadBackend) Caps() compute.Caps {
	if b.panics {
		panic(b.payload)
	}
	return b.v41EPLoadRetirementBackend.Caps()
}

// fak-test:runtime integration est=100ms lane=default
func TestV41EPOwnerModelAdapterBinding(t *testing.T) {
	for _, mode := range []string{"healthy", "report-budget", "cancel-after-close", "constructor-panic", "cancel-before-root", "panicnil-refusal", "missing-panic-settings", "malformed-panic-settings", "stale-modern-settings", "second-factory"} {
		t.Run(mode, func(t *testing.T) {
			t.Setenv("FAK_PAGED_KV", "off")
			// Explicitly reset modern semantics after the separate scoped loader test.
			t.Setenv("GODEBUG", "panicnil=0")
			if mode == "panicnil-refusal" {
				t.Setenv("GODEBUG", "panicnil=0,panicnil=1")
			}
			if mode == "malformed-panic-settings" {
				// Invalid dynamic input leaves the prior modern setting in place,
				// but the captured ambiguity must still refuse execution.
				t.Setenv("GODEBUG", "panicnil=0,panicnil=bad")
			}
			fixture := newV41EPGroupTCPFixture(t, 3)
			var joins [3]<-chan v41EPOwnerPhaseResult
			for rank := range joins {
				joins[rank] = fixture.startJoin(rank, fixture.req.Budgets.JoinNS)
			}
			for _, joined := range joins {
				if r := fixture.await(t, joined); r.err() != nil || r.panic != nil {
					fixture.fail(t, "actual group Join: %v panic=%v", r.err(), r.panic)
				}
			}
			req := fixture.req
			req.IgnoreEOS = true
			if mode == "report-budget" {
				req.Budgets.ForwardNS = time.Nanosecond // Reporting bound, not fixture I/O deadline.
			}
			var peers [3]*v41EPModelPeer
			for rank := range peers {
				peers[rank] = newV41EPModelPeer(t, fixture, rank, req)
			}
			root := peers[0]
			// This is an internal CPU model binding, not a successful model Load.
			// No lifecycle success receipt or ArtifactAuthenticated credit is made.
			if work, err := root.deps.Root(fixture.ctx, req, root.loaded); work != nil || err == nil || root.backend.constructors != 1 {
				fixture.fail(t, "Root ran before actual initial Session.Close")
			}
			if work, err := root.deps.Root(fixture.ctx, req, peers[1].loaded); work != nil || err == nil {
				fixture.fail(t, "cross-invocation bundle accepted")
			}
			changed := req
			changed.PromptTokenIDs = append(slices.Clone(req.PromptTokenIDs), 2)
			if work, err := root.deps.Root(fixture.ctx, changed, root.loaded); work != nil || err == nil {
				fixture.fail(t, "changed request accepted")
			}
			if err := root.loaded.CloseSession(); err != nil || root.backend.kv[0].frees != 1 {
				fixture.fail(t, "initial root Close did not return: %v", err)
			}
			if mode == "constructor-panic" {
				root.backend.interrupted = true // Arm only after the initial Close completed.
			}
			if mode == "missing-panic-settings" {
				root.adapter.panicMode = v41EPOwnerPanicModeInput{}
			}
			if mode == "stale-modern-settings" {
				// Deliberately violate capture stability: the live probe must still
				// refuse before a fresh Session can produce retirement evidence.
				t.Setenv("GODEBUG", "panicnil=1")
			}
			var follower [3]<-chan v41EPOwnerPhaseResult
			var entered [3]chan struct{}
			for rank := 1; rank < 3; rank++ {
				peer := peers[rank]
				entered[rank] = make(chan struct{})
				follower[rank] = fixture.startPhase(rank, fixture.req.Budgets.ForwardNS, func(ctx context.Context) error {
					close(entered[rank])
					return peer.deps.Follower(ctx, peer.loaded, func() error { peer.claims++; return nil })
				})
			}
			// Establish callback entry, not factory acceptance. A fast no-forward
			// root may otherwise cancel a phase before its adapter ever runs.
			// All joined-result assertions below remain unchanged; no Session
			// is claimed until the actual protocol asks for its first Prefill.
			barrier, cancelBarrier := context.WithTimeout(fixture.ctx, 2*time.Second)
			defer cancelBarrier()
			for rank := 1; rank < 3; rank++ {
				select {
				case <-entered[rank]:
				case <-barrier.Done():
					fixture.fail(t, "rank %d follower callback did not enter: %v", rank, barrier.Err())
				}
			}
			cancelBarrier()
			var work *V41EPOwnerRootWork
			rootRun := fixture.startPhase(0, fixture.req.Budgets.ForwardNS, func(ctx context.Context) error {
				if mode == "cancel-after-close" {
					cancelled, cancel := context.WithCancel(ctx)
					defer cancel()
					ctx = cancelled
					root.backend.afterFreshClose = cancel
				}
				if mode == "cancel-before-root" {
					cancelled, cancel := context.WithCancel(ctx)
					cancel()
					ctx = cancelled
				}
				var err error
				work, err = root.deps.Root(ctx, req, root.loaded)
				if err != nil {
					return err
				}
				if mode == "second-factory" {
					// A second real root Session sends another PREFILL at position 0.
					// The adapter's follower factory must refuse reusing its closed
					// Session before a second claim/handoff can occur.
					_, err := rawdecode.ExecuteModelContext(ctx, v41EPModelRawRequest(req), root.model, root.backend)
					return err
				}
				return root.deps.Shutdown(ctx, root.loaded)
			})
			rootResult := fixture.await(t, rootRun)
			if mode != "healthy" {
				fixture.cancel() // Join interrupted idle followers before reading or releasing.
			}
			var followerResult [3]v41EPOwnerPhaseResult
			for rank := 1; rank < 3; rank++ {
				followerResult[rank] = fixture.await(t, follower[rank])
			}
			retired, observeErr := root.loaded.RootRetirement()
			if observeErr != nil || !root.adapter.state.executionReturned {
				t.Fatalf("root snapshot preceded joined return: %+v %v", retired, observeErr)
			}
			switch mode {
			case "healthy", "second-factory":
				if work == nil || !retired.Available || !retired.Completed || retired.Err != nil {
					t.Fatalf("successful direct execution lost Close: work=%+v retirement=%+v", work, retired)
				}
				if mode == "healthy" && (rootResult.err() != nil || rootResult.panic != nil || !root.adapter.state.shutdownCalled) {
					t.Fatalf("actual model protocol failed: %+v", rootResult)
				}
			case "report-budget":
				if work != nil || rootResult.err() == nil || !retired.Available || !retired.Completed || root.adapter.state.rootComplete {
					t.Fatalf("later report error erased completed Close: result=%+v retirement=%+v", rootResult, retired)
				}
			case "cancel-after-close":
				if work != nil || !errors.Is(rootResult.err(), context.Canceled) || !retired.Available || !retired.Completed || root.adapter.state.rootComplete || !root.backend.cancelled {
					t.Fatalf("late cancellation erased observed Close: result=%+v retirement=%+v", rootResult, retired)
				}
			case "constructor-panic":
				if rootResult.panic != root.backend.primary || !errors.Is(rootResult.err(), root.backend.primary) ||
					retired.Available || retired.Completed || retired.Err != nil || root.backend.cleanupAttempts != 1 || root.backend.cleanupCompletions != 0 {
					t.Fatalf("opaque failed retirement was credited: result=%+v retirement=%+v", rootResult, retired)
				}
			case "panicnil-refusal", "missing-panic-settings", "malformed-panic-settings", "stale-modern-settings":
				want := "panicnil compatibility or ambiguous runtime setting"
				if mode == "missing-panic-settings" {
					want = "observed runtime panic settings unavailable"
				} else if mode == "stale-modern-settings" {
					want = "live panicnil compatibility refuses root execution"
				}
				if rootResult.err() == nil || !strings.Contains(rootResult.err().Error(), want) || retired.Available || retired.Completed || retired.Err != nil || len(root.backend.kv) != 1 || root.backend.constructors != 1 || root.adapter.state.rootComplete {
					t.Fatalf("unproven panic mode entered opaque execution: %+v %+v", rootResult, retired)
				}
			case "cancel-before-root":
				if !errors.Is(rootResult.err(), context.Canceled) || retired.Available || len(root.backend.kv) != 1 {
					t.Fatalf("canceled call gained execution evidence: %+v %+v", rootResult, retired)
				}
			}
			for rank := 1; rank < 3; rank++ {
				peer := peers[rank]
				r, err := peer.loaded.FollowerRetirement()
				if err != nil || !r.Available || !peer.adapter.state.executionReturned {
					t.Fatalf("rank %d result was not joined: %+v %v", rank, r, err)
				}
				accepted := mode != "constructor-panic" && mode != "cancel-before-root" && mode != "panicnil-refusal" &&
					mode != "missing-panic-settings" && mode != "malformed-panic-settings" && mode != "stale-modern-settings"
				if accepted {
					if peer.claims != 1 || r.AcceptedSessions != 1 || r.CloseAttempts != 1 || r.CloseCompletions != 1 || peer.backend.kv[0].frees != 1 {
						t.Fatalf("rank %d handoff/Close changed: claims=%d retirement=%+v", rank, peer.claims, r)
					}
				} else {
					if peer.claims != 0 || r.AcceptedSessions != 0 || r.CloseAttempts != 0 || r.CloseCompletions != 0 {
						t.Fatalf("rank %d claimed before Prefill: %+v", rank, r)
					}
					if err := peer.loaded.CloseSession(); err != nil {
						t.Fatal(err)
					}
				}
				if mode == "healthy" && followerResult[rank].err() != nil || mode == "second-factory" && (peer.adapter.state.followerResult.Err == nil || !strings.Contains(peer.adapter.state.followerResult.Err.Error(), "factory is one-shot")) {
					t.Fatalf("rank %d follower protocol result changed: %+v", rank, followerResult[rank])
				}
				if err := peer.deps.Follower(context.Background(), peer.loaded, func() error { peer.claims++; return nil }); err == nil {
					t.Fatal("second Follower accepted")
				}
			}
			if again, err := root.deps.Root(context.Background(), req, root.loaded); err == nil || again != nil {
				t.Fatal("second Root accepted")
			}
			for rank, peer := range peers {
				if peer.loaded.OwnerCounts != nil {
					t.Fatal("CPU binding gained authenticated owner statistics")
				}
				if err := peer.loaded.CloseModel(); err != nil || peer.modelCloses != 1 {
					t.Fatalf("rank %d actual model/pin release: %v closes=%d", rank, err, peer.modelCloses)
				}
			}
			fixture.retire(t)
			fixture.assertRetired(t, 2)
			if mode == "healthy" {
				oracle := model.NewSynthetic(v41EPModelTinyConfig())
				execution, err := rawdecode.ExecuteModelContext(t.Context(), v41EPModelRawRequest(req), oracle, compute.Default())
				closeErr := oracle.CloseWeights()
				generation, available := execution.GenerationObservation(0)
				tokens, tokenOK := generation.OutputTokenIDs()
				_, timingAvailable := execution.TimingObservation(0)
				if err != nil || closeErr != nil || !available || !tokenOK || len(tokens) != 2 || !slices.Equal(tokens, work.GeneratedTokenIDs) ||
					*work.PrefillTokens != 2 || *work.CompletedStepCalls != 1 || timingAvailable || execution.Runs[0].BackendExecution != nil ||
					len(root.backend.kv) != 2 || root.backend.kv[1].frees != 1 {
					t.Fatalf("actual internal model binding/scalar oracle disagreed: err=%v close=%v output=%+v execution=%+v", err, closeErr, work, execution)
				}
				t.Log("INTERNAL_CPU_MODEL_BINDING_CONFIRMED: actual Prefill, Step, follower handoff, Shutdown and fresh-session Close; no authenticated-loader success")
			}
		})
	}
}

type v41EPModelPeer struct {
	adapter     *v41EPOwnerModelAdapter
	deps        v41EPOwnerLifecycleDeps
	loaded      *v41EPOwnerLifecycleLoaded
	model       *model.Model
	backend     *v41EPModelCPUBackend
	claims      int
	modelCloses int
}

func newV41EPModelPeer(t *testing.T, f *v41EPGroupTCPFixture, rank int, req V41EPOwnerRequest) *v41EPModelPeer {
	t.Helper()
	p := &v41EPModelPeer{model: model.NewSynthetic(v41EPModelTinyConfig()), backend: &v41EPModelCPUBackend{v41EPRootRetirementBackend: &v41EPRootRetirementBackend{
		Backend: compute.Default(), primary: errors.New("adapter actual constructor panic"), secondary: errors.New("adapter actual constructor Close panic")}}}
	p.model.SetWeightCloser(v41EPRootRetirementCloser(func() error { p.modelCloses++; return nil }))
	initial, err := p.model.NewBackendSessionChecked(p.backend)
	if err != nil {
		_ = p.model.CloseWeights()
		f.fail(t, "initial checked CPU Session: %v", err)
	}
	g := f.groups[rank].Value.(*model.DistComm)
	var coordinator *model.EPDecodeCoordinator
	if rank == 0 {
		coordinator, err = model.NewEPDecodeCoordinator(g)
		if err != nil {
			initial.Close()
			_ = p.model.CloseWeights()
			f.fail(t, "actual root coordinator: %v", err)
		}
		p.model.SetEPDecodeCoordinator(coordinator)
	}
	p.deps, p.adapter = bindV41EPOwnerModelDeps(v41EPOwnerLifecycleDeps{}, v41EPOwnerPanicModeInput{environment: os.Getenv("GODEBUG"), observed: true})
	p.loaded, err = p.adapter.bindLoaded(req, p.backend, g, initial, coordinator)
	t.Cleanup(func() {
		f.close() // Join every fixture user before emergency ownership checks.
		if !p.adapter.state.initialAttempted && !p.adapter.state.factoryCalled {
			_ = p.loaded.CloseSession()
		}
		if !p.adapter.state.modelAttempted {
			_ = p.loaded.CloseModel()
		}
		// Retain/retire CPU fixture KV wrappers after evidence, without ever
		// retrying the failed opaque Session.Close.
		for _, kv := range p.backend.kv {
			if kv.frees == 0 {
				kv.Free()
			}
		}
	})
	if err != nil {
		f.fail(t, "internal CPU binding: %v", err)
	}
	return p
}

func v41EPModelTinyConfig() model.Config {
	return model.Config{ModelType: "llama", HiddenSize: 8, IntermediateSize: 16, VocabSize: 8,
		NumLayers: 1, NumHeads: 2, NumKVHeads: 1, HeadDim: 4, RMSNormEps: 1e-5, RopeTheta: 10000, EOSTokenID: -1}
}

func v41EPModelRawRequest(req V41EPOwnerRequest) rawdecode.Request {
	return rawdecode.Request{BackendName: "cpu-ref", PromptTokenIDs: slices.Clone(req.PromptTokenIDs),
		ContextLimit: req.ContextLimit, GeneratedTokenLimit: req.GeneratedTokenLimit, Repetitions: 1, IgnoreEOS: req.IgnoreEOS}
}

// Name is consulted while ExecuteModelContext describes its already-completed
// return. The cancellation control fires only after the fresh actual KV Free,
// never during initial Close or failed constructor retirement.
type v41EPModelCPUBackend struct {
	*v41EPRootRetirementBackend
	afterFreshClose context.CancelFunc
	cancelled       bool
}

func (b *v41EPModelCPUBackend) Name() string {
	if b.afterFreshClose != nil && len(b.kv) == 2 && b.kv[1].frees == 1 {
		b.cancelled = true
		b.afterFreshClose()
	}
	return b.Backend.Name()
}

// fak-test:runtime fast est=1ms lane=default
func TestV41EPOwnerModelAdapterPanicMode(t *testing.T) {
	for _, tc := range []struct {
		name, defaults, environment string
		refuse                      bool
	}{
		{"modern-default", "", "", false},
		{"old-compiled-default", "panicnil=1", "", true},
		{"environment-overrides-default", "panicnil=1", "panicnil=0", false},
		{"last-zero", "", "panicnil=1,panicnil=0", false},
		{"last-one", "", "panicnil=0,panicnil=1", true},
		{"positive-sign", "", "panicnil=+1", true},
		{"decimal-leading-zero", "", "panicnil=01", true},
		{"other-integer", "", "panicnil=2", false},
		{"substring-is-not-key", "", "otherpanicnil=1", false},
		{"space-is-not-key", "", " panicnil=1", false},
		{"invalid-keeps-prior", "", "panicnil=1,panicnil=bad", true},
		{"overflow", "", "panicnil=2147483648", true},
		{"empty-value", "", "panicnil=", true},
		{"bisect-is-not-runtime-integer", "", "panicnil=1#y", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := v41EPOwnerCheckPanicnil(tc.defaults, tc.environment); (err != nil) != tc.refuse {
				t.Fatalf("effective panic mode: %v refuse=%v", err, tc.refuse)
			}
		})
	}
	t.Setenv("GODEBUG", "panicnil=0")
	input := v41EPOwnerPanicModeInput{environment: os.Getenv("GODEBUG"), observed: true}
	_, adapter := bindV41EPOwnerModelDeps(v41EPOwnerLifecycleDeps{}, input)
	input.environment, input.observed = "panicnil=1", false
	if err := v41EPOwnerRootPanicMode(adapter.panicMode); err != nil {
		t.Fatalf("current modern runtime or copied input changed: %v", err)
	}
	if err := v41EPOwnerRootPanicMode(input); err == nil {
		t.Fatal("missing observation was accepted")
	}
	t.Setenv("GODEBUG", "panicnil=1")
	if err := v41EPOwnerRootPanicMode(adapter.panicMode); err == nil {
		t.Fatal("stale or forged modern input admitted live compatibility mode")
	}
	t.Setenv("GODEBUG", "panicnil=0")
	t.Setenv("GODEBUG", "panicnil=0,panicnil=bad")
	input = v41EPOwnerPanicModeInput{environment: os.Getenv("GODEBUG"), observed: true}
	if !v41EPOwnerModernPanicnil() || v41EPOwnerRootPanicMode(input) == nil {
		t.Fatal("modern runtime probe bypassed ambiguous captured settings")
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestV41EPOwnerModelAdapterPartialBinding(t *testing.T) {
	t.Setenv("FAK_PAGED_KV", "off")
	m := model.NewSynthetic(v41EPModelTinyConfig())
	be := &v41EPRootRetirementBackend{Backend: compute.Default()}
	initial, err := m.NewBackendSessionChecked(be)
	if err != nil {
		_ = m.CloseWeights()
		t.Fatal(err)
	}
	conn := &v41EPFollowerRetirementConn{reader: bytes.NewReader(nil)}
	group, err := model.Join(conn, 1, 3)
	if err != nil {
		initial.Close()
		_ = m.CloseWeights()
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close(); _ = group.Close() })
	req := v41EPOwnerRequestFixture()
	req.Repetitions = 2 // Reporting/binding validation fails after actual adoption.
	_, adapter := bindV41EPOwnerModelDeps(v41EPOwnerLifecycleDeps{}, v41EPOwnerPanicModeInput{environment: os.Getenv("GODEBUG"), observed: true})
	loaded, err := adapter.bindLoaded(req, be, group, initial, nil)
	if loaded == nil {
		initial.Close()
		_ = m.CloseWeights()
		t.Fatal("partial binding discarded its owned Session")
	}
	t.Cleanup(func() {
		if !adapter.state.initialAttempted {
			_ = loaded.CloseSession()
		}
		if !adapter.state.modelAttempted {
			_ = loaded.CloseModel()
		}
	})
	if err == nil || loaded.CloseSession == nil || loaded.CloseModel == nil || loaded.OwnerCounts != nil || conn.closes != 0 {
		t.Fatalf("partial ownership or borrowed resources changed: loaded=%+v err=%v", loaded, err)
	}
	if err := loaded.CloseSession(); err != nil || len(be.kv) != 1 || be.kv[0].frees != 1 {
		t.Fatalf("returned initial Session was not actually retired: %v", err)
	}
	if err := loaded.CloseModel(); err != nil || conn.closes != 0 {
		t.Fatalf("model pin or borrowed group changed: %v closes=%d", err, conn.closes)
	}
	if err := loaded.CloseSession(); err == nil || be.kv[0].frees != 1 {
		t.Fatal("second release retried Session.Close")
	}
}
