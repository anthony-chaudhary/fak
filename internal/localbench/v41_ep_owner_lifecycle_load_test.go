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
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=10ms lane=default
func TestV41EPOwnerLifecycleLoadRetirement(t *testing.T) {
	// This control must pass before the sole expected-failure marker. The real
	// loader explicitly observed that preflight cancellation began no owned work.
	healthy := v41EPRunLoadRetirement(t, true)
	if !healthy.result.Retirement.Available || !healthy.result.Retirement.Completed ||
		healthy.result.Retirement.Err != nil || healthy.out.Receipt.CleanupCompleted == nil || !*healthy.out.Receipt.CleanupCompleted {
		t.Fatalf("known no-owned-work retirement control changed: result=%+v receipt=%+v", healthy.result, healthy.out.Receipt)
	}
	t.Log("LIFECYCLE_LOAD_PREFLIGHT_RETIREMENT_CONFIRMED")

	// An opaque failed opener provides no retirement observation. This does not
	// demonstrate a leaked descriptor; it demonstrates unsupported receipt credit.
	failed := v41EPRunLoadRetirement(t, false)
	if failed.result.Retirement.Available || failed.result.Retirement.Completed || failed.result.Retirement.Err != nil {
		t.Fatalf("actual failed opener unexpectedly proved retirement: %+v", failed.result)
	}
	if failed.out.Receipt.CleanupCompleted != nil && *failed.out.Receipt.CleanupCompleted {
		t.Fatal("LIFECYCLE_LOAD_RETIREMENT_UNPROVEN: failed opaque loader returned unavailable retirement but lifecycle reported CleanupCompleted=true")
	}
	if failed.out.Receipt.CleanupCompleted != nil {
		t.Fatal("unavailable factory retirement must remain unknown without an observed release failure")
	}
}

type v41EPLoadRetirementFixtureResult struct {
	result ggufload.V41ExpertParallelSessionResult
	out    v41EPOwnerLifecycleOutcome
}

func v41EPRunLoadRetirement(t *testing.T, preflightCancel bool) v41EPLoadRetirementFixtureResult {
	t.Helper()
	t.Setenv("FAK_GGUF_MMAP", "off")
	f := newV41EPLifecycleFake()
	f.base, f.deps.Now = time.Now(), time.Now
	missing := filepath.Join(t.TempDir(), "missing.gguf")
	if _, err := os.Stat(missing); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing-file fixture is not absent: %v", err)
	}
	f.req.Shards = []ggufload.ShardIdentity{{Path: missing, Size: 1, RawFileSHA256: strings.Repeat("a", 64)}}
	manifest, err := json.Marshal(f.req.Shards)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(manifest)
	f.req.ExpectedShardSetSHA256 = hex.EncodeToString(sum[:])
	if _, err := V41EPOwnerRequestDigest(f.req); err != nil {
		t.Fatalf("declared missing-file request is invalid: %v", err)
	}
	conn := &v41EPFollowerRetirementConn{reader: bytes.NewReader(nil)}
	group, err := model.Join(conn, 1, 3)
	if err != nil {
		_ = conn.Close()
		t.Fatal(err)
	}
	retired := false
	t.Cleanup(func() {
		if !retired {
			_ = conn.Close()
			_ = group.Close()
		}
	})
	f.group.Value, f.group.Rank = group, 1
	f.backend.Identity.Device = f.req.Owners[1].ExpectedDevice
	f.group.StopIO = func(error) error { f.event("stop-io"); return conn.Close() }
	returned, closeAfterReturn := false, false
	f.group.Close = func() error {
		f.event("group")
		closeAfterReturn = returned
		return group.Close()
	}
	backend := &v41EPLoadRetirementBackend{}
	var result ggufload.V41ExpertParallelSessionResult
	observations := 0
	f.deps.LoadRetirement = func() (v41EPOwnerLoadRetirement, error) {
		observations++
		if !returned {
			return v41EPOwnerLoadRetirement{}, errors.New("load retirement observed before joined return")
		}
		r := result.Retirement
		return v41EPOwnerLoadRetirement{Available: r.Available, Completed: r.Completed, Err: r.Err}, nil
	}
	f.deps.Load = func(ctx context.Context, req V41EPOwnerRequest, _ *v41EPOwnerLifecycleBackend, _ *v41EPOwnerLifecycleGroup) (*v41EPOwnerLifecycleLoaded, error) {
		f.event("load")
		if preflightCancel {
			cancelled, cancel := context.WithCancel(ctx)
			cancel()
			ctx = cancelled
		}
		result = ggufload.LoadV41ExpertParallelSessionWithResult(ctx, ggufload.V41ExpertParallelSessionRequest{
			Shards: req.Shards, ExpectedShardSetSHA256: req.ExpectedShardSetSHA256, SourceSHA256: req.SourceSHA256,
			Group: group, Backend: backend,
		})
		returned = true // The synchronous loader and its cleanup have returned.
		return nil, result.Err
	}
	out := runV41EPOwnerLifecycle(context.Background(), f.req, 1, f.deps)
	if !returned || !closeAfterReturn || observations != 1 || f.count("load") != 1 || result.Session != nil || result.Coordinator != nil || result.OwnershipTransferred || result.Panicked || result.PrimaryPanic != nil || result.Err == nil {
		t.Fatalf("actual failed loader ownership changed: returned=%v result=%+v", returned, result)
	}
	want := error(os.ErrNotExist)
	if preflightCancel {
		want = context.Canceled
		if result.Err != context.Canceled || backend.caps != 0 || backend.dtypes != 0 || backend.routed != 0 {
			t.Fatalf("preflight refusal did owned work or changed error identity: result=%+v backend=%+v", result, backend)
		}
	} else if backend.caps != 2 || backend.dtypes != 2 || backend.routed != 1 || errors.Is(result.Err, context.Canceled) || errors.Is(result.Err, context.DeadlineExceeded) {
		t.Fatalf("failed-open discriminator changed: result=%+v backend=%+v", result, backend)
	}
	if !errors.Is(result.Err, want) || !errors.Is(out.Err, result.Err) || !errors.Is(out.Err, want) || out.PrimaryPanic != nil || out.Receipt.Status != "incomplete" {
		t.Fatalf("primary failed-load identity changed: result=%+v outcome=%+v", result, out)
	}
	if out.Receipt.ArtifactAuthenticated != nil || out.Receipt.ExecutionJoined != nil || out.Receipt.ShutdownCompleted != nil ||
		out.Receipt.RootWork.PrefillTokens != nil || out.Receipt.Phases.ForwardNS != nil || len(out.Receipt.CleanupFailures) != 0 ||
		f.has("session") || f.has("model") || f.has("root") || f.has("follower") || f.has("shutdown") ||
		f.count("group") != 1 || f.count("backend") != 1 || !f.before("load", "group") || !f.before("group", "backend") ||
		conn.closes != 1 || conn.written != 4 || conn.reader.Len() != 0 {
		t.Fatalf("joined refusal or unrelated releases changed: outcome=%+v events=%v conn=%+v", out, f.events(), conn)
	}
	retired = true
	return v41EPLoadRetirementFixtureResult{result: result, out: out}
}

// Capability declarations admit only the actual failed-open path. No compute
// implementation, native backend initialization, or weight allocation is used.
type v41EPLoadRetirementBackend struct {
	compute.Backend
	caps, dtypes, routed int
}

func (*v41EPLoadRetirementBackend) Name() string { return "fixture-capability-only" }
func (b *v41EPLoadRetirementBackend) Caps() compute.Caps {
	b.caps++
	return compute.Caps{DeviceMemory: true, UploadDtype: true}
}
func (b *v41EPLoadRetirementBackend) SupportsDeviceWeightDtype(dtype compute.Dtype) bool {
	b.dtypes++
	return dtype == compute.Q2_K || dtype == compute.Q3_K
}
func (b *v41EPLoadRetirementBackend) V41RoutedSwiGLUAvailable() bool {
	b.routed++
	return true
}
func (*v41EPLoadRetirementBackend) V41RoutedSwiGLU(compute.Tensor, compute.Tensor, float32, float32) (compute.Tensor, error) {
	panic("unexpected compute in failed-open fixture")
}

// fak-test:runtime fast est=30ms lane=default
func TestV41EPOwnerLifecycleLoadEvidence(t *testing.T) {
	t.Run("typed-failed-factory-observations", func(t *testing.T) {
		for _, mode := range []string{"complete", "failed-release", "unavailable", "missing", "unavailable-complete", "unavailable-error", "complete-error", "failed-without-error", "observer-error", "observer-panic", "factory-panic"} {
			t.Run(mode, func(t *testing.T) {
				f := newV41EPLifecycleFake()
				primary, release, observation := errors.New("load primary"), errors.New("owned release"), errors.New("observation failure")
				returned, observations := false, 0
				f.deps.Load = func(context.Context, V41EPOwnerRequest, *v41EPOwnerLifecycleBackend, *v41EPOwnerLifecycleGroup) (*v41EPOwnerLifecycleLoaded, error) {
					f.event("load")
					defer func() { returned = true }()
					if mode == "factory-panic" {
						panic(primary)
					}
					return nil, primary
				}
				f.deps.LoadRetirement = func() (v41EPOwnerLoadRetirement, error) {
					observations++
					if !returned {
						return v41EPOwnerLoadRetirement{}, errors.New("observation preceded factory unwind")
					}
					switch mode {
					case "complete", "factory-panic":
						return v41EPOwnerLoadRetirement{Available: true, Completed: true}, nil
					case "failed-release":
						return v41EPOwnerLoadRetirement{Available: true, Err: release}, nil
					case "unavailable-complete":
						return v41EPOwnerLoadRetirement{Completed: true}, nil
					case "unavailable-error":
						return v41EPOwnerLoadRetirement{Err: release}, nil
					case "complete-error":
						return v41EPOwnerLoadRetirement{Available: true, Completed: true, Err: release}, nil
					case "failed-without-error":
						return v41EPOwnerLoadRetirement{Available: true}, nil
					case "observer-error":
						return v41EPOwnerLoadRetirement{}, observation
					case "observer-panic":
						panic(observation)
					default:
						return v41EPOwnerLoadRetirement{}, nil
					}
				}
				if mode == "missing" {
					f.deps.LoadRetirement = nil
				}
				out := runV41EPOwnerLifecycle(context.Background(), f.req, 0, f.deps)
				wantObservations := 1
				if mode == "missing" {
					wantObservations = 0
				}
				if !returned || observations != wantObservations || !errors.Is(out.Err, primary) || out.Receipt.Status != "incomplete" ||
					out.Receipt.ArtifactAuthenticated != nil || out.Receipt.ExecutionJoined != nil || f.has("session") || f.has("model") ||
					f.has("root") || f.has("shutdown") || f.count("group") != 1 || f.count("backend") != 1 {
					t.Fatalf("failed factory result or retirement changed: mode=%s out=%+v events=%v", mode, out, f.events())
				}
				if (mode == "factory-panic") != (out.PrimaryPanic == primary) || mode != "factory-panic" && out.PrimaryPanic != nil {
					t.Fatalf("primary panic precedence changed: mode=%s panic=%v", mode, out.PrimaryPanic)
				}
				switch mode {
				case "complete", "factory-panic":
					if out.Receipt.CleanupCompleted == nil || !*out.Receipt.CleanupCompleted || len(out.Receipt.CleanupFailures) != 0 {
						t.Fatalf("known complete factory retirement lost: %+v", out.Receipt)
					}
				case "failed-release":
					if out.Receipt.CleanupCompleted == nil || *out.Receipt.CleanupCompleted || len(out.Receipt.CleanupFailures) != 1 || !errors.Is(out.Err, release) {
						t.Fatalf("known release failure lost: %+v", out)
					}
				default:
					if out.Receipt.CleanupCompleted != nil || len(out.Receipt.CleanupFailures) != 0 {
						t.Fatalf("unknown observation invented release evidence: %+v", out.Receipt)
					}
				}
				if (mode == "observer-error" || mode == "observer-panic") && !errors.Is(out.Err, observation) {
					t.Fatal("observer failure was lost")
				}
			})
		}
	})
	t.Run("partial-bundle-retirement-and-false-dominates-unknown", func(t *testing.T) {
		for _, mode := range []string{"complete", "unknown", "release-fails"} {
			t.Run(mode, func(t *testing.T) {
				f := newV41EPLifecycleFake()
				primary, release := errors.New("partial Load"), errors.New("partial model release")
				f.deps.Load = func(context.Context, V41EPOwnerRequest, *v41EPOwnerLifecycleBackend, *v41EPOwnerLifecycleGroup) (*v41EPOwnerLifecycleLoaded, error) {
					return f.loaded, primary
				}
				f.deps.LoadRetirement = func() (v41EPOwnerLoadRetirement, error) {
					return v41EPOwnerLoadRetirement{Available: mode == "complete", Completed: mode == "complete"}, nil
				}
				if mode == "release-fails" {
					f.loaded.CloseModel = func() error { f.event("model"); return release }
				}
				out := runV41EPOwnerLifecycle(context.Background(), f.req, 0, f.deps)
				if !errors.Is(out.Err, primary) || f.count("session") != 1 || f.count("model") != 1 || f.count("group") != 1 || f.count("backend") != 1 || out.Receipt.ArtifactAuthenticated != nil {
					t.Fatalf("partial bundle was not retired exactly once: %+v %v", out, f.events())
				}
				switch mode {
				case "complete":
					if out.Receipt.CleanupCompleted == nil || !*out.Receipt.CleanupCompleted || len(out.Receipt.CleanupFailures) != 0 {
						t.Fatal("complete factory and transferred releases were not credited")
					}
				case "unknown":
					if out.Receipt.CleanupCompleted != nil || len(out.Receipt.CleanupFailures) != 0 {
						t.Fatal("clean transferred releases erased unknown factory retirement")
					}
				case "release-fails":
					if out.Receipt.CleanupCompleted == nil || *out.Receipt.CleanupCompleted || len(out.Receipt.CleanupFailures) != 1 || !errors.Is(out.Err, release) {
						t.Fatal("known release failure did not dominate unknown")
					}
				}
			})
		}
	})
	t.Run("incomplete-nil-bundle-cannot-skip-observation", func(t *testing.T) {
		f := newV41EPLifecycleFake()
		f.deps.Load = func(context.Context, V41EPOwnerRequest, *v41EPOwnerLifecycleBackend, *v41EPOwnerLifecycleGroup) (*v41EPOwnerLifecycleLoaded, error) {
			return nil, nil
		}
		observations := 0
		f.deps.LoadRetirement = func() (v41EPOwnerLoadRetirement, error) { observations++; return v41EPOwnerLoadRetirement{}, nil }
		out := runV41EPOwnerLifecycle(context.Background(), f.req, 0, f.deps)
		if observations != 1 || out.Err == nil || out.Receipt.CleanupCompleted != nil || out.Receipt.ArtifactAuthenticated != nil || len(out.Receipt.CleanupFailures) != 0 || f.has("root") {
			t.Fatalf("nil ownership became completed transfer: %+v", out)
		}
	})
	t.Run("observation-skipped-before-Load-and-after-complete-transfer", func(t *testing.T) {
		for _, mode := range []string{"backend-failure", "normal-transfer", "cancel-after-transfer", "phase-expiry-after-transfer"} {
			t.Run(mode, func(t *testing.T) {
				f := newV41EPLifecycleFake()
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				observations := 0
				f.deps.LoadRetirement = func() (v41EPOwnerLoadRetirement, error) {
					observations++
					return v41EPOwnerLoadRetirement{}, errors.New("unexpected observation")
				}
				base := f.deps.Load
				f.deps.Load = func(c context.Context, r V41EPOwnerRequest, b *v41EPOwnerLifecycleBackend, g *v41EPOwnerLifecycleGroup) (*v41EPOwnerLifecycleLoaded, error) {
					loaded, err := base(c, r, b, g)
					if mode == "cancel-after-transfer" {
						cancel()
					} else if mode == "phase-expiry-after-transfer" {
						f.advance(f.req.Budgets.LoadNS)
					}
					return loaded, err
				}
				primary := errors.New("backend refusal")
				if mode == "backend-failure" {
					f.deps.Backend = func(context.Context, V41EPOwnerSelection) (*v41EPOwnerLifecycleBackend, error) { return nil, primary }
				}
				out := runV41EPOwnerLifecycle(ctx, f.req, 0, f.deps)
				if observations != 0 || out.Receipt.CleanupCompleted == nil || !*out.Receipt.CleanupCompleted || len(out.Receipt.CleanupFailures) != 0 {
					t.Fatalf("unneeded observation altered actual releases: mode=%s out=%+v", mode, out)
				}
				if mode == "normal-transfer" {
					if out.Err != nil || out.Receipt.Status != "complete" {
						t.Fatalf("successful transfer changed: %+v", out)
					}
				} else {
					want := error(primary)
					if mode == "cancel-after-transfer" {
						want = context.Canceled
					} else if mode == "phase-expiry-after-transfer" {
						want = context.DeadlineExceeded
					}
					if !errors.Is(out.Err, want) || out.Receipt.Status != "incomplete" || out.Receipt.ArtifactAuthenticated != nil || f.has("root") || f.has("shutdown") {
						t.Fatalf("phase failure became successful execution: %+v", out)
					}
				}
				if mode != "backend-failure" && (f.count("session") != 1 || f.count("model") != 1 || f.count("backend") != 1) {
					t.Fatal("successful bundle transfer lost exactly-once releases")
				}
			})
		}
	})
	t.Run("observer-waits-for-phase-watcher-join", func(t *testing.T) {
		f := newV41EPLifecycleFake()
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		entered, release := make(chan struct{}), make(chan struct{})
		var stop, releaseOnce sync.Once
		var observations atomic.Int32
		f.group.StopIO = func(error) error {
			stop.Do(func() { close(entered); <-release; f.event("stop-returned") })
			return nil
		}
		primary := errors.New("joined factory failure")
		f.deps.Load = func(context.Context, V41EPOwnerRequest, *v41EPOwnerLifecycleBackend, *v41EPOwnerLifecycleGroup) (*v41EPOwnerLifecycleLoaded, error) {
			return nil, primary
		}
		f.deps.LoadRetirement = func() (v41EPOwnerLoadRetirement, error) {
			observations.Add(1)
			if !f.has("stop-returned") {
				return v41EPOwnerLoadRetirement{}, errors.New("observer preceded watcher join")
			}
			return v41EPOwnerLoadRetirement{Available: true, Completed: true}, nil
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
		releaseOnce.Do(func() { close(release) })
		// Always join before any assertion or fixture retirement. The standard
		// test/process deadlines remain the final bound if an invariant breaks.
		out := <-done
		if !enteredOnTime || beforeJoin != 0 || observations.Load() != 1 || !errors.Is(out.Err, primary) || out.Receipt.CleanupCompleted == nil || !*out.Receipt.CleanupCompleted || len(out.Receipt.CleanupFailures) != 0 {
			t.Fatalf("observation did not follow joined stop work: before=%d after=%d out=%+v", beforeJoin, observations.Load(), out)
		}
	})
}
