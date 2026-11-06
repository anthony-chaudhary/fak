package agent

import (
	"context"
	"errors"
	"runtime"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=100ms lane=default
func TestOwnedPrefillObservationColdAndWarmBeforeEmission(t *testing.T) {
	p := reusePlanner(true, false, tinyCfg())
	ids := synthIDs(p.m.Cfg.VocabSize, 24, 71)
	// Seed outside the observer, so CPU cold-flight eligibility is irrelevant.
	_, _, _, _, _, _, _, _, err := p.generateReusedContextWithBias(context.Background(), ids, 1, 0, 0, 0, nil, 0, 0, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, warm := range []bool{true, false} {
		if !warm {
			p = reusePlanner(false, false, tinyCfg())
		}
		var events []string
		wantCached := 0
		if warm {
			wantCached = len(ids)
		}
		ctx := p.WithOwnedPrefillObservation(context.Background(), func(prompt, cached int) func(bool) {
			if prompt != len(ids) || cached != wantCached {
				t.Fatalf("counts=%d/%d want=%d/%d", prompt, cached, len(ids), wantCached)
			}
			events = append(events, "start")
			return func(completed bool) {
				if !completed {
					t.Fatal("successful prefill reported aborted")
				}
				events = append(events, "finish")
			}
		})
		ctx = p.WithExecutionDeadlineAdmission(ctx, func(context.Context, int, int, int) error {
			events = append(events, "deadline")
			return nil
		})
		_, _, _, _, _, _, _, _, err = p.generateReusedContextWithBias(ctx, ids, 1, 0, 0, 0, nil, 0, 0, nil, func(int) bool {
			if !slices.Equal(events, []string{"deadline", "start", "finish"}) {
				t.Fatalf("emission before finish: %v", events)
			}
			return false
		})
		if err != nil || !slices.Equal(events, []string{"deadline", "start", "finish"}) {
			t.Fatalf("warm=%v events=%v err=%v", warm, events, err)
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestOwnedPrefillObservationCancellationAndRetry(t *testing.T) {
	p := reusePlanner(false, false, tinyCfg())
	ids := synthIDs(p.m.Cfg.VocabSize, 12, 93)
	var completed []bool
	starts := 0
	base, cancel := context.WithCancel(context.Background())
	ctx := p.WithOwnedPrefillObservation(base, func(prompt, cached int) func(bool) {
		starts++
		if prompt != len(ids) || cached != 0 {
			t.Fatalf("counts=%d/%d", prompt, cached)
		}
		cancel()
		return func(ok bool) { completed = append(completed, ok) }
	})
	_, _, _, _, _, _, _, _, err := p.generateReusedContextWithBias(ctx, ids, 1, 0, 0, 0, nil, 0, 0, nil, nil)
	if !errors.Is(err, context.Canceled) || starts != 1 || !slices.Equal(completed, []bool{false}) {
		t.Fatalf("starts=%d finish=%v err=%v", starts, completed, err)
	}
	ctx = p.WithOwnedPrefillObservation(context.Background(), func(int, int) func(bool) {
		starts++
		return func(ok bool) { completed = append(completed, ok) }
	})
	for i := 0; i < 2; i++ {
		// No text callback: lifecycle must not depend on emitted text.
		_, _, _, _, _, _, _, _, err = p.generateReusedContextWithBias(ctx, ids, 1, 0, 0, 0, nil, 0, 0, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
	}
	if starts != 3 || !slices.Equal(completed, []bool{false, true, true}) {
		t.Fatalf("starts=%d finish=%v", starts, completed)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestOwnedPrefillObservationExactHitRefeed(t *testing.T) {
	p := reusePlanner(true, false, tinyCfg())
	ids := synthIDs(p.m.Cfg.VocabSize, 12, 93)
	sess := p.m.NewSession()
	sess.Prefill(ids)
	defer sess.Close()
	root, _ := p.tree.Lookup(nil)
	leaf := p.tree.Insert(root, ids, sess.Cache)
	p.tree.Done(leaf)
	starts, finishes := 0, 0
	ctx := p.WithOwnedPrefillObservation(context.Background(), func(prompt, cached int) func(bool) {
		starts++
		if prompt != len(ids) || cached != len(ids)-1 {
			t.Fatalf("refeed counts=%d/%d", prompt, cached)
		}
		return func(ok bool) {
			if !ok {
				t.Fatal("refeed failed")
			}
			finishes++
		}
	})
	_, _, _, _, _, _, _, _, err := p.generateReusedContextWithBias(ctx, ids, 1, 0, 0, 0, nil, 0, 0, nil, nil)
	if err != nil || starts != 1 || finishes != 1 {
		t.Fatalf("start/finish=%d/%d err=%v", starts, finishes, err)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestOwnedPrefillObservationDeclinesUnsupported(t *testing.T) {
	p := reusePlanner(true, false, tinyCfg())
	ids := synthIDs(p.m.Cfg.VocabSize, 12, 17)
	ctx := p.WithOwnedPrefillObservation(context.Background(), func(int, int) func(bool) { t.Fatal("unsupported observation"); return nil })
	other := reusePlanner(false, false, tinyCfg())
	if other.beginOwnedPrefillObservation(ctx, 12, 0) != nil {
		t.Fatal("cross-planner observation")
	}
	for _, pair := range [][2]int{{-1, 0}, {12, -1}, {12, 13}} {
		if p.beginOwnedPrefillObservation(ctx, pair[0], pair[1]) != nil {
			t.Fatal("invalid counts observed")
		}
	}
	p.SetSpeculativeEngine(&model.SpeculativeEngine{})
	if p.beginOwnedPrefillObservation(ctx, 12, 0) != nil {
		t.Fatal("speculation observed")
	}
	p.SetSpeculativeEngine(nil)
	// Cold reusable CPU requests are declined for every GOMAXPROCS value,
	// even when flight coalescing is not eligible at the observer seam.
	old := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(old)
	for _, procs := range []int{1, 2} {
		runtime.GOMAXPROCS(procs)
		p = reusePlanner(true, false, tinyCfg())
		ctx = p.WithOwnedPrefillObservation(context.Background(), func(int, int) func(bool) {
			t.Fatalf("cold CPU observation at GOMAXPROCS=%d", procs)
			return nil
		})
		_, _, _, _, _, _, _, _, err := p.generateReusedContextWithBias(ctx, ids, 1, 0, 0, 0, nil, 0, 0, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestOwnedPrefillObservationFinishOnceAndPanicUnwind(t *testing.T) {
	p := reusePlanner(false, false, tinyCfg())
	var completed []bool
	ctx := p.WithOwnedPrefillObservation(context.Background(), func(int, int) func(bool) {
		return func(ok bool) { completed = append(completed, ok) }
	})
	finish := p.beginOwnedPrefillObservation(ctx, 12, 0)
	finish(true)
	finish(false)
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("missing fixture panic")
			}
		}()
		finish := p.beginOwnedPrefillObservation(ctx, 12, 0)
		defer finish(false)
		panic("fixture")
	}()
	if !slices.Equal(completed, []bool{true, false}) {
		t.Fatalf("finish=%v", completed)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestOwnedPrefillObservationRouteExclusions(t *testing.T) {
	for _, vision := range []bool{false, true} {
		for _, populate := range []bool{false, true} {
			for _, cpu := range []bool{false, true} {
				for _, reuse := range []bool{false, true} {
					for _, matched := range []int{0, 12} {
						want := !vision && !populate
						if cpu && reuse && matched == 0 {
							want = false
						}
						if got := ownedPrefillRouteEligible(vision, populate, cpu, reuse, matched); got != want {
							t.Fatalf("vision=%v populate=%v cpu=%v reuse=%v matched=%d: got %v want %v", vision, populate, cpu, reuse, matched, got, want)
						}
					}
				}
			}
		}
	}
}

type ownedPrefillNamedBackend struct {
	compute.Backend
	name string
}

func (b ownedPrefillNamedBackend) Name() string { return b.name }

// fak-test:runtime fast est=1ms lane=default
func TestOwnedPrefillObservationBackendExclusions(t *testing.T) {
	for _, name := range []string{"cuda", " CUDA ", "vulkan", "rocm", "metal", ""} {
		p := &InKernelPlanner{backend: ownedPrefillNamedBackend{name: name}}
		starts := 0
		ctx := p.WithOwnedPrefillObservation(context.Background(), func(int, int) func(bool) { starts++; return nil })
		p.beginOwnedPrefillObservation(ctx, 12, 0)
		want := 0
		if name == "cuda" || name == " CUDA " {
			want = 1
		}
		if starts != want {
			t.Fatalf("backend=%q starts=%d want=%d", name, starts, want)
		}
	}
	for _, p := range []*InKernelPlanner{{metal: true}, {metalMTPCoordinator: &model.MetalMTPCoordinator{}}} {
		ctx := p.WithOwnedPrefillObservation(context.Background(), func(int, int) func(bool) { t.Fatal("unsupported planner observed"); return nil })
		if p.beginOwnedPrefillObservation(ctx, 12, 0) != nil {
			t.Fatal("unsupported planner finish")
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestTryOwnedPrefillObservationPreservesCaller(t *testing.T) {
	p, other := &InKernelPlanner{}, &InKernelPlanner{}
	calls := 0
	ctx := p.WithOwnedPrefillObservation(context.Background(), func(int, int) func(bool) { calls++; return nil })
	for _, target := range []*InKernelPlanner{p, other} {
		got, ok := target.TryWithOwnedPrefillObservation(ctx, func(int, int) func(bool) { t.Fatal("replacement observer called"); return nil })
		if ok || got != ctx {
			t.Fatal("existing observer replaced")
		}
	}
	p.beginOwnedPrefillObservation(ctx, 12, 0)
	if calls != 1 {
		t.Fatal(calls)
	}
	ctx, ok := p.TryWithOwnedPrefillObservation(context.Background(), func(int, int) func(bool) { calls++; return nil })
	if !ok {
		t.Fatal("empty context refused")
	}
	p.beginOwnedPrefillObservation(ctx, 12, 0)
	if calls != 2 {
		t.Fatal(calls)
	}
}
