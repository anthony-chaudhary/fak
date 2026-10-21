package agent

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const seedlessRNGTestTokens = 16

func seedlessRNGTestPlanner(t *testing.T, plannerSeed string) *InKernelPlanner {
	t.Helper()
	t.Setenv("FAK_INKERNEL_SEED", plannerSeed)
	m := model.NewSynthetic(tinyConcurrencyConfig())
	m.Quantize()
	p := NewInKernelPlanner(m, loadProbeTok(t), "synthetic-seedless-rng", false, nil, false)
	p.batchDecode = false
	return p
}

func seedlessRNGTestDraw(t *testing.T, p *InKernelPlanner, temperature float64, opts ...SampleOpt) string {
	t.Helper()
	opts = append([]SampleOpt{
		WithTemperature(&temperature),
		WithMaxTokens(seedlessRNGTestTokens),
		WithDecodeTrace(true),
		WithNativeDecodeTokenIDs(true),
	}, opts...)
	comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "Say anything."}}, nil, opts...)
	if err != nil {
		t.Fatal(err)
	}
	if comp.NativeDecodeTokenIDs == nil {
		t.Fatal("completion carries no native decode token IDs")
	}
	return fmt.Sprint(comp.NativeDecodeTokenIDs.TokenIDs)
}

func seedlessRNGTestDistinct(draws []string) int {
	seen := map[string]bool{}
	for _, draw := range draws {
		seen[draw] = true
	}
	return len(seen)
}

// fak-test:runtime fast est=200ms lane=default
func TestInKernelSeedlessSampledRequestsDrawIndependentStreams(t *testing.T) {
	p := seedlessRNGTestPlanner(t, "")
	draws := make([]string, 8)
	for i := range draws {
		draws[i] = seedlessRNGTestDraw(t, p, 1)
	}
	if got := seedlessRNGTestDistinct(draws); got < 2 {
		t.Fatalf("8 seedless T=1 requests produced %d distinct token streams, want >= 2: %v", got, draws)
	}
}

// fak-test:runtime fast est=300ms lane=default
func TestInKernelExplicitPlannerSeedReplaysRequestSequence(t *testing.T) {
	run := func() []string {
		p := seedlessRNGTestPlanner(t, "1234")
		draws := make([]string, 4)
		for i := range draws {
			draws[i] = seedlessRNGTestDraw(t, p, 1)
		}
		return draws
	}
	first, replay := run(), run()
	for i := range first {
		if first[i] != replay[i] {
			t.Fatalf("FAK_INKERNEL_SEED request %d = %s on a fresh planner, want %s", i, replay[i], first[i])
		}
	}
	if got := seedlessRNGTestDistinct(first); got < 2 {
		t.Fatalf("4 requests under FAK_INKERNEL_SEED produced %d distinct token streams, want >= 2: %v", got, first)
	}
}

// fak-test:runtime fast est=300ms lane=default
func TestInKernelRequestSeedReplayIgnoresPlannerSeedPolicy(t *testing.T) {
	seed := int64(-17)
	entropy := seedlessRNGTestPlanner(t, "")
	want := seedlessRNGTestDraw(t, entropy, 1, WithSeed(&seed))
	_ = seedlessRNGTestDraw(t, entropy, 1)
	if got := seedlessRNGTestDraw(t, entropy, 1, WithSeed(&seed)); got != want {
		t.Fatalf("same request seed after a seedless request = %s, want %s", got, want)
	}
	explicit := seedlessRNGTestPlanner(t, "99")
	_ = seedlessRNGTestDraw(t, explicit, 1)
	if got := seedlessRNGTestDraw(t, explicit, 1, WithSeed(&seed)); got != want {
		t.Fatalf("same request seed under FAK_INKERNEL_SEED = %s, want %s", got, want)
	}
}

// fak-test:runtime fast est=450ms lane=default
func TestInKernelGreedyIgnoresSeedlessStreams(t *testing.T) {
	entropy := seedlessRNGTestPlanner(t, "")
	want := seedlessRNGTestDraw(t, entropy, 0)
	for range 2 {
		if got := seedlessRNGTestDraw(t, entropy, 0); got != want {
			t.Fatalf("seedless greedy request = %s, want %s", got, want)
		}
	}
	interleaved := seedlessRNGTestPlanner(t, "1234")
	sampledOnly := seedlessRNGTestPlanner(t, "1234")
	for i, greedyBefore := range []bool{true, false, true} {
		if greedyBefore {
			if got := seedlessRNGTestDraw(t, interleaved, 0); got != want {
				t.Fatalf("greedy request under FAK_INKERNEL_SEED = %s, want %s", got, want)
			}
		}
		if got, want := seedlessRNGTestDraw(t, interleaved, 1), seedlessRNGTestDraw(t, sampledOnly, 1); got != want {
			t.Fatalf("sampled request %d after interleaved greedy requests = %s, want %s", i, got, want)
		}
	}
}

// fak-test:runtime fast est=700ms lane=default
func TestInKernelUnsetPlannerSeedFreshPlannersDrawFreshEntropy(t *testing.T) {
	draws := make([]string, 4)
	for i := range draws {
		p := seedlessRNGTestPlanner(t, "")
		draws[i] = seedlessRNGTestDraw(t, p, 1)
		if n := p.seedRequests.Load(); n != 0 {
			t.Fatalf("unset FAK_INKERNEL_SEED advanced the reproducible request index to %d, want 0", n)
		}
	}
	if got := seedlessRNGTestDistinct(draws); got < 2 {
		t.Fatalf("4 fresh planners with FAK_INKERNEL_SEED unset produced %d distinct first draws, want >= 2: %v", got, draws)
	}
}

func seedlessRNGOOMTestPlanner(t *testing.T, faultAfterEmits int) (*InKernelPlanner, *observerOOMBackend) {
	t.Helper()
	tok := loadProbeTok(t)
	cfg := tinyConcurrencyConfig()
	cfg.VocabSize = tok.Vocab()
	m := model.NewSynthetic(cfg)
	m.Quantize()
	emits := 0
	be := &observerOOMBackend{
		receiptOOMRetryBackend: receiptOOMRetryBackend{Backend: compute.Default()},
		shouldFail:             func() bool { return faultAfterEmits > 0 && emits >= faultAfterEmits },
		fault:                  &compute.DeviceAllocError{Bytes: 4096, Site: "seedless-complete-oom-retry", Class: compute.MemoryScratchpad},
	}
	p := NewInKernelPlanner(m, tok, "synthetic-seedless-oom", false, be, false)
	p.batchDecode = false
	p.cachePrimeEmitHook = func() { emits++ }
	return p, be
}

// fak-test:runtime fast est=250ms lane=default
func TestInKernelSeedlessCompleteOOMRetryReplaysRequestStream(t *testing.T) {
	t.Setenv("FAK_INKERNEL_SEED", "7")
	draw := func(p *InKernelPlanner) string {
		temperature, maxTokens := 1.5, 12
		comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "Say anything."}}, nil,
			WithTemperature(&temperature), WithMaxTokens(maxTokens))
		if err != nil {
			t.Fatal(err)
		}
		return comp.Message.Content
	}
	clean, _ := seedlessRNGOOMTestPlanner(t, 0)
	want, next := draw(clean), draw(clean)
	if want == next {
		t.Fatalf("FAK_INKERNEL_SEED requests 0 and 1 both emitted %q; the retry witness cannot tell their streams apart", want)
	}
	faulted, be := seedlessRNGOOMTestPlanner(t, 3)
	got := draw(faulted)
	if !be.failed {
		t.Fatal("OOM retry fixture did not inject a mid-decode failure")
	}
	if got != want {
		t.Fatalf("Complete OOM retry changed the seedless stream: got %q want %q (request 1 is %q)", got, want, next)
	}
	if n := faulted.seedRequests.Load(); n != 1 {
		t.Fatalf("one OOM-retried request consumed %d reproducible seed indexes, want 1", n)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestInKernelSeedlessStreamReplaysAfterOOMRetry(t *testing.T) {
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	ids := synthIDs(cfg.VocabSize, 8, 9192)
	run := func(seed int64, injectOOM bool) ([]int, bool) {
		measurement := &nativeInferenceMeasurement{startedAt: time.Now()}
		backend := compute.Default()
		var fault *receiptOOMRetryBackend
		if injectOOM {
			fault = &receiptOOMRetryBackend{Backend: backend, measurement: measurement}
			backend = fault
		}
		p := &InKernelPlanner{m: model.NewSynthetic(cfg), modelID: "seedless-oom-retry", backend: backend}
		ctx := withNativeSeedlessSampleSeed(context.Background(), seed)
		var emitted []int
		_, err := p.generateReusedWithOOMRetry(ctx, ids, 6, 5, 0, 0, nil, 0, 0, map[int]bool{}, func(token int) bool {
			emitted = append(emitted, token)
			return false
		}, func() {
			emitted = emitted[:0]
			measurement.reset()
		}, measurement)
		if err != nil {
			t.Fatal(err)
		}
		return emitted, fault != nil && fault.failed
	}
	want, _ := run(4242, false)
	if other, _ := run(4243, false); eqInts(other, want) {
		t.Fatalf("request stream ignored its resolved seed: both seeds emitted %v", want)
	}
	got, faulted := run(4242, true)
	if !faulted {
		t.Fatal("OOM retry fixture did not inject a first-attempt failure")
	}
	if !eqInts(got, want) {
		t.Fatalf("seedless OOM replay changed stream: got %v want %v", got, want)
	}
}
