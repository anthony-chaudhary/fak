package agent

import (
	"context"
	"errors"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

type cpuPrefixFlightGenerateResult struct {
	tokens    []int
	cacheable int
	matched   int
	tier      radixkv.SnapshotTier
	err       error
}

func runCPUPrefixFlightGenerate(p *InKernelPlanner, ctx context.Context, ids []int) cpuPrefixFlightGenerateResult {
	var result cpuPrefixFlightGenerateResult
	_, _, result.cacheable, result.matched, result.tier, _, _, _, result.err = p.generateReusedContextWithBias(
		ctx, ids, 4, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool {
			result.tokens = append(result.tokens, id)
			return false
		},
	)
	return result
}

func awaitCPUPrefixFlightCondition(t *testing.T, what string, ok func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !ok() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		runtime.Gosched()
	}
}

func receiveCPUPrefixFlightGenerate(t *testing.T, ch <-chan cpuPrefixFlightGenerateResult) cpuPrefixFlightGenerateResult {
	t.Helper()
	select {
	case result := <-ch:
		return result
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for CPU prefix-flight generation")
		return cpuPrefixFlightGenerateResult{}
	}
}

func cpuPrefixFlightPrompt(common []int, suffixLen, marker int, seed uint64, vocab int) []int {
	suffix := synthIDs(vocab, suffixLen, seed)
	suffix[0] = marker
	return append(append([]int(nil), common...), suffix...)
}

// TestInKernelCPUConcurrentPrefixFlight exercises the normal CPU planner seam:
// authorized cold lookup, transient shared-prefix handoff, independent suffix
// prefill, ordinary radix admission, and independent decode.
func TestInKernelCPUConcurrentPrefixFlight(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	priorProcs := runtime.GOMAXPROCS(1)
	t.Cleanup(func() { runtime.GOMAXPROCS(priorProcs) })

	t.Run("same-tenant siblings preserve output after immediate leader mutation", func(t *testing.T) {
		cfg := tinyCfg()
		cfg.EOSTokenID = -1
		p := reusePlanner(true, false, cfg)
		common := synthIDs(cfg.VocabSize, 128, 12749)
		prompts := [][]int{
			cpuPrefixFlightPrompt(common, 80, 7, 12750, cfg.VocabSize),
			cpuPrefixFlightPrompt(common, 90, 8, 12751, cfg.VocabSize),
			cpuPrefixFlightPrompt(common, 100, 9, 12752, cfg.VocabSize),
		}
		contexts := []context.Context{
			WithPrefixCacheIdentity(context.Background(), "tenant-a", "agent-a"),
			WithPrefixCacheIdentity(context.Background(), "tenant-a", "agent-b"),
			WithPrefixCacheIdentity(context.Background(), "tenant-a", "agent-c"),
		}

		leaderSession := p.m.NewSession()
		leaderEntered := make(chan struct{})
		releaseLeader := make(chan struct{})
		leaderDone := make(chan error, 1)
		var callbacks atomic.Int64
		var cloneBytes atomic.Int64
		go func() {
			_, _, matched, leader, err := p.prefixFlights.CoalesceSharedPrefixNS(
				contexts[0], inKernelPrefixFlightNamespace(contexts[0]), prompts[0], 64,
				func(context.Context) (*model.KVCache, []float32, error) {
					callbacks.Add(1)
					close(leaderEntered)
					<-releaseLeader
					logits := leaderSession.Prefill(prompts[0])
					cloneBytes.Store(leaderSession.Cache.ClonePayloadBytes())
					return leaderSession.Cache, logits, nil
				},
			)
			if err == nil && (!leader || matched != 0) {
				err = errors.New("manual flight did not lead")
			}
			// The leader continues immediately into decode and mutates its live KV.
			// Followers must be cloning the immutable publication, not this cache.
			if err == nil {
				leaderSession.Step(5)
				leaderSession.Cache.Truncate(1)
			}
			leaderDone <- err
		}()
		<-leaderEntered

		results := []chan cpuPrefixFlightGenerateResult{
			make(chan cpuPrefixFlightGenerateResult, 1),
			make(chan cpuPrefixFlightGenerateResult, 1),
		}
		for i := range results {
			idx := i + 1
			go func(out chan<- cpuPrefixFlightGenerateResult) {
				out <- runCPUPrefixFlightGenerate(p, contexts[idx], prompts[idx])
			}(results[i])
		}
		awaitCPUPrefixFlightCondition(t, "both planner followers to join", func() bool {
			return p.prefixFlights.Coalesced() == 2
		})
		close(releaseLeader)
		if err := <-leaderDone; err != nil {
			t.Fatal(err)
		}

		for i := range results {
			got := receiveCPUPrefixFlightGenerate(t, results[i])
			if got.err != nil {
				t.Fatalf("sibling %d: %v", i+1, got.err)
			}
			if got.matched != len(common) {
				t.Fatalf("sibling %d matched=%d, want %d", i+1, got.matched, len(common))
			}
			cold := reusePlanner(false, false, cfg)
			want := runCPUPrefixFlightGenerate(cold, contexts[i+1], prompts[i+1])
			if want.err != nil {
				t.Fatalf("cold sibling %d: %v", i+1, want.err)
			}
			if want.matched != 0 || !eqInts(got.tokens, want.tokens) {
				t.Fatalf("sibling %d flight output=%v matched=%d; cold=%v matched=%d", i+1, got.tokens, got.matched, want.tokens, want.matched)
			}
		}
		if callbacks.Load() != 1 {
			t.Fatalf("leader callbacks=%d, want 1", callbacks.Load())
		}
		perClone := cloneBytes.Load()
		if perClone <= 0 || p.prefixFlights.ClonePayloadBytes() != 3*perClone {
			t.Fatalf("flight clone bytes=%d, want publication plus two follower clones=%d", p.prefixFlights.ClonePayloadBytes(), 3*perClone)
		}
	})

	t.Run("automatic admission bypasses unconstrained CPU", func(t *testing.T) {
		previous := runtime.GOMAXPROCS(2)
		defer runtime.GOMAXPROCS(previous)
		cfg := tinyCfg()
		cfg.EOSTokenID = -1
		p := reusePlanner(true, false, cfg)
		common := synthIDs(cfg.VocabSize, 128, 13500)
		a := cpuPrefixFlightPrompt(common, 80, 7, 13501, cfg.VocabSize)
		b := cpuPrefixFlightPrompt(common, 90, 8, 13502, cfg.VocabSize)
		ctx := context.Background()
		entered, release := make(chan struct{}), make(chan struct{})
		leaderDone := make(chan error, 1)
		go func() {
			_, _, _, _, err := p.prefixFlights.CoalesceSharedPrefixNS(ctx, inKernelPrefixFlightNamespace(ctx), a, 64, func(context.Context) (*model.KVCache, []float32, error) {
				close(entered)
				<-release
				s := p.m.NewSession()
				return s.Cache, s.Prefill(a), nil
			})
			leaderDone <- err
		}()
		<-entered
		got := runCPUPrefixFlightGenerate(p, ctx, b)
		if got.err != nil {
			t.Fatal(got.err)
		}
		cold := runCPUPrefixFlightGenerate(reusePlanner(false, false, cfg), ctx, b)
		if cold.err != nil || !eqInts(got.tokens, cold.tokens) {
			t.Fatalf("unconstrained CPU output=%v err=%v, cold=%v err=%v", got.tokens, got.err, cold.tokens, cold.err)
		}
		if got.matched != 0 || p.prefixFlights.Coalesced() != 0 || p.prefixFlights.ClonePayloadBytes() != 0 {
			t.Fatalf("unconstrained CPU flight matched=%d coalesced=%d clone_bytes=%d, want 0/0/0", got.matched, p.prefixFlights.Coalesced(), p.prefixFlights.ClonePayloadBytes())
		}
		close(release)
		if err := <-leaderDone; err != nil {
			t.Fatal(err)
		}
	})

	t.Run("shorter exact follower refeeds last token", func(t *testing.T) {
		cfg := tinyCfg()
		cfg.EOSTokenID = -1
		p := reusePlanner(true, false, cfg)
		short := synthIDs(cfg.VocabSize, 128, 12800)
		long := cpuPrefixFlightPrompt(short, 80, 7, 12801, cfg.VocabSize)
		ctx := WithPrefixCacheIdentity(context.Background(), "tenant-short", "agent-a")
		leaderSession := p.m.NewSession()
		entered, release := make(chan struct{}), make(chan struct{})
		leaderDone := make(chan error, 1)
		go func() {
			_, _, _, _, err := p.prefixFlights.CoalesceSharedPrefixNS(ctx, inKernelPrefixFlightNamespace(ctx), long, 64, func(context.Context) (*model.KVCache, []float32, error) {
				close(entered)
				<-release
				return leaderSession.Cache, leaderSession.Prefill(long), nil
			})
			if err == nil {
				leaderSession.Step(5)
			}
			leaderDone <- err
		}()
		<-entered
		resultCh := make(chan cpuPrefixFlightGenerateResult, 1)
		go func() { resultCh <- runCPUPrefixFlightGenerate(p, ctx, short) }()
		awaitCPUPrefixFlightCondition(t, "shorter planner follower to join", func() bool { return p.prefixFlights.Coalesced() == 1 })
		close(release)
		if err := <-leaderDone; err != nil {
			t.Fatal(err)
		}
		got := receiveCPUPrefixFlightGenerate(t, resultCh)
		if got.err != nil {
			t.Fatal(got.err)
		}
		cold := runCPUPrefixFlightGenerate(reusePlanner(false, false, cfg), ctx, short)
		if cold.err != nil {
			t.Fatal(cold.err)
		}
		if got.matched != len(short)-1 {
			t.Fatalf("shorter exact follower matched=%d, want last-token refeed depth %d", got.matched, len(short)-1)
		}
		if !eqInts(got.tokens, cold.tokens) {
			t.Fatalf("shorter exact follower output=%v, cold=%v", got.tokens, cold.tokens)
		}
	})

	t.Run("tenant and legacy namespaces do not join", func(t *testing.T) {
		cfg := tinyCfg()
		cfg.EOSTokenID = -1
		p := NewInKernelPlanner(model.NewSynthetic(cfg), nil, "cpu-prefix-flight-scope", false, nil, false)
		p.quant = false
		common := synthIDs(cfg.VocabSize, 128, 13000)
		a := cpuPrefixFlightPrompt(common, 80, 7, 13001, cfg.VocabSize)
		b := cpuPrefixFlightPrompt(common, 90, 8, 13002, cfg.VocabSize)
		ctxA := WithPrefixCacheIdentity(context.Background(), "tenant-a", "agent-a")
		ctxB := WithPrefixCacheIdentity(context.Background(), "tenant-b", "agent-b")
		entered, release := make(chan struct{}), make(chan struct{})
		leaderDone := make(chan error, 1)
		go func() {
			_, _, _, _, err := p.prefixFlights.CoalesceSharedPrefixNS(ctxA, inKernelPrefixFlightNamespace(ctxA), a, 64, func(context.Context) (*model.KVCache, []float32, error) {
				close(entered)
				<-release
				s := p.m.NewSession()
				return s.Cache, s.Prefill(a), nil
			})
			leaderDone <- err
		}()
		<-entered

		for name, ctx := range map[string]context.Context{"tenant-b": ctxB, "legacy": context.Background()} {
			t.Run(name, func(t *testing.T) {
				got := runCPUPrefixFlightGenerate(p, ctx, b)
				if got.err != nil {
					t.Fatal(got.err)
				}
				if got.matched != 0 {
					t.Fatalf("isolated request matched=%d, want 0", got.matched)
				}
			})
		}
		if p.prefixFlights.Coalesced() != 0 {
			t.Fatalf("isolated namespaces coalesced=%d, want 0", p.prefixFlights.Coalesced())
		}
		close(release)
		if err := <-leaderDone; err != nil {
			t.Fatal(err)
		}
	})

	for _, bypass := range []struct {
		name string
		new  func(model.Config) *InKernelPlanner
	}{
		{name: "cache disabled", new: func(cfg model.Config) *InKernelPlanner { return reusePlanner(false, false, cfg) }},
		{name: "device backend", new: func(cfg model.Config) *InKernelPlanner {
			be := &countingBackend{Backend: compute.Default(), deviceMemory: true}
			p := NewInKernelPlanner(model.NewSynthetic(cfg), nil, "cpu-prefix-flight-device-bypass", false, be, false)
			p.quant = false
			return p
		}},
		{name: "recurrent CPU cache", new: func(model.Config) *InKernelPlanner {
			p := NewInKernelPlanner(model.NewSynthetic(tinyHybridCfg()), nil, "cpu-prefix-flight-recurrent-bypass", false, nil, false)
			p.quant = false
			if p.tree == nil {
				t.Fatal("recurrent CPU fixture must keep ordinary persistent prefix reuse enabled")
			}
			return p
		}},
	} {
		t.Run(bypass.name+" bypass", func(t *testing.T) {
			cfg := tinyCfg()
			cfg.EOSTokenID = -1
			p := bypass.new(cfg)
			common := synthIDs(cfg.VocabSize, 128, 14000)
			a := cpuPrefixFlightPrompt(common, 80, 7, 14001, cfg.VocabSize)
			b := cpuPrefixFlightPrompt(common, 90, 8, 14002, cfg.VocabSize)
			entered, release := make(chan struct{}), make(chan struct{})
			leaderDone := make(chan error, 1)
			go func() {
				ctx := context.Background()
				_, _, _, _, err := p.prefixFlights.CoalesceSharedPrefixNS(ctx, inKernelPrefixFlightNamespace(ctx), a, 64, func(context.Context) (*model.KVCache, []float32, error) {
					close(entered)
					<-release
					s := p.m.NewSession()
					return s.Cache, s.Prefill(a), nil
				})
				leaderDone <- err
			}()
			<-entered
			got := runCPUPrefixFlightGenerate(p, context.Background(), b)
			if got.err != nil {
				t.Fatal(got.err)
			}
			if got.matched != 0 || p.prefixFlights.Coalesced() != 0 {
				t.Fatalf("bypass matched=%d coalesced=%d, want 0/0", got.matched, p.prefixFlights.Coalesced())
			}
			close(release)
			if err := <-leaderDone; err != nil {
				t.Fatal(err)
			}
		})
	}
}
