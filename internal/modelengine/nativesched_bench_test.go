package modelengine

import (
	"context"
	"errors"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func BenchmarkEngineContinuousBatching(b *testing.B) {
	cfg := SyntheticConfig()
	cfg.HiddenSize = 192
	cfg.NumLayers = 4
	cfg.NumHeads = 6
	cfg.NumKVHeads = 3
	cfg.HeadDim = 32
	cfg.IntermediateSize = 512
	cfg.VocabSize = 512
	m := model.NewSynthetic(cfg)

	for _, lanes := range []int{1, 2, 4, 8} {
		calls := benchCalls(lanes)
		b.Run("legacy/B"+strconv.Itoa(lanes), func(b *testing.B) {
			benchLegacyLifecycle(b, m, calls)
		})
		b.Run("native/B"+strconv.Itoa(lanes), func(b *testing.B) {
			e := New()
			e.Preload(m)
			benchNativeScheduler(b, e, calls)
		})
	}
}

func benchCalls(n int) []*abi.ToolCall {
	calls := make([]*abi.ToolCall, n)
	for i := range calls {
		calls[i] = inlineCall("bench_tool_"+strconv.Itoa(i), `{"i":`+strconv.Itoa(i)+`}`)
	}
	return calls
}

func benchLegacyLifecycle(b *testing.B, m *model.Model, calls []*abi.ToolCall) {
	ctx := context.Background()
	b.ResetTimer()
	start := time.Now()
	for i := 0; i < b.N; i++ {
		reqs := make([]*legacyBenchRequest, len(calls))
		for j, c := range calls {
			prompt := tokenize(c.Tool, refBytes(ctx, c.Args), m.Cfg.VocabSize)
			reqs[j] = startLegacyBenchRequest(ctx, m, c.Tool, prompt)
		}
		drainLegacyBenchRequests(b, reqs)
		for j, r := range reqs {
			if _, err := r.result(); err != nil {
				b.Fatalf("legacy Result(%d): %v", j, err)
			}
		}
	}
	reportTokensPerSecond(b, len(calls), start)
}

func benchNativeScheduler(b *testing.B, e *Engine, calls []*abi.ToolCall) {
	ctx := context.Background()
	b.ResetTimer()
	start := time.Now()
	for i := 0; i < b.N; i++ {
		reqs := make([]abi.EngineRequest, len(calls))
		for j, c := range calls {
			r, err := e.Admit(ctx, c)
			if err != nil {
				b.Fatalf("Admit(%d): %v", j, err)
			}
			reqs[j] = r
		}
		var wg sync.WaitGroup
		for _, r := range reqs {
			wg.Add(1)
			go func(r abi.EngineRequest) {
				defer wg.Done()
				for range r.Tokens() {
				}
			}(r)
		}
		wg.Wait()
		for j, r := range reqs {
			res, err := r.Result()
			if err != nil {
				b.Fatalf("Result(%d): %v", j, err)
			}
			if res == nil || res.Status != abi.StatusOK {
				b.Fatalf("Result(%d) = %+v, want StatusOK", j, res)
			}
		}
	}
	reportTokensPerSecond(b, len(calls), start)
}

func reportTokensPerSecond(b *testing.B, lanes int, start time.Time) {
	elapsed := time.Since(start)
	if elapsed <= 0 {
		return
	}
	b.ReportMetric(float64(b.N*lanes)/elapsed.Seconds(), "req/s")
	b.ReportMetric(float64(b.N*lanes*genTokens)/elapsed.Seconds(), "tok/s")
}

type legacyBenchRequest struct {
	tokens chan abi.EngineToken
	done   chan struct{}
	res    *abi.Result
	err    error
}

func startLegacyBenchRequest(ctx context.Context, m *model.Model, tool string, prompt []int) *legacyBenchRequest {
	r := &legacyBenchRequest{
		tokens: make(chan abi.EngineToken),
		done:   make(chan struct{}),
	}
	go func() {
		defer close(r.tokens)
		defer close(r.done)
		sess := m.NewSession()
		gen := make([]int, 0, genTokens)
		logits := sess.Prefill(prompt)
		for i := 0; i < genTokens; i++ {
			if err := ctx.Err(); err != nil {
				r.err = err
				return
			}
			next := argmax(logits)
			gen = append(gen, next)
			select {
			case r.tokens <- abi.EngineToken{ID: next}:
			case <-ctx.Done():
				r.err = ctx.Err()
				return
			}
			if sess.M.Cfg.IsEOS(next) {
				break
			}
			logits = sess.Step(next)
		}
		r.res = assembleResult(ctx, tool, len(prompt), gen, nil)
	}()
	return r
}

func drainLegacyBenchRequests(b *testing.B, reqs []*legacyBenchRequest) {
	b.Helper()
	var wg sync.WaitGroup
	for _, r := range reqs {
		wg.Add(1)
		go func(r *legacyBenchRequest) {
			defer wg.Done()
			for range r.tokens {
			}
		}(r)
	}
	wg.Wait()
}

func (r *legacyBenchRequest) result() (*abi.Result, error) {
	<-r.done
	if r.err != nil {
		return nil, r.err
	}
	if r.res == nil {
		return nil, errors.New("legacy request produced nil result")
	}
	return r.res, nil
}
