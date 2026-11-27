package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

func TestServeInkling2StatusAndAutotuneCounters(t *testing.T) {
	adm := inkling2Admission{max: 4, gap: time.Millisecond, maxWait: 2 * time.Millisecond, clock: inkling2RealClock{}}
	e := newInkling2LoopBatchEngineWith(
		func(_ context.Context, _ []int, _ int, _ map[int]bool, emit func(int) bool) (agent.Inkling2LoopGeneration, error) {
			emit(1)
			return agent.Inkling2LoopGeneration{Generated: 1, Stopped: true}, nil
		},
		func(reqs []ggufload.Inkling2LoopBatchRequest) ([]ggufload.Inkling2LoopBatchResult, error) {
			res := make([]ggufload.Inkling2LoopBatchResult, len(reqs))
			for i, r := range reqs {
				r.Emit(2)
				res[i] = ggufload.Inkling2LoopBatchResult{Generated: 1, Stopped: true}
			}
			return res, nil
		},
		adm,
		32768,
	)
	e.loneCohort = true
	e.fitsMultiRow = func(positions int) bool { return positions <= 16384 }
	e.chunkStepdowns.Store(2)
	e.lastChunkSize.Store(256)
	go e.run()
	t.Cleanup(func() { close(e.queue) })

	// Trigger a C=1 long-context fallback (> 16384 positions).
	longPrompt := make([]int, 17000)
	if _, err := e.Generate(context.Background(), longPrompt, 64, nil, func(int) bool { return true }); err != nil {
		t.Fatalf("long Generate: %v", err)
	}

	env := ggufload.Inkling2ServeEnvelope{
		MaxCtx:             32768,
		DefaultChunk:       512,
		MinChunk:           128,
		TargetConcurrency:  4,
		KVDtype:            ggufload.Inkling2KVFP8,
		LongCtxC1Threshold: 16384,
	}
	provider := newInkling2ServeStatusProvider("inkling2-test", env, e)

	srv, err := gateway.New(gateway.Config{
		Model:               "inkling2-test",
		ServeStatusProvider: provider,
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	h := srv.Handler()

	req := httptest.NewRequest(http.MethodGet, "/v1/serve/status", nil)
	req.RemoteAddr = "127.0.0.1:12345"
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /v1/serve/status status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}

	var status gateway.ServeStatusSnapshot
	if err := json.Unmarshal(rec.Body.Bytes(), &status); err != nil {
		t.Fatalf("unmarshal /v1/serve/status: %v", err)
	}
	if status.MaxCtx != 32768 || status.KVDtype != "fp8" || status.DefaultChunk != 512 || status.LongCtxC1Threshold != 16384 {
		t.Fatalf("unexpected envelope in /v1/serve/status: %+v", status)
	}
	if status.ChunkStepdownsTotal != 2 || status.C1LongCtxFallbacksTotal != 1 || status.LastCohortWidth != 1 || status.LastChunkSize != 256 {
		t.Fatalf("unexpected autotune counters in /v1/serve/status: %+v", status)
	}

	mreq := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	mreq.RemoteAddr = "127.0.0.1:12345"
	mrec := httptest.NewRecorder()
	h.ServeHTTP(mrec, mreq)
	if mrec.Code != http.StatusOK {
		t.Fatalf("GET /metrics status = %d, want 200", mrec.Code)
	}
	metricsBody := mrec.Body.String()
	for _, want := range []string{
		"fak_inkling2_chunk_stepdowns_total 2",
		"fak_inkling2_c1_long_ctx_fallbacks_total 1",
		"fak_inkling2_last_cohort_width 1",
		"fak_inkling2_last_chunk_size 256",
	} {
		if !strings.Contains(metricsBody, want) {
			t.Errorf("/metrics missing %q\n--- body ---\n%s", want, metricsBody)
		}
	}
}
