package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestInkling2ServeStatusAndMetrics(t *testing.T) {
	srv, err := New(Config{
		Model:    "inkling2-test",
		EngineID: "inkernel",
		ServeStatusProvider: func() ServeStatusSnapshot {
			return ServeStatusSnapshot{
				MaxCtx:                  32768,
				KVDtype:                 "fp8",
				DefaultChunk:            512,
				MinChunk:                128,
				TargetConcurrency:       4,
				LongCtxC1Threshold:      16384,
				ChunkStepdownsTotal:     3,
				C1LongCtxFallbacksTotal: 2,
				LastCohortWidth:         4,
				LastChunkSize:           256,
			}
		},
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

	var got ServeStatusSnapshot
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal /v1/serve/status: %v", err)
	}
	if got.MaxCtx != 32768 || got.KVDtype != "fp8" || got.DefaultChunk != 512 || got.LongCtxC1Threshold != 16384 {
		t.Fatalf("unexpected envelope in /v1/serve/status: %+v", got)
	}
	if got.ChunkStepdownsTotal != 3 || got.C1LongCtxFallbacksTotal != 2 || got.LastCohortWidth != 4 || got.LastChunkSize != 256 {
		t.Fatalf("unexpected counters in /v1/serve/status: %+v", got)
	}

	mreq := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	mreq.RemoteAddr = "127.0.0.1:12345"
	mrec := httptest.NewRecorder()
	h.ServeHTTP(mrec, mreq)
	if mrec.Code != http.StatusOK {
		t.Fatalf("GET /metrics status = %d, want 200", mrec.Code)
	}
	body := mrec.Body.String()
	for _, want := range []string{
		"fak_inkling2_chunk_stepdowns_total 3",
		"fak_inkling2_c1_long_ctx_fallbacks_total 2",
		"fak_inkling2_last_cohort_width 4",
		"fak_inkling2_last_chunk_size 256",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("/metrics missing %q\n--- body ---\n%s", want, body)
		}
	}
}
