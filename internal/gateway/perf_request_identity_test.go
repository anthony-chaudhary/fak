package gateway

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

// fak-test:runtime fast est=10ms lane=default
func TestRequestObservationConsumedBodyAndUniqueIdentity(t *testing.T) {
	const body = `{"model":"same","messages":[{"role":"user","content":"private"}],"max_tokens":8}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	o := newRequestObservation(r)
	var partial perfledger.Record
	applyRequestObservation(o, &partial)
	if partial.WorkloadSHA256 != "" {
		t.Fatal("unread body claimed identity")
	}
	got, err := io.ReadAll(r.Body)
	if err != nil || !bytes.Equal(got, []byte(body)) {
		t.Fatal("body wrapper changed request")
	}
	var complete perfledger.Record
	applyRequestObservation(o, &complete)
	sum := sha256.Sum256([]byte(body))
	if complete.WorkloadSHA256 != hex.EncodeToString(sum[:]) || complete.WorkloadIdentityMode != "raw_request_body_sha256_v1" {
		t.Fatal("consumed workload identity missing")
	}
	other := newRequestObservation(httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
	if o.requestID == other.requestID || o.requestID == "" {
		t.Fatal("request identities reused")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestRequestObservationJoinsAdmissionAndChunkContext(t *testing.T) {
	s := newTestServer(t)
	s.SetModelLoadProfile(&ModelLoadProfile{TotalSeconds: 2})
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader("{}"))
	o := newRequestObservation(r)
	src := perfSourceFromRequest(r)
	src.observation = o
	r = r.WithContext(withPerfSource(r.Context(), src))
	_, _ = io.ReadAll(r.Body)
	ctx, _, _, _, _ := s.beginServedRequest(httptest.NewRecorder(), r)
	requestID, loadID := enginestep.RequestObservationFromContext(ctx)
	var success, failed perfledger.Record
	src.apply(&success)
	src.apply(&failed)
	if requestID == "" || requestID != success.RequestID || requestID != failed.RequestID || loadID != success.LoadLifecycle.LoadID || success.LoadLifecycle.RequestOrdinal != 1 || !success.LoadLifecycle.InitialAfterLoad {
		t.Fatal("request/load identity was not shared")
	}
	if success.ConfigSHA256 == "" || success.ConfigIdentityMode != "scalar_v1" {
		t.Fatal("scalar configuration identity missing")
	}
	if s.captureModelLoadLifecycle().RequestOrdinal != 2 {
		t.Fatal("admission counted more than once")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestRequestObservationPartialAndFailedReadsRemainUnknown(t *testing.T) {
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader("abcdef"))
	o := newRequestObservation(r)
	_, _ = r.Body.Read(make([]byte, 2))
	var rec perfledger.Record
	applyRequestObservation(o, &rec)
	if rec.WorkloadSHA256 != "" {
		t.Fatal("partial body claimed a workload match")
	}
	o.mu.Lock()
	o.failed = true
	o.mu.Unlock()
	_, _ = io.ReadAll(r.Body)
	applyRequestObservation(o, &rec)
	if rec.WorkloadSHA256 != "" {
		t.Fatal("failed read claimed a workload match")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestRequestObservationHTTPMiddlewareAndReadOnlyScrape(t *testing.T) {
	s := newTestServer(t)
	s.SetModelLoadProfile(&ModelLoadProfile{TotalSeconds: 1})
	var rows []perfledger.Record
	h := s.withMetrics(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			w.WriteHeader(http.StatusOK)
			return
		}
		_, _ = io.Copy(io.Discard, r.Body)
		ctx, _, _, _, _ := s.beginServedRequest(w, r)
		var row perfledger.Record
		perfSourceFrom(ctx).apply(&row)
		rows = append(rows, row)
		w.WriteHeader(http.StatusOK)
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/debug/vars", nil))
	for range 2 {
		r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader("{}"))
		r.Header.Set("X-Trace-Id", "shared-session")
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Header().Get("X-Fak-Request-Id") != rows[len(rows)-1].RequestID {
			t.Fatal("response identity differs from row")
		}
	}
	if rows[0].RequestID == rows[1].RequestID || rows[0].TraceID != rows[1].TraceID || rows[0].LoadLifecycle.RequestOrdinal != 1 || rows[1].LoadLifecycle.RequestOrdinal != 2 || rows[0].WorkloadSHA256 != rows[1].WorkloadSHA256 {
		t.Fatal("HTTP request/load/workload join failed")
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestRequestObservationPreservesPartialPhasesOnFailure(t *testing.T) {
	s := newTestServer(t)
	s.SetModelLoadProfile(&ModelLoadProfile{TotalSeconds: 2})
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader("{}"))
	src := perfSourceFromRequest(r)
	src.observation = newRequestObservation(r)
	r = r.WithContext(withPerfSource(r.Context(), src))
	ctx, _, _, _, _ := s.beginServedRequest(httptest.NewRecorder(), r)
	var unreached perfledger.Record
	src.apply(&unreached)
	if unreached.PartialNativePhases != nil {
		t.Fatal("unreached phases must remain unknown")
	}
	recorder := enginestep.New(8)
	recorder.ObservePhaseContext(ctx, enginestep.PhasePrefill, 2*time.Millisecond)
	recorder.ObservePhaseContext(ctx, enginestep.PhasePrefill, time.Millisecond)
	recorder.ObservePhaseContext(ctx, enginestep.PhaseDecode, 7*time.Millisecond)
	s.recordFailedTurn(ctx, localitySelfHosted, context.DeadlineExceeded, time.Now().Add(-time.Second), 0, false)
	rows, _, _ := s.metrics.perfRecordsSnapshot()
	if len(rows) != 1 {
		t.Fatal("failed request must retain one performance row")
	}
	failed := rows[0]
	p := failed.PartialNativePhases
	if failed.Error != perfledger.ErrorDeadline || failed.NativeTiming != nil || p == nil || p.Scope != "request_phase_elapsed" || p.PrefillMS == nil || *p.PrefillMS != 3 || p.PrefillObservations != 2 || p.DecodeMS == nil || *p.DecodeMS != 7 || p.DecodeObservations != 1 {
		t.Fatal("failed generation lost measured partial phase evidence")
	}
	if failed.RequestID != unreached.RequestID || failed.LoadLifecycle.LoadID != unreached.LoadLifecycle.LoadID {
		t.Fatal("failed row changed request or load identity")
	}
	complete := &perfledger.NativeTiming{PrefillMS: 12, DecodeMS: 30}
	success := perfledger.Record{NativeTiming: complete}
	src.apply(&success)
	if success.NativeTiming != complete || success.PartialNativePhases == nil || *success.PartialNativePhases.DecodeMS != 7 {
		t.Fatal("partial observations overwrote completed timing")
	}
	other := enginestep.WithRequestObservation(context.Background(), "other", "load")
	recorder.ObservePhaseContext(other, enginestep.PhaseDecode, time.Second)
	if got := requestPartialNativePhases(ctx); got == nil || *got.DecodeMS != 7 {
		t.Fatal("a different request contaminated partial observations")
	}
	raw, err := json.Marshal(failed)
	if err != nil {
		t.Fatal(err)
	}
	var durable perfledger.Record
	if err := json.Unmarshal(raw, &durable); err != nil || durable.PartialNativePhases == nil || *durable.PartialNativePhases.DecodeMS != 7 {
		t.Fatal("durable row lost partial phase observations")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestRequestObservationPartialPhaseUnknownAndMeasuredZero(t *testing.T) {
	if got := requestPartialNativePhases(nil); got != nil {
		t.Fatal("missing request context must remain unknown")
	}
	ctx := enginestep.WithRequestObservation(context.Background(), "known-zero", "")
	enginestep.New(1).ObservePhaseContext(ctx, enginestep.PhaseDecode, 0)
	got := requestPartialNativePhases(ctx)
	if got == nil || got.DecodeMS == nil || *got.DecodeMS != 0 || got.DecodeObservations != 1 || got.PrefillMS != nil {
		t.Fatal("observed zero and unavailable phase were conflated")
	}
	raw, err := json.Marshal(got)
	if err != nil || !strings.Contains(string(raw), `"decode_ms":0`) || strings.Contains(string(raw), `"prefill_ms"`) {
		t.Fatal("wire representation lost measured zero or fabricated prefill")
	}
}
