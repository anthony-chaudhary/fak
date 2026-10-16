package gateway

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

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
