package gateway

import (
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func batchReceiptRecorder() *httptest.ResponseRecorder {
	return httptest.NewRecorder()
}

// TestNativeBatchReceiptHeadersBuffered proves a valid coordinated receipt
// emits the exact three-header evidence set with cohort values preserved.
func TestNativeBatchReceiptHeadersBuffered(t *testing.T) {
	rr := batchReceiptRecorder()
	applyNativeBatchReceiptHeaders(rr, &agent.InKernelBatchReceipt{CohortID: 4294967297, CohortSize: 4, SharedPanels: 3, SharedMACs: 99})
	if got := rr.Header().Get(HeaderNativeCohortID); got != "4294967297" {
		t.Fatalf("%s=%q, want %q", HeaderNativeCohortID, got, "4294967297")
	}
	if got := rr.Header().Get(HeaderNativeCohortSize); got != "4" {
		t.Fatalf("%s=%q, want %q", HeaderNativeCohortSize, got, "4")
	}
	if got := rr.Header().Get(HeaderNativeSharedPanels); got != "3" {
		t.Fatalf("%s=%q, want %q", HeaderNativeSharedPanels, got, "3")
	}
}

// TestNativeBatchReceiptHeadersZeroSharedPanels proves a zero shared-panel
// count is preserved as an authoritative serial value, not omitted.
func TestNativeBatchReceiptHeadersZeroSharedPanels(t *testing.T) {
	rr := batchReceiptRecorder()
	applyNativeBatchReceiptHeaders(rr, &agent.InKernelBatchReceipt{CohortID: 7, CohortSize: 1, SharedPanels: 0})
	if got := rr.Header().Get(HeaderNativeSharedPanels); got != "0" {
		t.Fatalf("%s=%q, want present %q", HeaderNativeSharedPanels, got, "0")
	}
	if got := rr.Header().Get(HeaderNativeCohortID); got != "7" {
		t.Fatalf("%s=%q, want %q", HeaderNativeCohortID, got, "7")
	}
	if got := rr.Header().Get(HeaderNativeCohortSize); got != "1" {
		t.Fatalf("%s=%q, want %q", HeaderNativeCohortSize, got, "1")
	}
}

// TestNativeBatchReceiptHeadersOmitted proves a nil receipt or a receipt with
// no cohort writes no header at all, so absence stays "unknown".
func TestNativeBatchReceiptHeadersOmitted(t *testing.T) {
	for name, receipt := range map[string]*agent.InKernelBatchReceipt{
		"nil":         nil,
		"zero-cohort": {CohortID: 0, CohortSize: 4, SharedPanels: 2},
	} {
		t.Run(name, func(t *testing.T) {
			rr := batchReceiptRecorder()
			applyNativeBatchReceiptHeaders(rr, receipt)
			for _, h := range []string{HeaderNativeCohortID, HeaderNativeCohortSize, HeaderNativeSharedPanels} {
				if got := rr.Header().Get(h); got != "" {
					t.Fatalf("%s=%q, want absent header", h, got)
				}
			}
		})
	}
	rr := batchReceiptRecorder()
	applyNativeBatchReceiptHeaders(nil, &agent.InKernelBatchReceipt{CohortID: 1, CohortSize: 1})
	for _, h := range []string{HeaderNativeCohortID, HeaderNativeCohortSize, HeaderNativeSharedPanels} {
		if got := rr.Header().Get(h); got != "" {
			t.Fatalf("nil writer %s=%q, want absent header", h, got)
		}
	}
}

// TestNativeBatchReceiptHeadersMalformed proves malformed cohort size or
// shared-panel values refuse the WHOLE set rather than emitting a partial or
// inferred one.
func TestNativeBatchReceiptHeadersMalformed(t *testing.T) {
	for name, receipt := range map[string]*agent.InKernelBatchReceipt{
		"negative-size":   {CohortID: 9, CohortSize: -1, SharedPanels: 2},
		"zero-size":       {CohortID: 9, CohortSize: 0, SharedPanels: 2},
		"negative-shared": {CohortID: 9, CohortSize: 2, SharedPanels: -1},
	} {
		t.Run(name, func(t *testing.T) {
			rr := batchReceiptRecorder()
			applyNativeBatchReceiptHeaders(rr, receipt)
			for _, h := range []string{HeaderNativeCohortID, HeaderNativeCohortSize, HeaderNativeSharedPanels} {
				if got := rr.Header().Get(h); got != "" {
					t.Fatalf("malformed receipt emitted %s=%q, want no header set", h, got)
				}
			}
		})
	}
}

// TestNativeBatchReceiptHeadersStreamRefused proves the header set is never
// emitted when the receipt is unavailable, i.e. the buffered opt-in gate is
// what admits it — a streaming request never carries the native receipt at all.
func TestNativeBatchReceiptHeadersStreamRefused(t *testing.T) {
	srv := nativeReceiptServer(t)
	body := `{"messages":[{"role":"user","content":"x"}],"stream":true,"fak":{"native_inference_receipt":true}}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("streaming native receipt status=%d, want fail-closed 400", rr.Code)
	}
	for _, h := range []string{HeaderNativeCohortID, HeaderNativeCohortSize, HeaderNativeSharedPanels} {
		if got := rr.Header().Get(h); got != "" {
			t.Fatalf("refused streaming request emitted %s=%q", h, got)
		}
	}
}

// TestNativeBatchReceiptHeadersConcurrentIsolation proves two request-local
// header writers never cross-contaminate: each recorder carries only its own
// cohort triple.
func TestNativeBatchReceiptHeadersConcurrentIsolation(t *testing.T) {
	const n = 32
	recorders := make([]*httptest.ResponseRecorder, n)
	done := make(chan struct{}, n)
	for i := 0; i < n; i++ {
		go func(i int) {
			rr := batchReceiptRecorder()
			applyNativeBatchReceiptHeaders(rr, &agent.InKernelBatchReceipt{CohortID: uint64(i + 1), CohortSize: i + 1, SharedPanels: i})
			recorders[i] = rr
			done <- struct{}{}
		}(i)
	}
	for i := 0; i < n; i++ {
		<-done
	}
	for i, rr := range recorders {
		wantID := strconv.Itoa(i + 1)
		if got := rr.Header().Get(HeaderNativeCohortID); got != wantID {
			t.Fatalf("recorder[%d] %s=%q, want %q (cross-contamination)", i, HeaderNativeCohortID, got, wantID)
		}
		if got := rr.Header().Get(HeaderNativeCohortSize); got != wantID {
			t.Fatalf("recorder[%d] %s=%q, want %q", i, HeaderNativeCohortSize, got, wantID)
		}
		if got := rr.Header().Get(HeaderNativeSharedPanels); got != strconv.Itoa(i) {
			t.Fatalf("recorder[%d] %s=%q, want %q", i, HeaderNativeSharedPanels, got, strconv.Itoa(i))
		}
	}
}
