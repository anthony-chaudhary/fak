package metrics

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// Six in-memory HTTP cases; runtime estimate is unmeasured.
// fak-test:runtime fast est=1ms lane=default
func TestReadyHandlerRefusesDuringShutdown(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name       string
		target     string
		ready      bool
		accepting  bool
		detection  string
		wantCode   int
		wantStatus string
		wantCalls  int
	}{
		{"ready", "/ready", true, false, "", http.StatusOK, "ready", 0},
		{"starting", "/ready", false, false, "", http.StatusServiceUnavailable, "not_ready", 0},
		{"accepting", "/ready", false, true, "", http.StatusServiceUnavailable, "accepting", 0},
		{"slab warming", "/ready?slab=true", true, false, "warming_up", http.StatusServiceUnavailable, "slab_not_ready", 1},
		{"slab ready", "/ready?slab=true", true, false, "ready", http.StatusOK, "ready", 1},
		{"slab not requested", "/ready?slab=false", true, false, "warming_up", http.StatusOK, "ready", 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			startup := &StartupState{}
			startup.Ready.Store(tc.ready)
			startup.AcceptingConnections.Store(tc.accepting)
			c := NewCollector(startup, nil, nil, time.Time{})
			calls := 0
			c.ShardDetectionFunc = func() []string {
				calls++
				return []string{tc.detection}
			}
			before := httptest.NewRecorder()
			c.ReadyHandler(before, httptest.NewRequest(http.MethodGet, tc.target, nil))
			var body struct {
				Status string `json:"status"`
			}
			if err := json.Unmarshal(before.Body.Bytes(), &body); err != nil {
				t.Fatal(err)
			}
			if before.Code != tc.wantCode || body.Status != tc.wantStatus || calls != tc.wantCalls {
				t.Fatalf("before shutdown: code=%d status=%q calls=%d, want %d %q %d", before.Code, body.Status, calls, tc.wantCode, tc.wantStatus, tc.wantCalls)
			}

			c.SetShuttingDown()
			for i := 0; i < 2; i++ {
				after := httptest.NewRecorder()
				c.ReadyHandler(after, httptest.NewRequest(http.MethodGet, tc.target, nil))
				if after.Code != http.StatusServiceUnavailable || after.Header().Get("Content-Type") != "application/json" || after.Body.String() != "{\"status\":\"shutting_down\"}\n" {
					t.Fatalf("during shutdown: code=%d content-type=%q body=%q", after.Code, after.Header().Get("Content-Type"), after.Body.String())
				}
				if calls != tc.wantCalls {
					t.Fatalf("shutdown invoked slab callback: calls=%d, want %d", calls, tc.wantCalls)
				}
			}
			if startup.Ready.Load() != tc.ready || startup.AcceptingConnections.Load() != tc.accepting {
				t.Fatal("readiness handler changed startup state")
			}
		})
	}
}
