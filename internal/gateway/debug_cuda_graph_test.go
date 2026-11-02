package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:justify why=contract when=changed:internal/gateway/debug*
// fak-test:runtime fast est=20ms lane=default
func TestDebugVarsCUDAGraphReplays(t *testing.T) {
	original := debugCUDAGraphReplays
	t.Cleanup(func() { debugCUDAGraphReplays = original })
	srv := newTestServer(t)
	for _, count := range []uint64{0, 7} {
		debugCUDAGraphReplays = func() uint64 { return count }
		response := httptest.NewRecorder()
		srv.Handler().ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/debug/vars", nil))
		if response.Code != http.StatusOK {
			t.Fatalf("debug vars status=%d", response.Code)
		}
		var body struct {
			Kernel map[string]json.RawMessage `json:"kernel"`
		}
		if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
			t.Fatal(err)
		}
		raw, present := body.Kernel["cuda_graph_replays"]
		var got uint64
		if !present || string(raw) == "null" || json.Unmarshal(raw, &got) != nil || got != count {
			t.Fatalf("kernel.cuda_graph_replays=%s present=%t want=%d", raw, present, count)
		}
	}
	debugCUDAGraphReplays = original
	if got, want := debugCUDAGraphReplays(), compute.CUDAGraphReplays(); got != want {
		t.Fatalf("production counter=%d want=%d", got, want)
	}
}
