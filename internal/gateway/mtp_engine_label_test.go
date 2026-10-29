package gateway

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestResidentMTPHeaderNamesExecutingBackend pins the x-fak-speculative wire
// value to the receipt's backend label: a CUDA run reports "mtp-cuda", a Vulkan
// run keeps "mtp-vulkan", and an unknown engine label stamps nothing.
//
// fak-test:runtime fast est=20ms lane=default
func TestResidentMTPHeaderNamesExecutingBackend(t *testing.T) {
	abi.RegisterEngine("test", echoEngine{})
	for _, tc := range []struct {
		name, engine, wantHeader, wantDowngrade string
		used                                    bool
	}{
		{name: "cuda execution", engine: "mtp-cuda", used: true, wantHeader: "mtp-cuda"},
		{name: "vulkan execution", engine: "mtp-vulkan", used: true, wantHeader: "mtp-vulkan"},
		{name: "cuda downgrade", engine: "mtp-cuda", wantDowngrade: "proposal-error"},
		{name: "unknown engine", engine: "mtp-other", used: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv, err := New(Config{EngineID: "test", Model: "qwen38-vulkan"})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			receipt := &agent.VulkanMTPExecution{Engine: tc.engine, Used: tc.used, ProposalRounds: 1, TargetOperations: 1}
			if tc.wantDowngrade != "" {
				receipt.DowngradeReason = tc.wantDowngrade
			}
			srv.planner = &vulkanMTPGatewayPlanner{receipt: receipt}
			payload := `{"model":"qwen38-vulkan","messages":[{"role":"user","content":"hi"}],"temperature":0}`
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(payload))
			req.Header.Set("content-type", "application/json")
			w := httptest.NewRecorder()
			srv.Handler().ServeHTTP(w, req)
			resp := w.Result()
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				body, _ := io.ReadAll(resp.Body)
				t.Fatalf("status = %d, want 200; body=%s", resp.StatusCode, body)
			}
			if got := resp.Header.Get(HeaderSpeculative); got != tc.wantHeader {
				t.Fatalf("%s = %q, want %q", HeaderSpeculative, got, tc.wantHeader)
			}
			if got := resp.Header.Get(HeaderSpeculativeDowngrade); got != tc.wantDowngrade {
				t.Fatalf("%s = %q, want %q", HeaderSpeculativeDowngrade, got, tc.wantDowngrade)
			}
		})
	}
}
