package gateway

import (
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// A Metal command-buffer stall is a local, retryable resource stall — never an upstream fault —
// so upstreamErrorStatus must surface it as 503 with the distinct code "metal_command_buffer_stalled"
// instead of the generic 502. The error is deliberately %w-WRAPPED here to prove errors.As walks
// the chain (the observation seam may wrap it with the failing operation's context).
func TestUpstreamErrorStatus_MetalCommandBufferStall(t *testing.T) {
	stall := metalgemm.MetalCommandBufferStallError{
		Operation:          "native-attention",
		WaitedMilliseconds: 12000.5,
		LimitMilliseconds:  metalgemm.DefaultCommandBufferWaitLimit,
	}
	wrapped := fmt.Errorf("turnkey inference: %w", stall)

	status, code, msg := upstreamErrorStatus(wrapped)
	if status != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want %d", status, http.StatusServiceUnavailable)
	}
	if code != "metal_command_buffer_stalled" {
		t.Errorf("code = %q, want %q", code, "metal_command_buffer_stalled")
	}
	if !strings.Contains(msg, "native-attention") || !strings.Contains(msg, "12000.5") {
		t.Errorf("msg %q must name the operation and waited ms", msg)
	}
}
