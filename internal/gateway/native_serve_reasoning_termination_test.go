package gateway

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// fak-test:runtime fast est=100ms
func TestNativeReasoningOnlyTerminationFrame(t *testing.T) {
	const private = "private-reasoning-marker"
	err := fmt.Errorf("provider status 401 %s: %w", private, agent.ErrNoFinalAnswer)
	frame := nativeTerminationError(err)
	inner, ok := frame["error"].(map[string]any)
	if !ok || frame["type"] != "error" || inner["type"] != "terminated_run" || inner["message"] != "run terminated" {
		t.Fatalf("error frame = %#v", frame)
	}
	if inner["cause"] != agent.TerminationNoFinalAnswer || inner["evidence"] != "model ended without a final answer" {
		t.Fatalf("closed diagnostic = %#v", inner)
	}
	raw, marshalErr := json.Marshal(frame)
	if marshalErr != nil || strings.Contains(string(raw), private) || strings.Contains(string(raw), "401") {
		t.Fatalf("unsafe serialized frame: %s err=%v", raw, marshalErr)
	}
}
