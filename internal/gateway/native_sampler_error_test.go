package gateway

import (
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// fak-test:runtime fast est=1ms lane=default
func TestNativeSamplerNonFiniteLogitsClassification(t *testing.T) {
	status, code, message := upstreamErrorStatus(fmt.Errorf("lane detail: %w", agent.ErrNonFiniteLogits))
	if status != http.StatusInternalServerError || code != "native_sampler_nonfinite_logits" {
		t.Fatalf("classification %d/%s, want 500/native_sampler_nonfinite_logits", status, code)
	}
	if message == "" || strings.Contains(message, "lane detail") {
		t.Fatalf("unsafe error detail: %q", message)
	}
}
