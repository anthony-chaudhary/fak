package gateway

import (
	"errors"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// A V4.1 operation refusal at warmup reads backend_refused, and /healthz names the
// kind, stage and layer it stopped at, so a probe needs no goroutine dump.
// fak-test:runtime fast est=50ms
func TestRunWarmupFailureNamesV41ForwardSite(t *testing.T) {
	refusal := fmt.Errorf("warmup: %w", &model.V41IndexWeightScaleOperationError{
		Layer: 20, Cause: &model.V41ForwardError{Stage: "indexer", Layer: 20, Err: model.ErrV41ForwardStage},
	})
	_, body := warmupHealthz(t, failedWarmupServer(t, refusal))
	failure, _ := body["warmup_failure"].(map[string]any)
	site, _ := failure["forward_site"].(map[string]any)
	if failure["reason"] != "backend_refused" || site["kind"] != "index_weight_scale" || site["stage"] != "indexer" || site["layer"] != float64(20) {
		t.Fatalf("warmup_failure = %v, want backend_refused at index_weight_scale/indexer layer 20", failure)
	}
	_, body = warmupHealthz(t, failedWarmupServer(t, errors.New("boom")))
	if failure, _ := body["warmup_failure"].(map[string]any); failure["forward_site"] != nil {
		t.Fatalf("untyped warmup error reported a forward site: %v", failure)
	}
}
