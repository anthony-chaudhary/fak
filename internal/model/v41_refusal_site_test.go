package model

import (
	"errors"
	"fmt"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestV41RefusalSiteOf(t *testing.T) {
	stage := v41StageErr(v41StageAttention, 2, ErrV41ForwardStage)
	for _, c := range []struct {
		err  error
		want V41RefusalSite
		ok   bool
	}{
		{stage, V41RefusalSite{Kind: "stage", Stage: "attention", Layer: 2}, true},
		{fmt.Errorf("prefill: %w", v41IndexerScoreOperationErr(20, ErrV41ForwardStage)), V41RefusalSite{Kind: "indexer_score", Stage: "indexer", Layer: 20}, true},
		{&BackendForwardOperationError{Stage: "selection", Layer: -1, Cause: stage}, V41RefusalSite{Kind: "backend_forward", Stage: "selection", Layer: 2}, true},
		{&V41PrefillLatencyError{}, V41RefusalSite{Kind: "prefill_latency", Layer: -1}, true},
		{errors.New(stage.Error()), V41RefusalSite{Layer: -1}, false},
	} {
		got, ok := V41RefusalSiteOf(c.err)
		if got != c.want || ok != c.ok {
			t.Errorf("V41RefusalSiteOf(%T) = %+v, %v; want %+v, %v", c.err, got, ok, c.want, c.ok)
		}
	}
}
