package gateway

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=100ms lane=default
func TestGatewayNewValidatesEPFanoutTopology(t *testing.T) {
	t.Run("valid coverage reaches construction", func(t *testing.T) {
		t.Setenv("FAK_EP_FANOUT_ADDRS", "rank-1:8000,rank-2:8000")
		newTestServerWithConfig(t, Config{
			EngineID:            "test",
			Model:               "test-model",
			ExpertParallelRanks: 3,
		})
	})

	t.Run("missing followers fail closed", func(t *testing.T) {
		t.Setenv("FAK_EP_FANOUT_ADDRS", "")
		_, err := New(Config{ExpertParallelRanks: 3})
		var topologyErr *model.EPFanoutTopologyError
		if !errors.As(err, &topologyErr) {
			t.Fatalf("New error = %T %v, want *model.EPFanoutTopologyError", err, err)
		}
		if topologyErr.Reason != model.EPCoverMiscount {
			t.Fatalf("topology reason = %q, want %q", topologyErr.Reason, model.EPCoverMiscount)
		}
	})

	t.Run("existing validation keeps precedence", func(t *testing.T) {
		t.Setenv("FAK_EP_FANOUT_ADDRS", "")
		_, err := New(Config{SpeculativeMode: "dflash2", ExpertParallelRanks: 3})
		if err == nil {
			t.Fatal("New admitted an invalid DFlash2 configuration")
		}
		var topologyErr *model.EPFanoutTopologyError
		if errors.As(err, &topologyErr) {
			t.Fatalf("New returned topology error before DFlash2 validation: %v", err)
		}
	})
}

// fak-test:runtime fast est=100ms lane=default
func TestGatewayNewRefusesDuplicateEPFanoutBeforeContact(t *testing.T) {
	var contacts atomic.Int64
	follower := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		contacts.Add(1)
	}))
	t.Cleanup(follower.Close)
	t.Setenv("FAK_EP_FANOUT_ADDRS", follower.URL+","+follower.URL)

	_, err := New(Config{ExpertParallelRanks: 4})
	var topologyErr *model.EPFanoutTopologyError
	if !errors.As(err, &topologyErr) {
		t.Fatalf("New error = %T %v, want *model.EPFanoutTopologyError", err, err)
	}
	if topologyErr.Reason != model.EPCoverDuplicateRank {
		t.Fatalf("topology reason = %q, want %q", topologyErr.Reason, model.EPCoverDuplicateRank)
	}
	if got := contacts.Load(); got != 0 {
		t.Fatalf("follower contacts = %d, want 0 before gateway construction", got)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestGatewayNewSingleRankFanoutPreflightIsInert(t *testing.T) {
	for _, ranks := range []int{0, 1} {
		t.Run(string(rune('0'+ranks))+" ranks", func(t *testing.T) {
			t.Setenv("FAK_EP_FANOUT_ADDRS", "rank-1:8000,rank-1:8000")
			newTestServerWithConfig(t, Config{
				EngineID:            "test",
				Model:               "test-model",
				ExpertParallelRanks: ranks,
			})
		})
	}
}
