package gateway

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// pagedPrefixPlanner is a local-planner fake that reports the paged prefix ledger.
type pagedPrefixPlanner struct {
	kvMemoryStatsPlanner
	tel agent.InKernelPagedPrefixTelemetry
	ok  bool
}

func (p pagedPrefixPlanner) PagedPrefixTelemetry() (agent.InKernelPagedPrefixTelemetry, bool) {
	return p.tel, p.ok
}

func samplePagedPrefixTelemetry() agent.InKernelPagedPrefixTelemetry {
	return agent.InKernelPagedPrefixTelemetry{
		Leases: 7, ActiveLeases: 2, SharedPages: 5, AllocatedPages: 9, FailOpen: 1,
		DecodePages: 4, ActiveDecodePages: 3, DecodeFailOpen: 2,
		LivePages: 11, TokensPerPage: 16, PageKVElems: 64, SharedKVElems: 320,
	}
}

// [SW-VERIFIED] The healthz block is a closed contract: field names and values, and it
// never flips readiness.
func TestHealthzPagedPrefixReported(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = pagedPrefixPlanner{ok: true, tel: samplePagedPrefixTelemetry()}
	code, body := residentHealthzBody(t, srv)
	if code != 200 || body["ok"] != true {
		t.Fatalf("paged_prefix must not flip readiness: code=%d body=%v", code, body)
	}
	pp, ok := body["paged_prefix"].(map[string]any)
	if !ok {
		t.Fatalf("paged_prefix missing: %v", body)
	}
	want := map[string]any{
		"leases": float64(7), "active_leases": float64(2), "shared_pages": float64(5),
		"allocated_pages": float64(9), "fail_open": float64(1), "decode_pages": float64(4),
		"active_decode_pages": float64(3), "decode_fail_open": float64(2), "live_pages": float64(11),
		"tokens_per_page": float64(16), "page_kv_elems": float64(64), "shared_kv_elems": float64(320),
	}
	if len(pp) != len(want) {
		t.Errorf("paged_prefix has %d fields, want %d: %v", len(pp), len(want), pp)
	}
	for k, v := range want {
		if pp[k] != v {
			t.Errorf("paged_prefix[%q] = %v, want %v", k, pp[k], v)
		}
	}
}

func TestHealthzPagedPrefixAbsentWithoutLedger(t *testing.T) {
	srv := newTestServer(t)
	if _, body := residentHealthzBody(t, srv); body["paged_prefix"] != nil {
		t.Fatalf("non-reporting planner must omit paged_prefix: %v", body["paged_prefix"])
	}
	srv.planner = pagedPrefixPlanner{ok: false}
	if _, body := residentHealthzBody(t, srv); body["paged_prefix"] != nil {
		t.Fatalf("planner without a ledger must omit paged_prefix: %v", body["paged_prefix"])
	}
	if strings.Contains(srv.renderMetrics(), "fak_gateway_paged_prefix_") {
		t.Fatal("planner without a ledger must emit no paged prefix series")
	}
}

func TestPagedPrefixMetrics(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = pagedPrefixPlanner{ok: true, tel: samplePagedPrefixTelemetry()}
	text := srv.renderMetrics()
	for _, want := range []string{
		`fak_gateway_paged_prefix_leases_total 7`,
		`fak_gateway_paged_prefix_active_leases 2`,
		`fak_gateway_paged_prefix_pages_total{kind="shared"} 5`,
		`fak_gateway_paged_prefix_pages_total{kind="allocated"} 9`,
		`fak_gateway_paged_prefix_pool_full_skips_total{scope="prompt"} 1`,
		`fak_gateway_paged_prefix_pool_full_skips_total{scope="decode"} 2`,
		`fak_gateway_paged_prefix_decode_pages{kind="active"} 3`,
		`fak_gateway_paged_prefix_decode_pages{kind="total"} 4`,
	} {
		if !strings.Contains(text, want+"\n") {
			t.Errorf("missing %q", want)
		}
	}
}
