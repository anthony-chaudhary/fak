package servechain

import (
	"net/http"
	"testing"

	billingledger "github.com/anthony-chaudhary/fak/pkg/billing/ledger"
	billingmanifest "github.com/anthony-chaudhary/fak/pkg/billing/manifest"
)

// reservationFor fetches the reservation the chain made for one response's
// billing request id.
func reservationFor(t *testing.T, f *fixture, tenant, reqID string) *billingledger.Reservation {
	t.Helper()
	res, err := f.ledger.GetReservationByRequest(tenant, reqID)
	if err != nil {
		t.Fatalf("reservation for %s/%s: %v", tenant, reqID, err)
	}
	return res
}

// TestServeChainSettlesExactUnits proves the settlement arithmetic end to
// end: model-a's (10 fresh, 5 output) usage at 2.0/4.0 USD per million
// tokens settles 40 micro-USD units, model-b's (100, 50) at 1.0/2.0 settles
// 200, and the accounts and invariants hold after the settle queue drains.
//
// fak-test:runtime fast est=2s lane=default
func TestServeChainSettlesExactUnits(t *testing.T) {
	t.Parallel()
	f := newFixture(t, billingmanifest.Limits{})

	respA := f.post(t, "/v1/chat/completions", f.keyFor(t, "tenant-a"), "", `{"model":"model-a"}`)
	if respA.StatusCode != http.StatusOK {
		t.Fatalf("model-a: got %d, want 200", respA.StatusCode)
	}
	reqA := respA.Header.Get("X-Fak-Billing-Request-Id")
	respB := f.post(t, "/v1/completions", f.keyFor(t, "tenant-b"), "", `{"model":"model-b"}`)
	if respB.StatusCode != http.StatusOK {
		t.Fatalf("model-b: got %d, want 200", respB.StatusCode)
	}
	reqB := respB.Header.Get("X-Fak-Billing-Request-Id")

	f.finish() // drain the settle queue before asserting ledger state

	resA := reservationFor(t, f, "tenant-a", reqA)
	if resA.Status != billingledger.StatusSettled || resA.ActualUnits != 40 {
		t.Fatalf("model-a reservation: status %s actual %d, want settled 40", resA.Status, resA.ActualUnits)
	}
	resB := reservationFor(t, f, "tenant-b", reqB)
	if resB.Status != billingledger.StatusSettled || resB.ActualUnits != 200 {
		t.Fatalf("model-b reservation: status %s actual %d, want settled 200", resB.Status, resB.ActualUnits)
	}
	accA, err := f.ledger.GetAccount("tenant-a")
	if err != nil {
		t.Fatalf("account tenant-a: %v", err)
	}
	if accA.Settled != 40 {
		t.Fatalf("tenant-a settled %d, want 40", accA.Settled)
	}
	if err := f.ledger.CheckInvariants(); err != nil {
		t.Fatalf("ledger invariants: %v", err)
	}
}

// TestServeChainUnpricedModelStaysPending proves the unknown rule: a served
// model the price book does not price settles nothing — its reservation
// stays PENDING (never zero) — while the response itself is served.
//
// fak-test:runtime fast est=2s lane=default
func TestServeChainUnpricedModelStaysPending(t *testing.T) {
	t.Parallel()
	f := newFixture(t, billingmanifest.Limits{})

	resp := f.post(t, "/v1/chat/completions", f.keyFor(t, "tenant-a"), "", `{"model":"model-c"}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("model-c: got %d, want 200", resp.StatusCode)
	}
	reqID := resp.Header.Get("X-Fak-Billing-Request-Id")

	f.finish()

	res := reservationFor(t, f, "tenant-a", reqID)
	if res.Status != billingledger.StatusPending {
		t.Fatalf("unpriced model reservation: status %s, want pending", res.Status)
	}
	stats := f.chain.StatsSnapshot()
	if stats["pending"] != 1 || stats["settled"] != 0 {
		t.Fatalf("stats pending/settled: %d/%d, want 1/0", stats["pending"], stats["settled"])
	}
	if err := f.ledger.CheckInvariants(); err != nil {
		t.Fatalf("ledger invariants: %v", err)
	}
}

// TestServeChainUpstreamFailureReleases proves the transport-failure path:
// an unreachable upstream yields 502, the hold is released in full, and the
// account is left exactly as it started.
//
// fak-test:runtime fast est=2s lane=default
func TestServeChainUpstreamFailureReleases(t *testing.T) {
	t.Parallel()
	f := newClosedUpstreamFixture(t, billingmanifest.Limits{})

	resp := f.post(t, "/v1/chat/completions", f.keyFor(t, "tenant-a"), "", `{"model":"model-a"}`)
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("unreachable upstream: got %d, want 502", resp.StatusCode)
	}
	reqID := resp.Header.Get("X-Fak-Billing-Request-Id")

	f.finish()

	res := reservationFor(t, f, "tenant-a", reqID)
	if res.Status != billingledger.StatusReleased {
		t.Fatalf("failed-upstream reservation: status %s, want released", res.Status)
	}
	acc, err := f.ledger.GetAccount("tenant-a")
	if err != nil {
		t.Fatalf("account: %v", err)
	}
	if acc.Available != depositTenantA || acc.Reserved != 0 || acc.Settled != 0 {
		t.Fatalf("account after release: available %d reserved %d settled %d, want %d/0/0",
			acc.Available, acc.Reserved, acc.Settled, depositTenantA)
	}
	stats := f.chain.StatsSnapshot()
	if stats["proxy_errors"] != 1 || stats["released"] != 1 {
		t.Fatalf("stats proxy_errors/released: %d/%d, want 1/1", stats["proxy_errors"], stats["released"])
	}
}

// TestServeChainStreamedResponseStaysPending proves the documented streaming
// limitation: a non-JSON (SSE) upstream response is served and proxied, but
// its usage is not captured, so the reservation stays PENDING.
//
// fak-test:runtime fast est=2s lane=default
func TestServeChainStreamedResponseStaysPending(t *testing.T) {
	t.Parallel()
	f := newFixture(t, billingmanifest.Limits{})
	f.upstreamMu.Lock()
	f.upstreamCT = "text/event-stream"
	f.upstreamBody = "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n"
	f.upstreamMu.Unlock()

	resp := f.post(t, "/v1/chat/completions", f.keyFor(t, "tenant-a"), "", `{"model":"model-a"}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("streamed request: got %d, want 200", resp.StatusCode)
	}
	reqID := resp.Header.Get("X-Fak-Billing-Request-Id")

	f.finish()

	res := reservationFor(t, f, "tenant-a", reqID)
	if res.Status != billingledger.StatusPending {
		t.Fatalf("streamed reservation: status %s, want pending", res.Status)
	}
	stats := f.chain.StatsSnapshot()
	if stats["pending"] != 1 {
		t.Fatalf("stats pending: %d, want 1", stats["pending"])
	}
}
