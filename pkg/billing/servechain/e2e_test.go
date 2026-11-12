package servechain

import (
	"io"
	"net/http"
	"strings"
	"testing"

	billingledger "github.com/anthony-chaudhary/fak/pkg/billing/ledger"
	billingmanifest "github.com/anthony-chaudhary/fak/pkg/billing/manifest"
)

// TestServeChainEndToEndTwoTenants is the item's one end-to-end witness: a
// real HTTP chain over a real upstream serves a billed request for each of
// two tenants, both land in the ledger at their exact units, the tenant
// whose budget is exhausted after settlement is refused 402 on the next
// request, credentials are scrubbed, the model id is rewritten upstream, and
// the ledger's invariants hold.
//
// fak-test:runtime fast est=3s lane=default
func TestServeChainEndToEndTwoTenants(t *testing.T) {
	t.Parallel()
	f := newFixture(t, billingmanifest.Limits{})

	// Tenant A: one billed request through the full pipeline.
	respA := f.post(t, "/v1/chat/completions", f.keyFor(t, "tenant-a"), "sess-a",
		`{"model":"model-a","messages":[{"role":"user","content":"hi"}]}`)
	if respA.StatusCode != http.StatusOK {
		t.Fatalf("tenant-a request: got %d, want 200", respA.StatusCode)
	}
	reqA := respA.Header.Get("X-Fak-Billing-Request-Id")
	if reqA == "" {
		t.Fatal("tenant-a response carries no billing request id")
	}
	bodyA, err := io.ReadAll(respA.Body)
	if err != nil {
		t.Fatalf("read tenant-a body: %v", err)
	}
	if !strings.Contains(string(bodyA), `"content":"ok"`) {
		t.Fatalf("tenant-a body not proxied intact: %q", bodyA)
	}

	// Wire asserts: the upstream saw the rewritten model id, the merged
	// path (no doubled /v1), and none of the caller's credentials.
	if got := f.lastModel.Load(); got != "internal-a" {
		t.Fatalf("upstream saw model %v, want internal-a (served name must be rewritten)", got)
	}
	if got := f.lastPath.Load(); got != "/v1/chat/completions" {
		t.Fatalf("upstream saw path %v, want /v1/chat/completions (base path merged once)", got)
	}
	if got := f.lastAuth.Load(); got != "" {
		t.Fatalf("upstream saw Authorization %q, want none", got)
	}

	// Tenant B: the deposit covers exactly one hold. The first request
	// settles 200 units, leaving 999_800 — below the 1_000_000 hold — so
	// the second must be refused 402 at reservation.
	respB1 := f.post(t, "/v1/chat/completions", f.keyFor(t, "tenant-b"), "",
		`{"model":"model-b","messages":[{"role":"user","content":"hi"}]}`)
	if respB1.StatusCode != http.StatusOK {
		t.Fatalf("tenant-b first request: got %d, want 200", respB1.StatusCode)
	}
	reqB1 := respB1.Header.Get("X-Fak-Billing-Request-Id")

	respB2 := f.post(t, "/v1/chat/completions", f.keyFor(t, "tenant-b"), "",
		`{"model":"model-b","messages":[{"role":"user","content":"hi"}]}`)
	if respB2.StatusCode != http.StatusPaymentRequired {
		t.Fatalf("tenant-b second request: got %d, want 402", respB2.StatusCode)
	}
	if code := readErrorCode(t, respB2); code != "insufficient_balance" {
		t.Fatalf("tenant-b 402 code: got %q, want insufficient_balance", code)
	}
	reqB2 := respB2.Header.Get("X-Fak-Billing-Request-Id")

	f.finish() // drain the settle queue before asserting ledger state

	// Both billed requests settled their exact units.
	resA := reservationFor(t, f, "tenant-a", reqA)
	if resA.Status != billingledger.StatusSettled || resA.ActualUnits != 40 {
		t.Fatalf("tenant-a reservation: %s/%d, want settled/40", resA.Status, resA.ActualUnits)
	}
	resB1 := reservationFor(t, f, "tenant-b", reqB1)
	if resB1.Status != billingledger.StatusSettled || resB1.ActualUnits != 200 {
		t.Fatalf("tenant-b reservation: %s/%d, want settled/200", resB1.Status, resB1.ActualUnits)
	}
	// The refused request reserved nothing.
	if _, err := f.ledger.GetReservationByRequest("tenant-b", reqB2); err == nil {
		t.Fatal("refused tenant-b request left a reservation, want none")
	}

	// Accounts and invariants.
	accB, err := f.ledger.GetAccount("tenant-b")
	if err != nil {
		t.Fatalf("account tenant-b: %v", err)
	}
	if accB.Settled != 200 || accB.Available != depositTenantB-200 || accB.Reserved != 0 {
		t.Fatalf("tenant-b account: settled %d available %d reserved %d, want 200/%d/0",
			accB.Settled, accB.Available, accB.Reserved, depositTenantB-200)
	}
	if err := f.ledger.CheckInvariants(); err != nil {
		t.Fatalf("ledger invariants: %v", err)
	}

	// Chain counters: three requests in, two admitted and settled, one
	// refused at reservation, nothing pending.
	stats := f.chain.StatsSnapshot()
	want := map[string]uint64{"requests": 3, "admitted": 2, "rejected": 1, "settled": 2, "pending": 0, "released": 0, "proxy_errors": 0}
	for k, v := range want {
		if stats[k] != v {
			t.Fatalf("stats[%s]: got %d, want %d (full stats %v)", k, stats[k], v, stats)
		}
	}
}
