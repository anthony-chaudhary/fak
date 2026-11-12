package servechain

import (
	"encoding/json"
	"io"
	"net/http"
	"testing"
	"time"

	billingledger "github.com/anthony-chaudhary/fak/pkg/billing/ledger"
	billingmanifest "github.com/anthony-chaudhary/fak/pkg/billing/manifest"
)

// readErrorCode decodes the chain's error envelope and returns its code tag.
func readErrorCode(t *testing.T, resp *http.Response) string {
	t.Helper()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read error body: %v", err)
	}
	var envelope struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		t.Fatalf("decode error body %q: %v", body, err)
	}
	return envelope.Error.Code
}

// TestRateWindowFixedMinute proves the window's own contract: a saturated
// minute refuses, commits fold only into the current minute, and the
// window rolls when the wallclock minute advances.
//
// fak-test:runtime fast est=1s lane=default
func TestRateWindowFixedMinute(t *testing.T) {
	t.Parallel()
	w := newRateWindow(billingmanifest.Limits{RequestsPerMinute: 2})
	t0 := time.Unix(1_700_000_000-1_700_000_000%60, 0) // aligned to a minute boundary

	if w.saturated(t0) {
		t.Fatal("empty window reports saturated")
	}
	w.commitRequest(t0)
	if w.saturated(t0.Add(30 * time.Second)) {
		t.Fatal("one request saturated an rpm=2 window mid-minute")
	}
	w.commitRequest(t0)
	if !w.saturated(t0) {
		t.Fatal("rpm=2 window not saturated after two commits")
	}
	if !w.saturated(t0.Add(59 * time.Second)) {
		t.Fatal("window drained before its minute ended")
	}
	if w.saturated(t0.Add(61 * time.Second)) {
		t.Fatal("window did not roll after the minute advanced")
	}

	tw := newRateWindow(billingmanifest.Limits{TokensPerMinute: 100})
	if tw.saturated(t0) {
		t.Fatal("empty token window reports saturated")
	}
	tw.commitTokens(100, t0)
	if !tw.saturated(t0) {
		t.Fatal("tpm=100 window not saturated after 100 tokens")
	}
	if tw.saturated(t0.Add(61 * time.Second)) {
		t.Fatal("token window did not roll after a minute")
	}
}

// TestServeChainRateLimit429 proves the rate stage: the third request in a
// saturated rpm=2 minute is refused 429 before reservation, and the refused
// request consumed no rate or money.
//
// fak-test:runtime fast est=2s lane=default
func TestServeChainRateLimit429(t *testing.T) {
	t.Parallel()
	f := newFixture(t, billingmanifest.Limits{RequestsPerMinute: 2})
	key := f.keyFor(t, "tenant-a")

	for i := 0; i < 2; i++ {
		resp := f.post(t, "/v1/chat/completions", key, "", `{"model":"model-a"}`)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("request %d: got %d, want 200", i+1, resp.StatusCode)
		}
	}
	resp := f.post(t, "/v1/chat/completions", key, "", `{"model":"model-a"}`)
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("third request: got %d, want 429", resp.StatusCode)
	}
	if code := readErrorCode(t, resp); code != "rate_limited" {
		t.Fatalf("429 code: got %q, want rate_limited", code)
	}

	f.finish()
	reserves := 0
	for _, e := range f.ledger.Entries() {
		if e.Kind == billingledger.KindReserve {
			reserves++
		}
	}
	if reserves != 2 {
		t.Fatalf("ledger carries %d reserves, want 2 (refused request reserved nothing)", reserves)
	}
}

// TestServeChainEmptyBalance402 proves the reserve stage's money gate: a
// tenant with no deposit is refused 402 at reservation and holds nothing.
//
// fak-test:runtime fast est=1s lane=default
func TestServeChainEmptyBalance402(t *testing.T) {
	t.Parallel()
	f := newFixture(t, billingmanifest.Limits{})
	key, _, err := f.keys.Mint("tenant-c")
	if err != nil {
		t.Fatalf("mint tenant-c: %v", err)
	}

	resp := f.post(t, "/v1/chat/completions", key, "", `{"model":"model-a"}`)
	if resp.StatusCode != http.StatusPaymentRequired {
		t.Fatalf("empty balance: got %d, want 402", resp.StatusCode)
	}
	if code := readErrorCode(t, resp); code != "insufficient_balance" {
		t.Fatalf("402 code: got %q, want insufficient_balance", code)
	}

	f.finish()
	for _, e := range f.ledger.Entries() {
		if e.TenantID == "tenant-c" {
			t.Fatalf("refused request left a %s entry for tenant-c", e.Kind)
		}
	}
}

// TestServeChainDailyCap402 proves the deterministic cap path: usage folded
// synchronously at capture breaches the daily ceiling, and the NEXT request
// is refused 402 — no settle-queue race involved.
//
// fak-test:runtime fast est=2s lane=default
func TestServeChainDailyCap402(t *testing.T) {
	t.Parallel()
	f := newFixture(t, billingmanifest.Limits{DailyTokenCap: 15})
	key := f.keyFor(t, "tenant-a")

	resp := f.post(t, "/v1/chat/completions", key, "", `{"model":"model-a"}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("first request: got %d, want 200", resp.StatusCode)
	}
	resp = f.post(t, "/v1/chat/completions", key, "", `{"model":"model-a"}`)
	if resp.StatusCode != http.StatusPaymentRequired {
		t.Fatalf("second request: got %d, want 402", resp.StatusCode)
	}
	if code := readErrorCode(t, resp); code != "cap_reached" {
		t.Fatalf("402 code: got %q, want cap_reached", code)
	}
	if f.upstreamHits.Load() != 1 {
		t.Fatalf("upstream hit %d times, want 1 (the capped request never proxied)", f.upstreamHits.Load())
	}
}
