package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kvbudget"
)

func init() {
	specPathFor["/v1/fak/capacity/tiers"] = "/v1/fak/capacity/tiers"
	epFanoutExemptRoutes["/v1/fak/capacity/tiers"] = epExemptNoDecode
}

// fak-test:runtime fast est=10ms lane=default
func TestTierCapabilitiesEndpointAndHeaders(t *testing.T) {
	srv := &Server{}

	// 1. Unconfigured tier pool returns {"enabled": false} with 200 OK.
	reqDisabled := httptest.NewRequest(http.MethodGet, "/v1/fak/capacity/tiers", nil)
	recDisabled := httptest.NewRecorder()
	srv.handleFakCapacityTiers(recDisabled, reqDisabled)
	if recDisabled.Code != http.StatusOK {
		t.Fatalf("disabled status = %d, want 200", recDisabled.Code)
	}
	var disabledBody struct {
		Enabled bool `json:"enabled"`
	}
	if err := json.Unmarshal(recDisabled.Body.Bytes(), &disabledBody); err != nil {
		t.Fatalf("unmarshal disabled body: %v", err)
	}
	if disabledBody.Enabled {
		t.Fatalf("disabledBody.Enabled = true, want false")
	}

	// 2. Attach 384k tier pool and acquire 1 master slot.
	plan, err := kvbudget.AutotuneTiersFromTokenBudget(384*1024, kvbudget.ProfileUltracode)
	if err != nil {
		t.Fatalf("AutotuneTiersFromTokenBudget: %v", err)
	}
	pool, err := kvbudget.NewTierPool(plan.TotalTokenCapacity, plan.Specs)
	if err != nil {
		t.Fatalf("NewTierPool: %v", err)
	}
	ctl := NewAdmissionController(DefaultAdmissionPolicy())
	ctl.SetTierPool(pool)
	srv.SetAdmissionController(ctl)

	v, lease, _ := ctl.OfferTiered(SeqRequest{TraceID: "m1", Tokens: 131072})
	if v != VerdictAdmitted {
		t.Fatalf("OfferTiered(m1) = %v, want admitted", v)
	}
	defer ctl.CompleteTiered(lease)

	reqEnabled := httptest.NewRequest(http.MethodGet, "/v1/fak/capacity/tiers", nil)
	recEnabled := httptest.NewRecorder()
	srv.handleFakCapacityTiers(recEnabled, reqEnabled)
	if recEnabled.Code != http.StatusOK {
		t.Fatalf("enabled status = %d, want 200", recEnabled.Code)
	}
	var snap kvbudget.TierPoolSnapshot
	if err := json.Unmarshal(recEnabled.Body.Bytes(), &snap); err != nil {
		t.Fatalf("unmarshal enabled body: %v", err)
	}
	if !snap.Enabled || snap.TotalTokens != 384*1024 || snap.FreeTokens != 256*1024 {
		t.Fatalf("unexpected snapshot: %+v", snap)
	}
	if len(snap.Tiers) != 3 {
		t.Fatalf("len(snap.Tiers) = %d, want 3", len(snap.Tiers))
	}

	// 3. Verify WriteTierHeaders and FormatAvailableTiersHeader (includes elastic borrow headroom).
	recHdr := httptest.NewRecorder()
	WriteTierHeaders(recHdr, snap, kvbudget.TierWorker, 32768)
	if got := recHdr.Header().Get(HeaderContextTier); got != kvbudget.TierWorker {
		t.Fatalf("%s = %q, want %q", HeaderContextTier, got, kvbudget.TierWorker)
	}
	if got := recHdr.Header().Get(HeaderTierCapTokens); got != "32768" {
		t.Fatalf("%s = %q, want 32768", HeaderTierCapTokens, got)
	}
	wantElasticAvail := "master=2@131072,worker=8@32768,micro=64@4096"
	if got := recHdr.Header().Get(HeaderAvailTiers); got != wantElasticAvail {
		t.Fatalf("%s = %q, want %q", HeaderAvailTiers, got, wantElasticAvail)
	}

	// 4. Occupy 3 worker + 10 micro slots so 120k tokens remain (< 128k master cap),
	// then verify WriteTierShedResponse (429 + X-Fak-Available-Tiers + advice JSON).
	for i := 0; i < 3; i++ {
		wl, wv := pool.Acquire(kvbudget.TierWorker, 32768)
		if !wv.Admitted {
			t.Fatalf("setup worker-%d not admitted: %+v", i, wv)
		}
		defer pool.Release(wl)
	}
	for i := 0; i < 10; i++ {
		ml, mv := pool.Acquire(kvbudget.TierMicro, 4096)
		if !mv.Admitted {
			t.Fatalf("setup micro-%d not admitted: %+v", i, mv)
		}
		defer pool.Release(ml)
	}
	_, deniedVerdict := pool.Acquire(kvbudget.TierMaster, 131072)
	if deniedVerdict.Admitted {
		t.Fatalf("expected master acquire denial with 120k free tokens")
	}
	advice := BuildTierAdvice(deniedVerdict, 131072)
	recShed := httptest.NewRecorder()
	WriteTierShedResponse(recShed, advice)
	if recShed.Code != http.StatusTooManyRequests {
		t.Fatalf("shed status = %d, want 429", recShed.Code)
	}
	wantShedAvail := "master=0@131072,worker=3@32768,micro=30@4096"
	if got := recShed.Header().Get(HeaderAvailTiers); got != wantShedAvail {
		t.Fatalf("shed %s = %q, want %q", HeaderAvailTiers, got, wantShedAvail)
	}
	var shedBody struct {
		Error   string     `json:"error"`
		Message string     `json:"message"`
		Advice  TierAdvice `json:"advice"`
	}
	if err := json.Unmarshal(recShed.Body.Bytes(), &shedBody); err != nil {
		t.Fatalf("unmarshal shed body: %v", err)
	}
	if shedBody.Advice.Code != "DOWNGRADE_TO_WORKER" {
		t.Fatalf("shed advice.Code = %q, want DOWNGRADE_TO_WORKER", shedBody.Advice.Code)
	}
}
