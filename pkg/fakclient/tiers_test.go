package fakclient_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kvbudget"
	"github.com/anthony-chaudhary/fak/pkg/fakclient"
)

// fak-test:runtime fast est=10ms lane=default
func TestCapacityTiersRoundTrip(t *testing.T) {
	var gotMethod, gotPath, gotAuth, gotPrincipal string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		gotPrincipal = r.Header.Get("X-Fak-Principal")
		writeJSON(w, `{
			"enabled": true,
			"total_tokens": 393216,
			"free_tokens": 262144,
			"tiers": [
				{"name":"master","level":1,"max_context":131072,"min_slots":1,"active_slots":1,"available_now":0,"elastic_available":0,"preemptible":false},
				{"name":"worker","level":2,"max_context":32768,"min_slots":4,"active_slots":0,"available_now":4,"elastic_available":0,"preemptible":false},
				{"name":"micro","level":3,"max_context":4096,"min_slots":32,"active_slots":0,"available_now":32,"elastic_available":0,"preemptible":true}
			],
			"recommended_fanout": {
				"master_max_ctx": 131072,
				"worker_slots": 4,
				"worker_max_ctx": 32768,
				"micro_slots": 32,
				"micro_max_ctx": 4096
			}
		}`)
	}))
	t.Cleanup(ts.Close)

	c := fakclient.New(ts.URL, fakclient.WithAPIKey("tok"), fakclient.WithPrincipal("agent-1"))
	snap, err := c.CapacityTiers(context.Background())
	if err != nil {
		t.Fatalf("CapacityTiers returned unexpected error: %v", err)
	}
	if gotMethod != http.MethodGet || gotPath != "/v1/fak/capacity/tiers" {
		t.Fatalf("unexpected request: %s %s", gotMethod, gotPath)
	}
	if gotAuth != "Bearer tok" || gotPrincipal != "agent-1" {
		t.Fatalf("unexpected headers: auth=%q principal=%q", gotAuth, gotPrincipal)
	}
	if !snap.Enabled || snap.TotalTokens != 393216 || snap.FreeTokens != 262144 {
		t.Fatalf("unexpected top-level snapshot: %+v", snap)
	}
	if len(snap.Tiers) != 3 {
		t.Fatalf("expected 3 tiers, got %d", len(snap.Tiers))
	}
	if snap.Tiers[0].Name != "master" || snap.Tiers[0].AvailableNow != 0 || snap.Tiers[0].MaxContext != 131072 {
		t.Fatalf("unexpected master tier: %+v", snap.Tiers[0])
	}
	if snap.Tiers[2].Name != "micro" || !snap.Tiers[2].Preemptible || snap.Tiers[2].AvailableNow != 32 {
		t.Fatalf("unexpected micro tier: %+v", snap.Tiers[2])
	}
	wantFanout := fakclient.RecommendedFanout{
		MasterMaxCtx: 131072,
		WorkerSlots:  4,
		WorkerMaxCtx: 32768,
		MicroSlots:   32,
		MicroMaxCtx:  4096,
	}
	if snap.RecommendedFanout != wantFanout {
		t.Fatalf("RecommendedFanout = %+v, want %+v", snap.RecommendedFanout, wantFanout)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCapacityTiersAPIError(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":{"message":"tier pool disabled","type":"unavailable"}}`))
	}))
	t.Cleanup(ts.Close)

	c := fakclient.New(ts.URL)
	_, err := c.CapacityTiers(context.Background())
	if err == nil {
		t.Fatal("expected error on 503 response, got nil")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestParseAvailableTiersHeader(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want []fakclient.TierStatus
	}{
		{
			name: "aegis inkling canonical",
			in:   "master=0@131072,worker=4@32768,micro=32@4096",
			want: []fakclient.TierStatus{
				{Name: "master", AvailableNow: 0, MaxContext: 131072},
				{Name: "worker", AvailableNow: 4, MaxContext: 32768},
				{Name: "micro", AvailableNow: 32, MaxContext: 4096},
			},
		},
		{
			name: "whitespace and trailing comma",
			in:   " master = 1 @ 131072 , worker = 2 @ 32768 , ",
			want: []fakclient.TierStatus{
				{Name: "master", AvailableNow: 1, MaxContext: 131072},
				{Name: "worker", AvailableNow: 2, MaxContext: 32768},
			},
		},
		{
			name: "skips malformed segments",
			in:   "bad,noat=12,neg=-1@4096,zero=1@0,micro=8@4096",
			want: []fakclient.TierStatus{
				{Name: "micro", AvailableNow: 8, MaxContext: 4096},
			},
		},
		{
			name: "empty",
			in:   "   ",
			want: nil,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := fakclient.ParseAvailableTiersHeader(tc.in)
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("ParseAvailableTiersHeader(%q) = %+v, want %+v", tc.in, got, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCapacityTiersSnapshotParity(t *testing.T) {
	server := kvbudget.TierPoolSnapshot{
		Enabled:     true,
		TotalTokens: 393216,
		FreeTokens:  262144,
		Tiers: []kvbudget.TierStatus{
			{
				Name:             kvbudget.TierMaster,
				Level:            1,
				MaxContext:       131072,
				MinSlots:         1,
				ActiveSlots:      1,
				AvailableNow:     0,
				ElasticAvailable: 0,
				Preemptible:      false,
			},
			{
				Name:             kvbudget.TierMicro,
				Level:            3,
				MaxContext:       4096,
				MinSlots:         32,
				ActiveSlots:      4,
				AvailableNow:     28,
				ElasticAvailable: 1,
				Preemptible:      true,
			},
		},
		RecommendedFanout: kvbudget.FanoutRecommendation{
			MasterMaxCtx: 131072,
			WorkerSlots:  4,
			WorkerMaxCtx: 32768,
			MicroSlots:   28,
			MicroMaxCtx:  4096,
		},
	}
	serverJSON, err := json.Marshal(server)
	if err != nil {
		t.Fatalf("marshal server snapshot: %v", err)
	}
	var sdk fakclient.CapacityTiersSnapshot
	if err := json.Unmarshal(serverJSON, &sdk); err != nil {
		t.Fatalf("unmarshal into SDK CapacityTiersSnapshot: %v", err)
	}
	sdkJSON, err := json.Marshal(sdk)
	if err != nil {
		t.Fatalf("re-marshal SDK snapshot: %v", err)
	}
	if string(serverJSON) != string(sdkJSON) {
		t.Fatalf("SDK CapacityTiersSnapshot drifted from kvbudget.TierPoolSnapshot:\n server: %s\n sdk:    %s", serverJSON, sdkJSON)
	}
}
