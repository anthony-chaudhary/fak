package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// A streamed model can fault its host checkpoint without admitting a device ring.
// fak-test:runtime fast est=100ms lane=default
func TestCheckpointResidencyDefaultDebugReaderEmitsBoundedModelLifetimeSnapshot(t *testing.T) {
	srv := newTestServer(t)
	var ledger agent.MoEResidencyLedger
	if err := json.Unmarshal([]byte(`{"checkpoint":{"scope":"model_lifetime","reads":13,"hits":6,"bytes_read":53248,"evictions":3,"failures":2,"budget_bytes":1048576,"resident_bytes":327680,"peak_bytes":524288,"resident_count":5,"overlay_rows":7,"overlay_bytes_read":28672}}`), &ledger); err != nil {
		t.Fatal(err)
	}
	srv.planner = moeResidencyPlanner{ledger: ledger}

	rr := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/debug/vars", nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("GET /debug/vars status=%d body=%s", rr.Code, rr.Body.String())
	}
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(rr.Body.Bytes(), &doc); err != nil {
		t.Fatalf("decode /debug/vars: %v", err)
	}
	var block struct {
		Checkpoint map[string]any `json:"checkpoint"`
	}
	if err := json.Unmarshal(doc["moe_residency"], &block); err != nil {
		t.Fatalf("checkpoint-only moe_residency absent from served reader: %v; body=%s", err, rr.Body.String())
	}
	want := map[string]float64{
		"reads": 13, "hits": 6, "bytes_read": 52 << 10, "evictions": 3, "failures": 2,
		"budget_bytes": 1 << 20, "resident_bytes": 320 << 10, "peak_bytes": 512 << 10,
		"resident_count": 5, "overlay_rows": 7, "overlay_bytes_read": 28 << 10,
	}
	if block.Checkpoint["scope"] != "model_lifetime" {
		t.Fatalf("checkpoint scope=%v want model_lifetime", block.Checkpoint["scope"])
	}
	for key, value := range want {
		if got, ok := block.Checkpoint[key].(float64); !ok || got != value {
			t.Fatalf("checkpoint[%q]=%v want %.0f; block=%v", key, block.Checkpoint[key], value, block.Checkpoint)
		}
	}
	for _, forbidden := range []string{"last_error", "error", "payload"} {
		if _, ok := block.Checkpoint[forbidden]; ok {
			t.Fatalf("bounded checkpoint reader leaked forbidden field %q: %v", forbidden, block.Checkpoint)
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestCheckpointResidencyAbsentSnapshotStaysOffWire(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = moeResidencyPlanner{ledger: moeEngagedLedger()}
	rr := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/debug/vars", nil))
	var doc map[string]struct {
		Checkpoint json.RawMessage `json:"checkpoint"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &doc); err != nil {
		t.Fatal(err)
	}
	if len(doc["moe_residency"].Checkpoint) != 0 {
		t.Fatalf("absent checkpoint snapshot emitted on wire: %s", doc["moe_residency"].Checkpoint)
	}
}

// [SW-VERIFIED] An empty tier proves source wiring, not checkpoint IO or Flash throughput.
// fak-test:runtime fast est=200ms lane=default
func TestCheckpointResidencyOrdinaryNativeRequestReachesDefaultDebugReader(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	t.Setenv("FAK_INKERNEL_MAX_TOKENS", "2")
	const budget = int64(64 << 10)
	cfg := model.Config{HiddenSize: 32, NumLayers: 1, NumHeads: 4, NumKVHeads: 2, HeadDim: 8, IntermediateSize: 64, VocabSize: 512, RMSNormEps: 1e-5, RopeTheta: 10000, TieWordEmbeddings: true, EOSTokenID: -1}
	m := model.NewSynthetic(cfg)
	m.Quantize()
	m.SetExpertCheckpoint(model.NewExpertCheckpointTier(budget))
	srv := newEngineStepNativeServer(t)
	srv.planner = agent.NewInKernelPlanner(m, newByteLevelTokenizer(t), "native", false, nil, false)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"native","messages":[{"role":"user","content":"hi"}],"max_tokens":2,"temperature":0}`))
	req.RemoteAddr = "127.0.0.1:50001"
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("native completion status=%d body=%s", rec.Code, rec.Body.String())
	}

	rr := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/debug/vars", nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("GET /debug/vars status=%d body=%s", rr.Code, rr.Body.String())
	}
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(rr.Body.Bytes(), &doc); err != nil {
		t.Fatal(err)
	}
	var got struct {
		Requests   int64 `json:"requests"`
		Checkpoint struct {
			Scope       string `json:"scope"`
			Reads       int    `json:"reads"`
			BudgetBytes int64  `json:"budget_bytes"`
		} `json:"checkpoint"`
	}
	if err := json.Unmarshal(doc["moe_residency"], &got); err != nil {
		t.Fatalf("decode moe_residency: %v", err)
	}
	if got.Checkpoint.Scope != "model_lifetime" || got.Checkpoint.BudgetBytes != budget || got.Checkpoint.Reads != 0 {
		t.Fatalf("checkpoint scope=%q budget=%d reads=%d", got.Checkpoint.Scope, got.Checkpoint.BudgetBytes, got.Checkpoint.Reads)
	}
	if got.Requests != 0 {
		t.Fatalf("checkpoint-only request changed ring requests: got %d", got.Requests)
	}
}
