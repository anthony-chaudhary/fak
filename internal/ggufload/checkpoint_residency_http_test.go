package ggufload

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

type checkpointResidencyWire struct {
	Checkpoint struct {
		Scope     string `json:"scope"`
		Reads     int    `json:"reads"`
		Hits      int    `json:"hits"`
		BytesRead int64  `json:"bytes_read"`
	} `json:"checkpoint"`
}

// fak-test:runtime fast est=1s lane=default
func TestCheckpointResidencyHTTPReportsRealFileBackedCumulativeIO(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	t.Setenv("FAK_INKERNEL_MAX_TOKENS", "2")

	ws, err := OpenWeights(writeDeepSeek41ExpertHALFile(t))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ws.Close() })
	const prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
	tok, err := tokenizer.FromGGML(
		[]string{prompt, "a", "b", "c", "d", "e", "f", "g"},
		[]string{"a b"}, []int32{3, 1, 1, 1, 1, 1, 1, 1}, "gpt2",
	)
	if err != nil {
		t.Fatal(err)
	}
	const hostBudget = int64(64 << 20)
	m, err := ws.QuantModelQ4KProfileOptionsContext(t.Context(), nil, WithStreamedExperts(hostBudget))
	if err != nil {
		t.Fatal(err)
	}
	srv, err := gateway.New(gateway.Config{EngineID: "mock", Model: "native", Logf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	srv.SetPlanner(agent.NewInKernelPlanner(m, tok, "native", false, nil, false))

	read := func() checkpointResidencyWire {
		t.Helper()
		rr := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/debug/vars", nil))
		if rr.Code != http.StatusOK {
			t.Fatalf("GET /debug/vars status=%d body=%s", rr.Code, rr.Body.String())
		}
		var doc map[string]json.RawMessage
		if err := json.Unmarshal(rr.Body.Bytes(), &doc); err != nil {
			t.Fatal(err)
		}
		var got checkpointResidencyWire
		if err := json.Unmarshal(doc["moe_residency"], &got); err != nil {
			t.Fatal(err)
		}
		var bounded struct {
			Checkpoint map[string]json.RawMessage `json:"checkpoint"`
		}
		if err := json.Unmarshal(doc["moe_residency"], &bounded); err != nil {
			t.Fatal(err)
		}
		allowed := map[string]bool{
			"scope": true, "reads": true, "hits": true, "bytes_read": true,
			"evictions": true, "failures": true, "budget_bytes": true,
			"resident_bytes": true, "peak_bytes": true, "resident_count": true,
			"overlay_rows": true, "overlay_bytes_read": true,
		}
		for key, value := range bounded.Checkpoint {
			if !allowed[key] {
				t.Fatalf("checkpoint reader exposed unexpected field %q", key)
			}
			if key != "scope" {
				var number float64
				if err := json.Unmarshal(value, &number); err != nil {
					t.Fatalf("checkpoint field %q is not numeric", key)
				}
			}
		}
		return got
	}
	post := func(port string) {
		t.Helper()
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"native","messages":[{"role":"user","content":"hi"}],"max_tokens":2,"temperature":0}`))
		req.RemoteAddr = "127.0.0.1:" + port
		req.Header.Set("Content-Type", "application/json")
		rr := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rr, req)
		if rr.Code != http.StatusOK {
			t.Fatalf("POST /v1/chat/completions status=%d body=%s", rr.Code, rr.Body.String())
		}
	}

	post("50101")
	first := read().Checkpoint
	if first.Scope != "model_lifetime" || first.Reads == 0 || first.BytesRead == 0 {
		t.Fatalf("first snapshot scope=%q reads=%d bytes=%d", first.Scope, first.Reads, first.BytesRead)
	}
	post("50102")
	second := read().Checkpoint
	source := m.ExpertCheckpointStats()
	if second.Hits <= first.Hits {
		t.Fatalf("second request did not reuse retained file rows: hits %d->%d", first.Hits, second.Hits)
	}
	if second.Reads != source.Reads || second.Hits != source.Hits || second.BytesRead != source.BytesRead {
		t.Fatalf("served snapshot reads=%d hits=%d bytes=%d; source reads=%d hits=%d bytes=%d",
			second.Reads, second.Hits, second.BytesRead, source.Reads, source.Hits, source.BytesRead)
	}
	t.Logf("file-backed checkpoint cumulative reads=%d hits=%d bytes=%d", second.Reads, second.Hits, second.BytesRead)
}
