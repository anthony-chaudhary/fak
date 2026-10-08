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
	V41Phases *checkpointV41Wire `json:"v41_phases"`
}

type checkpointV41Wire struct {
	Scope   string                     `json:"scope"`
	Prefill map[string]json.RawMessage `json:"prefill"`
	Decode  map[string]json.RawMessage `json:"decode"`
}

func checkpointV41Number(t *testing.T, phase map[string]json.RawMessage, key string) float64 {
	t.Helper()
	value, ok := phase[key]
	if !ok {
		t.Fatalf("V41 phase missing numeric field %q", key)
	}
	var number float64
	if err := json.Unmarshal(value, &number); err != nil {
		t.Fatalf("V41 field %q is not numeric: %v", key, err)
	}
	return number
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
			V41Phases  struct {
				Scope   json.RawMessage            `json:"scope"`
				Prefill map[string]json.RawMessage `json:"prefill"`
				Decode  map[string]json.RawMessage `json:"decode"`
			} `json:"v41_phases"`
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
		allowedPhase := map[string]bool{
			"tokens": true, "attention_contraction_calls": true, "attention_contraction_nanos": true,
			"faults": true, "faulted_bytes": true, "fault_nanos": true,
			"dequant_bytes": true, "dequant_nanos": true, "contractions": true, "contraction_nanos": true,
			"mhc_projection_device_calls": true, "mhc_projection_host_calls": true, "mhc_projection_nanos": true,
			"mhc_projection_activation_upload_bytes": true, "mhc_projection_readback_bytes": true,
			"dense_projection_device_calls": true, "dense_projection_host_calls": true, "dense_projection_nanos": true,
			"dense_projection_activation_upload_bytes": true, "dense_projection_readback_bytes": true,
			"grouped_output_device_calls": true, "grouped_output_host_calls": true, "grouped_output_nanos": true,
			"grouped_output_activation_upload_bytes": true, "grouped_output_readback_bytes": true,
			"expert_activation_device_calls": true, "expert_activation_host_calls": true, "expert_activation_nanos": true,
			"expert_activation_readback_bytes": true, "incremental_engram_injections": true, "incremental_engram_nanos": true,
		}
		for phase, fields := range map[string]map[string]json.RawMessage{"prefill": bounded.V41Phases.Prefill, "decode": bounded.V41Phases.Decode} {
			if len(fields) != len(allowedPhase) {
				t.Fatalf("V41 %s field count = %d, want bounded %d-field schema", phase, len(fields), len(allowedPhase))
			}
			for key, value := range fields {
				if !allowedPhase[key] {
					t.Fatalf("V41 %s exposed unexpected field %q", phase, key)
				}
				var number float64
				if err := json.Unmarshal(value, &number); err != nil {
					t.Fatalf("V41 %s field %q is not numeric", phase, key)
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
	firstRead := read()
	first := firstRead.Checkpoint
	if first.Scope != "model_lifetime" || first.Reads == 0 || first.BytesRead == 0 {
		t.Fatalf("first snapshot scope=%q reads=%d bytes=%d", first.Scope, first.Reads, first.BytesRead)
	}
	if firstRead.V41Phases == nil {
		t.Fatal("first V41 snapshot absent, want model_lifetime")
	}
	if firstRead.V41Phases.Scope != "model_lifetime" {
		t.Fatalf("first V41 snapshot scope = %q, want model_lifetime", firstRead.V41Phases.Scope)
	}
	firstPrefillAttention := checkpointV41Number(t, firstRead.V41Phases.Prefill, "attention_contraction_calls")
	firstDecodeAttention := checkpointV41Number(t, firstRead.V41Phases.Decode, "attention_contraction_calls")
	if firstPrefillAttention == 0 || firstDecodeAttention == 0 {
		t.Fatalf("ordinary CPU request did not execute both V41 attention phases: prefill=%d decode=%d",
			int(firstPrefillAttention), int(firstDecodeAttention))
	}
	firstPrefillAttentionNanos := checkpointV41Number(t, firstRead.V41Phases.Prefill, "attention_contraction_nanos")
	firstDecodeAttentionNanos := checkpointV41Number(t, firstRead.V41Phases.Decode, "attention_contraction_nanos")
	if firstPrefillAttentionNanos <= 0 || firstDecodeAttentionNanos <= 0 {
		t.Fatalf("default V41 attention duration prefill=%dns decode=%dns, want both positive",
			int64(firstPrefillAttentionNanos), int64(firstDecodeAttentionNanos))
	}
	firstPrefillHost := checkpointV41Number(t, firstRead.V41Phases.Prefill, "dense_projection_host_calls")
	firstDecodeHost := checkpointV41Number(t, firstRead.V41Phases.Decode, "dense_projection_host_calls")
	if firstPrefillHost == 0 || firstDecodeHost == 0 {
		t.Fatalf("ordinary request did not report actual host execution: prefill=%d decode=%d",
			int(firstPrefillHost), int(firstDecodeHost))
	}
	firstPrefillDenseNanos := checkpointV41Number(t, firstRead.V41Phases.Prefill, "dense_projection_nanos")
	firstDecodeDenseNanos := checkpointV41Number(t, firstRead.V41Phases.Decode, "dense_projection_nanos")
	if firstPrefillDenseNanos <= 0 || firstDecodeDenseNanos <= 0 {
		t.Fatalf("default dense projection duration prefill=%dns decode=%dns, want both positive",
			int64(firstPrefillDenseNanos), int64(firstDecodeDenseNanos))
	}
	post("50102")
	secondRead := read()
	second := secondRead.Checkpoint
	source := m.ExpertCheckpointStats()
	if second.Hits <= first.Hits {
		t.Fatalf("second request did not reuse retained file rows: hits %d->%d", first.Hits, second.Hits)
	}
	if second.Reads != source.Reads || second.Hits != source.Hits || second.BytesRead != source.BytesRead {
		t.Fatalf("served snapshot reads=%d hits=%d bytes=%d; source reads=%d hits=%d bytes=%d",
			second.Reads, second.Hits, second.BytesRead, source.Reads, source.Hits, source.BytesRead)
	}
	phaseSource := m.V41ExpertFaultAttribution()
	if secondRead.V41Phases == nil {
		t.Fatal("second request dropped V41 phase snapshot while checkpoint data remained")
	}
	sourceJSON, err := json.Marshal(phaseSource)
	if err != nil {
		t.Fatal(err)
	}
	var sourcePhases struct {
		Prefill map[string]json.RawMessage `json:"prefill"`
		Decode  map[string]json.RawMessage `json:"decode"`
	}
	if err := json.Unmarshal(sourceJSON, &sourcePhases); err != nil {
		t.Fatal(err)
	}
	for phase, pair := range map[string][2]map[string]json.RawMessage{
		"prefill": {secondRead.V41Phases.Prefill, sourcePhases.Prefill},
		"decode":  {secondRead.V41Phases.Decode, sourcePhases.Decode},
	} {
		for key := range pair[0] {
			if got, want := checkpointV41Number(t, pair[0], key), checkpointV41Number(t, pair[1], key); got != want {
				t.Fatalf("served V41 %s.%s=%v, source=%v (snapshot must be latest, not summed)", phase, key, got, want)
			}
		}
	}
	secondPrefillAttention := checkpointV41Number(t, secondRead.V41Phases.Prefill, "attention_contraction_calls")
	secondDecodeAttention := checkpointV41Number(t, secondRead.V41Phases.Decode, "attention_contraction_calls")
	if secondPrefillAttention <= firstPrefillAttention || secondDecodeAttention <= firstDecodeAttention {
		t.Fatalf("second request did not advance both phase counters: prefill %d->%d decode %d->%d",
			int(firstPrefillAttention), int(secondPrefillAttention), int(firstDecodeAttention), int(secondDecodeAttention))
	}
	t.Logf("file-backed checkpoint cumulative reads=%d hits=%d bytes=%d", second.Reads, second.Hits, second.BytesRead)
}

// TestCheckpointResidencyHTTPReportsV41WithoutCheckpoint proves the default
// reader admits V4.1 model-lifetime phases independently of both optional
// residency sources. The real GGUF is loaded resident (no checkpoint tier), and
// an ordinary CPU request uses no observation flag. [SW-VERIFIED], not hardware.
// fak-test:runtime fast est=1s lane=default
func TestCheckpointResidencyHTTPReportsV41WithoutCheckpoint(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	t.Setenv("FAK_INKERNEL_MAX_TOKENS", "2")

	ws, err := OpenWeights(writeDeepSeek41ExpertHALFile(t))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ws.Close() })
	m, err := ws.QuantModelQ4KProfileOptionsContext(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	const prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
	tok, err := tokenizer.FromGGML(
		[]string{prompt, "a", "b", "c", "d", "e", "f", "g"},
		[]string{"a b"}, []int32{3, 1, 1, 1, 1, 1, 1, 1}, "gpt2",
	)
	if err != nil {
		t.Fatal(err)
	}
	srv, err := gateway.New(gateway.Config{EngineID: "mock", Model: "native", Logf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	srv.SetPlanner(agent.NewInKernelPlanner(m, tok, "native", false, nil, false))

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"native","messages":[{"role":"user","content":"hi"}],"max_tokens":2,"temperature":0}`))
	req.RemoteAddr = "127.0.0.1:50201"
	req.Header.Set("Content-Type", "application/json")
	post := httptest.NewRecorder()
	srv.Handler().ServeHTTP(post, req)
	if post.Code != http.StatusOK {
		t.Fatalf("POST /v1/chat/completions status=%d body=%s", post.Code, post.Body.String())
	}

	get := httptest.NewRecorder()
	srv.Handler().ServeHTTP(get, httptest.NewRequest(http.MethodGet, "/debug/vars", nil))
	if get.Code != http.StatusOK {
		t.Fatalf("GET /debug/vars status=%d body=%s", get.Code, get.Body.String())
	}
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(get.Body.Bytes(), &doc); err != nil {
		t.Fatal(err)
	}
	if len(doc["moe_residency"]) == 0 {
		t.Fatal("checkpointless, ringless V4.1 request omitted moe_residency from /debug/vars")
	}
	var got struct {
		Requests   int64              `json:"requests"`
		Checkpoint json.RawMessage    `json:"checkpoint"`
		V41Phases  *checkpointV41Wire `json:"v41_phases"`
	}
	if err := json.Unmarshal(doc["moe_residency"], &got); err != nil {
		t.Fatal(err)
	}
	if got.Requests != 0 {
		t.Fatalf("resident request ring count = %d, want 0", got.Requests)
	}
	if len(got.Checkpoint) != 0 && string(got.Checkpoint) != "null" {
		t.Fatalf("resident request exposed checkpoint JSON %s, want omitted", got.Checkpoint)
	}
	if got.V41Phases == nil {
		t.Fatal("checkpointless, ringless V4.1 request was omitted from /debug/vars")
	}
	if got.V41Phases.Scope != "model_lifetime" {
		t.Fatalf("V41 phase scope = %q, want model_lifetime", got.V41Phases.Scope)
	}
	if pre, dec := checkpointV41Number(t, got.V41Phases.Prefill, "attention_contraction_calls"), checkpointV41Number(t, got.V41Phases.Decode, "attention_contraction_calls"); pre == 0 || dec == 0 {
		t.Fatalf("checkpointless V41 attention calls prefill=%d decode=%d, want both positive", int(pre), int(dec))
	}
	if pre, dec := checkpointV41Number(t, got.V41Phases.Prefill, "attention_contraction_nanos"), checkpointV41Number(t, got.V41Phases.Decode, "attention_contraction_nanos"); pre <= 0 || dec <= 0 {
		t.Fatalf("checkpointless default attention duration prefill=%dns decode=%dns, want both positive", int64(pre), int64(dec))
	}
}
