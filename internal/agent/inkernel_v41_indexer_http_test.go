package agent_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// fak-test:justify why=contract when=changed:internal/agent/**
// fak-test:runtime medium est=3s lane=default
func TestV41IndexerProjectionOrdinaryHTTPSourceAndReader(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	t.Setenv("FAK_INKERNEL_MAX_TOKENS", "2")
	t.Setenv("FAK_SESSION_LEDGER_DIR", t.TempDir())
	m := v41CompressorHTTPModel(t)
	const prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
	tok, err := tokenizer.FromGGML([]string{prompt, "a", "b", "c", "d", "e", "f", "g"}, []string{"a b"}, []int32{3, 1, 1, 1, 1, 1, 1, 1}, "gpt2")
	if err != nil {
		t.Fatal(err)
	}
	planner := agent.NewInKernelPlanner(m, tok, "native", false, nil, false)
	srv, err := gateway.New(gateway.Config{EngineID: "mock", Model: "native", Logf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	srv.SetPlanner(planner)
	keys := []string{"indexer_projection_device_calls", "indexer_projection_host_calls", "indexer_projection_device_rows", "indexer_projection_host_rows", "indexer_projection_activation_upload_bytes", "indexer_projection_readback_bytes", "indexer_projection_nanos", "indexer_scoring_calls", "indexer_scoring_nanos"}
	initialRaw, err := json.Marshal(m.V41ExpertFaultAttribution())
	if err != nil {
		t.Fatal(err)
	}
	var initial map[string]map[string]json.RawMessage
	if err = json.Unmarshal(initialRaw, &initial); err != nil {
		t.Fatal(err)
	}
	for _, phase := range []string{"prefill", "decode"} {
		for _, key := range keys {
			var value *int64
			if err = json.Unmarshal(initial[phase][key], &value); err != nil || value == nil || *value != 0 {
				t.Fatalf("idle model lacks zero numeric indexer source %s.%s", phase, key)
			}
		}
	}
	if planner.MoEResidencyStats().V41Phases != nil {
		t.Fatal("idle planner fabricated a model-lifetime phase source")
	}
	var previous map[string]map[string]int64
	for request := 0; request < 2; request++ {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"native","messages":[{"role":"user","content":"hi"}],"max_tokens":2,"temperature":0}`))
		req.RemoteAddr = "127.0.0.1:50101"
		req.Header.Set("Content-Type", "application/json")
		completion := httptest.NewRecorder()
		srv.Handler().ServeHTTP(completion, req)
		if completion.Code != http.StatusOK {
			t.Fatalf("ordinary compressed completion status=%d body=%s", completion.Code, completion.Body.String())
		}
		var response struct {
			Usage struct {
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err = json.Unmarshal(completion.Body.Bytes(), &response); err != nil {
			t.Fatal(err)
		}
		if response.Usage.CompletionTokens != 2 {
			t.Fatal("ordinary request did not decode two tokens")
		}
		get := httptest.NewRequest(http.MethodGet, "/debug/vars", nil)
		get.RemoteAddr = "127.0.0.1:50102"
		reader := httptest.NewRecorder()
		srv.Handler().ServeHTTP(reader, get)
		if reader.Code != http.StatusOK {
			t.Fatal("default indexer reader unavailable")
		}
		var doc struct {
			MoE struct {
				Requests   int64           `json:"requests"`
				Checkpoint json.RawMessage `json:"checkpoint"`
				V41        struct {
					Scope           string `json:"scope"`
					Prefill, Decode map[string]json.RawMessage
					Projections     map[string]map[string]json.RawMessage `json:"projections"`
					Compressor      map[string]map[string]json.RawMessage `json:"compressor"`
					Indexer         map[string]map[string]json.RawMessage `json:"indexer"`
				} `json:"v41_phases"`
			} `json:"moe_residency"`
		}
		if err = json.Unmarshal(reader.Body.Bytes(), &doc); err != nil {
			t.Fatal(err)
		}
		if doc.MoE.V41.Scope != "model_lifetime" || doc.MoE.Requests != 0 || len(doc.MoE.Checkpoint) != 0 {
			t.Fatal("indexer source must remain default-readable independently of checkpoint/ring")
		}
		if len(doc.MoE.V41.Indexer) != 2 {
			t.Fatal("indexer reader must contain exactly prefill/decode")
		}
		sourceRaw, err := json.Marshal(m.V41ExpertFaultAttribution())
		if err != nil {
			t.Fatal(err)
		}
		var source map[string]map[string]json.RawMessage
		if err = json.Unmarshal(sourceRaw, &source); err != nil {
			t.Fatal(err)
		}
		current := map[string]map[string]int64{}
		for _, phase := range []string{"prefill", "decode"} {
			legacy := doc.MoE.V41.Prefill
			if phase == "decode" {
				legacy = doc.MoE.V41.Decode
			}
			for _, existing := range []struct {
				name   string
				fields map[string]json.RawMessage
				keys   []string
			}{
				{"legacy", legacy, v41HTTPLegacyKeys},
				{"projections", doc.MoE.V41.Projections[phase], v41HTTPProjectionKeys},
				{"compressor", doc.MoE.V41.Compressor[phase], v41HTTPCompressorKeys},
			} {
				if err := v41HTTPReaderMatchesSource(existing.fields, source[phase], existing.keys); err != nil {
					t.Fatalf("%s %s reader: %v", existing.name, phase, err)
				}
			}
			fields := doc.MoE.V41.Indexer[phase]
			if len(fields) != len(keys) {
				t.Fatalf("indexer %s fields=%d want source keys=%d", phase, len(fields), len(keys))
			}
			current[phase] = map[string]int64{}
			for _, key := range keys {
				var got, want *int64
				if err = json.Unmarshal(fields[key], &got); err != nil {
					t.Fatal(err)
				}
				if err = json.Unmarshal(source[phase][key], &want); err != nil {
					t.Fatal(err)
				}
				if got == nil || want == nil || *got != *want {
					t.Fatalf("ordinary default reader differs from real model source %s.%s", phase, key)
				}
				current[phase][key] = *got
				if _, ok := legacy[key]; ok {
					t.Fatal("indexer field contaminated legacy phase")
				}
			}
			if current[phase]["indexer_projection_host_calls"] <= 0 || current[phase]["indexer_projection_host_rows"] <= 0 || current[phase]["indexer_projection_nanos"] <= 0 || current[phase]["indexer_scoring_calls"] <= 0 || current[phase]["indexer_scoring_nanos"] <= 0 || current[phase]["indexer_projection_device_calls"] != 0 {
				t.Fatal("ordinary CPU request did not invoke actual indexer source")
			}
			if previous != nil && current[phase]["indexer_projection_host_calls"] <= previous[phase]["indexer_projection_host_calls"] {
				t.Fatal("second ordinary request did not advance indexer source")
			}
		}
		previous = current
		beforeClone, err := json.Marshal(planner.MoEResidencyStats())
		if err != nil {
			t.Fatal(err)
		}
		copyStats := planner.MoEResidencyStats()
		if !v41IndexerHTTPMutateCounter(reflect.ValueOf(copyStats.V41Phases)) {
			t.Fatal("indexer source reader lacks mutable value child for clone witness")
		}
		afterClone, err := json.Marshal(planner.MoEResidencyStats())
		if err != nil {
			t.Fatal(err)
		}
		if string(beforeClone) != string(afterClone) {
			t.Fatal("reader mutation changed retained indexer lifetime snapshot")
		}
	}
}

func v41IndexerHTTPMutateCounter(value reflect.Value) bool {
	if value.Kind() == reflect.Pointer {
		if value.IsNil() {
			return false
		}
		return v41IndexerHTTPMutateCounter(value.Elem())
	}
	if value.Kind() != reflect.Struct {
		return false
	}
	typ := value.Type()
	for index := 0; index < value.NumField(); index++ {
		field := value.Field(index)
		if typ.Field(index).Tag.Get("json") == "indexer_projection_host_calls" && field.CanSet() && field.CanInt() {
			field.SetInt(field.Int() + 999)
			return true
		}
		if v41IndexerHTTPMutateCounter(field) {
			return true
		}
	}
	return false
}
