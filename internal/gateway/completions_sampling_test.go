package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// fak-test:runtime fast est=100ms
func TestCompletionsSamplingCarriers(t *testing.T) {
	// Exercise the actual HTTP planner and ride wire, not just option setters.
	forwarded := make(chan map[string]any, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode upstream body: %v", err)
			http.Error(w, "invalid body", http.StatusBadRequest)
			return
		}
		forwarded <- body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer upstream.Close()
	srv := newTestServer(t)
	srv.planner = agent.NewHTTPPlanner(upstream.URL, "test-model", "")
	handler := srv.Handler()
	post := func(path string, stream bool, fields map[string]any) *httptest.ResponseRecorder {
		t.Helper()
		body := map[string]any{"model": "test-model", "stream": stream}
		if path == "/v1/completions" {
			body["prompt"] = "answer"
		} else {
			body["messages"] = []map[string]string{{"role": "user", "content": "answer"}}
		}
		for key, value := range fields {
			body[key] = value
		}
		raw, err := json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
		req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(raw))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		return rec
	}
	keys := []string{"seed", "logit_bias", "frequency_penalty", "presence_penalty", "top_k", "min_p"}
	for _, tc := range []struct {
		name   string
		fields map[string]any
	}{
		{"omitted", map[string]any{}},
		{"zero", map[string]any{"seed": float64(0), "logit_bias": map[string]any{"42": float64(0)}, "frequency_penalty": float64(0), "presence_penalty": float64(0), "top_k": float64(0), "min_p": float64(0)}},
		{"nonzero", map[string]any{"seed": float64(-11), "logit_bias": map[string]any{"42": -12.5}, "frequency_penalty": 0.5, "presence_penalty": -0.25, "top_k": float64(40), "min_p": 0.05}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, route := range []struct {
				path   string
				stream bool
			}{{"/v1/chat/completions", false}, {"/v1/completions", false}, {"/v1/completions", true}} {
				rec := post(route.path, route.stream, tc.fields)
				if rec.Code != http.StatusOK {
					t.Fatalf("%s stream=%v: status=%d body=%s", route.path, route.stream, rec.Code, rec.Body)
				}
				select {
				case got := <-forwarded:
					for _, key := range keys {
						want, present := tc.fields[key]
						value, exists := got[key]
						if exists != present || !reflect.DeepEqual(value, want) {
							t.Errorf("%s stream=%v: %s=%v (present=%v), want %v (present=%v)", route.path, route.stream, key, value, exists, want, present)
						}
					}
				default:
					t.Fatal("request never reached the ride engine")
				}
				if route.stream && !bytes.Contains(rec.Body.Bytes(), []byte("[DONE]")) {
					t.Fatalf("legacy stream did not finish: %s", rec.Body)
				}
			}
		})
	}
	// The same invalid values must fail before routing, with chat's typed code.
	for _, key := range []string{"frequency_penalty", "presence_penalty"} {
		for _, path := range []string{"/v1/chat/completions", "/v1/completions"} {
			rec := post(path, true, map[string]any{key: 3})
			if rec.Code != http.StatusBadRequest || !bytes.Contains(rec.Body.Bytes(), []byte(`"code":"sampling_param_out_of_range"`)) {
				t.Fatalf("%s %s: status=%d body=%s", path, key, rec.Code, rec.Body)
			}
		}
	}
	// Unsupported seed must remain an HTTP refusal, even on the legacy SSE route.
	planner := agent.NewHTTPPlanner(upstream.URL, "test-model", "")
	planner.Provider = agent.ProviderAnthropic
	srv.planner = planner
	for _, path := range []string{"/v1/chat/completions", "/v1/completions"} {
		for _, stream := range []bool{false, true} {
			rec := post(path, stream, map[string]any{"seed": 0})
			if rec.Code != http.StatusBadRequest || !bytes.Contains(rec.Body.Bytes(), []byte(`"code":"sampling_seed_unsupported"`)) {
				t.Fatalf("%s stream=%v: status=%d body=%s", path, stream, rec.Code, rec.Body)
			}
		}
	}
	select {
	case got := <-forwarded:
		t.Fatalf("refused sampling request reached upstream: %v", got)
	default:
	}
}
