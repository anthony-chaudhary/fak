package webbench

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync/atomic"
	"testing"
	"time"
)

// These tests use observed wire responses, not server-global counters or prompt estimates.
func TestServingRequestEvidenceWireAndMissingFields(t *testing.T) {
	for _, tc := range []struct {
		name, envelope string
		want           map[string]any
		exact          *int
	}{
		{"absent", "", nil, nil},
		{"null", `,"usage":null`, nil, nil},
		{"empty", `,"usage":{}`, map[string]any{"prompt_tokens": nil, "completion_tokens": nil, "total_tokens": nil, "cached_tokens": nil}, nil},
		{"partial", `,"usage":{"prompt_tokens":19,"completion_tokens":3}`, map[string]any{"prompt_tokens": float64(19), "completion_tokens": float64(3), "total_tokens": nil, "cached_tokens": nil}, evidenceInt(3)},
		{"zero", `,"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0,"prompt_tokens_details":{"cached_tokens":0}}`, map[string]any{"prompt_tokens": float64(0), "completion_tokens": float64(0), "total_tokens": float64(0), "cached_tokens": float64(0)}, evidenceInt(0)},
		{"aliases", `,"usage":{"input_tokens":23,"output_tokens":5,"total_tokens":28,"input_tokens_details":{"cached_tokens":11}}`, map[string]any{"prompt_tokens": float64(23), "completion_tokens": float64(5), "total_tokens": float64(28), "cached_tokens": float64(11)}, evidenceInt(5)},
		{"cache_only", `,"usage":{"prompt_tokens_details":{"cached_tokens":9}}`, map[string]any{"prompt_tokens": nil, "completion_tokens": nil, "total_tokens": nil, "cached_tokens": float64(9)}, nil},
	} {
		for _, streaming := range []bool{true, false} {
			t.Run(fmt.Sprintf("%s/sse=%t", tc.name, streaming), func(t *testing.T) {
				var calls atomic.Int32
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					calls.Add(1)
					if r.URL.Path != "/v1/chat/completions" || r.Header.Get("X-Fak-Project-Arm") != "evidence-arm" {
						t.Errorf("request path/arm %s/%q", r.URL.Path, r.Header.Get("X-Fak-Project-Arm"))
					}
					var body struct {
						Stream        bool `json:"stream"`
						StreamOptions struct {
							IncludeUsage bool `json:"include_usage"`
						} `json:"stream_options"`
					}
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil || !body.Stream || !body.StreamOptions.IncludeUsage {
						t.Errorf("stream usage request: %+v, %v", body, err)
					}
					if streaming {
						w.Header().Set("Content-Type", "text/event-stream")
						fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n")
						fmt.Fprintf(w, "data: {\"choices\":[]%s}\n\ndata: [DONE]\n\n", tc.envelope)
					} else {
						w.Header().Set("Content-Type", "application/json")
						fmt.Fprintf(w, `{"choices":[{"message":{"content":"hello"}}]%s}`, tc.envelope)
					}
				}))
				defer srv.Close()
				sample := MeasureSSERequest(context.Background(), srv.Client(), ServingTrackConfig{BaseURL: srv.URL + "/v1", ProjectArm: "evidence-arm"}, "fixture", ServingRequest{ID: "one", PromptTokensEstimate: 999, MaxOutputTokens: 8}, time.Second)
				if calls.Load() != 1 || sample.Status != "ok" {
					t.Fatalf("calls=%d sample=%+v", calls.Load(), sample)
				}
				encoded, err := json.Marshal(sample)
				if err != nil {
					t.Fatal(err)
				}
				var got map[string]any
				if err = json.Unmarshal(encoded, &got); err != nil {
					t.Fatal(err)
				}
				usage, exists := got["usage"]
				if !exists {
					t.Fatal("usage key missing; absence must be explicit null")
				}
				if tc.want == nil {
					if usage != nil {
						t.Fatalf("invented usage: %v", usage)
					}
				} else if !reflect.DeepEqual(usage, tc.want) {
					t.Fatalf("usage=%v want=%v", usage, tc.want)
				}
				if tc.exact == nil {
					if sample.OutputTokensExact != nil {
						t.Fatalf("invented completion count %d", *sample.OutputTokensExact)
					}
				} else if sample.OutputTokensExact == nil || *sample.OutputTokensExact != *tc.exact {
					t.Fatalf("completion count %v want %d", sample.OutputTokensExact, *tc.exact)
				}
			})
		}
	}
}

func evidenceInt(v int) *int { return &v }

func TestServingRequestEvidenceConcurrentSamplesRemainDistinct(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.Header.Get("X-Fak-Project-Arm") != "parallel-evidence" {
			t.Error("arm missing from one concurrent request")
		}
		var body struct {
			Messages []ChatMessage `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n")
		if body.Messages[0].Content == "measured" {
			fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":70,\"completion_tokens\":2,\"total_tokens\":72,\"prompt_tokens_details\":{\"cached_tokens\":64}}}\n\n")
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()
	work := []ServingRequest{{ID: "measured", Messages: []ChatMessage{{Role: "user", Content: "measured"}}}, {ID: "unknown", Messages: []ChatMessage{{Role: "user", Content: "unknown"}}}}
	report, err := RunServingParity(context.Background(), ServingParityConfig{Model: "fixture", Tracks: []ServingTrackConfig{{Track: TrackOurs, BaseURL: srv.URL + "/v1", ProjectArm: "parallel-evidence"}}, Workload: work, Concurrency: 2, Timeout: time.Second, Client: srv.Client()})
	if err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 2 || len(report.Tracks) != 1 {
		t.Fatalf("calls=%d tracks=%d", calls.Load(), len(report.Tracks))
	}
	track := report.Tracks[0]
	if track.ProjectArm != "parallel-evidence" || len(track.Samples) != 2 {
		t.Fatalf("report provenance: %+v", track)
	}
	if track.Samples[0].Usage == nil || track.Samples[0].Usage.CachedTokens == nil || *track.Samples[0].Usage.CachedTokens != 64 {
		t.Fatalf("measured sample lost usage: %+v", track.Samples[0])
	}
	if track.Samples[1].Usage != nil {
		t.Fatalf("unknown sample borrowed peer usage: %+v", track.Samples[1])
	}
}
