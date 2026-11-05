package serveradapter_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/serveradapter"
)

type assistantRoleDoer func(*http.Request) (*http.Response, error)

func (f assistantRoleDoer) Do(r *http.Request) (*http.Response, error) { return f(r) }

// TestProbeRequiresAssistantCompletion keeps echoed request roles from becoming
// chat-capability evidence, while preserving the successful assistant response.
// Estimated added wall-clock: 1 ms, unmeasured; nine injected HTTP calls, no I/O.
// fak-test:runtime fast est=1ms lane=default
func TestProbeRequiresAssistantCompletion(t *testing.T) {
	t.Parallel()
	for _, role := range []string{"assistant", "user", "system"} {
		t.Run(role, func(t *testing.T) {
			var calls []string
			client := assistantRoleDoer(func(req *http.Request) (*http.Response, error) {
				calls = append(calls, req.Method+" "+req.URL.Path)
				var body string
				switch req.URL.Path {
				case "/health":
					body = `{"status":"ok"}`
				case "/v1/models":
					body = `{"object":"list","data":[{"id":"code-model"}]}`
				case "/v1/chat/completions":
					body = fmt.Sprintf(`{"object":"chat.completion","model":"code-model","choices":[{"message":{"role":%q,"content":"OK"}}]}`, role)
				default:
					t.Fatalf("unexpected probe: %s", req.URL)
				}
				return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(body))}, nil
			})
			got, err := serveradapter.ProbeLlamaServer(context.Background(), client, serveradapter.ProbeTarget{
				BaseURL: "http://127.0.0.1:12345", ModelAlias: "code-model",
			})
			wantCalls := []string{"GET /health", "GET /v1/models", "POST /v1/chat/completions"}
			if !reflect.DeepEqual(calls, wantCalls) {
				t.Fatalf("calls = %v, want %v", calls, wantCalls)
			}
			wantCaps := []serveradapter.Capability{serveradapter.FeatureHealth, serveradapter.FeatureModelList}
			if role == "assistant" {
				if err != nil || !got.Ready {
					t.Fatalf("valid assistant response refused: ready=%v err=%v", got.Ready, err)
				}
				wantCaps = append(wantCaps, serveradapter.FeatureChat)
			} else {
				var probeErr *serveradapter.ProbeError
				if !errors.As(err, &probeErr) || probeErr.Kind != serveradapter.FailureMalformedResponse ||
					probeErr.Probe != serveradapter.ProbeChat || probeErr.StatusCode != http.StatusOK || got.Ready {
					t.Fatalf("non-assistant response accepted or misclassified: result=%+v err=%v", got, err)
				}
			}
			if !reflect.DeepEqual(got.Capabilities, wantCaps) {
				t.Fatalf("capabilities = %v, want %v", got.Capabilities, wantCaps)
			}
			if len(got.Observations) != len(wantCalls) || got.Observations[2].Probe != serveradapter.ProbeChat ||
				got.Observations[2].BodyDigest == "" || got.ProbeDigest == "" {
				t.Fatalf("probe evidence lost: %+v", got)
			}
		})
	}
}
