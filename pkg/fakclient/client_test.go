package fakclient_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/fakclient"
)

// fakeGateway is an httptest server that answers the fak-native surface with the
// example JSON documented in docs/fak/openapi.yaml. It records the last request's
// method, path, query, body, and the headers the SDK is responsible for setting,
// so a test can assert the client built the request correctly.
type fakeGateway struct {
	t      *testing.T
	method string
	path   string
	query  string
	body   string
	auth   string
	princ  string
}

func newFakeGateway(t *testing.T) (*fakeGateway, *httptest.Server) {
	t.Helper()
	g := &fakeGateway{t: t}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/fak/adjudicate", func(w http.ResponseWriter, r *http.Request) {
		g.record(r)
		// DENY — a refusal is a successful 200 carried in the verdict.
		writeJSON(w, `{"verdict":{"kind":"DENY","reason":"SELF_MODIFY","by":"selfmod","disposition":"ESCALATE","detail":{"claim":"fak/internal/kernel/kernel.go"}},"trace_id":"t-7f3a9c"}`)
	})
	mux.HandleFunc("/v1/fak/syscall", func(w http.ResponseWriter, r *http.Request) {
		g.record(r)
		writeJSON(w, `{"verdict":{"kind":"ALLOW","by":"tool"},"result":{"status":"OK","content":"{\"rows\":3}"},"trace_id":"t-7f3a9c"}`)
	})
	mux.HandleFunc("/v1/fak/admit", func(w http.ResponseWriter, r *http.Request) {
		g.record(r)
		writeJSON(w, `{"verdict":{"kind":"QUARANTINE","reason":"SECRET_EXFIL","disposition":"TERMINAL"},"result":{"status":"OK","content":"","meta":{"admit":"quarantined"}},"trace_id":"t-7f3a9c"}`)
	})
	mux.HandleFunc("/v1/fak/changes", func(w http.ResponseWriter, r *http.Request) {
		g.record(r)
		writeJSON(w, `{"events":[{"kind":"mutation","seq":43,"tool":"write_file","tags":["fs:/srv"],"world_ver":7,"trust_epoch":1}],"cursor":43}`)
	})
	mux.HandleFunc("/v1/fak/revoke", func(w http.ResponseWriter, r *http.Request) {
		g.record(r)
		writeJSON(w, `{"witness":"sha256:abc123","evicted":3,"trust_epoch":17}`)
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		g.record(r)
		writeJSON(w, `{"object":"list","data":[{"id":"qwen2.5:1.5b","object":"model","owned_by":"fak"}]}`)
	})
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		g.record(r)
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "ok")
	})
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return g, ts
}

func (g *fakeGateway) record(r *http.Request) {
	b, _ := io.ReadAll(r.Body)
	g.method = r.Method
	g.path = r.URL.Path
	g.query = r.URL.RawQuery
	g.body = string(b)
	g.auth = r.Header.Get("Authorization")
	g.princ = r.Header.Get("X-Fak-Principal")
}

func writeJSON(w http.ResponseWriter, body string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = io.WriteString(w, body)
}

func TestAdjudicateRefusalIs200(t *testing.T) {
	g, ts := newFakeGateway(t)
	c := fakclient.New(ts.URL)
	resp, err := c.Adjudicate(context.Background(), fakclient.SyscallRequest{
		Tool:      "write_file",
		Arguments: json.RawMessage(`{"path":"kernel.go"}`),
	})
	if err != nil {
		t.Fatalf("Adjudicate returned error for a DENY verdict; a refusal must be a 200, not an error: %v", err)
	}
	if g.method != http.MethodPost || g.path != "/v1/fak/adjudicate" {
		t.Fatalf("wrong request line: %s %s", g.method, g.path)
	}
	if !strings.Contains(g.body, `"tool":"write_file"`) {
		t.Fatalf("request body missing tool field: %s", g.body)
	}
	if resp.Verdict.Kind != "DENY" || resp.Verdict.Reason != "SELF_MODIFY" {
		t.Fatalf("verdict not parsed: %+v", resp.Verdict)
	}
	if resp.Verdict.Allowed() {
		t.Fatalf("DENY verdict reported Allowed()")
	}
	if resp.Verdict.Disposition != "ESCALATE" || resp.Verdict.Detail["claim"] == "" {
		t.Fatalf("disposition/detail not parsed: %+v", resp.Verdict)
	}
	if resp.TraceID != "t-7f3a9c" {
		t.Fatalf("trace_id not parsed: %q", resp.TraceID)
	}
}

func TestSyscallParsesAllowAndResult(t *testing.T) {
	g, ts := newFakeGateway(t)
	c := fakclient.New(ts.URL)
	resp, err := c.Syscall(context.Background(), fakclient.SyscallRequest{Tool: "read_file"})
	if err != nil {
		t.Fatal(err)
	}
	if g.path != "/v1/fak/syscall" {
		t.Fatalf("wrong path: %s", g.path)
	}
	if !resp.Verdict.Allowed() {
		t.Fatalf("expected ALLOW, got %q", resp.Verdict.Kind)
	}
	if resp.Result == nil || resp.Result.Status != "OK" || !strings.Contains(resp.Result.Content, "rows") {
		t.Fatalf("result envelope not parsed: %+v", resp.Result)
	}
}

func TestAdmitParsesQuarantine(t *testing.T) {
	g, ts := newFakeGateway(t)
	c := fakclient.New(ts.URL)
	resp, err := c.Admit(context.Background(), fakclient.AdmitRequest{
		Tool:   "fetch_url",
		Result: json.RawMessage(`{"page":"api_key=sk-..."}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	if g.path != "/v1/fak/admit" {
		t.Fatalf("wrong path: %s", g.path)
	}
	if resp.Verdict.Kind != "QUARANTINE" {
		t.Fatalf("expected QUARANTINE, got %q", resp.Verdict.Kind)
	}
	if resp.Result == nil || resp.Result.Meta["admit"] != "quarantined" {
		t.Fatalf("quarantine meta not parsed: %+v", resp.Result)
	}
}

func TestChangesForwardsSinceCursor(t *testing.T) {
	g, ts := newFakeGateway(t)
	c := fakclient.New(ts.URL)
	resp, err := c.Changes(context.Background(), 42)
	if err != nil {
		t.Fatal(err)
	}
	if g.method != http.MethodGet {
		t.Fatalf("Changes must be a GET, got %s", g.method)
	}
	if g.query != "since=42" {
		t.Fatalf("since cursor not forwarded as a query param: %q", g.query)
	}
	if len(resp.Events) != 1 || resp.Events[0].Tool != "write_file" || resp.Cursor != 43 {
		t.Fatalf("changes feed not parsed: %+v", resp)
	}
	// since==0 must omit the query entirely (read everything retained).
	_, _ = c.Changes(context.Background(), 0)
	if g.query != "" {
		t.Fatalf("since=0 must not emit a query param, got %q", g.query)
	}
}

func TestRevokeParsesEvictionCount(t *testing.T) {
	g, ts := newFakeGateway(t)
	c := fakclient.New(ts.URL)
	resp, err := c.Revoke(context.Background(), "sha256:abc123")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(g.body, `"witness":"sha256:abc123"`) {
		t.Fatalf("revoke body missing witness: %s", g.body)
	}
	if resp.Evicted != 3 || resp.TrustEpoch != 17 {
		t.Fatalf("revoke response not parsed: %+v", resp)
	}
}

func TestModelsAndHealth(t *testing.T) {
	_, ts := newFakeGateway(t)
	c := fakclient.New(ts.URL)
	models, err := c.Models(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(models.Data) != 1 || models.Data[0].ID != "qwen2.5:1.5b" {
		t.Fatalf("models not parsed: %+v", models)
	}
	if err := c.Health(context.Background()); err != nil {
		t.Fatalf("Health on a 200 must return nil: %v", err)
	}
}

func TestAuthAndPrincipalHeaders(t *testing.T) {
	g, ts := newFakeGateway(t)
	c := fakclient.New(ts.URL, fakclient.WithAPIKey("secret-key"), fakclient.WithPrincipal("tenant-7"))
	if _, err := c.Syscall(context.Background(), fakclient.SyscallRequest{Tool: "t"}); err != nil {
		t.Fatal(err)
	}
	if g.auth != "Bearer secret-key" {
		t.Fatalf("Authorization header not set to bearer token: %q", g.auth)
	}
	if g.princ != "tenant-7" {
		t.Fatalf("X-Fak-Principal header not forwarded: %q", g.princ)
	}
}

func TestAPIErrorOnNon2xx(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"message":"malformed request body","type":"invalid_request_error","code":null,"param":null}}`)
	}))
	defer ts.Close()
	c := fakclient.New(ts.URL)
	_, err := c.Adjudicate(context.Background(), fakclient.SyscallRequest{Tool: "t"})
	if err == nil {
		t.Fatal("expected an error on a 400")
	}
	apiErr, ok := err.(*fakclient.APIError)
	if !ok {
		t.Fatalf("error is not *APIError: %T %v", err, err)
	}
	if apiErr.StatusCode != http.StatusBadRequest {
		t.Fatalf("status not carried: %d", apiErr.StatusCode)
	}
	if apiErr.Type != "invalid_request_error" || !strings.Contains(apiErr.Message, "malformed") {
		t.Fatalf("error body not parsed: %+v", apiErr)
	}
}

// fak-test:runtime fast est=20ms lane=default
// Unmeasured estimate. Public methods use injected HTTP bodies; no gateway or model.
// errorBodyRoundTripper/partialErrorResponseBody are shared with the accepted
// failed streaming error-body regression in stream_test.go.
func TestHTTPMethodsFailedBodyPreserveStatusAndCause(t *testing.T) {
	methods := []struct {
		name, method, path string
		invoke             func(*fakclient.Client) (bool, error)
	}{
		{"adjudicate", http.MethodPost, "/v1/fak/adjudicate", func(c *fakclient.Client) (bool, error) {
			r, err := c.Adjudicate(context.Background(), fakclient.SyscallRequest{Tool: "t"})
			return r == nil, err
		}},
		{"syscall", http.MethodPost, "/v1/fak/syscall", func(c *fakclient.Client) (bool, error) {
			r, err := c.Syscall(context.Background(), fakclient.SyscallRequest{Tool: "t"})
			return r == nil, err
		}},
		{"admit", http.MethodPost, "/v1/fak/admit", func(c *fakclient.Client) (bool, error) {
			r, err := c.Admit(context.Background(), fakclient.AdmitRequest{Tool: "t"})
			return r == nil, err
		}},
		{"changes", http.MethodGet, "/v1/fak/changes?since=7", func(c *fakclient.Client) (bool, error) {
			r, err := c.Changes(context.Background(), 7)
			return r == nil, err
		}},
		{"revoke", http.MethodPost, "/v1/fak/revoke", func(c *fakclient.Client) (bool, error) {
			r, err := c.Revoke(context.Background(), "w")
			return r == nil, err
		}},
		{"models", http.MethodGet, "/v1/models", func(c *fakclient.Client) (bool, error) {
			r, err := c.Models(context.Background())
			return r == nil, err
		}},
		// Health has no response pointer. Its only result is the error checked below.
		{"health", http.MethodGet, "/healthz", func(c *fakclient.Client) (bool, error) { return true, c.Health(context.Background()) }},
	}
	for _, method := range methods {
		for _, status := range []int{http.StatusOK, http.StatusBadRequest, http.StatusServiceUnavailable} {
			for _, cause := range []error{context.Canceled, context.DeadlineExceeded, io.ErrUnexpectedEOF, errors.New("transport body failure")} {
				t.Run(method.name+"/"+http.StatusText(status)+"/"+cause.Error(), func(t *testing.T) {
					body := &partialErrorResponseBody{data: []byte(`{"error":{"message":"retained partial body","type":"upstream_error","code":"read-fault","param":"field"}}`), cause: cause}
					requests := 0
					c := fakclient.New("http://fixture.invalid", fakclient.WithHTTPClient(&http.Client{Transport: errorBodyRoundTripper(func(req *http.Request) (*http.Response, error) {
						requests++
						if req.Method != method.method || req.URL.RequestURI() != method.path {
							t.Fatalf("wrong public route: %s %s", req.Method, req.URL.RequestURI())
						}
						return &http.Response{StatusCode: status, Header: make(http.Header), Body: body, Request: req}, nil
					})}))
					nilResult, err := method.invoke(c)
					if !nilResult || !errors.Is(err, cause) {
						t.Fatalf("response or cause changed: nil=%v err=%v", nilResult, err)
					}
					var api *fakclient.APIError
					if status == http.StatusOK {
						if errors.As(err, &api) || err.Error() != "fak: read response: "+cause.Error() {
							t.Fatalf("2xx failure contract changed: %v", err)
						}
					} else if !errors.As(err, &api) || api.StatusCode != status || api.Message != "retained partial body" || api.Type != "upstream_error" || api.Code != "read-fault" || api.Param != "field" {
						t.Fatalf("API status/body lost: err=%v api=%+v", err, api)
					}
					if errors.Is(err, fakclient.ErrStreamInterrupted) {
						t.Fatal("HTTP body failure became SSE interruption")
					}
					if requests != 1 || body.reads != 1 || body.closes != 1 {
						t.Fatalf("request ownership: requests=%d reads=%d closes=%d", requests, body.reads, body.closes)
					}
				})
			}
		}
	}
}
