package main

// accounts_rehome_test.go — `fak accounts rehome` against a stub gateway speaking the
// POST /v1/fak/account/rehome wire: the applied swap renders from->to and exits 0, a
// gateway refusal (409 no sibling / 404 no roster) surfaces its message and exits 1,
// and the bearer key rides the Authorization header.

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type rehomeStub struct {
	lastMethod string
	lastPath   string
	lastAuth   string
	lastBody   map[string]string
	status     int
	body       string
}

func (g *rehomeStub) server(t *testing.T) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/fak/account/rehome", func(w http.ResponseWriter, r *http.Request) {
		g.lastMethod = r.Method
		g.lastPath = r.URL.Path
		g.lastAuth = r.Header.Get("Authorization")
		g.lastBody = map[string]string{}
		_ = json.NewDecoder(r.Body).Decode(&g.lastBody)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(g.status)
		_, _ = w.Write([]byte(g.body))
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func TestAccountsRehomeAppliedSwap(t *testing.T) {
	g := &rehomeStub{status: http.StatusOK,
		body: `{"from":"day26","from_email":"a@x.test","to":"july4","to_email":"b@x.test","reason":"operator_rehome"}`}
	srv := g.server(t)

	var out, errb bytes.Buffer
	code := runAccounts(&out, &errb, []string{"rehome", "--addr", srv.URL, "--key", "sekret", "--reason", "capped"})
	if code != 0 {
		t.Fatalf("exit %d, want 0; stderr=%s", code, errb.String())
	}
	if g.lastMethod != http.MethodPost || g.lastPath != "/v1/fak/account/rehome" {
		t.Fatalf("hit %s %s, want POST /v1/fak/account/rehome", g.lastMethod, g.lastPath)
	}
	if g.lastAuth != "Bearer sekret" {
		t.Fatalf("Authorization = %q, want the bearer key", g.lastAuth)
	}
	if g.lastBody["reason"] != "capped" {
		t.Fatalf("posted reason = %q, want capped", g.lastBody["reason"])
	}
	if !strings.Contains(out.String(), "day26 (a@x.test) -> july4 (b@x.test)") {
		t.Fatalf("output %q does not render the from->to swap", out.String())
	}
}

func TestAccountsRehomeRefusalExits1(t *testing.T) {
	g := &rehomeStub{status: http.StatusConflict,
		body: `{"error":{"message":"no available sibling seat","type":"invalid_request_error","code":"account_rehome_unavailable","param":null}}`}
	srv := g.server(t)

	var out, errb bytes.Buffer
	code := runAccounts(&out, &errb, []string{"rehome", "--addr", srv.URL})
	if code != 1 {
		t.Fatalf("exit %d, want 1 on a gateway refusal", code)
	}
	if !strings.Contains(errb.String(), "no available sibling seat") {
		t.Fatalf("stderr %q does not surface the gateway's refusal message", errb.String())
	}
}

func TestAccountsRehomeJSONPassthrough(t *testing.T) {
	g := &rehomeStub{status: http.StatusOK,
		body: `{"from":"a","to":"b","reason":"operator_rehome"}`}
	srv := g.server(t)

	var out, errb bytes.Buffer
	code := runAccounts(&out, &errb, []string{"rehome", "--addr", srv.URL, "--json"})
	if code != 0 {
		t.Fatalf("exit %d, want 0; stderr=%s", code, errb.String())
	}
	var res map[string]any
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("--json output not the raw response: %v (%q)", err, out.String())
	}
	if res["to"] != "b" {
		t.Fatalf("--json output = %v, want the gateway body verbatim", res)
	}
}

func TestAccountsRehomeUnreachableGateway(t *testing.T) {
	var out, errb bytes.Buffer
	// A closed port: the dial fails fast and the CLI must name the likely fix.
	code := runAccounts(&out, &errb, []string{"rehome", "--addr", "http://127.0.0.1:1"})
	if code != 1 {
		t.Fatalf("exit %d, want 1 on an unreachable gateway", code)
	}
	if !strings.Contains(errb.String(), "fak guard") {
		t.Fatalf("stderr %q should point the operator at the guard banner's gateway URL", errb.String())
	}
}
