package gateway

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"
)

// a2aPlaceholderHosts are the documentation-reserved names (RFC 2606) that must
// never appear in a SERVED A2A response. They are the failure this test exists to
// catch: they always resolve and never to fak, so a peer that dials one sees a
// timeout or a DNS error it blames on its own configuration rather than on the
// descriptor that lied to it. Fixtures and docs may keep them; a served body may not.
var a2aPlaceholderHosts = []string{"example.com", "example.org", "example.net", "example.edu"}

// scanNoPlaceholderHost fails if any documentation-reserved host survives into a
// served response body. It scans the RAW bytes rather than decoded fields so a
// placeholder that reappears in a field this test does not name still trips it.
func scanNoPlaceholderHost(t *testing.T, what string, raw []byte) {
	t.Helper()
	body := string(raw)
	for _, ph := range a2aPlaceholderHosts {
		if strings.Contains(body, ph) {
			t.Errorf("%s contains placeholder host %q — a served descriptor must name this process:\n%s", what, ph, body)
		}
	}
}

// TestA2AAgentCardEndpointIsLive is the #5642 witness: a gateway bound to an
// EPHEMERAL port must serve an agent card whose advertised endpoint resolves back to
// that same listener, and the endpoint must be dialable — proven here by actually
// dialing it and receiving the A2A route, not by inspecting a string.
//
// It covers BOTH served surfaces that carried the literal: the card's own Endpoint
// (handleA2AGetExtendedAgentCard) and the AgentCardURL stamped onto a created task
// (handleA2ASendMessage). Repairing only the card handler would leave a task record
// still handing peers a host that is not fak.
func TestA2AAgentCardEndpointIsLive(t *testing.T) {
	srv := newTestServer(t) // EngineID "test"

	// Ephemeral port: the address does not exist until the listener does, so nothing
	// built before Serve could have known it.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("bind ephemeral port: %v", err)
	}
	listenAddr := ln.Addr().String()
	base := "http://" + listenAddr

	ctx, cancel := context.WithCancel(context.Background())
	served := make(chan error, 1)
	go func() { served <- srv.Serve(ctx, ln) }()
	t.Cleanup(func() {
		cancel()
		select {
		case err := <-served:
			if err != nil && err != http.ErrServerClosed {
				t.Errorf("Serve returned unexpected error on shutdown: %v", err)
			}
		case <-time.After(6 * time.Second):
			t.Error("Serve did not return within 6s of ctx cancel")
		}
	})

	client := &http.Client{Timeout: 5 * time.Second}
	waitServing(t, client, base+"/healthz")

	// --- 1) The card's own Endpoint ------------------------------------------------
	code, raw := a2aGetRaw(t, client, base+"/a2a/v1/agent-card")
	if code != http.StatusOK {
		t.Fatalf("GET /a2a/v1/agent-card = %d, want 200 (body %s)", code, raw)
	}
	scanNoPlaceholderHost(t, "agent card", raw)

	var card struct {
		ID       string `json:"id"`
		Name     string `json:"name"`
		Endpoint string `json:"endpoint"`
	}
	if err := json.Unmarshal(raw, &card); err != nil {
		t.Fatalf("decode agent card: %v (body %s)", err, raw)
	}

	// The endpoint must name the listener that served it — this is the whole contract.
	if got := mustURLHost(t, "card endpoint", card.Endpoint); got != listenAddr {
		t.Errorf("card endpoint host = %q, want the serving listener %q (endpoint %q)", got, listenAddr, card.Endpoint)
	}

	// Identity reflects the configured engine rather than a fixed literal.
	if card.ID != "fleet-fak-test" {
		t.Errorf("card id = %q, want %q (engine id must reach the card)", card.ID, "fleet-fak-test")
	}
	if !strings.Contains(card.Name, "test") {
		t.Errorf("card name = %q, want it to name the configured engine", card.Name)
	}

	// Dial what the card advertised. A wrong host fails here as a dial error; a right
	// host but a wrong path fails here as a non-200 — both are the bug this closes.
	if code, body := a2aGetRaw(t, client, card.Endpoint+"/agent-card"); code != http.StatusOK {
		t.Fatalf("dialing the advertised endpoint %q/agent-card = %d, want 200 (body %s)", card.Endpoint, code, body)
	}

	// --- 2) The AgentCardURL stamped onto a created task ---------------------------
	taskID := a2aCreateTask(t, client, base)
	code, raw = a2aGetRaw(t, client, base+"/a2a/v1/tasks/"+taskID)
	if code != http.StatusOK {
		t.Fatalf("GET /a2a/v1/tasks/%s = %d, want 200 (body %s)", taskID, code, raw)
	}
	scanNoPlaceholderHost(t, "task record", raw)

	var task struct {
		AgentCardURL string `json:"agent_card_url"`
	}
	if err := json.Unmarshal(raw, &task); err != nil {
		t.Fatalf("decode task: %v (body %s)", err, raw)
	}
	if got := mustURLHost(t, "task agent_card_url", task.AgentCardURL); got != listenAddr {
		t.Errorf("task agent_card_url host = %q, want the serving listener %q (url %q)", got, listenAddr, task.AgentCardURL)
	}
	if code, body := a2aGetRaw(t, client, task.AgentCardURL); code != http.StatusOK {
		t.Fatalf("dialing the task's agent_card_url %q = %d, want 200 (body %s)", task.AgentCardURL, code, body)
	}
}

// TestA2ASelfBaseURLPrecedence pins the derivation order a2aSelfBaseURL documents.
// The live-listener test above proves the common path end to end; this one covers the
// edges that path cannot reach — a request with no authority, a wildcard bind, and a
// hostile Host header — without binding a socket per case.
func TestA2ASelfBaseURLPrecedence(t *testing.T) {
	bound := func(addr string) *Server {
		s := &Server{}
		if addr != "" {
			a := addr
			s.boundAddr.Store(&a)
		}
		return s
	}
	req := func(host string, overTLS bool) *http.Request {
		r, err := http.NewRequest(http.MethodGet, "/a2a/v1/agent-card", nil)
		if err != nil {
			t.Fatalf("build request: %v", err)
		}
		r.Host = host
		if overTLS {
			// A non-nil ConnectionState is exactly what net/http sets on a TLS request.
			r.TLS = &tls.ConnectionState{}
		}
		return r
	}

	cases := []struct {
		name  string
		srv   *Server
		r     *http.Request
		want  string
		about string
	}{
		{"request host wins", bound("127.0.0.1:9999"), req("gw.internal:8080", false), "http://gw.internal:8080",
			"the authority the caller dialed beats the bound socket"},
		{"tls implies https", bound(""), req("gw.internal:8443", true), "https://gw.internal:8443",
			"scheme comes from the connection, never a guess"},
		{"empty host falls back to listener", bound("127.0.0.1:9999"), req("", false), "http://127.0.0.1:9999",
			"an HTTP/1.0 client may omit Host; http:/// is not a descriptor"},
		{"wildcard bind becomes loopback", bound("[::]:8080"), req("", false), "http://127.0.0.1:8080",
			"a wildcard names no reachable host"},
		{"zero bind becomes loopback", bound("0.0.0.0:8080"), req("", false), "http://127.0.0.1:8080",
			"same, in IPv4 spelling"},
		{"hostile host rejected", bound("127.0.0.1:9999"), req("evil/../x", false), "http://127.0.0.1:9999",
			"a Host header is attacker-supplied; a malformed authority must not splice into the URL"},
		{"no host and no listener", bound(""), req("", false), "http://127.0.0.1",
			"loopback is true of every live process; an empty authority is true of none"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.srv.a2aSelfBaseURL(tc.r); got != tc.want {
				t.Errorf("a2aSelfBaseURL = %q, want %q (%s)", got, tc.want, tc.about)
			}
		})
	}

	// Forwarded headers are NOT a trust input: a direct caller must not be able to
	// dictate the authority the gateway advertises back to it.
	spoofed := req("gw.internal:8080", false)
	spoofed.Header.Set("X-Forwarded-Host", "attacker.invalid")
	spoofed.Header.Set("X-Forwarded-Proto", "https")
	if got := bound("").a2aSelfBaseURL(spoofed); got != "http://gw.internal:8080" {
		t.Errorf("a2aSelfBaseURL honored a forwarded header: got %q, want %q", got, "http://gw.internal:8080")
	}
}

// TestA2AIdentityFallsBackWhenUnconfigured keeps the unconfigured card byte-identical
// to what peers see today: only a gateway that WAS given an engine identity changes.
func TestA2AIdentityFallsBackWhenUnconfigured(t *testing.T) {
	id, name := (&Server{}).a2aIdentity()
	if id != "fleet-fak" || name != "Fleet fak Agent" {
		t.Errorf("unconfigured identity = (%q, %q), want the historical (%q, %q)", id, name, "fleet-fak", "Fleet fak Agent")
	}
	withEngine := &Server{engineID: "gpu-0"}
	if id, _ := withEngine.a2aIdentity(); id != "fleet-fak-gpu-0" {
		t.Errorf("configured identity id = %q, want %q", id, "fleet-fak-gpu-0")
	}
}

// a2aGetRaw issues an authenticated A2A GET and returns the status and RAW body, so a
// caller can scan bytes the decoded struct would drop.
func a2aGetRaw(t *testing.T, client *http.Client, target string) (int, []byte) {
	t.Helper()
	req, err := http.NewRequest(http.MethodGet, target, nil)
	if err != nil {
		t.Fatalf("build GET %s: %v", target, err)
	}
	req.Header.Set("X-Caller-ID", "peer-under-test")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("GET %s: %v (an unreachable advertised host fails HERE)", target, err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body of %s: %v", target, err)
	}
	return resp.StatusCode, raw
}

// a2aCreateTask posts one registry-valid A2A message and returns the created task id.
func a2aCreateTask(t *testing.T, client *http.Client, base string) string {
	t.Helper()
	body := `{"message_id":"msg-5642","from":"peer-under-test","to":"fleet-agent","content":{"method":"agent.ping"}}`
	req, err := http.NewRequest(http.MethodPost, base+"/a2a/v1/messages", strings.NewReader(body))
	if err != nil {
		t.Fatalf("build POST: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Caller-ID", "peer-under-test")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("POST /a2a/v1/messages: %v", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("POST /a2a/v1/messages = %d, want 201 (body %s)", resp.StatusCode, raw)
	}
	var created struct {
		TaskID string `json:"task_id"`
	}
	if err := json.Unmarshal(raw, &created); err != nil || created.TaskID == "" {
		t.Fatalf("decode created task: %v (body %s)", err, raw)
	}
	return created.TaskID
}

// mustURLHost parses an advertised URL and returns its authority, failing the test if
// the descriptor is not a parseable absolute URL at all.
func mustURLHost(t *testing.T, what, raw string) string {
	t.Helper()
	u, err := url.Parse(raw)
	if err != nil {
		t.Fatalf("%s %q is not a parseable URL: %v", what, raw, err)
	}
	if u.Scheme == "" || u.Host == "" {
		t.Fatalf("%s %q is not an absolute URL a peer could dial", what, raw)
	}
	return u.Host
}
