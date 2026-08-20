package harnessweb

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestAdversarialPublicGatewayURL(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want string
	}{
		{name: "empty"},
		{name: "script scheme", raw: "javascript:alert(1)"},
		{name: "scheme relative", raw: "//example.test/debug/vars"},
		{name: "missing host", raw: "https:///debug/vars"},
		{name: "control character", raw: "https://example.test/\nprivate"},
		{name: "credentials query and fragment removed", raw: "https://user:secret@example.test:8443/base/?token=private#part", want: "https://example.test:8443/base"},
		{name: "surrounding space removed", raw: "  http://127.0.0.1:8080/  ", want: "http://127.0.0.1:8080"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := publicGatewayURL(tt.raw); got != tt.want {
				t.Fatalf("publicGatewayURL(%q)=%q want %q", tt.raw, got, tt.want)
			}
		})
	}
}

func TestEdgeGatewayOverviewFailsClosed(t *testing.T) {
	hostile := `<script>document.location="https://attacker.invalid"</script>`
	tests := []struct {
		name          string
		status        int
		body          string
		wantReachable bool
		wantSessions  int
	}{
		{name: "empty document", status: http.StatusOK, body: `{}`, wantReachable: true},
		{name: "upstream error", status: http.StatusBadGateway, body: `provider down`},
		{name: "malformed document", status: http.StatusOK, body: `{"sessions":`},
		{name: "oversized document", status: http.StatusOK, body: `{"sessions":[{"trace_id":"` + strings.Repeat("x", (2<<20)+64) + `"}]}`},
		{name: "hostile session text remains data", status: http.StatusOK, body: fmt.Sprintf(`{"sessions":[{"trace_id":%q}],"fleet":{"machines":1,"sessions":1}}`, hostile), wantReachable: true, wantSessions: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tt.status)
				_, _ = io.WriteString(w, tt.body)
			}))
			defer gateway.Close()

			adapter := &liveAdapter{baseURL: gateway.URL, client: gateway.Client()}
			overview, sessions := adapter.overview(context.Background())
			if overview.Reachable != tt.wantReachable || len(sessions) != tt.wantSessions {
				t.Fatalf("overview=%+v sessions=%d", overview, len(sessions))
			}
			links := dashboardLinks(overview.URL)
			if len(links) == 0 {
				t.Fatal("dashboard links are empty")
			}
			for _, link := range links {
				if tt.wantReachable && link.URL == "" {
					t.Fatalf("reachable gateway left %q disabled", link.Label)
				}
				if !tt.wantReachable && link.URL != "" {
					t.Fatalf("unreachable gateway exposed %q as %q", link.Label, link.URL)
				}
			}
			if tt.name == "hostile session text remains data" {
				response := httptest.NewRecorder()
				request := httptest.NewRequest(http.MethodGet, "/api/status", nil)
				handlerWithLive(newStore(), adapter).ServeHTTP(response, request)
				if strings.Contains(response.Body.String(), "<script>") || !strings.Contains(response.Body.String(), `\u003cscript\u003e`) {
					t.Fatalf("status JSON did not HTML-escape hostile text: %s", response.Body.String())
				}
			}
		})
	}
}

func TestEdgeCapturedHomeKeepsDashboardAndRefreshEnvelope(t *testing.T) {
	links := dashboardLinks("")
	wantPaths := []string{"/", "/healthz", "/v1/fak/sessions", "/v1/fak/loops", "/v1/fak/fleet", "/v1/fak/tasks", "/metrics", "/debug/vars"}
	if len(links) != len(wantPaths) {
		t.Fatalf("dashboard links=%d want %d", len(links), len(wantPaths))
	}
	for i, want := range wantPaths {
		if links[i].Path != want {
			t.Fatalf("dashboard %d path=%q want %q", i, links[i].Path, want)
		}
	}
	for _, want := range []string{
		`setInterval(refreshOverview,5000)`,
		`@media(max-width:600px)`,
		`strong.textContent=item.label`,
		`desc.textContent=item.description`,
		`path.textContent=item.path`,
	} {
		if !strings.Contains(page, want) {
			t.Fatalf("captured home missing %q", want)
		}
	}
}
