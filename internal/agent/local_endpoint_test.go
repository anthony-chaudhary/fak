package agent

import (
	"strings"
	"testing"
)

func TestIsLocalEndpoint(t *testing.T) {
	cases := []struct {
		url  string
		want bool
	}{
		{"http://127.0.0.1:18560/v1", true},
		{"http://localhost:8080", true},
		{"http://LOCALHOST:8080/v1", true},
		{"http://gpu.localhost/v1", true},
		{"http://[::1]:8080/v1", true},
		{"http://10.0.0.5/v1", true},
		{"http://172.16.4.2/v1", true},
		{"http://192.168.1.149:18560/v1", true},
		{"http://[fd00::1]/v1", true},
		{"http://169.254.10.1/v1", true},
		{"http://[fe80::1]/v1", true},
		{"http://100.64.0.1/v1", true},
		{"http://100.127.255.254/v1", true},
		{"http://aegis2:18560/v1", true},
		{"http://box.local/v1", true},
		{"http://box.lan/v1", true},
		{"http://svc.internal/v1", true},
		{"http://router.home.arpa/v1", true},
		{"https://api.openai.com/v1", false},
		{"https://openrouter.ai/api/v1", false},
		{"http://8.8.8.8/v1", false},
		{"http://100.128.0.1/v1", false},
		{"http://172.32.0.1/v1", false},
		{"https://example.com.local.evil.com/v1", false},
		{"", false},
		{"   ", false},
		{"://bad", false},
		{"/v1/chat", false},
		{"http:///v1", false},
	}
	for _, tc := range cases {
		if got := IsLocalEndpoint(tc.url); got != tc.want {
			t.Errorf("IsLocalEndpoint(%q) = %v, want %v", tc.url, got, tc.want)
		}
	}
}

func TestTraceLogEventsFeedRender(t *testing.T) {
	var l TraceLog
	if got := l.Events(); got == nil {
		t.Fatal("Events() returned nil sink")
	}
	sink := l.Events()
	*sink = append(*sink, traceEvent{Turn: 1, Arm: "fak", Tool: "read_file", RawArgs: `{"path":"a.go"}`, Verdict: "ALLOW"})
	if len(l) != 1 {
		t.Fatalf("append through Events() not visible on TraceLog: len=%d", len(l))
	}
	out := string(l.Render())
	if !strings.Contains(out, "read_file") || !strings.Contains(out, "verdict=ALLOW") {
		t.Fatalf("Render missing event: %q", out)
	}
}
