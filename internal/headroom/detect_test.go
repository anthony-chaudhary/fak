package headroom

import (
	"context"
	"errors"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestDetectSources(t *testing.T) {
	tests := []struct {
		name string
		d    Detector
		want Presence
	}{
		{"env-cli", Detector{Env: func(string) string { return "http://localhost:8787/v1" }, ReadFile: func(string) ([]byte, error) { return nil, errors.New("must not read") }}, Presence{InFront: true, URL: "http://localhost:8787", Variant: "cli", Source: "env"}},
		{"desktop-settings", Detector{Env: func(string) string { return "" }, SettingsPath: "settings", ReadFile: func(string) ([]byte, error) {
			return []byte(`{"hooks":["~/.claude/hooks/headroom-rtk-rewrite.sh"]}`), nil
		}, ProbeURLs: []string{}}, Presence{InFront: true, URL: "http://127.0.0.1:6767", Variant: "desktop", Source: "settings"}},
		{"probe", Detector{Env: func(string) string { return "" }, SettingsPath: "missing", ReadFile: func(string) ([]byte, error) { return nil, os.ErrNotExist }, ProbeURLs: []string{"http://127.0.0.1:8787/healthz"}, Client: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader("ok"))}, nil
		})}}, Presence{InFront: true, URL: "http://127.0.0.1:8787", Variant: "cli", Source: "probe"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.d.Detect(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Fatalf("got %#v want %#v", got, tc.want)
			}
		})
	}
}

func TestDetectDoesNotGuessRemoteProxy(t *testing.T) {
	got, err := (Detector{Env: func(string) string { return "https://proxy.example/v1" }, SettingsPath: "missing", ReadFile: func(string) ([]byte, error) { return nil, os.ErrNotExist }, ProbeURLs: []string{}}).Detect(context.Background())
	if err != nil || got.InFront {
		t.Fatalf("got=%#v err=%v", got, err)
	}
}

func TestNativeDefersWhenHeadroomAlreadyInFront(t *testing.T) {
	in := Input{Bytes: []byte(strings.Repeat("duplicate line\n", 20)), Tool: "go test", UpstreamHeadroom: &Presence{InFront: true, URL: "http://localhost:8787", Variant: "cli", Source: "env"}}
	out, err := nativeCompressor{detectUpstream: func(context.Context) (Presence, error) { return *in.UpstreamHeadroom, nil }}.Compress(context.Background(), Input{Bytes: in.Bytes, Tool: in.Tool})
	if err != nil {
		t.Fatal(err)
	}
	if string(out.Bytes) != string(in.Bytes) || out.Status != "inert" || !strings.Contains(out.Reason, "double-compress") {
		t.Fatalf("out=%#v", out)
	}
}
