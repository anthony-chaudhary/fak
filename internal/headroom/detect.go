package headroom

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Presence describes an already-selected Headroom transport. Detection is
// observational only: it never changes ANTHROPIC_BASE_URL or user settings.
type Presence struct {
	InFront bool   `json:"in_front"`
	URL     string `json:"url,omitempty"`
	Variant string `json:"variant,omitempty"` // cli or desktop
	Source  string `json:"source,omitempty"`  // env, settings, or probe
}

// Detector exposes the environmental seams needed for deterministic tests.
type Detector struct {
	Env          func(string) string
	ReadFile     func(string) ([]byte, error)
	HomeDir      func() (string, error)
	Client       *http.Client
	ProbeURLs    []string
	SettingsPath string
}

// Detect reports whether Headroom is already in front of the provider.
func DetectUpstream(ctx context.Context) (Presence, error) { return (Detector{}).Detect(ctx) }

// Detect checks an explicit provider URL, Claude settings, then local health
// endpoints. Missing settings and unreachable probes mean "not detected".
func (d Detector) Detect(ctx context.Context) (Presence, error) {
	getenv := d.Env
	if getenv == nil {
		getenv = os.Getenv
	}
	if raw := strings.TrimSpace(getenv("ANTHROPIC_BASE_URL")); raw != "" {
		if p, ok := headroomURL(raw); ok {
			p.Source = "env"
			return p, nil
		}
	}
	readFile := d.ReadFile
	if readFile == nil {
		readFile = os.ReadFile
	}
	settings := d.SettingsPath
	if settings == "" {
		homeDir := d.HomeDir
		if homeDir == nil {
			homeDir = os.UserHomeDir
		}
		if home, err := homeDir(); err == nil {
			settings = filepath.Join(home, ".claude", "settings.json")
		}
	}
	if settings != "" {
		b, err := readFile(settings)
		if err == nil {
			var v any
			if err := json.Unmarshal(b, &v); err != nil {
				return Presence{}, errors.New("parse Claude settings: " + err.Error())
			}
			text := strings.ToLower(string(b))
			if strings.Contains(text, "headroom-rtk-rewrite.sh") || strings.Contains(text, ":6767") {
				return Presence{InFront: true, URL: "http://127.0.0.1:6767", Variant: "desktop", Source: "settings"}, nil
			}
		} else if !errors.Is(err, os.ErrNotExist) {
			return Presence{}, err
		}
	}
	probes := d.ProbeURLs
	if probes == nil {
		probes = []string{"http://127.0.0.1:8787/healthz", "http://127.0.0.1:6767/healthz"}
	}
	client := d.Client
	if client == nil {
		client = &http.Client{Timeout: 250 * time.Millisecond}
	}
	for _, endpoint := range probes {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
		if err != nil {
			continue
		}
		resp, err := client.Do(req)
		if err != nil {
			continue
		}
		resp.Body.Close()
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			p, ok := headroomURL(endpoint)
			if ok {
				p.Source = "probe"
				return p, nil
			}
		}
	}
	return Presence{}, nil
}

func headroomURL(raw string) (Presence, bool) {
	u, err := url.Parse(raw)
	if err != nil {
		return Presence{}, false
	}
	host := strings.ToLower(u.Hostname())
	port := u.Port()
	if host != "127.0.0.1" && host != "localhost" && host != "::1" {
		return Presence{}, false
	}
	variant := "cli"
	if port == "6767" {
		variant = "desktop"
	} else if port != "8787" {
		return Presence{}, false
	}
	u.Path, u.RawQuery, u.Fragment = "", "", ""
	return Presence{InFront: true, URL: strings.TrimSuffix(u.String(), "/"), Variant: variant}, true
}
