package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const dispatchProviderProbeTimeout = 3 * time.Second

var dispatchProviderProbeClient = &http.Client{Timeout: dispatchProviderProbeTimeout}

func dispatchProviderReachabilityCheck(launchCommand []string) map[string]any {
	baseURL := dispatchProviderURL(launchCommand)
	if baseURL == "" {
		return map[string]any{
			"id":          "provider_reachability",
			"evaluated":   false,
			"reason":      "launch has no guarded provider base URL",
			"next_action": "configure the guarded provider route before relying on this check",
		}
	}
	endpoint, err := dispatchDeepHealthURL(baseURL)
	if err != nil {
		return map[string]any{"id": "provider_reachability", "evaluated": true, "ok": false, "reason": "invalid guarded provider URL: " + err.Error()}
	}
	ctx, cancel := context.WithTimeout(context.Background(), dispatchProviderProbeTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return map[string]any{"id": "provider_reachability", "evaluated": true, "ok": false, "reason": err.Error()}
	}
	started := time.Now()
	resp, err := dispatchProviderProbeClient.Do(req)
	result := map[string]any{
		"id":         "provider_reachability",
		"evaluated":  true,
		"endpoint":   dispatchPublicEndpoint(endpoint),
		"elapsed_ms": time.Since(started).Milliseconds(),
	}
	if err != nil {
		result["ok"] = false
		result["reason"] = "guarded provider route unreachable: " + err.Error()
		result["next_action"] = "start or repair the configured fak gateway/upstream, then retry dispatch"
		return result
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	result["status"] = resp.StatusCode
	var health struct {
		OK           bool `json:"ok"`
		Reachability struct {
			Evaluated bool   `json:"evaluated"`
			OK        bool   `json:"ok"`
			Status    int    `json:"status"`
			Error     string `json:"error"`
		} `json:"provider_reachability"`
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		result["ok"] = false
		result["reason"] = fmt.Sprintf("guarded provider health returned HTTP %d", resp.StatusCode)
		return result
	}
	if err := json.Unmarshal(body, &health); err != nil {
		result["ok"] = false
		result["reason"] = "guarded provider health returned malformed JSON"
		return result
	}
	if !health.Reachability.Evaluated {
		result["ok"] = false
		result["reason"] = "guarded provider did not evaluate its upstream hop"
		return result
	}
	result["upstream_status"] = health.Reachability.Status
	if !health.OK || !health.Reachability.OK {
		result["ok"] = false
		result["reason"] = firstString(health.Reachability.Error, "guarded provider upstream is not ready")
		result["next_action"] = "repair the upstream behind the live fak gateway, then retry dispatch"
		return result
	}
	result["ok"] = true
	result["reason"] = "guarded provider route and upstream answered without a model turn"
	return result
}

func dispatchProviderURL(command []string) string {
	for i := 0; i+1 < len(command); i++ {
		if command[i] == "--base-url" {
			return strings.TrimSpace(command[i+1])
		}
	}
	return ""
}

func dispatchDeepHealthURL(baseURL string) (string, error) {
	u, err := url.Parse(baseURL)
	if err != nil {
		return "", err
	}
	if u.Scheme != "http" && u.Scheme != "https" || u.Host == "" {
		return "", fmt.Errorf("want an absolute http(s) URL")
	}
	u.Path = strings.TrimSuffix(u.Path, "/")
	if strings.HasSuffix(u.Path, "/v1") {
		u.Path = strings.TrimSuffix(u.Path, "/v1")
	}
	u.Path += "/healthz"
	q := u.Query()
	q.Set("deep", "1")
	u.RawQuery = q.Encode()
	return u.String(), nil
}

func dispatchPublicEndpoint(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	u.User = nil
	u.RawQuery = ""
	u.Fragment = ""
	return u.String()
}
