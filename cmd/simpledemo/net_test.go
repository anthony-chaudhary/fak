package main

import (
	"net/http"
	"os"
	"testing"
	"time"
)

// TestDownloadURLsResolve verifies the external-boundary claim that the auto-download
// path actually makes: that every URL modelDownload() emits for a recommended model
// resolves at the host. This is the network analog of the pathlint witness — it checks
// the claim against ground truth (a real range request) instead of trusting that the
// derivation "looks right". It caught nothing the day it was written only because the
// derivation was already fixed; the dash-form URLs it replaces returned 404.
//
// It hits the network, so it is gated on FAK_NET_TESTS=1 and skipped by default to keep
// `go test ./...` offline and fast. Run it in a network CI lane:
//
//	FAK_NET_TESTS=1 go test ./cmd/simpledemo/ -run TestDownloadURLsResolve -v
func TestDownloadURLsResolve(t *testing.T) {
	if os.Getenv("FAK_NET_TESTS") != "1" {
		t.Skip("set FAK_NET_TESTS=1 to run the download-URL reachability witness (hits the network)")
	}

	// The models simpledemo recommends / auto-downloads. Each must derive a URL that
	// the host serves; a dash-vs-dot or wrong-repo regression turns these red.
	models := []string{
		"Qwen2.5-0.5B-Instruct-Q8_0.gguf",
		"Qwen2.5-1.5B-Instruct-Q8_0.gguf",
		"Qwen2.5-3B-Instruct-Q4_K_M.gguf",
	}

	client := &http.Client{Timeout: 30 * time.Second}
	reachable := func(url string) (int, error) {
		req, err := http.NewRequest(http.MethodGet, url, nil)
		if err != nil {
			return 0, err
		}
		req.Header.Set("Range", "bytes=0-0") // ask for one byte; avoids pulling GBs
		resp, err := client.Do(req)
		if err != nil {
			return 0, err
		}
		defer resp.Body.Close()
		return resp.StatusCode, nil
	}
	ok := func(code int) bool { return code == http.StatusOK || code == http.StatusPartialContent }

	for _, m := range models {
		canonical, urls, derived := modelDownload(m)
		if !derived {
			t.Errorf("modelDownload(%q) failed to derive a URL", m)
			continue
		}
		// The primary host must serve it; the mirror is best-effort (some are flaky).
		code, err := reachable(urls[0])
		if err != nil {
			t.Errorf("%s: GET %s: %v", m, urls[0], err)
			continue
		}
		if !ok(code) {
			t.Errorf("%s: primary URL %s returned %d (canonical=%s) — derivation likely wrong (repo or dash-vs-dot)", m, urls[0], code, canonical)
		} else {
			t.Logf("%s -> %s (%d)", m, urls[0], code)
		}
	}

	// The tokenizer fallback (mradermacher ships none, so this must point at upstream Qwen).
	for _, u := range tokenizerURLs()[:1] {
		code, err := reachable(u)
		if err != nil {
			t.Errorf("tokenizer GET %s: %v", u, err)
			continue
		}
		if !ok(code) {
			t.Errorf("tokenizer URL %s returned %d — fallback host changed", u, code)
		}
	}
}
