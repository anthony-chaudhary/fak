package armbench

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseUsageCacheEvidence(t *testing.T) {
	u := map[string]any{"prompt_tokens": float64(1200), "completion_tokens": float64(20), "prompt_tokens_details": map[string]any{"cached_tokens": float64(1024)}}
	g := parseUsage(u)
	if g.CacheRead != 1024 || g.Input != 1200 {
		t.Fatalf("%+v", g)
	}
}
func TestSixArmColdWarmSummary(t *testing.T) {
	var cs []PassthroughCall
	for _, a := range []string{"direct-normal", "direct-caveman", "fak-passthrough-normal", "fak-passthrough-caveman", "fak-provider-cache-only-normal", "fak-provider-cache-only-caveman"} {
		cs = append(cs, PassthroughCall{Arm: a, Phase: "cold"}, PassthroughCall{Arm: a, Phase: "warm"})
	}
	s := summarizePassthrough(cs)
	if len(s) != 6 {
		t.Fatalf("got %d", len(s))
	}
	for _, x := range s {
		if x.Cold.Calls != 1 || x.Warm.Calls != 1 {
			t.Fatalf("%+v", x)
		}
	}
}
func TestProxyPreservesBody(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var v map[string]any
		json.NewDecoder(r.Body).Decode(&v)
		json.NewEncoder(w).Encode(v)
	}))
	defer up.Close()
	p, e := newBenchProxy(up.URL)
	if e != nil {
		t.Fatal(e)
	}
	defer p.Close()
	resp, e := http.Post(p.server.URL, "application/json", strings.NewReader(`{"x":1}`))
	if e != nil {
		t.Fatal(e)
	}
	defer resp.Body.Close()
}
func TestLivePassthrough(t *testing.T) {
	if os.Getenv("FAK_CAVEMAN_PASSTHROUGH_LIVE") != "1" {
		t.Skip("set live env")
	}
	root := filepath.Join("..", "..")
	_, err := RunCavemanPassthrough(t.Context(), PassthroughOptions{InputDir: filepath.Join(root, "docs", "_witnesses", "armbench-caveman-native", "inputs"), OutDir: filepath.Join(root, "docs", "_witnesses", "armbench-caveman-passthrough", "live-gpt-5.6-sol"), BaseURL: os.Getenv("OPENAI_BASE_URL"), APIKey: os.Getenv("OPENAI_API_KEY"), Model: "gpt-5.6-sol", Label: "live-gpt-5.6-sol", Trials: 3})
	if err != nil {
		t.Fatal(err)
	}
}
