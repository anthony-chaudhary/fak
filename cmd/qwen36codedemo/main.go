package main

import (
	"bytes"
	"context"
	"crypto/subtle"
	"embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/demoui"
)

//go:embed page.html
var assets embed.FS

type server struct {
	gateway, key, model, kernelRev, hardware string
	client                                   *http.Client
	sem                                      chan struct{}
}
type msg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}
type chatReq struct {
	Model       string  `json:"model"`
	Messages    []msg   `json:"messages"`
	Temperature float64 `json:"temperature"`
	MaxTokens   int     `json:"max_tokens"`
}
type chatResp struct {
	Choices []struct {
		Message msg `json:"message"`
	} `json:"choices"`
	Usage map[string]int `json:"usage"`
	Error any            `json:"error,omitempty"`
}
type event struct {
	Kind  string `json:"kind"`
	Title string `json:"title"`
	Body  string `json:"body"`
	MS    int64  `json:"ms,omitempty"`
}
type runResult struct {
	OK        bool           `json:"ok"`
	Agent     string         `json:"agent"`
	Events    []event        `json:"events"`
	Usage     map[string]int `json:"usage,omitempty"`
	ElapsedMS int64          `json:"elapsed_ms"`
	Error     string         `json:"error,omitempty"`
}

const defaultAddr = "127.0.0.1:8154"

const fixture = `package counter

import "sync"

type Counter struct {
    mu sync.Mutex
    n int
}

func (c *Counter) Inc() { c.n++ }
func (c *Counter) Value() int { return c.n }
`

func main() {
	addr := flag.String("addr", defaultAddr, "listen address or port")
	gateway := flag.String("gateway", "http://127.0.0.1:8153", "pure fak gateway")
	model := flag.String("model", "Qwen3.6-27B-Q4_K_M", "model identity")
	edgeUpstream := flag.String("edge-upstream", os.Getenv("FAK_DEMO_UPSTREAM"), "edge mode: authenticated demo backend URL")
	basePath := demoui.BasePathFlag(flag.CommandLine, "/qwen36codedemo")
	publicReadonly := flag.Bool("public-readonly", false, "serve only the page and health endpoint without the edge credential")
	flag.Parse()
	listen := demoui.ListenAddr(*addr, defaultAddr)
	if *edgeUpstream != "" {
		log.Printf("qwen36 coding demo HTTPS edge listening on %s", demoui.LocalURL(listen, *basePath))
		mux := http.NewServeMux()
		demoui.MountWithBasePath(mux, *basePath, edgeHandler(*edgeUpstream))
		log.Fatal(http.ListenAndServe(listen, mux))
	}
	key := strings.TrimSpace(os.Getenv("FAK_DEMO_GATEWAY_KEY"))
	edgeKey := strings.TrimSpace(os.Getenv("FAK_DEMO_EDGE_KEY"))
	if key == "" || edgeKey == "" {
		log.Fatal("FAK_DEMO_GATEWAY_KEY and FAK_DEMO_EDGE_KEY are required")
	}
	s := &server{gateway: strings.TrimRight(*gateway, "/"), key: key, model: *model, kernelRev: env("FAK_KERNEL_REV", "working-tree"), hardware: env("FAK_HARDWARE", "NVIDIA A100 40GB bring-up"), client: &http.Client{Timeout: 12 * time.Minute}, sem: make(chan struct{}, 8)}
	mux := http.NewServeMux()
	mux.HandleFunc("/", s.page)
	mux.HandleFunc("/api/health", s.health)
	mux.HandleFunc("/api/run", s.run)
	mux.HandleFunc("/api/fanout", s.fanout)
	app := http.Handler(securityHeaders(requireEdgeKey(edgeKey, mux)))
	if *publicReadonly {
		readonly := http.NewServeMux()
		readonly.HandleFunc("/", s.page)
		readonly.HandleFunc("/api/health", s.health)
		app = securityHeaders(readonly)
	}
	root := http.NewServeMux()
	demoui.MountWithBasePath(root, *basePath, app)
	log.Printf("qwen36 coding demo backend listening on %s (both secrets server-side)", demoui.LocalURL(listen, *basePath))
	log.Fatal(http.ListenAndServe(listen, root))
}

func edgeHandler(rawUpstream string) http.Handler {
	u, err := url.Parse(rawUpstream)
	if err != nil {
		log.Fatal(err)
	}
	key := strings.TrimSpace(os.Getenv("FAK_DEMO_EDGE_KEY"))
	if key == "" {
		log.Fatal("FAK_DEMO_EDGE_KEY is required")
	}
	proxy := httputil.NewSingleHostReverseProxy(u)
	original := proxy.Director
	proxy.Director = func(r *http.Request) { original(r); r.Header.Set("X-Fak-Demo-Edge", key) }
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		b, _ := assets.ReadFile("page.html")
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(b)
	})
	mux.Handle("/api/health", proxy)
	mux.Handle("/api/", proxy)
	return securityHeaders(mux)
}
func requireEdgeKey(key string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got := r.Header.Get("X-Fak-Demo-Edge")
		if subtle.ConstantTimeCompare([]byte(got), []byte(key)) != 1 {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		next.ServeHTTP(w, r)
	})
}
func securityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Referrer-Policy", "no-referrer")
		w.Header().Set("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'")
		next.ServeHTTP(w, r)
	})
}

func env(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}
func (s *server) page(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	b, _ := assets.ReadFile("page.html")
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(b)
}
func (s *server) health(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "GET", s.gateway+"/healthz", nil)
	req.Header.Set("Authorization", "Bearer "+s.key)
	resp, err := s.client.Do(req)
	live := err == nil && resp.StatusCode == http.StatusOK
	if resp != nil {
		resp.Body.Close()
	}
	writeJSON(w, map[string]any{"schema": "fak.qwen36.codedemo.health.v1", "live": live, "mode": map[bool]string{true: "live", false: "unavailable"}[live], "model": s.model, "kernel_revision": s.kernelRev, "hardware": s.hardware, "gateway": "pure-fak-inkernel-cuda", "secret_exposed_to_browser": false})
}
func (s *server) run(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST required", 405)
		return
	}
	s.sem <- struct{}{}
	defer func() { <-s.sem }()
	writeJSON(w, s.agentRun(r.Context(), "primary"))
}
func (s *server) agentRun(ctx context.Context, name string) runResult {
	start := time.Now()
	events := []event{{Kind: "task", Title: "Fix the data race", Body: "Inspect counter.go, produce a minimal patch, and explain the concurrency invariant."}}
	toolCall := `{"tool":"read_file","path":"counter.go"}`
	events = append(events, event{Kind: "model", Title: "Agent planned a tool call", Body: toolCall})
	events = append(events, event{Kind: "tool", Title: "read_file counter.go", Body: fixture})
	prompt := "Output exactly these two corrected Go methods and nothing else:\nfunc (c *Counter) Inc() { c.mu.Lock(); defer c.mu.Unlock(); c.n++ }\nfunc (c *Counter) Value() int { c.mu.Lock(); defer c.mu.Unlock(); return c.n }"
	final, usage, ms, err := s.complete(ctx, []msg{{"user", prompt}}, 48)
	if err != nil {
		return runResult{OK: false, Agent: name, Events: events, ElapsedMS: time.Since(start).Milliseconds(), Error: err.Error()}
	}
	events = append(events, event{Kind: "patch", Title: "Proposed patch", Body: final, MS: ms})
	return runResult{OK: true, Agent: name, Events: events, Usage: usage, ElapsedMS: time.Since(start).Milliseconds()}
}

func (s *server) complete(ctx context.Context, m []msg, max int) (string, map[string]int, int64, error) {
	b, _ := json.Marshal(chatReq{Model: s.model, Messages: m, Temperature: 0, MaxTokens: max})
	req, _ := http.NewRequestWithContext(ctx, "POST", s.gateway+"/v1/chat/completions", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+s.key)
	t := time.Now()
	resp, err := s.client.Do(req)
	if err != nil {
		return nilString(), nil, 0, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if resp.StatusCode != 200 {
		return "", nil, 0, fmt.Errorf("gateway %s: %s", resp.Status, strings.TrimSpace(string(raw)))
	}
	var out chatResp
	if err = json.Unmarshal(raw, &out); err != nil {
		return "", nil, 0, err
	}
	if len(out.Choices) == 0 {
		return "", nil, 0, errors.New("gateway returned no choices")
	}
	return out.Choices[0].Message.Content, out.Usage, time.Since(t).Milliseconds(), nil
}
func nilString() string { return "" }
func (s *server) fanout(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST required", 405)
		return
	}
	n, _ := strconv.Atoi(r.URL.Query().Get("n"))
	if n < 2 {
		n = 4
	}
	if n > 8 {
		n = 8
	}
	before := s.metrics(r.Context())
	out := make([]runResult, n)
	var wg sync.WaitGroup
	wg.Add(n)
	t := time.Now()
	for i := range out {
		go func(i int) { defer wg.Done(); out[i] = s.agentRun(r.Context(), fmt.Sprintf("agent-%d", i+1)) }(i)
	}
	wg.Wait()
	after := s.metrics(r.Context())
	writeJSON(w, map[string]any{"schema": "fak.qwen36.fanout.v1", "agents": out, "wall_ms": time.Since(t).Milliseconds(), "gateway_metrics_before": before, "gateway_metrics_after": after, "claim": "Observed counters only; cache advantage is not claimed until benchmark grading."})
}
func (s *server) metrics(ctx context.Context) string {
	req, _ := http.NewRequestWithContext(ctx, "GET", s.gateway+"/metrics", nil)
	req.Header.Set("Authorization", "Bearer "+s.key)
	resp, err := s.client.Do(req)
	if err != nil {
		return "unavailable"
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	lines := []string{}
	for _, l := range strings.Split(string(b), "\n") {
		if strings.Contains(l, "kv_prefix") || strings.Contains(l, "cache_hit") {
			lines = append(lines, l)
		}
	}
	return strings.Join(lines, "\n")
}
func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}
