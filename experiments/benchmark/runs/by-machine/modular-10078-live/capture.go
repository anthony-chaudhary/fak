//go:build ignore

// A bounded hardware recipe using fak's canonical sweep engine.
// Endpoint, bearer key, and output directory are private runtime inputs.
package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/webbench"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const expected = "red green blue yellow"

type transport struct {
	session string
	mu      sync.Mutex
	started int
	samples []map[string]any
}
type body struct {
	io.ReadCloser
	buf     bytes.Buffer
	status  int
	ordinal int
	trace   string
	owner   *transport
	once    sync.Once
}

func (b *body) Read(p []byte) (int, error) {
	n, e := b.ReadCloser.Read(p)
	b.buf.Write(p[:n])
	return n, e
}
func (b *body) Close() error {
	e := b.ReadCloser.Close()
	b.once.Do(func() {
		output := ""
		var native json.RawMessage
		var usage json.RawMessage
		scan := bufio.NewScanner(bytes.NewReader(b.buf.Bytes()))
		scan.Buffer(make([]byte, 4096), 1024*1024)
		for scan.Scan() {
			line := strings.TrimSpace(scan.Text())
			if !strings.HasPrefix(line, "data:") {
				continue
			}
			var event map[string]json.RawMessage
			if json.Unmarshal([]byte(strings.TrimSpace(strings.TrimPrefix(line, "data:"))), &event) != nil {
				continue
			}
			var choices []map[string]json.RawMessage
			json.Unmarshal(event["choices"], &choices)
			for _, choice := range choices {
				var delta map[string]string
				json.Unmarshal(choice["delta"], &delta)
				output += delta["content"]
			}
			var ext map[string]json.RawMessage
			json.Unmarshal(event["fak"], &ext)
			if len(ext["native_inference_receipt"]) > 0 {
				native = ext["native_inference_receipt"]
			}
			if len(event["usage"]) > 0 && string(event["usage"]) != "null" {
				usage = event["usage"]
			}
		}
		digest := sha256.Sum256(b.buf.Bytes())
		concurrencies := []int{1, 2, 4, 8, 16, 32}
		s := map[string]any{"request_ordinal": b.ordinal, "concurrency": concurrencies[(b.ordinal-1)/32], "trace_id": b.trace, "http_status": b.status, "output": output, "exact_output": b.status == 200 && strings.TrimSpace(output) == expected, "response_sha256": hex.EncodeToString(digest[:]), "native_receipt": native, "usage": usage}
		b.owner.mu.Lock()
		defer b.owner.mu.Unlock()
		b.owner.samples = append(b.owner.samples, s)
		if len(b.owner.samples)%32 == 0 {
			fmt.Fprintf(os.Stderr, "completed_requests=%d\n", len(b.owner.samples))
		}
	})
	return e
}
func (t *transport) RoundTrip(req *http.Request) (*http.Response, error) {
	t.mu.Lock()
	t.started++
	ordinal := t.started
	t.mu.Unlock()
	trace := fmt.Sprintf("%s-request-%03d", t.session, ordinal)
	r := req.Clone(req.Context())
	r.Header.Set("X-Trace-Id", trace)
	if r.Body != nil {
		raw, e := io.ReadAll(r.Body)
		if e != nil {
			return nil, e
		}
		r.Body.Close()
		var payload map[string]any
		if e = json.Unmarshal(raw, &payload); e != nil {
			return nil, e
		}
		payload["stream_options"] = map[string]any{"include_usage": true}
		raw, e = json.Marshal(payload)
		if e != nil {
			return nil, e
		}
		r.Body = io.NopCloser(bytes.NewReader(raw))
		r.ContentLength = int64(len(raw))
	}
	resp, e := http.DefaultTransport.RoundTrip(r)
	if e == nil {
		resp.Body = &body{ReadCloser: resp.Body, status: resp.StatusCode, ordinal: ordinal, trace: trace, owner: t}
	}
	return resp, e
}
func check(e error) {
	if e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
func write(path string, v any) {
	raw, e := json.MarshalIndent(v, "", "  ")
	check(e)
	check(os.WriteFile(path, append(raw, '\n'), 0600))
}
func main() {
	started := time.Now()
	model := flag.String("model", "", "exact verified model")
	engine := flag.String("engine-receipt", "", "sha256 of sanitized engine/capacity snapshot")
	capacity := flag.Int("capacity", 0, "verified admitted concurrency")
	out := flag.String("out-dir", "", "private capture directory")
	session := flag.String("session", "", "task-owned prefix for independent one-turn agent sessions")
	maxSeconds := flag.Int("max-seconds", 600, "total run deadline")
	flag.Parse()
	if *model == "" || *engine == "" || *capacity <= 0 || *out == "" || *session == "" || os.Getenv("MODULAR_BENCH_ENDPOINT") == "" || os.Getenv("MODULAR_BENCH_API_KEY") == "" {
		check(fmt.Errorf("verified configuration missing"))
	}
	check(os.MkdirAll(*out, 0700))
	var workload []webbench.ServingRequest
	for i := 0; i < 32; i++ {
		workload = append(workload, webbench.ServingRequest{ID: fmt.Sprintf("fixed-%03d", i+1), Messages: []webbench.ChatMessage{{Role: "system", Content: "Follow the instruction exactly."}, {Role: "user", Content: "Return exactly: " + expected}}, MaxOutputTokens: 16})
	}
	write(filepath.Join(*out, "workload.json"), workload)
	tr := &transport{session: *session}
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*maxSeconds)*time.Second)
	defer cancel()
	measurementStart := time.Now()
	report, e := webbench.RunServingSweep(ctx, webbench.ServingSweepConfig{
		GeneratedAt: started.UTC().Format(time.RFC3339), MachineID: "qwen38-native-vulkan-single-gpu", Model: *model,
		Tracks:    []webbench.ServingTrackConfig{{Track: webbench.TrackOurs, BaseURL: os.Getenv("MODULAR_BENCH_ENDPOINT"), Model: *model, APIKeyEnv: "MODULAR_BENCH_API_KEY"}},
		Contracts: map[webbench.ServingTrack]webbench.ServingSweepTrackContract{webbench.TrackOurs: {Track: webbench.TrackOurs, Model: *model, Engine: "fak-native", EngineReceiptDigest: *engine, BatchCapacity: *capacity, CapacitySource: "explicit-admission-budget-and-max-seqs-bound-to-engine-snapshot"}},
		Workload:  workload, Concurrencies: []int{1, 2, 4, 8, 16, 32}, GoodputSLO: 2 * time.Second, Timeout: 60 * time.Second,
		Client: &http.Client{Transport: tr, Timeout: 65 * time.Second},
	})
	tr.mu.Lock()
	write(filepath.Join(*out, "quality.json"), map[string]any{"schema": "fak.modular-capacity-quality/v1", "expected_output": expected, "sample_count": len(tr.samples), "samples": tr.samples, "session_policy": "each independent one-turn agent request has a distinct task-owned trace; no automatic resets or budget changes", "protocol_extensions": map[string]any{"stream_options.include_usage": true}, "runner_setup_seconds": measurementStart.Sub(started).Seconds(), "measurement_seconds": time.Since(measurementStart).Seconds(), "total_runner_seconds": time.Since(started).Seconds(), "max_run_seconds": *maxSeconds, "sla_claim": "none; goodput threshold is 2000 ms"})
	tr.mu.Unlock()
	if report != nil {
		check(webbench.WriteServingSweepReport(report, filepath.Join(*out, "receipt.json")))
	}
	check(e)
	fmt.Fprintf(os.Stderr, "measurement_seconds=%.3f quality_samples=%d\n", time.Since(measurementStart).Seconds(), len(tr.samples))
}
