package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentxbench"
)

func cmdAgentX(argv []string) {
	os.Exit(runAgentX(os.Stdout, os.Stderr, argv))
}

func runAgentX(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak agentx", flag.ContinueOnError)
	fs.SetOutput(stderr)

	endpoint := fs.String("endpoint", envOrDefault("FAK_AGENTX_ENDPOINT", "http://127.0.0.1:8155/v1"), "OpenAI-compatible chat/completions endpoint")
	model := fs.String("model", "Qwen3.8-27B-Q4_K_M", "model identifier")
	engine := fs.String("engine", "", "engine the endpoint serves (required for a live run)")
	hardware := fs.String("hardware", "", "hardware the endpoint runs on (required for a live run)")
	agents := fs.Int("agents", 5, "number of concurrent agent sessions")
	turns := fs.Int("turns", 3, "number of sequential turns per agent")
	maxTokens := fs.Int("max-tokens", 32, "maximum tokens per turn")
	temp := fs.Float64("temperature", 0.0, "sampling temperature")
	prefix := fs.String("prefix", "", "shared master goal prompt prefix")
	outPath := fs.String("out", "", "output receipt JSON path")
	selfcheck := fs.Bool("selfcheck", false, "execute hermetic self-check without external endpoint (SIMULATED, never evidence)")
	validateFile := fs.String("validate", "", "admit an existing receipt JSON file as evidence (refuses mock and legacy receipts)")
	jsonOutput := fs.Bool("json", false, "output JSON receipt to stdout")
	prefillSlots := fs.Int("prefill-slots", 0, "bound on concurrent in-flight requests during prefill; each request holds one slot until its first content token arrives (live runs only); 0 disables the gate")

	if err := fs.Parse(argv); err != nil {
		return 2
	}

	if *prefillSlots < 0 {
		fmt.Fprintf(stderr, "fak agentx: --prefill-slots must be >= 0\n")
		return 2
	}

	if *validateFile != "" {
		data, err := os.ReadFile(*validateFile)
		if err != nil {
			fmt.Fprintf(stderr, "fak agentx: read file %s: %v\n", *validateFile, err)
			return 1
		}
		var receipt agentxbench.AgentXReceipt
		if err := json.Unmarshal(data, &receipt); err != nil {
			fmt.Fprintf(stderr, "fak agentx: unmarshal receipt: %v\n", err)
			return 1
		}
		errs := agentxbench.EvidenceErrors(&receipt)
		if len(errs) > 0 {
			fmt.Fprintf(stderr, "FAIL: %d validation error(s):\n", len(errs))
			for _, e := range errs {
				fmt.Fprintf(stderr, "  - %s\n", e)
			}
			return 1
		}
		fmt.Fprintf(stdout, "PASS: receipt %s verified (%s %s on %s, %d requests, token counts %s, prefix speedup: %s; synthetic workload, not InferenceX-comparable)\n",
			receipt.BenchmarkID, receipt.ExecutionMode, receipt.Model, receipt.Hardware, receipt.Aggregated.TotalRequests,
			receipt.TokenCountSource, speedupText(receipt.Aggregated.PrefixSpeedupRatio))
		return 0
	}

	cfg := agentxbench.Config{
		EndpointURL:       *endpoint,
		Model:             *model,
		Engine:            *engine,
		Hardware:          *hardware,
		AgentCount:        *agents,
		TurnsPerAgent:     *turns,
		MaxTokens:         *maxTokens,
		Temperature:       *temp,
		SharedPrefix:      *prefix,
		DeterministicSeed: time.Now().UnixNano(),
		MockExecution:     *selfcheck,
	}

	runner := agentxbench.NewRunner(prefillClient(*prefillSlots))
	receipt, err := runner.Run(context.Background(), cfg)
	if err != nil {
		fmt.Fprintf(stderr, "fak agentx: execution failed: %v\n", err)
		return 1
	}

	if *outPath != "" {
		data, err := json.MarshalIndent(receipt, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "fak agentx: marshal receipt: %v\n", err)
			return 1
		}
		if err := os.WriteFile(*outPath, data, 0644); err != nil {
			fmt.Fprintf(stderr, "fak agentx: write output %s: %v\n", *outPath, err)
			return 1
		}
	}

	if *jsonOutput {
		data, _ := json.MarshalIndent(receipt, "", "  ")
		fmt.Fprintln(stdout, string(data))
	} else {
		fmt.Fprintf(stdout, "AgentX Benchmark Result: %s\n", receipt.BenchmarkID)
		fmt.Fprintf(stdout, "Mode / Workload: %s / %s (not InferenceX-comparable)\n", receipt.ExecutionMode, receipt.Workload)
		fmt.Fprintf(stdout, "Model / Engine : %s / %s\n", receipt.Model, receipt.Engine)
		fmt.Fprintf(stdout, "Hardware       : %s\n", receipt.Hardware)
		fmt.Fprintf(stdout, "Concurrency    : %d agents x %d turns (%d total requests)\n",
			receipt.AgentCount, receipt.TurnsPerAgent, receipt.Aggregated.TotalRequests)
		if *prefillSlots > 0 {
			fmt.Fprintf(stdout, "Prefill Gate   : %d slots (live runs only)\n", *prefillSlots)
		}
		fmt.Fprintf(stdout, "Success Rate   : %.1f%% (%d/%d)\n",
			receipt.Aggregated.SuccessRate*100.0, receipt.Aggregated.SuccessfulRequests, receipt.Aggregated.TotalRequests)
		fmt.Fprintf(stdout, "Cold TTFT Mean : %.2f ms\n", receipt.Aggregated.ColdTTFTMeanMS)
		fmt.Fprintf(stdout, "Warm TTFT Mean : %.2f ms\n", receipt.Aggregated.WarmTTFTMeanMS)
		fmt.Fprintf(stdout, "Prefix Speedup : %s\n", speedupText(receipt.Aggregated.PrefixSpeedupRatio))
		fmt.Fprintf(stdout, "ITL p50 / p95  : %.2f ms / %.2f ms\n",
			receipt.Aggregated.ITLP50MS, receipt.Aggregated.ITLP95MS)
		fmt.Fprintf(stdout, "Interactivity  : %.2f tok/s\n", receipt.Aggregated.NormalizedInteractivity)
		fmt.Fprintf(stdout, "Throughput     : %.2f req/s (%.2f out tok/s, %.2f cluster tok/s)\n",
			receipt.Aggregated.RequestThroughputPerSec,
			receipt.Aggregated.OutputTokenThroughputPerSec,
			receipt.Aggregated.ClusterTokenThroughputPerSec)
		fmt.Fprintf(stdout, "Status         : %s\n", receipt.ValidationStatus)
	}

	want := agentxbench.StatusVerifiedPass
	if *selfcheck {
		want = agentxbench.StatusSimulated
	}
	if receipt.ValidationStatus != want {
		return 1
	}
	return 0
}

// prefillGate bounds how many chat/completions requests may be in flight at
// once while their prompt is being processed. It adapts ai-dynamo/aiperf
// candidate aiperf-data-2 ("first-token callback releases prefill
// concurrency"): a request holds one slot from dispatch until its first
// content token arrives — the point at which the GPU-intensive prefill phase
// has effectively finished and the next queued prompt may enter. Streams that
// end without a content frame release the slot on the return path instead.
//
// ADAPT (Apache-2.0; re-expressed idiomatically in Go, no code copied):
//   - ai-dynamo/aiperf src/aiperf/transports/base_transports.py:27@ab610add
//     (FirstTokenCallback contract: fire with ttft_ns until the callback
//     reports true; "used to release prefill concurrency")
//   - ai-dynamo/aiperf src/aiperf/transports/aiohttp_client.py:259@ab610add
//     (separate fast path when no callback is registered)
//   - ai-dynamo/aiperf src/aiperf/credit/callback_handler.py:424@ab610add
//     (return path: prefill release if TTFT never arrived)
type prefillGate struct {
	slots chan struct{}
}

func newPrefillGate(n int) *prefillGate {
	if n <= 0 {
		return nil
	}
	return &prefillGate{slots: make(chan struct{}, n)}
}

// hold reserves one prefill slot, surfacing ctx's error if the context is
// canceled or expires while waiting for a free slot.
func (g *prefillGate) hold(ctx context.Context) (*prefillHold, error) {
	select {
	case g.slots <- struct{}{}:
		return &prefillHold{gate: g}, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// prefillHold is one reserved prefill slot. Release frees it exactly once so
// a double release can never hand the same slot out twice.
type prefillHold struct {
	gate *prefillGate
	once sync.Once
}

// release frees the held slot; subsequent calls are no-ops.
func (h *prefillHold) release() {
	h.once.Do(func() { <-h.gate.slots })
}

// firstTokenFrame is the minimum of an OpenAI-compatible stream chunk that
// carries generated content, mirroring internal/agentxbench's TTFT
// definition: role-only and usage-only frames carry no token.
type firstTokenFrame struct {
	Choices []struct {
		Delta struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"delta"`
	} `json:"choices"`
}

func (f firstTokenFrame) hasContent() bool {
	for _, c := range f.Choices {
		if c.Delta.Content != "" || c.Delta.ReasoningContent != "" {
			return true
		}
	}
	return false
}

// firstTokenBody passes a response body through unchanged and offers the
// first-token callback each content frame, with the TTFT in nanoseconds
// measured from dispatch plus the raw frame — the aiperf FirstTokenCallback
// contract: fire until the callback reports true ("meaningful content"). Once
// it does, no further inspection happens; aiperf likewise runs a separate
// fast path when no callback is registered at all. A stream that ends
// without an acknowledged content frame still fires the callback once on the
// return path, matching aiperf's prefill-release-if-TTFT-never-arrived rule.
// The return path also fires on Close, so a caller that stops reading at
// "data: [DONE]" and closes without draining to EOF still releases the slot.
// The callback is invoked while holding b.mu; it MUST be non-blocking and
// MUST NOT re-enter Read or Close. When no callback is set, the held slot
// (if any) is released directly instead, still exactly once.
type firstTokenBody struct {
	rc    io.ReadCloser
	r     *bufio.Reader
	hold  *prefillHold
	start time.Time
	cb    func(ttftNS int64, firstFrame string) bool
	mu    sync.Mutex // guards acked so the callback fires exactly once total
	acked bool       // callback reported true, or the return path already fired
	pend  []byte
}

// Read implements io.Reader over the wrapped body.
func (b *firstTokenBody) Read(p []byte) (int, error) {
	if b.pend != nil {
		n := copy(p, b.pend)
		b.pend = b.pend[n:]
		if len(b.pend) == 0 {
			b.pend = nil
		}
		return n, nil
	}
	line, err := b.r.ReadString('\n')
	if line == "" {
		// Stream ended (or errored) with nothing left: fire the return path
		// once if no content frame was ever acknowledged.
		b.releaseReturnPath()
		return 0, err
	}
	if strings.HasPrefix(line, "data: ") {
		data := strings.TrimPrefix(line, "data: ")
		if data != "[DONE]" {
			var frame firstTokenFrame
			if json.Unmarshal([]byte(data), &frame) == nil && frame.hasContent() {
				ttftNS := time.Since(b.start).Nanoseconds()
				if ttftNS < 0 {
					ttftNS = 0
				}
				b.tryAck(ttftNS, data)
			}
		}
	}
	n := copy(p, line)
	if n < len(line) {
		b.pend = []byte(line[n:])
	}
	return n, err
}

// tryAck fires the first-token callback for a content frame under the same
// lock as releaseReturnPath, so the callback fires exactly once total even
// if Read and Close run concurrently. It records the acknowledgement only
// when the callback reports true, per the aiperf contract. With no callback
// set, it releases the held slot directly (if any) and marks the body
// acknowledged, so the return path stays a no-op.
func (b *firstTokenBody) tryAck(ttftNS int64, frame string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.acked {
		return
	}
	if b.cb != nil {
		if !b.cb(ttftNS, frame) {
			return
		}
	} else if b.hold != nil {
		b.hold.release()
	}
	b.acked = true
}

// releaseReturnPath fires the return-path callback — an empty frame with a
// zero TTFT — at most once total, and is safe to call from both Read and
// Close. It is a no-op once a content frame has been acknowledged. With no
// callback set, it releases the held slot directly (if any) instead.
func (b *firstTokenBody) releaseReturnPath() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.acked {
		return
	}
	b.acked = true
	if b.cb != nil {
		b.cb(0, "")
	} else if b.hold != nil {
		b.hold.release()
	}
}

// Close closes the wrapped body. The return path fires here too, so a caller
// that stops reading at "data: [DONE]" and closes without draining to EOF
// still releases the held prefill slot.
func (b *firstTokenBody) Close() error {
	b.releaseReturnPath()
	return b.rc.Close()
}

// gatedTransport holds one prefill slot per in-flight request until that
// request's first content token arrives (or the stream ends without one),
// bounding concurrent prompt processing. When no slots are requested the
// runner keeps its default client and this transport is never built — the
// separate fast path aiperf uses when no callback is registered.
type gatedTransport struct {
	base http.RoundTripper
	gate *prefillGate
}

// RoundTrip implements http.RoundTripper.
func (t *gatedTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	hold, err := t.gate.hold(req.Context())
	if err != nil {
		return nil, err
	}
	// TTFT is measured from dispatch, after the slot is held: waiting in the
	// gate must not inflate the prefill latency the callback reports.
	start := time.Now()
	resp, err := t.base.RoundTrip(req)
	if err != nil {
		hold.release()
		return nil, err
	}
	if resp.Body == nil {
		hold.release()
		return resp, nil
	}
	resp.Body = &firstTokenBody{
		rc:    resp.Body,
		r:     bufio.NewReader(resp.Body),
		hold:  hold,
		start: start,
		cb: func(ttftNS int64, firstFrame string) bool {
			hold.release()
			return true
		},
	}
	return resp, nil
}

// prefillClient builds the http.Client handed to the AgentX runner. A zero
// slot count returns nil so NewRunner installs its default client and live-run
// behavior is unchanged; otherwise every request passes through the prefill
// gate. The timeout matches NewRunner's default client.
func prefillClient(slots int) *http.Client {
	if slots <= 0 {
		return nil
	}
	return &http.Client{
		Timeout:   120 * time.Second,
		Transport: &gatedTransport{base: http.DefaultTransport, gate: newPrefillGate(slots)},
	}
}

// speedupText prints n/a when no warm turn was observed.
func speedupText(ratio float64) string {
	if ratio <= 0 {
		return "n/a (no warm turn)"
	}
	return fmt.Sprintf("%.2fx", ratio)
}
