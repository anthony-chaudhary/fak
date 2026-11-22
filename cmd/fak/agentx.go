package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sort"
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
	parentPID := fs.Int("parent-pid", agentxDefaultParentPID(), "direct-parent supervisor PID captured at launch, not a wrapper shell's parent (env "+agentxParentPIDEnv+"); when set, exit if it is no longer the parent and, on Linux, die with it via PR_SET_PDEATHSIG; 0 disables")

	if err := fs.Parse(argv); err != nil {
		return 2
	}

	if *prefillSlots < 0 {
		fmt.Fprintf(stderr, "fak agentx: --prefill-slots must be >= 0\n")
		return 2
	}
	if *parentPID < 0 {
		fmt.Fprintf(stderr, "fak agentx: --parent-pid must be >= 0\n")
		return 2
	}
	if !installAgentXParentDeathGuard(stderr, *parentPID) {
		return 1
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

// NOTICE (attribution): the layered prefix allocator below is a direct port
// of the layered KV-cache prefix model from ai-dynamo/aiperf, file
// src/aiperf/dataset/agentic_code_gen/prefix_model.py at commit
// ab610add4e8876fca223197596c728e17365f381 (ai-dynamo/aiperf@ab610add),
// SPDX-License-Identifier: Apache-2.0, Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved. Layer structure, constants,
// and allocation logic are mirrored one-to-one; only the language changed.

// Layered prefix model constants, mirroring aiperf's prefix_model.py.
const (
	maxGroupBlocks   = 200
	maxGroups        = 1_000
	maxSessionBlocks = 4_000
)

// prefixLayerConfig mirrors the CacheLayerConfig fields the aiperf prefix
// allocator consumes: L1 (global) tokens, L1.5 (group-shared) tokens, and the
// number of L1.5 groups.
type prefixLayerConfig struct {
	Layer1Tokens  int
	Layer15Tokens int
	Layer15Groups int
}

// prefixAllocator allocates deterministic hash IDs for the layered prefix
// model, mirroring aiperf's PrefixAllocator:
//
//	L1:   tools + system prompt. Identical across ALL sessions (globally cached).
//	L1.5: group-shared prefix (repo instructions and context). Shared within a group.
//	L2:   session-specific prefix (initial files). Unique per session at turn 0.
//	L3:   conversation history added from turn 1 onward. Grows each turn.
//
// Hash ID layout:
//
//	L1:     [0 .. l1Blocks-1]                                      (global)
//	L1.5:   [l1Blocks + groupID*maxGroupBlocks .. +N]              (per group)
//	L2+L3:  [sessionBase .. sessionBase+M]                         (per session)
//
// On session reset: a new session gets fresh L2+L3, L1 and L1.5 preserved.
type prefixAllocator struct {
	blockSize         int
	l1Blocks          int
	l15Blocks         int
	l1IDs             []int
	prefixBlocks      int
	sessionRegionBase int
}

// newPrefixAllocator validates the configuration and precomputes the layer
// geometry, mirroring aiperf's PrefixAllocator.__init__.
func newPrefixAllocator(cfg prefixLayerConfig, blockSize int) (*prefixAllocator, error) {
	if blockSize <= 0 {
		return nil, fmt.Errorf("block_size must be > 0")
	}
	l1Blocks := ceilDiv(cfg.Layer1Tokens, blockSize)
	l15Blocks := ceilDiv(cfg.Layer15Tokens, blockSize)
	if cfg.Layer15Groups > maxGroups {
		return nil, fmt.Errorf("num_groups=%d exceeds MAX_GROUPS=%d", cfg.Layer15Groups, maxGroups)
	}
	if l15Blocks > maxGroupBlocks {
		return nil, fmt.Errorf("layer1_5_tokens=%d needs %d blocks, exceeding MAX_GROUP_BLOCKS=%d",
			cfg.Layer15Tokens, l15Blocks, maxGroupBlocks)
	}
	l1IDs := make([]int, l1Blocks)
	for i := range l1IDs {
		l1IDs[i] = i
	}
	return &prefixAllocator{
		blockSize:    blockSize,
		l1Blocks:     l1Blocks,
		l15Blocks:    l15Blocks,
		l1IDs:        l1IDs,
		prefixBlocks: l1Blocks + l15Blocks,
		// Session region starts after all group regions.
		sessionRegionBase: l1Blocks + maxGroups*maxGroupBlocks,
	}, nil
}

// L1Blocks reports the number of globally shared L1 blocks.
func (a *prefixAllocator) L1Blocks() int { return a.l1Blocks }

// L15Blocks reports the number of group-shared L1.5 blocks.
func (a *prefixAllocator) L15Blocks() int { return a.l15Blocks }

// PrefixBlocks reports L1 + L1.5 blocks (shared prefix before session-specific content).
func (a *prefixAllocator) PrefixBlocks() int { return a.prefixBlocks }

// PrefixTokens reports the token span reserved by the shared prefix (L1 + L1.5)
// as WHOLE blocks. L1 and L1.5 each round up to a whole number of blocks
// independently, so the reserved prefix occupies PrefixBlocks*BlockSize tokens,
// which can exceed Layer1Tokens + Layer15Tokens. Session (L2) content must
// begin past this boundary; otherwise a small L2 remainder gets absorbed into
// the last shared prefix block, making that shared hash ID the partial final
// block in one session and a full block in another. That maps one hash ID to
// two block sizes and fails trace reconstruction on replay.
func (a *prefixAllocator) PrefixTokens() int { return a.prefixBlocks * a.blockSize }

// BlockSize reports the KV block size in tokens.
func (a *prefixAllocator) BlockSize() int { return a.blockSize }

// GroupBase computes the base hash ID offset for a group's L1.5 blocks.
func (a *prefixAllocator) GroupBase(groupID int) int {
	return a.l1Blocks + groupID*maxGroupBlocks
}

// SessionBase computes the base hash ID offset for a session's L2+L3 blocks.
func (a *prefixAllocator) SessionBase(sessionIndex int) int {
	return a.sessionRegionBase + sessionIndex*maxSessionBlocks
}

// ensureSessionBlockBudget rejects inputs whose session-level block demand
// exceeds the per-session region.
func (a *prefixAllocator) ensureSessionBlockBudget(sessionNeeded, inputLength int) error {
	if sessionNeeded > maxSessionBlocks {
		return fmt.Errorf("input_length=%d needs %d session blocks, exceeding MAX_SESSION_BLOCKS=%d",
			inputLength, sessionNeeded, maxSessionBlocks)
	}
	return nil
}

// l15IDs returns the L1.5 block IDs for a group.
func (a *prefixAllocator) l15IDs(groupID int) []int {
	base := a.GroupBase(groupID)
	ids := make([]int, a.l15Blocks)
	for i := range ids {
		ids[i] = base + i
	}
	return ids
}

// TurnHashIDs generates the full hash ID array for a turn: L1 ++ L1.5 ++
// session IDs, with len(result) == ceil(inputLength/BlockSize). On turn 0
// (prevSessionIDs == nil) everything beyond L1+L1.5 is session prefix (L2);
// on turn 1+ the carried session blocks are extended by new L3 growth.
func (a *prefixAllocator) TurnHashIDs(sessionIndex, groupID, inputLength int, prevSessionIDs []int) ([]int, error) {
	totalBlocks := 0
	if inputLength > 0 {
		totalBlocks = ceilDiv(inputLength, a.blockSize)
	}

	l1Used := min(a.l1Blocks, totalBlocks)
	l1 := a.l1IDs[:l1Used]

	remaining := totalBlocks - l1Used
	if remaining <= 0 {
		return append([]int(nil), l1...), nil
	}

	l15Used := min(a.l15Blocks, remaining)
	l15 := a.l15IDs(groupID)[:l15Used]

	sessionNeeded := remaining - l15Used
	if sessionNeeded < 0 {
		sessionNeeded = 0
	}
	if sessionNeeded == 0 {
		out := make([]int, 0, len(l1)+len(l15))
		out = append(out, l1...)
		out = append(out, l15...)
		return out, nil
	}

	if err := a.ensureSessionBlockBudget(sessionNeeded, inputLength); err != nil {
		return nil, err
	}
	base := a.SessionBase(sessionIndex)

	var session []int
	if prevSessionIDs == nil {
		session = make([]int, sessionNeeded)
		for i := range session {
			session[i] = base + i
		}
	} else {
		newBlocks := sessionNeeded - len(prevSessionIDs)
		if newBlocks > 0 {
			nextID := base
			if len(prevSessionIDs) > 0 {
				nextID = prevSessionIDs[len(prevSessionIDs)-1] + 1
			}
			session = make([]int, 0, sessionNeeded)
			session = append(session, prevSessionIDs...)
			for i := 0; i < newBlocks; i++ {
				session = append(session, nextID+i)
			}
		} else {
			session = append([]int(nil), prevSessionIDs[:sessionNeeded]...)
		}
	}

	out := make([]int, 0, len(l1)+len(l15)+len(session))
	out = append(out, l1...)
	out = append(out, l15...)
	out = append(out, session...)
	return out, nil
}

// SessionBlocksNeeded computes how many session-level blocks (L2+L3) are
// needed for a given input length.
func (a *prefixAllocator) SessionBlocksNeeded(inputLength int) int {
	totalBlocks := 0
	if inputLength > 0 {
		totalBlocks = ceilDiv(inputLength, a.blockSize)
	}
	l1Used := min(a.l1Blocks, totalBlocks)
	remaining := totalBlocks - l1Used
	l15Used := min(a.l15Blocks, remaining)
	sessionNeeded := remaining - l15Used
	if sessionNeeded < 0 {
		sessionNeeded = 0
	}
	return sessionNeeded
}

// BuildHashIDs constructs the full hash ID array from L1 + L1.5 + the
// provided session IDs, mirroring aiperf's build_hash_ids.
func (a *prefixAllocator) BuildHashIDs(groupID, inputLength int, sessionIDs []int) ([]int, error) {
	totalBlocks := 0
	if inputLength > 0 {
		totalBlocks = ceilDiv(inputLength, a.blockSize)
	}

	l1Used := min(a.l1Blocks, totalBlocks)
	l1 := a.l1IDs[:l1Used]

	remaining := totalBlocks - l1Used
	if remaining <= 0 {
		return append([]int(nil), l1...), nil
	}

	l15Used := min(a.l15Blocks, remaining)
	l15 := a.l15IDs(groupID)[:l15Used]

	sessionNeeded := remaining - l15Used
	if sessionNeeded < 0 {
		sessionNeeded = 0
	}
	if sessionNeeded == 0 {
		out := make([]int, 0, len(l1)+len(l15))
		out = append(out, l1...)
		out = append(out, l15...)
		return out, nil
	}

	if err := a.ensureSessionBlockBudget(sessionNeeded, inputLength); err != nil {
		return nil, err
	}
	if len(sessionIDs) < sessionNeeded {
		return nil, fmt.Errorf("input_length=%d needs %d session IDs, got %d",
			inputLength, sessionNeeded, len(sessionIDs))
	}

	out := make([]int, 0, len(l1)+len(l15)+sessionNeeded)
	out = append(out, l1...)
	out = append(out, l15...)
	out = append(out, sessionIDs[:sessionNeeded]...)
	return out, nil
}

// ExtractSessionIDs extracts everything after L1 + L1.5 from a full hash ID
// array, returning nil when the array holds no session content.
func (a *prefixAllocator) ExtractSessionIDs(hashIDs []int) []int {
	if len(hashIDs) <= a.prefixBlocks {
		return nil
	}
	return append([]int(nil), hashIDs[a.prefixBlocks:]...)
}

// ceilDiv is integer ceiling division for non-negative a and positive b,
// mirroring math.ceil(a/b) in the source.
func ceilDiv(a, b int) int {
	return (a + b - 1) / b
}

// NOTICE / attribution (Apache-2.0 reuse):
//
// The session synthesizer below is an idiomatic Go re-expression (no code
// copied) of the session-synthesis reset semantics in:
//
//	ai-dynamo/aiperf @ ab610ad (ab610add4e8876fca223197596c728e17365f381),
//	Apache-2.0 (NVIDIA CORPORATION & AFFILIATES, 2025-2026)
//	  - src/aiperf/dataset/agentic_code_gen/session_synthesizer.py
//	    (SessionSynthesizer per-session state machine; probabilistic reset)
//	  - src/aiperf/dataset/agentic_code_gen/prefix_model.py
//	    (PrefixAllocator.turn_hash_ids; "On session reset: new session gets
//	     fresh L2+L3, L1 and L1.5 preserved.")
//	  - src/aiperf/dataset/agentic_code_gen/models.py
//	    (SynthesizedTurn / SynthesizedSession / ResetConfig)
//
// Upstream license header: SPDX-License-Identifier: Apache-2.0.

// aiperf prefix-model block budget constants (prefix_model.py).
const (
	agentxMaxGroupBlocks   = 200
	agentxMaxGroups        = 1000
	agentxMaxSessionBlocks = 4000
)

// synthTurn mirrors aiperf's SynthesizedTurn (models.py): one synthesized turn
// within a session.
type synthTurn struct {
	turnIndex    int
	inputLength  int
	outputLength int
	newTokens    int
	delayMS      float64
	timestampMS  float64
	hashIDs      []int
}

// sessionSynthConfig is the base configuration a sessionSynthesizer re-seeds
// from on reset.
type sessionSynthConfig struct {
	MaxPromptTokens      int
	BlockSize            int
	Layer1Tokens         int
	Layer15Tokens        int
	NumGroups            int
	ZipfAlpha            float64
	L2Mu, L2Sigma        float64
	L2Min, L2Max         float64
	GenMu, GenSigma      float64
	OutputMin            int
	ResetBaseProbability float64
	ResetContextScaling  float64
	Seed                 int64
}

// sessionSynthesizer mirrors the per-session state machine of aiperf's
// SessionSynthesizer (session_synthesizer.py), focused on the session-reset
// semantics: when a session is reset, the synthesized state (turns, hashes,
// budget) is cleared and re-seeded from the base, with L1 and L1.5 preserved
// and fresh L2+L3 allocated to the new session.
type sessionSynthesizer struct {
	// base config (the re-seed source)
	maxPromptTokens      int
	blockSize            int
	l1Blocks             int
	l15Blocks            int
	numGroups            int
	zipfAlpha            float64
	groupWeights         []float64
	sessionRegionBase    int
	maxSessionBlocks     int
	maxGroupBlocks       int
	l2Mu, l2Sigma        float64
	l2Min, l2Max         float64
	genMu, genSigma      float64
	outputMin            int
	resetBaseProbability float64
	resetContextScaling  float64

	// per-session synthesized state (cleared + re-seeded on Reset)
	sessionCounter int
	sessionIndex   int
	sessionID      string
	groupID        int
	turns          []synthTurn
	hashes         []int
	prevInput      int
	prevOutput     int
	turnIdx        int
	timestampMS    float64
	budget         int
	endReason      string

	rng *rand.Rand
}

// newSessionSynthesizer builds a sessionSynthesizer from cfg, pre-computing the
// L1/L1.5 block counts and the Zipf group weights (source __init__).
func newSessionSynthesizer(cfg sessionSynthConfig) *sessionSynthesizer {
	l1Blocks := (cfg.Layer1Tokens + cfg.BlockSize - 1) / cfg.BlockSize
	l15Blocks := (cfg.Layer15Tokens + cfg.BlockSize - 1) / cfg.BlockSize
	weights := make([]float64, cfg.NumGroups)
	sum := 0.0
	for k := 1; k <= cfg.NumGroups; k++ {
		w := 1.0 / math.Pow(float64(k), cfg.ZipfAlpha)
		weights[k-1] = w
		sum += w
	}
	for i := range weights {
		weights[i] /= sum
	}
	return &sessionSynthesizer{
		maxPromptTokens:   cfg.MaxPromptTokens,
		blockSize:         cfg.BlockSize,
		l1Blocks:          l1Blocks,
		l15Blocks:         l15Blocks,
		numGroups:         cfg.NumGroups,
		zipfAlpha:         cfg.ZipfAlpha,
		groupWeights:      weights,
		sessionRegionBase: l1Blocks + agentxMaxGroups*agentxMaxGroupBlocks,
		maxSessionBlocks:  agentxMaxSessionBlocks,
		maxGroupBlocks:    agentxMaxGroupBlocks,
		l2Mu:              cfg.L2Mu, l2Sigma: cfg.L2Sigma,
		l2Min: cfg.L2Min, l2Max: cfg.L2Max,
		genMu: cfg.GenMu, genSigma: cfg.GenSigma,
		outputMin:            cfg.OutputMin,
		resetBaseProbability: cfg.ResetBaseProbability,
		resetContextScaling:  cfg.ResetContextScaling,
		rng:                  rand.New(rand.NewSource(cfg.Seed)),
	}
}

// nextSessionIndex returns the current session counter and advances it,
// mirroring aiperf's _next_session_index.
func (s *sessionSynthesizer) nextSessionIndex() int {
	idx := s.sessionCounter
	s.sessionCounter++
	return idx
}

// newSessionID mints a fresh session id, mirroring aiperf's "sess-<uuid
// hex[:12]>" format derived from 16 random bytes.
func (s *sessionSynthesizer) newSessionID() string {
	var b [16]byte
	for i := range b {
		b[i] = byte(s.rng.Int())
	}
	return "sess-" + fmt.Sprintf("%x", b[:])[:12]
}

// sampleGroupID ports aiperf's _sample_group_id: a Zipf-weighted draw over the
// L1.5 groups.
func (s *sessionSynthesizer) sampleGroupID() int {
	if s.numGroups <= 1 {
		return 0
	}
	r := s.rng.Float64()
	cum := 0.0
	for i, w := range s.groupWeights {
		cum += w
		if r < cum {
			return i
		}
	}
	return s.numGroups - 1
}

// lognormal draws a lognormal sample, exp(N(mu, sigma)). With sigma == 0 it is
// deterministic (exp(mu)), matching the source's numpy lognormal degenerate case.
func (s *sessionSynthesizer) lognormal(mu, sigma float64) float64 {
	if sigma == 0 {
		return math.Exp(mu)
	}
	return math.Exp(mu + sigma*s.rng.NormFloat64())
}

// sampleInitialContext ports aiperf's _sample_initial_context: L1 + L1.5
// (reserved as whole blocks) + sampled L2, clamped to max_prompt_tokens - 1.
func (s *sessionSynthesizer) sampleInitialContext() int {
	l2f := s.lognormal(s.l2Mu, s.l2Sigma)
	if s.l2Min > 0 && l2f < s.l2Min {
		l2f = s.l2Min
	}
	if s.l2Max > 0 && l2f > s.l2Max {
		l2f = s.l2Max
	}
	l2 := int(l2f)
	if l2 < 1 {
		l2 = 1
	}
	fixedPrefix := (s.l1Blocks + s.l15Blocks) * s.blockSize
	ctx := fixedPrefix + l2
	if maxCtx := s.maxPromptTokens - 1; ctx > maxCtx {
		ctx = maxCtx
	}
	return ctx
}

// sampleOutputLength ports aiperf's _sample_output_length: a lognormal draw
// with a hard floor at outputMin.
func (s *sessionSynthesizer) sampleOutputLength() int {
	v := int(s.lognormal(s.genMu, s.genSigma))
	if v < s.outputMin {
		v = s.outputMin
	}
	return v
}

// shouldReset ports aiperf's _should_reset: the context-dependent
// probabilistic reset check. P(reset) = base * (1 + (scaling-1) *
// inputLength/maxPromptTokens). A non-positive base probability disables it.
func (s *sessionSynthesizer) shouldReset(inputLength int) bool {
	if s.resetBaseProbability <= 0 {
		return false
	}
	ratio := float64(inputLength) / float64(s.maxPromptTokens)
	p := s.resetBaseProbability * (1.0 + (s.resetContextScaling-1.0)*ratio)
	return s.rng.Float64() < p
}

// turnHashIDs ports aiperf's PrefixAllocator.turn_hash_ids (prefix_model.py):
// returns L1 ++ L1.5 ++ session blocks, len == ceil(inputLength/blockSize).
// With prevSessionIDs nil (a fresh session / turn 0), the session blocks are
// allocated from the session base; otherwise they extend the previous session
// blocks.
func (s *sessionSynthesizer) turnHashIDs(sessionIndex, groupID, inputLength int, prevSessionIDs []int) []int {
	totalBlocks := 0
	if inputLength > 0 {
		totalBlocks = (inputLength + s.blockSize - 1) / s.blockSize
	}
	l1Used := s.l1Blocks
	if l1Used > totalBlocks {
		l1Used = totalBlocks
	}
	l1 := make([]int, l1Used)
	for i := range l1 {
		l1[i] = i
	}
	remaining := totalBlocks - l1Used
	if remaining <= 0 {
		return l1
	}
	l15Used := s.l15Blocks
	if l15Used > remaining {
		l15Used = remaining
	}
	l15 := make([]int, l15Used)
	gb := s.l1Blocks + groupID*s.maxGroupBlocks
	for i := range l15 {
		l15[i] = gb + i
	}
	sessionNeeded := remaining - l15Used
	if sessionNeeded < 0 {
		sessionNeeded = 0
	}
	if sessionNeeded == 0 {
		return append(l1, l15...)
	}
	sBase := s.sessionRegionBase + sessionIndex*s.maxSessionBlocks
	var session []int
	if prevSessionIDs == nil {
		session = make([]int, sessionNeeded)
		for i := range session {
			session[i] = sBase + i
		}
	} else {
		newBlocks := sessionNeeded - len(prevSessionIDs)
		if newBlocks > 0 {
			nextID := sBase
			if len(prevSessionIDs) > 0 {
				nextID = prevSessionIDs[len(prevSessionIDs)-1] + 1
			}
			session = make([]int, 0, sessionNeeded)
			session = append(session, prevSessionIDs...)
			for i := 0; i < newBlocks; i++ {
				session = append(session, nextID+i)
			}
		} else {
			session = make([]int, sessionNeeded)
			copy(session, prevSessionIDs)
		}
	}
	out := make([]int, 0, len(l1)+len(l15)+len(session))
	out = append(out, l1...)
	out = append(out, l15...)
	out = append(out, session...)
	return out
}

// Reset clears the synthesized state (turns, hashes, budget) and re-seeds from
// the base, mirroring aiperf's session-reset semantics: "On session reset: new
// session gets fresh L2+L3, L1 and L1.5 preserved." (prefix_model.py). The new
// session gets a fresh session index, session id, group, and initial context
// (L1 + L1.5 + sampled L2); its hash ids are regenerated from the session base
// with no carried session blocks, and the context budget is re-seeded.
func (s *sessionSynthesizer) Reset() {
	s.sessionIndex = s.nextSessionIndex()
	s.sessionID = s.newSessionID()
	s.groupID = s.sampleGroupID()
	initialCtx := s.sampleInitialContext()
	outputLen := s.sampleOutputLength()
	hashIDs := s.turnHashIDs(s.sessionIndex, s.groupID, initialCtx, nil)
	s.turns = []synthTurn{{
		turnIndex:    0,
		inputLength:  initialCtx,
		outputLength: outputLen,
		newTokens:    initialCtx,
		delayMS:      0.0,
		timestampMS:  0.0,
		hashIDs:      hashIDs,
	}}
	s.hashes = hashIDs
	s.budget = s.maxPromptTokens - initialCtx
	s.timestampMS = 0.0
	s.prevInput = initialCtx
	s.prevOutput = outputLen
	s.turnIdx = 1
	s.endReason = ""
}

// NOTICE / attribution (Apache-2.0 reuse):
//
// The tool-call delta merge below is an idiomatic Go re-expression (no code
// copied) of the streaming tool-call delta assembly in:
//
//	ai-dynamo/aiperf @ ab610ad (ab610add4e8876fca223197596c728e17365f381),
//	Apache-2.0 (NVIDIA CORPORATION & AFFILIATES, 2025-2026)
//	  - src/aiperf/endpoints/openai_chat.py
//	    (ChatEndpoint._merge_tool_call_delta and
//	     ChatEndpoint._merge_legacy_function_call_delta: fold streaming
//	     tool_calls / function_call deltas into an index-keyed accumulator)
//
// Upstream license header: SPDX-License-Identifier: Apache-2.0.

// toolCallDelta mirrors one element of a streaming "tool_calls" array in an
// OpenAI chat.completion.chunk delta. Index is a pointer so an omitted index
// is distinguishable from an explicit one; the source falls back to the count
// of existing slots when the server omits it.
type toolCallDelta struct {
	Index    *int                   `json:"index"`
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Function *toolCallDeltaFunction `json:"function"`
}

// toolCallDeltaFunction mirrors the "function" object of a tool-call delta.
// Arguments is a pointer so an absent "arguments" key is distinguishable from
// a present one, matching the source's `"arguments" in fn_delta` check.
type toolCallDeltaFunction struct {
	Name      string  `json:"name"`
	Arguments *string `json:"arguments"`
}

// toolCallSlot is one accumulated tool call, keyed by its stream index.
type toolCallSlot struct {
	id       string
	typ      string
	function *toolCallSlotFunction
}

// toolCallSlotFunction is the accumulated "function" object of a tool call.
type toolCallSlotFunction struct {
	name      string
	arguments string
}

// toolCallDeltaAccumulator folds streaming tool_calls deltas into an
// index-keyed accumulator, mirroring aiperf's _merge_tool_call_delta.
type toolCallDeltaAccumulator struct {
	slots map[int]*toolCallSlot
}

// newToolCallDeltaAccumulator returns an empty accumulator.
func newToolCallDeltaAccumulator() *toolCallDeltaAccumulator {
	return &toolCallDeltaAccumulator{slots: make(map[int]*toolCallSlot)}
}

// mergeToolCallDelta merges one streaming tool_calls delta into the
// index-keyed accumulator, mirroring aiperf's _merge_tool_call_delta:
//   - index defaults to the count of existing slots when omitted
//   - "id" and "type" are set when present (non-empty)
//   - function "name" is REPLACED when present (non-empty)
//   - function "arguments" is APPENDED across chunks when the key is present
func (a *toolCallDeltaAccumulator) mergeToolCallDelta(d toolCallDelta) {
	idx := len(a.slots)
	if d.Index != nil {
		idx = *d.Index
	}
	slot := a.slots[idx]
	if slot == nil {
		slot = &toolCallSlot{}
		a.slots[idx] = slot
	}
	if d.ID != "" {
		slot.id = d.ID
	}
	if d.Type != "" {
		slot.typ = d.Type
	}
	fn := d.Function
	if fn == nil {
		return
	}
	if fn.Name == "" && fn.Arguments == nil {
		// Empty function object: nothing to accumulate, mirroring the
		// source's `if not fn_delta: return`.
		return
	}
	if slot.function == nil {
		slot.function = &toolCallSlotFunction{}
	}
	if fn.Name != "" {
		slot.function.name = fn.Name
	}
	if fn.Arguments != nil {
		slot.function.arguments += *fn.Arguments
	}
}

// mergeLegacyFunctionCallDelta merges a legacy streaming function_call delta
// into a synthesised slot at index 0, mirroring aiperf's
// _merge_legacy_function_call_delta. Legacy chunks carry no index, so
// everything lands in slot 0; the function name is APPENDED (concatenated)
// rather than replaced, matching the source.
func (a *toolCallDeltaAccumulator) mergeLegacyFunctionCallDelta(fn *toolCallDeltaFunction) {
	if fn == nil {
		return
	}
	slot := a.slots[0]
	if slot == nil {
		slot = &toolCallSlot{typ: "function"}
		a.slots[0] = slot
	}
	if slot.typ == "" {
		slot.typ = "function"
	}
	if slot.function == nil {
		slot.function = &toolCallSlotFunction{}
	}
	if fn.Name != "" {
		slot.function.name += fn.Name
	}
	if fn.Arguments != nil {
		slot.function.arguments += *fn.Arguments
	}
}

// merged returns the accumulated tool calls in index order, mirroring the
// source's `[tool_calls_by_index[k] for k in sorted(tool_calls_by_index)]`.
func (a *toolCallDeltaAccumulator) merged() []toolCallSlot {
	if len(a.slots) == 0 {
		return nil
	}
	idxs := make([]int, 0, len(a.slots))
	for i := range a.slots {
		idxs = append(idxs, i)
	}
	sort.Ints(idxs)
	out := make([]toolCallSlot, 0, len(idxs))
	for _, i := range idxs {
		out = append(out, *a.slots[i])
	}
	return out
}
