package main

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// TestTurnkeyConcurrentBatch is the #1590 witness: the turnkey `fak up` path must
// COALESCE N concurrent same-prefix requests onto ONE batched forward when the batch
// is admissible, and must fall back to the typed per-request path when it is not.
//
// It is deterministic and device-free: a synthetic Qwen3.5-hybrid model drives the
// CPU dense forward, the planner's device half is relaxed through the exported test
// seam, and a ready-barrier holds the cohort leader until every request has arrived
// so the fan-out genuinely coalesces (rather than depending on runtime.Gosched
// racing N goroutines). No external service, no GPU, no Metal.
func TestTurnkeyConcurrentBatch(t *testing.T) {
	tok := testProbeTokenizer(t)
	const fanout = 4
	prompts := make([]string, fanout)
	for i := range prompts {
		prompts[i] = "shared-prefix fan-out probe"
	}

	newPlanner := func(t *testing.T) *agent.InKernelPlanner {
		t.Helper()
		cfg := fakmodel.Config{
			HiddenSize:        32,
			NumLayers:         2,
			NumHeads:          4,
			NumKVHeads:        2,
			HeadDim:           8,
			IntermediateSize:  64,
			VocabSize:         320,
			RMSNormEps:        1e-5,
			RopeTheta:         10000,
			TieWordEmbeddings: true,
			EOSTokenID:        -1,
			// A linear_attention layer makes this a qwen35-family hybrid, which is
			// the architecture the coalescer admits.
			LayerTypes:          []string{"linear_attention"},
			LinearConvKernelDim: 3,
			LinearKeyHeadDim:    8,
			LinearNumKeyHeads:   2,
			LinearValueHeadDim:  8,
			LinearNumValueHeads: 4,
		}
		m := fakmodel.NewSynthetic(cfg)
		m.Quantize()
		p := agent.NewInKernelPlannerWithConfig(m, tok, "synthetic-turnkey-batch", false, nil, false, agent.InKernelPlannerConfig{})
		if p == nil {
			t.Fatal("planner is nil")
		}
		return p
	}

	// Serial reference: the same prompt on its own (the per-request baseline).
	serial := newPlanner(t)
	ref, err := serial.Complete(context.Background(), []agent.Message{{Role: agent.RoleUser, Content: prompts[0]}}, nil)
	if err != nil {
		t.Fatalf("serial complete: %v", err)
	}

	// --- Coalesce path -------------------------------------------------------
	coalescing := newPlanner(t)
	restoreSeam := coalescing.AdmitCoalescedDecodeForTest()
	defer restoreSeam()

	// The turnkey constructor must WIRE the coalescer in, not require FAK_INKERNEL_BATCH.
	turnkeyWired := newTurnkeyInKernelPlanner(
		fakmodel.NewSynthetic(fakmodel.Config{LayerTypes: []string{"linear_attention"}}),
		nil, "turnkey-wiring-probe", false, nil, false, 0,
	)
	if !turnkeyWired.BatchDecodeEnabled() {
		t.Fatal("newTurnkeyInKernelPlanner did not enable the batched-decode coalescer (#1590 routing missing)")
	}
	if serial.BatchDecodeEnabled() {
		t.Fatal("a default planner must not be batch-decode enabled")
	}

	released := make(chan struct{})
	restoreHook := coalescing.SetCoalesceReadyHookForTest(func() {
		// The cohort leader blocks here until the test has observed all N requests
		// queued, so the fan-out coalesces into ONE cohort deterministically.
		<-released
	})
	defer restoreHook()

	results := make([]*agent.Completion, fanout)
	errs := make([]error, fanout)
	var wg sync.WaitGroup
	for i := 0; i < fanout; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i], errs[i] = coalescing.Complete(
				context.Background(),
				[]agent.Message{{Role: agent.RoleUser, Content: prompts[i]}},
				nil,
			)
		}(i)
	}

	// Wait until every fan-out request has queued for the cohort, then release the
	// leader. Bounded so a wiring regression fails instead of hanging forever.
	waitForCoalesceReady(t, coalescing, fanout, 10*time.Second)
	close(released)
	wg.Wait()

	var cohortID uint64
	cohorts := map[uint64]int{}
	for i := 0; i < fanout; i++ {
		if errs[i] != nil {
			t.Fatalf("coalesced[%d]: %v", i, errs[i])
		}
		if results[i].Message.Content != ref.Message.Content {
			t.Fatalf("coalesced[%d] content %q != serial %q", i, results[i].Message.Content, ref.Message.Content)
		}
		r := results[i].InKernelBatch
		if r == nil {
			t.Fatalf("coalesced[%d] carried no batch receipt: coalescer was bypassed", i)
		}
		if r.CohortSize < 2 {
			t.Fatalf("coalesced[%d] cohort size %d, want >= 2 (fan-out not batched)", i, r.CohortSize)
		}
		cohorts[r.CohortID]++
		if cohortID == 0 {
			cohortID = r.CohortID
		}
	}
	if len(cohorts) != 1 {
		t.Fatalf("fan-out split across %d cohorts (%v), want one shared cohort", len(cohorts), cohorts)
	}
	if cohorts[cohortID] < 2 {
		t.Fatalf("cohort %d held %d requests, want >= 2", cohortID, cohorts[cohortID])
	}

	// --- Typed fallback path -------------------------------------------------
	// The same fan-out with the coalescer OFF must still serve every request
	// through the per-request path, carrying NO batch receipt (the typed fallback).
	fallback := newPlanner(t)
	fbuf := make([]*agent.Completion, fanout)
	fErrs := make([]error, fanout)
	var fwg sync.WaitGroup
	for i := 0; i < fanout; i++ {
		fwg.Add(1)
		go func(i int) {
			defer fwg.Done()
			fbuf[i], fErrs[i] = fallback.Complete(
				context.Background(),
				[]agent.Message{{Role: agent.RoleUser, Content: prompts[i]}},
				nil,
			)
		}(i)
	}
	fwg.Wait()
	for i := 0; i < fanout; i++ {
		if fErrs[i] != nil {
			t.Fatalf("fallback[%d]: %v", i, fErrs[i])
		}
		if fbuf[i].Message.Content != ref.Message.Content {
			t.Fatalf("fallback[%d] content %q != serial %q", i, fbuf[i].Message.Content, ref.Message.Content)
		}
		if fbuf[i].InKernelBatch != nil {
			t.Fatalf("fallback[%d] unexpectedly coalesced: %+v", i, fbuf[i].InKernelBatch)
		}
	}

	// --- Streaming framing preserved under concurrent fan-out ----------------
	// Drive the real turnkey HTTP handler with N concurrent same-prefix streamed
	// completions; each must terminate with text_completion frames and data: [DONE],
	// proving the wire framing survives concurrency.
	assertConcurrentStreamFraming(t, coalescing, prompts)
}

// assertConcurrentStreamFraming fans out streamed /v1/completions through the real
// turnkey handler and requires each response to be well-formed SSE terminated by
// data: [DONE].
func assertConcurrentStreamFraming(t *testing.T, planner *agent.InKernelPlanner, prompts []string) {
	t.Helper()
	ts := newTurnkeyCompletionsTestServer(planner)
	defer ts.Close()

	var wg sync.WaitGroup
	errCh := make(chan string, len(prompts))
	for _, p := range prompts {
		wg.Add(1)
		body, _ := json.Marshal(map[string]any{
			"model":      "local",
			"prompt":     p,
			"max_tokens": 4,
			"stream":     true,
		})
		go func(body []byte) {
			defer wg.Done()
			resp, err := http.Post(ts.URL, "application/json", strings.NewReader(string(body)))
			if err != nil {
				errCh <- err.Error()
				return
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				errCh <- "status " + resp.Status
				return
			}
			var sawChunk, sawDone bool
			sc := bufio.NewScanner(resp.Body)
			for sc.Scan() {
				line := strings.TrimSpace(sc.Text())
				if line == "data: [DONE]" {
					sawDone = true
					break
				}
				if strings.HasPrefix(line, "data: ") {
					var chunk struct {
						Object string `json:"object"`
					}
					if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err == nil && chunk.Object == "text_completion" {
						sawChunk = true
					}
				}
			}
			if !sawChunk || !sawDone {
				errCh <- "stream framing incomplete (chunk=" + boolStr(sawChunk) + " done=" + boolStr(sawDone) + ")"
			}
		}(body)
	}
	wg.Wait()
	close(errCh)
	for msg := range errCh {
		t.Fatalf("concurrent streamed completion framing: %s", msg)
	}
}

func boolStr(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

// waitForCoalesceReady polls until n requests are queued for the next cohort, bounded
// so a wiring regression fails the test instead of hanging forever.
func waitForCoalesceReady(t *testing.T, p *agent.InKernelPlanner, n int, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if p.CoalesceReadyLenForTest() >= n {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("timed out waiting for %d requests to reach the coalescer (saw %d)", n, p.CoalesceReadyLenForTest())
}
