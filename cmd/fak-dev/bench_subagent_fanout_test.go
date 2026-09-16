package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestBenchSubagentFanoutHelp(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runBenchSubagentFanout(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Fatalf("expected exit 0 for --help, got %d, stderr=%s", code, stderr.String())
	}
	for _, want := range []string{
		"-model",
		"-quant",
		"-mem-fraction",
		"-fanout",
		"-arms",
		"-verify-contract",
	} {
		if !strings.Contains(stderr.String(), want) {
			t.Errorf("help output missing flag %q:\n%s", want, stderr.String())
		}
	}
}

func TestBenchSubagentFanoutContractValidation(t *testing.T) {
	t.Run("CompliantConfig", func(t *testing.T) {
		cfg := &FanoutBenchConfig{
			Model:          "Qwen/Qwen2.5-Coder-7B-Instruct",
			Quantization:   "Q4_K_M",
			MemoryFraction: 0.85,
			FanoutSweep:    []int{1, 4, 8, 16, 32},
			Arms:           []string{"no_reuse", "sglang", "vllm", "fak"},
		}
		val := validateContractInvariants(cfg)
		if !val.Compliant {
			t.Fatalf("expected config to be compliant, violations: %v", val.Violations)
		}
		if !val.AllFourArmsPresent {
			t.Errorf("expected all 4 arms present")
		}
		if !val.FixedMemoryFractionVerified {
			t.Errorf("expected fixed memory fraction 0.85 verified")
		}
		if !val.FanoutSweepVerified {
			t.Errorf("expected fanout sweep [1,4,8,16,32] verified")
		}
	})

	t.Run("MissingArmViolation", func(t *testing.T) {
		cfg := &FanoutBenchConfig{
			Model:          "Qwen/Qwen2.5-Coder-7B-Instruct",
			Quantization:   "Q4_K_M",
			MemoryFraction: 0.85,
			FanoutSweep:    []int{1, 4, 8, 16, 32},
			Arms:           []string{"no_reuse", "sglang", "fak"}, // missing vllm
		}
		val := validateContractInvariants(cfg)
		if val.Compliant {
			t.Fatal("expected compliance failure when arm is missing")
		}
		if val.AllFourArmsPresent {
			t.Error("AllFourArmsPresent should be false")
		}
		foundViolation := false
		for _, v := range val.Violations {
			if strings.Contains(v, "vllm") {
				foundViolation = true
				break
			}
		}
		if !foundViolation {
			t.Errorf("expected violation mentioning vllm, got: %v", val.Violations)
		}
	})

	t.Run("InvalidMemoryFractionViolation", func(t *testing.T) {
		cfg := &FanoutBenchConfig{
			Model:          "Qwen/Qwen2.5-Coder-7B-Instruct",
			Quantization:   "Q4_K_M",
			MemoryFraction: 0.90, // Contract requires 0.85
			FanoutSweep:    []int{1, 4, 8, 16, 32},
			Arms:           []string{"no_reuse", "sglang", "vllm", "fak"},
		}
		val := validateContractInvariants(cfg)
		if val.Compliant {
			t.Fatal("expected compliance failure when memory fraction != 0.85")
		}
		if val.FixedMemoryFractionVerified {
			t.Error("FixedMemoryFractionVerified should be false")
		}
	})

	t.Run("IncompleteFanoutSweepViolation", func(t *testing.T) {
		cfg := &FanoutBenchConfig{
			Model:          "Qwen/Qwen2.5-Coder-7B-Instruct",
			Quantization:   "Q4_K_M",
			MemoryFraction: 0.85,
			FanoutSweep:    []int{1, 4, 8}, // missing 16, 32
			Arms:           []string{"no_reuse", "sglang", "vllm", "fak"},
		}
		val := validateContractInvariants(cfg)
		if val.Compliant {
			t.Fatal("expected compliance failure when sweep is incomplete")
		}
		if val.FanoutSweepVerified {
			t.Error("FanoutSweepVerified should be false")
		}
	})
}

func TestBenchSubagentFanoutSimulationRunJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{
		"-fanout", "1,4,8,16,32",
		"-arms", "no_reuse,sglang,vllm,fak",
		"-model", "Qwen/Qwen2.5-Coder-7B-Instruct",
		"-quant", "Q4_K_M",
		"-mem-fraction", "0.85",
		"-trials", "2",
		"-json",
	}

	code := runBenchSubagentFanout(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("runBenchSubagentFanout failed with code %d, stderr: %s", code, stderr.String())
	}

	var receipt SubagentFanoutReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("failed to decode receipt JSON: %v\nOutput: %s", err, stdout.String())
	}

	if receipt.Schema != SubagentFanoutComparisonSchema {
		t.Errorf("schema = %q, want %q", receipt.Schema, SubagentFanoutComparisonSchema)
	}
	if receipt.ContractIssue != ContractIssue6036 {
		t.Errorf("contract issue = %q, want %q", receipt.ContractIssue, ContractIssue6036)
	}
	if !receipt.Contract.Compliant {
		t.Errorf("contract audit should pass, violations: %v", receipt.Contract.Violations)
	}
	if math.Abs(receipt.MemoryFraction-0.85) > 1e-4 {
		t.Errorf("memory fraction = %f, want 0.85", receipt.MemoryFraction)
	}

	// Must contain 5 fanout values * 4 arms = 20 cells
	wantCells := len(CanonicalFanoutSweep) * len(CanonicalArms)
	if len(receipt.Results) != wantCells {
		t.Fatalf("got %d results, want %d", len(receipt.Results), wantCells)
	}

	for _, res := range receipt.Results {
		if res.Error != "" {
			t.Errorf("cell N=%d Arm=%s returned error: %s", res.FanoutN, res.Arm, res.Error)
		}
		// TTFT percentiles order check
		if res.TTFT.P50 > res.TTFT.P95 || res.TTFT.P95 > res.TTFT.P99 {
			t.Errorf("invalid TTFT percentile ordering for N=%d Arm=%s: p50=%.2f, p95=%.2f, p99=%.2f",
				res.FanoutN, res.Arm, res.TTFT.P50, res.TTFT.P95, res.TTFT.P99)
		}
		// ITL jitter check
		if res.ITL.JitterMs < 0 {
			t.Errorf("negative ITL jitter for N=%d Arm=%s: %f", res.FanoutN, res.Arm, res.ITL.JitterMs)
		}
		// Hit rate check for no-reuse arm
		if res.Arm == ArmNoReuse && res.PrefixHitRate != 0.0 {
			t.Errorf("ArmNoReuse should have 0.0 hit rate, got %f", res.PrefixHitRate)
		}
		// At N > 1, reuse arms must have positive hit rate
		if res.Arm != ArmNoReuse && res.FanoutN > 1 && res.PrefixHitRate <= 0.0 {
			t.Errorf("Arm %s at N=%d should have positive hit rate, got %f", res.Arm, res.FanoutN, res.PrefixHitRate)
		}
	}
}

func TestPercentileCalculation(t *testing.T) {
	sorted := []float64{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}

	p50 := calcPercentile(sorted, 50.0)
	if math.Abs(p50-55.0) > 1e-4 {
		t.Errorf("p50 = %f, want 55.0", p50)
	}

	p0 := calcPercentile(sorted, 0.0)
	if math.Abs(p0-10.0) > 1e-4 {
		t.Errorf("p0 = %f, want 10.0", p0)
	}

	p100 := calcPercentile(sorted, 100.0)
	if math.Abs(p100-100.0) > 1e-4 {
		t.Errorf("p100 = %f, want 100.0", p100)
	}
}

func TestDistributionComputations(t *testing.T) {
	samples := []float64{10.0, 12.0, 14.0, 16.0, 18.0, 20.0}
	dist := computeDistribution(samples)
	if dist.Count != 6 {
		t.Errorf("count = %d, want 6", dist.Count)
	}
	if math.Abs(dist.Mean-15.0) > 1e-4 {
		t.Errorf("mean = %f, want 15.0", dist.Mean)
	}
	if dist.Min != 10.0 || dist.Max != 20.0 {
		t.Errorf("min/max = %f/%f, want 10/20", dist.Min, dist.Max)
	}

	itl := computeITLStats(samples)
	if itl.Count != 6 {
		t.Errorf("count = %d, want 6", itl.Count)
	}
	if itl.JitterMs <= 0 {
		t.Errorf("jitter should be positive, got %f", itl.JitterMs)
	}
}

func TestPrettyRenderOutput(t *testing.T) {
	cfg := &FanoutBenchConfig{
		Model:          "Qwen/Qwen2.5-Coder-7B-Instruct",
		Quantization:   "Q4_K_M",
		MemoryFraction: 0.85,
		FanoutSweep:    []int{1, 4},
		Arms:           []string{"no_reuse", "fak"},
		Trials:         1,
		PrefixTokens:   1024,
		SuffixTokens:   128,
		DecodeTokens:   16,
		Seed:           42,
	}
	harness := NewFanoutBenchmarkHarness(cfg)
	receipt, err := harness.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	receipt.Contract = validateContractInvariants(cfg)

	var buf bytes.Buffer
	renderPrettyReceipt(&buf, receipt)
	output := buf.String()

	if !strings.Contains(output, "APPLES-TO-APPLES SUBAGENT FANOUT BENCHMARK HARNESS") {
		t.Errorf("missing header in pretty render:\n%s", output)
	}
	if !strings.Contains(output, "No-reuse baseline") || !strings.Contains(output, "FAK native engine") {
		t.Errorf("missing arm labels in pretty render:\n%s", output)
	}
}

// TestSubagentFanoutLlamaCPPArm proves the additional llama.cpp arm is
// parseable/selectable and that a LIVE cell against an OpenAI-compatible
// httptest endpoint produces a well-formed FanoutArmResult labeled llamacpp.
// No network or external service is required.
func TestSubagentFanoutLlamaCPPArm(t *testing.T) {
	// 1. Arm identifier and description are registered.
	if ArmLLamaCPP != "llamacpp" {
		t.Fatalf("ArmLLamaCPP = %q, want %q", ArmLLamaCPP, "llamacpp")
	}
	if desc := ArmDescription[ArmLLamaCPP]; desc == "" {
		t.Fatalf("ArmDescription[%q] must be registered", ArmLLamaCPP)
	}

	// 2. The arm is NOT one of the 4 mandatory contract arms.
	for _, a := range CanonicalArms {
		if a == ArmLLamaCPP {
			t.Fatalf("CanonicalArms must stay the frozen 4; got %q", ArmLLamaCPP)
		}
	}

	// 3. -arms accepts llamacpp and -llamacpp-url wires the endpoint.
	cfg, err := parseFanoutFlags(&bytes.Buffer{}, []string{
		"-arms", "no_reuse,sglang,vllm,fak,llamacpp",
		"-llamacpp-url", "http://127.0.0.1:18081",
	})
	if err != nil {
		t.Fatalf("parseFanoutFlags: %v", err)
	}
	found := false
	for _, a := range cfg.Arms {
		if a == ArmLLamaCPP {
			found = true
		}
	}
	if !found {
		t.Fatalf("llamacpp not selectable via -arms; got %v", cfg.Arms)
	}
	if got := cfg.Endpoints[ArmLLamaCPP]; got != "http://127.0.0.1:18081" {
		t.Fatalf("llamacpp endpoint = %q, want %q", got, "http://127.0.0.1:18081")
	}

	// 4. A LIVE cell against an OpenAI-compatible streaming endpoint yields a
	//    well-formed result labeled llamacpp. The fixture must also serve /props
	//    with an adequate per-slot context, because the harness now fails a cell
	//    closed unless the reference provably serves the frozen geometry
	//    (issue #13134).
	var hits int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/props":
			// Per-slot n_ctx must cover P+S+D = 64+16+4 = 84; serve 4096.
			fmt.Fprint(w, `{"total_slots":8,"default_generation_settings":{"n_ctx":4096}}`)
			return
		case "/v1/completions":
			// handled below
		default:
			http.NotFound(w, r)
			return
		}
		atomic.AddInt64(&hits, 1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		for i := 0; i < 4; i++ {
			fmt.Fprintf(w, "data: {\"choices\":[{\"text\":\"tok\"}]}\n\n")
			if flusher != nil {
				flusher.Flush()
			}
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		if flusher != nil {
			flusher.Flush()
		}
	}))
	defer srv.Close()

	harness := NewFanoutBenchmarkHarness(&FanoutBenchConfig{
		Model:          "Qwen/Qwen2.5-Coder-7B-Instruct",
		Quantization:   "Q4_K_M",
		MemoryFraction: FixedMemoryFraction,
		FanoutSweep:    []int{2},
		Arms:           []string{ArmLLamaCPP},
		PrefixTokens:   64,
		SuffixTokens:   16,
		DecodeTokens:   4,
		Trials:         1,
		Live:           true,
		Endpoints:      map[string]string{ArmLLamaCPP: srv.URL},
		Seed:           42,
	})
	receipt, err := harness.Run(context.Background())
	if err != nil {
		t.Fatalf("live harness run: %v", err)
	}
	if atomic.LoadInt64(&hits) == 0 {
		t.Fatal("live cell never reached the httptest endpoint")
	}
	if len(receipt.Results) != 1 {
		t.Fatalf("got %d results, want 1", len(receipt.Results))
	}
	res := receipt.Results[0]
	if res.Error != "" {
		t.Fatalf("live cell returned error: %s", res.Error)
	}
	if res.Arm != ArmLLamaCPP {
		t.Errorf("result arm = %q, want %q", res.Arm, ArmLLamaCPP)
	}
	if res.ArmDescription != ArmDescription[ArmLLamaCPP] {
		t.Errorf("arm description = %q, want %q", res.ArmDescription, ArmDescription[ArmLLamaCPP])
	}
	if res.FanoutN != 2 {
		t.Errorf("fanout N = %d, want 2", res.FanoutN)
	}
	if res.PrefixTokens != 64 || res.SuffixTokens != 16 || res.DecodeTokens != 4 {
		t.Errorf("trace geometry not preserved: P=%d S=%d D=%d", res.PrefixTokens, res.SuffixTokens, res.DecodeTokens)
	}
	if res.TotalPromptTokens != int64(2)*(64+16) {
		t.Errorf("total prompt tokens = %d, want %d", res.TotalPromptTokens, int64(2)*(64+16))
	}
	if res.TTFT.Count == 0 {
		t.Errorf("expected TTFT samples from the live stream")
	}
	if res.OutputHash == "" {
		t.Errorf("expected non-empty output hash")
	}
	// At N > 1 the arm has a shared-prefix cache, so reuse must be accounted.
	if res.ReusedTokens <= 0 || res.PrefixHitRate <= 0 {
		t.Errorf("expected positive reuse accounting at N=2, got reused=%d rate=%f", res.ReusedTokens, res.PrefixHitRate)
	}
	// The live cell must carry the serve-completeness witness and be admitted.
	if res.ServeCompleteness == nil {
		t.Fatal("live cell must carry ServeCompleteness")
	}
	if !res.ServeCompleteness.ControlArmServed {
		t.Errorf("adequately sized reference must be admitted as a control: %+v", res.ServeCompleteness)
	}

	// 5. Contract validation still requires the frozen 4; llamacpp alone is
	//    not compliant, which proves CanonicalArms did not absorb it.
	if val := validateContractInvariants(&FanoutBenchConfig{
		Model:          "Qwen/Qwen2.5-Coder-7B-Instruct",
		Quantization:   "Q4_K_M",
		MemoryFraction: FixedMemoryFraction,
		FanoutSweep:    CanonicalFanoutSweep,
		Arms:           []string{ArmLLamaCPP},
	}); val.Compliant {
		t.Error("llamacpp alone must not satisfy the 4-arm contract")
	}
}

// fanoutPropsServer stands up a server-free llama-server-shaped endpoint that
// answers /props with a fixed per-slot n_ctx (the served capacity) and
// /v1/completions with a small SSE stream. It is the whole reference surface the
// harness touches, so a cell can be driven entirely in-process.
func fanoutPropsServer(t *testing.T, perSlotCtx, totalSlots int, completionsHits *int64) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/props":
			// Mirrors the live shape: per-slot n_ctx == -c/--parallel, plus total_slots.
			fmt.Fprintf(w, `{"total_slots":%d,"default_generation_settings":{"n_ctx":%d}}`, totalSlots, perSlotCtx)
		case "/v1/completions":
			if completionsHits != nil {
				atomic.AddInt64(completionsHits, 1)
			}
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			flusher, _ := w.(http.Flusher)
			for i := 0; i < 4; i++ {
				fmt.Fprint(w, "data: {\"choices\":[{\"text\":\"tok\"}]}\n\n")
				if flusher != nil {
					flusher.Flush()
				}
			}
			fmt.Fprint(w, "data: [DONE]\n\n")
			if flusher != nil {
				flusher.Flush()
			}
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestSubagentFanoutServeGeometryShortfallFailsClosed is the issue #13134
// witness: a reference server whose per-slot context cannot hold the frozen
// P+S+D geometry must fail the cell CLOSED with a structured reason naming the
// observed limit, and must never be counted as a measured/compared arm.
func TestSubagentFanoutServeGeometryShortfallFailsClosed(t *testing.T) {
	// Served per-slot capacity 128 < P+S+D = 256+64+16 = 336, so the reference
	// cannot hold even one frozen prompt and the cell must fail closed.
	var completionsHits int64
	srv := fanoutPropsServer(t, 128, 4, &completionsHits)

	h := NewFanoutBenchmarkHarness(&FanoutBenchConfig{
		Model:          "Qwen/Qwen2.5-Coder-7B-Instruct",
		Quantization:   "Q4_K_M",
		MemoryFraction: FixedMemoryFraction,
		FanoutSweep:    []int{2},
		Arms:           []string{ArmLLamaCPP},
		PrefixTokens:   256,
		SuffixTokens:   64,
		DecodeTokens:   16,
		Trials:         1,
		Live:           true,
		Endpoints:      map[string]string{ArmLLamaCPP: srv.URL},
		Seed:           42,
	})
	receipt, err := h.Run(context.Background())
	if err != nil {
		t.Fatalf("harness run returned a top-level error (cells must fail, not the run): %v", err)
	}
	if len(receipt.Results) != 1 {
		t.Fatalf("got %d results, want 1", len(receipt.Results))
	}
	res := receipt.Results[0]

	// The cell failed closed.
	if res.Error == "" {
		t.Fatal("under-capacity reference must produce a failed cell, got none")
	}
	if res.ServeCompleteness == nil {
		t.Fatal("failed cell must carry a structured ServeCompleteness witness")
	}
	sc := res.ServeCompleteness
	if sc.ControlArmServed {
		t.Error("ControlArmServed must be false when the reference cannot hold the prompt")
	}
	if sc.Refusal != FanoutServeRefusePromptOverflow {
		t.Errorf("refusal = %q, want %q", sc.Refusal, FanoutServeRefusePromptOverflow)
	}
	if sc.EvidenceGap {
		t.Error("a measured shortfall is not an evidence gap")
	}
	if sc.ObservedPerSlotTokens != 128 {
		t.Errorf("observed per-slot = %d, want 128 (the served limit)", sc.ObservedPerSlotTokens)
	}
	if sc.ObservedTotalSlots != 4 {
		t.Errorf("observed total slots = %d, want 4", sc.ObservedTotalSlots)
	}
	if sc.PromptTokens != 320 { // P+S
		t.Errorf("declared prompt geometry = %d, want 320", sc.PromptTokens)
	}
	if sc.RequestedTotalTokens != 336 { // P+S+D
		t.Errorf("requested total = %d, want 336", sc.RequestedTotalTokens)
	}
	// The refusal must name the observed limit so the reader sees the cause.
	if !strings.Contains(sc.Detail, "128") {
		t.Errorf("refusal detail must name the observed limit 128: %q", sc.Detail)
	}
	if !strings.Contains(sc.Detail, "336") {
		t.Errorf("refusal detail must name the required geometry 336: %q", sc.Detail)
	}
	// The whole point: the control never ran, so no measured request was spent.
	if got := atomic.LoadInt64(&completionsHits); got != 0 {
		t.Errorf("an under-capacity control must not spend measured requests, got %d completions", got)
	}
	// It must not be counted as a served/comparable arm.
	if sc.ControlArmServed {
		t.Error("under-capacity reference must not be counted as a measured arm")
	}

	// The receipt summary must surface the unserved control, and the pretty
	// render must name the structured refusal so a reader cannot mistake a
	// broken reference for a measured null result.
	if got, ok := receipt.Summary["control_arms_unserved"]; !ok || got != 1 {
		t.Errorf("summary control_arms_unserved = %v (ok=%v), want 1", got, ok)
	}
	var buf bytes.Buffer
	renderPrettyReceipt(&buf, receipt)
	if !strings.Contains(buf.String(), FanoutServeRefusePromptOverflow) {
		t.Errorf("pretty render must name the refusal %q:\n%s", FanoutServeRefusePromptOverflow, buf.String())
	}
}

// TestSubagentFanoutServeGeometryOK proves a correctly sized reference is
// unaffected: the cell is admitted, ControlArmServed is true, and the measured
// requests actually ran.
func TestSubagentFanoutServeGeometryOK(t *testing.T) {
	var completionsHits int64
	// served 4096/slot >= P+S+D = 64+16+4 = 84
	srv := fanoutPropsServer(t, 4096, 8, &completionsHits)

	h := NewFanoutBenchmarkHarness(&FanoutBenchConfig{
		Model:          "Qwen/Qwen2.5-Coder-7B-Instruct",
		Quantization:   "Q4_K_M",
		MemoryFraction: FixedMemoryFraction,
		FanoutSweep:    []int{2},
		Arms:           []string{ArmLLamaCPP},
		PrefixTokens:   64,
		SuffixTokens:   16,
		DecodeTokens:   4,
		Trials:         1,
		Live:           true,
		Endpoints:      map[string]string{ArmLLamaCPP: srv.URL},
		Seed:           42,
	})
	receipt, err := h.Run(context.Background())
	if err != nil {
		t.Fatalf("live harness run: %v", err)
	}
	if len(receipt.Results) != 1 {
		t.Fatalf("got %d results, want 1", len(receipt.Results))
	}
	res := receipt.Results[0]
	if res.Error != "" {
		t.Fatalf("correctly sized reference must not fail: %s", res.Error)
	}
	if res.ServeCompleteness == nil {
		t.Fatal("live cell must carry a ServeCompleteness witness")
	}
	sc := res.ServeCompleteness
	if !sc.ControlArmServed {
		t.Errorf("ControlArmServed must be true; refusal=%q detail=%q", sc.Refusal, sc.Detail)
	}
	if sc.Refusal != "" {
		t.Errorf("no refusal expected on the ok path, got %q", sc.Refusal)
	}
	if sc.ObservedPerSlotTokens != 4096 {
		t.Errorf("observed per-slot = %d, want 4096", sc.ObservedPerSlotTokens)
	}
	if atomic.LoadInt64(&completionsHits) == 0 {
		t.Fatal("an admitted control must actually run its measured requests")
	}
}

// TestSubagentFanoutServeGeometryUnobservedFailsClosed proves the anti-vacuity
// path: when the served limit cannot be read (no /props), the cell fails closed
// as an evidence gap rather than assuming a default capacity.
func TestSubagentFanoutServeGeometryUnobservedFailsClosed(t *testing.T) {
	var completionsHits int64
	// A server with NO /props endpoint: only completions exists.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/completions" {
			http.NotFound(w, r)
			return
		}
		atomic.AddInt64(&completionsHits, 1)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	h := NewFanoutBenchmarkHarness(&FanoutBenchConfig{
		Model:          "Qwen/Qwen2.5-Coder-7B-Instruct",
		Quantization:   "Q4_K_M",
		MemoryFraction: FixedMemoryFraction,
		FanoutSweep:    []int{1},
		Arms:           []string{ArmLLamaCPP},
		PrefixTokens:   64,
		SuffixTokens:   16,
		DecodeTokens:   4,
		Trials:         1,
		Live:           true,
		Endpoints:      map[string]string{ArmLLamaCPP: srv.URL},
		Seed:           42,
	})
	receipt, err := h.Run(context.Background())
	if err != nil {
		t.Fatalf("harness run: %v", err)
	}
	res := receipt.Results[0]
	if res.Error == "" {
		t.Fatal("unreadable served capacity must fail the cell closed")
	}
	if res.ServeCompleteness == nil {
		t.Fatal("must carry ServeCompleteness with the evidence gap")
	}
	sc := res.ServeCompleteness
	if sc.ControlArmServed {
		t.Error("unobserved capacity must never be admitted")
	}
	if sc.Refusal != FanoutServeRefuseCapacityUnobserved {
		t.Errorf("refusal = %q, want %q", sc.Refusal, FanoutServeRefuseCapacityUnobserved)
	}
	if !sc.EvidenceGap {
		t.Error("an unreadable limit is an evidence gap")
	}
	if got := atomic.LoadInt64(&completionsHits); got != 0 {
		t.Errorf("unobserved control must not spend measured requests, got %d", got)
	}
}

// TestSubagentFanoutServeRefusalVocabularyIsClosed pins the refusal tokens so a
// future edit cannot silently invent a new one.
func TestSubagentFanoutServeRefusalVocabularyIsClosed(t *testing.T) {
	want := map[string]bool{
		FanoutServeRefusePromptOverflow:     true,
		FanoutServeRefuseCapacityUnobserved: true,
		FanoutServeRefuseProbeError:         true,
	}
	if len(FanoutServeRefusalVocabulary) != len(want) {
		t.Fatalf("vocabulary has %d tokens, want %d", len(FanoutServeRefusalVocabulary), len(want))
	}
	for _, tok := range FanoutServeRefusalVocabulary {
		if !want[tok] {
			t.Errorf("unexpected refusal token %q", tok)
		}
	}
}
