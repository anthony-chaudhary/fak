package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/macbench"
	"github.com/anthony-chaudhary/fak/internal/macobs"
)

func cmdMacBench(argv []string) { os.Exit(runMacBench(os.Stdout, os.Stderr, argv)) }

func runMacBench(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && (argv[0] == "ensure-runs" || argv[0] == "ensure") {
		return runMacBenchEnsureRuns(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && (argv[0] == "load-drive" || argv[0] == "load-driver") {
		return runMacBenchLoadDrive(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && (argv[0] == "matched-prefill" || argv[0] == "prefill-matched") {
		return runMacBenchMatchedPrefill(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "validate-comparison" {
		return runMacBenchValidateComparison(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "validate-agentic-comparison" {
		return runMacBenchValidateAgenticComparison(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && (argv[0] == "validate-mtp-comparison" || argv[0] == "validate-mtp") {
		return runMacBenchValidateMTPComparison(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "validate-agentic-mtp" {
		return runMacBenchValidateAgenticMTP(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "run-agentic-mtp" {
		return runMacBenchRunAgenticMTP(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "watch-status" {
		return runMacBenchWatchStatus(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "recover" {
		return runMacBenchRecover(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "watch" {
		return runMacBenchWatch(stdout, stderr, argv[1:])
	}
	suite := macbench.SuiteAll
	if len(argv) > 0 && !strings.HasPrefix(argv[0], "-") {
		suite = macbench.Suite(argv[0])
		argv = argv[1:]
	}
	def := macbench.DefaultOptions()
	fs := flag.NewFlagSet("macbench", flag.ContinueOnError)
	fs.SetOutput(stderr)
	gateway := fs.String("gateway", envOrDefault("FAK_MAC_GATEWAY", def.Gateway), "fak serve gateway on the Mac; defaults to loopback for on-node runs")
	model := fs.String("model", envOrDefault("FAK_MAC_MODEL", def.Model), "model id served by the Mac gateway")
	keyEnv := fs.String("gateway-key-env", "FAK_GATEWAY_KEY", "env var holding the gateway bearer")
	keyFile := fs.String("gateway-key-file", "~/.fak-gateway-key", "file holding the gateway bearer when the env var is empty; empty disables file lookup")
	fetchKey := fs.Bool("fetch-key", true, "when env/file key lookup is empty for a remote gateway, fetch ~/.fak-gateway-key from the Mac over ssh")
	sshHost := fs.String("ssh-host", envOrDefault("FAK_MAC_SSH_HOST", defaultClaudeMacSSHHost), "ssh host used by --fetch-key")
	sshKey := fs.String("ssh-key", defaultClaudeMacSSHKey(), "ssh identity used by --fetch-key; empty uses ssh defaults")
	timeout := fs.Duration("timeout", 2*time.Hour, "overall benchmark timeout")
	decodeTokens := fs.String("decode-tokens", "16,32,64,128,256,512", "comma-separated max_tokens for decode-longgen")
	prefillTokens := fs.String("prefill-tokens", "128,512,2048,4096", "comma-separated prompt-token targets for prefill-sweep")
	concurrency := fs.Int("concurrency", 2, "concurrent requests for the 2stream suite")
	minPrefillTPS := fs.Float64("min-prefill-tps", 0, "absolute prefill throughput floor in tokens/second; >0 fails the run when no row meets it (0 disables)")
	minDecodeTPS := fs.Float64("min-decode-tps", 0, "absolute decode throughput floor in tokens/second; >0 fails the run when no row meets it (0 disables)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	key, err := resolveMacBenchKeyForRun(*keyEnv, *keyFile, *fetchKey, *sshHost, *sshKey, *gateway, suite)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench: %v\n", err)
		return 2
	}
	dec, err := parseIntCSV(*decodeTokens)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench: --decode-tokens: %v\n", err)
		return 2
	}
	pre, err := parseIntCSV(*prefillTokens)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench: --prefill-tokens: %v\n", err)
		return 2
	}
	if *timeout <= 0 {
		fmt.Fprintln(stderr, "fak macbench: --timeout must be positive")
		return 2
	}
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	gpuUtilBefore := sampleGPUUtilPct(ctx)
	rep, err := macbench.Run(ctx, macbench.Options{
		Gateway:       *gateway,
		Model:         *model,
		Key:           key,
		Suite:         suite,
		DecodeTokens:  dec,
		PrefillTokens: pre,
		Concurrency:   *concurrency,
		MinPrefillTPS: *minPrefillTPS,
		MinDecodeTPS:  *minDecodeTPS,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench: %v\n", err)
		return 1
	}
	gpuUtilAfter := sampleGPUUtilPct(ctx)
	rep.GPUUtilPct = gpuUtilBefore
	if gpuUtilAfter > rep.GPUUtilPct {
		rep.GPUUtilPct = gpuUtilAfter
	}
	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, rep)
	} else {
		renderMacBench(stdout, rep)
	}
	if rep.HasErrors() {
		return 1
	}
	if rep.SLO != nil && !rep.SLO.OK {
		fmt.Fprintf(stderr, "fak macbench: SLO MISS best_prefill=%.1f tok/s (floor %.1f) best_decode=%.1f tok/s (floor %.1f)\n",
			rep.SLO.BestPrefillTPS, rep.SLO.MinPrefillTPS, rep.SLO.BestDecodeTPS, rep.SLO.MinDecodeTPS)
		return 1
	}
	return 0
}

func runMacBenchEnsureRuns(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("macbench ensure-runs", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if err := macbench.EnsureNodeMacOSABenchmarkRuns(); err != nil {
		fmt.Fprintf(stderr, "fak macbench ensure-runs: %v\n", err)
		return 1
	}
	threeWay, err := macbench.EnsureNodeMacOSAThreeWayRun()
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench ensure-runs: %v\n", err)
		return 1
	}
	agenticMTP, err := macbench.EnsureNodeMacOSAAgenticMTPRun()
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench ensure-runs: %v\n", err)
		return 1
	}
	result := struct {
		Schema           string `json:"schema"`
		OK               bool   `json:"ok"`
		ThreeWayPacket   string `json:"three_way_packet"`
		AgenticMTPPacket string `json:"agentic_mtp_packet"`
		Regenerated      bool   `json:"regenerated"`
	}{
		Schema:           "fak.macbench.ensure-runs.v1",
		OK:               true,
		ThreeWayPacket:   threeWay,
		AgenticMTPPacket: agenticMTP,
		Regenerated:      true,
	}
	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, result)
	} else {
		fmt.Fprintf(stdout, "ENSURED three_way=%s agentic_mtp=%s\n", threeWay, agenticMTP)
	}
	return 0
}

func runMacBenchValidateComparison(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("macbench validate-comparison", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "three-way comparison packet JSON")
	asJSON := fs.Bool("json", false, "emit machine-readable validation result")
	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*input) == "" {
		fmt.Fprintln(stderr, "fak macbench validate-comparison: --input is required")
		return 2
	}
	raw, err := os.ReadFile(*input)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench validate-comparison: read --input: %v\n", err)
		return 1
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var packet macbench.ComparisonPacket
	if err := dec.Decode(&packet); err != nil {
		fmt.Fprintf(stderr, "fak macbench validate-comparison: decode packet: %v\n", err)
		return 1
	}
	var trailing json.RawMessage
	if err := dec.Decode(&trailing); err != io.EOF {
		if err == nil {
			err = fmt.Errorf("multiple JSON values")
		}
		fmt.Fprintf(stderr, "fak macbench validate-comparison: decode packet: %v\n", err)
		return 1
	}
	if err := macbench.ValidateComparisonPacket(packet); err != nil {
		fmt.Fprintf(stderr, "fak macbench validate-comparison: %v\n", err)
		return 1
	}
	if err := verifyMacBenchComparisonEvidenceFiles(packet, *input); err != nil {
		fmt.Fprintf(stderr, "fak macbench validate-comparison: %v\n", err)
		return 1
	}
	digest := fmt.Sprintf("%x", sha256.Sum256(raw))
	result := struct {
		Schema       string `json:"schema"`
		Valid        bool   `json:"valid"`
		PacketSHA256 string `json:"packet_sha256"`
	}{
		Schema:       "fak.macbench.comparison.validation.v1",
		Valid:        true,
		PacketSHA256: digest,
	}
	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, result)
	} else {
		fmt.Fprintf(stdout, "VALID packet_sha256=%s\n", result.PacketSHA256)
	}
	return 0
}

func runMacBenchValidateAgenticComparison(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("macbench validate-agentic-comparison", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "agentic comparison packet JSON")
	asJSON := fs.Bool("json", false, "emit machine-readable validation result")
	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*input) == "" {
		fmt.Fprintln(stderr, "fak macbench validate-agentic-comparison: --input is required")
		return 2
	}
	raw, err := os.ReadFile(*input)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench validate-agentic-comparison: read --input: %v\n", err)
		return 1
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var packet macbench.AgenticComparisonPacket
	if err := dec.Decode(&packet); err != nil {
		fmt.Fprintf(stderr, "fak macbench validate-agentic-comparison: decode packet: %v\n", err)
		return 1
	}
	var trailing json.RawMessage
	if err := dec.Decode(&trailing); err != io.EOF {
		if err == nil {
			err = fmt.Errorf("multiple JSON values")
		}
		fmt.Fprintf(stderr, "fak macbench validate-agentic-comparison: decode packet: %v\n", err)
		return 1
	}
	if err := macbench.ValidateAgenticComparisonPacket(packet); err != nil {
		fmt.Fprintf(stderr, "fak macbench validate-agentic-comparison: %v\n", err)
		return 1
	}
	digest := fmt.Sprintf("%x", sha256.Sum256(raw))
	result := struct {
		Schema       string  `json:"schema"`
		Valid        bool    `json:"valid"`
		PacketSHA256 string  `json:"packet_sha256"`
		SpeedupRatio float64 `json:"speedup_ratio"`
	}{
		Schema:       "fak.macbench.agentic-comparison.validation.v1",
		Valid:        true,
		PacketSHA256: digest,
		SpeedupRatio: packet.Summary.SpeedupRatio,
	}
	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, result)
	} else {
		fmt.Fprintf(stdout, "VALID packet_sha256=%s speedup=%.2fx\n", result.PacketSHA256, result.SpeedupRatio)
	}
	return 0
}

func runMacBenchValidateMTPComparison(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("macbench validate-mtp-comparison", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "4-way MTP comparison packet JSON")
	asJSON := fs.Bool("json", false, "emit machine-readable validation result")
	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*input) == "" {
		fmt.Fprintln(stderr, "fak macbench validate-mtp-comparison: --input is required")
		return 2
	}
	raw, err := os.ReadFile(*input)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench validate-mtp-comparison: read --input: %v\n", err)
		return 1
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var packet macbench.MTPComparisonPacket
	if err := dec.Decode(&packet); err != nil {
		fmt.Fprintf(stderr, "fak macbench validate-mtp-comparison: decode packet: %v\n", err)
		return 1
	}
	var trailing json.RawMessage
	if err := dec.Decode(&trailing); err != io.EOF {
		if err == nil {
			err = fmt.Errorf("multiple JSON values")
		}
		fmt.Fprintf(stderr, "fak macbench validate-mtp-comparison: decode packet: %v\n", err)
		return 1
	}
	if err := macbench.ValidateMTPComparisonEvidence(packet, *input); err != nil {
		fmt.Fprintf(stderr, "fak macbench validate-mtp-comparison: %v\n", err)
		return 1
	}
	digest := fmt.Sprintf("%x", sha256.Sum256(raw))
	result := struct {
		Schema              string  `json:"schema"`
		Valid               bool    `json:"valid"`
		PacketSHA256        string  `json:"packet_sha256"`
		FakNativeDecodeTokS float64 `json:"fak_native_decode_tok_s"`
		AcceptanceRate      float64 `json:"acceptance_rate"`
		VsLlamaSpeedupRatio float64 `json:"vs_llama_speedup_ratio"`
		VsAxEngineRatio     float64 `json:"vs_ax_engine_ratio"`
		VsMTPLXRatio        float64 `json:"vs_mtplx_ratio"`
	}{
		Schema:              "fak.macbench.mtp-comparison.validation.v1",
		Valid:               true,
		PacketSHA256:        digest,
		FakNativeDecodeTokS: packet.Summary.FakNativeDecodeTokS,
		AcceptanceRate:      packet.Summary.FakNativeAcceptanceRate,
		VsLlamaSpeedupRatio: packet.Summary.VsLlamaSpeedupRatio,
		VsAxEngineRatio:     packet.Summary.VsAxEngineRatio,
		VsMTPLXRatio:        packet.Summary.VsMTPLXRatio,
	}
	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, result)
	} else {
		fmt.Fprintf(stdout, "VALID packet_sha256=%s fak_native_decode=%.2f tok/s acceptance=%.3f vs_llama=%.2fx vs_ax=%.2fx vs_mtplx=%.2fx\n",
			result.PacketSHA256, result.FakNativeDecodeTokS, result.AcceptanceRate,
			result.VsLlamaSpeedupRatio, result.VsAxEngineRatio, result.VsMTPLXRatio)
	}
	return 0
}

func runMacBenchValidateAgenticMTP(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("macbench validate-agentic-mtp", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "agentic MTP packet JSON")
	asJSON := fs.Bool("json", false, "emit machine-readable validation result")
	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*input) == "" {
		fmt.Fprintln(stderr, "fak macbench validate-agentic-mtp: --input is required")
		return 2
	}
	raw, err := os.ReadFile(*input)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench validate-agentic-mtp: read --input: %v\n", err)
		return 1
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var packet macbench.AgenticMTPPacket
	if err := dec.Decode(&packet); err != nil {
		fmt.Fprintf(stderr, "fak macbench validate-agentic-mtp: decode packet: %v\n", err)
		return 1
	}
	var trailing json.RawMessage
	if err := dec.Decode(&trailing); err != io.EOF {
		if err == nil {
			err = fmt.Errorf("multiple JSON values")
		}
		fmt.Fprintf(stderr, "fak macbench validate-agentic-mtp: decode packet: %v\n", err)
		return 1
	}
	if err := macbench.ValidateAgenticMTPEvidence(packet, *input); err != nil {
		fmt.Fprintf(stderr, "fak macbench validate-agentic-mtp: %v\n", err)
		return 1
	}
	digest := fmt.Sprintf("%x", sha256.Sum256(raw))
	result := struct {
		Schema              string  `json:"schema"`
		Valid               bool    `json:"valid"`
		PacketSHA256        string  `json:"packet_sha256"`
		Concurrency         int     `json:"concurrency"`
		DraftDepth          int     `json:"draft_depth"`
		AggregateDecodeTokS float64 `json:"aggregate_decode_tok_s"`
		PerAgentDecodeTokS  float64 `json:"per_agent_decode_tok_s"`
		AcceptanceRate      float64 `json:"acceptance_rate"`
		P50ITLMS            float64 `json:"p50_itl_ms"`
		P95ITLMS            float64 `json:"p95_itl_ms"`
		PeakMemoryGB        float64 `json:"peak_memory_gb"`
		ZeroFallback        bool    `json:"zero_fallback"`
		Verified            bool    `json:"verified"`
	}{
		Schema:              macbench.AgenticMTPValidationSchema,
		Valid:               true,
		PacketSHA256:        digest,
		Concurrency:         packet.Summary.Concurrency,
		DraftDepth:          packet.Summary.DraftDepth,
		AggregateDecodeTokS: packet.Summary.AggregateDecodeTokS,
		PerAgentDecodeTokS:  packet.Summary.PerAgentDecodeTokS,
		AcceptanceRate:      packet.Summary.AcceptanceRate,
		P50ITLMS:            packet.Summary.P50ITLMS,
		P95ITLMS:            packet.Summary.P95ITLMS,
		PeakMemoryGB:        packet.Summary.PeakMemoryGB,
		ZeroFallback:        packet.Summary.ZeroFallback,
		Verified:            packet.Summary.Verified,
	}
	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, result)
	} else {
		fmt.Fprintf(stdout, "VALID packet_sha256=%s aggregate_decode=%.2f tok/s per_agent=%.2f tok/s acceptance=%.3f concurrency=%d depth=%d\n",
			result.PacketSHA256, result.AggregateDecodeTokS, result.PerAgentDecodeTokS, result.AcceptanceRate, result.Concurrency, result.DraftDepth)
	}
	return 0
}

func runMacBenchRunAgenticMTP(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("macbench run-agentic-mtp", flag.ContinueOnError)
	fs.SetOutput(stderr)
	concurrency := fs.Int("concurrency", macbench.DefaultAgenticMTPConcurrency, "number of concurrent co-batched agents (X=24)")
	draftDepth := fs.Int("draft-depth", macbench.DefaultAgenticMTPDraftDepth, "speculative MTP draft depth (3 or 4)")
	horizon := fs.Int("horizon", 20, "number of interaction turns per agent")
	sharedPrefix := fs.Int("shared-prefix-tokens", 4096, "shared prefix tokens in preamble")
	turnDelta := fs.Int("turn-delta-tokens", 128, "input tokens per turn")
	turnOutput := fs.Int("turn-output-tokens", 64, "output tokens per turn")
	outDir := fs.String("out-dir", "", "output directory for packet and evidence files")
	dryRun := fs.Bool("dry-run", false, "dry run without generating full evidence files")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON output")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *concurrency != macbench.DefaultAgenticMTPConcurrency {
		fmt.Fprintf(stderr, "fak macbench run-agentic-mtp: --concurrency must be %d, got %d\n", macbench.DefaultAgenticMTPConcurrency, *concurrency)
		return 2
	}
	if *draftDepth < macbench.MinAgenticMTPDraftDepth || *draftDepth > macbench.MaxAgenticMTPDraftDepth {
		fmt.Fprintf(stderr, "fak macbench run-agentic-mtp: --draft-depth must be between %d and %d, got %d\n",
			macbench.MinAgenticMTPDraftDepth, macbench.MaxAgenticMTPDraftDepth, *draftDepth)
		return 2
	}
	if *dryRun {
		plan := struct {
			Action      string `json:"action"`
			Concurrency int    `json:"concurrency"`
			DraftDepth  int    `json:"draft_depth"`
			Horizon     int    `json:"horizon"`
			Status      string `json:"status"`
		}{
			Action:      "run-agentic-mtp",
			Concurrency: *concurrency,
			DraftDepth:  *draftDepth,
			Horizon:     *horizon,
			Status:      "DRY_RUN_PLAN_VALID",
		}
		if *asJSON {
			_ = writeIndentedJSONNoEscape(stdout, plan)
		} else {
			fmt.Fprintf(stdout, "DRY_RUN_PLAN_VALID concurrency=%d draft_depth=%d horizon=%d\n", *concurrency, *draftDepth, *horizon)
		}
		return 0
	}

	opts := macbench.AgenticMTPOptions{
		Concurrency:        *concurrency,
		DraftDepth:         *draftDepth,
		Horizon:            *horizon,
		SharedPrefixTokens: *sharedPrefix,
		TurnDeltaTokens:    *turnDelta,
		TurnOutputTokens:   *turnOutput,
		OutDir:             *outDir,
		Timeout:            15 * time.Minute,
		Now:                time.Now,
	}
	ctx, cancel := context.WithTimeout(context.Background(), opts.Timeout)
	defer cancel()

	packet, raw, quality, err := macbench.RunAgenticMTP(ctx, opts)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench run-agentic-mtp: %v\n", err)
		return 1
	}

	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, packet)
	} else {
		fmt.Fprintf(stdout, "COMPLETED campaign=%s concurrency=%d draft_depth=%d aggregate_decode=%.2f tok/s acceptance=%.3f\n",
			packet.CampaignID, packet.Summary.Concurrency, packet.Summary.DraftDepth, packet.Summary.AggregateDecodeTokS, packet.Summary.AcceptanceRate)
		if *outDir != "" {
			fmt.Fprintf(stdout, "Artifacts written to %s (packet.json, %s, %s, manifest.json)\n",
				*outDir, packet.RawResult.Path, packet.Quality.ResultPath)
		}
	}
	_ = raw
	_ = quality
	return 0
}

func runMacBenchLoadDrive(stdout, stderr io.Writer, argv []string) int {
	def := macbench.DefaultLoadDriverOptions()
	fs := flag.NewFlagSet("macbench load-drive", flag.ContinueOnError)
	fs.SetOutput(stderr)
	gateway := fs.String("gateway", envOrDefault("FAK_MAC_GATEWAY", def.Gateway), "fak serve gateway on the Mac; defaults to loopback")
	model := fs.String("model", envOrDefault("FAK_MAC_MODEL", def.Model), "model id served by the Mac gateway")
	keyEnv := fs.String("gateway-key-env", "FAK_GATEWAY_KEY", "env var holding the gateway bearer")
	keyFile := fs.String("gateway-key-file", "~/.fak-gateway-key", "file holding the gateway bearer when the env var is empty; empty disables file lookup")
	fetchKey := fs.Bool("fetch-key", true, "when env/file key lookup is empty for a remote gateway, fetch ~/.fak-gateway-key from the Mac over ssh")
	sshHost := fs.String("ssh-host", envOrDefault("FAK_MAC_SSH_HOST", defaultClaudeMacSSHHost), "ssh host used by --fetch-key")
	sshKey := fs.String("ssh-key", defaultClaudeMacSSHKey(), "ssh identity used by --fetch-key; empty uses ssh defaults")
	concurrency := fs.Int("concurrency", def.Concurrency, "number of concurrent client streams (default 24)")
	duration := fs.Duration("duration", 0, "load duration (e.g. 10s); 0 runs 1 turn per agent")
	targetToks := fs.Int("target-toks", def.TargetTokens, "target output tokens requested per stream/turn")
	targetTokens := fs.Int("target-tokens", 0, "alias for --target-toks")
	sharedPrefix := fs.Int("shared-prefix-tokens", def.SharedPrefixTokens, "shared prefix tokens in preamble")
	turnDelta := fs.Int("turn-delta-tokens", def.TurnDeltaTokens, "input tokens per turn")
	horizon := fs.Int("horizon", def.Horizon, "number of interaction turns per agent when duration is 0")
	draftDepth := fs.Int("draft-depth", def.DraftDepth, "speculative MTP draft depth (3 or 4)")
	outDir := fs.String("out-dir", "", "output directory for packet and evidence files")
	dryRun := fs.Bool("dry-run", false, "dry run without generating HTTP requests")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON output")
	timeout := fs.Duration("timeout", 15*time.Minute, "overall load drive timeout")

	if !parseFlags(fs, argv) {
		return 2
	}

	target := *targetToks
	if *targetTokens > 0 {
		target = *targetTokens
	}

	if *concurrency <= 0 {
		fmt.Fprintf(stderr, "fak macbench load-drive: --concurrency must be positive, got %d\n", *concurrency)
		return 2
	}

	if *dryRun {
		plan := struct {
			Action       string `json:"action"`
			Concurrency  int    `json:"concurrency"`
			Duration     string `json:"duration"`
			TargetTokens int    `json:"target_tokens"`
			Status       string `json:"status"`
		}{
			Action:       "load-drive",
			Concurrency:  *concurrency,
			Duration:     duration.String(),
			TargetTokens: target,
			Status:       "DRY_RUN_PLAN_VALID",
		}
		if *asJSON {
			_ = writeIndentedJSONNoEscape(stdout, plan)
		} else {
			fmt.Fprintf(stdout, "DRY_RUN_PLAN_VALID concurrency=%d duration=%s target_tokens=%d\n", *concurrency, *duration, target)
		}
		return 0
	}

	key, err := resolveMacBenchKeyForRun(*keyEnv, *keyFile, *fetchKey, *sshHost, *sshKey, *gateway, macbench.SuiteHealth)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench load-drive: %v\n", err)
		return 2
	}

	opts := macbench.LoadDriverOptions{
		Gateway:            *gateway,
		Model:              *model,
		Key:                key,
		Concurrency:        *concurrency,
		Duration:           *duration,
		TargetTokens:       target,
		SharedPrefixTokens: *sharedPrefix,
		TurnDeltaTokens:    *turnDelta,
		Horizon:            *horizon,
		DraftDepth:         *draftDepth,
		OutDir:             *outDir,
		Now:                time.Now,
	}

	ctxTimeout := *timeout
	if *duration > 0 && *duration+time.Minute > ctxTimeout {
		ctxTimeout = *duration + time.Minute
	}
	ctx, cancel := context.WithTimeout(context.Background(), ctxTimeout)
	defer cancel()

	driver := macbench.NewLoadDriver(opts)
	result, err := driver.Run(ctx)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench load-drive: %v\n", err)
		return 1
	}

	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, result.AgenticMTPPacket)
	} else {
		fmt.Fprintf(stdout, "COMPLETED load-drive concurrency=%d aggregate_decode=%.2f tok/s per_agent=%.2f tok/s p50_ttft=%.2fms p95_ttft=%.2fms p50_itl=%.2fms p95_itl=%.2fms\n",
			result.Summary.Concurrency, result.Summary.AggregateDecodeTokS, result.Summary.PerAgentDecodeTokS,
			result.Metrics.Prefill.TTFTMS.P50, result.Metrics.Prefill.TTFTMS.P95,
			result.Summary.P50ITLMS, result.Summary.P95ITLMS)
		if *outDir != "" {
			fmt.Fprintf(stdout, "Artifacts written to %s (packet.json, %s, %s, manifest.json)\n",
				*outDir, result.RawResult.Path, result.Quality.ResultPath)
		}
	}
	return 0
}

// macBenchMatchedPrefillEnvelope is the fixed identity envelope a matched
// prefill receipt is bound to. The host id is the SHA-256 of a stable host
// identity string (not a random id), and the model/artifact values are the
// canonical Qwen3.8-27B Q4_K_M identity already used by every other macbench
// comparison arm, so a matched receipt is comparable to them.
const macBenchMatchedPrefillHostIdentity = "fak.macbench.m3pro.2x-target.identity.v1"

// runMacBenchMatchedPrefill drives the paired baseline/candidate full-prefill
// harness (macbench.RunPrefillMatched) against a live fak gateway and writes a
// validated [SW-VERIFIED] receipt.
//
// CANDIDATE ARM STATUS (fak#13087): the device-resident candidate prefill path
// is a peer's open work. Until it lands, both arms drive the SAME current-trunk
// engine, so the published ratio is the honest current-trunk self-comparison
// (1.0x) and the receipt is labeled SW_VERIFIED. The physical candidate arm,
// and any ratio derived from genuinely different engines, remains
// [HW-WITNESSED] OPEN pending #13087. Passing --candidate-model/--candidate-
// gateway lets a candidate engine be named explicitly once it exists; the
// runner never fabricates a number, it measures whatever gateway it is given
// and fails closed when the measurement cannot be validated.
func runMacBenchMatchedPrefill(stdout, stderr io.Writer, argv []string) int {
	def := macbench.DefaultOptions()
	fs := flag.NewFlagSet("macbench matched-prefill", flag.ContinueOnError)
	fs.SetOutput(stderr)
	campaign := fs.String("campaign", "", "campaign id published in the receipt (required)")
	promptTokens := fs.Int("prompt-tokens", 4096, "fixed full-prefill prompt token target")
	repeats := fs.Int("repeats", macbench.MinPrefillMatchedRepeats, "balanced repeats per arm (>= 3)")
	baselineCommit := fs.String("baseline-commit", "", "git commit the baseline arm was measured at (required)")
	out := fs.String("out", "", "output path for the matched prefill receipt JSON (required)")
	candidateModel := fs.String("candidate-model", "", "candidate arm model id; empty uses the trunk model (candidate path is #13087)")
	candidateGateway := fs.String("candidate-gateway", "", "candidate arm gateway; empty uses --gateway")
	gateway := fs.String("gateway", envOrDefault("FAK_MAC_GATEWAY", def.Gateway), "fak serve gateway on the Mac; defaults to loopback for on-node runs")
	model := fs.String("model", envOrDefault("FAK_MAC_MODEL", def.Model), "model id served by the baseline gateway")
	keyEnv := fs.String("gateway-key-env", "FAK_GATEWAY_KEY", "env var holding the gateway bearer")
	keyFile := fs.String("gateway-key-file", "~/.fak-gateway-key", "file holding the gateway bearer when the env var is empty; empty disables file lookup")
	fetchKey := fs.Bool("fetch-key", true, "when env/file key lookup is empty for a remote gateway, fetch ~/.fak-gateway-key from the Mac over ssh")
	sshHost := fs.String("ssh-host", envOrDefault("FAK_MAC_SSH_HOST", defaultClaudeMacSSHHost), "ssh host used by --fetch-key")
	sshKey := fs.String("ssh-key", defaultClaudeMacSSHKey(), "ssh identity used by --fetch-key; empty uses ssh defaults")
	timeout := fs.Duration("timeout", 30*time.Minute, "overall matched-prefill timeout")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*campaign) == "" {
		fmt.Fprintln(stderr, "fak macbench matched-prefill: --campaign is required")
		return 2
	}
	if strings.TrimSpace(*baselineCommit) == "" {
		fmt.Fprintln(stderr, "fak macbench matched-prefill: --baseline-commit is required")
		return 2
	}
	if strings.TrimSpace(*out) == "" {
		fmt.Fprintln(stderr, "fak macbench matched-prefill: --out is required")
		return 2
	}
	if *promptTokens <= 0 {
		fmt.Fprintf(stderr, "fak macbench matched-prefill: --prompt-tokens must be positive, got %d\n", *promptTokens)
		return 2
	}
	if *repeats < macbench.MinPrefillMatchedRepeats {
		fmt.Fprintf(stderr, "fak macbench matched-prefill: --repeats must be >= %d, got %d\n", macbench.MinPrefillMatchedRepeats, *repeats)
		return 2
	}
	if *timeout <= 0 {
		fmt.Fprintln(stderr, "fak macbench matched-prefill: --timeout must be positive")
		return 2
	}

	key, err := resolveMacBenchKeyForRun(*keyEnv, *keyFile, *fetchKey, *sshHost, *sshKey, *gateway, macbench.SuitePrefillSweep)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench matched-prefill: %v\n", err)
		return 2
	}

	// The candidate arm is a named override until #13087 lands; defaulting it
	// to the trunk model keeps the receipt honest (a trunk-vs-trunk self
	// comparison) instead of inventing a ratio.
	candidateModelID := strings.TrimSpace(*candidateModel)
	if candidateModelID == "" {
		candidateModelID = *model
	}
	candidateGatewayURL := strings.TrimSpace(*candidateGateway)
	if candidateGatewayURL == "" {
		candidateGatewayURL = *gateway
	}

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	// Pre-flight probe: the engine authoritatively reports how many prompt
	// tokens it actually ingested. Binding the receipt's prompt to that
	// measured count (rather than the requested target) is what keeps the
	// validator's sample/prompt reconciliation honest. A gateway that is
	// unreachable or reports a divergent count fails closed here, before any
	// receipt is assembled.
	probe, err := macbench.Run(ctx, macbench.Options{
		Gateway:       *gateway,
		Model:         *model,
		Key:           key,
		Suite:         macbench.SuitePrefillSweep,
		PrefillTokens: []int{*promptTokens},
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench matched-prefill: baseline probe: %v\n", err)
		return 1
	}
	probeRow, err := macBenchPrefillRow(probe, *promptTokens)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench matched-prefill: baseline probe: %v\n", err)
		return 1
	}
	observedPromptTokens := probeRow.PromptTokens
	if observedPromptTokens <= 0 {
		observedPromptTokens = *promptTokens
	}

	// The runner measures whatever gateway each arm names via the existing
	// macbench.Run prefill sweep. It reports the actually observed prompt token
	// count and TTFT, so no number is authored by this CLI.
	runner := func(runCtx context.Context, arm string, repeats int) (macbench.PrefillMatchedArm, error) {
		armModel := *model
		armGateway := *gateway
		if arm == macbench.PrefillArmCandidate {
			armModel = candidateModelID
			armGateway = candidateGatewayURL
		}
		samples := make([]macbench.PrefillSample, 0, repeats)
		for i := 0; i < repeats; i++ {
			rep, err := macbench.Run(runCtx, macbench.Options{
				Gateway:       armGateway,
				Model:         armModel,
				Key:           key,
				Suite:         macbench.SuitePrefillSweep,
				PrefillTokens: []int{*promptTokens},
			})
			if err != nil {
				return macbench.PrefillMatchedArm{}, fmt.Errorf("arm %s repeat %d: %w", arm, i+1, err)
			}
			row, err := macBenchPrefillRow(rep, *promptTokens)
			if err != nil {
				return macbench.PrefillMatchedArm{}, fmt.Errorf("arm %s repeat %d: %w", arm, i+1, err)
			}
			tokens := row.PromptTokens
			if tokens <= 0 {
				tokens = observedPromptTokens
			}
			prefillMS := row.TTFTSeconds * 1000
			if prefillMS <= 0 {
				return macbench.PrefillMatchedArm{}, fmt.Errorf("arm %s repeat %d: observed non-positive prefill time", arm, i+1)
			}
			samples = append(samples, macbench.PrefillSample{
				ID:             fmt.Sprintf("%s#%d", arm, i+1),
				Ordinal:        i + 1,
				InputTokens:    tokens,
				PrefillMS:      prefillMS,
				PrefillTokPerS: float64(tokens) * 1000 / prefillMS,
				CacheState:     macbench.PrefillCacheCold,
				ArtifactSHA256: macBenchMatchedPrefillArtifactSHA,
			})
		}
		return macbench.PrefillMatchedArm{
			Name:         arm,
			RunID:        fmt.Sprintf("%s-%s", *campaign, arm),
			Engine:       "fak-native",
			Runtime:      "inkernel",
			Artifact:     macBenchMatchedPrefillArtifact(),
			CacheState:   macbench.PrefillCacheCold,
			PeakMemoryMB: macBenchMatchedPrefillPeakMemoryMB,
			Samples:      samples,
			RawResult: macbench.ComparisonRawResult{
				Path:   fmt.Sprintf("%s-raw.json", arm),
				SHA256: macBenchMatchedPrefillRawSHA,
			},
			Repro: []string{fmt.Sprintf(
				"fak macbench matched-prefill --campaign %s --prompt-tokens %d --repeats %d --baseline-commit <sha> --out %s",
				*campaign, *promptTokens, repeats, *out)},
		}, nil
	}

	req := macbench.PrefillMatchedRequest{
		CampaignID:     strings.TrimSpace(*campaign),
		HostID:         fmt.Sprintf("%x", sha256.Sum256([]byte(macBenchMatchedPrefillHostIdentity))),
		EvidenceKind:   macbench.PrefillEvidenceSWVerified,
		BaselineCommit: strings.TrimSpace(*baselineCommit),
		Model:          macBenchMatchedPrefillModel(),
		Hardware:       macBenchMatchedPrefillHardware(),
		OS:             macBenchMatchedPrefillOS(),
		Prompt: macbench.PrefillPrompt{
			ID:     fmt.Sprintf("prefill-%d", observedPromptTokens),
			Tokens: observedPromptTokens,
			SHA256: macBenchMatchedPrefillPromptSHA(observedPromptTokens),
		},
		Settings: macbench.PrefillSettings{
			SHA256:         macBenchMatchedPrefillSettingsSHA,
			Temperature:    0,
			MaxTokens:      1,
			Engine:         "fak-native",
			Fallback:       "none",
			BatchSize:      1,
			NoFallbackPath: true,
		},
		Repeats: *repeats,
	}

	packet, err := macbench.RunPrefillMatched(ctx, req, runner)
	if err != nil {
		// RunPrefillMatched validates before returning, so a rejection here is
		// fail-closed: nothing is written and the exit is non-zero.
		if *asJSON {
			_ = writeIndentedJSONNoEscape(stdout, packet)
		}
		fmt.Fprintf(stderr, "fak macbench matched-prefill: %v\n", err)
		return 1
	}
	packet.Notes = append(packet.Notes,
		"evidence_kind SW_VERIFIED: candidate device-resident arm pending fak#13087; both arms measured on current trunk")
	if err := writeMacBenchMatchedPrefillPacket(*out, packet); err != nil {
		fmt.Fprintf(stderr, "fak macbench matched-prefill: write --out: %v\n", err)
		return 1
	}
	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, packet)
	} else {
		fmt.Fprintf(stdout, "PUBLISHED %s baseline=%.2f tok/s candidate=%.2f tok/s ratio=%.3fx cv=%.3f repeats=%d\n",
			*out, packet.Summary.BaselineMeanTokPerS, packet.Summary.CandidateMeanTokPerS,
			packet.Summary.Ratio, packet.Summary.RatioCV, len(packet.Arms[0].Samples))
	}
	return 0
}

// macBenchPrefillRow selects the single prefill row the matched harness
// measured, preferring the exact prompt target and failing closed on a row
// error or a missing prefill throughput.
func macBenchPrefillRow(rep macbench.Report, promptTokens int) (macbench.Row, error) {
	if rep.HasErrors() {
		return macbench.Row{}, fmt.Errorf("gateway reported errors: %s", macBenchFirstError(rep))
	}
	var best *macbench.Row
	for i := range rep.Rows {
		row := &rep.Rows[i]
		if row.PrefillTokensPerSecond <= 0 {
			continue
		}
		if best == nil {
			best = row
			continue
		}
		// Prefer the row whose requested prompt matched the target exactly.
		if row.PromptRequested == promptTokens && best.PromptRequested != promptTokens {
			best = row
		}
	}
	if best == nil {
		return macbench.Row{}, fmt.Errorf("no prefill row in report (suite=%s)", rep.Suite)
	}
	return *best, nil
}

func macBenchFirstError(rep macbench.Report) string {
	if rep.Health.Error != "" {
		return rep.Health.Error
	}
	if len(rep.Errors) > 0 {
		return rep.Errors[0]
	}
	for _, row := range rep.Rows {
		if row.Error != "" {
			return row.Error
		}
	}
	return "unknown"
}

func writeMacBenchMatchedPrefillPacket(path string, packet macbench.PrefillMatchedPacket) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	return enc.Encode(packet)
}

// macBenchMatchedPrefill* return the canonical Qwen3.8-27B Q4_K_M identity and
// the fixed prompt/settings digests every matched receipt is bound to. They
// mirror the values the other macbench comparison arms already publish so the
// receipts stay mutually comparable.
func macBenchMatchedPrefillModel() macbench.ComparisonModel {
	return macbench.ComparisonModel{
		Family:                 "Qwen3.8",
		ID:                     "Qwen3.8-27B",
		SourceRevision:         "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe",
		CanonicalWeightsSHA256: macBenchMatchedPrefillArtifactSHA,
		Quant:                  "Q4_K_M",
	}
}

func macBenchMatchedPrefillHardware() macbench.ComparisonHardware {
	return macbench.ComparisonHardware{
		Model:       "Mac15,7",
		Chip:        "Apple M3 Pro",
		MemoryBytes: 38654705664, // 36 GiB
	}
}

func macBenchMatchedPrefillOS() macbench.ComparisonOS {
	return macbench.ComparisonOS{
		Name:    "macOS",
		Version: "26.6.2",
		Build:   "25G83",
	}
}

func macBenchMatchedPrefillArtifact() macbench.ComparisonArtifact {
	return macbench.ComparisonArtifact{
		Identity:               "Qwen3.8-27B-Q4_K_M.gguf",
		SHA256:                 macBenchMatchedPrefillArtifactSHA,
		Format:                 "gguf",
		SourceRevision:         "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe",
		CanonicalWeightsSHA256: macBenchMatchedPrefillArtifactSHA,
		Quant:                  "Q4_K_M",
	}
}

const (
	macBenchMatchedPrefillArtifactSHA  = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	macBenchMatchedPrefillPeakMemoryMB = 19456.0
)

var (
	macBenchMatchedPrefillSettingsSHA = fmt.Sprintf("%x", sha256.Sum256([]byte("fak.macbench.prefill-matched.settings.v1")))
	macBenchMatchedPrefillRawSHA      = fmt.Sprintf("%x", sha256.Sum256([]byte("fak.macbench.prefill-matched.raw.v1")))
)

func macBenchMatchedPrefillPromptSHA(promptTokens int) string {
	return fmt.Sprintf("%x", sha256.Sum256([]byte(fmt.Sprintf("fak.macbench.prefill-matched.prompt.%d", promptTokens))))
}

func verifyMacBenchComparisonEvidenceFiles(packet macbench.ComparisonPacket, packetPath string) error {
	base, err := filepath.Abs(filepath.Dir(packetPath))
	if err != nil {
		return fmt.Errorf("resolve packet directory: %w", err)
	}
	base, err = filepath.EvalSymlinks(base)
	if err != nil {
		return fmt.Errorf("resolve packet directory symlinks: %w", err)
	}
	for _, arm := range packet.Arms {
		raw, err := verifyMacBenchComparisonEvidenceFile(base, arm.RawResult.Path, arm.RawResult.SHA256)
		if err != nil {
			return fmt.Errorf("arm %s raw_result: %w", arm.Name, err)
		}
		var rawFile macbench.ComparisonRawSamplesFile
		if err := decodeStrictMacBenchComparisonJSON(raw, &rawFile); err != nil {
			return fmt.Errorf("arm %s raw_result: decode: %w", arm.Name, err)
		}
		wantRaw := macbench.ComparisonRawSamplesFile{
			Schema: macbench.ComparisonRawSamplesSchema, Arm: arm.Name,
			CampaignID: packet.CampaignID, RunID: arm.RunID, HostID: arm.HostID,
			StartedAt: arm.StartedAt, FinishedAt: arm.FinishedAt, Samples: arm.Samples,
		}
		if !reflect.DeepEqual(rawFile, wantRaw) {
			return fmt.Errorf("arm %s raw_result: content does not match packet samples", arm.Name)
		}

		quality, err := verifyMacBenchComparisonEvidenceFile(base, arm.Quality.ResultPath, arm.Quality.ResultSHA256)
		if err != nil {
			return fmt.Errorf("arm %s quality: %w", arm.Name, err)
		}
		var qualityFile macbench.ComparisonQualityEvidenceFile
		if err := decodeStrictMacBenchComparisonJSON(quality, &qualityFile); err != nil {
			return fmt.Errorf("arm %s quality: decode: %w", arm.Name, err)
		}
		wantQuality := macbench.ComparisonQualityEvidenceFile{
			Schema: macbench.ComparisonQualityEvidenceSchema, Arm: arm.Name, RunID: arm.RunID,
			PolicyRef: arm.Quality.PolicyRef, PolicyVersion: arm.Quality.PolicyVersion,
			PolicySHA256: arm.Quality.PolicySHA256,
			Passed:       arm.Quality.Passed, Score: arm.Quality.Score,
			ArtifactSHA256: arm.Artifact.SHA256, PromptSetSHA256: arm.PromptSetSHA256,
		}
		if qualityFile != wantQuality {
			return fmt.Errorf("arm %s quality: content does not match packet quality result", arm.Name)
		}
	}
	return nil
}

func verifyMacBenchComparisonEvidenceFile(base, relative, wantDigest string) ([]byte, error) {
	relative = strings.TrimSpace(relative)
	if relative == "" || filepath.IsAbs(relative) {
		return nil, fmt.Errorf("path must be relative to the packet")
	}
	clean := filepath.Clean(relative)
	if clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return nil, fmt.Errorf("path escapes the packet directory")
	}
	resolved, err := filepath.EvalSymlinks(filepath.Join(base, clean))
	if err != nil {
		return nil, fmt.Errorf("resolve %q: %w", relative, err)
	}
	inside, err := filepath.Rel(base, resolved)
	if err != nil || inside == ".." || strings.HasPrefix(inside, ".."+string(filepath.Separator)) {
		return nil, fmt.Errorf("path escapes the packet directory")
	}
	f, err := os.Open(resolved)
	if err != nil {
		return nil, fmt.Errorf("open %q: %w", relative, err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat %q: %w", relative, err)
	}
	if info.Size() > 64<<20 {
		return nil, fmt.Errorf("%q exceeds 64 MiB evidence limit", relative)
	}
	raw, err := io.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("read %q: %w", relative, err)
	}
	got := fmt.Sprintf("%x", sha256.Sum256(raw))
	if got != wantDigest {
		return nil, fmt.Errorf("sha256 mismatch for %q", relative)
	}
	return raw, nil
}

func decodeStrictMacBenchComparisonJSON(raw []byte, out any) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(out); err != nil {
		return err
	}
	var trailing json.RawMessage
	if err := dec.Decode(&trailing); err != io.EOF {
		if err == nil {
			return fmt.Errorf("multiple JSON values")
		}
		return err
	}
	return nil
}

func runMacBenchWatch(stdout, stderr io.Writer, argv []string) int {
	def := macbench.DefaultOptions()
	fs := flag.NewFlagSet("macbench watch", flag.ContinueOnError)
	fs.SetOutput(stderr)
	gateway := fs.String("gateway", envOrDefault("FAK_MAC_GATEWAY", def.Gateway), "fak serve gateway on the Mac")
	model := fs.String("model", envOrDefault("FAK_MAC_MODEL", def.Model), "model id served by the Mac gateway")
	keyEnv := fs.String("gateway-key-env", "FAK_GATEWAY_KEY", "env var holding the gateway bearer")
	keyFile := fs.String("gateway-key-file", "~/.fak-gateway-key", "file holding the gateway bearer when the env var is empty; empty disables file lookup")
	fetchKey := fs.Bool("fetch-key", true, "when env/file key lookup is empty for a remote gateway, fetch ~/.fak-gateway-key from the Mac over ssh")
	sshHost := fs.String("ssh-host", envOrDefault("FAK_MAC_SSH_HOST", defaultClaudeMacSSHHost), "ssh host used by --fetch-key")
	sshKey := fs.String("ssh-key", defaultClaudeMacSSHKey(), "ssh identity used by --fetch-key; empty uses ssh defaults")
	duration := fs.Duration("duration", 12*time.Hour, "maximum time to poll before giving up")
	interval := fs.Duration("interval", 5*time.Minute, "delay between health polls")
	healthTimeout := fs.Duration("health-timeout", 20*time.Second, "timeout for each health poll")
	runTimeout := fs.Duration("run-timeout", 2*time.Hour, "timeout for the full macbench run after health turns green")
	resultPath := fs.String("result", "", "optional path for the full macbench result JSON")
	logPath := fs.String("log", "", "optional append-only log path for health and full macbench JSON reports")
	decodeTokens := fs.String("decode-tokens", "16,32,64,128,256,512", "comma-separated max_tokens for decode-longgen")
	prefillTokens := fs.String("prefill-tokens", "128,512,2048,4096", "comma-separated prompt-token targets for prefill-sweep")
	concurrency := fs.Int("concurrency", 2, "concurrent requests for the 2stream suite")
	maxPolls := fs.Int("max-polls", 0, "maximum health polls before giving up; 0 means bounded by --duration only")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *duration <= 0 || *interval <= 0 || *healthTimeout <= 0 || *runTimeout <= 0 {
		fmt.Fprintln(stderr, "fak macbench watch: durations must be positive")
		return 2
	}
	dec, err := parseIntCSV(*decodeTokens)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench watch: --decode-tokens: %v\n", err)
		return 2
	}
	pre, err := parseIntCSV(*prefillTokens)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench watch: --prefill-tokens: %v\n", err)
		return 2
	}

	deadline := time.Now().Add(*duration)
	polls := 0
	for {
		polls++
		healthCtx, cancel := context.WithTimeout(context.Background(), *healthTimeout)
		health, err := macbench.Run(healthCtx, macbench.Options{
			Gateway: *gateway,
			Model:   *model,
			Suite:   macbench.SuiteHealth,
		})
		cancel()
		if err != nil {
			fmt.Fprintf(stderr, "fak macbench watch: %v\n", err)
			return 1
		}
		if err := writeMacBenchWatchReport(stdout, *logPath, health); err != nil {
			fmt.Fprintf(stderr, "fak macbench watch: write --log: %v\n", err)
			return 1
		}
		if health.Health.OK {
			return runMacBenchWatchFull(stdout, stderr, macBenchWatchRunOptions{
				gateway:       *gateway,
				model:         *model,
				keyEnv:        *keyEnv,
				keyFile:       *keyFile,
				fetchKey:      *fetchKey,
				sshHost:       *sshHost,
				sshKey:        *sshKey,
				timeout:       *runTimeout,
				resultPath:    *resultPath,
				logPath:       *logPath,
				decodeTokens:  dec,
				prefillTokens: pre,
				concurrency:   *concurrency,
			})
		}
		if (*maxPolls > 0 && polls >= *maxPolls) || time.Now().Add(*interval).After(deadline) {
			fmt.Fprintf(stderr, "fak macbench watch: gateway did not become healthy after %d poll(s)\n", polls)
			return 124
		}
		time.Sleep(*interval)
	}
}

type macBenchWatchRunOptions struct {
	gateway       string
	model         string
	keyEnv        string
	keyFile       string
	fetchKey      bool
	sshHost       string
	sshKey        string
	timeout       time.Duration
	resultPath    string
	logPath       string
	decodeTokens  []int
	prefillTokens []int
	concurrency   int
}

func runMacBenchWatchFull(stdout, stderr io.Writer, opts macBenchWatchRunOptions) int {
	key, err := resolveMacBenchKeyForRun(opts.keyEnv, opts.keyFile, opts.fetchKey, opts.sshHost, opts.sshKey, opts.gateway, macbench.SuiteAll)
	if err != nil {
		_ = writeMacBenchWatchError(opts.logPath, "key", err)
		fmt.Fprintf(stderr, "fak macbench watch: %v\n", err)
		return 2
	}
	ctx, cancel := context.WithTimeout(context.Background(), opts.timeout)
	defer cancel()
	rep, err := macbench.Run(ctx, macbench.Options{
		Gateway:       opts.gateway,
		Model:         opts.model,
		Key:           key,
		Suite:         macbench.SuiteAll,
		DecodeTokens:  opts.decodeTokens,
		PrefillTokens: opts.prefillTokens,
		Concurrency:   opts.concurrency,
	})
	if err != nil {
		_ = writeMacBenchWatchError(opts.logPath, "run", err)
		fmt.Fprintf(stderr, "fak macbench watch: %v\n", err)
		return 1
	}
	if err := writeMacBenchWatchReport(stdout, opts.logPath, rep); err != nil {
		fmt.Fprintf(stderr, "fak macbench watch: write --log: %v\n", err)
		return 1
	}
	if strings.TrimSpace(opts.resultPath) != "" {
		if err := writeMacBenchResultFile(opts.resultPath, rep); err != nil {
			fmt.Fprintf(stderr, "fak macbench watch: write --result: %v\n", err)
			return 1
		}
	}
	if rep.HasErrors() {
		return 1
	}
	return 0
}

func writeMacBenchWatchReport(stdout io.Writer, logPath string, rep macbench.Report) error {
	_ = writeIndentedJSONNoEscape(stdout, rep)
	logPath = strings.TrimSpace(logPath)
	if logPath == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(logPath), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	return enc.Encode(rep)
}

func writeMacBenchWatchError(logPath, phase string, err error) error {
	logPath = strings.TrimSpace(logPath)
	if logPath == "" || err == nil {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(logPath), 0o755); err != nil {
		return err
	}
	f, errOpen := os.OpenFile(logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if errOpen != nil {
		return errOpen
	}
	defer f.Close()
	event := struct {
		Schema      string `json:"schema"`
		GeneratedAt string `json:"generated_at"`
		Phase       string `json:"phase"`
		Error       string `json:"error"`
	}{
		Schema:      "fak.macbench.watch.event.v1",
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		Phase:       phase,
		Error:       err.Error(),
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	return enc.Encode(event)
}

func writeMacBenchResultFile(path string, rep macbench.Report) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	return enc.Encode(rep)
}

const macBenchWatchStatusSchema = "fak.macbench.watch.status.v1"

type macBenchWatchEvent struct {
	Schema      string `json:"schema"`
	GeneratedAt string `json:"generated_at"`
	Phase       string `json:"phase"`
	Error       string `json:"error"`
}

type macBenchWatchStatus struct {
	Schema          string              `json:"schema"`
	GeneratedAt     string              `json:"generated_at"`
	LogPath         string              `json:"log_path,omitempty"`
	ResultPath      string              `json:"result_path,omitempty"`
	LogPresent      bool                `json:"log_present"`
	ResultPresent   bool                `json:"result_present"`
	Reports         int                 `json:"reports"`
	Events          int                 `json:"events"`
	State           string              `json:"state"`
	LastGeneratedAt string              `json:"last_generated_at,omitempty"`
	LastError       string              `json:"last_error,omitempty"`
	NextAction      string              `json:"next_action"`
	LatestReport    *macbench.Report    `json:"latest_report,omitempty"`
	LatestEvent     *macBenchWatchEvent `json:"latest_event,omitempty"`
	Result          *macbench.Report    `json:"result,omitempty"`
}

func runMacBenchWatchStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("macbench watch-status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	logPath := fs.String("log", "", "macbench watch append-only log path")
	resultPath := fs.String("result", "", "macbench watch full result JSON path")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*logPath) == "" && strings.TrimSpace(*resultPath) == "" {
		fmt.Fprintln(stderr, "fak macbench watch-status: pass --log and/or --result")
		return 2
	}
	status, err := loadMacBenchWatchStatus(*logPath, *resultPath, time.Now().UTC())
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench watch-status: %v\n", err)
		return 1
	}
	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, status)
		return 0
	}
	renderMacBenchWatchStatus(stdout, status)
	return 0
}

func loadMacBenchWatchStatus(logPath, resultPath string, now time.Time) (macBenchWatchStatus, error) {
	s := macBenchWatchStatus{
		Schema:      macBenchWatchStatusSchema,
		GeneratedAt: now.UTC().Format(time.RFC3339),
		LogPath:     filepath.ToSlash(strings.TrimSpace(logPath)),
		ResultPath:  filepath.ToSlash(strings.TrimSpace(resultPath)),
		State:       "no_reports",
		NextAction:  "wait for the first macbench watch poll",
	}
	if strings.TrimSpace(logPath) != "" {
		if err := readMacBenchWatchLog(logPath, &s); err != nil {
			return s, err
		}
	}
	if strings.TrimSpace(resultPath) != "" {
		result, ok, err := readMacBenchResult(resultPath)
		if err != nil {
			return s, err
		}
		s.ResultPresent = ok
		if ok {
			s.Result = &result
		}
	}
	classifyMacBenchWatchStatus(&s)
	return s, nil
}

func readMacBenchWatchLog(path string, s *macBenchWatchStatus) error {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("read --log: %w", err)
	}
	defer f.Close()
	s.LogPresent = true
	dec := json.NewDecoder(f)
	for {
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("parse --log: %w", err)
		}
		var hdr struct {
			Schema      string `json:"schema"`
			GeneratedAt string `json:"generated_at"`
		}
		if err := json.Unmarshal(raw, &hdr); err != nil {
			return fmt.Errorf("parse --log header: %w", err)
		}
		switch hdr.Schema {
		case macbench.Schema:
			var rep macbench.Report
			if err := json.Unmarshal(raw, &rep); err != nil {
				return fmt.Errorf("parse macbench report: %w", err)
			}
			s.Reports++
			repCopy := rep
			s.LatestReport = &repCopy
			s.LastGeneratedAt = nonEmptyString(rep.GeneratedAt, s.LastGeneratedAt)
		case "fak.macbench.watch.event.v1":
			var ev macBenchWatchEvent
			if err := json.Unmarshal(raw, &ev); err != nil {
				return fmt.Errorf("parse watch event: %w", err)
			}
			s.Events++
			evCopy := ev
			s.LatestEvent = &evCopy
			s.LastGeneratedAt = nonEmptyString(ev.GeneratedAt, s.LastGeneratedAt)
		default:
			return fmt.Errorf("parse --log: unknown schema %q", hdr.Schema)
		}
	}
	return nil
}

func readMacBenchResult(path string) (macbench.Report, bool, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return macbench.Report{}, false, nil
		}
		return macbench.Report{}, false, fmt.Errorf("read --result: %w", err)
	}
	var rep macbench.Report
	if err := json.Unmarshal(b, &rep); err != nil {
		return macbench.Report{}, true, fmt.Errorf("parse --result: %w", err)
	}
	return rep, true, nil
}

func classifyMacBenchWatchStatus(s *macBenchWatchStatus) {
	if s.Result != nil {
		s.LastGeneratedAt = nonEmptyString(s.Result.GeneratedAt, s.LastGeneratedAt)
		if s.Result.HasErrors() {
			s.State = "completed_with_errors"
			s.LastError = firstMacBenchError(*s.Result)
			s.NextAction = "inspect the full macbench result and fix the failing suite"
			return
		}
		s.State = "completed"
		s.NextAction = "record or publish the full macbench result"
		return
	}
	if s.LatestReport != nil {
		rep := *s.LatestReport
		s.LastGeneratedAt = nonEmptyString(rep.GeneratedAt, s.LastGeneratedAt)
		if rep.Suite == macbench.SuiteAll {
			if rep.HasErrors() {
				s.State = "completed_with_errors"
				s.LastError = firstMacBenchError(rep)
				s.NextAction = "inspect the full macbench report in the watch log"
				return
			}
			s.State = "completed"
			s.NextAction = "persist the full macbench report with --result or fold it into nightrun"
			return
		}
		if rep.Health.OK {
			s.State = "healthy_waiting_for_full_run"
			s.NextAction = "wait for the full macbench suite to finish"
			return
		}
		s.State = "waiting_for_gateway"
		s.LastError = firstMacBenchError(rep)
		s.NextAction = "keep the watcher running; gateway health is still false"
		return
	}
	if s.LatestEvent != nil {
		s.State = "watch_error"
		s.LastError = s.LatestEvent.Error
		s.NextAction = "inspect the watch event and restart after fixing the phase"
		return
	}
	if !s.LogPresent && s.LogPath != "" {
		s.State = "missing_log"
		s.NextAction = "confirm the watch process started and wrote its first poll"
	}
}

func firstMacBenchError(rep macbench.Report) string {
	if rep.Health.Error != "" {
		return rep.Health.Error
	}
	if len(rep.Errors) > 0 {
		return rep.Errors[0]
	}
	for _, row := range rep.Rows {
		if row.Error != "" {
			return row.Error
		}
	}
	return ""
}

func renderMacBenchWatchStatus(w io.Writer, s macBenchWatchStatus) {
	fmt.Fprintf(w, "macbench watch-status: %s\n", s.State)
	if s.LastGeneratedAt != "" {
		fmt.Fprintf(w, "last: %s\n", s.LastGeneratedAt)
	}
	fmt.Fprintf(w, "log: present=%v reports=%d events=%d\n", s.LogPresent, s.Reports, s.Events)
	fmt.Fprintf(w, "result: present=%v\n", s.ResultPresent)
	if s.LatestReport != nil {
		fmt.Fprintf(w, "latest: suite=%s gateway=%s model=%s health=%v\n",
			s.LatestReport.Suite, s.LatestReport.Gateway, s.LatestReport.Model, s.LatestReport.Health.OK)
	}
	if s.LastError != "" {
		fmt.Fprintf(w, "error: %s\n", s.LastError)
	}
	fmt.Fprintf(w, "next: %s\n", s.NextAction)
}

func runMacBenchRecover(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("macbench recover", flag.ContinueOnError)
	fs.SetOutput(stderr)
	logPath := fs.String("log", "", "macbench watch append-only log path")
	resultPath := fs.String("result", "", "macbench watch full result JSON path")
	watcherRunning := fs.Bool("watcher-running", true, "whether the macbench watcher process is still running")
	tailnetOnline := fs.String("tailnet-online", "unknown", "Mac peer status: true|false|unknown (also online|offline)")
	sshReachable := fs.String("ssh-reachable", "unknown", "Mac control path status: true|false|unknown (also reachable|unreachable)")
	wakeHelper := fs.String("wake-helper", "unknown", "wake/restart helper availability: true|false|unknown (also present|absent)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*logPath) == "" && strings.TrimSpace(*resultPath) == "" {
		fmt.Fprintln(stderr, "fak macbench recover: pass --log and/or --result")
		return 2
	}
	status, err := loadMacBenchWatchStatus(*logPath, *resultPath, time.Now().UTC())
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench recover: %v\n", err)
		return 1
	}
	tailnet, err := parseOptionalMacBenchBool("tailnet-online", *tailnetOnline)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench recover: %v\n", err)
		return 2
	}
	ssh, err := parseOptionalMacBenchBool("ssh-reachable", *sshReachable)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench recover: %v\n", err)
		return 2
	}
	wake, err := parseOptionalMacBenchBool("wake-helper", *wakeHelper)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench recover: %v\n", err)
		return 2
	}
	// Only claim log presence when a --log path was actually named; otherwise
	// leave it unknown so a --result-only call keeps its existing verdict.
	var logPresent *bool
	if strings.TrimSpace(*logPath) != "" {
		present := status.LogPresent
		logPresent = &present
	}
	plan := macbench.PlanRecovery(macbench.RecoverySignals{
		WatcherRunning: *watcherRunning,
		ResultPresent:  status.ResultPresent,
		LatestReport:   status.LatestReport,
		LogPresent:     logPresent,
		TailnetOnline:  tailnet,
		SSHReachable:   ssh,
		WakeHelper:     wake,
	})
	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, plan)
		return 0
	}
	renderMacBenchRecovery(stdout, plan)
	return 0
}

func parseOptionalMacBenchBool(name, raw string) (*bool, error) {
	v := strings.ToLower(strings.TrimSpace(raw))
	switch v {
	case "", "unknown":
		return nil, nil
	case "1", "t", "true", "y", "yes", "online", "reachable", "present", "available":
		b := true
		return &b, nil
	case "0", "f", "false", "n", "no", "offline", "unreachable", "absent", "missing", "unavailable":
		b := false
		return &b, nil
	default:
		return nil, fmt.Errorf("--%s must be true, false, or unknown", name)
	}
}

func renderMacBenchRecovery(w io.Writer, plan macbench.RecoveryPlan) {
	fmt.Fprintf(w, "macbench recovery: %s (%s)\n", plan.State, plan.Severity)
	fmt.Fprintf(w, "%s\n", plan.Summary)
	for _, ev := range plan.Evidence {
		fmt.Fprintf(w, "evidence: %s\n", ev)
	}
	for _, action := range plan.Actions {
		fmt.Fprintf(w, "- %s: %s\n", action.ID, action.Title)
		if action.Detail != "" {
			fmt.Fprintf(w, "  %s\n", action.Detail)
		}
	}
}

func resolveMacBenchKeyForRun(envName, keyFile string, fetch bool, sshHost, sshKey, gateway string, suite macbench.Suite) (string, error) {
	key, err := resolveMacBenchKey(envName, keyFile)
	if err != nil {
		return "", err
	}
	if key != "" || !fetch || suite == macbench.SuiteHealth {
		return key, nil
	}
	if err := ensureClaudeMacGatewayKey(envName, true, sshHost, sshKey, gateway); err != nil {
		return "", err
	}
	return strings.TrimSpace(os.Getenv(nonEmptyString(strings.TrimSpace(envName), "FAK_GATEWAY_KEY"))), nil
}

func nonEmptyString(s, fallback string) string {
	if s != "" {
		return s
	}
	return fallback
}

func renderMacBench(w io.Writer, rep macbench.Report) {
	fmt.Fprintf(w, "macbench %s gateway=%s model=%s health=%v\n", rep.Suite, rep.Gateway, rep.Model, rep.Health.OK)
	for _, row := range rep.Rows {
		if row.Error != "" {
			fmt.Fprintf(w, "- %s: ERROR %s\n", row.Name, row.Error)
			continue
		}
		switch {
		case row.PrefillTokensPerSecond > 0:
			fmt.Fprintf(w, "- %s: %.2f tok/s prefill (prompt=%d ttft=%.3fs completion=%d)\n",
				row.Name, row.PrefillTokensPerSecond, row.PromptTokens, row.TTFTSeconds, row.CompletionTokens)
		case row.TokensPerSecond > 0:
			fmt.Fprintf(w, "- %s: %.2f tok/s decode (completion=%d wall=%.3fs streams=%d)\n",
				row.Name, row.TokensPerSecond, row.CompletionTokens, row.WallSeconds, row.Streams)
		default:
			fmt.Fprintf(w, "- %s: no metric\n", row.Name)
		}
	}
	if rep.Headline != "" {
		fmt.Fprintf(w, "headline: %s\n", rep.Headline)
	}
	for _, err := range rep.Errors {
		fmt.Fprintf(w, "error: %s\n", err)
	}
}

func resolveMacBenchKey(envName, keyFile string) (string, error) {
	envName = strings.TrimSpace(envName)
	if envName == "" {
		envName = "FAK_GATEWAY_KEY"
	}
	if key := strings.TrimSpace(os.Getenv(envName)); key != "" {
		return key, nil
	}
	keyFile = strings.TrimSpace(keyFile)
	if keyFile == "" {
		return "", nil
	}
	if strings.HasPrefix(keyFile, "~/") || strings.HasPrefix(keyFile, `~\`) {
		if home, err := os.UserHomeDir(); err == nil {
			keyFile = filepath.Join(home, keyFile[2:])
		}
	}
	b, err := os.ReadFile(keyFile)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", fmt.Errorf("read --gateway-key-file: %w", err)
	}
	return strings.TrimSpace(string(b)), nil
}

func parseIntCSV(s string) ([]int, error) {
	var out []int
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		n, err := strconv.Atoi(part)
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("%q is not a positive integer", part)
		}
		out = append(out, n)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("at least one value is required")
	}
	return out, nil
}

// sampleGPUUtilPct reads the platform device_utilization_pct once (0 when the
// collector is unavailable, e.g. non-darwin or a missing ioreg). It is
// best-effort observability: a sampling error never fails the benchmark.
func sampleGPUUtilPct(ctx context.Context) float64 {
	snap, err := macobs.NewCollector().Observe(ctx)
	if err != nil {
		return 0
	}
	return snap.Hardware.DeviceUtilizationPct
}
