package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"time"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
)

var (
	wholeTokenOut    = flag.String("whole-token-witness", "", "write a separate native Metal P32/greedy decode wall-clock witness; no PhaseProfiler")
	wholeTokenRoute  = flag.String("whole-token-expect", "whole-token", "required executed route: whole-token or per-layer (source-control arm)")
	wholeTokenPrompt = flag.String("whole-token-prompt-ids", "", "exact comma-separated 32-token prompt for the witness")
)

type wholeTokenReport struct {
	BindingSHA256 string                                    `json:"binding_sha256"`
	Host          nativeHostIdentity                        `json:"host"`
	Operations    []wholeTokenOperation                     `json:"operations"`
	Schema        string                                    `json:"schema"`
	Engine        string                                    `json:"engine"`
	Device        string                                    `json:"device"`
	ExpectedRoute string                                    `json:"expected_route"`
	Prompt        []int                                     `json:"prompt_ids"`
	Tokens        []int                                     `json:"forwarded_token_ids"`
	GreedyNext    []int                                     `json:"greedy_next_token_ids"`
	Phases        map[string]float64                        `json:"phase_seconds"`
	TotalSeconds  float64                                   `json:"total_seconds"`
	Prefill       model.Qwen35MetalForwardSequenceReceipt   `json:"prefill"`
	Decode        []model.Qwen35MetalForwardSequenceReceipt `json:"decode_operations,omitempty"`
	Handoff       model.Qwen35DecodeHandoffReceipt          `json:"handoff"`
	PeakRSSBytes  uint64                                    `json:"peak_rss_bytes"`
	Artifact      nativeFileIdentity                        `json:"artifact"`
	Source        nativeSourceIdentity                      `json:"source"`
	Binary        nativeFileIdentity                        `json:"binary"`
	Controls      map[string]any                            `json:"controls,omitempty"`
	Error         string                                    `json:"error,omitempty"`
}

func validateWholeTokenFlags(f *benchFlags) error {
	if *wholeTokenOut == "" {
		return nil
	}
	if *wholeTokenRoute != "whole-token" && *wholeTokenRoute != "per-layer" {
		return errors.New("whole-token-expect requires whole-token or per-layer")
	}
	ids, err := parsePromptIDs(*wholeTokenPrompt)
	if err != nil || len(ids) != 32 {
		return errors.New("whole-token witness requires exactly 32 explicit prompt IDs")
	}
	if *f.gguf == "" || !*f.q4k || !*f.metal || *f.backendName != "legacy" || *f.decodeSteps < 1 || *f.decodeSteps > 64 {
		return errors.New("whole-token witness requires -gguf -q4k -metal -backend=legacy and 1..64 decode steps")
	}
	if rawDecodeEnabled() || *f.nativeProfileOut != "" || *f.nativeProfileCompare != "" || *f.nativeProfileReadback != "" || *f.qwenSwapOut != "" || *f.qwenSwapReadback != "" || *f.loadOnly || *f.smoke || *f.verify || *f.preflight || *f.phaseProfile || *f.workloadPath != "" || *f.checkpoint != "" || *f.resume != "" {
		return errors.New("whole-token witness is exclusive with other benchmark/profile/readback modes")
	}
	if *macbenchMTP || *macbenchMTPAlt || *macbenchMTPDryRun || *macbenchMTPDryRunAlt || *macbenchMTPReadback != "" || *macbenchMTPReadbackAlt != "" {
		return errors.New("whole-token witness is exclusive with MTP comparison modes")
	}
	return nil
}

// runWholeToken owns an isolated session through cleanup. An unchanged terminal
// receipt is not evidence of another operation: decline/fallback must fail the
// candidate witness, even if an earlier Step successfully used the graph.
func runWholeToken(s *model.Session, prompt []int, steps int, route string, started time.Time, closeWeights func() error) (report wholeTokenReport, err error) {
	report = wholeTokenReport{Schema: "fak.whole-token-witness/1", Engine: "fak-native", Device: metalgemm.DeviceName(), ExpectedRoute: route, Prompt: append([]int(nil), prompt...), Phases: map[string]float64{}}
	report.Phases["load_setup"] = time.Since(started).Seconds()
	defer func() {
		if p := recover(); p != nil {
			err = fmt.Errorf("whole-token forward failed: %v", p)
		}
		t := time.Now()
		if s != nil {
			s.Close()
		}
		if closeWeights != nil {
			err = errors.Join(err, closeWeights())
		}
		report.Phases["cleanup"] = time.Since(t).Seconds()
		report.TotalSeconds = time.Since(started).Seconds()
		if err != nil {
			report.Error = err.Error()
		}
	}()
	if s == nil || s.M == nil || s.Backend != nil || !s.Q4K || !s.MetalQ4K || s.PhaseProfiler != nil || len(prompt) != 32 || steps < 1 || steps > 64 || (route != "whole-token" && route != "per-layer") {
		return report, errors.New("whole-token witness requires a fresh backend-nil Metal Q4_K session without profiler")
	}
	for _, id := range prompt {
		if id < 0 || id >= s.M.Cfg.VocabSize {
			return report, errors.New("prompt ID outside vocabulary")
		}
	}
	t := time.Now()
	if err = s.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
		return report, err
	}
	report.Phases["session_setup"] = time.Since(t).Seconds()
	t = time.Now()
	logits := s.Prefill(prompt)
	report.Phases["prefill"] = time.Since(t).Seconds()
	t = time.Now()
	if executed, e := s.FinalizeQwen35MetalGDNPreprojectedSequence(); e != nil || !executed {
		return report, fmt.Errorf("resident handoff executed=%v: %w", executed, e)
	}
	report.Phases["finalize"] = time.Since(t).Seconds()
	report.Prefill = s.Qwen35MetalForwardSequenceReceipt()
	if report.Prefill.Tokens != 32 || !report.Prefill.CompletedWait {
		return report, errors.New("missing physical P32 prefill receipt")
	}
	for i := 0; i < steps; i++ {
		t = time.Now()
		id, e := wholeTokenGreedy(logits)
		if e != nil {
			return report, e
		}
		report.Phases["host_selection_and_receipt_bookkeeping"] += time.Since(t).Seconds()
		before := s.Qwen35MetalForwardSequenceReceipt()
		counts := s.Qwen35DecodeHandoffReceipt()
		base := s.Cache.Len()
		t = time.Now()
		logits = s.Step(id) // Includes the existing output head and cache bookkeeping.
		report.Phases["decode_with_head"] += time.Since(t).Seconds()
		t = time.Now()
		after := s.Qwen35MetalForwardSequenceReceipt()
		nextCounts := s.Qwen35DecodeHandoffReceipt()
		report.Handoff = nextCounts
		operation := wholeTokenOperation{Before: before, After: after, CountsBefore: counts, CountsAfter: nextCounts, CacheBefore: base, CacheAfter: s.Cache.Len()}
		if err := validateWholeTokenOperation(operation, route); err != nil {
			return report, err
		}
		report.Operations = append(report.Operations, operation)
		if route == "whole-token" {
			report.Decode = append(report.Decode, after)
		}
		next, e := wholeTokenGreedy(logits)
		if e != nil {
			return report, e
		}
		report.Tokens = append(report.Tokens, id)
		report.GreedyNext = append(report.GreedyNext, next)
		report.Phases["host_selection_and_receipt_bookkeeping"] += time.Since(t).Seconds()
	}
	report.PeakRSSBytes, err = peakRSSBytes()
	return report, err
}

func wholeTokenGreedy(logits []float32) (int, error) {
	if len(logits) == 0 {
		return 0, errors.New("empty logits")
	}
	best := 0
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return 0, errors.New("non-finite logits")
		}
		if v > logits[best] {
			best = i
		}
	}
	return best, nil
}

func runWholeTokenCLI(f *benchFlags, m *model.Model, started time.Time, newSession func() *model.Session) error {
	// Bind all loaded weights, including non-streamed synthetic/diagnostic stores,
	// to the existing idempotent lifetime helper before any identity can fail.
	if !streamQ4KEnabled(f) {
		f.bindWeightCloser(m.CloseWeights)
	}
	return runWithTransferredWeightLifetime(f, func() error {
		artifact, err := fileIdentity(*f.gguf)
		if err != nil {
			return err
		}
		source, binary, err := captureNativeBuild()
		if err != nil {
			return err
		}
		if source.Modified {
			return errors.New("whole-token artifact witness requires an immutable committed build; dirty diff identity cannot bind untracked sources")
		}
		host, err := captureNativeHost()
		if err != nil {
			return err
		}
		prompt, err := parsePromptIDs(*wholeTokenPrompt)
		if err != nil {
			return err
		}
		report, runErr := runWholeToken(newSession(), prompt, *f.decodeSteps, *wholeTokenRoute, started, f.closeTransferredWeights)
		report.Artifact, report.Source, report.Binary, report.Host = artifact, source, binary, host
		report.Controls = loadReportIdentity(f)
		report.Controls["argv"] = append([]string(nil), os.Args[1:]...)
		environment := make(map[string]string)
		for _, key := range nativeProfileDeniedEnvironment {
			if value, present := os.LookupEnv(key); present {
				environment[key] = value
			}
		}
		for _, key := range []string{"FAK_METAL_STREAM_Q4K", "FAK_Q4K", nativeProfileSequenceSelector, nativeProfileDecodeHandoffControl} {
			if value, present := os.LookupEnv(key); present {
				environment[key] = value
			}
		}
		report.Controls["environment"] = environment
		options := make(map[string]string)
		flag.VisitAll(func(f *flag.Flag) { options[f.Name] = f.Value.String() })
		report.Controls["flags"] = options
		report.Controls["logical_cpus"] = runtime.NumCPU()
		report.Controls["gomaxprocs"] = runtime.GOMAXPROCS(0)
		report.Controls["workers"] = model.NumWorkers()
		report.Controls["q8_decode_workers"] = model.Q8DecodeWorkers()
		report.BindingSHA256, err = wholeTokenBinding(report)
		if err != nil {
			return errors.Join(runErr, err)
		}
		data, err := json.MarshalIndent(report, "", "  ")
		if err == nil {
			err = os.WriteFile(*wholeTokenOut, append(data, '\n'), 0o600)
		}
		return errors.Join(runErr, err)
	})
}

// Operation pairs are retained in the serialized report so readback can reject
// stale receipts without relying on pointer identity after JSON decoding.
type wholeTokenOperation struct {
	Before       model.Qwen35MetalForwardSequenceReceipt `json:"before"`
	After        model.Qwen35MetalForwardSequenceReceipt `json:"after"`
	CountsBefore model.Qwen35DecodeHandoffReceipt        `json:"counts_before"`
	CountsAfter  model.Qwen35DecodeHandoffReceipt        `json:"counts_after"`
	CacheBefore  int                                     `json:"cache_before"`
	CacheAfter   int                                     `json:"cache_after"`
}

func validateWholeTokenOperation(op wholeTokenOperation, route string) error {
	before, err := sha256JSON(op.Before)
	if err != nil {
		return err
	}
	after, err := sha256JSON(op.After)
	if err != nil {
		return err
	}
	if op.CacheAfter != op.CacheBefore+1 {
		return errors.New("Step did not advance exactly one cache position")
	}
	a, c, n := op.After, op.CountsBefore, op.CountsAfter
	switch route {
	// The promoted trunk whole-token route records its acceptance on
	// BlockAcceptedCalls and rewrites the HAL forward receipt to a fresh
	// Tokens=1 executed receipt. Requiring ResidentGDNAcceptedCalls to advance
	// would wrongly reject that real route; requiring it to stay unchanged
	// still rejects a per-layer GDN fallback.
	case "whole-token":
		if before == after || a.Tokens != 1 || !a.Committed || !a.CompletedWait || a.CommandBuffers != 1 || a.TerminalWaits != 1 || a.TerminalReadbacks != 1 || a.IntermediateReadbacks != 0 || a.IntermediateWaits != 0 || a.Path != model.Qwen35MetalGDNSequenceForwardPath || a.EvidenceState != model.Qwen35MetalSequenceEvidenceExecuted || n.BlockAcceptedCalls != c.BlockAcceptedCalls+1 || n.ResidentGDNAcceptedCalls != c.ResidentGDNAcceptedCalls || n.MixerAcceptedCalls != c.MixerAcceptedCalls {
			return errors.New("Step lacks a fresh successful whole-token receipt; fallback is non-qualifying")
		}
	case "per-layer":
		if before != after || n.BlockAcceptedCalls != c.BlockAcceptedCalls+1 || n.MixerAcceptedCalls != c.MixerAcceptedCalls || n.ResidentGDNAcceptedCalls != c.ResidentGDNAcceptedCalls {
			return errors.New("source-control Step did not execute exactly one prior AUTO per-layer block route")
		}
	default:
		return errors.New("unknown whole-token route")
	}
	return nil
}

func wholeTokenBinding(report wholeTokenReport) (string, error) {
	report.BindingSHA256 = ""
	return sha256JSON(report)
}

// This validates an individual serialized observation, not source A/B
// comparability. Comparative qualification requires the separate pairing gate.
func validateWholeTokenReport(report wholeTokenReport) error {
	digest, err := wholeTokenBinding(report)
	if err != nil {
		return err
	}
	if report.BindingSHA256 == "" || digest != report.BindingSHA256 {
		return errors.New("whole-token report binding mismatch")
	}
	if report.Error != "" {
		return errors.New("whole-token report records failed execution")
	}
	if len(report.Operations) == 0 || len(report.Operations) != len(report.Tokens) {
		return errors.New("whole-token operations missing")
	}
	for _, op := range report.Operations {
		if err := validateWholeTokenOperation(op, report.ExpectedRoute); err != nil {
			return err
		}
	}
	return nil
}
