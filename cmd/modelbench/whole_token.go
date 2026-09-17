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

	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
)

var (
	wholeTokenOut    = flag.String("whole-token-witness", "", "write a separate native Metal P32/greedy decode wall-clock witness; no PhaseProfiler")
	wholeTokenRoute  = flag.String("whole-token-expect", "whole-token", "required executed route: whole-token or per-layer (source-control arm)")
	wholeTokenPrompt = flag.String("whole-token-prompt-ids", "", "exact comma-separated 32-token prompt for the witness")
)

type wholeTokenReport struct {
	BindingSHA256 string                `json:"binding_sha256"`
	Host          nativeHostIdentity    `json:"host"`
	Operations    []wholeTokenOperation `json:"operations"`
	Schema        string                `json:"schema"`
	Engine        string                `json:"engine"`
	Device        string                `json:"device"`
	ExpectedRoute string                `json:"expected_route"`
	// SequencePath is the capability token the live session declared at build
	// time. Readback pins every operation receipt to it, so a serialized report
	// cannot be revalidated under a fabricated capability path.
	SequencePath string                       `json:"sequence_path"`
	Prompt       []int                        `json:"prompt_ids"`
	Tokens       []int                        `json:"forwarded_token_ids"`
	GreedyNext   []int                        `json:"greedy_next_token_ids"`
	Phases       map[string]float64           `json:"phase_seconds"`
	TotalSeconds float64                      `json:"total_seconds"`
	Prefill      model.WholeSequenceReceipt   `json:"prefill"`
	Decode       []model.WholeSequenceReceipt `json:"decode_operations,omitempty"`
	Handoff      model.WholeSequenceHandoff   `json:"handoff"`
	PeakRSSBytes uint64                       `json:"peak_rss_bytes"`
	Artifact     nativeFileIdentity           `json:"artifact"`
	Source       nativeSourceIdentity         `json:"source"`
	Binary       nativeFileIdentity           `json:"binary"`
	Controls     map[string]any               `json:"controls,omitempty"`
	Error        string                       `json:"error,omitempty"`
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
	if *f.gguf == "" || !*f.q4k || *f.decodeSteps < 1 || *f.decodeSteps > 64 {
		return errors.New("whole-token witness requires -gguf -q4k and 1..64 decode steps")
	}
	// Device admission is capability-gated, not Metal-gated: either the legacy
	// Metal lane (-metal -backend=legacy, whose resolved whole-sequence owner is
	// the Metal graph) or a named compute.Backend that advertises the
	// whole-sequence prefill seam (the TICKET-02 route). A legacy run without
	// -metal is refused by name rather than silently running the witness on a lane
	// it did not select; a -backend run must omit -metal because that flag selects
	// the legacy path.
	if *f.backendName == "legacy" {
		if !*f.metal {
			return errors.New("whole-token witness on the legacy lane requires -metal, or name a -backend that provides a whole-sequence prefill")
		}
	} else if *f.metal {
		return errors.New("-metal selects the legacy Metal path; omit it when naming -backend")
	}
	if rawDecodeEnabled() || *f.nativeProfileOut != "" || *f.nativeProfileCompare != "" || *f.nativeProfileReadback != "" || *f.qwenSwapOut != "" || *f.qwenSwapReadback != "" || *f.loadOnly || *f.smoke || *f.verify || *f.preflight || *f.phaseProfile || *f.workloadPath != "" || *f.checkpoint != "" || *f.resume != "" {
		return errors.New("whole-token witness is exclusive with other benchmark/profile/readback modes")
	}
	if *macbenchMTP || *macbenchMTPAlt || *macbenchMTPDryRun || *macbenchMTPDryRunAlt || *macbenchMTPReadback != "" || *macbenchMTPReadbackAlt != "" {
		return errors.New("whole-token witness is exclusive with MTP comparison modes")
	}
	return nil
}

// wholeTokenDeviceIdentity resolves the whole-token witness's reported device
// identity device-neutrally: the active session's selected compute.Backend when
// present, else the legacy metalgemm.DeviceName() fallback (Metal path,
// byte-identical). A nil session, or a session with no selected backend, is the
// legacy direct path and MUST fall back to keep its report byte-identical.
func wholeTokenDeviceIdentity(s *model.Session) string {
	if s == nil || s.Backend == nil {
		return metalgemm.DeviceName()
	}
	name := s.Backend.Name()
	tier := s.Backend.Tier()
	if tier == "" {
		return name
	}
	return name + "/" + tier
}

// runWholeToken owns an isolated session through cleanup. An unchanged terminal
// receipt is not evidence of another operation: decline/fallback must fail the
// candidate witness, even if an earlier Step successfully used the graph.
func runWholeToken(s *model.Session, prompt []int, steps int, route string, started time.Time, closeWeights func() error) (report wholeTokenReport, err error) {
	report = wholeTokenReport{Schema: "fak.whole-token-witness/1", Engine: "fak-native", Device: wholeTokenDeviceIdentity(s), ExpectedRoute: route, Prompt: append([]int(nil), prompt...), Phases: map[string]float64{}}
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
	if s == nil || s.M == nil || s.PhaseProfiler != nil || len(prompt) != 32 || steps < 1 || steps > 64 || (route != "whole-token" && route != "per-layer") {
		return report, errors.New("whole-token witness requires a fresh whole-sequence-capable session without profiler")
	}
	// Admission is capability-gated before any prompt byte is consumed: the
	// selected session must advertise a whole-sequence owner. A session with no
	// such capability is refused by name (a typed refusal), never silently run
	// through a per-layer fallback.
	capabilityPath, admissible := s.WholeSequenceCapability()
	if !admissible {
		return report, errors.New("whole-token witness requires a whole-sequence-capable session: no native whole-sequence owner is available for this build/device")
	}
	// The legacy lane is exactly the backend-nil resident-Q4_K Metal session; the
	// backend lane is any session whose selected compute.Backend advertises the
	// whole-sequence prefill seam.
	backendLane := s.Backend != nil
	if !backendLane && (!s.Q4K || !s.MetalQ4K) {
		return report, errors.New("whole-token witness on the legacy lane requires a fresh resident-Q4_K Metal session")
	}
	for _, id := range prompt {
		if id < 0 || id >= s.M.Cfg.VocabSize {
			return report, errors.New("prompt ID outside vocabulary")
		}
	}
	// Consume the backend-neutral whole-sequence seam: the witness never names the
	// concrete Qwen35 Metal receipt or backend types. The adapter behind this
	// interface is the unchanged Metal path on the legacy lane, and the backend
	// whole-sequence prefill route on the backend lane.
	sequence := s.WholeSequence()
	if path := sequence.WholeSequencePath(); path != "" {
		capabilityPath = path
	}
	report.SequencePath = capabilityPath
	t := time.Now()
	if err = sequence.EnableWholeSequence(); err != nil {
		return report, err
	}
	report.Phases["session_setup"] = time.Since(t).Seconds()
	t = time.Now()
	logits := s.Prefill(prompt)
	report.Phases["prefill"] = time.Since(t).Seconds()
	t = time.Now()
	if executed, e := sequence.FinalizeWholeSequence(); e != nil || !executed {
		return report, fmt.Errorf("resident handoff executed=%v: %w", executed, e)
	}
	report.Phases["finalize"] = time.Since(t).Seconds()
	report.Prefill = sequence.WholeSequenceReceipt()
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
		before := sequence.WholeSequenceReceipt()
		counts := sequence.WholeSequenceHandoffReceipt()
		base := s.SequencePosition()
		t = time.Now()
		logits = s.Step(id) // Includes the existing output head and cache bookkeeping.
		report.Phases["decode_with_head"] += time.Since(t).Seconds()
		t = time.Now()
		after := sequence.WholeSequenceReceipt()
		nextCounts := sequence.WholeSequenceHandoffReceipt()
		report.Handoff = nextCounts
		operation := wholeTokenOperation{WholeSequenceOperation: model.WholeSequenceOperation{Before: before, After: after, CountsBefore: counts, CountsAfter: nextCounts, CacheBefore: base, CacheAfter: s.SequencePosition()}}
		if backendLane {
			if err := validateWholeTokenBackendOperation(operation, route, sequence); err != nil {
				return report, err
			}
		} else if err := validateWholeTokenOperation(operation, route, sequence, report.SequencePath); err != nil {
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
	if backendLane {
		// The backend lane owns no per-token terminal Metal receipt; the honest
		// evidence is the whole-sequence route still reporting an executed,
		// available receipt after the finite decode loop, plus the finite-logits
		// check the greedy selection above already enforces at every step.
		final := sequence.WholeSequenceReceipt()
		if !final.Available || final.EvidenceState != sequence.WholeSequenceExecutedEvidence() {
			return report, errors.New("whole-token backend decode loop lost its qualifying whole-sequence route")
		}
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
			data, err = benchcli.MarshalReport(data)
		}
		if err == nil {
			err = os.WriteFile(*wholeTokenOut, append(data, '\n'), 0o600)
		}
		return errors.Join(runErr, err)
	})
}

// wholeTokenOperation embeds the backend-neutral operation pair retained in the
// serialized report so readback can reject stale receipts without relying on
// pointer identity after JSON decoding.
type wholeTokenOperation struct {
	model.WholeSequenceOperation
}

// validateWholeTokenOperation delegates to the backend-neutral model validator,
// reading the path and executed-evidence tokens from the seam rather than naming
// a concrete Qwen35/Metal constant. A nil sequence means readback of a serialized
// artifact: the caller supplies the capability token recorded at build time
// (report.SequencePath) so a fabricated path still fails.
func validateWholeTokenOperation(op wholeTokenOperation, route string, sequence model.WholeSequenceSession, recordedPath string) error {
	path := recordedPath
	executed := model.WholeSequenceEvidenceExecuted
	if sequence != nil {
		path = sequence.WholeSequencePath()
		executed = sequence.WholeSequenceExecutedEvidence()
	}
	return model.ValidateWholeSequenceOperation(route, path, executed, op.WholeSequenceOperation)
}

// validateWholeTokenBackendOperation is the backend lane's whole-token lockstep
// gate. A compute.Backend-selected session runs its prompt as one batched
// whole-sequence prefill and decodes through the HAL; it owns no per-token
// terminal Metal command-buffer receipt, so requiring the resident-Metal
// Tokens==1/CommandBuffers==1 lockstep would refuse a real backend. What this
// lane can honestly witness is: the sequence advanced exactly one position, the
// whole-sequence route still reports an executed/available receipt, and the
// prefill-level evidence is unchanged across the step (the route is a
// prefill-level fact, not a per-decode one). A backend that cannot hold the
// route fails FinalizeWholeSequence before this ever runs.
func validateWholeTokenBackendOperation(op wholeTokenOperation, route string, sequence model.WholeSequenceSession) error {
	if op.CacheAfter != op.CacheBefore+1 {
		return errors.New("whole-token backend Step did not advance exactly one sequence position")
	}
	switch route {
	case "whole-token":
		if !op.After.Available || op.After.EvidenceState != sequence.WholeSequenceExecutedEvidence() || !op.After.CompletedWait {
			return errors.New("whole-token backend Step lacks a live executed whole-sequence receipt; fallback is non-qualifying")
		}
		// The backend lane's receipt is prefill-level, so its tokens field tracks
		// the sequence position rather than a per-step command buffer. A zero-
		// progress step (no token advanced) is the corruption this rejects.
		if op.After.Tokens <= op.Before.Tokens {
			return errors.New("whole-token backend Step advanced no whole-sequence token")
		}
	case "per-layer":
		// The backend lane does not distinguish a per-layer source-control arm at
		// the receipt level; the route status is the same executed prefill. Only
		// the position advance is contract-bearing here.
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
	if report.ExpectedRoute == "whole-token" && report.SequencePath == "" {
		return errors.New("whole-token report records no capability path")
	}
	for _, op := range report.Operations {
		// A readback validates a serialized artifact: it pins each receipt to the
		// capability token the live session recorded at build time.
		if err := validateWholeTokenOperation(op, report.ExpectedRoute, nil, report.SequencePath); err != nil {
			return err
		}
	}
	return nil
}
