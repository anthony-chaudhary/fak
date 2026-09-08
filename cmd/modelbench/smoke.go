package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// newGGUFLoadProfiler enables default progress for lean and resident/streamed Q4_K
// GGUF loads. Only lean and streamed paths support detailed phase profiles.
// Resident Q4_K progress counts collected tensors, before packing and finalization.
// Returns nil when neither progress nor a supported detailed profile is requested.
func newGGUFLoadProfiler(f *benchFlags) *ggufload.LoadProfiler {
	profiledGGUF := *f.gguf != "" && (*f.lean || streamQ4KEnabled(f))
	wantLoadProfile := (*f.loadProfile || *f.loadProfileTrace || *f.phaseProfile) && profiledGGUF
	progressGGUF := *f.gguf != "" && (*f.lean || *f.q4k)
	wantProgress := *f.loadProgress && progressGGUF
	if !wantLoadProfile && !wantProgress {
		return nil
	}
	lp := ggufload.NewLoadProfiler()
	if wantProgress {
		lp.Progress = os.Stderr // stream load % to stderr so a large multi-minute load is not silent
		if *f.q4k && !streamQ4KEnabled(f) {
			fmt.Fprintln(lp.Progress, "fak: resident Q4_K progress counts collected GGUF tensors; Q8 packing and model finalization may continue after 100%")
		}
	}
	if *f.loadProfileTrace {
		lp.Trace = os.Stderr
		lp.Every = *f.loadProfileTraceEvery
	}
	return lp
}

func loadReportIdentity(f *benchFlags) map[string]any {
	return map[string]any{
		"source":              loadSource(*f.hf, *f.gguf, *f.dir, *f.lean, *f.q4k, streamQ4KEnabled(f)),
		"stream_q4k":          streamQ4KEnabled(f),
		"load_worker_control": currentLoadWorkerControl(),
	}
}

func ggufLoadProfileIdentity(f *benchFlags) (mode, source string) {
	if streamQ4KEnabled(f) {
		return "gguf-streamed-dense-q4k", loadSource(*f.hf, *f.gguf, *f.dir, *f.lean, *f.q4k, true)
	}
	return "gguf-lean-q8", *f.gguf
}

// Closed-vocabulary -smoke statuses.
const (
	smokeStatusLoaded        = "SMOKE_LOADED"         // load finished within the deadline
	smokeStatusTimeout       = "SMOKE_LOAD_TIMEOUT"   // load exceeded -smoke-deadline (aborted)
	smokeStatusOK            = "SMOKE_OK"             // forward ran and produced finite logits
	smokeStatusForwardFailed = "SMOKE_FORWARD_FAILED" // forward panicked or produced NaN/Inf
)

// smokeOutcome is the PURE deadline decision for the -smoke load: given whether the load finished
// and how long it took against the deadline, it returns the closed status. Factored out so the
// timeout logic is unit-testable without a real multi-minute load.
func smokeOutcome(done bool, elapsed, deadline time.Duration) string {
	if !done {
		return smokeStatusTimeout
	}
	if deadline > 0 && elapsed > deadline {
		return smokeStatusTimeout
	}
	return smokeStatusLoaded
}

// loadModelMaybeDeadline bounds smoke loads. Q4_K loaders cooperate with cancellation and
// return only after their workers and checkpoint reader are drained; other loaders retain the
// historical race-and-exit behavior.
func loadModelMaybeDeadline(f *benchFlags, lp *ggufload.LoadProfiler) (*model.Model, string, error) {
	if !*f.smoke || *f.smokeDeadline <= 0 {
		return loadModel(f, lp)
	}
	if *f.q4k {
		start := time.Now()
		ctx, cancel := context.WithTimeout(context.Background(), *f.smokeDeadline)
		defer cancel()
		m, name, err := loadModelContext(ctx, f, lp)
		elapsed := time.Since(start)
		if errors.Is(err, context.DeadlineExceeded) {
			smokeTimeoutReporter(f, elapsed)
			return nil, "", nil // unreachable in production: reportSmokeTimeout exits
		}
		return m, name, err
	}
	type loadRes struct {
		m    *model.Model
		name string
		err  error
	}
	ch := make(chan loadRes, 1)
	start := time.Now()
	go func() {
		m, name, err := loadModel(f, lp)
		ch <- loadRes{m, name, err}
	}()
	select {
	case r := <-ch:
		// Won the race within the deadline window.
		return r.m, r.name, r.err
	case <-time.After(*f.smokeDeadline):
		// The deadline fired first. smokeOutcome (the pure, tested classifier) names this
		// SMOKE_LOAD_TIMEOUT; report it and exit. The load goroutine is abandoned (the process
		// exits), so a load that would have run for an hour is bounded by -smoke-deadline.
		elapsed := time.Since(start)
		if smokeOutcome(false, elapsed, *f.smokeDeadline) == smokeStatusTimeout {
			reportSmokeTimeout(f, elapsed)
		}
		return nil, "", nil // unreachable: reportSmokeTimeout exits
	}
}

// reportSmokeTimeout emits the SMOKE_LOAD_TIMEOUT artifact (with the last progress visible on
// stderr from the load profiler) and exits non-zero.
var smokeTimeoutReporter = reportSmokeTimeout

func reportSmokeTimeout(f *benchFlags, elapsed time.Duration) {
	fmt.Fprintf(os.Stderr, "fak: -smoke load exceeded -smoke-deadline %s (%.0fs elapsed) — aborting\n", *f.smokeDeadline, elapsed.Seconds())
	report := map[string]any{
		"app_version":     appversion.Current(),
		"engine":          "fak modelbench smoke",
		"smoke_status":    smokeStatusTimeout,
		"elapsed_seconds": elapsed.Seconds(),
		"deadline":        f.smokeDeadline.String(),
	}
	for key, value := range loadReportIdentity(f) {
		report[key] = value
	}
	writeReport(f, report)
	os.Exit(1)
}

// allFinite reports whether every logit is a finite number — the cheapest proof a forward pass
// produced real output rather than NaN/Inf (a broken kernel or a config mismatch).
func allFinite(logits []float32) bool {
	if len(logits) == 0 {
		return false
	}
	for _, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return false
		}
	}
	return true
}

func loadSource(hf, gguf, dir string, lean, q4k, streamQ4K bool) string {
	if gguf != "" {
		if q4k {
			if streamQ4K {
				return gguf + " (streamed dense Q4_K)"
			}
			return gguf + " (resident Q4_K)"
		}
		return gguf
	}
	if hf == "" {
		return dir
	}
	if lean {
		return filepath.Join(hf, "model.safetensors") + " (quantize-at-load)"
	}
	return filepath.Join(hf, "model.safetensors")
}

func writeReport(f *benchFlags, report map[string]any) {
	b, _ := benchcli.MarshalReport(report)
	if *f.out != "" {
		if err := os.WriteFile(*f.out, b, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write:", err)
			f.exit(1)
		}
		fmt.Fprintln(os.Stderr, "wrote", *f.out)
		return
	}
	fmt.Println(string(b))
}

func phaseTable(p *model.PhaseProfile) string {
	if p == nil {
		return ""
	}
	n := 8
	if len(p.Phases) < n {
		n = len(p.Phases)
	}
	s := fmt.Sprintf("[fak phase] %s tokens=%d steps=%d total=%.1f ms bottleneck=%s\n",
		p.Mode, p.Tokens, p.Steps, p.TotalMS, p.Bottleneck)
	for i := 0; i < n; i++ {
		ph := p.Phases[i]
		s += fmt.Sprintf("  %-28s %7.1f ms %5.1f%% calls=%d\n", ph.Phase, ph.MS, ph.TimePct, ph.Calls)
	}
	return s
}
