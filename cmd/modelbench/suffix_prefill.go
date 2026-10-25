package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

var suffixBuildIdentity = captureNativeBuild

var suffixPrefillInput = flag.String("suffix-prefill-workload", "", "bounded JSON prefix_ids/suffix_ids/next_token/repetitions/absolute_tolerance/relative_tolerance; real GGUF suffix witness")
var suffixPrefillOutput = flag.String("suffix-prefill-out", "", "exclusive-create JSON receipt for suffix-prefill-workload (session execution failures only; load/build-identity failures precede receipt)")

type suffixPrefillWorkload struct {
	Prefix            []int   `json:"prefix_ids"`
	Suffix            []int   `json:"suffix_ids"`
	Next              int     `json:"next_token"`
	Repetitions       int     `json:"repetitions"`
	AbsoluteTolerance float64 `json:"absolute_tolerance"`
	RelativeTolerance float64 `json:"relative_tolerance"`
}

func (w suffixPrefillWorkload) validate(vocab int) error {
	if len(w.Prefix) == 0 || len(w.Suffix) == 0 || len(w.Prefix)+len(w.Suffix) > 16384 || w.Repetitions < 1 || w.Repetitions > 10 {
		return errors.New("require nonempty prefix/suffix, at most 16384 combined tokens and 1..10 repetitions")
	}
	for _, v := range []float64{w.AbsoluteTolerance, w.RelativeTolerance} {
		if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			return errors.New("tolerances must be finite and nonnegative")
		}
	}
	for _, ids := range [][]int{w.Prefix, w.Suffix, {w.Next}} {
		for _, id := range ids {
			if id < 0 || (vocab > 0 && id >= vocab) {
				return fmt.Errorf("token ID %d outside vocabulary", id)
			}
		}
	}
	return nil
}

func prepareSuffixPrefill(f *benchFlags) (*suffixPrefillWorkload, error) {
	if *suffixPrefillInput == "" && *suffixPrefillOutput == "" {
		return nil, nil
	}
	if *suffixPrefillInput == "" || *suffixPrefillOutput == "" || *f.gguf == "" {
		return nil, errors.New("suffix witness requires -gguf, -suffix-prefill-workload and -suffix-prefill-out")
	}
	if *wholeTokenOut != "" || rawDecodeEnabled() || *f.nativeProfileOut != "" || *f.nativeProfileCompare != "" || *f.nativeProfileReadback != "" || *f.qwenSwapOut != "" || *f.qwenSwapReadback != "" || *f.loadOnly || *f.smoke || *f.verify || *f.preflight || *f.phaseProfile || *f.workloadPath != "" || *f.checkpoint != "" || *f.resume != "" || *macbenchMTP || *macbenchMTPAlt || *macbenchMTPDryRun || *macbenchMTPDryRunAlt || *macbenchMTPReadback != "" || *macbenchMTPReadbackAlt != "" {
		return nil, errors.New("suffix witness is exclusive with other benchmark/profile/readback modes")
	}
	out, err := filepath.Abs(*suffixPrefillOutput)
	if err != nil {
		return nil, err
	}
	for _, p := range []string{*suffixPrefillInput, *f.gguf} {
		a, e := filepath.Abs(p)
		if e != nil {
			return nil, e
		}
		if a == out {
			return nil, errors.New("receipt must not replace input or checkpoint")
		}
	}
	if _, err := os.Lstat(out); err == nil {
		return nil, errors.New("receipt already exists")
	} else if !os.IsNotExist(err) {
		return nil, err
	}
	in, err := os.Open(*suffixPrefillInput)
	if err != nil {
		return nil, err
	}
	defer in.Close()
	data, err := io.ReadAll(io.LimitReader(in, (1<<20)+1))
	if err != nil {
		return nil, err
	}
	if len(data) > 1<<20 {
		return nil, errors.New("workload exceeds 1 MiB")
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var w suffixPrefillWorkload
	if err := dec.Decode(&w); err != nil {
		return nil, err
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		return nil, errors.New("workload must contain exactly one JSON value")
	}
	return &w, w.validate(0)
}

type suffixPrefillSession interface {
	Prefill([]int) []float32
	Step(int) []float32
	SequencePosition() int
	Close()
}

type suffixGPUObservation struct {
	Status        string                 `json:"status"`
	TimerDomain   string                 `json:"timer_domain"`
	Scope         string                 `json:"scope"`
	Correlation   string                 `json:"event_correlation"`
	StartUnixNS   int64                  `json:"host_start_unix_ns"`
	EndUnixNS     int64                  `json:"host_end_unix_ns"`
	Profile       *compute.KernelProfile `json:"kernel_timestamp_delta,omitempty"`
	HostBreakdown string                 `json:"enqueue_wait_sync_breakdown"`
}

func suffixGPUWindow(before, after *compute.KernelProfile, backend string, start, end time.Time) suffixGPUObservation {
	o := suffixGPUObservation{Status: "unavailable", TimerDomain: "backend GPU timestamp profile; frequency/resolution/calibration unavailable in API; kernel sum is not GPU wall time", Scope: "process-global; concurrent device work can contaminate this window", Correlation: "phase boundaries only; no session or event identifiers exposed", StartUnixNS: start.UnixNano(), EndUnixNS: end.UnixNano(), HostBreakdown: "enqueue/wait/synchronization not separately exposed; transfer rows only when backend records them"}
	if backend == "vulkan" {
		o.TimerDomain = "Vulkan bottom-of-pipe intervals include intervening barriers/copies; not isolated kernel-active time or production critical path; frequency/resolution/calibration unavailable in Go API"
		o.Scope += "; FAK_VULKAN_TIMESTAMP_PROFILE=1 forces serialized diagnostic batches and disables pipelined batch cycles even if timestamps unavailable; compare optimization arms only under identical profiling controls (vulkan_shim.cpp:1609-1616)"
	}
	p := compute.KernelProfileDelta(before, after)
	if p == nil || p.Backend != backend || p.Ops == 0 {
		return o
	}
	o.Profile = p
	o.Status = "observed; coverage must be inspected"
	if p.Truncated > 0 || p.UntimedCompute > 0 || p.UntimedTransfer > 0 || p.BarriersElided > 0 {
		o.Status = "partial; truncated/untimed/barrier coverage gaps"
	}
	return o
}

type suffixPrefillSample struct {
	Backend      string                          `json:"resolved_backend"`
	Device       string                          `json:"device_identity"`
	GPU          map[string]suffixGPUObservation `json:"gpu_phases"`
	PrefixNanos  int64                           `json:"prefix_ns"`
	SuffixNanos  int64                           `json:"suffix_ns"`
	NextNanos    int64                           `json:"next_step_ns"`
	TotalNanos   int64                           `json:"total_ns"`
	SuffixDigest string                          `json:"suffix_logits_sha256"`
	NextDigest   string                          `json:"next_logits_sha256"`
	SuffixMaxAbs float64                         `json:"suffix_max_abs_error"`
	NextMaxAbs   float64                         `json:"next_max_abs_error"`
	Error        string                          `json:"error,omitempty"`
}

type suffixPrefillReport struct {
	TimingScope           string                `json:"timing_scope"`
	Schema                string                `json:"schema"`
	Model                 string                `json:"model"`
	Config                model.Config          `json:"model_config"`
	Artifact              nativeFileIdentity    `json:"artifact"`
	ArtifactCoverage      string                `json:"artifact_identity_coverage"`
	ArtifactRequirement   string                `json:"artifact_identity_requirement"`
	HashingEffect         string                `json:"identity_hashing_effect"`
	Source                nativeSourceIdentity  `json:"source"`
	Binary                nativeFileIdentity    `json:"binary"`
	Backend               string                `json:"requested_backend"`
	Quant                 bool                  `json:"quant"`
	Q4K                   bool                  `json:"q4k"`
	Metal                 bool                  `json:"metal"`
	GOOS                  string                `json:"goos"`
	GOARCH                string                `json:"goarch"`
	GOMAXPROCS            int                   `json:"gomaxprocs"`
	SuffixControl         string                `json:"suffix_control"`
	Route                 string                `json:"execution_route"`
	Workload              suffixPrefillWorkload `json:"workload"`
	Reference             suffixPrefillSample   `json:"serial_step_numerical_control"`
	Controls              map[string]string     `json:"controls"`
	ReferenceSuffixDigest string                `json:"reference_suffix_logits_sha256"`
	ReferenceNextDigest   string                `json:"reference_next_logits_sha256"`
	Samples               []suffixPrefillSample `json:"samples"`
	Error                 string                `json:"error,omitempty"`
}

// No route receipt is exposed by the current model API. These measurements
// establish actual seeded Session calls, never execution of a requested fast path.
func suffixPrefillPass(factory func() suffixPrefillSession, w suffixPrefillWorkload, serial bool, vocab int) (sample suffixPrefillSample, suffix, next []float32, err error) {
	return suffixPrefillPassWithClock(factory, w, serial, vocab, time.Now)
}

func suffixPrefillPassWithClock(factory func() suffixPrefillSession, w suffixPrefillWorkload, serial bool, vocab int, now func() time.Time) (sample suffixPrefillSample, suffix, next []float32, err error) {
	started := now()
	var s suffixPrefillSession
	completed := false
	defer func() {
		p := recover()
		if !completed && (p != nil || err == nil) {
			err = errors.Join(err, fmt.Errorf("forward panic or incomplete call: %v", p))
		}
		if s != nil {
			func() {
				closed := false
				defer func() {
					p := recover()
					if !closed {
						err = errors.Join(err, fmt.Errorf("session cleanup panic: %v", p))
					}
				}()
				s.Close()
				closed = true
			}()
		}
		sample.TotalNanos = now().Sub(started).Nanoseconds()
		if err != nil {
			sample.Error = err.Error()
		}
	}()
	s = factory()
	if s == nil {
		return sample, nil, nil, errors.New("nil session")
	}
	sample.Backend = "unknown"
	sample.Device = "unavailable"
	if actual, ok := s.(*model.Session); ok {
		sample.Backend = "cpu-legacy"
		sample.Device = "host"
		if actual.Backend != nil {
			sample.Backend = actual.Backend.Name()
			sample.Device = fmt.Sprintf("backend tier %v; physical device ID unavailable", actual.Backend.Tier())
		} else if actual.Metal {
			sample.Backend = "metal"
			sample.Device = wholeTokenDeviceIdentity(actual)
		}
	}
	sample.GPU = map[string]suffixGPUObservation{}
	if s.SequencePosition() != 0 {
		return sample, nil, nil, errors.New("session is not fresh")
	}
	gpuBefore := compute.KernelProfileSnapshot()
	t := now()
	prefix := s.Prefill(w.Prefix)
	end := now()
	sample.PrefixNanos = end.Sub(t).Nanoseconds()
	sample.GPU["prefix"] = suffixGPUWindow(gpuBefore, compute.KernelProfileSnapshot(), sample.Backend, t, end)
	if err = checkSuffixLogits(prefix, vocab); err != nil {
		return
	}
	if s.SequencePosition() != len(w.Prefix) {
		err = errors.New("prefix cursor mismatch")
		return
	}
	gpuBefore = compute.KernelProfileSnapshot()
	t = now()
	if serial {
		for _, id := range w.Suffix {
			suffix = s.Step(id)
			if err = checkSuffixLogits(suffix, vocab); err != nil {
				return
			}
		}
	} else {
		suffix = s.Prefill(w.Suffix)
	}
	end = now()
	sample.SuffixNanos = end.Sub(t).Nanoseconds()
	sample.GPU["suffix"] = suffixGPUWindow(gpuBefore, compute.KernelProfileSnapshot(), sample.Backend, t, end)
	if err = checkSuffixLogits(suffix, vocab); err != nil {
		return
	}
	suffix = append([]float32(nil), suffix...)
	if s.SequencePosition() != len(w.Prefix)+len(w.Suffix) {
		err = errors.New("suffix cursor mismatch")
		return
	}
	gpuBefore = compute.KernelProfileSnapshot()
	t = now()
	next = s.Step(w.Next)
	end = now()
	sample.NextNanos = end.Sub(t).Nanoseconds()
	sample.GPU["next_step"] = suffixGPUWindow(gpuBefore, compute.KernelProfileSnapshot(), sample.Backend, t, end)
	if err = checkSuffixLogits(next, vocab); err != nil {
		return
	}
	next = append([]float32(nil), next...)
	if s.SequencePosition() != len(w.Prefix)+len(w.Suffix)+1 {
		err = errors.New("next cursor mismatch")
		return
	}
	sample.SuffixDigest, err = sha256JSON(suffix)
	if err != nil {
		return
	}
	sample.NextDigest, err = sha256JSON(next)
	completed = true
	return
}

func checkSuffixLogits(v []float32, vocab int) error {
	if len(v) != vocab || vocab < 1 {
		return errors.New("empty or wrong-sized logits")
	}
	for _, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return errors.New("nonfinite logits")
		}
	}
	return nil
}

func compareSuffixLogits(got, want []float32, w suffixPrefillWorkload) (float64, error) {
	if len(got) != len(want) || len(got) == 0 {
		return 0, errors.New("logit shape mismatch")
	}
	maxErr := 0.0
	for i, x := range got {
		d := math.Abs(float64(x) - float64(want[i]))
		maxErr = math.Max(maxErr, d)
		if math.IsNaN(d) || math.IsInf(d, 0) || d > w.AbsoluteTolerance+w.RelativeTolerance*math.Abs(float64(want[i])) {
			return maxErr, fmt.Errorf("logit %d exceeds declared tolerance", i)
		}
	}
	return maxErr, nil
}

func measureSuffixPrefill(factory func() suffixPrefillSession, w suffixPrefillWorkload, vocab int) (report suffixPrefillReport, err error) {
	return measureSuffixPrefillWithClock(factory, w, vocab, time.Now)
}

func measureSuffixPrefillWithClock(factory func() suffixPrefillSession, w suffixPrefillWorkload, vocab int, now func() time.Time) (report suffixPrefillReport, err error) {
	report.Schema = "fak.suffix-prefill/1"
	report.TimingScope = "host phase wall durations; sample total includes session setup, prefix, suffix, next Step, validation, profiling reads and cleanup; excludes model load, artifact hashing and serial numerical control"
	report.Workload = w
	report.Route = "unverified; candidate-only unpaired timings cannot be attributed to batching"
	defer func() {
		if err != nil {
			report.Error = err.Error()
		}
	}()
	if err = w.validate(vocab); err != nil {
		return
	}
	reference, refSuffix, refNext, e := suffixPrefillPassWithClock(factory, w, true, vocab, now)
	report.Reference = reference
	if e != nil {
		err = fmt.Errorf("serial-Step numerical reference: %w", e)
		return
	}
	report.ReferenceSuffixDigest = reference.SuffixDigest
	report.ReferenceNextDigest = reference.NextDigest
	for i := 0; i < w.Repetitions; i++ {
		sample, suffix, next, e := suffixPrefillPassWithClock(factory, w, false, vocab, now)
		if e == nil {
			sample.SuffixMaxAbs, e = compareSuffixLogits(suffix, refSuffix, w)
		}
		if e == nil {
			sample.NextMaxAbs, e = compareSuffixLogits(next, refNext, w)
		}
		if e != nil {
			sample.Error = e.Error()
		}
		report.Samples = append(report.Samples, sample)
		if e != nil {
			err = fmt.Errorf("sample %d: %w", i, e)
			return
		}
	}
	return
}

func runSuffixPrefillCLI(f *benchFlags, m *model.Model, name string, w *suffixPrefillWorkload, newSession func() *model.Session) error {
	// Identity hashing is deliberately outside sample timings. This reads the
	// supplied input file only. Split GGUF loaders may resolve additional shards;
	// this digest does not authenticate that full set and may warm the file cache.
	artifact, err := fileIdentity(*f.gguf)
	if err != nil {
		return err
	}
	source, binary, err := suffixBuildIdentity()
	if err != nil {
		return err
	}
	report, runErr := measureSuffixPrefill(func() suffixPrefillSession { return newSession() }, *w, m.Cfg.VocabSize)
	report.Model = name
	report.Config = m.Cfg
	report.Artifact = artifact
	report.ArtifactCoverage = "input-file-only; partial checkpoint coverage"
	report.ArtifactRequirement = "full loaded shard-set manifest and digest verification unavailable on this loader route; required before whole-checkpoint identity claims"
	report.HashingEffect = "input file and executable hashed after loading, outside sample timings; reads may warm page cache before numerical reference and measured samples; cold-cache performance is not established"
	report.Source = source
	report.Binary = binary
	report.Backend = *f.backendName
	report.Quant = *f.quant
	report.Q4K = *f.q4k
	report.Metal = *f.metal
	report.GOOS = runtime.GOOS
	report.GOARCH = runtime.GOARCH
	report.GOMAXPROCS = runtime.GOMAXPROCS(0)
	report.SuffixControl = os.Getenv("FAK_V41_SUFFIX_UBATCH")
	report.Controls = map[string]string{}
	for _, key := range []string{"FAK_WORKERS", "FAK_BUDGET", "FAK_PAGED_KV", "FAK_V41_SUFFIX_UBATCH", "FAK_VULKAN_TIMESTAMP_PROFILE", "FAK_VULKAN_TIMESTAMP_PROFILE_EVERY", "FAK_VULKAN_DISPATCH_PROFILE"} {
		report.Controls[key] = os.Getenv(key)
	}
	b, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return errors.Join(runErr, err)
	}
	out, err := os.OpenFile(*suffixPrefillOutput, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return errors.Join(runErr, err)
	}
	_, writeErr := out.Write(append(b, '\n'))
	closeErr := out.Close()
	return errors.Join(runErr, writeErr, closeErr)
}
