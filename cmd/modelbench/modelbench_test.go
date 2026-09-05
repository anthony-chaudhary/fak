// Unit tests for the pure helpers in modelbench's main package: the CSV
// size parser, the prompt-length clamp, and the greedy-token argmax. All three
// are deterministic and resource-free (no model file, GPU, or network), so the
// expected values below are computed by hand from the functions' actual logic.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/mathx"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

func testBool(v bool) *bool { return &v }

func testString(v string) *string                                                      { return &v }
func testInt(v int) *int                                                               { return &v }
func testFloat64(v float64) *float64                                                   { return &v }
func testDuration(v time.Duration) *time.Duration                                      { return &v }
func testDecodeHandoff(v model.Qwen35DecodeHandoffMode) *model.Qwen35DecodeHandoffMode { return &v }

type testCloser func() error

func (c testCloser) Close() error { return c() }

func testCompleteBenchFlags() *benchFlags {
	return &benchFlags{
		dir: testString("fixture-dir"), hf: testString(""), gguf: testString(""),
		lean: testBool(false), q4k: testBool(false), streamQ4K: testBool(false),
		name: testString(""), out: testString(""), prefillSizesCSV: testString("16"),
		prefillReps: testInt(1), decodeReps: testInt(1), decodeSteps: testInt(1), decodePrompt: testInt(1),
		quant: testBool(false), metal: testBool(false), verify: testBool(false),
		backendName: testString("legacy"), q4kGateUpSlab: testBool(false), vulkanQ4KProfile: testBool(false), vulkanStageQ4K: testBool(false), requireNonReference: testBool(false),
		workloadPath: testString(""), workloadPrefillCap: testInt(0), loadOnly: testBool(false),
		loadProfile: testBool(false), loadProfileTrace: testBool(false), loadProfileTraceEvery: testInt(25),
		phaseProfile: testBool(false), budget: testFloat64(0), preflight: testBool(false), smoke: testBool(false),
		smokeDeadline: testDuration(90 * time.Second), fitCheck: testBool(true), loadProgress: testBool(true),
		checkpoint: testString(""), resume: testString(""), nativeProfileOut: testString(""), nativeProfileReadback: testString(""),
		nativeProfileCompare: testString(""), nativeDecodeHandoff: testDecodeHandoff(model.Qwen35DecodeHandoffAuto),
		qwenSwapOut: testString(""), qwenSwapReadback: testString(""),
	}
}

func testBenchFlags(q4k, quant, metal bool) *benchFlags {
	return &benchFlags{q4k: testBool(q4k), quant: testBool(quant), metal: testBool(metal), q4kGateUpSlab: testBool(false)}
}

func TestStreamQ4KValidation(t *testing.T) {
	valid := testCompleteBenchFlags()
	*valid.gguf = "fixture.gguf"
	*valid.q4k = true
	*valid.streamQ4K = true
	*valid.loadProfile = true
	*valid.loadProfileTrace = true
	if err := validateFlagCombinations(valid); err != nil {
		t.Fatalf("valid streamed Q4_K load rejected: %v", err)
	}
	*valid.nativeProfileOut = "profile.json"
	*valid.metal = true
	*valid.decodePrompt = 32
	*valid.decodeSteps = 64
	if err := validateFlagCombinations(valid); err != nil {
		t.Fatalf("valid streamed native-performance profile rejected: %v", err)
	}

	tests := []struct {
		name string
		edit func(*benchFlags)
		want string
	}{
		{name: "missing q4k", edit: func(f *benchFlags) { *f.q4k = false }, want: "requires exact -gguf and -q4k"},
		{name: "missing gguf", edit: func(f *benchFlags) { *f.gguf = "" }, want: "requires exact -gguf and -q4k"},
		{name: "lean remains incompatible", edit: func(f *benchFlags) { *f.lean = true }, want: "omit -lean"},
		{name: "native profile still requires metal", edit: func(f *benchFlags) {
			*f.nativeProfileOut = "profile.json"
			*f.decodePrompt = 32
			*f.decodeSteps = 64
		}, want: "requires -gguf, -q4k, -metal"},
		{name: "native profile still requires P32", edit: func(f *benchFlags) {
			*f.nativeProfileOut = "profile.json"
			*f.metal = true
			*f.decodePrompt = 31
			*f.decodeSteps = 64
		}, want: "-decode-prompt=32"},
		{name: "native profile still requires T64", edit: func(f *benchFlags) {
			*f.nativeProfileOut = "profile.json"
			*f.metal = true
			*f.decodePrompt = 32
			*f.decodeSteps = 63
		}, want: "-decode-steps=64"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := testCompleteBenchFlags()
			*f.gguf = "fixture.gguf"
			*f.q4k = true
			*f.streamQ4K = true
			tt.edit(f)
			err := validateFlagCombinations(f)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("validation error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestQ4KLoaderSelectorPreservesResidentDefault(t *testing.T) {
	originalResident, originalStreamed := loadResidentQ4K, loadStreamedDenseQ4K
	defer func() {
		loadResidentQ4K, loadStreamedDenseQ4K = originalResident, originalStreamed
	}()
	var residentCalls, streamedCalls int
	var gotProfiler *ggufload.LoadProfiler
	loadResidentQ4K = func(_ context.Context, path string, p *ggufload.LoadProfiler) (*model.Model, error) {
		residentCalls++
		gotProfiler = p
		return &model.Model{}, nil
	}
	loadStreamedDenseQ4K = func(_ context.Context, path string, p *ggufload.LoadProfiler) (*model.Model, error) {
		streamedCalls++
		gotProfiler = p
		return &model.Model{}, nil
	}

	f := testCompleteBenchFlags()
	*f.gguf = "/tmp/fixture.gguf"
	*f.q4k = true
	profiler := ggufload.NewLoadProfiler()
	_, name, err := loadModel(f, profiler)
	if err != nil {
		t.Fatal(err)
	}
	if residentCalls != 1 || streamedCalls != 0 || gotProfiler != profiler {
		t.Fatalf("resident selection calls resident=%d streamed=%d profiler=%p want %p", residentCalls, streamedCalls, gotProfiler, profiler)
	}
	if name != "fixture.gguf [gguf-q4k]" {
		t.Fatalf("resident label = %q", name)
	}

	*f.streamQ4K = true
	_, name, err = loadModel(f, profiler)
	if err != nil {
		t.Fatal(err)
	}
	if residentCalls != 1 || streamedCalls != 1 || gotProfiler != profiler {
		t.Fatalf("streamed selection calls resident=%d streamed=%d profiler=%p want %p", residentCalls, streamedCalls, gotProfiler, profiler)
	}
	if name != "fixture.gguf [gguf-q4k-streamed-dense]" {
		t.Fatalf("streamed label = %q", name)
	}
}

func TestStreamQ4KProfilerIdentity(t *testing.T) {
	f := testCompleteBenchFlags()
	*f.gguf = "fixture.gguf"
	*f.q4k = true
	*f.streamQ4K = true
	*f.loadProfile = true
	lp := newGGUFLoadProfiler(f)
	if lp == nil || lp.Progress == nil {
		t.Fatal("streamed Q4_K must be progress/profile capable")
	}
	mode, source := ggufLoadProfileIdentity(f)
	profile := lp.Snapshot(mode, source, 123)
	if profile.Mode != "gguf-streamed-dense-q4k" || profile.Source != "fixture.gguf (streamed dense Q4_K)" {
		t.Fatalf("profile identity = mode %q source %q", profile.Mode, profile.Source)
	}

	*f.streamQ4K = false
	if got := newGGUFLoadProfiler(f); got != nil {
		t.Fatal("resident Q4_K default unexpectedly changed its historical profiler behavior")
	}
}

func TestLoadWorkerControlReadback(t *testing.T) {
	tests := []struct {
		name          string
		literal       string
		explicit      bool
		wantSource    string
		wantEffective int
	}{
		{name: "unset", wantSource: "unset"},
		{name: "valid explicit preserves literal", literal: " 1 ", explicit: true, wantSource: "explicit", wantEffective: 1},
		{name: "invalid explicit stays typed", literal: "many", explicit: true, wantSource: "explicit"},
		{name: "zero explicit is invalid", literal: "0", explicit: true, wantSource: "explicit"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			control := readLoadWorkerControl(func(string) (string, bool) { return tt.literal, tt.explicit }, 12)
			if control.FAKGGUFLoadWorkers != tt.literal || control.Source != tt.wantSource || control.GOMAXPROCS != 12 {
				t.Fatalf("control = %+v", control)
			}
			if tt.wantEffective == 0 {
				if control.EffectiveCount != nil {
					t.Fatalf("invalid/unset control derived effective=%d", *control.EffectiveCount)
				}
				encoded, err := json.Marshal(control)
				if err != nil {
					t.Fatal(err)
				}
				if strings.Contains(string(encoded), `"effective_count"`) {
					t.Fatalf("invalid/unset control emitted a numeric effective count: %s", encoded)
				}
			} else if control.EffectiveCount == nil || *control.EffectiveCount != tt.wantEffective {
				t.Fatalf("effective = %v, want %d", control.EffectiveCount, tt.wantEffective)
			}
		})
	}
}

func TestStreamQ4KSourceFingerprintAndReportIdentity(t *testing.T) {
	t.Setenv("FAK_GGUF_LOAD_WORKERS", " 1 ")
	f := testCompleteBenchFlags()
	*f.gguf = "fixture.gguf"
	*f.q4k = true
	*f.streamQ4K = true

	wantSource := "fixture.gguf (streamed dense Q4_K)"
	if got := loadSource("", *f.gguf, "", false, true, true); got != wantSource {
		t.Fatalf("load source = %q, want %q", got, wantSource)
	}
	fingerprint := modelbenchFingerprint(f, "fixture")
	if fingerprint["stream_q4k"] != true || fingerprint["source"] != wantSource {
		t.Fatalf("fingerprint stream identity = %#v", fingerprint)
	}
	fpControl, ok := fingerprint["load_worker_control"].(loadWorkerControl)
	if !ok || fpControl.EffectiveCount == nil || *fpControl.EffectiveCount != 1 || fpControl.FAKGGUFLoadWorkers != " 1 " {
		t.Fatalf("fingerprint worker control = %#v", fingerprint["load_worker_control"])
	}
	report := loadReportIdentity(f)
	if report["stream_q4k"] != true || report["source"] != wantSource {
		t.Fatalf("report stream identity = %#v", report)
	}
	reportControl, ok := report["load_worker_control"].(loadWorkerControl)
	if !ok || reportControl.EffectiveCount == nil || *reportControl.EffectiveCount != 1 {
		t.Fatalf("report worker control = %#v", report["load_worker_control"])
	}
}

func TestTransferredWeightCloserRunsExactlyOnce(t *testing.T) {
	t.Run("normal and repeated cleanup", func(t *testing.T) {
		f := testCompleteBenchFlags()
		*f.streamQ4K = true
		calls := 0
		m := &model.Model{}
		m.SetWeightCloser(testCloser(func() error { calls++; return nil }))
		if !bindLoadedModelWeights(f, m) {
			t.Fatal("streamed model ownership was not transferred to the command guard")
		}
		if err := f.closeTransferredWeights(); err != nil {
			t.Fatal(err)
		}
		if err := f.closeTransferredWeights(); err != nil {
			t.Fatal(err)
		}
		if calls != 1 {
			t.Fatalf("closer calls = %d, want 1", calls)
		}
	})

	t.Run("terminal status closes before exit", func(t *testing.T) {
		f := testCompleteBenchFlags()
		var events []string
		f.bindWeightCloser(func() error { events = append(events, "close"); return nil })
		f.processExit = func(code int) { events = append(events, fmt.Sprintf("exit:%d", code)) }
		f.exit(2)
		_ = f.closeTransferredWeights()
		if !reflect.DeepEqual(events, []string{"close", "exit:2"}) {
			t.Fatalf("terminal lifecycle = %v", events)
		}
	})

	t.Run("close error forces failure status", func(t *testing.T) {
		f := testCompleteBenchFlags()
		f.bindWeightCloser(func() error { return fmt.Errorf("close failed") })
		gotCode := 0
		f.processExit = func(code int) { gotCode = code }
		f.exit(2)
		if gotCode != 1 {
			t.Fatalf("exit code = %d, want 1 after close failure", gotCode)
		}
	})
}

func TestStreamedNativeProfileRetainsAndClosesWeightsExactlyOnce(t *testing.T) {
	tests := []struct {
		name    string
		runErr  error
		wantErr string
	}{
		{name: "success"},
		{name: "profile error", runErr: fmt.Errorf("capture failed"), wantErr: "capture failed"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := testCompleteBenchFlags()
			*f.streamQ4K = true
			f.processExit = func(int) {}
			calls := 0
			m := &model.Model{}
			m.SetWeightCloser(testCloser(func() error {
				calls++
				return nil
			}))
			if !bindLoadedModelWeights(f, m) {
				t.Fatal("streamed native profile did not bind model weight ownership")
			}
			err := runWithTransferredWeightLifetime(f, func() error {
				if calls != 0 {
					t.Fatal("checkpoint closed before native profile completed")
				}
				return tt.runErr
			})
			if tt.wantErr == "" && err != nil {
				t.Fatalf("native profile returned error: %v", err)
			}
			if tt.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tt.wantErr)) {
				t.Fatalf("native profile error = %v, want substring %q", err, tt.wantErr)
			}
			f.exit(1)
			_ = f.closeTransferredWeights()
			if calls != 1 {
				t.Fatalf("weight closer calls = %d, want exactly 1", calls)
			}
		})
	}
}

func TestNativeProfileFailureFinishesExecutionBeforeTransferredWeights(t *testing.T) {
	f := testCompleteBenchFlags()
	*f.streamQ4K = true
	var events []string
	f.bindWeightCloser(func() error {
		events = append(events, "weights")
		return nil
	})

	err := runWithTransferredWeightLifetime(f, func() (err error) {
		finishProfile := onceFinishNativeProfile(func() { events = append(events, "session") })
		defer finishProfile()
		return fmt.Errorf("injected profile failure")
	})
	if err == nil || !strings.Contains(err.Error(), "injected profile failure") {
		t.Fatalf("profile error = %v", err)
	}
	if !reflect.DeepEqual(events, []string{"session", "weights"}) {
		t.Fatalf("failure cleanup order = %v, want [session weights]", events)
	}
}

func TestNativeProfileFinishDoesNotRepeat(t *testing.T) {
	calls := 0
	finishProfile := onceFinishNativeProfile(func() { calls++ })
	finishProfile()
	finishProfile()
	if calls != 1 {
		t.Fatalf("session closer calls = %d, want exactly 1", calls)
	}
}

func TestResidentNativeProfileDoesNotBindWeightCloser(t *testing.T) {
	f := testCompleteBenchFlags()
	*f.gguf = "fixture.gguf"
	*f.q4k = true
	*f.metal = true
	*f.decodePrompt = 32
	*f.decodeSteps = 64
	*f.nativeProfileOut = "profile.json"
	if err := validateFlagCombinations(f); err != nil {
		t.Fatalf("historical non-stream native profile rejected: %v", err)
	}
	m := &model.Model{}
	calls := 0
	m.SetWeightCloser(testCloser(func() error { calls++; return nil }))
	if bindLoadedModelWeights(f, m) {
		t.Fatal("resident native profile unexpectedly transferred streamed weight ownership")
	}
	if err := runWithTransferredWeightLifetime(f, func() error { return nil }); err != nil {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Fatalf("resident native profile closer calls = %d, want 0", calls)
	}
}

func TestParsePositiveInts(t *testing.T) {
	tests := []struct {
		name    string
		in      string
		want    []int
		wantErr bool
	}{
		{name: "default sizes", in: "16,64,256", want: []int{16, 64, 256}},
		{name: "single", in: "8", want: []int{8}},
		{name: "trims whitespace", in: " 1 , 2 ", want: []int{1, 2}},
		{name: "skips empty fields", in: "3,,4", want: []int{3, 4}},
		{name: "trailing comma", in: "5,", want: []int{5}},
		{name: "empty string", in: "", wantErr: true},
		{name: "only separators", in: " , ", wantErr: true},
		{name: "zero rejected", in: "0", wantErr: true},
		{name: "negative rejected", in: "-5", wantErr: true},
		{name: "non-numeric rejected", in: "abc", wantErr: true},
		{name: "mixed valid then invalid", in: "4,nope", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parsePositiveInts(tt.in)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("parsePositiveInts(%q) = %v, want error", tt.in, got)
				}
				if got != nil {
					t.Fatalf("parsePositiveInts(%q) returned %v on error, want nil slice", tt.in, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("parsePositiveInts(%q) unexpected error: %v", tt.in, err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("parsePositiveInts(%q) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

func TestCapPositive(t *testing.T) {
	tests := []struct {
		name string
		n    int
		cap  int
		want int
	}{
		{name: "no cap passes through", n: 10, cap: 0, want: 10},
		{name: "over cap is clamped", n: 100, cap: 50, want: 50},
		{name: "exactly at cap unchanged", n: 50, cap: 50, want: 50},
		{name: "under cap unchanged", n: 3, cap: 5, want: 3},
		{name: "zero with no cap floored to one", n: 0, cap: 0, want: 1},
		{name: "negative floored to one", n: -3, cap: 0, want: 1},
		{name: "zero under positive cap floored to one", n: 0, cap: 5, want: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := capPositive(tt.n, tt.cap); got != tt.want {
				t.Fatalf("capPositive(%d, %d) = %d, want %d", tt.n, tt.cap, got, tt.want)
			}
		})
	}
}

func TestArgmax(t *testing.T) {
	tests := []struct {
		name string
		in   []float32
		want int
	}{
		{name: "middle max", in: []float32{0.1, 0.9, 0.3}, want: 1},
		{name: "first on ties", in: []float32{5, 5, 5}, want: 0},
		{name: "all negative", in: []float32{-3, -1, -2}, want: 1},
		{name: "single element", in: []float32{42}, want: 0},
		{name: "last is max", in: []float32{1, 2, 3, 4}, want: 3},
		{name: "empty slice", in: []float32{}, want: 0},
		{name: "below default floor", in: []float32{-math.MaxFloat32 / 2}, want: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := mathx.ArgmaxF32(tt.in); got != tt.want {
				t.Fatalf("ArgmaxF32(%v) = %d, want %d", tt.in, got, tt.want)
			}
		})
	}
}

func TestSmokeOutcome(t *testing.T) {
	const dl = 90 * time.Second
	tests := []struct {
		name     string
		done     bool
		elapsed  time.Duration
		deadline time.Duration
		want     string
	}{
		{name: "finished under deadline", done: true, elapsed: 5 * time.Second, deadline: dl, want: smokeStatusLoaded},
		{name: "not finished (deadline fired)", done: false, elapsed: dl, deadline: dl, want: smokeStatusTimeout},
		{name: "finished but over deadline", done: true, elapsed: 2 * dl, deadline: dl, want: smokeStatusTimeout},
		{name: "finished, no deadline set", done: true, elapsed: time.Hour, deadline: 0, want: smokeStatusLoaded},
		{name: "exactly at deadline counts as loaded", done: true, elapsed: dl, deadline: dl, want: smokeStatusLoaded},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := smokeOutcome(tt.done, tt.elapsed, tt.deadline); got != tt.want {
				t.Fatalf("smokeOutcome(%v, %v, %v) = %s, want %s", tt.done, tt.elapsed, tt.deadline, got, tt.want)
			}
		})
	}
}

// TestQ8UploadGate locks the modelbench -quant admission gate (#472): the -quant
// (Q8_0) forward may run on a compute backend ONLY when that backend advertises
// Caps().UploadDtype, because the wired Q8 HAL path keys off it. An f32-only backend
// must refuse -quant instead of silently running the f32 path under a Q8 flag, and the
// plain f32 path (-quant=false) must stay unchanged regardless of the backend's caps.
// The Q8-capable case is exercised on hardware by internal/model's
// TestHALVulkanQ8ForwardMatchesComputeQ8; this is its host-free decision-logic twin.
func TestQ8UploadGate(t *testing.T) {
	tests := []struct {
		name       string
		quant      bool
		caps       compute.Caps
		wantRefuse bool
	}{
		{name: "quant on f32-only backend refuses", quant: true, caps: compute.Caps{}, wantRefuse: true},
		{name: "quant on Q8-upload backend allowed", quant: true, caps: compute.Caps{UploadDtype: true}, wantRefuse: false},
		{name: "f32 path on f32-only backend unchanged", quant: false, caps: compute.Caps{}, wantRefuse: false},
		{name: "f32 path on Q8-upload backend unchanged", quant: false, caps: compute.Caps{UploadDtype: true}, wantRefuse: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := q8UploadUnsupported(tt.quant, tt.caps); got != tt.wantRefuse {
				t.Fatalf("q8UploadUnsupported(%v, %+v) = %v, want %v", tt.quant, tt.caps, got, tt.wantRefuse)
			}
		})
	}

	// Ground the gate in the real cpu-ref backend the issue names as the f32-only
	// example: it must lack UploadDtype, so -quant against it is refused while the
	// f32 path stays open. This binds the predicate to a production backend's caps
	// rather than a hand-built struct, so a regression in either side is caught.
	be, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("cpu-ref backend must be registered")
	}
	if be.Caps().UploadDtype {
		t.Fatalf("cpu-ref unexpectedly advertises UploadDtype; the Q8-upload gate would no longer refuse it")
	}
	if !q8UploadUnsupported(true, be.Caps()) {
		t.Fatalf("cpu-ref (f32-only) must refuse -quant")
	}
	if q8UploadUnsupported(false, be.Caps()) {
		t.Fatalf("cpu-ref must keep the f32 path open when -quant is false")
	}
}

func TestQ4KMetalSessionFlagsUseMetalQ4K(t *testing.T) {
	f := testBenchFlags(true, false, true)
	s := &model.Session{}
	applyLegacySessionFlags(s, f)

	if !s.Q4K {
		t.Fatalf("Q4K flag was not applied")
	}
	if !s.MetalQ4K {
		t.Fatalf("-q4k -metal must route through Session.MetalQ4K")
	}
	if s.Metal {
		t.Fatalf("-q4k -metal must not route through the Q8/f16 Session.Metal lane")
	}
	if s.Quant {
		t.Fatalf("-q4k -metal must not force the separate Q8_0 session flag")
	}
}

func TestQ4KSlabExplicitFlagReachesModelbenchSession(t *testing.T) {
	t.Setenv("FAK_Q4K_GATEUP_SLAB", "0")
	f := testBenchFlags(true, false, false)
	*f.q4kGateUpSlab = true
	s := &model.Session{}
	applyLegacySessionFlags(s, f)
	if !s.Q4KGateUpOutputSlab {
		t.Fatal("explicit modelbench Q4_K slab setting did not reach session")
	}
}

func TestQ8MetalSessionFlagsKeepMetalLane(t *testing.T) {
	f := testBenchFlags(false, true, true)
	s := &model.Session{}
	applyLegacySessionFlags(s, f)

	if !s.Quant || !s.Metal {
		t.Fatalf("Q8 -metal should keep Quant+Metal set, got Quant=%v Metal=%v", s.Quant, s.Metal)
	}
	if s.Q4K || s.MetalQ4K {
		t.Fatalf("Q8 -metal should not set Q4K/MetalQ4K, got Q4K=%v MetalQ4K=%v", s.Q4K, s.MetalQ4K)
	}
}

func TestNUMAReplicasFlagAndWiring(t *testing.T) {
	cfg := model.Config{
		HiddenSize:       256,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		HeadDim:          64,
		IntermediateSize: 512,
		VocabSize:        256,
	}
	m := model.NewSynthetic(cfg)
	defer m.FreeNUMAReplicas()

	f := &benchFlags{
		numaReplicas: testString("2"),
	}

	lbl := m.ApplyNUMAWeightReplicas(*f.numaReplicas)
	if !m.NUMAReplicasEnabled() {
		t.Fatalf("expected NUMA replicas enabled with flag=2, got label %s", lbl)
	}
	if !strings.Contains(m.NUMAReplicasLabel(), "nodes=2") {
		t.Fatalf("label does not indicate 2 nodes: %s", m.NUMAReplicasLabel())
	}

	// Disable via flag "off"
	*f.numaReplicas = "off"
	lbl = m.ApplyNUMAWeightReplicas(*f.numaReplicas)
	if m.NUMAReplicasEnabled() {
		t.Fatalf("expected NUMA replicas disabled with flag=off, got label %s", lbl)
	}

	// Fallback via FAK_NUMA_REPLICAS environment variable
	t.Setenv("FAK_NUMA_REPLICAS", "2")
	*f.numaReplicas = ""
	lbl = m.ApplyNUMAWeightReplicas(*f.numaReplicas)
	if !m.NUMAReplicasEnabled() {
		t.Fatalf("expected NUMA replicas enabled via env FAK_NUMA_REPLICAS=2, got label %s", lbl)
	}

	t.Setenv("FAK_NUMA_REPLICAS", "off")
	lbl = m.ApplyNUMAWeightReplicas("")
	if m.NUMAReplicasEnabled() {
		t.Fatalf("expected NUMA replicas disabled via env FAK_NUMA_REPLICAS=off, got label %s", lbl)
	}
}

func TestResolveMetalDoesNotForceQuantForQ4K(t *testing.T) {
	f := testBenchFlags(true, false, true)
	resolveMetal(f)

	if *f.quant {
		t.Fatalf("-q4k -metal must not force Quantize(); LoadModelQ4K already owns the resident mixed store")
	}
}

func TestDescribeEngineLabelsQ4KMetal(t *testing.T) {
	f := testBenchFlags(true, false, true)
	engine, precision, _ := describeEngine(f, nil, nil)

	if !strings.Contains(engine, "Metal Q4_K") {
		t.Fatalf("engine %q does not identify the Q4_K Metal scorer", engine)
	}
	if !strings.Contains(precision, "MetalQ4K") {
		t.Fatalf("precision %q does not identify MetalQ4K", precision)
	}
}

func TestAllFinite(t *testing.T) {
	tests := []struct {
		name string
		in   []float32
		want bool
	}{
		{name: "all finite", in: []float32{-1, 0, 3.5, 1e9}, want: true},
		{name: "empty is not a valid forward result", in: []float32{}, want: false},
		{name: "contains NaN", in: []float32{1, float32(math.NaN()), 3}, want: false},
		{name: "contains +Inf", in: []float32{1, float32(math.Inf(1)), 3}, want: false},
		{name: "contains -Inf", in: []float32{float32(math.Inf(-1)), 2}, want: false},
		{name: "single finite", in: []float32{42}, want: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := allFinite(tt.in); got != tt.want {
				t.Fatalf("allFinite(%v) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

func testNativeProfileControls() map[string]string {
	controls := make(map[string]string, len(nativeProfileRequiredEnvironment)+len(nativeProfileDeniedEnvironment)+8)
	for key, value := range nativeProfileRequiredEnvironment {
		controls[key] = value
	}
	for _, key := range nativeProfileDeniedEnvironment {
		controls[key] = nativeProfileUnset
	}
	controls[nativeControlGGUFMMap] = "1"
	controls[nativeProfileSequenceSelector] = nativeProfileSelectorOff
	controls[nativeProfileDecodeHandoffControl] = model.Qwen35DecodeHandoffAuto.String()
	controls[nativeControlFlagBudget] = "0"
	controls[nativeControlLogicalCPUs] = strconv.Itoa(runtime.NumCPU())
	controls[nativeControlGOMAXPROCS] = strconv.Itoa(runtime.GOMAXPROCS(0))
	controls[nativeControlWorkers] = strconv.Itoa(model.NumWorkers())
	controls[nativeControlQ8Workers] = strconv.Itoa(model.Q8DecodeWorkers())
	controls[nativeControlWorkerBudget] = "default(GOMAXPROCS)"
	return controls
}

func TestNativeProfileControlsRefuseBeforeRun(t *testing.T) {
	required := map[string]string{}
	declarations := make([]string, 0, len(nativeProfileRequiredEnvironment))
	for key, value := range nativeProfileRequiredEnvironment {
		required[key] = value
		declarations = append(declarations, key+"="+value)
	}
	required[nativeControlGGUFMMap] = "0"
	declarations = append(declarations, nativeControlGGUFMMap+"=0")
	lookup := func(key string) (string, bool) { value, ok := required[key]; return value, ok }
	if _, err := nativeProfileControlEnvironment(lookup, declarations, 0, model.Qwen35DecodeHandoffAuto); err != nil {
		t.Fatalf("documented control envelope rejected: %v", err)
	}
	for _, selector := range []string{nativeProfileSelectorOff, nativeProfileSelectorOn} {
		t.Run("typed selector "+selector, func(t *testing.T) {
			env := mapsStringClone(required)
			env[nativeProfileSequenceSelector] = selector
			decls := append(append([]string(nil), declarations...), nativeProfileSequenceSelector+"="+selector)
			lookup := func(key string) (string, bool) { value, ok := env[key]; return value, ok }
			controls, err := nativeProfileControlEnvironment(lookup, decls, 0, model.Qwen35DecodeHandoffAuto)
			if err != nil || controls[nativeProfileSequenceSelector] != selector {
				t.Fatalf("typed selector rejected or not captured: controls=%v err=%v", controls, err)
			}
		})
	}
	for _, mode := range []model.Qwen35DecodeHandoffMode{model.Qwen35DecodeHandoffControl, model.Qwen35DecodeHandoffMixer} {
		t.Run("graded handoff "+mode.String(), func(t *testing.T) {
			env := mapsStringClone(required)
			env[nativeProfileSequenceSelector] = nativeProfileSelectorOn
			decls := append(append([]string(nil), declarations...), nativeProfileSequenceSelector+"="+nativeProfileSelectorOn)
			lookup := func(key string) (string, bool) { value, ok := env[key]; return value, ok }
			controls, err := nativeProfileControlEnvironment(lookup, decls, 0, mode)
			if err != nil || controls[nativeProfileDecodeHandoffControl] != mode.String() {
				t.Fatalf("graded handoff rejected or missing: controls=%v err=%v", controls, err)
			}
		})
	}
	if _, err := nativeProfileControlEnvironment(lookup, declarations, 0, model.Qwen35DecodeHandoffControl); err == nil {
		t.Fatal("CONTROL accepted without sequence ON")
	}

	tests := []struct {
		name        string
		key         string
		value       string
		budget      float64
		declaration string
	}{
		{name: "forced q8 upload", key: "FAK_METAL_Q8_UPLOAD", value: "1"},
		{name: "free q4k cpu", key: "FAK_Q4K_FREE_CPU", value: "1"},
		{name: "q4k mm", key: "FAK_Q4K_MM", value: "1"},
		{name: "q8 gemm group", key: "FAK_Q8_GEMM_GROUP", value: "1"},
		{name: "workers", key: "FAK_WORKERS", value: "4"},
		{name: "budget env", key: "FAK_BUDGET", value: "0.5"},
		{name: "mmap untyped", key: nativeControlGGUFMMap, value: "true"},
		{name: "sequence selector untyped", key: nativeProfileSequenceSelector, value: "true"},
		{name: "budget flag", budget: 0.5},
		{name: "unknown fak control", declaration: "FAK_FUTURE_Q4K_SWITCH=1"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			env := make(map[string]string, len(required)+1)
			decls := append([]string(nil), declarations...)
			for key, value := range required {
				env[key] = value
			}
			if test.key != "" {
				env[test.key] = test.value
				decls = append(decls, test.key+"="+test.value)
			}
			if test.declaration != "" {
				decls = append(decls, test.declaration)
			}
			lookup := func(key string) (string, bool) { value, ok := env[key]; return value, ok }
			if _, err := nativeProfileControlEnvironment(lookup, decls, test.budget, model.Qwen35DecodeHandoffAuto); err == nil {
				t.Fatal("behavior-changing control was accepted")
			}
		})
	}
}

func testNativeProfileReceipt(t *testing.T) ([]byte, nativeperf.ProfileBundle, nativeProfileReceipt) {
	t.Helper()
	t.Setenv("FAK_GGUF_MMAP", "1")
	executionSession := metalgemm.NewExecutionSession()
	executionSession.Record(metalgemm.ExecutionSnapshot{Events: []metalgemm.ExecutionEvent{{
		Operation: metalgemm.ExecutionQ4KGEMM, CommandBufferID: 1, Committed: true,
		CompletedWait: true, HostReadback: true, Encoders: 2, GPUMilliseconds: 3,
		WaitMilliseconds: 4, TimingAvailable: true,
	}}}, nil)
	execution, err := executionSession.Receipt()
	if err != nil {
		t.Fatal(err)
	}
	phases := make([]nativeperf.ProfilePhase, 0, 6)
	for i, name := range []string{"load-setup", "prefill", "first-token", "steady-decode", "verification", "teardown"} {
		phases = append(phases, nativeperf.ProfilePhase{Name: name, StartMilliseconds: float64(i), DurationMilliseconds: 1})
	}
	profile := nativeperf.ProfileBundle{
		Schema:     nativeperf.ProfileSchema,
		EnvelopeID: "qwen38-27b-q4km-m3pro-p32-t64",
		Execution:  nativeperf.ExecutionIdentity{Engine: "fak-native", ForwardPath: "metal/qwen35-hybrid-session-v1"},
		Phases:     phases,
		Metal: &nativeperf.MetalCounters{
			CommandBuffers: 1, Encoders: 2, DispatchMilliseconds: 3, WaitMilliseconds: 4,
			ResidentBytes: 1, WorkingSetBytes: 1,
		},
		AttributionUnavailable: &nativeperf.AttributionUnavailable{Reason: nativeperf.AttributionUnavailableCapture, Detail: "test capture has no per-lever attribution"},
	}
	profileBytes, err := json.MarshalIndent(profile, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	profileBytes = append(profileBytes, '\n')
	profileSHA := sha256.Sum256(profileBytes)
	config := map[string]any{"model_type": "qwen3_5_text", "hidden_size": float64(5120)}
	configSHA, err := sha256JSON(config)
	if err != nil {
		t.Fatal(err)
	}
	fallbackEvents := []model.MetalFallbackEvent{}
	fallbackSHA, err := sha256JSON(fallbackEvents)
	if err != nil {
		t.Fatal(err)
	}
	q4kResidency := (&model.Model{}).Q4KResidencyReceipt()
	handoffReceipt := model.Qwen35DecodeHandoffReceipt{Mode: model.Qwen35DecodeHandoffAuto}
	receipt := nativeProfileReceipt{
		Schema:        nativeProfileReceiptSchema,
		ProfileSHA256: fmt.Sprintf("%x", profileSHA),
		EnvelopeID:    profile.EnvelopeID,
		Artifact: nativeArtifactIdentity{
			nativeFileIdentity: nativeFileIdentity{Bytes: nativeProfileArtifactBytes, SHA256: "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"},
			Model:              "unsloth/Qwen3.8-27B-GGUF", ModelRevision: "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe",
		},
		ModelConfig: config, ModelConfigSHA256: configSHA,
		Host:                nativeHostIdentity{GOOS: "darwin", GOARCH: "arm64", CPU: "Apple M3 Pro", MetalDevice: "Apple M3 Pro", GPUCores: 18, MemoryBytes: 36 << 30, MetalWorkingSetBytes: 1},
		Source:              nativeSourceIdentity{Revision: strings.Repeat("a", 40)},
		Binary:              nativeFileIdentity{Bytes: 123, SHA256: strings.Repeat("b", 64)},
		Controls:            testNativeProfileControls(),
		Execution:           execution,
		Fallbacks:           model.MetalFallbackReceipt{Schema: "fak-metal-fallback-receipt/v1", Events: fallbackEvents, EventsSHA256: fallbackSHA},
		Q4KResidency:        &q4kResidency,
		Qwen35DecodeHandoff: &handoffReceipt,
	}
	receipt.BindingSHA256, err = nativeReceiptBinding(receipt)
	if err != nil {
		t.Fatal(err)
	}
	return profileBytes, profile, receipt
}

func TestAppendNativeProfilePhaseUsesPreviousEncodedEnd(t *testing.T) {
	// Independent nanosecond-to-millisecond conversions can round a cumulative
	// phase start below the previous encoded end at real model-load durations.
	load := 83753567806 * time.Nanosecond
	prefill := 8307754683 * time.Nanosecond
	oldEnd := float64(load.Nanoseconds())/1e6 + float64(prefill.Nanoseconds())/1e6
	oldNextStart := float64((load + prefill).Nanoseconds()) / 1e6
	if oldNextStart >= oldEnd {
		t.Fatal("test durations no longer reproduce cumulative conversion overlap")
	}

	phases := appendNativeProfilePhase(nil, "load-setup", load)
	phases = appendNativeProfilePhase(phases, "prefill", prefill)
	phases = appendNativeProfilePhase(phases, "first-token", time.Nanosecond)
	previousEnd := phases[1].StartMilliseconds + phases[1].DurationMilliseconds
	if phases[2].StartMilliseconds != previousEnd {
		t.Fatalf("next phase start = %.17g, want previous encoded end %.17g", phases[2].StartMilliseconds, previousEnd)
	}
}

func TestNativeProfileReceiptBindsAllEvidence(t *testing.T) {
	profileBytes, profile, receipt := testNativeProfileReceipt(t)
	if err := validateNativeProfileReceipt(profileBytes, profile, receipt); err != nil {
		t.Fatalf("valid receipt rejected: %v", err)
	}
	// The nested receipt is an additive, versioned extension. A companion produced by the older
	// v1 writer omits it and retains its original binding shape, so readback stays compatible.
	legacy := receipt
	legacy.Q4KResidency = nil
	legacy.Qwen35DecodeHandoff = nil
	legacy.Controls = mapsStringClone(receipt.Controls)
	delete(legacy.Controls, nativeProfileDecodeHandoffControl)
	legacy.BindingSHA256, _ = nativeReceiptBinding(legacy)
	if err := validateNativeProfileReceipt(profileBytes, profile, legacy); err != nil {
		t.Fatalf("legacy v1 receipt rejected: %v", err)
	}

	tests := []struct {
		name string
		edit func(*nativeProfileReceipt)
	}{
		{name: "artifact bytes", edit: func(r *nativeProfileReceipt) { r.Artifact.Bytes++ }},
		{name: "model config", edit: func(r *nativeProfileReceipt) { r.ModelConfig["hidden_size"] = float64(1) }},
		{name: "host", edit: func(r *nativeProfileReceipt) { r.Host.GPUCores = 17 }},
		{name: "source", edit: func(r *nativeProfileReceipt) { r.Source.Revision = "" }},
		{name: "binary", edit: func(r *nativeProfileReceipt) { r.Binary.SHA256 = strings.Repeat("c", 64) }},
		{name: "raw event", edit: func(r *nativeProfileReceipt) { r.Execution.Events[0].Encoders++ }},
		{name: "fallback aggregate", edit: func(r *nativeProfileReceipt) { r.Fallbacks.PromisedCPUFallbacks++ }},
		{name: "forced q8 upload", edit: func(r *nativeProfileReceipt) { r.Controls["FAK_METAL_Q8_UPLOAD"] = "1" }},
		{name: "worker budget", edit: func(r *nativeProfileReceipt) { r.Controls[nativeControlWorkers] = "0" }},
		{name: "residency count", edit: func(r *nativeProfileReceipt) { r.Q4KResidency.MappedSuccess.Tensors++ }},
		{name: "residency control", edit: func(r *nativeProfileReceipt) { r.Q4KResidency.FAKGGUFMMap = "0" }},
		{name: "residency digest", edit: func(r *nativeProfileReceipt) { r.Q4KResidency.IntegritySHA256 = strings.Repeat("d", 64) }},
		{name: "handoff mode", edit: func(r *nativeProfileReceipt) { r.Qwen35DecodeHandoff.Mode = model.Qwen35DecodeHandoffMixer }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			copy := receipt
			copy.ModelConfig = mapsClone(receipt.ModelConfig)
			copy.Controls = mapsStringClone(receipt.Controls)
			copy.Execution.Events = append([]metalgemm.ExecutionEvent(nil), receipt.Execution.Events...)
			q4kResidencyCopy := *receipt.Q4KResidency
			copy.Q4KResidency = &q4kResidencyCopy
			handoffCopy := *receipt.Qwen35DecodeHandoff
			copy.Qwen35DecodeHandoff = &handoffCopy
			test.edit(&copy)
			if err := validateNativeProfileReceipt(profileBytes, profile, copy); err == nil {
				t.Fatal("tampered receipt was accepted")
			}
		})
	}
}

func mapsStringClone(in map[string]string) map[string]string {
	out := make(map[string]string, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}

func mapsClone(in map[string]any) map[string]any {
	out := make(map[string]any, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}

func TestNativeProfileReadbackRecomputesCompanion(t *testing.T) {
	profileBytes, _, receipt := testNativeProfileReceipt(t)
	dir := t.TempDir()
	profilePath := filepath.Join(dir, "profile.json")
	receiptPath := nativeReceiptPath(profilePath)
	receiptBytes, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(profilePath, profileBytes, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(receiptPath, receiptBytes, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := runNativeProfileReadback(profilePath); err != nil {
		t.Fatalf("readback failed: %v", err)
	}
}

func TestNativeProfileRefusesAnyPromisedMetalFallback(t *testing.T) {
	if err := requireNoMetalFallbacks(0); err != nil {
		t.Fatalf("zero fallback rejected: %v", err)
	}
	if err := requireNoMetalFallbacks(1); err == nil {
		t.Fatal("nonzero fallback accepted")
	}
}

func TestQ4KSmokeDeadlineReportsOnlyAfterLoaderCleanup(t *testing.T) {
	originalResident := loadResidentQ4K
	originalReporter := smokeTimeoutReporter
	defer func() {
		loadResidentQ4K = originalResident
		smokeTimeoutReporter = originalReporter
	}()

	var cleaned atomic.Bool
	loadResidentQ4K = func(ctx context.Context, _ string, _ *ggufload.LoadProfiler) (*model.Model, error) {
		<-ctx.Done()
		// This stands in for the loader's joined workers and closed checkpoint reader.
		cleaned.Store(true)
		return nil, ctx.Err()
	}
	var reported atomic.Bool
	smokeTimeoutReporter = func(_ *benchFlags, _ time.Duration) {
		if !cleaned.Load() {
			t.Fatal("timeout reported before Q4_K loader cleanup completed")
		}
		reported.Store(true)
	}

	f := testCompleteBenchFlags()
	*f.gguf = "fixture.gguf"
	*f.q4k = true
	*f.smoke = true
	*f.smokeDeadline = time.Millisecond
	m, name, err := loadModelMaybeDeadline(f, nil)
	if err != nil || m != nil || name != "" {
		t.Fatalf("loadModelMaybeDeadline = (%v, %q, %v), want timeout-report return", m, name, err)
	}
	if !reported.Load() {
		t.Fatal("SMOKE_LOAD_TIMEOUT was not reported")
	}
}
