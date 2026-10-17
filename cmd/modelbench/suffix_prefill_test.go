package main

import (
	"context"
	"encoding/json"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:justify why=regression when=changed:cmd/modelbench/suffix_prefill.go,cmd/modelbench/main.go
// fak-test:runtime short est=2s lane=default
func TestSuffixPrefillActualCPUSessions(t *testing.T) {
	t.Setenv("FAK_PAGED_KV", "0")
	m := model.NewSynthetic(model.Config{ModelType: "llama", HiddenSize: 8, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: 8, IntermediateSize: 16, VocabSize: 16, RMSNormEps: 1e-5, RopeTheta: 10000})
	defer m.CloseWeights()
	w := suffixPrefillWorkload{Prefix: []int{1, 2}, Suffix: []int{3, 4, 5}, Next: 6, Repetitions: 2, AbsoluteTolerance: 1e-5, RelativeTolerance: 1e-5}
	calls := 0
	report, err := measureSuffixPrefill(func() suffixPrefillSession { calls++; return m.NewSession() }, w, m.Cfg.VocabSize)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 3 || len(report.Samples) != 2 || report.ReferenceSuffixDigest == "" || report.ReferenceNextDigest == "" || !strings.Contains(report.Route, "unverified") {
		t.Fatalf("incomplete witness: calls=%d report=%+v", calls, report)
	}
	for _, s := range report.Samples {
		if s.SuffixDigest == "" || s.NextDigest == "" || s.PrefixNanos <= 0 || s.SuffixNanos <= 0 || s.NextNanos <= 0 || s.TotalNanos <= 0 {
			t.Fatalf("incomplete sample: %+v", s)
		}
	}
}

type suffixTestSession struct {
	pos, closed int
	calls       []string
	fault       string
	borrowed    []float32
}

func (s *suffixTestSession) Prefill(ids []int) []float32 {
	s.calls = append(s.calls, "prefill")
	if s.fault == "nilpanic" {
		panic(nil)
	}
	if s.fault == "panic" || s.fault == "panic-close" {
		panic("selected backend failure")
	}
	s.pos += len(ids)
	return s.logits()
}
func (s *suffixTestSession) Step(id int) []float32 {
	s.calls = append(s.calls, "step")
	s.pos++
	return s.logits()
}
func (s *suffixTestSession) SequencePosition() int {
	if s.fault == "cursor" && s.pos > 0 {
		return -1
	}
	return s.pos
}
func (s *suffixTestSession) Close() {
	s.closed++
	if s.fault == "close" || s.fault == "panic-close" {
		panic("cleanup failed")
	}
}
func (s *suffixTestSession) logits() []float32 {
	if s.fault == "empty" {
		return nil
	}
	if s.borrowed == nil {
		s.borrowed = make([]float32, 2)
	}
	s.borrowed[0] = float32(s.pos)
	s.borrowed[1] = -float32(s.pos)
	if s.fault == "nan" {
		s.borrowed[0] = float32(math.NaN())
	}
	return s.borrowed
}

// fak-test:justify why=regression when=changed:cmd/modelbench/suffix_prefill.go
// fak-test:runtime short est=1s lane=default
func TestSuffixPrefillProtocolFailuresAndBorrowedLogits(t *testing.T) {
	w := suffixPrefillWorkload{Prefix: []int{0, 1}, Suffix: []int{1, 0}, Next: 1, Repetitions: 2}
	var sessions []*suffixTestSession
	factory := func() suffixPrefillSession { s := &suffixTestSession{}; sessions = append(sessions, s); return s }
	report, err := measureSuffixPrefill(factory, w, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(sessions) != 3 || len(report.Samples) != 2 {
		t.Fatal("wrong number of isolated sessions")
	}
	for i, s := range sessions {
		if s.closed != 1 {
			t.Fatalf("session %d leaked", i)
		}
		want := "prefill,prefill,step"
		if i == 0 {
			want = "prefill,step,step,step"
		}
		if strings.Join(s.calls, ",") != want {
			t.Fatalf("calls=%v", s.calls)
		}
	}
	if report.Samples[0].SuffixDigest == report.Samples[0].NextDigest {
		t.Fatal("borrowed suffix logits overwritten by next Step")
	}
	for _, fault := range []string{"panic", "nilpanic", "panic-close", "close", "empty", "nan", "cursor"} {
		t.Run(fault, func(t *testing.T) {
			t.Setenv("GODEBUG", "panicnil=1")
			calls := 0
			var bad *suffixTestSession
			r, e := measureSuffixPrefill(func() suffixPrefillSession {
				calls++
				s := &suffixTestSession{}
				if calls == 2 {
					s.fault = fault
					bad = s
				}
				return s
			}, w, 2)
			if e == nil || r.Error == "" || len(r.Samples) != 1 || r.Samples[0].Error == "" || calls != 2 || bad.closed != 1 {
				t.Fatalf("failed to retain/refuse fault: err=%v report=%+v calls=%d", e, r, calls)
			}
			if fault == "panic-close" && (!strings.Contains(e.Error(), "selected backend failure") || !strings.Contains(e.Error(), "cleanup failed")) {
				t.Fatalf("primary or cleanup failure lost: %v", e)
			}
		})
	}
}

// fak-test:justify why=regression when=changed:cmd/modelbench/suffix_prefill.go
// fak-test:runtime short est=1s lane=default
func TestSuffixPrefillValidationAndTolerance(t *testing.T) {
	valid := suffixPrefillWorkload{Prefix: []int{0}, Suffix: []int{1}, Next: 0, Repetitions: 1}
	for _, mutate := range []func(*suffixPrefillWorkload){func(w *suffixPrefillWorkload) { w.Prefix = nil }, func(w *suffixPrefillWorkload) { w.Suffix = nil }, func(w *suffixPrefillWorkload) { w.Next = -1 }, func(w *suffixPrefillWorkload) { w.Next = 2 }, func(w *suffixPrefillWorkload) { w.Repetitions = 11 }, func(w *suffixPrefillWorkload) { w.AbsoluteTolerance = math.NaN() }, func(w *suffixPrefillWorkload) { w.RelativeTolerance = -1 }, func(w *suffixPrefillWorkload) { w.Suffix = make([]int, 16384) }} {
		w := valid
		mutate(&w)
		called := false
		if _, err := measureSuffixPrefill(func() suffixPrefillSession { called = true; return nil }, w, 2); err == nil || called {
			t.Fatal("invalid workload reached model")
		}
	}
	w := valid
	w.AbsoluteTolerance = .01
	if _, err := compareSuffixLogits([]float32{1.005}, []float32{1}, w); err != nil {
		t.Fatal(err)
	}
	if _, err := compareSuffixLogits([]float32{1.02}, []float32{1}, w); err == nil {
		t.Fatal("drift accepted")
	}
	w.AbsoluteTolerance = 0
	if _, err := compareSuffixLogits([]float32{1.005}, []float32{1}, w); err == nil {
		t.Fatal("undeclared tolerance")
	}
}

// fak-test:justify why=regression when=changed:cmd/modelbench/suffix_prefill.go,cmd/modelbench/main.go
// fak-test:runtime short est=1s lane=default
func TestSuffixPrefillJSONPreloadValidation(t *testing.T) {
	oldIn, oldOut := *suffixPrefillInput, *suffixPrefillOutput
	defer func() { *suffixPrefillInput, *suffixPrefillOutput = oldIn, oldOut }()
	dir := t.TempDir()
	*suffixPrefillInput = filepath.Join(dir, "workload.json")
	*suffixPrefillOutput = filepath.Join(dir, "receipt.json")
	f := testCompleteBenchFlags()
	*f.gguf = filepath.Join(dir, "not-loaded.gguf")
	valid := `{"prefix_ids":[0],"suffix_ids":[1],"next_token":0,"repetitions":1,"absolute_tolerance":0,"relative_tolerance":0}`
	for _, raw := range []string{`{"unknown":1}`, valid + ` {}`, strings.Replace(valid, `"next_token":0`, `"next_token":-1`, 1)} {
		if err := os.WriteFile(*suffixPrefillInput, []byte(raw), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := prepareSuffixPrefill(f); err == nil {
			t.Fatalf("accepted %s", raw)
		}
	}
	if err := os.WriteFile(*suffixPrefillInput, []byte(valid), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := prepareSuffixPrefill(f); err != nil {
		t.Fatal(err)
	}
	*f.smoke = true
	if _, err := prepareSuffixPrefill(f); err == nil {
		t.Fatal("conflicting mode accepted")
	}
	*f.smoke = false
	if err := os.WriteFile(*suffixPrefillOutput, []byte("preserve"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := prepareSuffixPrefill(f); err == nil {
		t.Fatal("existing receipt accepted")
	}
}

// fak-test:justify why=integration when=changed:cmd/modelbench/suffix_prefill.go,cmd/modelbench/main.go
// fak-test:runtime medium est=8s lane=default
func TestSuffixPrefillExistingGGUFLoader(t *testing.T) {
	t.Setenv("FAK_PAGED_KV", "0")
	f := testCompleteBenchFlags()
	*f.gguf = engProjGGUF(t)
	*f.q4k = true
	m, _, err := loadModelContext(context.Background(), f, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer m.CloseWeights()
	w := suffixPrefillWorkload{Prefix: []int{1, 2}, Suffix: []int{3, 4}, Next: 5, Repetitions: 1, AbsoluteTolerance: 1e-5, RelativeTolerance: 1e-5}
	report, err := measureSuffixPrefill(func() suffixPrefillSession { return newBenchSession(m, f, nil) }, w, m.Cfg.VocabSize)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Samples) != 1 {
		t.Fatalf("missing loaded model sample: %+v", report)
	}
}

// fak-test:justify why=regression when=changed:cmd/modelbench/suffix_prefill.go
// fak-test:runtime short est=1s lane=default
func TestSuffixPrefillGPUCoverage(t *testing.T) {
	now := time.Now()
	if got := suffixGPUWindow(nil, nil, "cpu-legacy", now, now); got.Profile != nil || got.Status != "unavailable" {
		t.Fatal(got)
	}
	before := &compute.KernelProfile{Backend: "test-gpu"}
	after := &compute.KernelProfile{Backend: "test-gpu", Ops: 1, GPUNS: 100, Rows: []compute.KernelRow{{Kernel: "test", Calls: 1, NS: 100}}}
	got := suffixGPUWindow(before, after, "test-gpu", now, now)
	if got.Profile == nil || got.Profile.GPUNS != 100 || !strings.Contains(got.Scope, "process-global") {
		t.Fatal(got)
	}
	after.UntimedCompute = 1
	if got := suffixGPUWindow(before, after, "test-gpu", now, now); !strings.Contains(got.Status, "partial") {
		t.Fatal(got)
	}
	if got := suffixGPUWindow(before, after, "other", now, now); got.Profile != nil {
		t.Fatal("cross-backend profile accepted")
	}
	before.GPUNS = 200
	if got := suffixGPUWindow(before, after, "test-gpu", now, now); got.Profile != nil {
		t.Fatal("counter reset accepted")
	}
}

// fak-test:justify why=integration when=changed:cmd/modelbench/suffix_prefill.go
// fak-test:runtime short est=2s lane=default
func TestSuffixPrefillCLIReceiptAndExclusiveOutput(t *testing.T) {
	oldOut, oldIdentity := *suffixPrefillOutput, suffixBuildIdentity
	defer func() { *suffixPrefillOutput = oldOut; suffixBuildIdentity = oldIdentity }()
	// Tests do not carry a production VCS stamp. Bind a declared fixture stamp
	// plus the actual test executable digest; production keeps captureNativeBuild.
	suffixBuildIdentity = func() (nativeSourceIdentity, nativeFileIdentity, error) {
		p, e := os.Executable()
		if e != nil {
			return nativeSourceIdentity{}, nativeFileIdentity{}, e
		}
		b, e := fileIdentity(p)
		return nativeSourceIdentity{Revision: "test-fixture"}, b, e
	}
	dir := t.TempDir()
	*suffixPrefillOutput = filepath.Join(dir, "receipt.json")
	f := testCompleteBenchFlags()
	*f.gguf = filepath.Join(dir, "artifact.gguf")
	if err := os.WriteFile(*f.gguf, []byte("identity fixture; loader separately covered"), 0600); err != nil {
		t.Fatal(err)
	}
	m := model.NewSynthetic(model.Config{ModelType: "llama", HiddenSize: 8, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: 8, IntermediateSize: 16, VocabSize: 16, RMSNormEps: 1e-5, RopeTheta: 10000})
	defer m.CloseWeights()
	w := suffixPrefillWorkload{Prefix: []int{1}, Suffix: []int{2}, Next: 3, Repetitions: 1, AbsoluteTolerance: 1e-5, RelativeTolerance: 1e-5}
	if err := runSuffixPrefillCLI(f, m, "fixture", &w, m.NewSession); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(*suffixPrefillOutput)
	if err != nil {
		t.Fatal(err)
	}
	var report suffixPrefillReport
	if err = json.Unmarshal(b, &report); err != nil {
		t.Fatal(err)
	}
	identity, _ := fileIdentity(*f.gguf)
	if report.ArtifactCoverage != "input-file-only; partial checkpoint coverage" || report.ArtifactRequirement == "" || !strings.Contains(report.HashingEffect, "warm page cache") || report.Artifact != identity || report.Binary.SHA256 == "" || len(report.Samples) != 1 || report.Samples[0].Backend != "cpu-legacy" {
		t.Fatalf("unbound report: %+v", report)
	}
	if err := runSuffixPrefillCLI(f, m, "fixture", &w, m.NewSession); err == nil {
		t.Fatal("overwrote existing receipt")
	}
	after, _ := os.ReadFile(*suffixPrefillOutput)
	if string(after) != string(b) {
		t.Fatal("receipt changed")
	}
	*suffixPrefillOutput = filepath.Join(dir, "failure.json")
	if err := runSuffixPrefillCLI(f, m, "fixture", &w, func() *model.Session { panic("selected CLI failure") }); err == nil {
		t.Fatal("failure accepted")
	}
	raw, err := os.ReadFile(*suffixPrefillOutput)
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &report); err != nil || report.Error == "" || report.Reference.Error == "" {
		t.Fatalf("failure receipt missing: %s %v", raw, err)
	}
}
