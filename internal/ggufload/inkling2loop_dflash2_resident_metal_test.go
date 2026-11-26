//go:build darwin && arm64 && cgo

package ggufload

import (
	"errors"
	"os"
	"slices"
	"syscall"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func inkling2RusageFaults() inkling2DFlash2Faults {
	var ru syscall.Rusage
	if syscall.Getrusage(syscall.RUSAGE_SELF, &ru) != nil {
		return inkling2DFlash2Faults{}
	}
	return inkling2DFlash2Faults{int64(ru.Majflt), int64(ru.Minflt)}
}

// FAK_INKLING2_TEST_GGUF with the DFlash2 checkpoint beside it A/Bs the
// per-request draft open (uncached: map, store and wire the checkpoint every
// request) against the process checkpoint cache: the draft's first-call wait
// and page faults, and tok/s against plain decode of the same workload in the
// same run. The first spec run after each plain run is logged as the request
// that follows plain decode and kept out of the steady-state means. Ids must
// equal plain greedy on every run.
//
// fak-test:runtime slow est=600s
func TestInkling2MetalSpecDFlash2Resident(t *testing.T) {
	path := os.Getenv("FAK_INKLING2_TEST_GGUF")
	if path == "" {
		t.Skip("set FAK_INKLING2_TEST_GGUF to run on Apple GPU")
	}
	dir, err := Inkling2DFlash2AdjacentDir(path)
	if err != nil || dir == "" {
		t.Skipf("no %s beside %s (%v)", Inkling2DFlash2DirName, path, err)
	}
	var lease *gpulease.Lease
	for attempt := 0; ; attempt++ {
		lease, err = gpulease.Acquire(gpulease.Options{Timeout: 5 * time.Minute, Logf: t.Logf})
		if !errors.Is(err, gpulease.ErrTimeout) || attempt == 5 {
			break
		}
	}
	if errors.Is(err, gpulease.ErrTimeout) {
		t.Skipf("NOT RUN: %v", err)
	}
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()

	m, be, err := OpenInkling2LoopMetalDevice(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	sbe, release, err := inkling2MetalChunkBackendFor(be)
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	t.Setenv("FAK_INKLING2_MTP_DIR", "")
	t.Setenv("FAK_INKLING2_SPEC", "7")
	t.Setenv("FAK_INKLING2_SPEC_DRAFT", inkling2DraftDFlash)

	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	table, ok := f.GGMLTokenizer()
	if !ok {
		t.Fatal("checkpoint has no embedded BPE tokenizer")
	}
	tok, err := tokenizer.FromGGML(table.Tokens, table.Merges, table.TokenTypes, table.Pre)
	if err != nil {
		t.Fatal(err)
	}
	control := func(s string) tokenizer.Segment { return tokenizer.Segment{Text: s, Trusted: true} }
	chat := func(user string) []int {
		ids, err := tok.EncodeSegments([]tokenizer.Segment{
			control("<|message_system|><|content_text|>"), {Text: "Thinking effort level: 0.0"}, control("<|end_message|>"),
			control("<|message_user|><|content_text|>"), {Text: user}, control("<|end_message|>"),
		})
		if err != nil {
			t.Fatal(err)
		}
		return ids
	}
	workloads := []struct {
		name   string
		prompt []int
	}{
		{"code", chat("Write a Go function that returns the n-th Fibonacci number iteratively. Reply with code only.")},
		{"chat", chat("Tell me something interesting about octopuses.")},
		{"math", chat("What is 17 times 23? Show the multiplication step by step.")},
	}
	const n, depth = 64, 7
	if _, _, err := inkling2ResidentGreedy(m, be, workloads[0].prompt, 8); err != nil {
		t.Fatal(err)
	}
	inkling2DFlash2FaultCounter = inkling2RusageFaults
	inkling2DFlash2Fences = func() uint64 { st, _ := compute.MetalInkling2Stats(); return st.Submissions }
	defer func() {
		inkling2DFlash2FaultCounter, inkling2DFlash2Fences, inkling2DFlash2Uncached = nil, nil, false
	}()

	type agg struct {
		runs                    int
		ratio, wait, tokPerStep float64
		firstMaj, firstMin      int64
	}
	sums := map[string]*agg{}
	ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }
	spec := func(mode, phase string, w int, want []int, plain float64) {
		inkling2DFlash2Uncached = mode == "uncached"
		inkling2DFlash2LastRun = inkling2DFlash2RunStats{}
		f0 := inkling2RusageFaults()
		t0 := time.Now()
		got, _, err := inkling2SpecGreedy(m, sbe, workloads[w].prompt, n, depth)
		wall := time.Since(t0)
		run := inkling2RusageFaults().sub(f0)
		inkling2DFlash2Uncached = false
		if err != nil {
			t.Fatalf("%s %s: %v", mode, workloads[w].name, err)
		}
		if !slices.Equal(got, want) {
			t.Errorf("%s %s %s decoded %v, plain %v", mode, phase, workloads[w].name, got, want)
		}
		st, r, o := inkling2LastSpec, inkling2DFlash2LastRun, inkling2DFlash2LastOpen
		steps := st.Verifies + st.PlainSteps
		tps := float64(len(got)) / float64(max(1, steps))
		wait := o.took + r.attach + r.first
		ratio := st.DecodeTokensPerSecond() / plain
		t.Logf("  %s %-6s %s: %.2f tok/s (%.2fx plain), %.2f tok/step, depths %v; call#0 wait %.1f ms = open %.1f (maj %d min %d) + attach %.1f (maj %d min %d) + first draft %.1f (maj %d min %d); run faults maj %d min %d, wall %.0f ms",
			mode, phase, workloads[w].name, st.DecodeTokensPerSecond(), ratio, tps, r.depths, ms(wait),
			ms(o.took), o.faults.major, o.faults.minor, ms(r.attach), r.attachFaults.major, r.attachFaults.minor,
			ms(r.first), r.firstFaults.major, r.firstFaults.minor, run.major, run.minor, ms(wall))
		key := mode + " " + phase
		if sums[key] == nil {
			sums[key] = &agg{}
		}
		a := sums[key]
		a.runs++
		a.ratio += ratio
		a.wait += ms(wait)
		a.tokPerStep += tps
		a.firstMaj += o.faults.major + r.attachFaults.major + r.firstFaults.major
		a.firstMin += o.faults.minor + r.attachFaults.minor + r.firstFaults.minor
	}
	// the cache serves a long-lived process, so it is primed once before timing
	if _, _, err := inkling2SpecGreedy(m, sbe, workloads[0].prompt, 8, depth); err != nil {
		t.Fatal(err)
	}
	const reps = 2
	for rep := 0; rep < reps; rep++ {
		modes := []string{"uncached", "cached"}
		if rep%2 == 1 {
			modes[0], modes[1] = modes[1], modes[0]
		}
		for _, mode := range modes {
			for w := range workloads {
				want, plain, err := inkling2ResidentGreedy(m, be, workloads[w].prompt, n)
				if err != nil {
					t.Fatal(err)
				}
				t.Logf("rep %d %s %s: plain %.2f tok/s, %s", rep, mode, workloads[w].name, plain.DecodeTokensPerSecond(), inkling2MetalSpecLoad())
				spec(mode, "after-plain", w, want, plain.DecodeTokensPerSecond())
				spec(mode, "steady", w, want, plain.DecodeTokensPerSecond())
				spec(mode, "steady", w, want, plain.DecodeTokensPerSecond())
			}
		}
	}
	for _, key := range []string{"uncached after-plain", "cached after-plain", "uncached steady", "cached steady"} {
		a := sums[key]
		k := float64(max(1, a.runs))
		t.Logf("%s over %d runs: mean %.3fx plain, %.3f tok/step, call#0 wait %.1f ms, call#0 faults maj %.0f min %.0f",
			key, a.runs, a.ratio/k, a.tokPerStep/k, a.wait/k, float64(a.firstMaj)/k, float64(a.firstMin)/k)
	}
}
