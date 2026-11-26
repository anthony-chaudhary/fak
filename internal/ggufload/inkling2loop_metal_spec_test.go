//go:build darwin && arm64 && cgo

package ggufload

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// inkling2MetalChunkBackendFor is be when it has every op the speculative chunk
// driver calls; the production Metal backend implements them all.
func inkling2MetalChunkBackendFor(be compute.Backend) (compute.Backend, func(), error) {
	if !inkling2SpecCapable(be) {
		return nil, nil, fmt.Errorf("backend %s has no inkling2 prefill and verify ops", be.Name())
	}
	return be, func() {}, nil
}

// inkling2MetalSpecLoad is the host load beside a timing: every number here is
// taken on a shared machine.
func inkling2MetalSpecLoad() string {
	out, err := exec.Command("sysctl", "-n", "vm.loadavg").Output()
	if err != nil {
		return "load average unavailable"
	}
	return "load average " + strings.TrimSpace(string(out))
}

// FAK_INKLING2_TEST_GGUF runs fak's chain speculation on the Apple GPU: the
// multi-row verify must be width-invariant on the real model, and speculative
// greedy must reproduce the resident path's ids with accepts, rejects and
// history rewinds in play.
//
// fak-test:runtime slow est=240s
func TestInkling2MetalSpecGreedyMatchesResident(t *testing.T) {
	path := os.Getenv("FAK_INKLING2_TEST_GGUF")
	if path == "" {
		t.Skip("set FAK_INKLING2_TEST_GGUF to run on Apple GPU")
	}
	lease, err := gpulease.Acquire(gpulease.Options{Timeout: 8 * time.Minute, Logf: t.Logf})
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
	const n = 64
	prompt := inkling2OracleCases[0].prompt

	// The first run compiles every kernel and faults the weights in.
	var ref []int
	var plain Inkling2LoopGreedyStats
	for run := range 2 {
		if ref, plain, err = inkling2ResidentGreedy(m, be, prompt, n); err != nil {
			t.Fatal(err)
		}
		t.Logf("resident run %d: prefill %s, %d decode steps in %s (%.2f tok/s), %s",
			run, plain.Prefill, plain.DecodeSteps, plain.Decode, plain.DecodeTokensPerSecond(), inkling2MetalSpecLoad())
	}
	t.Logf("resident ids %v", ref)

	sbe, release, err := inkling2MetalChunkBackendFor(be)
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	if inkling2VerifyMMA(sbe) {
		ref, plain = inkling2MetalPlain(t, m, be, sbe, prompt, n)
		t.Logf("MMA plain ids %v", ref)
	}
	// FAK_INKLING2_GRAPH stays at its default: Metal has no graph capture, and
	// a full-width verify has to fall back to the eager chunk on its own.
	t.Setenv("FAK_INKLING2_SPEC", "4")

	t.Run("verify rows are width invariant", func(t *testing.T) { inkling2MetalSpecWidths(t, m, sbe, prompt, ref) })

	check := func(name string, depth int, mk func() inkling2Drafter) {
		got, st, err := inkling2ResidentSpecGreedy(m, sbe, prompt, n, depth, func(*inkling2Resident) (inkling2Drafter, error) { return mk(), nil })
		if err != nil {
			t.Fatalf("%s depth %d: %v", name, depth, err)
		}
		perVerify := float64(st.Accepted) / float64(max(1, st.Verifies))
		t.Logf("spec %s depth %d: %.2f tok/s (plain %.2f); %d verifies, accepted %d/%d (%.2f per verify), plain steps %d, verify %.1f ms/call, prefill %s, %s",
			name, depth, st.DecodeTokensPerSecond(), plain.DecodeTokensPerSecond(), st.Verifies, st.Accepted, st.Drafted, perVerify,
			st.PlainSteps, float64(st.Verify.Microseconds())/1000/float64(max(1, st.Verifies)), st.Prefill, inkling2MetalSpecLoad())
		if !slices.Equal(got, ref) {
			t.Errorf("spec %s depth %d decoded %v, resident %v", name, depth, got, ref)
		}
	}
	lookup := func() inkling2Drafter { return newInkling2LookupDrafter(prompt) }
	oracle := func(corrupt func(int) bool) func() inkling2Drafter {
		return func() inkling2Drafter { return &inkling2OracleDrafter{ref: ref, corrupt: corrupt} }
	}
	check("prompt lookup", 4, lookup)
	check("prompt lookup", 8, lookup)
	check("oracle with rejects", 4, oracle(func(i int) bool { return i%5 == 2 }))
	check("oracle with rejects", 7, oracle(func(i int) bool { return i%9 == 4 || i%13 == 0 }))
	for _, depth := range []int{1, 2, 4, 8} {
		check("oracle", depth, oracle(nil))
	}
	t.Run("verify cost by width", func(t *testing.T) { inkling2MetalSpecCost(t, m, sbe, prompt, ref) })
	t.Run("lookup draft by workload", func(t *testing.T) { inkling2MetalSpecLookupWorkloads(t, path, m, be, sbe) })
}

// inkling2MetalSpecLookupWorkloads times plain greedy against the production
// lookup draft, selected through FAK_INKLING2_SPEC_DRAFT, on a prompt the reply
// repeats and on open chat, holding the ids to plain greedy on both.
func inkling2MetalSpecLookupWorkloads(t *testing.T, path string, m *Inkling2LoopModel, be, sbe compute.Backend) {
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
	const code = `func sumPositive(xs []int) int {
	acc := 0
	for _, x := range xs {
		if x > 0 {
			acc += x
		}
	}
	return acc
}`
	workloads := []struct {
		name   string
		prompt []int
	}{
		{"code edit", chat("Rename the variable acc to total in this Go function and reply with the whole function only, no prose:\n\n" + code)},
		{"open chat", chat("Tell me something interesting about octopuses.")},
	}
	t.Setenv("FAK_INKLING2_SPEC_DRAFT", inkling2DraftLookup)
	const n = 96
	for _, w := range workloads {
		want, plain := inkling2MetalPlain(t, m, be, sbe, w.prompt, n)
		for _, depth := range []int{4, 8} {
			got, _, err := inkling2SpecGreedy(m, sbe, w.prompt, n, depth)
			if err != nil {
				t.Fatalf("%s lookup depth %d: %v", w.name, depth, err)
			}
			st := inkling2LastSpec
			t.Logf("workload %s (%d prompt tokens): plain %.2f tok/s, lookup depth %d %.2f tok/s (%.2fx); %d verifies, accepted %d/%d, plain steps %d, %s",
				w.name, len(w.prompt), plain.DecodeTokensPerSecond(), depth, st.DecodeTokensPerSecond(),
				st.DecodeTokensPerSecond()/plain.DecodeTokensPerSecond(), st.Verifies, st.Accepted, st.Drafted, st.PlainSteps, inkling2MetalSpecLoad())
			if !slices.Equal(got, want) {
				t.Errorf("%s lookup depth %d decoded %v, plain %v", w.name, depth, got, want)
			}
		}
	}
}

// inkling2MetalSpecSession prefills prompt into a session with verify state
// for the deepest draft and returns it with a rewind to that point.
func inkling2MetalSpecSession(t *testing.T, m *Inkling2LoopModel, be compute.Backend, prompt, ref []int) (*inkling2Resident, func()) {
	s, err := newInkling2Resident(m, be)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.close)
	if err := s.enableSpec(inkling2SpecMaxDraft); err != nil {
		t.Fatal(err)
	}
	next, err := s.prefill(context.Background(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	if next != ref[0] {
		t.Fatalf("chunked prefill predicts %d, resident %d", next, ref[0])
	}
	base := s.pos
	hp := s.hp
	histFloats := (2*hp.KVHeads*hp.HeadDim + 2*hp.Hidden) * (hp.ConvKernel - 1)
	hists := make([][]float32, len(s.hist))
	for slot := range hists {
		if hists[slot], err = s.dev.Ink2Read(s.hist[slot], histFloats); err != nil {
			t.Fatal(err)
		}
	}
	return s, func() {
		for slot := range hists {
			if err := s.dev.Ink2Write(s.hist[slot], hists[slot]); err != nil {
				t.Fatal(err)
			}
		}
		s.pos = base
	}
}

// inkling2MetalSpecWidths feeds the same eight tokens through verify chunks of
// 1, 2, 4 and 8 rows from one state and requires every row's logits to carry
// the same bits at every width.
func inkling2MetalSpecWidths(t *testing.T, m *Inkling2LoopModel, be compute.Backend, prompt, ref []int) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	s, rewind := inkling2MetalSpecSession(t, m, be, prompt, ref)
	const rows = 8
	toks := ref[:rows]
	vocab := s.output.rows
	var want []float32
	var wantIDs []int
	for _, width := range []int{1, 2, 4, 8} {
		rewind()
		logits, ids := make([]float32, 0, rows*vocab), make([]int, 0, rows)
		for at0 := 0; at0 < rows; at0 += width {
			got, err := s.verify(toks[at0 : at0+width])
			if err != nil {
				t.Fatalf("width %d: %v", width, err)
			}
			l, err := s.dev.Ink2Read(s.verifyLogits, width*vocab)
			if err != nil {
				t.Fatal(err)
			}
			if err := s.commitVerify(width, width); err != nil {
				t.Fatal(err)
			}
			logits, ids = append(logits, l...), append(ids, got...)
		}
		if width == 1 {
			want, wantIDs = logits, ids
			if !slices.Equal(ids, ref[1:rows+1]) {
				t.Errorf("single-row verifies predict %v, resident %v", ids, ref[1:rows+1])
			}
			continue
		}
		if !slices.Equal(ids, wantIDs) {
			t.Errorf("width %d predicts %v, single rows %v", width, ids, wantIDs)
		}
		for i := range want {
			if math.Float32bits(logits[i]) != math.Float32bits(want[i]) {
				t.Fatalf("width %d: row %d logit %d = %v (%#x), alone %v (%#x)", width, i/vocab, i%vocab,
					logits[i], math.Float32bits(logits[i]), want[i], math.Float32bits(want[i]))
			}
		}
		t.Logf("width %d: %d rows of %d logits bit-identical to single-row verifies", width, rows, vocab)
	}
}

// inkling2MetalSpecCost logs the wall time of one verify chunk by width, each
// from the same state.
func inkling2MetalSpecCost(t *testing.T, m *Inkling2LoopModel, be compute.Backend, prompt, ref []int) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	s, rewind := inkling2MetalSpecSession(t, m, be, prompt, ref)
	const reps = 4
	feed := ref[:inkling2SpecMaxDraft+1]
	var one time.Duration
	for _, width := range []int{1, 2, 3, 5, 9} {
		var total time.Duration
		for r := 0; r < reps+1; r++ {
			rewind()
			t0 := time.Now()
			if _, err := s.verify(feed[:width]); err != nil {
				t.Fatalf("width %d: %v", width, err)
			}
			if r > 0 {
				total += time.Since(t0)
			}
		}
		per := total / reps
		if width == 1 {
			one = per
		}
		t.Logf("verify cost: %d rows %.1f ms = %.2fx one row, %.1f ms/row, %s",
			width, float64(per.Microseconds())/1000, per.Seconds()/one.Seconds(), float64(per.Microseconds())/1000/float64(width), inkling2MetalSpecLoad())
	}
}

// FAK_INKLING2_TEST_GGUF times verify chunks of 1, 2, 5 and 8 rows from one
// state, interleaved and best of several, and with FAK_INK2_PROFILE set logs
// the per-dispatch GPU profile of each width: where an extra row's cost goes.
//
// fak-test:runtime slow est=90s
func TestInkling2MetalSpecVerifyCostByWidth(t *testing.T) {
	path := os.Getenv("FAK_INKLING2_TEST_GGUF")
	if path == "" {
		t.Skip("set FAK_INKLING2_TEST_GGUF to run on Apple GPU")
	}
	lease, err := gpulease.Acquire(gpulease.Options{Timeout: 8 * time.Minute, Logf: t.Logf})
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
	prompt := inkling2OracleCases[0].prompt
	ref, _, err := inkling2ResidentGreedy(m, be, prompt, inkling2SpecMaxDraft+2)
	if err != nil {
		t.Fatal(err)
	}
	sbe, release, err := inkling2MetalChunkBackendFor(be)
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	t.Setenv("FAK_INKLING2_SPEC", "4")
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	s, rewind := inkling2MetalSpecSession(t, m, sbe, prompt, ref)
	widths := []int{1, 2, 5, 8}
	feed := ref[:widths[len(widths)-1]]
	for _, w := range widths {
		rewind()
		if _, err := s.verify(feed[:w]); err != nil {
			t.Fatalf("warm width %d: %v", w, err)
		}
	}
	reps := 6
	if v := os.Getenv("FAK_INKLING2_VERIFY_REPS"); v != "" {
		fmt.Sscan(v, &reps)
	}
	arms := []string{"new"}
	if inkling2VerifyMMA(sbe) {
		arms = []string{"gemv", "mma"}
	}
	best := make([][]time.Duration, len(arms))
	for a := range arms {
		best[a] = make([]time.Duration, len(widths))
	}
	for r := 0; r < reps; r++ {
		for a, arm := range arms {
			switch arm {
			case "gemv":
				t.Setenv("FAK_INKLING2_VERIFY_MMA", "0")
			case "mma":
				t.Setenv("FAK_INKLING2_VERIFY_MMA", "1")
			}
			for i, w := range widths {
				rewind()
				t0 := time.Now()
				if _, err := s.verify(feed[:w]); err != nil {
					t.Fatalf("width %d: %v", w, err)
				}
				if d := time.Since(t0); best[a][i] == 0 || d < best[a][i] {
					best[a][i] = d
				}
			}
		}
	}
	ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }
	for a, name := range arms {
		var line strings.Builder
		for i, w := range widths {
			extra := 0.0
			if w > 1 {
				extra = (ms(best[a][i]) - ms(best[a][0])) / float64(w-1)
			}
			fmt.Fprintf(&line, " w%d=%.1fms(%.2fx,+%.1fms/row)", w, ms(best[a][i]), best[a][i].Seconds()/best[a][0].Seconds(), extra)
		}
		t.Logf("verify_cost_by_width %s best of %d:%s; %s", name, reps, line.String(), inkling2MetalSpecLoad())
	}
	if p, ok := compute.MetalInkling2Profile(); ok && p.Enabled() {
		if len(arms) > 1 {
			t.Setenv("FAK_INKLING2_VERIFY_MMA", "1")
		}
		for _, w := range widths {
			rewind()
			if err := compute.MetalInkling2ProfileReset(); err != nil {
				t.Fatal(err)
			}
			st0, _ := compute.MetalInkling2Stats()
			t0 := time.Now()
			if _, err := s.verify(feed[:w]); err != nil {
				t.Fatal(err)
			}
			wall := time.Since(t0)
			st1, _ := compute.MetalInkling2Stats()
			p, _ := compute.MetalInkling2Profile()
			t.Logf("verify profile width %d:\n%s", w, p.Format(1, wall, st1.Sub(st0)))
		}
	}
}
