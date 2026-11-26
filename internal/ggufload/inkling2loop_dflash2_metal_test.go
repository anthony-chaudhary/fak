//go:build darwin && arm64 && cgo

package ggufload

import (
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// FAK_INKLING2_TEST_GGUF with the DFlash2 checkpoint beside it runs fak's
// chain speculation with the device DFlash2 draft on the Apple GPU: on code,
// open chat and math, at depths 4 and 7, the speculative ids must equal plain
// resident greedy, and acceptance and tok/s are logged beside prompt lookup.
//
// fak-test:runtime slow est=600s
func TestInkling2MetalSpecDFlash2GreedyMatchesResident(t *testing.T) {
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
		if !errors.Is(err, gpulease.ErrTimeout) || attempt == 2 {
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
	// a set depth enables the batched verify state a spec session needs
	t.Setenv("FAK_INKLING2_SPEC", "4")
	if got := inkling2ResolveSpecDraft(inkling2DraftAuto, "", inkling2DFlash2DirFor(m, sbe)); got != inkling2DraftDFlash {
		t.Fatalf("auto with %s beside the GGUF resolves to %q", Inkling2DFlash2DirName, got)
	}

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
		{"open chat", chat("Tell me something interesting about octopuses.")},
		{"math", chat("What is 17 times 23? Show the multiplication step by step.")},
	}
	const n = 64
	// The first run compiles every kernel and faults the target weights in.
	if _, _, err := inkling2ResidentGreedy(m, be, workloads[0].prompt, 8); err != nil {
		t.Fatal(err)
	}
	submissions := func() uint64 { st, _ := compute.MetalInkling2Stats(); return st.Submissions }
	inkling2DFlash2Fences = submissions
	defer func() { inkling2DFlash2Fences = nil }()
	type tally struct {
		verifies, accepted, drafted, draftedVerifies int
		draft                                        time.Duration
		run                                          inkling2DFlash2RunStats
	}
	totals := map[string]*tally{}
	for _, w := range workloads {
		want, plain, err := inkling2ResidentGreedy(m, be, w.prompt, n)
		if err != nil {
			t.Fatalf("%s plain: %v", w.name, err)
		}
		t.Logf("workload %s (%d prompt tokens): plain %.2f tok/s, %s", w.name, len(w.prompt), plain.DecodeTokensPerSecond(), inkling2MetalSpecLoad())
		for _, run := range []struct {
			kind              string
			depth             int
			rowsHead, unfused bool
		}{{inkling2DraftDFlash, 4, false, false}, {inkling2DraftDFlash, 7, true, false}, {inkling2DraftDFlash, 7, false, false},
			{inkling2DraftDFlash, 7, false, true}, {inkling2DraftLookup, 4, false, false}} {
			t.Setenv("FAK_INKLING2_SPEC_DRAFT", run.kind)
			inkling2DFlash2RowsHead, inkling2DFlash2Unfused = run.rowsHead, run.unfused
			inkling2DFlash2LastRun = inkling2DFlash2RunStats{}
			sub0 := submissions()
			got, _, err := inkling2SpecGreedy(m, sbe, w.prompt, n, run.depth)
			subs := submissions() - sub0
			inkling2DFlash2RowsHead, inkling2DFlash2Unfused = false, false
			if err != nil {
				t.Fatalf("%s %s depth %d: %v", w.name, run.kind, run.depth, err)
			}
			st := inkling2LastSpec
			label := run.kind
			if run.rowsHead {
				label += " (rows head)"
			}
			if run.unfused {
				label += " (unfused)"
			}
			t.Logf("  %s depth %d: %.2f tok/s (%.2fx plain); %d verifies, accepted %d/%d (mean %.2f per verify), plain steps %d, draft %.1f ms/call, verify %.1f ms/call",
				label, run.depth, st.DecodeTokensPerSecond(), st.DecodeTokensPerSecond()/plain.DecodeTokensPerSecond(),
				st.Verifies, st.Accepted, st.Drafted, float64(st.Accepted)/float64(max(1, st.Verifies)), st.PlainSteps,
				float64(st.Draft.Microseconds())/1000/float64(max(1, st.Verifies)),
				float64(st.Verify.Microseconds())/1000/float64(max(1, st.Verifies)))
			if !slices.Equal(got, want) {
				t.Errorf("%s %s depth %d decoded %v, plain %v", w.name, run.kind, run.depth, got, want)
			}
			key := run.kind + " d" + string(rune('0'+run.depth))
			if run.rowsHead {
				key += " rows head"
			}
			if run.unfused {
				key += " unfused"
			}
			if totals[key] == nil {
				totals[key] = &tally{}
			}
			c := totals[key]
			c.verifies += st.Verifies
			c.accepted += st.Accepted
			c.drafted += st.Drafted
			c.draft += st.Draft
			if run.kind != inkling2DraftDFlash {
				c.draftedVerifies += st.Verifies
				continue
			}
			r := inkling2DFlash2LastRun
			dv := 0
			for _, k := range r.depths[1:] {
				dv += k
			}
			c.draftedVerifies += dv
			for i := range r.depths {
				c.run.depths[i] += r.depths[i]
				c.run.stepTime[i] += r.stepTime[i]
			}
			for j := range r.judged {
				c.run.judged[j] += r.judged[j]
				c.run.accepted[j] += r.accepted[j]
			}
			c.run.draftFences += r.draftFences
			t.Logf("    depths %v; %.2f command buffers per drafted call, %.2f per verify over the run (%d)",
				r.depths, float64(r.draftFences)/float64(max(1, dv)), float64(subs)/float64(max(1, st.Verifies)), subs)
			t.Logf("    per-position acceptance %s", inkling2DFlash2PositionLine(r))
		}
	}
	for _, key := range []string{"dflash2 d4", "dflash2 d7 rows head", "dflash2 d7", "dflash2 d7 unfused", "lookup d4"} {
		c := totals[key]
		// an undrafted width-1 verify accepts nothing by construction, so only drafted verifies count
		t.Logf("%s over %d workloads: mean accepted %.2f per drafted verify (%d/%d drafted over %d of %d verifies), draft %.1f ms/drafted call",
			key, len(workloads), float64(c.accepted)/float64(max(1, c.draftedVerifies)), c.accepted, c.drafted, c.draftedVerifies, c.verifies,
			float64(c.draft.Microseconds())/1000/float64(max(1, c.draftedVerifies)))
		if strings.HasPrefix(key, "dflash2") {
			t.Logf("    depth histogram %v; acceptance %s", c.run.depths, inkling2DFlash2PositionLine(c.run))
		}
	}
	if c := totals["dflash2 d7"]; float64(c.accepted)/float64(max(1, c.draftedVerifies)) < 1.5 {
		t.Errorf("dflash2 depth 7 accepts %.2f per drafted verify; a wired draft accepts several", float64(c.accepted)/float64(max(1, c.draftedVerifies)))
	}
}

// inkling2DFlash2PositionLine renders a run's conditional acceptance per draft
// position and its verify+commit wall time by drafted length.
func inkling2DFlash2PositionLine(r inkling2DFlash2RunStats) string {
	var b strings.Builder
	for j := range r.judged {
		if r.judged[j] > 0 {
			fmt.Fprintf(&b, " p%d %d/%d=%.2f", j, r.accepted[j], r.judged[j], float64(r.accepted[j])/float64(r.judged[j]))
		}
	}
	b.WriteString("; verify ms by drafted n:")
	for n, k := range r.depths {
		if k > 0 {
			fmt.Fprintf(&b, " %d:%.1f", n, float64(r.stepTime[n].Microseconds())/1000/float64(k))
		}
	}
	return b.String()
}
