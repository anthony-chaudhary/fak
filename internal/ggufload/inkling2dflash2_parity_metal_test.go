//go:build darwin && arm64 && cgo

package ggufload

import (
	"context"
	"errors"
	"os"
	"slices"
	"sort"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// The device DFlash2 draft and the host reference, fed the same device target
// taps, teacher-forced over the target's greedy continuation: at every anchor
// both propose a full block, and the ids must agree per block. The device
// draft's per-call time and both acceptance means are logged.
//
// fak-test:runtime slow est=900s
func TestInkling2MetalDFlash2ProposalParity(t *testing.T) {
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
	t.Setenv("FAK_INKLING2_SPEC", "7")
	dm, err := OpenInkling2DFlash2(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer dm.Close()
	gamma := dm.Config.DFlash.Block - 1
	// the host reference reads the LM head on the CPU
	tm, err := OpenInkling2Loop(path)
	if err != nil {
		t.Fatal(err)
	}
	defer tm.Close()

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
		{"math", chat("What is 17 times 23? Show the multiplication step by step.")},
	}
	if len(inkling2OracleCases) > 0 {
		workloads = append(workloads, struct {
			name   string
			prompt []int
		}{"oracle " + inkling2OracleCases[0].name, inkling2OracleCases[0].prompt})
	}
	const n = 40
	var blocks, agreeBlocks, agreeIDs, devAcc, hostAcc int
	var draftTimes []time.Duration
	for _, w := range workloads {
		ref, _, err := inkling2ResidentGreedy(m, be, w.prompt, n)
		if err != nil {
			t.Fatalf("%s plain: %v", w.name, err)
		}
		tokens := append(append([]int(nil), w.prompt...), ref...)
		s, err := newInkling2Resident(m, sbe)
		if err != nil {
			t.Fatal(err)
		}
		if err := s.enableSpec(gamma); err != nil {
			s.close()
			t.Fatal(err)
		}
		d, err := s.attachDFlash2(dm)
		if err != nil {
			s.close()
			t.Fatal(err)
		}
		hd, err := NewInkling2DFlash2Draft(dm, Inkling2DFlash2TargetOf(tm))
		if err != nil {
			t.Fatal(err)
		}
		H := dm.Config.Hidden
		var commitErr error
		d.tapSink = func(T int, taps [][]float32) {
			for r := 0; r < T && commitErr == nil; r++ {
				row := make([][]float32, len(taps))
				for i := range taps {
					row[i] = taps[i][r*H : (r+1)*H]
				}
				commitErr = hd.Commit(hd.Len(), row)
			}
		}
		if _, err := s.prefill(context.Background(), w.prompt); err != nil {
			t.Fatal(err)
		}
		var wBlocks, wAgree, wDev, wHost int
		for p := len(w.prompt); p+gamma < len(tokens); p++ {
			if commitErr != nil {
				t.Fatal(commitErr)
			}
			if hd.Len() != p || s.pos != p {
				t.Fatalf("%s: host context %d, device %d, anchor at %d", w.name, hd.Len(), s.pos, p)
			}
			t0 := time.Now()
			got, err := d.propose(tokens[p], p, gamma)
			draftTimes = append(draftTimes, time.Since(t0))
			if err != nil {
				t.Fatal(err)
			}
			hb, err := hd.Propose(tokens[p], gamma)
			if err != nil {
				t.Fatal(err)
			}
			want := hb.Tokens
			same := 0
			for same < gamma && got[same] == want[same] {
				same++
			}
			accept := func(ids []int) int {
				a := 0
				for a < gamma && ids[a] == tokens[p+1+a] {
					a++
				}
				return a
			}
			wDev += accept(got)
			wHost += accept(want)
			wBlocks++
			agreeIDs += same
			if same == gamma {
				wAgree++
			} else {
				sc := slices.Clone(hb.Scores[same])
				sort.Slice(sc, func(i, j int) bool { return sc[i] > sc[j] })
				t.Logf("%s @%d: device %v host %v diverge at %d (host top-2 score margin %.4g)", w.name, p, got, want, same, sc[0]-sc[1])
			}
			// feed the true token: a one-row verify writes its context
			if _, err := s.verify([]int{tokens[p]}); err != nil {
				t.Fatal(err)
			}
			if err := s.commitVerify(1, 1); err != nil {
				t.Fatal(err)
			}
		}
		if w.name == workloads[0].name {
			inkling2DFlash2ProfileParts(t, d, tokens[len(tokens)-1], s.pos, gamma)
		}
		d.close()
		s.close()
		t.Logf("%s: %d blocks, identical %d, device accept mean %.2f/%d, host %.2f/%d", w.name, wBlocks, wAgree,
			float64(wDev)/float64(wBlocks), gamma, float64(wHost)/float64(wBlocks), gamma)
		blocks += wBlocks
		agreeBlocks += wAgree
		devAcc += wDev
		hostAcc += wHost
	}
	slices.Sort(draftTimes)
	ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }
	t.Logf("parity: %d blocks, identical %d (%.1f%%), id agreement %.2f/%d; teacher-forced accept device %.3f host %.3f; draft ms/call p50 %.1f min %.1f p90 %.1f; %s",
		blocks, agreeBlocks, 100*float64(agreeBlocks)/float64(max(1, blocks)), float64(agreeIDs)/float64(max(1, blocks)), gamma,
		float64(devAcc)/float64(max(1, blocks)), float64(hostAcc)/float64(max(1, blocks)),
		ms(draftTimes[len(draftTimes)/2]), ms(draftTimes[0]), ms(draftTimes[len(draftTimes)*9/10]), inkling2MetalSpecLoad())
	if blocks < 30 {
		t.Fatalf("only %d blocks compared", blocks)
	}
	if float64(devAcc) < 0.9*float64(hostAcc) {
		t.Errorf("device accepts %d of the host's %d teacher-forced drafts", devAcc, hostAcc)
	}
}

// inkling2DFlash2ProfileParts times the draft's layers, LM head and selector
// each fenced alone, best of several runs.
func inkling2DFlash2ProfileParts(t *testing.T, d *ink2DFlash2Draft, anchor, pos, depth int) {
	s := d.s
	fence := func() {
		if _, err := s.dev.Ink2Read(d.path, 1); err != nil {
			t.Fatal(err)
		}
	}
	best := func(f func()) float64 {
		b := time.Duration(1 << 62)
		for range 8 {
			copy(d.rows, d.tokenRow(anchor))
			s.fail(s.dev.Ink2Write(d.x, d.rows))
			s.fail(s.dev.Ink2Write(d.pos, []float32{intBits(pos)}))
			s.fail(s.dev.Ink2Write(d.anchor, []float32{intBits(anchor)}))
			fence()
			t0 := time.Now()
			f()
			fence()
			b = min(b, time.Since(t0))
		}
		return float64(b.Microseconds()) / 1000
	}
	layers := best(d.layersForward)
	head := best(func() { d.head(depth) })
	rowsHead := best(func() {
		inkling2DFlash2RowsHead = true
		d.head(depth)
		inkling2DFlash2RowsHead = false
	})
	rowsAll := best(func() {
		inkling2DFlash2RowsHead = true
		d.forward(depth)
		inkling2DFlash2RowsHead = false
	})
	sel := best(func() { d.selectPath(depth) })
	all := best(func() { d.forward(depth) })
	if s.err != nil {
		t.Fatal(s.err)
	}
	t.Logf("draft parts (best of 8, ms): layers %.2f, head %.2f (rows head %.2f), select %.2f, whole %.2f (with rows head %.2f)",
		layers, head, rowsHead, sel, all, rowsAll)
}
