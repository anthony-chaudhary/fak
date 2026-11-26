//go:build darwin && arm64 && cgo

package ggufload

import (
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/inklingmetal"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// The vendored Splash draft (af4g64 weights, Metal target taps) is the
// comparator only: the host reference's greedy chain should agree with
// Session.Draft on the oracle prompts' first anchor up to quantization.
func TestInkling2DFlash2ParityWithVendoredDraft(t *testing.T) {
	path := os.Getenv("FAK_INKLING2_GGUF")
	if path == "" {
		t.Skip("FAK_INKLING2_GGUF not set (Apple GPU comparator plus host target forward)")
	}
	dir, err := Inkling2DFlash2AdjacentDir(path)
	if err != nil || dir == "" {
		t.Skipf("no %s beside %s (%v)", Inkling2DFlash2DirName, path, err)
	}
	if !inklingmetal.Available() {
		t.Skip("Inkling Metal unavailable on this host")
	}
	vm, err := inklingmetal.OpenWithDraft(path, dir)
	if err != nil {
		t.Fatal(err)
	}
	defer vm.Close()
	tm, err := OpenInkling2Loop(path)
	if err != nil {
		t.Fatal(err)
	}
	defer tm.Close()
	dm, err := OpenInkling2DFlash2(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer dm.Close()
	gamma := dm.Config.DFlash.Block - 1
	var agreeFirst, agreeAll, total int
	for _, c := range inkling2OracleCases[:2] {
		s, err := vm.NewSession(256)
		if err != nil {
			t.Fatal(err)
		}
		ids := make([]int32, len(c.prompt))
		for i, v := range c.prompt {
			ids[i] = int32(v)
		}
		anchor, err := s.PrefillID(ids)
		if err != nil {
			s.Close()
			t.Fatal(err)
		}
		want, err := s.Draft(anchor, gamma)
		s.Close()
		if err != nil {
			t.Fatal(err)
		}
		if int(anchor) != c.greedy[0] {
			t.Fatalf("%s: Metal anchor %d, oracle %d", c.name, anchor, c.greedy[0])
		}
		h, err := model.NewInkling2LoopHost(tm.HP, tm.Weights)
		if err != nil {
			t.Fatal(err)
		}
		h.FeatureVisits = dm.Config.DFlash.Taps
		d, err := NewInkling2DFlash2Draft(dm, Inkling2DFlash2TargetOf(tm))
		if err != nil {
			t.Fatal(err)
		}
		for p, tok := range c.prompt {
			h.PrefillToken(tok, false)
			if err := d.Commit(p, h.Features); err != nil {
				t.Fatal(err)
			}
		}
		got, err := d.Propose(int(anchor), gamma)
		if err != nil {
			t.Fatal(err)
		}
		n := 0
		for n < len(want) && n < len(got.Tokens) && int(want[n]) == got.Tokens[n] {
			n++
		}
		t.Logf("%s: anchor %d, vendored %v, host %v, common prefix %d, oracle %v", c.name, anchor, want, got.Tokens, n, c.greedy[1:1+gamma])
		total++
		agreeAll += n
		if n > 0 {
			agreeFirst++
		}
	}
	t.Logf("parity: first-token agreement %d/%d, mean common prefix %.2f/%d", agreeFirst, total, float64(agreeAll)/float64(total), gamma)
	if agreeFirst < total {
		t.Fatalf("host reference disagrees with the vendored draft on a first proposal")
	}
}
