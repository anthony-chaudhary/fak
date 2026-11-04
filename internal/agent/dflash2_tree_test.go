package agent

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2TreeWidthBoundRefusal(t *testing.T) {
	p, target := dflash2ServingFixture(t)
	// The fixture model carries no layer geometry; give it the real target
	// depth and rebind the fingerprint before the attach can see it.
	p.m.Cfg.NumLayers = 64
	fingerprint, err := dflash2TargetFingerprint(p.m, p.tok)
	if err != nil {
		t.Fatal(err)
	}
	target.fingerprint = fingerprint
	target.options = DFlash2TargetOptions{Request: model.Qwen38DFlash2RequestOptions{TargetPanel: model.DFlash2TargetPanel8}}

	ctx := context.Background()
	s, err := p.AttachDFlash2Tree(DFlash2TreeSpec{Taps: []int{0, 32}})
	if err != nil {
		t.Fatalf("attach: %v", err)
	}
	if !reflect.DeepEqual(s.Taps(), []int{0, 32}) || s.Width() != 8 {
		t.Fatalf("taps=%v width=%d", s.Taps(), s.Width())
	}

	// Before build: tree requests hit the TreesNotBuilt equivalent; kept-row
	// absorption hits the NotDrafting equivalent.
	if _, err := s.Nodes(ctx, 4); !errors.Is(err, ErrDFlash2TreesNotBuilt) {
		t.Fatalf("unbuilt nodes: %v", err)
	}
	if err := s.AbsorbKept([]int{1}); !errors.Is(err, ErrDFlash2NotDrafting) {
		t.Fatalf("unbuilt absorb: %v", err)
	}
	if err := s.Build(ctx); err != nil {
		t.Fatalf("build: %v", err)
	}
	if err := s.Build(ctx); err == nil {
		t.Fatal("second build admitted")
	}
	if _, err := s.Nodes(ctx, 4); !errors.Is(err, ErrDFlash2NotDrafting) {
		t.Fatalf("undrafted nodes: %v", err)
	}
	if err := s.Begin(ctx); err != nil {
		t.Fatalf("begin: %v", err)
	}
	if err := s.Begin(ctx); err == nil {
		t.Fatal("reentrant begin admitted")
	}

	// Offers are bounded by the target width: exactly the first n node IDs,
	// and one past the bound refuses with the width sentinel.
	got, err := s.Nodes(ctx, 8)
	if err != nil || !reflect.DeepEqual(got, []int{0, 1, 2, 3, 4, 5, 6, 7}) {
		t.Fatalf("full-width offer=%v err=%v", got, err)
	}
	if got, err := s.Nodes(ctx, 3); err != nil || !reflect.DeepEqual(got, []int{0, 1, 2}) {
		t.Fatalf("partial offer=%v err=%v", got, err)
	}
	if _, err := s.Nodes(ctx, 9); !errors.Is(err, ErrDFlash2TreeWidthExceeded) {
		t.Fatalf("over-width offer: %v", err)
	}
	if _, err := s.Nodes(ctx, 0); err == nil {
		t.Fatal("empty offer admitted")
	}

	// Kept target rows absorb into drafter state and survive Finish.
	if err := s.AbsorbKept([]int{5, 6}); err != nil {
		t.Fatalf("absorb: %v", err)
	}
	if err := s.AbsorbKept(nil); err == nil {
		t.Fatal("empty absorption admitted")
	}
	if err := s.AbsorbKept([]int{-1}); err == nil {
		t.Fatal("negative row admitted")
	}
	if !reflect.DeepEqual(s.KeptRows(), []int{5, 6}) {
		t.Fatalf("kept=%v", s.KeptRows())
	}
	if err := s.Finish(); err != nil {
		t.Fatalf("finish: %v", err)
	}
	if _, err := s.Nodes(ctx, 4); !errors.Is(err, ErrDFlash2NotDrafting) {
		t.Fatalf("finished nodes: %v", err)
	}
	if err := s.AbsorbKept([]int{7}); !errors.Is(err, ErrDFlash2NotDrafting) {
		t.Fatalf("finished absorb: %v", err)
	}
	if !reflect.DeepEqual(s.KeptRows(), []int{5, 6}) {
		t.Fatalf("kept lost across finish: %v", s.KeptRows())
	}
	if err := s.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("repeat close: %v", err)
	}
	if _, err := s.Nodes(ctx, 4); err == nil {
		t.Fatal("retired stream offered nodes")
	}

	t.Run("refusals", func(t *testing.T) {
		target.options = DFlash2TargetOptions{}
		if _, err := p.AttachDFlash2Tree(DFlash2TreeSpec{Taps: []int{0}}); err == nil {
			t.Fatal("serial panel admitted a tree")
		}
		target.options = DFlash2TargetOptions{Request: model.Qwen38DFlash2RequestOptions{TargetPanel: model.DFlash2TargetPanel8}}
		for _, spec := range []DFlash2TreeSpec{{}, {Taps: []int{64}}, {Taps: []int{-1}}, {Taps: []int{1, 1}}} {
			if _, err := p.AttachDFlash2Tree(spec); err == nil {
				t.Fatalf("tap spec %+v admitted", spec)
			}
		}
		// A live lease means a forward can precede the tap declaration.
		lease := &dflash2Lease{}
		target.mu.Lock()
		target.leases[lease] = struct{}{}
		target.mu.Unlock()
		if _, err := p.AttachDFlash2Tree(DFlash2TreeSpec{Taps: []int{0}}); !errors.Is(err, ErrDFlash2Busy) {
			t.Fatalf("busy attach: %v", err)
		}
		target.mu.Lock()
		delete(target.leases, lease)
		target.mu.Unlock()
	})
}
