package model

import (
	"errors"
	"sync/atomic"
	"testing"
)

// fak-test:runtime fast est=1s lane=default
func TestV41ForwardAdmittedExpertMemo(t *testing.T) {
	m := v41IncrementalPlainModel(t, 2)
	expert := layerName(0, "ffn.experts.0.w1.weight")
	if _, ok := m.manifest[expert]; !ok {
		t.Fatalf("fixture has no manifest expert %s", expert)
	}
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatal(err)
	}
	if atomic.LoadPointer(&m.v41ExpertsAdmitted) == nil {
		t.Fatal("successful expert admission was not memoized")
	}

	t.Run("store change re-walks experts and refuses", func(t *testing.T) {
		meta := m.manifest[expert]
		delete(m.manifest, expert)
		defer func() { m.manifest[expert] = meta }()
		if err := m.v41ForwardAdmitted(); !errors.Is(err, ErrV41ForwardStage) {
			t.Fatalf("missing expert admitted from the memo: %v", err)
		}
	})

	t.Run("geometry change re-walks experts and refuses", func(t *testing.T) {
		saved := m.Cfg.MoEIntermediateSize
		defer func() { m.Cfg.MoEIntermediateSize = saved }()
		m.Cfg.MoEIntermediateSize++
		if err := m.v41ForwardAdmitted(); !errors.Is(err, ErrV41ForwardStage) {
			t.Fatalf("changed expert width admitted from the memo: %v", err)
		}
	})

	t.Run("restored model admits again", func(t *testing.T) {
		if err := m.v41ForwardAdmitted(); err != nil {
			t.Fatalf("restored model refused: %v", err)
		}
	})
}
